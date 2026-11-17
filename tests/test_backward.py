"""The memory-bounded gradients (`h3mlx.backward`) against the plain one (`train.loss_and_gradients`), which is itself
tested against torch and PEFT in `test_train_gradients.py`. Also here: the two facts about MLX that shape the
checkpointed path, each with a minimal test, and the direction of the memory measurement that
`bench/train_backward_memory.py` prints in full.
"""

import time

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx, to_mx
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from test_dit_mixed_precision import rel_err, to_mixed
from test_dit_tiny import TINY
from test_train_gradients import TABLE_FED, largest_scores, mlx_groups, sampling_forward_loss, setup, sharp_setup, swapped

from h3mlx import backward, lora, ops, train
from h3mlx.pipeline import Noise

ALL = sorted(lora.wired_shapes(TINY))  # an adapter on every Linear: trainable matrices before, in and after the blocks
BLOCKS_ONLY = lora.block_modules(TINY)[:12]  # the two transformer blocks' linears: nothing before or after them
WAYS = {"checkpointed": backward.loss_and_gradients_checkpointed, "blockwise": backward.loss_and_gradients_blockwise}
TOLERANCE = 1e-5  # per gradient tensor, relative, against the plain gradient
UNORDERED = 2e-3  # the same on the GPU for the tensors fed by a bf16 table's gradient, whose sum has no fixed order there


def groups(mixed=False, stacked=True, modules=ALL, sigma=0.5, n_refs=1):
    s = setup(n_refs, sigma, subject_modules=modules, std=0.1)
    frozen, trainable = mlx_groups(s, base=to_mixed(s["model"]) if mixed else None, stacked=stacked)
    return frozen, trainable, (s["cfg"], s["example"], to_mx(s["text"]))


def evaluated(function, *args, **kwargs):
    loss, grads = function(*args, **kwargs)
    mx.eval(loss, grads)
    return loss, grads


def worst_errors(grads, want) -> tuple[float, float]:
    """The largest relative error of a gradient tensor, (among those a modulation table's gradient feeds, among the
    rest). The audio head's adapter is apart: a picture's loss does not read the audio output, so its gradient is
    exactly zero, in the plain gradient too."""
    worst = {True: 0.0, False: 0.0}
    for k, g in grads.items():
        if k.startswith("audio_proj_out."):
            assert not np.array(g).any() and not np.array(want[k]).any(), k
        else:
            assert float(mx.abs(g).max()) > 0, k
            table_fed = any(part in k for part in TABLE_FED)
            worst[table_fed] = max(worst[table_fed], rel_err(g, want[k]))
    return worst[True], worst[False]


# --- the gradients ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mixed", [False, True], ids=["float32", "bf16 base"])
def test_checkpointed_loss_is_the_loss(mixed):
    """Checkpointing a block changes how its gradient is computed, not what it computes: the loss is
    `train.example_loss`'s bit for bit (which is the sampling forward's, `test_train_gradients.py`)."""
    frozen, trainable, args = groups(mixed)
    for weight in (None, 2.5):
        want = train.example_loss(trainable, frozen, *args, weight=weight)
        got = backward.checkpointed_loss(trainable, frozen, *args, weight=weight)
        assert got.dtype == mx.float32 and float(got) == float(want)


@pytest.mark.parametrize("way", list(WAYS))
@pytest.mark.parametrize("stacked", [False, True], ids=["one adapter", "two adapters"])
@pytest.mark.parametrize("modules", [ALL, BLOCKS_ONLY], ids=["every linear", "block linears"])
@pytest.mark.parametrize("precision", ["float32", "bf16 base on the cpu stream", "bf16 base"])
def test_bounded_gradients_equal_the_plain_gradient(way, precision, stacked, modules):
    """Each way's loss and its gradient for every trainable tensor, against the plain gradient: float32 and a bf16
    base with float32 factors, a trainable adapter alone and beside a frozen assistant, an adapter on every Linear
    (so the stages before and after the blocks have trainable matrices too) and one on block linears only. The keys
    are the trainable group's: nothing for the assistant or the base.

    The tolerance is 1e-5 per tensor (measured: exactly equal, but for the tensors a modulation table's gradient
    feeds, at 3e-7). One case is given more room. Over a bf16 base on the default device, the float32 sum behind a
    bf16 table's gradient has no fixed order, and a sum that lands on a rounding boundary can come out one bf16 step
    away from one run to the next. For the tensors such a table feeds (the time embedder's, the AdaLN projections'
    and the final norm projection's) the tolerance there is 2e-3; none moved in the runs measured, and on the CPU
    stream, where the order is fixed, the same cases hold at 1e-5."""
    frozen, trainable, args = groups(precision != "float32", stacked, modules)

    def compare():
        want_loss, want = evaluated(train.loss_and_gradients, trainable, frozen, *args)
        again = evaluated(train.loss_and_gradients, trainable, frozen, *args)[1]
        loss, grads = evaluated(WAYS[way], trainable, frozen, *args)
        return want_loss, want, loss, grads, worst_errors(again, want)

    if precision.endswith("cpu stream"):
        with mx.stream(mx.cpu):
            want_loss, want, loss, grads, rerun = compare()
    else:
        want_loss, want, loss, grads, rerun = compare()
    assert loss.dtype == mx.float32 and abs(float(loss) - float(want_loss)) <= 1e-6 * float(want_loss)
    assert list(grads) == list(trainable) and set(grads) == set(want)
    assert not any(".assistant." in k or ".lora_" not in k for k in grads)
    table_fed, rest = worst_errors(grads, want)
    print(f"{way}, {precision}, stacked {stacked}, {len(grads)} tensors: worst {rest:.1e}, table-fed {table_fed:.1e} "
          f"(the plain gradient against its own rerun: {rerun[1]:.1e}, {rerun[0]:.1e})")
    assert rest < TOLERANCE
    assert table_fed < (UNORDERED if precision == "bf16 base" else TOLERANCE)
    assert all(g.dtype == mx.float32 and g.shape == trainable[k].shape for k, g in grads.items())


@pytest.mark.parametrize("way", list(WAYS))
@pytest.mark.parametrize("gain", [30, 50])
def test_bounded_gradients_through_sharp_attention_over_a_bf16_base(way, gain):
    """A bf16 base whose attention scores are in the tens of thousands, at a head width the fused attention kernels take
    (`test_train_gradients.sharp_setup`): each bounded way is still the plain gradient, which there is the right one.
    With the sampling forward's attention in the training forward, the same way returns a gradient more than a
    thousand times as large, or one that is not finite: a bounded backward pass that went back to the bf16 call
    would show here."""
    s = sharp_setup(1, gain)
    frozen, trainable = mlx_groups(s, base=to_mixed(s["model"]))
    args = (s["cfg"], s["example"], to_mx(s["text"]))
    assert min(largest_scores(trainable, frozen, *args)) > 12_000
    want_loss, want = evaluated(train.loss_and_gradients, trainable, frozen, *args)
    loss, grads = evaluated(WAYS[way], trainable, frozen, *args)
    assert float(loss) == float(want_loss)
    worst = max(worst_errors(grads, want))
    with swapped("attention_for_training", ops.attention):
        _, before = evaluated(WAYS[way], trainable, frozen, *args)
    size = lambda g: float(np.sqrt(sum(float((v.astype(mx.float32) ** 2).sum()) for v in g.values())))  # noqa: E731
    print(f"{way}, gain {gain}: worst tensor {worst:.1e} from the plain gradient; norm {size(grads):.3g}, with the bf16 "
          f"call {size(before):.3g}")
    assert worst < TOLERANCE
    if mx.default_device() == mx.gpu:
        assert not size(before) < 1e3 * size(grads)


@pytest.mark.parametrize("way", list(WAYS))
def test_bounded_gradients_at_other_levels_and_layouts(way):
    """Text-to-image and two references, a level near pure noise and one where the picture's rows share a timestep
    with the references', and a loss weight."""
    for n_refs, sigma in ((0, 0.9), (2, 0.0005)):
        frozen, trainable, args = groups(sigma=sigma, n_refs=n_refs)
        want_loss, want = train.loss_and_gradients(trainable, frozen, *args, weight=1.7)
        loss, grads = WAYS[way](trainable, frozen, *args, weight=1.7)
        assert abs(float(loss) - float(want_loss)) <= 1e-6 * float(want_loss)
        assert max(worst_errors(grads, want)) < TOLERANCE


@pytest.mark.parametrize("way", list(WAYS))
def test_bounded_gradients_over_a_quantized_base(way):
    """The base's Linear weights 8-bit quantized, in the frozen group: captured by a checkpointed block, and constants
    of each block's product, as they are constants of the plain gradient."""
    s = setup(1, 0.5, cfg=dict(TINY, hidden_size=64, ffn_dim=96), std=0.1)
    frozen, trainable = mlx_groups(s)
    quantized = {k: (ops.QuantizedWeight(v, 8, 32) if k.endswith(".weight") and ".lora_" not in k and v.ndim == 2
                     and v.shape[1] % 32 == 0 else v) for k, v in frozen.items()}
    assert sum(isinstance(v, ops.QuantizedWeight) for v in quantized.values()) >= 18
    args = (s["cfg"], s["example"], to_mx(s["text"]))
    want_loss, want = train.loss_and_gradients(trainable, quantized, *args)
    loss, grads = WAYS[way](trainable, quantized, *args)
    assert abs(float(loss) - float(want_loss)) <= 1e-6 * float(want_loss)
    assert max(worst_errors(grads, want)) < TOLERANCE


@pytest.mark.parametrize("function", [backward.checkpointed_loss, *WAYS.values()])
def test_bounded_gradients_refuse_what_the_loss_refuses(function):
    """`train.check_inputs`, before anything is computed: matrices the forward would not read, and a text that is not
    the layout's."""
    frozen, trainable, (cfg, example, text) = groups(modules=BLOCKS_ONLY)
    with pytest.raises(ValueError, match="`trainable` is empty"):
        function({}, frozen, cfg, example, text)
    off, switched_off = lora.split(lora.set_active(lora.join(frozen, trainable), ("assistant",)), "subject")
    with pytest.raises(ValueError, match="is not active on"):
        function(switched_off, off, cfg, example, text)
    with pytest.raises(ValueError, match="text must be an array of shape"):
        function(trainable, frozen, cfg, example, text[:, :1])
    with pytest.raises(ValueError, match="key.s. are in both groups"):
        function(trainable, lora.join(frozen, trainable), cfg, example, text)
    with pytest.raises(ValueError, match="the config has 1 transformer_blocks, and the weights do not"):
        function(trainable, frozen, dict(cfg, num_layers=1), example, text)


def test_a_config_of_fewer_blocks_than_the_weights_is_refused():
    """Weights and trainable matrices for four blocks, a config that says two. Unchecked, each of the three gradients
    ran a two-block model and returned gradients of exactly zero for blocks 2 and 3 (the block-by-block one after
    filing their matrices under the stage before the blocks). All three refuse now, and `_stages` on its own refuses
    a key on a block the config does not have."""
    cfg4 = dict(TINY, num_layers=4)
    s = setup(1, 0.5, cfg=cfg4, subject_modules=lora.block_modules(cfg4), std=0.1)
    frozen, trainable = mlx_groups(s)
    cfg2 = dict(cfg4, num_layers=2)
    text = to_mx(s["text"])
    for function in (train.loss_and_gradients, *WAYS.values(), train.example_loss, backward.checkpointed_loss):
        with pytest.raises(ValueError, match="the config has 2 transformer_blocks, and the weights do not"):
            function(trainable, frozen, cfg2, s["example"], text)
        function(trainable, frozen, cfg4, s["example"], text)
    with pytest.raises(ValueError, match="on blocks this config does not have, e.g. 'transformer_blocks.2"):
        backward._stages(trainable, cfg2)
    # what the unchecked forward gave: zeros for the blocks it never ran
    unchecked = mx.grad(lambda t: sampling_forward_loss({**frozen, **t}, cfg2, s["example"], text))(trainable)
    left_out = [k for k in unchecked if k.startswith(("transformer_blocks.2.", "transformer_blocks.3."))]
    assert len(left_out) == 24 and all(not np.array(unchecked[k]).any() for k in left_out)
    assert any(np.array(unchecked[k]).any() for k in unchecked if k.startswith("transformer_blocks.1."))


def test_the_blockwise_gradient_reports_after_every_stage():
    """`after_stage` is called after each evaluation: the start, every block on the way up, the end, and every block
    on the way back. It changes nothing, and an exception raised in it stops the gradient there, which is how a
    trainer enforces a memory limit in the middle of a step."""
    frozen, trainable, args = groups(modules=BLOCKS_ONLY)
    calls = []
    want_loss, want = WAYS["blockwise"](trainable, frozen, *args)
    loss, grads = WAYS["blockwise"](trainable, frozen, *args, after_stage=lambda: calls.append(mx.get_peak_memory()))
    assert len(calls) == 2 * TINY["num_layers"] + 2 and float(loss) == float(want_loss)
    assert all(np.array_equal(np.array(grads[k]), np.array(want[k])) for k in want)

    class Stop(Exception):
        pass

    seen = []

    def third_time():
        seen.append(1)
        if len(seen) == 3:
            raise Stop

    with pytest.raises(Stop):
        WAYS["blockwise"](trainable, frozen, *args, after_stage=third_time)
    assert len(seen) == 3


def test_stages_cover_every_trainable_key_once():
    """Which stage of the forward a trainable matrix belongs to: the input projections, the token refiner and the time
    embedder come before the blocks; each block has its own, its AdaLN projection included; the final projection and
    the heads come after."""
    _, trainable, _ = groups()
    head, blocks, tail = backward._stages(trainable, TINY)
    assert sorted(head + [k for b in blocks for k in b] + tail) == sorted(trainable) and len(blocks) == TINY["num_layers"]
    assert all(k.startswith(f"transformer_blocks.{i}.") for i, b in enumerate(blocks) for k in b)
    assert any("adaln_proj.linear" in k for k in blocks[0]) and len(blocks[0]) == len(blocks[1]) == 14
    assert {k.split(".lora_")[0] for k in tail} == {"norm_out.linear", "proj_out", "audio_proj_out"}
    modules = {k.split(".lora_")[0] for k in head}
    assert {"proj_in", "audio_proj_in", "context_embedder", "time_embedder.linear_1", "time_embedder.linear_2"} <= modules
    assert all(m.startswith(("token_refiner.", "time_embedder.")) or m in ("proj_in", "audio_proj_in", "context_embedder") for m in modules)


# --- two facts about MLX the checkpointed path rests on ------------------------------------------------------------------


def test_a_checkpointed_function_gives_no_gradient_to_what_it_captures():
    """`mx.checkpoint` differentiates with respect to its arguments. An array the function captures from outside gets
    a zero gradient, with no error, even when the outer function is differentiated with respect to it. Passed as an
    argument it gets the right one. (For a frozen weight, zero is the gradient wanted: those are captured.)"""
    w = mx.array([[1.0, 2.0], [3.0, 4.0]])
    x = mx.array([1.0, -1.0])
    plain = mx.grad(lambda w: (mx.tanh(x @ w) ** 2).sum())(w)
    as_argument = mx.grad(lambda w: (mx.checkpoint(lambda x, w: mx.tanh(x @ w))(x, w) ** 2).sum())(w)
    captured = mx.grad(lambda w: (mx.checkpoint(lambda x: mx.tanh(x @ w))(x) ** 2).sum())(w)
    assert float(mx.abs(plain).max()) > 0.1
    assert np.array_equal(np.array(as_argument), np.array(plain))
    assert not np.array(captured).any(), "silently zero"
    # the input still gets its gradient through a function that captures the weight
    through = mx.grad(lambda x: (mx.checkpoint(lambda x: mx.tanh(x @ w))(x) ** 2).sum())(x)
    assert np.array_equal(np.array(through), np.array(mx.grad(lambda x: (mx.tanh(x @ w) ** 2).sum())(x)))


def chain_peak(checkpoint: bool, evaluate: bool, blocks: int = 16, rows: int = 1024, width: int = 256,
               device=mx.cpu) -> tuple[float, float]:
    """(peak of MLX's active memory in MB over the start, the value) of one gradient through a chain of blocks. By
    default on the CPU stream, where a buffer is released as soon as nothing needs it; `device=None` is the default
    device."""

    def run():
        mx.random.seed(0)
        weights = [mx.random.normal((width, width)) * 0.05 for _ in range(blocks)]
        a, b = mx.random.normal((4, width)) * 0.05, mx.random.normal((width, 4)) * 0.05
        x0 = mx.random.normal((rows, width))
        mx.eval(weights, a, b, x0)

        def block(x, a, b, w):
            h = mx.tanh(x @ w.T + (x @ a.T) @ b.T)
            return x + 0.5 * (mx.tanh(h * 1.1) + 0.1 * mx.sin(h))

        def loss(factors):
            x = x0
            for w in weights:
                run = mx.checkpoint(lambda x, a, b, w=w: block(x, a, b, w)) if checkpoint else (lambda x, a, b, w=w: block(x, a, b, w))
                x = run(x, *factors)
                if evaluate:
                    mx.eval(x)  # what the sampling forward does after every block
            return (x ** 2).mean()

        mx.clear_cache()
        mx.reset_peak_memory()
        start = mx.get_active_memory()
        value, grads = mx.value_and_grad(loss)([a, b])
        mx.eval(value, grads)
        return (mx.get_peak_memory() - start) / 1e6, float(value)

    if device is None:
        return run()
    with mx.stream(device):
        return run()


def test_an_eval_in_the_loop_defeats_the_checkpoint():
    """A chain of sixteen blocks, differentiated four ways. Checkpointed, with nothing evaluated inside, the peak is
    below the plain gradient's: block intermediates are recomputed, not kept. Put the sampling forward's `mx.eval`
    after each block and the saving is gone: an array evaluated while it is being traced keeps its whole graph, so the
    checkpointed chain peaks where the plain one with the same evals does, above both chains without them. The values
    are the same four times. Measured on the CPU stream. On the default device one evaluation holds its buffers
    until the device has run them, and on this chain the checkpoint saves nothing there either way
    (`bench/train_backward_memory.py` prints both)."""
    plain, value = chain_peak(checkpoint=False, evaluate=False)
    checkpointed, same = chain_peak(checkpoint=True, evaluate=False)
    plain_eval, _ = chain_peak(checkpoint=False, evaluate=True)
    checkpointed_eval, also = chain_peak(checkpoint=True, evaluate=True)
    print(f"peak MB: plain {plain:.0f}, checkpointed {checkpointed:.0f}, plain with evals {plain_eval:.0f}, "
          f"checkpointed with evals {checkpointed_eval:.0f}")
    assert value == same == also
    assert checkpointed < 0.8 * plain, "the checkpoint saves memory when nothing is evaluated inside"
    assert checkpointed_eval > 1.5 * checkpointed, "and stops saving when the loop evaluates"
    assert checkpointed_eval > 0.9 * plain_eval


# --- the direction of the memory measurement ----------------------------------------------------------------------------


def synthetic(layers: int, side: int = 48, seed: int = 0):
    """A transformer larger than the tests' tiny one (hidden 96, about 600 rows) in the release's mixed precision,
    with a float32 adapter on its attention and feed-forward linears, and one text-to-image example: the arguments of
    a gradient function. Random weights; nothing about it is the real model's."""
    cfg = dict(TINY, hidden_size=96, ffn_dim=192, num_layers=layers)
    torch.manual_seed(seed)
    base = state_mx(to_mixed(MiniMaxH3Transformer3DModel(**cfg).eval()))
    g = torch.Generator().manual_seed(seed + 1)
    fresh = lora.init(cfg, rank=4, seed=seed)
    tensors = {k: (v if ".lora_A." in k else mx.array((torch.randn(v.shape, generator=g) * 0.05).numpy()))
               for k, v in fresh.tensors.items()}
    frozen, trainable = lora.split(lora.attach(base, cfg, lora.Adapter(tensors, fresh.alpha), keep_dtype=True, name="subject"), "subject")
    latents = mx.array(torch.randn(1, 1, side, side, cfg["in_channels"], generator=g).numpy())
    example = train.build_example(cfg, np.array([1] * 16), latents, 0.5, Noise(seed), condition_t=0.999)
    text = mx.array(torch.randn(1, 16, cfg["text_dim"], generator=g).numpy())
    mx.eval(frozen, trainable, example.video_rows, example.audio_rows, example.target, text)
    return trainable, frozen, cfg, example, text


def through_the_sampling_forward(trainable, frozen, cfg, example, text):
    """The gradient taken through `ops.dit_forward`, which evaluates after every block: what differentiating the
    sampling forward as it is costs."""
    return mx.value_and_grad(lambda t: sampling_forward_loss({**frozen, **t}, cfg, example, text))(trainable)


MEASURED = {"sampling forward": through_the_sampling_forward, "plain": train.loss_and_gradients, **WAYS}


def gradient_peak(way: str, layers: int, device=None, side: int = 48) -> tuple[float, float, float]:
    """(peak of MLX's active memory in MB over the start, seconds, the loss) of one gradient of the synthetic model."""

    def run():
        args = synthetic(layers, side)
        mx.clear_cache()
        mx.reset_peak_memory()
        start, began = mx.get_active_memory(), time.perf_counter()
        loss, grads = MEASURED[way](*args)
        mx.eval(loss, grads)
        return (mx.get_peak_memory() - start) / 1e6, time.perf_counter() - began, float(loss)

    if device is None:
        return run()
    with mx.stream(device):
        return run()


def test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept():
    """Only directions are asserted, from 4 blocks to 16, as ratios of growth, and no absolute number.

    On the default device: the block-by-block gradient's peak grows by less than a twentieth of what the plain
    gradient's does (it keeps one block input per block; measured 1 to 3%. Without the evaluation after each block of
    its forward walk it measured 6 to 17%). On the CPU stream, where MLX releases a buffer as soon as nothing needs
    it and the numbers repeat: that again; the checkpointed gradient's growth is below half the plain one's; and the
    plain gradient's, whose forward evaluates nothing, is below 0.6 of the growth of a gradient taken through the
    sampling forward, which evaluates after every block. The checkpointed path is not asserted on the default
    device, where what it saves depends on how the device batches its work. A synthetic model of about 600 rows and
    hidden size 96 says nothing about 50 real blocks."""
    peaks = {(way, n): gradient_peak(way, n)[0] for way in ("plain", "blockwise") for n in (4, 16)}
    growth = {way: peaks[way, 16] - peaks[way, 4] for way in ("plain", "blockwise")}
    print("default device, MB:", {k: round(v) for k, v in peaks.items()})
    assert growth["plain"] > 0 and growth["blockwise"] < 0.05 * growth["plain"]

    on_cpu = {(way, n): gradient_peak(way, n, mx.cpu)[0] for way in MEASURED for n in (4, 16)}
    cpu_growth = {way: on_cpu[way, 16] - on_cpu[way, 4] for way in MEASURED}
    print("cpu stream, MB:", {k: round(v) for k, v in on_cpu.items()})
    assert cpu_growth["blockwise"] < 0.05 * cpu_growth["plain"] and cpu_growth["checkpointed"] < 0.5 * cpu_growth["plain"]
    assert 0 < cpu_growth["plain"] < 0.6 * cpu_growth["sampling forward"]
