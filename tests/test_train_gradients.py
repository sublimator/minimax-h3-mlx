"""The loss of a training example and its gradient on the tiny transformer, against torch autograd and PEFT.

Torch side: the diffusers transformer with PEFT's layers, a frozen assistant adapter and a trainable one whose
parameters are float32 (`cast_training_params`), fed the example that diffusers' own functions assemble
(`test_train_example.theirs`), `mse_loss` on the generated video rows, `backward`. Ours: `train.loss_and_gradients`
with the weights as two arguments. In float32 the two agree closely; over a bf16 base each is scored against the
float32 run, as the forward's mixed-precision tests do.

A randomly initialised model's attention scores, q . k before the scale, are in the tens, and a pretrained model's
reach the tens of thousands. The `sharpened` models here have scores that large (`SHARP`, at a head width the default
device's fused attention kernels take), which is where the attention's bf16 backward goes wrong
(`test_attention_backward.py`).
"""

import contextlib
import dataclasses
import math

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx, to_mx, to_np
from diffusers.training_utils import cast_training_params
from test_dit_mixed_precision import rel_err, to_mixed
from test_dit_tiny import TINY
from test_hyperflow import randomised
from test_lora import GUARD, random_adapter
from test_lora_stack import ASSISTANT_MODULES, SUBJECT_MODULES, activate, peft_stack
from test_train_example import TAGS, latents_of, ours, theirs

from h3mlx import lora, ops, train
from h3mlx.pipeline import Noise


def setup(n_refs=1, sigma=0.5, seed=2, cfg=TINY, std=0.3, subject_modules=SUBJECT_MODULES, subject=None):
    """A model, a frozen and a trainable adapter, one example built both ways, and a prompt's embeddings."""
    model = randomised(cfg, seed)
    assistant_modules = lora.block_modules(cfg)
    assistant = random_adapter(cfg, {m: 4 for m in assistant_modules}, {m: 4 for m in assistant_modules}, seed=seed + 50, std=std)
    if subject is None:
        subject = random_adapter(cfg, {m: 3 for m in subject_modules}, {m: 6 for m in subject_modules}, seed=seed + 70, std=std)
    picture, refs = latents_of(n_refs, seed=seed)
    text = torch.randn(1, len(TAGS), cfg["text_dim"], generator=torch.Generator().manual_seed(seed))
    return dict(cfg=dict(cfg), model=model, assistant=assistant, subject=subject, text=text,
                example=ours(picture, refs, sigma, seed=seed), want=theirs(picture, refs, sigma, seed=seed))


def torch_model(s, tmp_path, base=None, stacked=True):
    """The PEFT model a trainer holds: the subject's parameters trainable and float32, everything else frozen.
    Loading an adapter into a bf16 model rounds its matrices to bf16, and `cast_training_params` widens the dtype
    without bringing the values back; a trainer's matrices are not on that grid after their first step, so the
    subject's float32 values are put back, the ones our side holds."""
    names = ("assistant", "subject") if stacked else ("subject",)
    m = activate(peft_stack(s["model"] if base is None else base, tmp_path, s["assistant"], s["subject"]), names)
    for name, prm in m.named_parameters():
        prm.requires_grad_(".lora_" in name and ".subject." in name)
    cast_training_params(m, dtype=torch.float32)
    exact = lora.runtime_tensors(s["subject"], "subject")
    parameters = dict(m.named_parameters())
    with torch.no_grad():
        for name, value in exact.items():
            parameters[name].copy_(torch.from_numpy(to_np(value.astype(mx.float32))))
    return m


def torch_loss(m, w, text, target=None):
    """`mse_loss` on the generated video rows of the model's forward on an example from diffusers' functions (`theirs`)."""
    video_pred, _ = m(w["video_rows"][None], w["audio_rows"][None], text, w["timestep"], w["indices"],
                      token_tags=w["token_tags"], position_ids=w["position_ids"], video_indices=w["video_indices"],
                      audio_indices=w["audio_indices"], text_indices=w["text_indices"], return_dict=False)
    return torch.nn.functional.mse_loss(video_pred[:, w["n_cond"]:].float(), (w["target"] if target is None else target)[None])


def torch_side(s, tmp_path, base=None, stacked=True, target=None):
    """(loss, {parameter name: gradient or None}, the PEFT model) after one `backward`."""
    m = torch_model(s, tmp_path, base, stacked)
    loss = torch_loss(m, s["want"], s["text"], target)
    loss.backward()
    return float(loss.detach()), {n: p.grad for n, p in m.named_parameters()}, m


def mlx_groups(s, base=None, stacked=True):
    """(frozen, trainable) for `train.loss_and_gradients`: the assistant attached as loaded (its base's dtype), the
    subject kept float32."""
    cfg = s["cfg"]
    params = state_mx(s["model"] if base is None else base)
    if stacked:
        params = lora.attach(params, cfg, s["assistant"], name="assistant")
    return lora.split(lora.attach(params, cfg, s["subject"], keep_dtype=True, name="subject"), "subject")


def flat(grads, keys):
    return np.concatenate([to_np(grads[k]).ravel() for k in keys])


TABLE_FED = ("time_embedder.", "adaln_proj.linear", "norm_out.linear")  # modules a modulation table's gradient reaches


def cancelling_adapter(cfg, modules, half, delta, seed, std=0.1):
    """An adapter that only float32 products get right. lora_A's second half is minus its first half plus
    `delta` * N, lora_B's second half repeats its first, and the scaling is 1 / delta: the update is B1 (N x), the
    difference of two products `1 / delta` times its size. With delta = 2**-10, float32 keeps 14 bits of it and bf16
    none."""
    g = torch.Generator().manual_seed(seed)
    shapes = lora.wired_shapes(cfg)
    tensors, alpha = {}, {}
    for m in modules:
        out_f, in_f = shapes[m]
        a1, n = torch.randn(half, in_f, generator=g) * std, torch.randn(half, in_f, generator=g) * std
        b1 = torch.randn(out_f, half, generator=g) * std
        tensors[m + ".lora_A.weight"] = to_mx(torch.cat([a1, -a1 + delta * n]))
        tensors[m + ".lora_B.weight"] = to_mx(torch.cat([b1, b1], dim=1))
        alpha[m] = 2 * half / delta
    return lora.Adapter(tensors, alpha)


SHARP = dict(TINY, num_layers=4, attention_head_dim=64)  # a head width the fused attention kernel takes; the release's is 128


def sharpened(model, gain, refiner=True):
    """The model with the `norm_q` and `norm_k` weights of every attention (the blocks' and, with `refiner`, the
    token refiner's) multiplied by `gain`, in place: attention scores `gain` squared times as large. At 30 the
    largest are about 15,000 and more, which a bf16 score tile gets wrong by 4 in the logit and more."""
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if name.endswith(("norm_q.weight", "norm_k.weight")) and (refiner or name.startswith("transformer_blocks.")):
                prm.mul_(gain)
    return model


def sharp_setup(seed, gain, cfg=SHARP, refiner=True):
    """`setup` on a sharpened model, with the trainable adapter on the block and refiner matrices (what a run
    trains)."""
    s = setup(seed % 3, (0.9, 0.5, 0.2)[seed % 3], seed=seed, cfg=cfg, std=0.1, subject_modules=lora.block_modules(cfg))
    sharpened(s["model"], gain, refiner)
    return s


@contextlib.contextmanager
def swapped(name, function):
    """`ops.<name>` replaced for the block: the training forward looks its gather and its attention up in `ops` when
    it runs, so this is the training forward with another one."""
    kept = getattr(ops, name)
    setattr(ops, name, function)
    try:
        yield
    finally:
        setattr(ops, name, kept)


def attention_by_hand(q, k, v, mask=None):
    """Attention written out in float32 with MLX's plain ops (scores, softmax, product), the result cast back: an
    independent backward pass for `ops.attention_for_training`'s."""
    qt, kt, vt = (t.astype(mx.float32).transpose(0, 2, 1, 3) for t in (q, k, v))
    scores = (qt * (1.0 / math.sqrt(q.shape[-1]))) @ kt.transpose(0, 1, 3, 2)
    return (mx.softmax(scores, axis=-1, precise=True) @ vt).transpose(0, 2, 1, 3).astype(q.dtype)


def largest_scores(trainable, frozen, cfg, example, text):
    """The largest unscaled score |q . k| of each attention the training forward runs, in order (the refiner's, then
    each block's). The logit is the score times 1 / sqrt(head width)."""
    seen = []
    plain = ops.attention_for_training

    def record(q, k, v, mask=None):
        qt, kt = (t.astype(mx.float32).transpose(0, 2, 1, 3) for t in (q, k))
        seen.append(float(mx.abs(qt @ kt.transpose(0, 1, 3, 2)).max()))
        return plain(q, k, v, mask)

    with swapped("attention_for_training", record):
        mx.eval(train.example_loss(trainable, frozen, cfg, example, text))
    return seen


def bf16_logit_error(score, width):
    """What a score held in bf16 can be off by, as a logit: half a bf16 step at the score (eight significant bits),
    times the scale 1 / sqrt(head width)."""
    return 2.0 ** (math.floor(math.log2(score)) - 8) / math.sqrt(width)


def evaluated(function, *args, **kwargs):
    loss, grads = function(*args, **kwargs)
    mx.eval(loss, grads)
    return loss, grads


def sampling_forward_loss(params, cfg, example, text, weight=None):
    """The example's loss on one dict of weights through `ops.dit_forward`, the forward sampling runs."""
    lay = example.layout
    video_pred, _ = ops.dit_forward(
        params, cfg, example.video_rows[None], example.audio_rows[None], text, mx.array(example.timestep),
        mx.array(example.timestep_indices), token_tags=mx.array(lay.token_tags),
        position_ids=mx.array(lay.position_ids.astype(np.float32)), video_indices=mx.array(lay.video_indices),
        audio_indices=mx.array(lay.audio_indices), text_indices=mx.array(lay.text_indices))
    return train.flow_match_loss(example.generated(video_pred), example.target[None], weight)


# --- float32 -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_refs,sigma", [(0, 0.9), (1, 0.5), (2, 0.2), (1, 0.0005)])
@pytest.mark.parametrize("stacked", [True, False])
def test_loss_and_gradients_are_torchs(tmp_path, n_refs, sigma, stacked):
    """Float32 throughout: the loss and the gradient with respect to both matrices of every trainable Linear, with a
    frozen adapter beside it and without, for text-to-image and edits, at four noise levels (at the last, the
    picture's rows and the references' share a timestep)."""
    s = setup(n_refs, sigma)
    want_loss, want, _ = torch_side(s, tmp_path, stacked=stacked)
    frozen, trainable = mlx_groups(s, stacked=stacked)
    loss, grads = train.loss_and_gradients(trainable, frozen, s["cfg"], s["example"], to_mx(s["text"]))
    assert loss.dtype == mx.float32 and abs(float(loss) - want_loss) <= 1e-5 * want_loss
    assert set(grads) == set(trainable) == {k for k, g in want.items() if g is not None}
    worst = max(rel_err(g, want[k]) for k, g in grads.items())
    print(f"float32, {n_refs} references, sigma {sigma}, stacked {stacked}: worst gradient tensor {worst:.1e}")
    assert worst < 1e-4
    for k, g in grads.items():
        assert g.dtype == mx.float32 and g.shape == trainable[k].shape and float(mx.abs(g).max()) > 0, k


def test_at_pefts_init_the_gradient_reaches_lora_b_alone(tmp_path):
    """A fresh adapter (`lora.init`: lora_B zero) is where training starts. The loss is then the model's without the
    adapter, lora_A's gradient is exactly zero on both sides (it is multiplied by lora_B), and lora_B's is torch's."""
    fresh = lora.init(TINY, rank=3, alpha=6, seed=1)
    s = setup(1, 0.5, subject=fresh)
    want_loss, want, _ = torch_side(s, tmp_path)
    frozen, trainable = mlx_groups(s)
    loss, grads = train.loss_and_gradients(trainable, frozen, s["cfg"], s["example"], to_mx(s["text"]))
    assert abs(float(loss) - want_loss) <= 1e-5 * want_loss
    for k, g in grads.items():
        if ".lora_A." in k:
            assert not bool(mx.any(g != 0)) and not want[k].any(), k
        else:
            assert rel_err(g, want[k]) < 1e-4 and float(mx.abs(g).max()) > 0, k
    without = sampling_forward_loss(lora.set_active(lora.join(frozen, trainable), ("assistant",)), s["cfg"],
                                    s["example"], to_mx(s["text"]))
    assert abs(float(without) - float(loss)) <= 1e-6 * float(loss), "a zero lora_B changes nothing in the forward"


def test_weight_scales_the_loss_and_the_gradient(tmp_path):
    s = setup(1, 0.5)
    frozen, trainable = mlx_groups(s)
    args = (trainable, frozen, s["cfg"], s["example"], to_mx(s["text"]))
    loss, grads = train.loss_and_gradients(*args)
    weighted, weighted_grads = train.loss_and_gradients(*args, weight=2.5)
    assert abs(float(weighted) - 2.5 * float(loss)) <= 1e-6 * float(weighted)
    assert max(rel_err(weighted_grads[k], 2.5 * to_np(g)) for k, g in grads.items()) < 1e-5


# --- who gets a gradient -----------------------------------------------------------------------------------------------


def test_no_gradient_for_the_base_or_the_assistant(tmp_path):
    """Torch: after `backward` only the subject's parameters hold a gradient; the base's and the assistant's hold
    None. Ours: the gradients are a dict with the trainable group's keys and nothing else, and the frozen group comes
    back untouched. The check is not empty: differentiated on purpose, the assistant's matrices and the base weights
    do have non-zero gradients here, so a function that differentiated everything would return them."""
    s = setup(1, 0.5)
    _, want, peft_model = torch_side(s, tmp_path)
    assert all(g is None for n, g in want.items() if ".subject." not in n)
    assert sum(g is not None for g in want.values()) == 2 * len(SUBJECT_MODULES)
    assert all(not p.requires_grad for n, p in peft_model.named_parameters() if ".assistant." in n or ".lora_" not in n)

    frozen, trainable = mlx_groups(s)
    before = dict(frozen)
    text = to_mx(s["text"])
    loss, grads = train.loss_and_gradients(trainable, frozen, s["cfg"], s["example"], text)
    assert set(grads) == set(trainable) == {k for k in want if ".subject." in k}
    assert not any(".assistant." in k or ".lora_" not in k for k in grads)
    # (MLX rebuilds the tuples of adapter names in the dict it is handed; the arrays are the very same objects)
    assert set(frozen) == set(before)
    assert all(frozen[k] is before[k] if isinstance(before[k], mx.array) else frozen[k] == before[k] for k in frozen)
    assert all(lora.RUNTIME_RE.match(k)["name"] == "subject" for k in trainable)
    # what a wrong grouping would hand back
    assistant = {k: v for k, v in frozen.items() if ".assistant." in k and k.endswith(".weight")}
    rest = {k: v for k, v in frozen.items() if k not in assistant}
    _, leaked = mx.value_and_grad(lambda a: train.example_loss(trainable, {**rest, **a}, s["cfg"], s["example"], text))(assistant)
    assert len(leaked) == 2 * len(ASSISTANT_MODULES) and all(float(mx.abs(g).max()) > 0 for g in leaked.values())
    base_weight = {"transformer_blocks.0.attn.to_q.weight": frozen["transformer_blocks.0.attn.to_q.weight"]}
    _, leaked = mx.value_and_grad(lambda w: train.example_loss(trainable, {**frozen, **w}, s["cfg"], s["example"], text))(base_weight)
    assert float(mx.abs(leaked["transformer_blocks.0.attn.to_q.weight"]).max()) > 0


# --- the sign of the target --------------------------------------------------------------------------------------------


def test_the_targets_sign_reaches_the_loss_and_the_gradient(tmp_path):
    """The torch side's target here is not `x_0 - noise`: it is the velocity the scheduler's step needs, solved from
    `x_0 = x_t + (1 - t) * v` in float64 on the rows diffusers' functions made. Our example's loss and gradient match
    it. With the example's target negated (`noise - x_0`), the loss and the gradient are far from torch's."""
    s = setup(1, 0.5)
    w = s["want"]
    noised = w["video_rows"][w["n_cond"]:].double()
    t = float(np.float32(1) - np.float32(0.5))
    from_step = ((w["clean"].double() - noised) / (1.0 - t)).float()
    want_loss, want, _ = torch_side(s, tmp_path, target=from_step)
    frozen, trainable = mlx_groups(s)
    text = to_mx(s["text"])
    loss, grads = train.loss_and_gradients(trainable, frozen, s["cfg"], s["example"], text)
    keys = sorted(grads)
    assert abs(float(loss) - want_loss) <= 1e-5 * want_loss and rel_err(flat(grads, keys), flat(want, keys)) < 1e-4
    flipped = dataclasses.replace(s["example"], target=-s["example"].target)
    wrong_loss, wrong = train.loss_and_gradients(trainable, frozen, s["cfg"], flipped, text)
    apart = rel_err(flat(wrong, keys), flat(want, keys))
    print(f"flipped target: loss {float(wrong_loss):.4f} against {want_loss:.4f}, gradient {apart:.2f} away")
    # a random model's output is nearly uncorrelated with the target, so the loss moves by percents; the gradient,
    # which is linear in (prediction - target), moves by its own size
    assert abs(float(wrong_loss) - want_loss) > 0.01 * want_loss
    assert apart > 10 * GUARD


# --- a bf16 base ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradients_over_a_bf16_base_track_torchs(tmp_path, seed):
    """The reference setting for training: the base in the release's mixed precision (bf16 blocks), the assistant in
    its base's dtype as loaded, the trainable matrices float32 and their products float32. Gradients are float32. There
    is no exact answer in bf16, so each side is scored against the float32 run: MLX's gradient error is at most 1.5
    times torch's own (plus 1e-3), and both losses are within 1% of the float32 loss. Four layers, adapters on the
    attention and feed-forward matrices."""
    cfg = dict(TINY, num_layers=4)
    modules = lora.block_modules(cfg)
    s = setup(seed % 3, (0.9, 0.5, 0.2)[seed % 3], seed=seed, cfg=cfg, std=0.1, subject_modules=modules)
    truth_loss, truth, _ = torch_side(s, tmp_path)
    mixed = to_mixed(s["model"])
    torch_loss, torch_grads, peft_model = torch_side(s, tmp_path, base=mixed)
    block = "transformer_blocks.0.attn.to_q"
    dtypes = {n.split(block + ".")[1]: p.dtype for n, p in peft_model.named_parameters() if n.startswith(block + ".")}
    assert dtypes == {"base_layer.weight": torch.bfloat16, "lora_A.assistant.weight": torch.bfloat16,
                      "lora_B.assistant.weight": torch.bfloat16, "lora_A.subject.weight": torch.float32,
                      "lora_B.subject.weight": torch.float32}

    frozen, trainable = mlx_groups(s, base=mixed)
    assert frozen[block + ".weight"].dtype == frozen[block + ".lora_A.assistant.weight"].dtype == mx.bfloat16
    assert all(v.dtype == mx.float32 for v in trainable.values())
    loss, grads = train.loss_and_gradients(trainable, frozen, s["cfg"], s["example"], to_mx(s["text"]))
    assert loss.dtype == mx.float32 and all(g.dtype == mx.float32 for g in grads.values())
    keys = sorted(grads)
    reference = flat(truth, keys)
    e_torch, e_mlx = rel_err(flat(torch_grads, keys), reference), rel_err(flat(grads, keys), reference)
    print(f"seed {seed}: gradient error against float32, torch {e_torch:.4f}, mlx {e_mlx:.4f}; "
          f"loss error torch {abs(torch_loss - truth_loss) / truth_loss:.4f}, mlx {abs(float(loss) - truth_loss) / truth_loss:.4f}")
    assert 1e-4 < e_torch < 0.1, "bf16 is visible in torch's own gradient, and small"
    assert e_mlx <= 1.5 * e_torch + 1e-3
    assert abs(torch_loss - truth_loss) < 0.01 * truth_loss and abs(float(loss) - truth_loss) < 0.01 * truth_loss


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_products_with_float32_matrices_are_float32_over_a_bf16_base(tmp_path, seed):
    """The test above cannot tell float32 products from bf16 ones: with ordinary matrices either is within bf16's
    noise of the float32 run. This adapter can (`cancelling_adapter`): its update is the small difference of two
    large products. Torch with PEFT's float32 matrices over the bf16 base stays as close to the float32 run as it
    does with any adapter, and so does ours. The same matrices held in bf16, which is what products in the base's
    dtype would compute, are tens of percent away."""
    cfg = dict(TINY, num_layers=4)
    modules = lora.block_modules(cfg)
    subject = cancelling_adapter(cfg, modules, half=2, delta=2.0 ** -10, seed=seed + 70)
    s = setup(seed % 3, (0.5, 0.9, 0.2)[seed % 3], seed=seed, cfg=cfg, std=0.1, subject_modules=modules, subject=subject)
    truth_loss, truth, _ = torch_side(s, tmp_path)
    mixed = to_mixed(s["model"])
    torch_loss, torch_grads, peft_model = torch_side(s, tmp_path, base=mixed)
    name = "transformer_blocks.0.attn.to_q.lora_A.subject.weight"
    assert np.array_equal(dict(peft_model.named_parameters())[name].detach().numpy(), to_np(subject.tensors[name.replace(".subject", "")]))

    frozen, trainable = mlx_groups(s, base=mixed)
    args = (s["cfg"], s["example"], to_mx(s["text"]))
    loss, grads = train.loss_and_gradients(trainable, frozen, *args)
    in_bf16 = {k: v.astype(mx.bfloat16) for k, v in trainable.items()}
    assert any(not np.array_equal(to_np(in_bf16[k].astype(mx.float32)), to_np(v)) for k, v in trainable.items())
    half_loss, half_grads = train.loss_and_gradients(in_bf16, frozen, *args)
    keys = sorted(grads)
    reference = flat(truth, keys)
    e_torch, e_mlx, e_half = (rel_err(flat(g, keys), reference) for g in (torch_grads, grads, half_grads))
    print(f"seed {seed}: gradient error against float32, torch {e_torch:.4f}, mlx {e_mlx:.4f}, mlx with bf16 matrices {e_half:.4f}; "
          f"loss {truth_loss:.4f}, torch {torch_loss:.4f}, mlx {float(loss):.4f}, bf16 matrices {float(half_loss):.4f}")
    assert 1e-4 < e_torch < 0.05, "torch's float32 products keep the difference"
    assert e_mlx <= 1.5 * e_torch + 1e-3
    assert abs(float(loss) - truth_loss) < 0.01 * truth_loss
    assert e_half > 0.25 and e_half > 10 * e_mlx, "bf16 products do not: the test can tell the two apart"


# --- the forward that is differentiated ----------------------------------------------------------------------------------


@pytest.mark.parametrize("base", ["float32", "bf16, head width 64", "bf16, head width 32"])
@pytest.mark.parametrize("n_refs,sigma", [(0, 0.9), (1, 0.5), (2, 0.0005)])
def test_the_training_forward_is_the_sampling_forward(base, n_refs, sigma):
    """`train.example_loss` runs the forward in stages with nothing evaluated between them, gathers the modulation
    tables' rows through float32 and runs the attention in float32; `ops.dit_forward`, which sampling runs and which
    is tested against diffusers, evaluates after every block and does both in the weights' dtype. None of the three
    changes a value where it matters: the loss is the same bit for bit in float32, and over a bf16 base at a head
    width the default device's fused attention kernel takes (64 here, 128 in the release), where the bf16 attention
    call already computes in float32 and rounds. With adapters on every Linear.

    At a head width that kernel does not take (32, the other tiny models'), MLX computes the sampling forward's
    attention in bf16, and the training forward's is the more exact of the two: the losses are within 0.2% of each
    other (measured: 5e-4, 3e-5 and 2e-4) and not the same bits."""
    cfg = dict(TINY, attention_head_dim=64) if base.endswith("64") else dict(TINY)
    s = setup(n_refs, sigma, cfg=cfg, subject_modules=sorted(lora.wired_shapes(cfg)), std=0.1)
    frozen, trainable = mlx_groups(s, base=None if base == "float32" else to_mixed(s["model"]))
    cfg, example, text = s["cfg"], s["example"], to_mx(s["text"])
    for weight in (None, 2.5):
        want = sampling_forward_loss(lora.join(frozen, trainable), cfg, example, text, weight)
        got = train.example_loss(trainable, frozen, cfg, example, text, weight)
        assert got.dtype == want.dtype == mx.float32
        if base.endswith("32") or (base != "float32" and mx.default_device() != mx.gpu):
            assert abs(float(got) - float(want)) < 2e-3 * float(want)
        else:
            assert float(got) == float(want)
    without_adapter = sampling_forward_loss(lora.set_active(lora.join(frozen, trainable), ()), cfg, example, text)
    assert abs(float(without_adapter) - float(want) / 2.5) > 1e-3, "the adapters count in the loss compared"


# --- attention with logits in the thousands ------------------------------------------------------------------------------


@pytest.mark.parametrize("gain", [30, 50])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradients_through_sharp_attention_over_a_bf16_base(tmp_path, seed, gain):
    """The setting a real run is in: a bf16 base whose attention scores are in the tens of thousands, in the blocks
    and in the token refiner (a bf16 score would be off by 4 to 32 in the logit), with float32 matrices on
    their linears.

    * The loss is still the sampling forward's, bit for bit.
    * The gradient is the gradient of the same forward with the attention written out by hand in float32
      (`attention_by_hand`: MLX's plain ops, another backward pass), within 2% as a whole (measured: at most 0.5%).
    * With the sampling forward's attention in the training forward, which is what training ran before, the same
      gradient is more than a hundred times too large or not finite (measured: 640 times at the least, 1e5 to 3e26
      times in the other five cases).

    Printed, not asserted: how far each gradient is from the float32 model's, torch's own bf16 gradient included
    (measured: ours 0.3 to 1.1 of the gradient's size, torch's 0.7 to 33). With attention this sharp a bf16 base is
    not the float32 model: rounding q and k to bf16 moves the logits by several units, and neither torch's gradient
    nor ours stays near the float32 one."""
    s = sharp_setup(seed, gain)
    truth = torch_side(s, tmp_path)[1]
    mixed = to_mixed(s["model"])
    theirs = torch_side(s, tmp_path, base=mixed)[1]
    frozen, trainable = mlx_groups(s, base=mixed)
    args = (s["cfg"], s["example"], to_mx(s["text"]))
    scores = largest_scores(trainable, frozen, *args)
    width = SHARP["attention_head_dim"]
    errors = [bf16_logit_error(score, width) for score in scores]
    assert len(scores) == 1 + SHARP["num_layers"] and min(scores) > 12_000 and min(errors) >= 4, scores

    loss, grads = evaluated(train.loss_and_gradients, trainable, frozen, *args)
    assert all(g.dtype == mx.float32 for g in grads.values())
    on_gpu = mx.default_device() == mx.gpu
    if on_gpu:
        assert float(loss) == float(sampling_forward_loss(lora.join(frozen, trainable), *args))
    with swapped("attention_for_training", attention_by_hand):
        want_loss, want = evaluated(train.loss_and_gradients, trainable, frozen, *args)
    with swapped("attention_for_training", ops.attention):
        _, before = evaluated(train.loss_and_gradients, trainable, frozen, *args)
    keys = sorted(grads)
    reference, exact = flat(want, keys), flat(truth, keys)
    ours, old = rel_err(flat(grads, keys), reference), rel_err(flat(before, keys), reference)
    print(f"gain {gain}, seed {seed}: largest scores {min(scores):,.0f} to {max(scores):,.0f} (a bf16 score is off by "
          f"up to {min(errors):g} to {max(errors):g} in the logit); against the by-hand float32 "
          f"attention, ours {ours:.1e}, the bf16 call {old:.1e}; against the float32 model, ours "
          f"{rel_err(flat(grads, keys), exact):.2f}, torch over the bf16 base {rel_err(flat(theirs, keys), exact):.2f}")
    assert abs(float(loss) - float(want_loss)) <= 1e-5 * float(want_loss)
    assert ours < 2e-2
    if on_gpu:
        assert not old < 1e2, "the bf16 call's backward is what the reference tells apart"


@pytest.mark.parametrize("gain", [30, 50])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradients_through_sharp_attention_in_float32_are_torchs(tmp_path, seed, gain):
    """Float32 throughout, where there is an exact reference: the loss and the whole gradient against torch and PEFT
    with the blocks' scores in the tens of thousands (measured: the gradient within 1e-3; single tensors whose gradient is
    nearly zero, under attention this sharp, are not scored one by one). The refiner is left as it is here: with its
    attention as sharp as the blocks' at a gain of 50, float32 on two machines' arithmetic is itself 10% apart in
    one of these cases, which says how steep such a model is and nothing about either backward pass."""
    s = sharp_setup(seed, gain, refiner=False)
    want_loss, want, _ = torch_side(s, tmp_path)
    frozen, trainable = mlx_groups(s)
    args = (s["cfg"], s["example"], to_mx(s["text"]))
    assert min(largest_scores(trainable, frozen, *args)[1:]) > 16_000
    loss, grads = train.loss_and_gradients(trainable, frozen, *args)
    keys = sorted(grads)
    error = rel_err(flat(grads, keys), flat(want, keys))
    print(f"float32, gain {gain}, seed {seed}: gradient {error:.1e} from torch's")
    assert abs(float(loss) - want_loss) <= 1e-5 * want_loss and error < 5e-3


# --- the gradient of a modulation table ----------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,torch_dtype", [(mx.bfloat16, torch.bfloat16), (mx.float16, torch.float16), (mx.float32, torch.float32)])
def test_the_training_gathers_gradient_is_torchs_index_select(dtype, torch_dtype):
    """Six table rows gathered into 6,000 sequence rows, each multiplied by its own factor and summed: the gradient
    with respect to the table is, per table row, the sum of about a thousand factors.

    The reference module gathers with `index_select`, whose backward returns that sum rounded once to the table's
    dtype. `ops.take_rows_for_training` returns the same bits for a bf16 and a float16 table (on the CPU stream, where
    MLX's sum has a fixed order; on the default device nearly every element). `ops.take_rows`, the sampling
    forward's gather, sums in the table's dtype: over a bf16 table it is about 3% from the exact sum and mostly not
    torch's bits. In the forward the two gathers are the same bit for bit."""
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 6, 6000)
    factors = mx.array((rng.standard_normal((6000, 64)) + 0.25).astype(np.float32)).astype(dtype)
    table = mx.array(rng.standard_normal((6, 64)).astype(np.float32)).astype(dtype)
    special = mx.array(np.float32([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40, 3.4e38, -1.0])).astype(dtype)[:, None]
    for t in (table, special):
        a, b = ops.take_rows_for_training(t, mx.array(rows)), ops.take_rows(t, mx.array(rows))
        assert a.dtype == b.dtype == dtype and a.shape == b.shape
        assert np.array_equal(to_np(a.astype(mx.float32)).view(np.uint32), to_np(b.astype(mx.float32)).view(np.uint32))

    factors_np = to_np(factors.astype(mx.float32))
    exact = np.zeros((6, 64))
    np.add.at(exact, rows, factors_np.astype(np.float64))
    theirs = torch.from_numpy(to_np(table.astype(mx.float32))).to(torch_dtype).requires_grad_(True)
    (theirs.index_select(0, torch.from_numpy(rows)) * torch.from_numpy(factors_np).to(torch_dtype)).float().sum().backward()
    theirs = theirs.grad.float().numpy()

    def gradient(take, device=None):
        f = mx.grad(lambda t: (take(t, mx.array(rows)) * factors).astype(mx.float32).sum())
        if device is None:
            g = f(table)
        else:
            with mx.stream(device):
                g = f(table)
                mx.eval(g)
        assert g.dtype == dtype
        return to_np(g.astype(mx.float32))

    error = lambda g: float(np.linalg.norm(g - exact) / np.linalg.norm(exact))  # noqa: E731
    ours, ours_default, plain = gradient(ops.take_rows_for_training, mx.cpu), gradient(ops.take_rows_for_training), gradient(ops.take_rows, mx.cpu)
    print(f"{dtype}: against the exact sum, torch {error(theirs):.5f}, training gather {error(ours):.5f}, plain gather "
          f"{error(plain):.5f}; equal to torch's bits: {np.mean(ours == theirs):.3f}, {np.mean(ours_default == theirs):.3f} "
          f"on the default device, plain {np.mean(plain == theirs):.3f}")
    assert np.array_equal(ours, theirs), "torch's bits"
    assert error(ours_default) < 1.2 * error(theirs) + 1e-6
    if dtype != mx.float32:  # the default device's float32 sum has no fixed order; rounded to half precision it is theirs
        assert np.mean(ours_default == theirs) > 0.99
    if dtype == mx.bfloat16:
        assert error(theirs) < 0.003 and error(plain) > 0.02 and np.mean(plain == theirs) < 0.5
    if dtype == mx.float16:
        assert error(plain) > 5 * error(theirs) and np.mean(plain == theirs) < 0.5


def test_a_tables_gradient_over_a_bf16_base_is_the_rows_sum():
    """The whole gradient, over a bf16 base, for adapters on the Linears a modulation table's gradient reaches (the
    time embedder, each block's AdaLN projection, the final norm's projection), on an example of about 2,300 rows.

    The reference is the same example with a timestep entry of its own for every row. The tables then have a row per
    sequence row, each used once, so their gather's backward sums nothing: the rows' cotangents are summed later, in
    float32, by the product with the float32 matrices. The values in the forward are the same and so is the loss, bit
    for bit. With shared tables `train.loss_and_gradients` is within 1% of that reference (one rounding of each sum
    to bf16 apart). With the sampling forward's gather in the training forward, which sums in bf16, the same
    gradients are more than 5% away. Every other tensor's gradient is identical three times. On the CPU stream, for
    a fixed order of sums."""
    cfg = dict(TINY)
    torch.manual_seed(0)
    model = to_mixed(randomised(cfg, 0))
    g = torch.Generator().manual_seed(1)
    modules = sorted(lora.wired_shapes(cfg))
    subject = random_adapter(cfg, {m: 4 for m in modules}, {m: 4 for m in modules}, seed=3, std=0.1)
    frozen, trainable = lora.split(lora.attach(state_mx(model), cfg, subject, keep_dtype=True, name="subject"), "subject")
    latents = to_mx(torch.randn(1, 1, 96, 96, cfg["in_channels"], generator=g))
    shared = train.build_example(cfg, np.array([1] * 8), latents, 0.5, Noise(0), condition_t=0.999)
    text = to_mx(torch.randn(1, 8, cfg["text_dim"], generator=g))
    rows = len(shared.layout.token_tags)
    assert rows > 2000 and len(shared.timestep) == 1
    own_row = dataclasses.replace(shared, timestep=shared.timestep[shared.timestep_indices], timestep_indices=np.arange(rows))
    with mx.stream(mx.cpu):
        want_loss, want = train.loss_and_gradients(trainable, frozen, cfg, own_row, text)
        loss, grads = train.loss_and_gradients(trainable, frozen, cfg, shared, text)
        with swapped("take_rows_for_training", ops.take_rows):
            in_bf16 = evaluated(train.loss_and_gradients, trainable, frozen, cfg, shared, text)[1]
        mx.eval(want_loss, want, loss, grads, in_bf16)
    assert float(loss) == float(want_loss)
    fed = [k for k in grads if any(part in k for part in TABLE_FED)]
    rest = [k for k in grads if k not in fed and not k.startswith("audio_proj_out.")]
    assert len(fed) == 2 * (2 + cfg["num_layers"] + 1) and len(rest) > 30
    for k in rest:
        assert np.array_equal(to_np(grads[k]), to_np(want[k])) and np.array_equal(to_np(in_bf16[k]), to_np(want[k])), k
    ours = {k: rel_err(grads[k], want[k]) for k in fed}
    summed_in_bf16 = {k: rel_err(in_bf16[k], want[k]) for k in fed}
    print(f"{rows} rows: table-fed gradients against the one-row-per-row reference, ours {max(ours.values()):.4f} at "
          f"worst, summed in bf16 {min(summed_in_bf16.values()):.3f} to {max(summed_in_bf16.values()):.3f}")
    assert max(ours.values()) < 0.01
    assert min(summed_in_bf16.values()) > 0.05, "the reference tells the two sums apart"


# --- what a loss refuses, and a quantized base ----------------------------------------------------------------------------


def test_the_loss_refuses_matrices_the_forward_would_not_read():
    """Each of these ran before the check, with no error, and returned gradients of exactly zero for matrices the
    forward did not read: it reads a named adapter's matrices only where the Linear lists the name as active. They
    are refused now, by the loss and by its gradient."""
    s = setup(1, 0.5)
    cfg, example, text = s["cfg"], s["example"], to_mx(s["text"])
    with_assistant = lora.attach(state_mx(s["model"]), cfg, s["assistant"], name="assistant")
    both = lora.attach(with_assistant, cfg, s["subject"], keep_dtype=True, name="subject")
    cases = {
        "a file's keys beside a named adapter": (s["subject"].tensors, with_assistant, "a file's keys are not read"),
        "runtime names never attached": (lora.runtime_tensors(s["subject"], "subject"), with_assistant, "is not active on"),
        "an adapter switched off": (*reversed(lora.split(lora.set_active(both, ("assistant",)), "subject")), "is not active on"),
        "nothing to train": ({}, lora.split(both, "subject")[0], "`trainable` is empty"),
    }
    for what, (trainable, frozen, message) in cases.items():
        for function in (train.example_loss, train.loss_and_gradients):
            with pytest.raises(ValueError, match=message):
                function(trainable, frozen, cfg, example, text)
    # what the first three would have returned: a loss that does not depend on the matrices. (A file's keys are the
    # exception on a Linear with no named adapter: there they are read, as an unnamed adapter whose scaling is 1.)
    shared = tuple(s["assistant"].modules)
    for what in list(cases)[:3]:
        trainable, frozen, _ = cases[what]
        ignored = mx.grad(lambda t: sampling_forward_loss({**frozen, **t}, cfg, example, text))(dict(trainable))
        unread = [k for k in ignored if what != "a file's keys beside a named adapter" or k.startswith(shared)]
        assert len(unread) >= 24 and all(not to_np(ignored[k]).any() for k in unread), what
        assert len(unread) == len(ignored) or any(to_np(ignored[k]).any() for k in set(ignored) - set(unread))


def test_the_loss_refuses_a_text_of_the_wrong_shape():
    """The forward scatters the text rows into the packed sequence, and that assignment broadcasts: one row of text
    would fill every text position without an error. The loss takes exactly (1, the layout's text rows, text dim)."""
    s = setup(1, 0.5)
    frozen, trainable = mlx_groups(s)
    cfg, example, text = s["cfg"], s["example"], to_mx(s["text"])
    rows = len(example.layout.text_indices)
    assert text.shape == (1, rows, cfg["text_dim"]) and rows > 1
    for bad in (text[:, :1], text[:, :-1], text[0], mx.concatenate([text, text]), text[:, :, :-1], s["text"].numpy()):
        for function in (train.example_loss, train.loss_and_gradients):
            with pytest.raises(ValueError, match="text must be an array of shape"):
                function(trainable, frozen, cfg, example, bad)
    one_row = sampling_forward_loss({**frozen, **trainable}, cfg, example, text[:, :1])  # what the check is for
    assert np.isfinite(float(one_row)) and float(one_row) != float(train.example_loss(trainable, frozen, cfg, example, text))


def test_gradients_over_a_quantized_base():
    """The base's Linear weights 8-bit quantized (`ops.QuantizedWeight`), inside the frozen group: the loss and the
    trainable matrices' gradients are those over the same base with each quantized weight replaced by the matrix it
    stands for. The quantized objects are constants of the differentiation like any other frozen entry."""
    s = setup(1, 0.5, cfg=dict(TINY, hidden_size=64, ffn_dim=96))  # widths a group of 32 divides
    frozen, trainable = mlx_groups(s)
    cfg, example, text = s["cfg"], s["example"], to_mx(s["text"])
    quantized, dense = dict(frozen), dict(frozen)
    for k, v in frozen.items():
        if k.endswith(".weight") and ".lora_" not in k and v.ndim == 2 and v.shape[1] % 32 == 0:
            quantized[k] = ops.QuantizedWeight(v, 8, 32)
            dense[k] = mx.dequantize(quantized[k].wq, quantized[k].scales, quantized[k].biases, group_size=32, bits=8)
    count = sum(isinstance(v, ops.QuantizedWeight) for v in quantized.values())
    assert count >= len(lora.block_modules(cfg)) and all(k in quantized and k in dense for k in frozen)
    want_loss, want = train.loss_and_gradients(trainable, dense, cfg, example, text)
    loss, grads = train.loss_and_gradients(trainable, quantized, cfg, example, text)
    plain_loss, plain = train.loss_and_gradients(trainable, frozen, cfg, example, text)
    assert abs(float(loss) - float(want_loss)) <= 1e-5 * float(want_loss)
    worst = max(rel_err(grads[k], want[k]) for k in grads)
    moved = rel_err(flat(grads, sorted(grads)), flat(plain, sorted(grads)))
    print(f"{count} quantized weights: worst gradient tensor {worst:.1e} from the dequantized base's, {moved:.1e} from the unquantized")
    assert set(grads) == set(trainable) and worst < 1e-4
    assert all(float(mx.abs(g).max()) > 0 for g in grads.values())
    assert moved > 1e-4, "8 bits is not the original weights: the comparison is against what the quantized weight holds"
