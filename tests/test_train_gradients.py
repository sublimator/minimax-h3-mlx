"""The loss of a training example and its gradient on the tiny transformer, against torch autograd and PEFT.

Torch side: the diffusers transformer with PEFT's layers, a frozen assistant adapter and a trainable one whose
parameters are float32 (`cast_training_params`), fed the example that diffusers' own functions assemble
(`test_train_example.theirs`), `mse_loss` on the generated video rows, `backward`. Ours: `train.loss_and_gradients`
with the weights as two arguments. In float32 the two agree closely; over a bf16 base each is scored against the
float32 run, as the forward's mixed-precision tests do.
"""

import dataclasses

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


def torch_side(s, tmp_path, base=None, stacked=True, target=None):
    """(loss, {parameter name: gradient or None}, the PEFT model): the subject's parameters trainable and float32,
    everything else frozen."""
    names = ("assistant", "subject") if stacked else ("subject",)
    m = activate(peft_stack(s["model"] if base is None else base, tmp_path, s["assistant"], s["subject"]), names)
    for name, prm in m.named_parameters():
        prm.requires_grad_(".lora_" in name and ".subject." in name)
    cast_training_params(m, dtype=torch.float32)
    w = s["want"]
    video_pred, _ = m(w["video_rows"][None], w["audio_rows"][None], s["text"], w["timestep"], w["indices"],
                      token_tags=w["token_tags"], position_ids=w["position_ids"], video_indices=w["video_indices"],
                      audio_indices=w["audio_indices"], text_indices=w["text_indices"], return_dict=False)
    target = w["target"] if target is None else target
    loss = torch.nn.functional.mse_loss(video_pred[:, w["n_cond"]:].float(), target[None])
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
