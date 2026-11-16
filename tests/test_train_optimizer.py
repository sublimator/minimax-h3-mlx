"""The optimizer step (`train.adamw`, `train.optimizer_step`) against torch: `torch.optim.AdamW` after
`clip_grad_norm_`, on the tiny transformer with PEFT's layers, the same examples in the same order on both sides.
Also here, each with a minimal test: MLX's AdamW without bias correction is not torch's, and half-precision matrices
do not take updates of the size a trainer makes.
"""

import inspect
import math

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
import torch
from conftest import to_mx, to_np
from test_dit_mixed_precision import to_mixed
from test_dit_tiny import TINY
from test_train_example import TAGS, latents_of, ours, theirs
from test_train_gradients import mlx_groups, setup, torch_loss, torch_model

from h3mlx import lora, train

STEPS = 12
EXAMPLES = [(1, 0.5), (0, 0.9), (2, 0.2)]  # (reference pictures, sigma), cycled: the gradient's norm differs between them
# two settings with every value different, so that none of them can be wired to a constant; the second has an epsilon
# of the gradients' size, where the update is no longer indifferent to the gradient's scale, and so to the clipping
SETTINGS = {
    "torch's defaults but the rate": dict(learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
    "all different": dict(learning_rate=3e-4, betas=(0.8, 0.95), eps=1e-3, weight_decay=0.1),
}
# the bound on the gradient's norm, set between the norms the three examples give from each start (0.06 to 0.17 from
# PEFT's initialisation, 0.32 to 0.62 from random matrices), so that some steps are clipped and some are not
MAX_NORM = {("torch's defaults but the rate", "PEFT's initialisation"): 0.1, ("all different", "PEFT's initialisation"): 0.09,
            ("torch's defaults but the rate", "both matrices random"): 0.5, ("all different", "both matrices random"): 0.45}
TOLERANCE = 1e-4  # the matrices' movement since the start, relative, against torch's; the loss within 1e-5


def examples(seed=2):
    """The same pictures both ways: ours (`train.build_example`) and from diffusers' functions (`theirs`)."""
    out = []
    for i, (n_refs, sigma) in enumerate(EXAMPLES):
        picture, refs = latents_of(n_refs, seed=seed + i)
        out.append((ours(picture, refs, sigma, seed=seed + i), theirs(picture, refs, sigma, seed=seed + i)))
    return out


def torch_steps(s, tmp_path, data, steps, *, learning_rate, betas, eps, weight_decay, max_grad_norm, base=None):
    """(losses, gradient norms before clipping, the trainable parameters after each step) of torch's loop."""
    m = torch_model(s, tmp_path, base)
    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(params.values(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    losses, norms, after = [], [], []
    for i in range(steps):
        opt.zero_grad()
        loss = torch_loss(m, data[i % len(data)][1], s["text"])
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(params.values(), max_grad_norm)))
        opt.step()
        losses.append(float(loss.detach()))
        after.append({n: p.detach().clone().numpy() for n, p in params.items()})
    return losses, norms, after


def mlx_steps(s, data, steps, optimizer, *, max_grad_norm, base=None):
    frozen, trainable = mlx_groups(s, base=base)
    text = to_mx(s["text"])
    losses, norms, after = [], [], []
    for i in range(steps):
        loss, grads = train.loss_and_gradients(trainable, frozen, s["cfg"], data[i % len(data)][0], text)
        trainable, norm = train.optimizer_step(optimizer, trainable, grads, max_grad_norm=max_grad_norm)
        losses.append(float(loss))
        norms.append(norm)
        after.append({k: to_np(v) for k, v in trainable.items()})
    return losses, norms, after


def movement(after, start, keys):
    return np.concatenate([(after[k] - start[k]).ravel() for k in keys])


def apart(got, want):
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# --- N steps -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("start", ["PEFT's initialisation", "both matrices random"])
def test_steps_are_torchs(tmp_path, setting, start):
    """Twelve steps over three examples in turn, float32, a frozen assistant beside the trained adapter: after every
    step the loss is torch's to 1e-5, the gradient's norm before clipping to 1e-5, and how far the matrices have
    moved from their start to 1e-4 (relative, over all of them). The clipping bites on some steps and not on others,
    so it is part of what is compared. From PEFT's initialisation (`lora_B` zero), where training starts, and from
    random matrices."""
    hp, max_norm = SETTINGS[setting], MAX_NORM[setting, start]
    subject = lora.init(TINY, rank=3, alpha=6, seed=1) if start.startswith("PEFT") else None
    s = setup(1, 0.5, subject=subject, std=0.1)
    data = examples()
    want_losses, want_norms, want = torch_steps(s, tmp_path, data, STEPS, **hp, max_grad_norm=max_norm)
    losses, norms, got = mlx_steps(s, data, STEPS, train.adamw(**hp), max_grad_norm=max_norm)
    start_values = {k: to_np(v) for k, v in lora.runtime_tensors(s["subject"], "subject").items()}
    keys = sorted(start_values)
    worst = 0.0
    for step in range(STEPS):
        assert abs(losses[step] - want_losses[step]) <= 1e-5 * want_losses[step], step
        assert abs(norms[step] - want_norms[step]) <= 1e-5 * want_norms[step], step
        worst = max(worst, apart(movement(got[step], start_values, keys), movement(want[step], start_values, keys)))
    clipped = [n > max_norm for n in want_norms]
    moved = float(np.abs(movement(want[-1], start_values, keys)).max())
    print(f"{setting}, {start}: movement {worst:.1e} from torch's at worst over {STEPS} steps (largest element moved "
          f"{moved:.1e}); first example's loss {want_losses[0]:.4f} to {want_losses[9]:.4f}; clipped on {sum(clipped)} "
          f"steps of {STEPS}; norms {[round(n, 3) for n in want_norms]}")
    assert worst < TOLERANCE
    assert any(clipped) and not all(clipped), "the clipping is on for some steps and off for others"
    assert moved > 5 * hp["learning_rate"], "the matrices moved"
    assert want_losses[9] < want_losses[0] and losses[9] < losses[0], "and the first example's loss fell, on both sides"
    assert all(v.dtype == np.float32 for v in got[-1].values()) and set(got[-1]) == set(want[-1])


def test_steps_over_a_bf16_base_stay_near_torchs(tmp_path):
    """The same loop over a base in the release's mixed precision, the trained matrices float32. The two sides'
    gradients are 1% apart there (bf16 arithmetic is not the same in MLX and torch), and Adam's first steps move
    every element by the learning rate in the direction of its gradient's sign, so an element whose gradient is near
    zero can go the other way. What is asserted is loose on purpose: the losses within 0.5% at every step, and the
    movement within 25% of torch's (measured 3 to 12%)."""
    hp = SETTINGS["torch's defaults but the rate"]
    s = setup(1, 0.5, std=0.1)
    data = examples()
    mixed = to_mixed(s["model"])
    want_losses, _, want = torch_steps(s, tmp_path, data, STEPS, **hp, max_grad_norm=0.5, base=mixed)
    losses, _, got = mlx_steps(s, data, STEPS, train.adamw(**hp), max_grad_norm=0.5, base=mixed)
    start_values = {k: to_np(v) for k, v in lora.runtime_tensors(s["subject"], "subject").items()}
    keys = sorted(start_values)
    errors = [apart(movement(got[i], start_values, keys), movement(want[i], start_values, keys)) for i in range(STEPS)]
    loss_errors = [abs(a - b) / b for a, b in zip(losses, want_losses)]
    print(f"bf16 base: movement {min(errors):.3f} to {max(errors):.3f} from torch's, losses within {max(loss_errors):.1e}")
    assert max(errors) < 0.25 and max(loss_errors) < 5e-3 and max(errors) > 10 * TOLERANCE


# --- the clipping --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_norm", [0.05, 0.5, 5.0, math.inf])
def test_clipping_is_torchs(max_norm):
    """One step of plain gradient descent shows the clipped gradients themselves: the matrices move by the learning
    rate times torch's gradients after `clip_grad_norm_`, and the norm returned is the one torch returns, taken
    before clipping. Below the norm, above it, and with no bound."""
    g = torch.Generator().manual_seed(0)
    values = {f"m{i}.lora_{x}.w": torch.randn(shape, generator=g) for i, shape in enumerate([(3, 8), (8, 3), (2, 5)]) for x in "AB"}
    grads = {k: torch.randn(v.shape, generator=g) * 0.1 for k, v in values.items()}
    parameters = [torch.nn.Parameter(v.clone()) for v in values.values()]
    for p, gradient in zip(parameters, grads.values()):
        p.grad = gradient.clone()
    want_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))
    assert 0.05 < want_norm < 5.0
    updated, norm = train.optimizer_step(optim.SGD(learning_rate=1.0), {k: to_mx(v) for k, v in values.items()},
                                         {k: to_mx(v) for k, v in grads.items()}, max_grad_norm=max_norm)
    assert isinstance(norm, float) and abs(norm - want_norm) <= 1e-6 * want_norm
    for (k, v), p in zip(values.items(), parameters):
        applied = v.numpy() - to_np(updated[k])  # the gradient that was applied, at a learning rate of 1
        assert np.allclose(applied, p.grad.numpy(), rtol=1e-5, atol=1e-7), k
    scale = float(np.linalg.norm(np.concatenate([p.grad.numpy().ravel() for p in parameters]))) / want_norm
    assert abs(scale - min(1.0, max_norm / want_norm)) < 1e-5


# --- two facts about the optimizer -----------------------------------------------------------------------------------------


def test_mlx_adamw_needs_bias_correction_to_be_torchs():
    """MLX's AdamW leaves the bias correction out unless asked (`bias_correction=False` is its default), and torch's
    always applies it. On the first step torch moves every element by the learning rate; MLX's default moves it by
    `(1 - beta1) / sqrt(1 - beta2)` times that, 3.16 times for betas of (0.9, 0.999). With `bias_correction=True`,
    which is what `train.adamw` passes, five steps are torch's."""
    hp = dict(betas=(0.9, 0.999), eps=1e-12, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    start = torch.randn(6, 5, generator=g)
    gradients = [torch.randn(6, 5, generator=g) for _ in range(5)]

    p = torch.nn.Parameter(start.clone())
    opt = torch.optim.AdamW([p], lr=1e-3, **hp)
    theirs = []
    for gradient in gradients:
        p.grad = gradient.clone()
        opt.step()
        theirs.append(p.detach().clone().numpy())

    def run(optimizer):
        w, out = {"w": to_mx(start)}, []
        for gradient in gradients:
            w = optimizer.apply_gradients({"w": to_mx(gradient)}, w)
            out.append(to_np(w["w"]))
        return out

    assert optim.AdamW(learning_rate=1e-3).bias_correction is False, "MLX's default"
    default = run(optim.AdamW(learning_rate=1e-3, betas=list(hp["betas"]), eps=hp["eps"], weight_decay=0.0))
    corrected = run(train.adamw(learning_rate=1e-3, **hp))
    first = np.abs(default[0] - start.numpy()) / np.abs(theirs[0] - start.numpy())
    assert np.allclose(first, (1 - 0.9) / math.sqrt(1 - 0.999), rtol=1e-3) and 3.1 < first.mean() < 3.2
    for step in range(5):
        assert apart(corrected[step] - start.numpy(), theirs[step] - start.numpy()) < 1e-5, step
        assert apart(default[step] - start.numpy(), theirs[step] - start.numpy()) > 0.5, step


def adamw_moves(dtype, low, high, steps=10, learning_rate=1e-4):
    """(values before, after) `steps` of MLX's AdamW, bias-corrected, on elements of magnitude in [low, high) and
    either sign, with a fixed gradient of about 1e-2 an element."""
    rng = np.random.default_rng(0)
    magnitudes = rng.uniform(low, high, (48, 48)) if high > low else np.full((48, 48), low)
    start = mx.array((magnitudes * rng.choice([-1.0, 1.0], magnitudes.shape)).astype(np.float32)).astype(dtype)
    gradient = mx.array((rng.standard_normal(magnitudes.shape) * 1e-2).astype(np.float32)).astype(dtype)
    optimizer = optim.AdamW(learning_rate=learning_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, bias_correction=True)
    w = {"w": start}
    for _ in range(steps):
        w = optimizer.apply_gradients({"w": gradient}, w)
    return to_np(start.astype(mx.float32)), to_np(w["w"].astype(mx.float32))


def test_half_precision_matrices_do_not_take_updates_of_1e_4():
    """Why the trained matrices are float32. Ten AdamW steps at a learning rate of 1e-4, a fixed gradient:

    * float32: every element moves by 1e-3, ten steps' worth.
    * bf16, magnitudes of 2**-5 (0.031) and up: not one element changes. The spacing of bf16 values there is 2.4e-4
      or more, and an update of 1e-4 is under half of it, so every step rounds back to where it started.
    * bf16, magnitudes in [2**-6, 2**-5): every element moves, by one whole spacing (1.2e-4) a step: 22% too far.
    * bf16 from zero, which is `lora_B`'s start: moves by the learning rate. Small elements are not stuck; they are
      coarse, and they get stuck as they grow.
    * float16: the second moment of a 1e-2 gradient, 1e-7, is below float16's smallest normal number and its square
      root's reciprocal overflows: the update is NaN or infinite for most elements.

    `train.optimizer_step` refuses anything but float32."""
    before, after = adamw_moves(mx.float32, 2.0 ** -5, 1.0)
    assert np.allclose(np.abs(after - before), 1e-3, rtol=0.02)
    before, after = adamw_moves(mx.bfloat16, 2.0 ** -5, 1.0)
    assert np.array_equal(before, after), "stuck"
    before, after = adamw_moves(mx.bfloat16, 2.0 ** -6, 2.0 ** -5, steps=1)
    moved = np.abs(after - before)
    assert (moved > 0).mean() > 0.99 and np.allclose(moved[moved > 0], 2.0 ** -13, rtol=1e-6), "one spacing, not 1e-4"
    before, after = adamw_moves(mx.bfloat16, 0.0, 0.0, steps=1)
    assert np.allclose(np.abs(after - before), 1e-4, rtol=0.03)
    before, after = adamw_moves(mx.float16, 2.0 ** -5, 1.0, steps=1)
    assert (~np.isfinite(after)).mean() > 0.25


# --- what the step refuses, and no defaults -----------------------------------------------------------------------------


def test_optimizer_step_refusals():
    w = {"a.lora_A.x.weight": mx.ones((2, 3)), "a.lora_B.x.weight": mx.ones((3, 2))}
    g = {k: mx.ones(v.shape) * 0.1 for k, v in w.items()}
    make = lambda: train.adamw(learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)  # noqa: E731
    for dtype in (mx.bfloat16, mx.float16):
        with pytest.raises(ValueError, match="must be float32"):
            train.optimizer_step(make(), {k: v.astype(dtype) for k, v in w.items()}, g, max_grad_norm=1.0)
    with pytest.raises(ValueError, match="keys must be the trainable matrices' keys"):
        train.optimizer_step(make(), w, {"a.lora_A.x.weight": g["a.lora_A.x.weight"]}, max_grad_norm=1.0)
    with pytest.raises(ValueError, match="keys must be the trainable matrices' keys"):
        train.optimizer_step(make(), {}, {}, max_grad_norm=1.0)
    for bad in (0.0, -1.0, math.nan, None, "1", True):
        with pytest.raises(ValueError, match="max_grad_norm must be a positive number"):
            train.optimizer_step(make(), w, g, max_grad_norm=bad)
    for bad in (math.nan, math.inf):
        poisoned = {**g, "a.lora_B.x.weight": g["a.lora_B.x.weight"] * bad}
        optimizer = make()
        with pytest.raises(ValueError, match="norm is .*: not applied"):
            train.optimizer_step(optimizer, w, poisoned, max_grad_norm=1.0)
        assert optimizer.state.get("step", 0) == 0, "the optimizer's state is untouched"
    updated, norm = train.optimizer_step(make(), w, g, max_grad_norm=math.inf)  # and a good step goes through
    assert set(updated) == set(w) and all(not np.array_equal(to_np(updated[k]), to_np(w[k])) for k in w)
    assert abs(norm - 0.1 * math.sqrt(12)) < 1e-6


def test_no_hyper_parameter_has_a_default():
    """Every hyper-parameter is the caller's: keyword-only, no default. (The recipe's values for them were not read.)"""
    for function, names in ((train.adamw, {"learning_rate", "betas", "eps", "weight_decay"}),
                            (train.optimizer_step, {"max_grad_norm"})):
        parameters = inspect.signature(function).parameters
        keyword_only = {n for n, p in parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}
        assert keyword_only == names and all(parameters[n].default is inspect.Parameter.empty for n in names)
    with pytest.raises(TypeError):
        train.adamw(learning_rate=1e-4)
    with pytest.raises(TypeError):
        train.optimizer_step(train.adamw(learning_rate=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0), {}, {})
    assert train.adamw(learning_rate=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).bias_correction is True
