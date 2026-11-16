"""The forward process, the velocity target and the loss, against diffusers' MiniMaxH3Scheduler and torch.

`scale_noise` is the scheduler's own; the target has no function to compare with in diffusers, so it is pinned by what
the scheduler does with it: one Euler step from `x_t` along the target, to sigma 0, must give back `x_0`. The loss is
torch's `mse_loss`; the weighted form is written out in torch here.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, to_mx, to_np
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from test_dit_mixed_precision import rel_err
from test_lora import to_mx_half

from h3mlx import ops, train

SHAPE = (3, 40, 16)  # examples, rows, channels


def draws(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(SHAPE, generator=g).to(dtype), torch.randn(SHAPE, generator=g).to(dtype)


# --- the forward process -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.7, 0.999, 1.0])
def test_scale_noise_with_a_float_is_the_schedulers(t, dtype):
    x0, noise = draws(dtype=dtype)
    want = MiniMaxH3Scheduler().scale_noise(x0, t, noise)
    got = train.scale_noise(to_mx_half(x0), t, to_mx_half(noise))
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    assert np.array_equal(to_np(got), to_np(want))
    if t == 1.0:
        assert np.array_equal(to_np(got), to_np(x0)), "t = 1 is clean"
    if t == 0.0:
        assert np.array_equal(to_np(got), to_np(noise)), "t = 0 is noise"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (3,), (3, 1)])
def test_scale_noise_with_an_array_is_the_schedulers(shape, dtype):
    """A 0-d timestep, one per example, and one already shaped for the rows; `1 - t` in the sample's dtype."""
    x0, noise = draws(1, dtype)
    t = torch.tensor([0.999, 0.25, 0.6])[: max(1, int(np.prod(shape)))].reshape(shape)
    want = MiniMaxH3Scheduler().scale_noise(x0, t, noise)
    got = train.scale_noise(to_mx(x0), to_mx(t), to_mx(noise))
    assert np.array_equal(to_np(got), to_np(want))


def test_float_and_array_timesteps_differ_as_in_the_scheduler():
    """The two paths are not the same arithmetic: a float keeps `1 - t` in double precision, an array takes it in
    float32. The array path is the expression `image.generate` noises its pictures with."""
    x0, noise = (to_mx(a) for a in draws(2))
    as_array = train.scale_noise(x0, mx.array(0.999), noise)
    t32 = np.float32(0.999)
    assert np.array_equal(to_np(as_array), to_np(mx.array(t32) * x0 + mx.array(np.float32(1) - t32) * noise))
    as_float = train.scale_noise(x0, 0.999, noise)
    assert not np.array_equal(to_np(as_float), to_np(as_array))
    assert_close(as_float, as_array, atol=1e-6, what="the same up to the float32 rounding of 1 - t")


# --- the target --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0.0, 0.25, 0.7, 0.999])
def test_one_euler_step_along_the_target_lands_on_x0(t):
    """Noise a sample to `t`, take the target as the model's output, and step to sigma 0: the scheduler (`step`) and
    our `flow_step` both return `x_0`, to float32 rounding. With the target's sign flipped (`noise - x_0`, the target
    under the usual flow-match convention `x_0 = x_t - sigma * v`) the same step lands at `2 * x_t - x_0`."""
    x0, noise = draws(3)
    sigma = 1.0 - t
    tol = 4e-6  # a few float32 steps at the size of these values (|x| up to about 4)

    def step_to_zero(v, x):  # a two-point grid [sigma, 0], taken as given: one step, and it ends at sigma 0
        sch = MiniMaxH3Scheduler(shift_provided_sigmas=False)
        sch.set_timesteps(sigmas=torch.tensor([sigma, 0.0]))
        return sch.step(v, sch.timesteps[0], x, return_dict=False)[0], sch

    x_t = MiniMaxH3Scheduler().scale_noise(x0, t, noise)
    target = to_torch(train.velocity_target(to_mx(x0), to_mx(noise)))
    landed, sch = step_to_zero(target, x_t)
    assert_close(landed, x0, atol=tol, what="the scheduler's step along our target")

    m_xt = train.scale_noise(to_mx(x0), t, to_mx(noise))
    m_target = train.velocity_target(to_mx(x0), to_mx(noise))
    m_landed = ops.flow_step(m_target, float(sch.timesteps[0]), m_xt, float(sch.sigmas[0]), 0.0)
    assert_close(m_landed, x0, atol=tol, what="flow_step along our target")

    # the counterexample: the other sign misses by 2 * (1 - t) * (x_0 - noise), far outside the tolerance
    flipped, _ = step_to_zero(-target, x_t)
    assert_close(flipped, 2 * x_t - x0, atol=4 * tol, what="where the flipped target lands")
    assert float((flipped - x0).abs().max()) > 1000 * tol
    m_flipped = ops.flow_step(-m_target, float(sch.timesteps[0]), m_xt, float(sch.sigmas[0]), 0.0)
    assert float(mx.abs(m_flipped - to_mx(x0)).max()) > 1000 * tol


def test_target_is_the_velocity_the_step_equation_implies():
    """Solve the scheduler's `x_0 = x_t + (1 - t) * v` for `v` in float64, without our target: it is `x_0 - noise`
    for every `t`, and it is not `noise - x_0`."""
    x0, noise = draws(4)
    target = to_np(train.velocity_target(to_mx(x0), to_mx(noise)))
    for t in (0.0, 0.3, 0.9):
        x_t = to_np(MiniMaxH3Scheduler().scale_noise(x0, t, noise)).astype(np.float64)
        v = (x0.numpy().astype(np.float64) - x_t) / (1.0 - t)
        assert_close(target, v, atol=2e-5, what=f"t = {t}")
        assert rel_err(-target, v) > 1.9, "the flipped target is the negative: a relative error of 2"


# --- the loss ----------------------------------------------------------------------------------------------------------


def to_torch(a):
    return torch.from_numpy(to_np(a))


@pytest.mark.parametrize("pred_dtype,target_dtype", [(torch.float32, torch.float32), (torch.bfloat16, torch.float32),
                                                     (torch.bfloat16, torch.bfloat16)])
def test_loss_is_torchs_mse(pred_dtype, target_dtype):
    """In float32 whatever arrives: the prediction comes in the model's dtype. Squaring and averaging in bf16 would be
    off in the third digit."""
    pred, target = draws(5)
    pred, target = pred.to(pred_dtype), target.to(target_dtype)
    want = torch.nn.functional.mse_loss(pred.float(), target.float())
    got = train.flow_match_loss(to_mx(pred), to_mx(target))
    assert got.dtype == mx.float32 and got.shape == ()
    assert_close(got, want, atol=0, rtol=2e-6, what="unweighted")
    assert float(train.flow_match_loss(to_mx(target), to_mx(target))) == 0.0


@pytest.mark.parametrize("weight", [2.5, [0.5, 2.0, 7.0]])
def test_weighted_loss(weight):
    """The weighted mean a per-example weighting needs, written out in torch: each example's mean of w * err^2, then
    the mean over examples. A weight per example must stay with its example."""
    pred, target = draws(6)
    w = torch.tensor(weight)
    sq = (w.reshape(-1, 1, 1) if w.ndim else w) * (pred.float() - target.float()) ** 2
    want = sq.reshape(SHAPE[0], -1).mean(1).mean()
    got = train.flow_match_loss(to_mx(pred), to_mx(target), weight if isinstance(weight, float) else to_mx(w))
    assert_close(got, want, atol=0, rtol=2e-6, what="weighted")
    if w.ndim:
        reversed_w = train.flow_match_loss(to_mx(pred), to_mx(target), to_mx(w.flip(0)))
        assert abs(float(reversed_w) - float(want)) > 1e-3 * float(want), "the guard: weights on the wrong examples"
        assert abs(float(train.flow_match_loss(to_mx(pred), to_mx(target))) - float(want)) > 0.1 * float(want)


def test_loss_gradient_is_torchs():
    pred, target = draws(7)
    w = torch.tensor([0.5, 2.0, 7.0])
    p = pred.clone().requires_grad_(True)
    ((w.reshape(-1, 1, 1) * (p - target) ** 2).mean()).backward()
    got = mx.grad(lambda x: train.flow_match_loss(x, to_mx(target), to_mx(w)))(to_mx(pred))
    assert_close(got, p.grad, atol=1e-9, rtol=2e-6, what="d loss / d pred")


def test_loss_sees_the_sign_of_the_target():
    """A model that predicts the step equation's velocity exactly has zero loss against our target, and a loss of
    4 * mean((x_0 - noise)^2) against the flipped one."""
    x0, noise = draws(8)
    t = 0.3
    x_t = to_np(MiniMaxH3Scheduler().scale_noise(x0, t, noise)).astype(np.float64)
    perfect = mx.array(((x0.numpy().astype(np.float64) - x_t) / (1.0 - t)).astype(np.float32))
    target = train.velocity_target(to_mx(x0), to_mx(noise))
    assert float(train.flow_match_loss(perfect, target)) < 1e-10
    flipped = float(train.flow_match_loss(perfect, -target))
    assert abs(flipped - 4 * float(((x0 - noise) ** 2).mean())) < 1e-3 * flipped and flipped > 1.0
