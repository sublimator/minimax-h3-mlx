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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", [np.float32, np.float64, int])
def test_scale_noise_with_a_numpy_scalar_is_the_schedulers(kind, dtype):
    """A numpy scalar is a number to the scheduler, not a tensor: `1 - t` is taken in the scalar's own precision and
    each product is tensor-times-number. Bit for bit here too."""
    x0, noise = draws(9, dtype)
    for value in ((0, 1) if kind is int else (0.0, 0.25, 0.3, 0.7, 0.999, 1.0)):
        want = MiniMaxH3Scheduler().scale_noise(x0, kind(value), noise)
        got = train.scale_noise(to_mx_half(x0), kind(value), to_mx_half(noise))
        assert isinstance(want, torch.Tensor) and want.dtype == dtype
        assert np.array_equal(to_np(got), to_np(want)), value


def test_a_number_and_a_tensor_of_the_same_value_differ_in_half_precision():
    """On a bf16 sample, `np.float32(0.999)` multiplies in float32 and rounds once, while a 0-d tensor of the same
    value is first cast to bf16, where 0.999 is 1.0 and `1 - t` is 0. The scheduler does the same; both are kept."""
    x0 = torch.zeros(1, 4, dtype=torch.bfloat16)
    noise = torch.full((1, 4), 100.0, dtype=torch.bfloat16)
    sch = MiniMaxH3Scheduler()
    for timestep, m_timestep in ((np.float32(0.999), np.float32(0.999)), (torch.tensor(0.999), mx.array(np.float32(0.999)))):
        want = sch.scale_noise(x0, timestep, noise)
        got = train.scale_noise(to_mx(x0), m_timestep, to_mx(noise))
        assert np.array_equal(to_np(got), to_np(want))
    assert float(sch.scale_noise(x0, np.float32(0.999), noise)[0, 0]) > 0.09
    assert float(sch.scale_noise(x0, torch.tensor(0.999), noise)[0, 0]) == 0.0


def test_scale_noise_refuses_a_numpy_array_as_the_scheduler_does():
    """The scheduler has no path for a numpy array: its `timestep * sample` is a TypeError, 0-d or not. Ours raises
    one that says what to do. Wrapped in `mx.array`, the timesteps `sample_sigmas` leads to are a tensor's."""
    x0, noise = draws(10)
    sch = MiniMaxH3Scheduler()
    for timestep in (np.float32([0.2, 0.7, 0.5]), np.float32([0.5]), np.array(0.999, dtype=np.float32)):
        with pytest.raises(TypeError):
            sch.scale_noise(x0, timestep, noise)
        with pytest.raises(TypeError, match=r"mx\.array\(timesteps\)"):
            train.scale_noise(to_mx(x0), timestep, to_mx(noise))
    sigmas, _ = train.sample_sigmas(np.float32([0.2, 0.7, 0.01]), train.training_sigmas(1000, 12.0))
    timesteps = np.float32(1) - sigmas
    want = sch.scale_noise(x0, torch.from_numpy(timesteps), noise)
    got = train.scale_noise(to_mx(x0), mx.array(timesteps), to_mx(noise))
    assert np.array_equal(to_np(got), to_np(want)) and not np.array_equal(to_np(got), to_np(x0))


def test_scale_noise_timestep_axes():
    """More axes than the sample: the scheduler broadcasts into a result larger than the sample, which is never what
    noising one sample means; ours refuses. Several timesteps against one example (a leading axis of 1) is ordinary
    broadcasting on both sides and is kept."""
    sch = MiniMaxH3Scheduler()
    too_many = torch.tensor([[[0.5]], [[0.25]]])
    assert sch.scale_noise(torch.ones(8, 16), too_many, torch.zeros(8, 16)).shape == (2, 8, 16)
    with pytest.raises(ValueError, match="more axes than the sample"):
        train.scale_noise(mx.ones((8, 16)), to_mx(too_many), mx.zeros((8, 16)))
    x0, noise = draws(11)
    three = torch.tensor([0.5, 0.2, 0.1])
    want = sch.scale_noise(x0[:1], three, noise[:1])
    got = train.scale_noise(to_mx(x0[:1]), to_mx(three), to_mx(noise[:1]))
    assert got.shape == (3,) + SHAPE[1:] and np.array_equal(to_np(got), to_np(want))


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


def test_loss_refuses_shapes_that_do_not_line_up():
    """A prediction that still holds nine condition rows against a one-row target: torch's `mse_loss` warns and
    broadcasts the target over all ten rows. Ours raises. The weight is checked the same way: an axis of it must be 1
    or the size of the leading axis it meets."""
    pred, target = torch.ones(1, 10, 16), torch.zeros(1, 1, 16)
    with pytest.warns(UserWarning, match="target size"):
        assert float(torch.nn.functional.mse_loss(pred, target)) == 1.0
    with pytest.raises(ValueError, match="differ in shape"):
        train.flow_match_loss(to_mx(pred), to_mx(target))
    with pytest.raises(ValueError, match="differ in shape"):
        train.flow_match_loss(to_mx(target), to_mx(pred))

    p, t = (to_mx(a) for a in draws(12))
    wrong = [mx.ones(SHAPE + (2,)),  # more axes than the prediction
             mx.array([1.0, 2.0]),  # two weights for three examples
             mx.ones((SHAPE[1],)), mx.ones((SHAPE[2],))]  # one per row, one per channel: not the leading axis
    for w in wrong:
        with pytest.raises(ValueError, match="does not line up"):
            train.flow_match_loss(p, t, w)
    with pytest.raises(ValueError, match="does not line up"):  # (3,) against one example of three rows would broadcast
        train.flow_match_loss(mx.ones((1, 3, 3)), mx.zeros((1, 3, 3)), mx.array([1.0, 2.0, 3.0]))
    w = [0.5, 2.0, 7.0]
    per_example = float(train.flow_match_loss(p, t, mx.array(w)))
    for same in (mx.array(w).reshape(3, 1, 1), mx.array(w).reshape(3, 1), np.float32(w), w):
        assert float(train.flow_match_loss(p, t, same)) == per_example
    assert float(train.flow_match_loss(p, t, mx.array([2.0]))) == float(train.flow_match_loss(p, t, 2.0))


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
