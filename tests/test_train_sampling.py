"""Sampling the noise level and weighting the loss, against diffusers' `training_utils` and its schedulers.

What is bit for bit and what is not:

* the uniform draw, both loss weightings and the sigma grid are the oracle's values exactly;
* the logit-normal and mode densities start from the same torch draw, but numpy's `exp` and `cos` are not torch's:
  the results agree to a few float32 steps (2 and 6 at worst over the cases below), which once in tens of thousands
  of draws moves a sample to the neighbouring grid level.

The lookup of a sigma from `u` has no function in diffusers' library code; it is written out in torch here.
"""

import numpy as np
import pytest
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from h3mlx import train

SEEDS = range(40)
SIZES = (1, 5, 15, 16, 33, 1000)  # torch's normal fills 16 or more values by another routine than fewer
DENSITIES = [("uniform", {}), ("logit_normal", dict(logit_mean=0.0, logit_std=1.0)),
             ("logit_normal", dict(logit_mean=0.3, logit_std=1.2)), ("logit_normal", dict(logit_mean=-2.0, logit_std=3.0)),
             ("mode", dict(mode_scale=1.29)), ("mode", dict(mode_scale=0.5))]


def steps_apart(a, b):
    """The largest distance between two float32 arrays, counted in representable values."""
    return int(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max())


def both(scheme, n, seed, **kw):
    want = compute_density_for_timestep_sampling(scheme, n, generator=torch.Generator().manual_seed(seed), **kw)
    got = train.density_for_timestep_sampling(scheme, n, generator=torch.Generator().manual_seed(seed), **kw)
    return got, want


def h3_sigmas(n, shift):
    """diffusers' MiniMaxH3Scheduler grid for n + 1 points, as a torch tensor with its terminal 0."""
    sch = MiniMaxH3Scheduler(shift=shift)
    sch.set_timesteps(n + 1)
    return sch


# --- the density -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme,kw", DENSITIES)
def test_density_matches_diffusers(scheme, kw):
    worst = 0
    for seed in SEEDS:
        for n in SIZES:
            got, want = both(scheme, n, seed, **kw)
            assert got.dtype == np.float32 and got.shape == (n,) and got.min() >= 0 and got.max() <= 1
            worst = max(worst, steps_apart(got, want.numpy()))
    print(f"{scheme} {kw}: at most {worst} float32 steps from torch")
    # uniform: the same draw, nothing else. The others: the same draw, then numpy's exp or cos in place of torch's
    assert worst <= {"uniform": 0, "logit_normal": 4, "mode": 12}[scheme]


def test_density_guards():
    """Wrong readings of the arguments are far outside the rounding allowance above, and the generator is used."""
    got, want = both("logit_normal", 1000, 0, logit_mean=0.3, logit_std=1.2)
    swapped = train.density_for_timestep_sampling("logit_normal", 1000, logit_mean=1.2, logit_std=0.3,
                                                  generator=torch.Generator().manual_seed(0))
    assert np.abs(swapped - want.numpy()).max() > 0.1
    mode, _ = both("mode", 1000, 0, mode_scale=1.29)
    uniform, _ = both("uniform", 1000, 0)
    assert np.abs(mode - uniform).max() > 0.1, "mode reshapes the uniform draw it starts from"
    assert not np.array_equal(uniform, both("uniform", 1000, 1)[0])
    g, tg = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    for scheme, kw in DENSITIES[:2]:  # two calls on one generator: the second continues the stream, as theirs
        got = train.density_for_timestep_sampling(scheme, 20, generator=g, **kw)
        assert steps_apart(got, compute_density_for_timestep_sampling(scheme, 20, generator=tg, **kw).numpy()) <= 4
    torch.manual_seed(0)  # no generator: torch's global one, as theirs
    assert np.array_equal(train.density_for_timestep_sampling("uniform", 1000), uniform)


def test_density_parameters_are_required_and_finite():
    """Leave out a scheme's parameter and theirs raises a TypeError (None times a tensor, or torch.normal's argument
    check); ours raises one too, where it used to return NaN for `mode`. Give a NaN or an infinite parameter and
    theirs returns values that are NaN, infinite, or pinned at 0 and 1, without a word; ours raises. A negative or NaN
    std is torch.normal's own RuntimeError on both sides, and a zero std gives 0.5 on both."""
    nan, inf = float("nan"), float("inf")
    for scheme, kw in (("mode", {}), ("logit_normal", {}), ("logit_normal", dict(logit_mean=0.0)), ("logit_normal", dict(logit_std=1.0))):
        with pytest.raises(TypeError):
            compute_density_for_timestep_sampling(scheme, 4, **kw)
        with pytest.raises(TypeError, match="is required"):
            train.density_for_timestep_sampling(scheme, 4, **kw)
    for scheme, kw in (("mode", dict(mode_scale=nan)), ("mode", dict(mode_scale=inf)),
                       ("logit_normal", dict(logit_mean=nan, logit_std=1.0)), ("logit_normal", dict(logit_mean=inf, logit_std=1.0)),
                       ("logit_normal", dict(logit_mean=0.0, logit_std=inf))):
        theirs = compute_density_for_timestep_sampling(scheme, 8, generator=torch.Generator().manual_seed(0), **kw)
        assert not ((theirs > 0) & (theirs < 1)).any(), "theirs: nothing usable, and no error"
        with pytest.raises(ValueError, match="must be finite"):
            train.density_for_timestep_sampling(scheme, 8, generator=torch.Generator().manual_seed(0), **kw)
    for call in (compute_density_for_timestep_sampling, train.density_for_timestep_sampling):
        for std in (-1.0, -inf, nan):
            with pytest.raises(RuntimeError, match="std >= 0"):
                call("logit_normal", 4, logit_mean=0.0, logit_std=std)
        assert np.array_equal(np.asarray(call("logit_normal", 4, logit_mean=0.0, logit_std=0.0)), np.full(4, 0.5, np.float32))


# --- the loss weighting ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["sigma_sqrt", "cosmap", "none"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loss_weighting_is_diffusers(scheme, dtype):
    sigmas = (torch.rand(2000, generator=torch.Generator().manual_seed(3)) * 0.999 + 0.001).to(dtype)
    want = compute_loss_weighting_for_sd3(scheme, sigmas)
    got = train.loss_weighting_for_sd3(scheme, sigmas.numpy())
    assert str(got.dtype) == str(want.dtype).split(".")[-1]  # sigma_sqrt is float32 even from float64, as theirs
    assert np.array_equal(got, want.numpy())


def test_loss_weighting_values():
    half = np.float32([0.5, 1.0, 0.25])
    assert np.array_equal(train.loss_weighting_for_sd3("sigma_sqrt", half), np.float32([4.0, 1.0, 16.0]))
    assert np.allclose(train.loss_weighting_for_sd3("cosmap", half), [4 / np.pi, 2 / np.pi, 2 / (np.pi * 0.625)], rtol=1e-6)
    assert np.array_equal(train.loss_weighting_for_sd3("none", half), np.ones(3, np.float32))


def test_loss_weighting_takes_float32_or_float64_only():
    """An `mx.array` of float32 levels is taken like numpy's, bit for bit theirs. Half precision is refused: theirs
    computes a bf16 tensor's weighting in bf16 (shown: not the float32 values), which a numpy result can't hold, and an
    upcast nobody asked for would be neither. Integers are refused too."""
    import mlx.core as mx

    levels = np.float32([0.5, 0.3, 0.9])
    for scheme in ("sigma_sqrt", "cosmap", "none"):
        want = compute_loss_weighting_for_sd3(scheme, torch.from_numpy(levels)).numpy()
        got = train.loss_weighting_for_sd3(scheme, mx.array(levels))
        assert got.dtype == np.float32 and np.array_equal(got, want)
        in_bf16 = compute_loss_weighting_for_sd3(scheme, torch.from_numpy(levels).to(torch.bfloat16))
        if scheme != "none":
            assert not np.array_equal(in_bf16.float().numpy(), want), "theirs on bf16 is bf16 arithmetic"
        for refused in (mx.array(levels).astype(mx.bfloat16), mx.array(levels).astype(mx.float16), levels.astype(np.float16),
                        np.array([1, 2]), mx.array([1, 2])):
            with pytest.raises(ValueError, match="float32 or float64"):
                train.loss_weighting_for_sd3(scheme, refused)
    assert train.loss_weighting_for_sd3("cosmap", [0.5, 0.25]).dtype == np.float64, "Python floats are float64, as numpy's"


# --- the grid ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,shift", [(1000, 12.0), (1000, 3.0), (1000, 1.0), (64, 1.0), (8, 5.5), (1, 12.0)])
def test_training_sigmas_are_the_schedulers(n, shift):
    """The grid is the port's own `flow_sigmas`, so it is MiniMaxH3Scheduler's for n + 1 points, bit for bit, without
    the terminal 0; `1 - sigma` is the scheduler's timesteps. The generic flow-match scheduler's training grid (the
    same levels, k / n shifted, by other arithmetic) is within a few float32 steps, not equal."""
    grid = train.training_sigmas(n, shift)
    sch = h3_sigmas(n, shift)
    assert grid.dtype == np.float32 and len(grid) == n
    assert np.array_equal(grid, sch.sigmas[:-1].numpy()) and float(sch.sigmas[-1]) == 0.0
    assert np.array_equal(np.float32(1) - grid, sch.timesteps.numpy())
    assert grid[0] == 1.0 and grid[-1] > 0 and (np.diff(grid) < 0).all(), "from pure noise down, never clean"
    generic = FlowMatchEulerDiscreteScheduler(num_train_timesteps=n, shift=shift).sigmas.numpy()
    assert len(generic) == n and steps_apart(grid, generic) <= 4
    if n == 1000 and shift == 12.0:
        assert not np.array_equal(grid, generic), "close, and not the same arithmetic"


@pytest.mark.parametrize("shift", [float("nan"), float("inf"), 1e39, 1e-50])
def test_a_shift_that_is_not_a_positive_finite_float32_is_refused(shift):
    """NaN and infinity pass a `shift <= 0` test, 1e39 is infinite as a float32 and 1e-50 is zero. diffusers' scheduler
    takes each and returns a grid with NaN in it, and so did the port's grid functions, on the sampling path too.
    They raise now; valid shifts are untouched (the grid tests above and the scheduler's own are bit for bit)."""
    from h3mlx import ops
    from h3mlx.denoise import Schedules, shift_grid

    theirs = h3_sigmas(4, shift).sigmas
    assert torch.isnan(theirs).any(), "theirs: a grid with NaN, and no error"
    for build in (lambda: ops.flow_sigmas(5, shift), lambda: train.training_sigmas(4, shift),
                  lambda: shift_grid([1.0, 0.5, 0.0], shift), lambda: Schedules.make(5, video_shift=shift),
                  lambda: Schedules.make(5, audio_shift=shift)):
        with pytest.raises(ValueError, match="must be positive and finite"):
            build()
    assert not ops.usable_shift(shift) and ops.usable_shift(12.0) and ops.usable_shift(1e-30)


def test_training_sigmas_refusals():
    with pytest.raises(ValueError, match="merges neighbouring"):
        train.training_sigmas(1_000_000, 12.0)  # float32 can't hold a million distinct levels at this shift
    with pytest.raises(ValueError, match="shift must be positive"):
        train.training_sigmas(1000, 0.0)
    with pytest.raises(ValueError, match="shift must be positive"):
        train.training_sigmas(1000, -3.0)
    with pytest.raises(ValueError):
        train.training_sigmas(0, 12.0)
    assert not np.array_equal(train.training_sigmas(1000, 12.0), train.training_sigmas(1000, 3.0))


# --- from a seed to sigmas ---------------------------------------------------------------------------------------------


def torch_lookup(u, sigmas):
    """An SD3-style trainer's lookup, written out: `indices = (u * num_train_timesteps).long()`, then the grid."""
    indices = (u * len(sigmas)).long()
    return sigmas[indices], indices


@pytest.mark.parametrize("n,shift", [(1000, 12.0), (64, 3.0), (7, 1.0)])
def test_same_seed_same_sigmas_uniform(n, shift):
    """Uniform density: the same torch seed gives the same sigmas as the torch chain, bit for bit."""
    grid = train.training_sigmas(n, shift)
    sigmas = h3_sigmas(n, shift).sigmas[:-1]
    for seed in SEEDS:
        got_u, want_u = both("uniform", 257, seed)
        want, want_i = torch_lookup(want_u, sigmas)
        got, got_i = train.sample_sigmas(got_u, grid)
        assert got.dtype == np.float32 and np.array_equal(got_i, want_i.numpy()) and np.array_equal(got, want.numpy())


@pytest.mark.parametrize("scheme,kw", DENSITIES[1:])
def test_same_seed_nearly_same_sigmas_otherwise(scheme, kw):
    """Logit-normal and mode: a `u` a float32 step or two from torch's lands on the same grid level unless it sits on
    a boundary. Over 40,000 draws on a 1,000-level grid at most a handful move, and only to the next level."""
    n = 1000
    grid = train.training_sigmas(n, 12.0)
    sigmas = h3_sigmas(n, 12.0).sigmas[:-1]
    moved = total = 0
    for seed in SEEDS:
        got_u, want_u = both(scheme, 1000, seed, **kw)
        keep = want_u < 1.0  # torch's lookup can't take u = 1.0
        _, want_i = torch_lookup(want_u[keep], sigmas)
        _, got_i = train.sample_sigmas(got_u[keep.numpy()], grid)
        diff = np.abs(got_i - want_i.numpy())
        assert diff.max() <= 1
        moved, total = moved + int(diff.sum()), total + diff.size
    print(f"{scheme} {kw}: {moved} of {total} draws on a neighbouring level")
    assert moved <= total // 2000


def test_sample_sigmas_refuses_nan_and_a_bad_grid():
    """A NaN is neither below 0 nor above 1, so a range test written that way lets it through; cast to an index it is
    0 on this hardware, sigma 1.0: pure noise for every such draw, and no error. The torch lookup does the same. Ours
    raises. An empty grid was an IndexError and a 2-D one returned whole rows; both are refused."""
    grid = train.training_sigmas(1000, 12.0)
    nan = float("nan")
    theirs, theirs_i = torch_lookup(torch.tensor([nan, 0.5]), torch.from_numpy(grid))
    assert theirs_i.tolist() == [0, 500] and float(theirs[0]) == 1.0  # arm64's float-to-integer cast of a NaN
    for u in (np.float32([nan, 0.5]), np.float32([0.5, nan]), [nan], np.float32(nan)):
        with pytest.raises(ValueError, match="a NaN does not"):
            train.sample_sigmas(u, grid)
    for bad in ([], np.zeros((0,), np.float32), grid.reshape(10, 100), np.float32(0.5)):
        with pytest.raises(ValueError, match="one non-empty row"):
            train.sample_sigmas([0.5], bad)
    for bad in (np.float32([1.0, nan, 0.2]), np.float32([float("inf"), 0.5])):  # a level that is not a number
        with pytest.raises(ValueError, match="not finite"):
            train.sample_sigmas([0.5], bad)


def test_sample_sigmas_lookup():
    grid = train.training_sigmas(10, 12.0)
    u = np.float32([0.0, 0.0999, 0.1, 0.55, 0.9999, 1.0])
    sigmas, indices = train.sample_sigmas(u, grid)
    assert indices.tolist() == [0, 0, 1, 5, 9, 9], "floor, and u = 1.0 kept on the last level"
    assert np.array_equal(sigmas, grid[[0, 0, 1, 5, 9, 9]]) and sigmas[0] == 1.0, "u = 0 is pure noise"
    with pytest.raises(IndexError):  # the torch lookup refuses u = 1.0: ours differs on that one value
        torch_lookup(torch.tensor([1.0]), torch.from_numpy(grid))
    for bad in ([-0.1], [1.5]):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            train.sample_sigmas(bad, grid)
    # the product is float32, as torch's: float32(0.005) is just under 0.005, and times 1000 it rounds up to 5.0
    thousand = train.training_sigmas(1000, 12.0)
    edge = np.float32([0.005, 0.009, 0.01])
    assert (edge.astype(np.float64) * 1000).astype(np.int64).tolist() == [4, 8, 9], "what a float64 product would pick"
    want, want_i = torch_lookup(torch.from_numpy(edge), torch.from_numpy(thousand))
    got, got_i = train.sample_sigmas(edge, thousand)
    assert got_i.tolist() == want_i.tolist() == [5, 9, 10] and np.array_equal(got, want.numpy())
    assert train.sample_sigmas([], grid)[0].shape == (0,)
