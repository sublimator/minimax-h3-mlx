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


def test_training_sigmas_refusals():
    with pytest.raises(ValueError, match="merges neighbouring"):
        train.training_sigmas(1_000_000, 12.0)  # float32 can't hold a million distinct levels at this shift
    with pytest.raises(ValueError, match="shift must be positive"):
        train.training_sigmas(1000, 0.0)
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
