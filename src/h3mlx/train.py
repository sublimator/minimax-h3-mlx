"""Pieces of a flow-matching trainer for the MLX transformer: the forward process, the target, the loss, and sampling
the noise level. No training loop here.

MiniMax H3's transformer predicts a data-ward velocity. Its scheduler (diffusers `scheduling_minimax_h3.py`) works in
`t = 1 - sigma`, with `t = 1` clean: `scale_noise` mixes `x_t = t * x_0 + (1 - t) * noise`, and `step` recovers
`x_0 = x_t + (1 - t) * v` (`ops.flow_step`). The velocity that takes `x_t` back to `x_0` is therefore `x_0 - noise`.
That is the opposite sign to the usual flow-match convention, which the scheduler's own notes give as
`x_0 = x_t - sigma * v`: there the target is `noise - x_0`.

The noise level is sampled as diffusers' SD3-style trainers sample it (`training_utils.py`:
`compute_density_for_timestep_sampling`, `compute_loss_weighting_for_sd3`), over a shifted grid built by this port's
own `ops.flow_sigmas`. Random draws come from a torch generator, as the sampling noise does (`pipeline.Noise`).
"""

import math

import mlx.core as mx
import numpy as np

from . import ops

# --- the forward process, the target and the loss --------------------------------------------------------------------


def scale_noise(sample: mx.array, timestep, noise: mx.array) -> mx.array:
    """`MiniMaxH3Scheduler.scale_noise`: `x_t = t * x_0 + (1 - t) * noise`, `t` in [0, 1] and 1 clean.

    A Python float keeps `1 - t` at double precision until it meets the noise (in float32, `1 - 0.999` comes out
    1.3e-5 relative too small). An array is cast to the sample's dtype and lined up with its leading axes (a 0-d `t`,
    or one per example), and `1 - t` is then taken in that dtype, which is how the pipelines noise their conditioning
    rows."""
    if not isinstance(timestep, mx.array):
        t = float(timestep)
        return ops.times_scalar(sample, t) + ops.times_scalar(noise, 1.0 - t)
    t = timestep.astype(sample.dtype)
    t = t.reshape(t.shape + (1,) * (sample.ndim - t.ndim))
    return t * sample + (1.0 - t) * noise


def velocity_target(sample: mx.array, noise: mx.array) -> mx.array:
    """What the transformer is trained to predict for `scale_noise(sample, t, noise)`, at every `t`: `x_0 - noise`.
    From the scheduler's `x_0 = x_t + (1 - t) * v` with `x_t = t * x_0 + (1 - t) * noise`: `(1 - t) * v` must equal
    `(1 - t) * (x_0 - noise)`. Not `noise - x_0`."""
    return sample - noise


def flow_match_loss(pred: mx.array, target: mx.array, weight=None) -> mx.array:
    """`mean(weight * (pred - target) ** 2)` over every element, in float32 whatever the inputs' dtypes. `weight` is a
    number, or an array lined up with the leading axes (one weight per example, from `loss_weighting_for_sd3`).

    Pass the rows the model is trained on and nothing else: the generated video rows, picked by the layout as `denoise`
    picks them (`pred[:, layout.num_condition_video_rows:]`). Condition rows are inputs. Rows can't be told apart by
    their timestep index: at `t >= 0.999` a generated row and a condition row share one."""
    err = pred.astype(mx.float32) - target.astype(mx.float32)
    sq = err * err
    if weight is not None:
        w = weight.astype(mx.float32) if isinstance(weight, mx.array) else mx.array(weight, dtype=mx.float32)
        sq = w.reshape(w.shape + (1,) * (sq.ndim - w.ndim)) * sq
    return mx.mean(sq)


# --- sampling the noise level ------------------------------------------------------------------------------------------


def density_for_timestep_sampling(weighting_scheme: str, batch_size: int, logit_mean: float | None = None,
                                  logit_std: float | None = None, mode_scale: float | None = None,
                                  generator=None) -> np.ndarray:
    """diffusers' `compute_density_for_timestep_sampling`: `batch_size` values `u` in [0, 1], float32.

    * `"logit_normal"`: the sigmoid of a normal draw (`logit_mean`, `logit_std`).
    * `"mode"`: `1 - u - mode_scale * (cos(pi * u / 2) ** 2 - 1 + u)` of a uniform draw.
    * anything else: a uniform draw.

    The draw is torch's, from `generator` (a `torch.Generator`; None is torch's global one), so it is the same stream
    as theirs. The arithmetic after it is numpy float32 in their order: equal to theirs bit for bit except for the
    sigmoid's `exp` and the `cos`, which numpy and torch round differently in the last place or two."""
    import torch

    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator).numpy()
        return (np.float32(1) / (np.float32(1) + np.exp(-u))).astype(np.float32)
    if weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), generator=generator).numpy()
        c = np.cos(np.float32(math.pi) * u / np.float32(2))
        return (np.float32(1) - u - np.float32(mode_scale) * (c * c - np.float32(1) + u)).astype(np.float32)
    return torch.rand(size=(batch_size,), generator=generator).numpy()


def loss_weighting_for_sd3(weighting_scheme: str, sigmas) -> np.ndarray:
    """diffusers' `compute_loss_weighting_for_sd3`, on float32 (or float64) noise levels.

    * `"sigma_sqrt"`: `sigma ** -2`, which torch computes as `1 / (sigma * sigma)`; float32 (infinite at sigma 0).
    * `"cosmap"`: `2 / (pi * (1 - 2 * sigma + 2 * sigma ** 2))`, in the input's dtype.
    * anything else: ones.

    `sigma` is the noise level in both conventions (1 is pure noise), so nothing changes for this model's `1 - t`."""
    s = np.asarray(sigmas)
    if s.dtype not in (np.float32, np.float64):
        s = s.astype(np.float32)
    one, two = s.dtype.type(1), s.dtype.type(2)
    if weighting_scheme == "sigma_sqrt":
        return (one / (s * s)).astype(np.float32)
    if weighting_scheme == "cosmap":
        bot = one - two * s + two * (s * s)
        return (two / (s.dtype.type(math.pi) * bot)).astype(s.dtype)
    return np.ones_like(s)


def training_sigmas(num_points: int, shift: float) -> np.ndarray:
    """`num_points` noise levels to train at, from 1.0 (pure noise) down, float32: the scheduler's own shifted grid
    for `num_points + 1` points (`ops.flow_sigmas`: `linspace(1, 0, n)` through `s * sigma / (1 + (s - 1) * sigma)`)
    without its last point, sigma 0, where the sample is clean and there is nothing to predict. Both numbers are the
    caller's: this module has no default shift or grid size."""
    grid = ops.flow_sigmas(num_points + 1, shift)[:-1]
    if len(grid) != num_points:
        raise ValueError(f"shift {shift} merges neighbouring float32 levels of a {num_points}-point grid "
                         f"({len(grid)} distinct): ask for fewer points")
    return grid


def sample_sigmas(u, grid) -> tuple[np.ndarray, np.ndarray]:
    """One noise level per `u`: `grid[floor(u * len(grid))]`, the multiplication in float32 (an SD3-style trainer's
    `(u * num_train_timesteps).long()` lookup). Returns (sigmas, indices). `u = 1.0` would index one past the end,
    which torch refuses; it is given the last level here (the logit-normal sigmoid rounds to 1.0 for a large draw, and
    `"mode"` maps a draw of 0 to 1.0). The model's timestep for a level is `1 - sigma` in float32."""
    grid = np.asarray(grid)
    u = np.asarray(u, dtype=np.float32)
    if u.size and (u.min() < 0 or u.max() > 1):
        raise ValueError(f"u must lie in [0, 1], got [{u.min()}, {u.max()}]")
    indices = np.minimum((u * np.float32(len(grid))).astype(np.int64), len(grid) - 1)
    return grid[indices], indices
