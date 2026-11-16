"""Pieces of a flow-matching trainer for the MLX transformer: the forward process, the target and the loss. No
training loop here.

MiniMax H3's transformer predicts a data-ward velocity. Its scheduler (diffusers `scheduling_minimax_h3.py`) works in
`t = 1 - sigma`, with `t = 1` clean: `scale_noise` mixes `x_t = t * x_0 + (1 - t) * noise`, and `step` recovers
`x_0 = x_t + (1 - t) * v` (`ops.flow_step`). The velocity that takes `x_t` back to `x_0` is therefore `x_0 - noise`.
That is the opposite sign to the usual flow-match convention, which the scheduler's own notes give as
`x_0 = x_t - sigma * v`: there the target is `noise - x_0`.
"""

import mlx.core as mx

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
    number, or an array lined up with the leading axes (one weight per example).

    Pass the rows the model is trained on and nothing else: the generated video rows, picked by the layout as `denoise`
    picks them (`pred[:, layout.num_condition_video_rows:]`). Condition rows are inputs. Rows can't be told apart by
    their timestep index: at `t >= 0.999` a generated row and a condition row share one."""
    err = pred.astype(mx.float32) - target.astype(mx.float32)
    sq = err * err
    if weight is not None:
        w = weight.astype(mx.float32) if isinstance(weight, mx.array) else mx.array(weight, dtype=mx.float32)
        sq = w.reshape(w.shape + (1,) * (sq.ndim - w.ndim)) * sq
    return mx.mean(sq)
