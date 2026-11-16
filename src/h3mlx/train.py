"""Pieces of a flow-matching trainer for the MLX transformer: the forward process, the target, the loss, one training
example and its loss and gradient, the optimizer step, and sampling the noise level. No training loop here.

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
import numbers
from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from . import lora, ops
from .denoise import row_timesteps
from .packing import Layout, Ref, build_packed_sequence, build_ref2va_packed_sequence

# --- the forward process, the target and the loss --------------------------------------------------------------------


def scale_noise(sample: mx.array, timestep, noise: mx.array) -> mx.array:
    """`MiniMaxH3Scheduler.scale_noise`: `x_t = t * x_0 + (1 - t) * noise`, `t` in [0, 1] and 1 clean.

    The timestep's kind decides the arithmetic, as in the scheduler:

    * A number (a Python float or int, or a numpy scalar) takes `1 - t` in its own precision before it meets the
      noise: double for a Python float (in float32, `1 - 0.999` comes out 1.3e-5 relative too small), float32 for an
      `np.float32`. Each product is then tensor-times-number.
    * An `mx.array` is cast to the sample's dtype and lined up with its leading axes (a 0-d `t`, or one per example),
      and `1 - t` is taken in that dtype, which is how the pipelines noise their conditioning rows.
    * A numpy array is a TypeError, as it is for the scheduler: wrap it, `mx.array(timesteps)`, to have it treated as
      a tensor (the timesteps `1 - sigma` from `sample_sigmas` are numpy).

    One difference, on purpose: a timestep with more axes than the sample is refused. The scheduler broadcasts it into
    a result larger than the sample."""
    if isinstance(timestep, np.ndarray):
        raise TypeError("a numpy array of timesteps is not accepted (nor by the scheduler): pass a number, or "
                        "mx.array(timesteps) to have them treated as a tensor")
    if not isinstance(timestep, mx.array):
        return ops.times_scalar(sample, float(timestep)) + ops.times_scalar(noise, float(1.0 - timestep))
    if timestep.ndim > sample.ndim:
        raise ValueError(f"a timestep of shape {timestep.shape} has more axes than the sample {sample.shape}")
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
    their timestep index: at `t >= 0.999` a generated row and a condition row share one.

    `pred` and `target` must have the same shape, and each axis of `weight` must be 1 or the size of the leading axis
    it lines up with. torch's `mse_loss` broadcasts a target of another shape with a warning; here it is an error,
    because a prediction that still holds its condition rows would be averaged against a one-row target without a
    word."""
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape: pass the same "
                         "rows of both")
    err = pred.astype(mx.float32) - target.astype(mx.float32)
    sq = err * err
    if weight is not None:
        w = weight.astype(mx.float32) if isinstance(weight, mx.array) else mx.array(weight, dtype=mx.float32)
        if w.ndim > sq.ndim or any(a not in (1, b) for a, b in zip(w.shape, sq.shape)):
            raise ValueError(f"a weight of shape {tuple(w.shape)} does not line up with the leading axes of "
                             f"{tuple(sq.shape)}")
        sq = w.reshape(w.shape + (1,) * (sq.ndim - w.ndim)) * sq
    return mx.mean(sq)


# --- one training example: a picture, with or without reference pictures -------------------------------------------

# The steps below follow a named recipe: the MiniMax H3 / P-H3-Image model files of ostris/ai-toolkit at commit
# d6a18ad7 (MIT), a third-party trainer. It is not the model maker's recipe, which is not published. Where diffusers
# has the same step for sampling (the layout, the rows, the row timesteps, the forward process) the port's own tested
# function is used; what comes from the recipe alone is said so at each place.


@dataclass
class Example:
    """One picture as the transformer sees it in training (`build_example`)."""

    layout: Layout
    video_rows: mx.array  # (condition rows + generated rows, patch dim): the references' rows, then the noised picture's
    audio_rows: mx.array  # (audio rows, audio channels): silence noised at the audio level
    target: mx.array  # (generated rows, patch dim): `velocity_target` of the picture's rows, x_0 - noise
    clean_rows: mx.array  # (generated rows, patch dim): the picture's rows before noising
    timestep: np.ndarray  # the distinct row timesteps, sorted, float32
    timestep_indices: np.ndarray  # each row's index into them
    sigma: np.float32  # the picture's noise level; its rows' timestep is 1 - sigma

    @property
    def num_condition_rows(self) -> int:
        return self.layout.num_condition_video_rows

    def generated(self, video_pred: mx.array) -> mx.array:
        """The transformer's video output (batch, video rows, patch dim) without the condition rows: what the loss is
        taken over, against `target`. The rows are told apart by their place in the layout, as `denoise` does it."""
        return video_pred[:, self.num_condition_rows:]


def loss_mask(layout: Layout) -> np.ndarray:
    """True at the packed-sequence rows a picture's loss covers: the generated video rows. Text rows, audio rows and
    condition rows are False. Read off the layout's indices, never off timesteps: at a timestep of the condition level
    or above, a generated row and a condition row carry the same value."""
    mask = np.zeros(len(layout.token_tags), dtype=bool)
    mask[layout.video_indices[layout.num_condition_video_rows:]] = True
    return mask


def encode_references(components, pictures, *, float16_round: bool) -> list:
    """The latents of an edit example's reference pictures: `image.encode_pictures` (a seed-42 posterior sample each),
    with or without its float16 round trip, which the caller must choose. The pipeline rounds, so `True` is what the
    model is given at sampling by diffusers and by this port. The recipe turns the rounding off for P-H3-Image, so
    `False` is what that trainer shows the model."""
    from .image import encode_pictures

    return encode_pictures(components, pictures, float16_round=float16_round)


def build_example(cfg: dict, text_token_tags, latents: mx.array, sigma, noise, *, condition_t: float,
                  condition_latents=(), audio_sigma=None) -> Example:
    """One training example for a single picture with zero or more reference pictures.

    `latents` is the picture, normalised, (1, 1, h, w, channels), and `condition_latents` the references in packed
    order, each (1, 1, h_i, w_i, channels) at its own size, as the VAE encode returns them. `text_token_tags` has one
    tag per prompt row: 1 for text, 0 for a vision token (tagged as video), which are the two the picture pipelines'
    encoders give. `sigma` in (0, 1] is the picture's noise level; it must also be large enough that the timestep
    `1 - sigma` is below 1 in float32, which is any sigma above 2**-25 (about 3e-8). Nothing is clamped. `noise(shape)`
    returns float32 draws (`pipeline.Noise`). `condition_t` is the timestep the references are noised to and held at;
    it has no default here (the pipeline and the recipe both use 0.999).

    * Layout: text-to-image's without references, the edit's with them (`image.generate`'s two, diffusers').
    * Draws, in the pipeline's order: one per reference, the picture's, the audio rows'.
    * References: `scale_noise` to `condition_t` given as an array, so `1 - t` is taken in float32 as the pipeline
      takes it. (The recipe writes the same mix with a Python float, whose `1 - t` is double: 1.3e-5 relative apart.)
    * The picture: `scale_noise` to `t = 1 - sigma` in float32, also as an array, so the factor on the noise is the
      `1 - t` that the sampler's step multiplies the velocity by. Its target is `velocity_target`.
    * Audio rows (recipe): a picture has no soundtrack, so they are silence noised at the audio level, and no loss is
      taken on them. `audio_sigma=None` means the picture's own level, which is the recipe's rule for P-H3-Image. The
      recipe multiplies the noise by sigma itself, this by `1 - t`: sigma exactly down to 0.5, and below that sigma
      after the float32 rounding of `t = 1 - sigma`, within 6e-8 of it.
    * Row timesteps: `denoise.row_timesteps`; the references sit at `max(t, condition_t)`, as in the pipeline and the
      recipe."""
    from .image import AUDIO_CHANNELS, NUM_AUDIO_LATENTS

    sigma32 = np.float32(sigma)
    audio32 = sigma32 if audio_sigma is None else np.float32(audio_sigma)
    for what, value in (("sigma", sigma32), ("audio_sigma", audio32)):
        if not (np.isfinite(value) and 0 < value <= 1):
            raise ValueError(f"{what} must lie in (0, 1], got {value}")
        if np.float32(1) - value >= 1:
            raise ValueError(f"{what} = {value} is too small: the timestep 1 - {what} rounds to 1.0 in float32, so the "
                             "rows would not be noised at all. It must be above 2**-25, about 3e-8")
    if not (math.isfinite(condition_t) and 0 <= condition_t <= 1):
        raise ValueError(f"condition_t must lie in [0, 1], got {condition_t}")
    patch = tuple(cfg["patch_size"])
    channels = cfg["in_channels"]
    condition_latents = list(condition_latents)
    for what, x in [("latents", latents)] + [(f"condition_latents[{i}]", x) for i, x in enumerate(condition_latents)]:
        if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 1 or x.shape[4] != channels or 0 in x.shape[2:4]:
            raise ValueError(f"{what} must be (1, 1, height, width, {channels}) with a height and a width, "
                             f"got {tuple(x.shape)}")
    lh, lw = latents.shape[2], latents.shape[3]

    tags = np.asarray(text_token_tags)
    if tags.ndim != 1 or tags.size == 0 or not np.issubdtype(tags.dtype, np.integer) or not np.isin(tags, (0, 1)).all():
        raise ValueError("text_token_tags must be one integer tag per prompt row, each 1 (text) or 0 (a vision token), "
                         f"and at least one row; got {tags.tolist() if tags.size <= 16 else tags.shape}")
    if condition_latents:
        layout = build_ref2va_packed_sequence(tags, [Ref("image")] * len(condition_latents),
                                              [tuple(x.shape[1:4]) for x in condition_latents], [], 1, lh, lw,
                                              NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)
    else:
        layout = build_packed_sequence(tags, 1, lh, lw, NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)

    t, t_audio = np.float32(1) - sigma32, np.float32(1) - audio32
    rows = []
    for x in condition_latents:
        b, f, h, w, ch = x.shape
        drawn = mx.array(noise((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
        noised = scale_noise(x.astype(mx.float32), mx.array(np.float32(condition_t)), drawn)
        rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
    clean = latents.astype(mx.float32).transpose(0, 4, 1, 2, 3)
    drawn = mx.array(noise((1, channels, 1, lh, lw)))
    rows.append(ops.patchify(scale_noise(clean, mx.array(t), drawn), patch))
    audio_noise = mx.array(noise((NUM_AUDIO_LATENTS * AUDIO_CHANNELS, cfg["audio_in_channels"])))
    audio_rows = scale_noise(mx.zeros(audio_noise.shape, dtype=mx.float32), mx.array(t_audio), audio_noise)

    timestep, timestep_indices = row_timesteps(layout, float(t), float(t_audio), max(float(t), float(condition_t)), 1.0)
    return Example(layout=layout, video_rows=mx.concatenate(rows), audio_rows=audio_rows,
                   target=ops.patchify(velocity_target(clean, drawn), patch), clean_rows=ops.patchify(clean, patch),
                   timestep=timestep, timestep_indices=timestep_indices, sigma=sigma32)


# --- the loss of an example, and its gradient ----------------------------------------------------------------------


def check_inputs(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array) -> None:
    """What every loss and gradient of an example checks first.

    * `trainable` is what the forward will train (`lora.check_trainable`: not empty, runtime names, active in
      `frozen`, both matrices, nothing shared with `frozen`, the adapter whole).
    * The weights and the matrices are this config's (`lora.check_config`): the weights have the config's blocks and
      no more, and every trained matrix sits on a Linear the config has, in that Linear's shape. A config of fewer
      blocks than the weights would run a shorter model, and the matrices of the blocks left out would get gradients
      of exactly zero.
    * `text` is exactly (1, the layout's text rows, text dim). A text of one row would be broadcast over every text
      position by the forward's scatter.

    Each of these would otherwise go wrong without an error."""
    lora.check_trainable(trainable, frozen)
    lora.check_config(trainable, frozen, cfg)
    want = (1, len(example.layout.text_indices), cfg["text_dim"])
    if not isinstance(text, mx.array) or tuple(text.shape) != want:
        raise ValueError(f"text must be an array of shape {want} (1, the layout's text rows, text dim), got "
                         f"{tuple(text.shape) if hasattr(text, 'shape') else type(text).__name__}")


class ExampleForward:
    """The transformer's forward on one example, for differentiating, in three stages: `start` (the projections, the
    token refiner, the scatter and the time embedding), `block` (one transformer block) and `finish` (the final norm,
    the heads and the loss). Each stage takes its weights as a dict, so a caller chooses which of them are arguments
    of a transform.

    These are `ops.dit_forward`'s operations in its order, and the loss is the same bit for bit. Two things differ,
    and neither changes a value. Nothing is evaluated between the stages: the sampling forward evaluates after every
    block, and an array evaluated while it is being traced keeps its graph. And the modulation tables' rows are
    gathered with `ops.take_rows_for_training`, whose gradient over a half-precision table is the sum torch's is."""

    def __init__(self, cfg: dict, example: Example, text: mx.array, weight=None):
        lay = example.layout
        self.cfg, self.example, self.text, self.weight = cfg, example, text, weight
        self.video, self.audio = example.video_rows[None], example.audio_rows[None]
        self.timestep, self.t_rows = mx.array(example.timestep), mx.array(example.timestep_indices)
        self.video_indices, self.audio_indices = mx.array(lay.video_indices), mx.array(lay.audio_indices)
        self.text_indices = mx.array(lay.text_indices)
        self.seq = len(lay.token_tags)
        self.adaln_indices = self.t_rows * ops.MODALITY_NUM + mx.array(lay.token_tags)
        self.rotary = ops.rope_cos_sin(mx.array(lay.position_ids.astype(np.float32)),
                                       ops.rope_inv_freq(cfg["rope_freq_dim"], cfg["rope_theta"]))

    def start(self, p: dict) -> tuple[mx.array, mx.array]:
        """(the packed rows before the first block, the time embedding) from the whole dict of weights."""
        x = ops.dit_inputs(p, self.cfg, self.video, self.audio, self.text, self.seq, self.video_indices,
                           self.audio_indices, self.text_indices)
        return x, ops.two_time_embedding(self.timestep, None, p, self.cfg)

    def block(self, x: mx.array, temb: mx.array, block_p: dict) -> mx.array:
        """One block on its own weights (the keys under `transformer_blocks.<i>.`, without that prefix)."""
        cfg = self.cfg
        return ops.dit_block(x, temb, self.adaln_indices, self.rotary, block_p, cfg["num_attention_heads"],
                             cfg["hidden_size"], cfg["norm_eps"], cfg["qk_norm_eps"], take=ops.take_rows_for_training)

    def finish(self, x: mx.array, temb: mx.array, p: dict) -> mx.array:
        """The loss from the last block's output, from the whole dict of weights."""
        shift, scale = ops.norm_out_tables(temb, p)
        video_pred, _ = ops.dit_outputs(x, p, self.cfg, shift, scale, self.t_rows, self.video_indices,
                                        self.audio_indices, take=ops.take_rows_for_training)
        return flow_match_loss(self.example.generated(video_pred), self.example.target[None], self.weight)


def example_loss(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None) -> mx.array:
    """One example's loss: the transformer's forward on the example's rows with the weights `{**frozen, **trainable}`
    (`ExampleForward`), then `flow_match_loss` of the generated video rows against the example's target. Nothing is
    taken from the audio output or from the condition rows.

    The weights come in as two arguments, the trainable matrices first (`lora.split`), so the function can be
    differentiated with respect to them and to nothing else; `check_inputs` refuses a `trainable` the forward would
    not use and a `text` that is not (1, text rows, text dim), the prompt's embeddings. Products with float32 LoRA
    matrices are float32 whatever the base's dtype (`ops.plinear`), and the loss is float32."""
    check_inputs(trainable, frozen, cfg, example, text)
    p = {**frozen, **trainable}
    forward = ExampleForward(cfg, example, text, weight)
    x, temb = forward.start(p)
    for i in range(cfg["num_layers"]):
        x = forward.block(x, temb, ops._sub(p, f"transformer_blocks.{i}."))
    return forward.finish(x, temb, p)


def loss_and_gradients(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None):
    """(loss, gradients) of `example_loss`. The gradients are a dict with `trainable`'s keys and no others: the base
    weights and any other adapter in `frozen` are constants of the differentiation and get none."""
    check_inputs(trainable, frozen, cfg, example, text)
    return mx.value_and_grad(example_loss)(trainable, frozen, cfg, example, text, weight)


# --- the optimizer step ------------------------------------------------------------------------------------------------


def _real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def adamw(*, learning_rate: float, betas, eps: float, weight_decay: float) -> optim.AdamW:
    """MLX's AdamW set up to compute `torch.optim.AdamW`'s update: with `bias_correction=True`. MLX leaves the bias
    correction out by default, and its first step is then `(1 - beta1) / sqrt(1 - beta2)` times torch's, 3.2 times
    for betas of (0.9, 0.999). No hyper-parameter has a default here: every one is the caller's to give.

    The arguments are checked as torch's constructor checks them, which MLX's does not: a learning rate and a weight
    decay of at least 0, betas in [0, 1), none of them NaN. Three things are stricter than torch's: they must be
    finite; the learning rate is a number (a schedule is not built); and `eps` must be above 0. torch takes an `eps`
    of 0, and so does the arithmetic until a gradient element is exactly zero: the update is then 0 / 0. That is
    every element of `lora_A` on the first step from PEFT's initialisation, where `lora_B` is zero."""
    for what, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay)):
        if not _real(value) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{what} must be a finite number of at least 0, got {value!r}")
    betas = list(betas)
    if len(betas) != 2 or not all(_real(b) and 0 <= b < 1 for b in betas):
        raise ValueError(f"betas must be two numbers in [0, 1), got {betas!r}")
    if not _real(eps) or not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a finite number above 0, got {eps!r}: with 0, a gradient element that is "
                         "exactly zero is updated by 0 / 0")
    return optim.AdamW(learning_rate=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay,
                       bias_correction=True)


def optimizer_step(optimizer, trainable: dict, grads: dict, *, max_grad_norm: float) -> tuple[dict, float]:
    """One step: (the updated matrices, the gradients' global norm before clipping), both evaluated.

    The gradients are scaled down to a global norm of `max_grad_norm` when theirs is larger, as torch's
    `clip_grad_norm_` scales them (`math.inf` clips nothing), and the optimizer applies them. Refused before any
    matrix is touched:

    * matrices that are not float32. At a learning rate of 1e-4 a bf16 element of magnitude 2**-5 or more does not
      move at all, because the update is under half its spacing, and a smaller one moves by whole steps of its
      spacing; in float16 the second moment of small gradients underflows and the update is NaN. So the trainable
      matrices are float32 over whatever base (`lora.attach(..., keep_dtype=True)`).
    * gradients whose keys are not the matrices' keys, or that are not float32 arrays of their matrix's shape. A
      gradient of another shape that broadcasts would be applied, and would leave moments of its own shape in the
      optimizer. A bf16 gradient would have its norm summed in bf16 by MLX's clip (which widens float16 and not
      bf16), a sum that stalls, and would leave bf16 moments;
    * a `max_grad_norm` that is not a positive number;
    * a gradient norm that is not finite, which torch's clip refuses under `error_if_nonfinite` and otherwise lets
      through into the weights."""
    if not trainable or set(grads) != set(trainable):
        raise ValueError(f"the gradients' keys must be the trainable matrices' keys: {len(grads)} against {len(trainable)}")
    narrow = sorted(k for k, v in trainable.items() if v.dtype != mx.float32)
    if narrow:
        raise ValueError(f"the trainable matrices must be float32; {len(narrow)} are not, e.g. {narrow[0]} "
                         f"({trainable[narrow[0]].dtype}): half-precision matrices do not take updates of this size")
    wrong = sorted(k for k, v in trainable.items()
                   if not isinstance(grads[k], mx.array) or grads[k].dtype != mx.float32 or grads[k].shape != v.shape)
    if wrong:
        g = grads[wrong[0]]
        found = f"{g.dtype} {tuple(g.shape)}" if isinstance(g, mx.array) else type(g).__name__
        raise ValueError(f"each gradient must be a float32 array of its matrix's shape; {len(wrong)} are not, e.g. "
                         f"{wrong[0]}: {found} for a matrix of {tuple(trainable[wrong[0]].shape)}")
    if not isinstance(max_grad_norm, numbers.Real) or isinstance(max_grad_norm, bool) or not max_grad_norm > 0:
        raise ValueError(f"max_grad_norm must be a positive number (math.inf to clip nothing), got {max_grad_norm!r}")
    clipped, norm = optim.clip_grad_norm(grads, float(max_grad_norm))
    norm = float(norm)
    if not math.isfinite(norm):
        raise ValueError(f"the gradients' norm is {norm}: not applied")
    updated = optimizer.apply_gradients(clipped, trainable)
    updated = {k: updated[k] for k in trainable}
    mx.eval(updated, optimizer.state)
    return updated, norm


def optimizer_state(optimizer, trainable: dict) -> dict:
    """The optimizer's state as a flat dict of arrays, for a checkpoint: the step count and, for each trainable
    matrix, Adam's two moments, under `m.<key>` and `v.<key>`. The learning rate is not state here: it is constant
    and an argument (`adamw`). Before the first step there is no state to save, which is an error."""
    state = optimizer.state
    missing = sorted(k for k in trainable if k not in state or set(state[k]) != {"m", "v"})
    if missing or "step" not in state:
        raise ValueError(f"the optimizer has no Adam state for {len(missing)} of {len(trainable)} matrices: it has not "
                         "stepped on them yet")
    out = {"step": mx.array(int(state["step"].item()), dtype=mx.uint64)}
    for key in trainable:
        out["m." + key], out["v." + key] = state[key]["m"], state[key]["v"]
    return out


def restore_optimizer(optimizer, trainable: dict, saved: dict) -> None:
    """Put a saved state (`optimizer_state`) into a new optimizer made with the same hyper-parameters, so that its
    next step is the step the saved one would have taken. The saved moments must be exactly the trainable matrices',
    in their shapes and float32."""
    want = {"step"} | {f"{moment}.{key}" for key in trainable for moment in "mv"}
    if set(saved) != want:
        raise ValueError(f"the saved optimizer state is not for these matrices: {len(set(saved) ^ want)} names differ, "
                         f"e.g. {sorted(set(saved) ^ want)[:2]}")
    step = int(np.array(saved["step"]).item())
    if step < 1:
        raise ValueError(f"a saved optimizer state has taken at least one step, got {step}")
    for key, matrix in trainable.items():
        for moment in "mv":
            got = saved[f"{moment}.{key}"]
            if got.shape != matrix.shape or got.dtype != mx.float32:
                raise ValueError(f"{moment}.{key}: saved as {got.dtype} {tuple(got.shape)}, the matrix is float32 "
                                 f"{tuple(matrix.shape)}")
    optimizer.init(trainable)
    state = optimizer.state
    state["step"] = mx.array(step, dtype=state["step"].dtype)
    for key in trainable:
        state[key] = {"m": saved["m." + key], "v": saved["v." + key]}
    mx.eval(state)


# --- sampling the noise level ------------------------------------------------------------------------------------------


def _density_parameters(scheme: str, **parameters) -> None:
    """A scheme's parameters must be given (a TypeError when one is not, as theirs raises) and finite (theirs returns
    NaN, infinite or pinned values for a NaN or infinite parameter, without a word). A std that is negative or NaN is
    left to `torch.normal`, which raises its own RuntimeError for it, in theirs too."""
    for name, value in parameters.items():
        if value is None:
            raise TypeError(f"`{name}` is required for the {scheme!r} density")
        if name == "logit_std" and not value >= 0:
            continue
        if not math.isfinite(value):
            raise ValueError(f"`{name}` must be finite for the {scheme!r} density, got {value}")


def density_for_timestep_sampling(weighting_scheme: str, batch_size: int, logit_mean: float | None = None,
                                  logit_std: float | None = None, mode_scale: float | None = None,
                                  generator=None) -> np.ndarray:
    """diffusers' `compute_density_for_timestep_sampling`: `batch_size` values `u` in [0, 1], float32.

    * `"logit_normal"`: the sigmoid of a normal draw (`logit_mean`, `logit_std`).
    * `"mode"`: `1 - u - mode_scale * (cos(pi * u / 2) ** 2 - 1 + u)` of a uniform draw.
    * anything else: a uniform draw.

    The draw is torch's, from `generator` (a `torch.Generator`; None is torch's global one), so it is the same stream
    as theirs. The arithmetic after it is numpy float32 in their order: equal to theirs bit for bit except for the
    sigmoid's `exp` and the `cos`, which numpy and torch round differently in the last place or two.

    A scheme's own parameters are required, as in theirs (TypeError), and must be finite, which theirs does not check
    (ValueError here). A negative or NaN `logit_std` is torch's own RuntimeError on both sides."""
    import torch

    if weighting_scheme == "logit_normal":
        _density_parameters(weighting_scheme, logit_mean=logit_mean, logit_std=logit_std)
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator).numpy()
        return (np.float32(1) / (np.float32(1) + np.exp(-u))).astype(np.float32)
    if weighting_scheme == "mode":
        _density_parameters(weighting_scheme, mode_scale=mode_scale)
        u = torch.rand(size=(batch_size,), generator=generator).numpy()
        c = np.cos(np.float32(math.pi) * u / np.float32(2))
        return (np.float32(1) - u - np.float32(mode_scale) * (c * c - np.float32(1) + u)).astype(np.float32)
    return torch.rand(size=(batch_size,), generator=generator).numpy()


def loss_weighting_for_sd3(weighting_scheme: str, sigmas) -> np.ndarray:
    """diffusers' `compute_loss_weighting_for_sd3`, on float32 or float64 noise levels (numpy, or an `mx.array`).

    * `"sigma_sqrt"`: `sigma ** -2`, which torch computes as `1 / (sigma * sigma)`; float32 (infinite at sigma 0).
    * `"cosmap"`: `2 / (pi * (1 - 2 * sigma + 2 * sigma ** 2))`, in the input's dtype.
    * anything else: ones.

    `sigma` is the noise level in both conventions (1 is pure noise), so nothing changes for this model's `1 - t`.

    Half-precision and integer levels are refused. Theirs computes a bf16 tensor's weighting in bf16, three digits of
    it; the numpy result here can't hold bf16, and upcasting without saying so would be neither theirs nor exact."""
    if isinstance(sigmas, mx.array):
        if sigmas.dtype not in (mx.float32, mx.float64):
            raise ValueError(f"sigmas must be float32 or float64, got {sigmas.dtype}: cast them first")
        sigmas = np.array(sigmas)
    s = np.asarray(sigmas)
    if s.dtype not in (np.float32, np.float64):
        raise ValueError(f"sigmas must be float32 or float64, got {s.dtype}: cast them first")
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
    `"mode"` maps a draw of 0 to 1.0). The model's timestep for a level is `1 - sigma` in float32; `scale_noise` takes
    those as `mx.array(1 - sigmas)`.

    Every `u` must lie in [0, 1], which a NaN does not: cast to an index a NaN is 0 on this hardware, pure noise for
    every such draw, here and in the torch lookup. The grid must be one non-empty row of finite levels."""
    grid = np.asarray(grid)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(f"the grid must be one non-empty row of noise levels, got shape {grid.shape}")
    if not np.isfinite(grid).all():
        raise ValueError("the grid holds levels that are not finite")
    u = np.asarray(u, dtype=np.float32)
    inside = (u >= 0) & (u <= 1)
    if not inside.all():
        raise ValueError(f"every u must lie in [0, 1] (a NaN does not): got {u[~inside][:4].tolist()}")
    indices = np.minimum((u * np.float32(len(grid))).astype(np.int64), len(grid) - 1)
    return grid[indices], indices
