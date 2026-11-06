"""The MiniMax H3 denoise loop in MLX (diffusers 0.42.1 `denoise.py` + `MiniMaxH3SetTimestepsStep`).

One forward per step serves every row: generated video rows step down the video schedule (shift 12), generated audio
rows down the audio schedule (shift 3), visual conditioning rows stay pinned at max(t, 0.999), audio reference rows at
1.0, and text rows inherit the video timestep. Only the generated rows are ever written. The checkpoint is
guidance-distilled: no unconditional pass.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import ops
from .packing import Layout

KEYFRAME_NOISE_AUG = 0.999


def shift_grid(raw, shift: float) -> np.ndarray:
    """A raw sigma grid through the exponential shift, `s * sigma / (1 + (s - 1) * sigma)` in float32: HyperFlow's
    `validate_sigmas` + `shift_sigmas` (the formula of `MiniMaxH3Scheduler.set_timesteps`)."""
    if shift <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}")
    base = np.asarray(raw, dtype=np.float32).reshape(-1)
    if base.size < 2 or not (base[1:] < base[:-1]).all() or base[0] > 1.0 or base[-1] != 0.0:
        raise ValueError(f"a sigma grid must be strictly decreasing from at most 1.0 to exactly 0.0, got {base.tolist()}")
    return (np.float32(shift) * base / (np.float32(1.0) + np.float32(shift - 1.0) * base)).astype(np.float32)


@dataclass
class Schedules:
    video_sigmas: np.ndarray
    audio_sigmas: np.ndarray
    two_time: bool = False  # HyperFlow: every step is also conditioned on its endpoint r = 1 - sigma[i + 1]

    @classmethod
    def make(cls, num_inference_steps: int, video_shift: float = 12.0, audio_shift: float = 3.0):
        return cls(ops.flow_sigmas(num_inference_steps, video_shift), ops.flow_sigmas(num_inference_steps, audio_shift))

    @classmethod
    def from_grid(cls, raw, video_shift: float = 12.0, audio_shift: float = 3.0, two_time: bool = True):
        """A fixed raw grid (HyperFlow's 9 points), shifted per modality, used verbatim by the Euler steps."""
        return cls(shift_grid(raw, video_shift), shift_grid(raw, audio_shift), two_time)

    @property
    def video_endpoints(self):
        return (np.float32(1) - self.video_sigmas[1:]).astype(np.float32)

    @property
    def audio_endpoints(self):
        return (np.float32(1) - self.audio_sigmas[1:]).astype(np.float32)

    @property
    def video_timesteps(self):
        return (np.float32(1) - self.video_sigmas[:-1]).astype(np.float32)

    @property
    def audio_timesteps(self):
        return (np.float32(1) - self.audio_sigmas[:-1]).astype(np.float32)


def assert_finite(what: str, a: mx.array) -> None:
    """Raise if `a` holds a NaN or an infinity (a silent NaN would otherwise become a black or silent clip)."""
    if not bool(mx.all(mx.isfinite(a.astype(mx.float32)))):
        raise FloatingPointError(f"non-finite values in {what}")


def row_timesteps(layout: Layout, video_t: float, audio_t: float, cond_video_t: float, cond_audio_t: float):
    """Per-row timesteps reduced to (sorted distinct values, inverse index), like `build_row_timesteps`."""
    seq = len(layout.video_indices) + len(layout.audio_indices) + len(layout.text_indices)
    rt = np.full(seq, np.float32(video_t), dtype=np.float32)
    rt[layout.video_indices[: layout.num_condition_video_rows]] = cond_video_t
    rt[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_t
    rt[layout.audio_indices[: layout.num_condition_audio_rows]] = cond_audio_t
    uniq, inv = np.unique(rt, return_inverse=True)
    return uniq, inv


def row_time_pairs(layout: Layout, video_t: float, video_r: float, audio_t: float, audio_r: float,
                   cond_video_t: float, cond_audio_t: float):
    """HyperFlow's `build_row_time_pairs`: every row gets (t, r); conditioning rows keep r = t. Reduced to the distinct
    pairs, sorted lexicographically like `torch.unique(dim=0)`, and each row's index into them."""
    seq = len(layout.video_indices) + len(layout.audio_indices) + len(layout.text_indices)
    t = np.full(seq, np.float32(video_t), dtype=np.float32)
    r = np.full(seq, np.float32(video_r), dtype=np.float32)
    cv = layout.video_indices[: layout.num_condition_video_rows]
    t[cv] = r[cv] = cond_video_t
    ga = layout.audio_indices[layout.num_condition_audio_rows :]
    t[ga], r[ga] = audio_t, audio_r
    ca = layout.audio_indices[: layout.num_condition_audio_rows]
    t[ca] = r[ca] = cond_audio_t
    pairs, inv = np.unique(np.stack([t, r], axis=-1), axis=0, return_inverse=True)
    return pairs[:, 0].copy(), pairs[:, 1].copy(), inv.reshape(-1)


def denoise(
    params: dict,
    cfg: dict,
    layout: Layout,
    prompt_embeds: mx.array,
    video_rows: mx.array,
    audio_rows: mx.array,
    schedules: Schedules,
    on_step=None,
    cache: "ops.ModulationCache | None" = None,
    check_finite: bool = False,
):
    """video_rows (n_video, patch_dim) with conditioning rows first; audio_rows (n_audio, audio_ch) with reference rows
    first; prompt_embeds (1, n_text, text_dim). Returns the final (video_rows, audio_rows). With `cache`
    (`ops.ModulationCache` built from `schedule_calls`), the AdaLN projections are not evaluated per step."""
    lay = {
        "token_tags": mx.array(layout.token_tags),
        "position_ids": mx.array(layout.position_ids.astype(np.float32)),
        "video_indices": mx.array(layout.video_indices),
        "audio_indices": mx.array(layout.audio_indices),
        "text_indices": mx.array(layout.text_indices),
    }
    ncv, nca = layout.num_condition_video_rows, layout.num_condition_audio_rows
    vts, ats = schedules.video_timesteps, schedules.audio_timesteps
    for i, (t, ta) in enumerate(zip(vts, ats)):
        cond_t = max(float(t), KEYFRAME_NOISE_AUG)
        if schedules.two_time:
            uniq, ends, inv = row_time_pairs(layout, float(t), float(schedules.video_endpoints[i]), float(ta),
                                             float(schedules.audio_endpoints[i]), cond_t, 1.0)
            endpoint = mx.array(ends)
        else:
            uniq, inv = row_timesteps(layout, float(t), float(ta), cond_t, 1.0)
            endpoint = None
        v_pred, a_pred = ops.dit_forward(
            params, cfg, video_rows[None], audio_rows[None], prompt_embeds, mx.array(uniq), mx.array(inv), **lay,
            cache=cache, endpoint=endpoint,
        )
        new_v = ops.flow_step(v_pred[0, ncv:].astype(mx.float32), float(t), video_rows[ncv:],
                              float(schedules.video_sigmas[i]), float(schedules.video_sigmas[i + 1]))
        new_a = ops.flow_step(a_pred[0, nca:].astype(mx.float32), float(ta), audio_rows[nca:],
                              float(schedules.audio_sigmas[i]), float(schedules.audio_sigmas[i + 1]))
        video_rows = mx.concatenate([video_rows[:ncv], new_v]) if ncv else new_v
        audio_rows = mx.concatenate([audio_rows[:nca], new_a]) if nca else new_a
        mx.eval(video_rows, audio_rows)
        if check_finite:
            assert_finite(f"step {i} video rows", video_rows)
            assert_finite(f"step {i} audio rows", audio_rows)
        if on_step:
            on_step(i, t)
    return video_rows, audio_rows


def schedule_calls(layout: Layout, schedules: Schedules) -> list[np.ndarray]:
    """Each model call's distinct timesteps over the whole run, exactly as `denoise` passes them: what a
    ModulationCache needs to build bit-identical tables."""
    if schedules.two_time:
        raise NotImplementedError("ModulationCache is keyed by t alone; HyperFlow's tables depend on (t, r)")
    return [row_timesteps(layout, float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0)[0]
            for t, ta in zip(schedules.video_timesteps, schedules.audio_timesteps)]
