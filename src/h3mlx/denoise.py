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


@dataclass
class Schedules:
    video_sigmas: np.ndarray
    audio_sigmas: np.ndarray

    @classmethod
    def make(cls, num_inference_steps: int, video_shift: float = 12.0, audio_shift: float = 3.0):
        return cls(ops.flow_sigmas(num_inference_steps, video_shift), ops.flow_sigmas(num_inference_steps, audio_shift))

    @property
    def video_timesteps(self):
        return (np.float32(1) - self.video_sigmas[:-1]).astype(np.float32)

    @property
    def audio_timesteps(self):
        return (np.float32(1) - self.audio_sigmas[:-1]).astype(np.float32)


def row_timesteps(layout: Layout, video_t: float, audio_t: float, cond_video_t: float, cond_audio_t: float):
    """Per-row timesteps reduced to (sorted distinct values, inverse index), like `build_row_timesteps`."""
    seq = len(layout.video_indices) + len(layout.audio_indices) + len(layout.text_indices)
    rt = np.full(seq, np.float32(video_t), dtype=np.float32)
    rt[layout.video_indices[: layout.num_condition_video_rows]] = cond_video_t
    rt[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_t
    rt[layout.audio_indices[: layout.num_condition_audio_rows]] = cond_audio_t
    uniq, inv = np.unique(rt, return_inverse=True)
    return uniq, inv


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
):
    """video_rows (n_video, patch_dim) with conditioning rows first; audio_rows (n_audio, audio_ch) with reference rows
    first; prompt_embeds (1, n_text, text_dim). Returns the final (video_rows, audio_rows). With `cache`
    (`ops.ModulationCache` built from `schedule_timesteps`), the AdaLN projections are not evaluated per step."""
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
        uniq, inv = row_timesteps(layout, float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0)
        v_pred, a_pred = ops.dit_forward(
            params, cfg, video_rows[None], audio_rows[None], prompt_embeds, mx.array(uniq), mx.array(inv), **lay,
            cache=cache,
        )
        new_v = ops.flow_step(v_pred[0, ncv:].astype(mx.float32), float(t), video_rows[ncv:],
                              float(schedules.video_sigmas[i]), float(schedules.video_sigmas[i + 1]))
        new_a = ops.flow_step(a_pred[0, nca:].astype(mx.float32), float(ta), audio_rows[nca:],
                              float(schedules.audio_sigmas[i]), float(schedules.audio_sigmas[i + 1]))
        video_rows = mx.concatenate([video_rows[:ncv], new_v]) if ncv else new_v
        audio_rows = mx.concatenate([audio_rows[:nca], new_a]) if nca else new_a
        mx.eval(video_rows, audio_rows)
        if on_step:
            on_step(i, t)
    return video_rows, audio_rows


def schedule_timesteps(layout: Layout, schedules: Schedules) -> np.ndarray:
    """Every distinct timestep any row takes over the whole run: what a ModulationCache needs."""
    vals = []
    for t, ta in zip(schedules.video_timesteps, schedules.audio_timesteps):
        uniq, _ = row_timesteps(layout, float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0)
        vals.append(uniq)
    return np.unique(np.concatenate(vals))
