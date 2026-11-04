"""The denoise loop against a torch loop built from diffusers' own pieces (transformer forward, MiniMaxH3Scheduler x2,
MiniMaxH3SetTimestepsStep.build_row_timesteps, the 0.999 / 1.0 condition levels), tiny model, ref2va layout."""

import numpy as np
import pytest
import torch
from conftest import assert_close, state_mx, to_mx
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from test_dit_tiny import TINY

from h3mlx.denoise import Schedules, denoise, row_timesteps
from h3mlx.packing import Ref, build_ref2va_packed_sequence


def tiny_layout():
    # one image reference (1 latent frame, 8x8 -> 16 rows), one audio reference (2 x 5 rows), 3 target latent frames
    return build_ref2va_packed_sequence(np.array([1] * 4 + [0] * 3 + [1] * 2), [Ref("image"), Ref("audio")],
                                        [(1, 8, 8)], [10], num_latent_frames=3, latent_height=8, latent_width=8,
                                        num_audio_latents=6)


def test_row_timesteps_match():
    lay = tiny_layout()
    want = MiniMaxH3SetTimestepsStep.build_row_timesteps(torch.from_numpy(lay.video_indices),
                                                         torch.from_numpy(lay.audio_indices),
                                                         lay.num_condition_video_rows, lay.num_condition_audio_rows,
                                                         len(lay.text_indices), 0.31, 0.12, 0.999, 1.0)
    uniq, inv = row_timesteps(lay, 0.31, 0.12, 0.999, 1.0)
    assert np.array_equal(uniq, want[0].numpy()) and np.array_equal(inv, want[1].numpy())


@pytest.mark.parametrize("steps", [3, 5])
def test_denoise_loop_matches(steps):
    torch.manual_seed(7)
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if "norm" in name and prm.ndim == 1:
                prm.normal_()
    lay = tiny_layout()
    n_v, n_a, n_t = len(lay.video_indices), len(lay.audio_indices), len(lay.text_indices)
    video = torch.randn(n_v, 16)
    audio = torch.randn(n_a, 6)
    text = torch.randn(1, n_t, 20)

    # torch reference loop, as MiniMaxH3DenoiseStep runs it
    vs, as_ = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    vs.set_timesteps(steps)
    as_.set_timesteps(steps)
    v, a = video.clone(), audio.clone()
    ncv, nca = lay.num_condition_video_rows, lay.num_condition_audio_rows
    t_lay = {k: torch.from_numpy(getattr(lay, k)) for k in ("token_tags", "video_indices", "audio_indices", "text_indices")}
    t_lay["position_ids"] = torch.from_numpy(lay.position_ids)
    with torch.no_grad():
        for i, (t, ta) in enumerate(zip(vs.timesteps, as_.timesteps)):
            uniq, inv = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                t_lay["video_indices"], t_lay["audio_indices"], ncv, nca, n_t, float(t), float(ta), max(float(t), 0.999), 1.0)
            vp, ap = model(v[None], a[None], text, uniq, inv, **t_lay, return_dict=False)
            v[ncv:] = vs.step(vp[0, ncv:].float(), t, v[ncv:], return_dict=False)[0]
            a[nca:] = as_.step(ap[0, nca:].float(), as_.timesteps[i], a[nca:], return_dict=False)[0]

    got_v, got_a = denoise(state_mx(model), dict(model.config), lay, to_mx(text), to_mx(video), to_mx(audio),
                           Schedules.make(steps))
    assert_close(got_v[:ncv], video[:ncv], atol=0, what="conditioning rows untouched")
    assert_close(got_v, v, atol=5e-4, rtol=5e-4, what="video rows after the loop")
    assert_close(got_a, a, atol=5e-4, rtol=5e-4, what="audio rows after the loop")
