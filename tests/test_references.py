"""Canvas and reference geometry against diffusers 0.42.1, exact."""

import numpy as np
import pytest
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as mp
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep as Setup
from PIL import Image

from h3mlx import references as R


@pytest.mark.parametrize("aw,ah", [(16, 9), (9, 16), (1, 1), (4, 3), (21, 9), (1080, 1920), (3840, 1600), (4, 1), (1, 4)])
def test_canvas_matches(aw, ah):
    assert R.resolve_canvas_size(aw, ah) == mp.resolve_canvas_size(aw, ah, 32, 768, 768 * 1344)


@pytest.mark.parametrize("n", [1, 100, 121, 124, 125, 200, 360])
def test_frame_counts_match(n):
    a = R.align_num_frames(n)
    assert a == mp.align_num_frames(n, 17, 5)
    assert R.video_latent_num_frames(a) == mp.video_latent_num_frames(a, 17, 5)
    assert R.audio_latent_num_frames(a) == mp.audio_latent_num_frames(a)


@pytest.mark.parametrize("w,h", [(1080, 1920), (512, 512), (3000, 2000), (640, 360), (2048, 2048)])
def test_image_reference_resize_matches(w, h):
    rng = np.random.default_rng(w + h)
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    got = R.normalize_image(img)
    # the diffusers path: size rule from MiniMaxH3Ref2VASetupStep.__call__, resize by VaeImageProcessor (lanczos)
    scale = 2048 / min(w, h)
    tw, th = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
    want = img if img.size == (tw, th) else VaeImageProcessor(vae_scale_factor=16).resize(img, height=th, width=tw)
    assert got.size == want.size
    assert np.array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("fps,n_in,size", [(24.0, 40, (360, 640)), (30.0, 50, (360, 640)), (25.0, 30, (432, 768)),
                                           (12.0, 20, (768, 1344)), (60.0, 90, (100, 100))])
def test_video_normalization_matches(fps, n_in, size):
    rng = np.random.default_rng(n_in)
    frames = rng.integers(0, 256, (n_in, *size, 3), dtype=np.uint8)
    want = Setup._normalize_video_condition(frames, fps, 22, 32, 768, 768 * 1344, 24.0)
    got = R.normalize_video_frames(frames, fps, 22)
    assert got.shape == want.shape and np.array_equal(got, want)
