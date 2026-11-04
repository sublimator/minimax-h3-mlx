"""Inputs diffusers 0.42.1 refuses and an earlier h3mlx accepted. Failing here is the finding."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from h3mlx.pipeline import AudioRef, Components, ImageRef, VideoRef, setup
from h3mlx.vae import VaeConfig


def components():
    return Components(
        tokenizer=None,
        processor=None,
        encode_text=None,
        vae=SimpleNamespace(cfg=VaeConfig()),
        vae_latents_mean=(),
        vae_latents_std=(),
        audio_vae=None,
        audio_latents_mean=(),
        audio_latents_std=(),
        dit_params={},
        dit_cfg={"patch_size": (1, 2, 2)},
    )


def still():
    return ImageRef(Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)))


def test_off_multiple_canvas_is_refused():
    """100x160 is not a multiple of 32. Diffusers raises (before_encoder.py 397).
    setup() keeps 100x160; 100 // 16 is 6, and the decode is 96 px tall."""
    with pytest.raises(ValueError):
        setup(components(), [still()], 100, 160, 124)


def test_1080p_canvas_is_refused():
    """1080x1920 is the size a file actually has. 1080 // 16 is 67, which patchify then rejects."""
    with pytest.raises(ValueError):
        setup(components(), [still()], 1080, 1920, 124)


def test_audio_only_is_refused():
    wave = np.zeros((2, 32000 * 6), np.float32)
    with pytest.raises(ValueError):
        setup(components(), [AudioRef(wave, 32000)], None, None, 124)


def test_four_videos_are_refused():
    vids = [VideoRef(np.zeros((30, 64, 64, 3), np.uint8), 24.0) for _ in range(4)]
    with pytest.raises(ValueError):
        setup(components(), vids, None, None, 124)


def test_empty_references_are_refused():
    with pytest.raises(ValueError):
        setup(components(), [], None, None, 124)


def test_48khz_soundtrack_is_resampled():
    """A video file's audio is 48000 Hz. Diffusers resamples with torchaudio.
    Six seconds truncated to the 124-frame duration (124/24 s) at 32 kHz is 165333 samples."""
    wave = np.zeros((1, 48000 * 6), np.float32)
    _, _, _, refs = setup(components(), [AudioRef(wave, 48000), still()], None, None, 124)
    # diffusers truncates at the native rate, int(124/24 * 48000) = 248000 samples, then torchaudio resamples
    # 48k -> 32k (3 -> 2 after the gcd) to ceil(2 * 248000 / 3) = 165334 (was int(124/24 * 32000) = 165333)
    assert refs[0].audio.shape == (2, 165334)
