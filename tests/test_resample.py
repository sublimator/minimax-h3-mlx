"""The vendored resampler against real torchaudio 2.9.1 (+ torch 2.9.1) outputs, generated in a separate environment
because torchaudio's wheels pin an older torch than ours (tests/data/resample_ref.npz)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from h3mlx.pipeline import _normalize_audio
from h3mlx.resample import resample

REF = np.load(Path(__file__).parent / "data" / "resample_ref.npz")


@pytest.mark.parametrize("sr", [48000, 44100, 22050, 16000])
@pytest.mark.parametrize("ch", [1, 2])
def test_resample_matches_torchaudio(sr, ch):
    w = torch.from_numpy(REF[f"in_{sr}_{ch}"])
    if ch == 1:
        w = w.expand(2, -1).contiguous()
    got = resample(w, sr, 32000).numpy()
    want = REF[f"out_{sr}_{ch}"]
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=2e-6, rtol=0)


def test_normalize_audio_truncates_upmixes_resamples():
    w = REF["in_48000_1"]
    got = _normalize_audio(w, 48000, 32000, max_duration=0.05)  # 2400 samples kept at 48 kHz
    assert got.shape == (2, 1600)
    want = resample(torch.from_numpy(w[:, :2400]).expand(2, -1).contiguous(), 48000, 32000).numpy()
    np.testing.assert_array_equal(got, want)
