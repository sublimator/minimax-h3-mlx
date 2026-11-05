"""The ref2va presentation against diffusers 0.42.1, with the released Qwen tokenizer (MiniMax-H3 @ 42ed227)."""

from types import SimpleNamespace

import pytest
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep as Step
from conftest import RELEASE
from transformers import AutoTokenizer

from h3mlx.packing import Ref
from h3mlx.presentation import build_presentation, sample_video_condition_frames

TOKENIZER_DIR = RELEASE / "tokenizer"


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER_DIR.exists():
        pytest.skip("release tokenizer not present (run scripts/fetch_test_fixtures.py)")
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@pytest.mark.parametrize("n_frames,fps,sample_fps,tp", [(124, 24.0, 2.0, 2), (49, 24.0, 2.0, 2), (37, 24.0, 2.0, 2),
                                                        (200, 24.0, 3.0, 2), (13, 24.0, 2.0, 2), (360, 24.0, 2.0, 2)])
def test_video_condition_sampling_matches(n_frames, fps, sample_fps, tp):
    import numpy as np

    frames = np.arange(n_frames)[:, None, None, None] * np.ones((1, 2, 2, 3), dtype=np.int64)
    want_frames, want_ts = Step._sample_video_condition_frames(frames, fps, sample_fps, tp)
    idx, ts = sample_video_condition_frames(n_frames, fps, sample_fps, tp)
    assert [int(f[0, 0, 0]) for f in want_frames] == idx
    assert ts == want_ts


def test_video_condition_sampling_too_short_raises():
    with pytest.raises(ValueError):
        sample_video_condition_frames(12, 24.0, 2.0, 2)


CASES = {
    "one picture": ([("image", False)], [256], [], []),
    "two pictures and audio": ([("image", False), ("audio", True), ("image", False)], [256, 1024], [], []),
    "video with sound then picture": ([("video", True), ("image", False)], [64], [120], [[0.2, 1.2, 2.2]]),
    "two silent videos": ([("video", False), ("video", False)], [], [48, 80], [[0.2, 1.2], [0.2]]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_presentation_matches(tok, case):
    refs_spec, img_counts, vid_counts, vid_ts = CASES[case]
    prompt = "A woman with copper hair in a camel coat turns to the camera and smiles. Audio: rain on stone."
    want = Step._build_presentation(tok, prompt, [SimpleNamespace(kind=k, has_audio=a) for k, a in refs_spec],
                                    img_counts, vid_counts, vid_ts)
    got = build_presentation(tok, prompt, [Ref(k, a) for k, a in refs_spec], img_counts, vid_counts, vid_ts)
    assert got == want
    assert len(got[0]) == len(got[1])


def test_round_half_to_even_label(tok):
    """The first 2 fps pair is (0 + 0.5) / 2 = 0.25 and renders as '<0.2 seconds>'."""
    _, ts = sample_video_condition_frames(124, 24.0, 2.0, 2)
    assert ts[0] == 0.25 and f"<{ts[0]:.1f} seconds>" == "<0.2 seconds>"
