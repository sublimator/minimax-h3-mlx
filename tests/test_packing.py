"""Packed layouts against diffusers 0.42.1, bit for bit (float64 positions, indices, tags, counts)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
)

from h3mlx.packing import Ref, build_packed_sequence, build_ref2va_packed_sequence

AUDIO_CH, AUDIO_TAG, VIDEO_TAG = 2, 2, 0


def text_tags(n, n_vision=0, seed=0):
    """Text rows are tagged 1, except a reference's vision block inside the text, tagged 0."""
    t = torch.ones(n, dtype=torch.long)
    if n_vision:
        start = torch.randint(0, n - n_vision + 1, (1,), generator=torch.Generator().manual_seed(seed)).item()
        t[start : start + n_vision] = 0
    return t


def same(layout, ref_tuple):
    pos, tags, vi, ai, ti, ncv, nca = ref_tuple
    assert np.array_equal(layout.position_ids, pos.numpy()), "positions differ"
    assert layout.position_ids.dtype == np.float64
    for got, want, name in ((layout.token_tags, tags, "tags"), (layout.video_indices, vi, "video"),
                            (layout.audio_indices, ai, "audio"), (layout.text_indices, ti, "text")):
        assert np.array_equal(got, want.numpy()), f"{name} differ"
    assert (layout.num_condition_video_rows, layout.num_condition_audio_rows) == (ncv, nca)


@pytest.mark.parametrize("anchors", [(), ("first",), ("last",), ("first", "last")])
@pytest.mark.parametrize("frames,lh,lw,n_audio", [(31, 24, 42, 205), (6, 16, 16, 40), (16, 30, 54, 101), (21, 8, 14, 3)])
def test_t2va_fl2va_layout_matches(anchors, frames, lh, lw, n_audio):
    tags = text_tags(37, n_vision=12 if anchors else 0)
    want = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        tags, frames, lh, lw, n_audio, (1, 2, 2), AUDIO_CH, AUDIO_TAG, VIDEO_TAG, anchors
    )
    got = build_packed_sequence(tags.numpy(), frames, lh, lw, n_audio, (1, 2, 2), AUDIO_CH, AUDIO_TAG, VIDEO_TAG, anchors)
    same(got, want)


REF_CASES = {
    "one image": [("image", False)],
    "three images": [("image", False), ("image", False), ("image", False)],
    "audio then image": [("audio", False), ("image", False)],
    "video with audio": [("video", True)],
    "video without audio": [("video", False)],
    "long video, long soundtrack": [("video", True), ("image", False)],  # audio wins max(): span order irrelevant
    "long silent video (ulp case)": [("video", False), ("image", False)],  # the span decides: order matters
    "long video, short soundtrack (ulp case)": [("video", True), ("image", False)],
    "mixed": [("image", False), ("video", True), ("audio", False), ("video", False), ("image", False)],
}


@pytest.mark.parametrize("case", list(REF_CASES))
def test_ref2va_layout_matches(case):
    g = torch.Generator().manual_seed(len(case))
    refs = [Ref(k, a) for k, a in REF_CASES[case]]
    long = "long" in case
    shapes, audio_rows = [], []
    for r in refs:
        if r.kind == "image":
            shapes.append((1, 2 * int(torch.randint(4, 20, (1,), generator=g)), 2 * int(torch.randint(4, 20, (1,), generator=g))))
        elif r.kind == "video":
            frames = 26 if long else int(torch.randint(2, 9, (1,), generator=g))
            shapes.append((frames, 16, 28))
            if r.has_audio:
                audio_rows.append(AUDIO_CH * (20 if "short soundtrack" in case else frames * 7 + 3))
        else:
            audio_rows.append(AUDIO_CH * int(torch.randint(5, 60, (1,), generator=g)))
    tags = text_tags(53, n_vision=20 * sum(r.kind != "audio" for r in refs) // 2)
    cond = [torch.zeros(1, 24, *s) for s in shapes]
    aud = [torch.zeros(n, 32) for n in audio_rows]
    want = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        tags, [SimpleNamespace(kind=r.kind, has_audio=r.has_audio) for r in refs], cond, aud,
        31, 24, 42, 205, (1, 2, 2), AUDIO_CH, AUDIO_TAG, VIDEO_TAG,
    )
    got = build_ref2va_packed_sequence(tags.numpy(), refs, shapes, audio_rows, 31, 24, 42, 205, (1, 2, 2),
                                       AUDIO_CH, AUDIO_TAG, VIDEO_TAG)
    same(got, want)


def test_ref2va_reference_order_changes_the_clock():
    """The reference order is part of the request: swapping two references must change the target positions."""
    args = dict(num_latent_frames=6, latent_height=16, latent_width=16, num_audio_latents=40)
    a = build_ref2va_packed_sequence(np.ones(5), [Ref("image"), Ref("audio")], [(1, 8, 8)], [20], **args)
    b = build_ref2va_packed_sequence(np.ones(5), [Ref("audio"), Ref("image")], [(1, 8, 8)], [20], **args)
    assert a.position_ids.shape == b.position_ids.shape
    assert not np.array_equal(a.position_ids, b.position_ids)
