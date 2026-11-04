"""Packed-sequence layouts of MiniMax H3, in numpy (float64 positions, int64 indices), mirroring diffusers 0.42.1
`modular_pipelines/minimax_h3/before_denoise.py`:

- `build_packed_sequence`: t2va / fl2va, `[text | keyframe conditions | target audio | target video]`
- `build_ref2va_packed_sequence`: ref2va, `[text | reference blocks | target audio | target video]`

The positions must match the reference bit for bit, including its two different float64 summation orders.
"""

from dataclasses import dataclass

import numpy as np

from .ops import ROPE_FRAME_RESCALE, ROPE_FRAMES_PER_LATENT, frame_position_grid, temporal_position_grid


@dataclass(frozen=True)
class Ref:
    """What the packer reads of a reference: its kind ('image' | 'video' | 'audio') and whether a video has audio."""

    kind: str
    has_audio: bool = False


@dataclass
class Layout:
    position_ids: np.ndarray  # (seq, 3) float64
    token_tags: np.ndarray  # (seq,) int64
    video_indices: np.ndarray
    audio_indices: np.ndarray
    text_indices: np.ndarray
    num_condition_video_rows: int
    num_condition_audio_rows: int


def _fill_audio_positions(pos, start, stop, num_audio_latents, rotary_time, width_grid, audio_channels):
    time = rotary_time + np.arange(num_audio_latents, dtype=np.float64)
    pos[start:stop, 0] = np.tile(time, audio_channels)
    pos[start:stop, 2] = np.concatenate(
        [np.full(num_audio_latents, float(width_grid[0])), np.full(num_audio_latents, float(width_grid[-1]))]
    )


def _tags(seq, text_token_tags, video_indices, audio_indices, text_indices, audio_tag, video_tag):
    tags = np.empty(seq, dtype=np.int64)
    tags[text_indices] = np.asarray(text_token_tags, dtype=np.int64)
    tags[audio_indices] = audio_tag
    tags[video_indices] = video_tag
    return tags


def build_packed_sequence(
    text_token_tags,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size=(1, 2, 2),
    audio_channels: int = 2,
    audio_tag: int = 2,
    video_tag: int = 0,
    keyframe_anchors: tuple[str, ...] = (),
) -> Layout:
    _, ph, pw = patch_size
    rows_per_frame = (latent_height // ph) * (latent_width // pw)
    n_text = len(text_token_tags)
    n_cond = len(keyframe_anchors) * rows_per_frame
    n_audio = num_audio_latents * audio_channels
    n_video = num_latent_frames * rows_per_frame
    seq = n_text + n_cond + n_audio + n_video
    cond_start, audio_start = n_text, n_text + n_cond
    video_start = audio_start + n_audio

    pos = np.zeros((seq, 3), dtype=np.float64)
    pos[:n_text, 0] = np.arange(n_text, dtype=np.float64)
    frame_grid, width_grid = frame_position_grid(latent_height, latent_width, ph, pw)

    for i, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            t = float(n_text)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64) * ROPE_FRAME_RESCALE  # numpy pairwise sum, as the ref
            for off in range(len(ROPE_FRAMES_PER_LATENT)):
                spans[off :: len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[off]
            t = float(n_text) + float(spans.sum()) - ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")
        r0, r1 = cond_start + i * rows_per_frame, cond_start + (i + 1) * rows_per_frame
        pos[r0:r1, 0] = t
        pos[r0:r1, 1:] = frame_grid

    _fill_audio_positions(pos, audio_start, video_start, num_audio_latents, float(n_text), width_grid, audio_channels)
    vpos = np.empty((num_latent_frames, rows_per_frame, 3), dtype=np.float64)
    vpos[:, :, 0] = temporal_position_grid(num_latent_frames, float(n_text))[:, None]
    vpos[:, :, 1:] = frame_grid[None]
    pos[video_start:] = vpos.reshape(-1, 3)

    video_indices = np.concatenate([np.arange(cond_start, audio_start), np.arange(video_start, seq)])
    audio_indices = np.arange(audio_start, video_start)
    text_indices = np.arange(n_text)
    tags = _tags(seq, text_token_tags, video_indices, audio_indices, text_indices, audio_tag, video_tag)
    return Layout(pos, tags, video_indices, audio_indices, text_indices, n_cond, 0)


def build_ref2va_packed_sequence(
    text_token_tags,
    references: list[Ref],
    reference_video_shapes: list[tuple[int, int, int]],
    reference_audio_rows: list[int],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size=(1, 2, 2),
    audio_channels: int = 2,
    audio_tag: int = 2,
    video_tag: int = 0,
) -> Layout:
    """`reference_video_shapes`: (latent frames, latent h, latent w) per image/video reference in packed order;
    `reference_audio_rows`: row count per audio-bearing reference (audio refs and videos with audio), in packed order.
    The order of references advances the shared audio/video rotary clock, so it is part of the request."""
    _, ph, pw = patch_size
    n_text = len(text_token_tags)
    n_target_video = num_latent_frames * (latent_height // ph) * (latent_width // pw)
    n_target_audio = num_audio_latents * audio_channels
    vis = iter(reference_video_shapes)
    aud = iter(reference_audio_rows)
    n_ref_video = sum(f * (h // ph) * (w // pw) for f, h, w in reference_video_shapes)
    n_ref_audio = sum(reference_audio_rows)
    seq = n_text + n_ref_video + n_ref_audio + n_target_audio + n_target_video

    pos = np.zeros((seq, 3), dtype=np.float64)
    pos[:n_text, 0] = np.arange(n_text, dtype=np.float64)
    target_frame_grid, target_width_grid = frame_position_grid(latent_height, latent_width, ph, pw)

    video_idx, audio_idx = [], []
    cursor = n_text
    rotary_time = float(n_text)
    for ref in references:
        if ref.kind == "image":
            f, h, w = next(vis)
            n = f * (h // ph) * (w // pw)
            video_idx.append(np.arange(cursor, cursor + n))
            grid, _ = frame_position_grid(h, w, ph, pw)
            pos[cursor : cursor + n, 0] = rotary_time
            pos[cursor : cursor + n, 1:] = grid
            cursor += n
            rotary_time += 1.0  # an image takes one integer rotary slot
        elif ref.kind == "audio":
            n = next(aud)
            lat = n // audio_channels
            audio_idx.append(np.arange(cursor, cursor + n))
            _fill_audio_positions(pos, cursor, cursor + n, lat, rotary_time, target_width_grid, audio_channels)
            cursor += n
            rotary_time += float(lat)
        elif ref.kind == "video":
            na = next(aud) if ref.has_audio else 0
            lat = na // audio_channels
            f, h, w = next(vis)
            nv = f * (h // ph) * (w // pw)
            a0, v0 = cursor, cursor + na
            cursor = v0 + nv
            audio_idx.append(np.arange(a0, v0))
            video_idx.append(np.arange(v0, cursor))
            grid, wgrid = frame_position_grid(h, w, ph, pw)
            _fill_audio_positions(pos, a0, v0, lat, rotary_time, wgrid, audio_channels)
            pos[v0:cursor, 0] = np.repeat(temporal_position_grid(f, rotary_time), grid.shape[0])
            pos[v0:cursor, 1:] = np.tile(grid, (f, 1))
            # sequential float64 sum, NOT numpy's pairwise sum (differs in the last ulp from 16 latent frames on)
            span = sum(ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(f))
            rotary_time += max(float(lat), span)
        else:
            raise ValueError(f"reference kind must be 'image', 'video' or 'audio', got {ref.kind!r}")

    audio_start = cursor
    video_start = audio_start + n_target_audio
    _fill_audio_positions(pos, audio_start, video_start, num_audio_latents, rotary_time, target_width_grid, audio_channels)
    pos[video_start:, 0] = np.repeat(temporal_position_grid(num_latent_frames, rotary_time), target_frame_grid.shape[0])
    pos[video_start:, 1:] = np.tile(target_frame_grid, (num_latent_frames, 1))

    video_indices = np.concatenate(video_idx + [np.arange(video_start, seq)])
    audio_indices = np.concatenate(audio_idx + [np.arange(audio_start, video_start)])
    text_indices = np.arange(n_text)
    tags = _tags(seq, text_token_tags, video_indices, audio_indices, text_indices, audio_tag, video_tag)
    return Layout(pos, tags, video_indices, audio_indices, text_indices, n_ref_video, n_ref_audio)
