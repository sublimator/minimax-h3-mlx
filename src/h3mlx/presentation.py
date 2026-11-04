"""The Qwen3-VL presentation of a ref2va request (diffusers 0.42.1 `encoders.py`,
`MiniMaxH3Ref2VATextEncoderStep._build_presentation` / `_sample_video_condition_frames`).

Pure Python over a tokenizer: labels per reference numbered per modality (`"<Picture i>: "` + an image vision block,
`"<Audio j>: "` alone, `"<Video k>: "` + one `"<t seconds>"` label and video vision block per merged frame pair), then
the prompt verbatim; no chat template, no special tokens. Vision blocks are tagged video (0), text is tagged 1.
"""

from .packing import Ref


def sample_video_condition_frames(num_frames: int, fps: float, sample_fps: float, temporal_patch: int):
    """Frame indices the conditioner reads (every fps/sample_fps-th, deduplicated) and one timestamp per merged block
    (mean of each group of `temporal_patch`, last timestamp repeated to fill the group)."""
    stride = fps / sample_fps
    indices, cursor = [], 0.0
    while round(cursor) < num_frames:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(f"a reference video needs at least {minimum} frames at {fps:g} fps, got {num_frames}")
    ts = [i / sample_fps for i in range(len(indices))]
    ts += [ts[-1]] * (-len(ts) % temporal_patch)
    blocks = [(ts[i] + ts[i + temporal_patch - 1]) / 2 for i in range(0, len(ts), temporal_patch)]
    return indices, blocks


def build_presentation(
    tokenizer,
    prompt: str,
    references: list[Ref],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
    video_block_timestamps: list[list[float]],
    text_tag: int = 1,
    video_tag: int = 0,
) -> tuple[list[int], list[int]]:
    def text(s):
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        return ids, [text_tag] * len(ids)

    def vision(pad, n):
        ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad)] * n
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return ids, [video_tag] * len(ids)

    ids, tags = [], []

    def emit(seg):
        ids.extend(seg[0])
        tags.extend(seg[1])

    n = {"image": 0, "video": 0, "audio": 0}
    for r in references:
        if r.has_audio:
            n["audio"] += 1
            emit(text(f"<Audio {n['audio']}>: "))
        if r.kind == "image":
            n["image"] += 1
            emit(text(f"<Picture {n['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[n["image"] - 1]))
        elif r.kind == "video":
            n["video"] += 1
            emit(text(f"<Video {n['video']}>: "))
            for t in video_block_timestamps[n["video"] - 1]:
                emit(text(f"<{t:.1f} seconds>"))  # round half to even: a 2 fps pair renders "<0.2 seconds>"
                emit(vision("<|video_pad|>", video_block_token_counts[n["video"] - 1]))
    emit(text(prompt))
    return ids, tags

