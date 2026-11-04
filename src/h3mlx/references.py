"""Reference and canvas geometry of MiniMax H3 (diffusers 0.42.1 `modular_pipeline.resolve_canvas_size`,
`before_encoder.MiniMaxH3Ref2VASetupStep`): numpy + PIL, no torch.

- the target canvas: 16:9 by default, short edge 768, area cap 768 * 1344, both axes rounded to 32;
- an image reference: its own short edge 2048 (upscaling included, no area cap), axes rounded to 32, LANCZOS;
- a video reference: frames held onto the 24 fps grid (ffmpeg `fps`-filter style), truncated to the generated frame
  count, put on the canvas its own aspect ratio resolves to, LANCZOS.
"""

import math

import numpy as np
from PIL import Image

FPS = 24
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
REFERENCE_IMAGE_SHORT_EDGE = 2048
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5
AUDIO_LATENTS_PER_SECOND = 40


def resolve_canvas_size(aspect_w, aspect_h, multiple=CANVAS_MULTIPLE, short_edge=CANVAS_SHORT_EDGE,
                        max_pixels=CANVAS_MAX_PIXELS, min_ratio=0.25, max_ratio=4.0) -> tuple[int, int]:
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError(f"aspect must be positive, got {aspect_w}:{aspect_h}")
    r = aspect_w / aspect_h
    if not min_ratio <= r <= max_ratio:
        raise ValueError(f"aspect {r:g} outside [{min_ratio:g}, {max_ratio:g}]")
    w, h = (short_edge * r, float(short_edge)) if r >= 1.0 else (float(short_edge), short_edge / r)
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        w, h = w * s, h * s
    return max(multiple, round(h / multiple) * multiple), max(multiple, round(w / multiple) * multiple)


def align_num_frames(n: int) -> int:
    if n < 1:
        raise ValueError(f"num_frames must be positive, got {n}")
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        n += 1
    return n


def video_latent_num_frames(n: int) -> int:
    if n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be 17 * n + 5, got {n}")
    return (n - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(n: int, fps: float = FPS) -> int:
    return int(round(n / fps * AUDIO_LATENTS_PER_SECOND))


def image_reference_size(width: int, height: int, short_edge=REFERENCE_IMAGE_SHORT_EDGE, multiple=CANVAS_MULTIPLE):
    """(target_width, target_height) for an image reference."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"image must be within 1:4 and 4:1, got {width}x{height}")
    s = short_edge / min(width, height)
    return max(multiple, round(width * s / multiple) * multiple), max(multiple, round(height * s / multiple) * multiple)


def normalize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    size = image_reference_size(*image.size)
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def normalize_video_frames(frames: np.ndarray, fps: float, num_frames: int, multiple: int = CANVAS_MULTIPLE,
                           short_edge: int = CANVAS_SHORT_EDGE, max_pixels: int = CANVAS_MAX_PIXELS) -> np.ndarray:
    """uint8 (T, H, W, 3) at `fps` -> 24 fps, truncated to num_frames, on its own aspect's canvas (LANCZOS)."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"video must be (T, H, W, 3), got {frames.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if fps != FPS:
        scale = FPS / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)
    frames = frames[:num_frames]
    h, w = resolve_canvas_size(frames.shape[2], frames.shape[1], multiple, short_edge, max_pixels)
    if frames.shape[1:3] == (h, w):
        return frames
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.LANCZOS)) for f in frames])
