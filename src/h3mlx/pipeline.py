"""ref2va end to end in MLX, mirroring diffusers 0.42.1 `MiniMaxH3Blocks` for a request with `references`:

setup (canvas, 17n+5 frames, normalise references) -> Qwen3-VL presentation + hidden_states[layer] ->
reference latents (VAE condition recipe; audio posterior mean) -> ref2va layout -> noise (conditions first, then target
video, then target audio rows, from ONE generator) -> two-schedule denoise on `transformer_ref` -> decode video + audio.

Randomness: `Noise` draws from torch's CPU generator in channels-first shapes, so a seed reproduces diffusers' draws.
The VAE conditioning posterior is sampled from a separate generator seeded 42, as the release does.
"""

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from PIL import Image

from . import ops
from . import references as R
from .denoise import Schedules, denoise
from .packing import Ref, build_ref2va_packed_sequence
from .presentation import build_presentation, sample_video_condition_frames
from .vae import VideoVAE, encode_condition

PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NOISE_AUG = 0.999
ENCODE_SEED = 42
AUDIO_CHANNELS = 2


@dataclass
class ImageRef:
    image: Image.Image
    kind = "image"
    has_audio = False


@dataclass
class VideoRef:
    frames: np.ndarray  # (T, H, W, 3) uint8
    fps: float = 24.0
    audio: np.ndarray | None = None  # (channels, samples) at sample_rate
    sample_rate: int | None = None
    kind = "video"

    @property
    def has_audio(self):
        return self.audio is not None


@dataclass
class AudioRef:
    audio: np.ndarray  # (channels, samples)
    sample_rate: int | None = None
    kind = "audio"
    has_audio = True


class Noise:
    """One torch CPU generator, drawn in order, channels-first shapes (diffusers' `randn_tensor` on CPU, float32)."""

    def __init__(self, seed: int):
        import torch

        self._torch = torch
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, shape) -> np.ndarray:
        return self._torch.randn(tuple(shape), generator=self.g, dtype=self._torch.float32).numpy()


@dataclass
class Components:
    tokenizer: object  # transformers Qwen2TokenizerFast
    processor: object  # transformers Qwen3VLProcessor (CPU preprocessing only)
    encode_text: object  # callable(token_ids: list[int], vision_inputs: dict[str, np.ndarray]) -> mx.array (1, n, D)
    vae: VideoVAE
    vae_latents_mean: tuple
    vae_latents_std: tuple
    audio_vae: object  # h3mlx.audio_vae.AudioVAE
    audio_latents_mean: tuple
    audio_latents_std: tuple
    dit_params: dict
    dit_cfg: dict
    audio_sampling_rate: int = 32000
    reference_image_short_edge: int = R.REFERENCE_IMAGE_SHORT_EDGE
    canvas_short_edge: int = R.CANVAS_SHORT_EDGE
    canvas_max_pixels: int = R.CANVAS_MAX_PIXELS
    video_sample_fps: float = 2.0
    extra: dict = field(default_factory=dict)

    @property
    def canvas_multiple(self):
        return self.vae.cfg.spatial_ratio * self.dit_cfg["patch_size"][2]


MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS, MAX_REFERENCES = 9, 3, 3, 12


def _normalize_audio(wave: np.ndarray, sr: int, target_sr: int, max_duration: float) -> np.ndarray:
    """`MiniMaxH3Ref2VASetupStep._normalize_audio_condition`: truncate at the native rate, mono -> stereo, one
    torchaudio sinc resample (h3mlx.resample is torchaudio's own code)."""
    import torch

    from .resample import resample

    w = torch.as_tensor(np.asarray(wave))
    if w.ndim != 2 or w.shape[0] not in (1, 2):
        raise ValueError(f"audio must be (channels, samples) mono or stereo, got {tuple(w.shape)}")
    w = w.to(torch.float32)[:, : int(max_duration * sr)]
    if w.shape[0] != 2:
        w = w.expand(2, -1).contiguous()
    return resample(w, int(sr), int(target_sr)).numpy()


def setup(c: Components, references, height, width, num_frames):
    """`MiniMaxH3Ref2VASetupStep`: validate, resolve the canvas and frame count, normalise references in order."""
    m = c.canvas_multiple
    if (height is None) != (width is None):
        raise ValueError("height and width have to be passed together, or neither of them")
    if height is not None and (height % m or width % m):
        raise ValueError(f"height and width must be multiples of {m}, got {height}x{width}")
    if not references:
        raise ValueError("ref2va needs at least one reference")
    kinds = [getattr(r, "kind", None) for r in references]
    if any(k not in ("image", "video", "audio") for k in kinds):
        raise ValueError(f"references must be ImageRef, VideoRef or AudioRef, got kinds {kinds}")
    for kind, limit in (("image", MAX_IMAGES), ("video", MAX_VIDEOS), ("audio", MAX_AUDIOS)):
        if kinds.count(kind) > limit:
            raise ValueError(f"at most {limit} {kind} references, got {kinds.count(kind)}")
    if len(kinds) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} references in total, got {len(kinds)}")
    if set(kinds) == {"audio"}:
        raise ValueError("an audio reference has to be paired with at least one image or video reference")
    if height is None:
        height, width = R.resolve_canvas_size(16, 9, m, c.canvas_short_edge, c.canvas_max_pixels)
    num_frames = R.align_num_frames(num_frames)
    if not 5.0 <= num_frames / R.FPS <= 15.0:
        raise ValueError(f"duration must be 5-15 s at 24 fps, got {num_frames} frames")
    out = []
    for r in references:
        wave = None
        if r.has_audio:
            wave = _normalize_audio(r.audio, r.sample_rate or c.audio_sampling_rate, c.audio_sampling_rate, num_frames / R.FPS)
        if r.kind == "image":
            out.append(ImageRef(R.normalize_image(r.image, c.reference_image_short_edge, m)))
        elif r.kind == "video":
            frames = R.normalize_video_frames(r.frames, float(r.fps), num_frames, m, c.canvas_short_edge,
                                              c.canvas_max_pixels)
            out.append(VideoRef(frames, float(R.FPS), wave, None if wave is None else c.audio_sampling_rate))
        else:
            out.append(AudioRef(wave, c.audio_sampling_rate))
    return height, width, num_frames, out


def vision_features(c: Components, refs):
    proc = c.processor
    merge = proc.image_processor.merge_size**2
    vision, img_counts, vid_counts, vid_ts = {}, [], [], []
    images = [r.image for r in refs if r.kind == "image"]
    if images:
        f = proc.image_processor(images=images, return_tensors="np")
        vision["pixel_values"], vision["image_grid_thw"] = f["pixel_values"], f["image_grid_thw"]
        img_counts = [int(np.prod(g)) // merge for g in f["image_grid_thw"]]
    videos = [r for r in refs if r.kind == "video"]
    if videos:
        tp = proc.video_processor.temporal_patch_size
        sampled = []
        for r in videos:
            idx, ts = sample_video_condition_frames(r.frames.shape[0], float(R.FPS), c.video_sample_fps, tp)
            sampled.append(np.stack([r.frames[i] for i in idx]))
            vid_ts.append(ts)
        f = proc.video_processor(videos=sampled, do_sample_frames=False, return_tensors="np")
        vision["pixel_values_videos"], vision["video_grid_thw"] = f["pixel_values_videos"], f["video_grid_thw"]
        vid_counts = [int(g[1]) * int(g[2]) // merge for g in f["video_grid_thw"]]
        for ts, g in zip(vid_ts, f["video_grid_thw"]):
            if int(g[0]) != len(ts):
                raise ValueError(f"processor made {int(g[0])} video blocks, presentation labels {len(ts)}")
    return vision, img_counts, vid_counts, vid_ts


def encode_references(c: Components, refs):
    cond, audio_rows = [], []
    a_mean = mx.array(c.audio_latents_mean)
    a_std = mx.array(c.audio_latents_std)
    for r in refs:
        if r.kind in ("image", "video"):
            if r.kind == "image":
                px = np.asarray(r.image)[None, None]  # (1, 1, H, W, 3)
            else:
                n = r.frames.shape[0]
                n = max(1, (n - R.LATENTS_PER_CHUNK) // R.FRAMES_PER_CHUNK) * R.FRAMES_PER_CHUNK + R.LATENTS_PER_CHUNK
                px = r.frames[:n][None]
            fresh = Noise(ENCODE_SEED)  # a fresh seed-42 generator per condition, as the release does

            def noise_cl(shape_cl, fresh=fresh):  # torch draws (B, C, F, H, W); hand back channels-last
                b, f, h, w, ch = shape_cl
                return mx.array(fresh((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))

            cond.append(encode_condition(c.vae, mx.array(px), PIXEL_MEAN, PIXEL_STD, c.vae_latents_mean,
                                         c.vae_latents_std, noise_cl))
        if r.has_audio:
            wave = mx.array(r.audio[:, :, None])  # (2, samples, 1): stereo channels as two batch items
            lat = c.audio_vae.encode_mean(wave)  # (2, frames, 32)
            audio_rows.append(((lat - a_mean) / a_std).reshape(-1, lat.shape[-1]))
    return cond, audio_rows


def ref2va(c: Components, prompt: str, references, num_frames=124, height=None, width=None, num_inference_steps=50,
           seed=0, on_step=None, debug: dict | None = None):
    """Returns (video float32 (F, H, W, 3) in [0, 1], audio float32 (2, samples), sampling rate). `debug`, if given,
    collects the intermediates under diffusers' state names."""
    dbg = debug if debug is not None else {}
    height, width, num_frames, refs = setup(c, references, height, width, num_frames)
    vision, ic, vc, vts = vision_features(c, refs)
    ids, tags = build_presentation(c.tokenizer, prompt, [Ref(r.kind, r.has_audio) for r in refs], ic, vc, vts)
    text = c.encode_text(ids, vision)
    dbg.update(prompt_embeds=text, text_token_tags=np.asarray(tags), token_ids=ids)

    cond, ref_audio = encode_references(c, refs)
    dbg.update(condition_latents=cond, audio_condition_latents=ref_audio)
    ratio = c.vae.cfg.spatial_ratio
    lf, lh, lw = R.video_latent_num_frames(num_frames), height // ratio, width // ratio
    na = R.audio_latent_num_frames(num_frames)
    layout = build_ref2va_packed_sequence(
        np.asarray(tags), [Ref(r.kind, r.has_audio) for r in refs], [tuple(x.shape[1:4]) for x in cond],
        [int(a.shape[0]) for a in ref_audio], lf, lh, lw, na, tuple(c.dit_cfg["patch_size"]), AUDIO_CHANNELS,
    )

    gen = Noise(seed)
    patch = tuple(c.dit_cfg["patch_size"])
    rows = []
    for x in cond:  # one draw per condition, packed order, then held at t = 0.999
        b, f, h, w, ch = x.shape
        nz = mx.array(gen((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
        t32 = np.float32(NOISE_AUG)  # scale_noise: t * x + (1 - t) * noise, both factors in float32
        noised = mx.array(t32) * x + mx.array(np.float32(1) - t32) * nz
        rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
    lat_ch = c.dit_cfg["in_channels"]
    target = mx.array(gen((1, lat_ch, lf, lh, lw)))
    rows.append(ops.patchify(target, patch))
    video_rows = mx.concatenate(rows)
    audio_ch = c.dit_cfg["audio_in_channels"]
    audio_rows = mx.concatenate(ref_audio + [mx.array(gen((na * AUDIO_CHANNELS, audio_ch)))])

    dbg.update(layout=layout, initial_video_rows=video_rows, initial_audio_rows=audio_rows)
    v, a = denoise(c.dit_params, c.dit_cfg, layout, text, video_rows, audio_rows, Schedules.make(num_inference_steps),
                   on_step=on_step)

    dbg.update(final_video_rows=v, final_audio_rows=a)
    v_lat = ops.unpatchify(v[layout.num_condition_video_rows :], lat_ch, lf, lh, lw, patch)  # (1, C, F, H, W)
    v_lat = v_lat.transpose(0, 2, 3, 4, 1) * mx.array(c.vae_latents_std) + mx.array(c.vae_latents_mean)
    px = c.vae.decode(v_lat)[0]  # (frames, H, W, 3)
    video = mx.clip(px * mx.array(PIXEL_STD) + mx.array(PIXEL_MEAN), 0.0, 1.0)

    a_lat = a[layout.num_condition_audio_rows :].reshape(AUDIO_CHANNELS, na, audio_ch)
    a_lat = a_lat * mx.array(c.audio_latents_std) + mx.array(c.audio_latents_mean)
    audio = c.audio_vae.decode(a_lat)[:, :, 0]  # (2, samples)
    mx.eval(video, audio)
    return np.array(video), np.array(audio), c.audio_sampling_rate
