"""P-H3-Image (Pruna's image model built from MiniMax H3) in MLX, mirroring diffusers' `PH3ImageBlocks`
(`modular_pipelines/p_h3_image/`, diffusers main at d3d8af89) for both of its workflows:

- `text2image`: the prompt as written (no chat template, no special tokens) -> Qwen3-VL-4B hidden states 13, 19, 25,
  31 and the final normed one -> the text connector -> H3's t2va layout for one latent frame and two audio latents ->
  8 Euler steps over the checkpoint's own sigma grid -> the image VAE's single-frame ViT decoder.
- `edit` (`image_conditioned`): up to 8 pictures, the first one edited. Each is resized for the VAE (its own size,
  kept between 1 and 8 MP) and for the text encoder (about 1 MP); the presentation is H3's `"<Picture i>: "` + vision
  block per picture, then `"Edit <Picture 1>. "` + the instruction; the pictures are encoded with H3's conditioning
  recipe and packed as image references (H3's ref2va layout), held at t = 0.999.

The transformer is H3's (`MiniMaxH3Transformer3DModel`, the same weight names), so `ops.dit_forward`, `packing` and
`denoise` run it unchanged. The two audio latents are denoised alongside and discarded. No guidance.
"""

import math
import warnings
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from PIL import Image

from . import load, ops
from .denoise import Schedules, assert_finite, denoise, schedule_calls
from .packing import Ref, build_packed_sequence, build_ref2va_packed_sequence
from .pipeline import ENCODE_SEED, NOISE_AUG, PIXEL_MEAN, PIXEL_STD, Noise
from .presentation import build_presentation
from .vae import VideoVAE, encode_condition

SAMPLE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
HIDDEN_STATE_INDICES = (13, 19, 25, 31, 36)
NUM_AUDIO_LATENTS = 2
AUDIO_CHANNELS = 2
MAX_EDIT_IMAGES = 8
DEFAULT_SIZE = 2912  # 8 MP square, the pipeline's text-to-image default
EDIT_PREFIXES = ("Edit <Picture 1>.", "subject_definitions:")


@dataclass
class Settings:
    """`modular_model_index.json`'s pipeline config (the defaults are the release's)."""

    reference_min_pixels: int = 1024**2
    reference_max_pixels: int = 8 * 1024**2
    canvas_min_pixels: int = 4 * 1024**2
    vision_pixels: int = 1_000_000
    sample_sigmas: tuple = SAMPLE_SIGMAS

    @classmethod
    def from_index(cls, cfg: dict):
        names = cls.__dataclass_fields__
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in names})


@dataclass
class Components:
    tokenizer: object  # transformers Qwen2TokenizerFast
    processor: object  # transformers Qwen3VLProcessor (CPU preprocessing only)
    encode_states: object  # callable(token_ids, vision_inputs) -> list of (1, n, 2560) hidden states
    connector: dict  # text_connector params (float32)
    vae: VideoVAE  # single_frame
    vae_latents_mean: tuple
    vae_latents_std: tuple
    dit_params: dict
    dit_cfg: dict
    settings: Settings = field(default_factory=Settings)
    # the conditioning is cast to the text encoder's dtype, as the pipeline does (`text_encoder.dtype`); None: the dtype
    # its hidden states come out in
    text_dtype: object = None

    @property
    def canvas_multiple(self):
        return self.vae.cfg.spatial_ratio * self.dit_cfg["patch_size"][2]


# -- sizes (before_encoder.py) ---------------------------------------------------------------------------------------


def size_at_area(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
    """`_size_at_area`: (width, height) at about `pixels`, own aspect ratio, sides multiples of `multiple`; a size
    already on the multiple and within 20% of `pixels` is kept."""
    if width % multiple == 0 and height % multiple == 0 and 0.8 * pixels <= width * height <= 1.2 * pixels:
        return width, height
    ratio = width / height
    return (max(multiple, round(math.sqrt(pixels * ratio) / multiple) * multiple),
            max(multiple, round(math.sqrt(pixels / ratio) / multiple) * multiple))


def size_in_area_range(width: int, height: int, min_pixels: int, max_pixels: int, multiple: int) -> tuple[int, int]:
    """`_size_in_area_range`: own size, or scaled to the nearer bound when outside it; sides rounded to `multiple`."""
    area = width * height
    scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
    return tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (width, height))


def round_canvas(height: int, width: int, multiple: int) -> tuple[int, int]:
    rounded = tuple(max(multiple, round(side / multiple) * multiple) for side in (height, width))
    if rounded != (height, width):
        warnings.warn(f"height and width are rounded to multiples of {multiple}: {height}x{width} becomes "
                      f"{rounded[0]}x{rounded[1]}", stacklevel=2)
    return rounded


def text2image_size(height, width, multiple: int) -> tuple[int, int]:
    """`PH3ImagePrepareLayoutStep`: each side defaults to 2912 on its own, then both are rounded to the multiple."""
    return round_canvas(height or DEFAULT_SIZE, width or DEFAULT_SIZE, multiple)


def resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """`VaeImageProcessor.resize` on a PIL image (its default resample, Lanczos); a no-op at the same size."""
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def prepare_edit(c: Components, images, height=None, width=None):
    """`PH3ImageEditResizeStep`: -> (height, width, reference_images for the VAE, vision_images for the encoder)."""
    images = [im.convert("RGB") for im in (images if isinstance(images, (list, tuple)) else [images])]
    if not images:
        raise ValueError("an edit needs at least one image")
    if len(images) > MAX_EDIT_IMAGES:
        raise ValueError(f"an edit takes at most {MAX_EDIT_IMAGES} images, got {len(images)}")
    if (height is None) != (width is None):
        raise ValueError("height and width have to be passed together, or neither of them")
    s, m = c.settings, c.canvas_multiple
    ref_sizes = [size_in_area_range(*im.size, s.reference_min_pixels, s.reference_max_pixels, m) for im in images]
    if height is not None:
        height, width = round_canvas(height, width, m)
    else:
        canvas_min = max(s.canvas_min_pixels, s.reference_min_pixels)
        ref_sizes[0] = size_in_area_range(*images[0].size, canvas_min, s.reference_max_pixels, m)
        width, height = ref_sizes[0]
    vision_sizes = [size_at_area(*im.size, s.vision_pixels, m) for im in images]
    return (height, width, [resize(im, sz) for im, sz in zip(images, ref_sizes)],
            [resize(im, sz) for im, sz in zip(images, vision_sizes)])


# -- text ------------------------------------------------------------------------------------------------------------


def connect(p: dict, states, out_dtype=mx.bfloat16) -> mx.array:
    """`PH3ImageTextConnector`: concatenate the hidden states, standardise, one linear map, destandardise, and put the
    learned sink in place of token 0. In float32 (the connector is kept in float32), cast to `out_dtype` after."""
    f32 = mx.float32
    x = mx.concatenate([s.astype(f32) for s in states], axis=-1)
    x = (x - p["input_mean"].astype(f32)) / p["input_std"].astype(f32)
    x = (x @ p["proj.weight"].astype(f32).T) * p["output_std"].astype(f32) + p["output_mean"].astype(f32)
    sink = mx.broadcast_to(p["sink_embedding"].astype(f32), (x.shape[0], 1, x.shape[-1]))
    return mx.concatenate([sink, x[:, 1:]], axis=1).astype(out_dtype)


def edit_prompt(prompt: str) -> str:
    """The instruction as the edit presentation encodes it: `"Edit <Picture 1>. "` in front, unless it already starts
    with that or is an upsampler brief."""
    return prompt if prompt.lstrip().startswith(EDIT_PREFIXES) else f"Edit <Picture 1>. {prompt}"


def edit_presentation(c: Components, prompt: str, vision_images):
    """`PH3ImageEditTextEncoderStep`'s tokens: (ids, tags, vision inputs)."""
    f = c.processor.image_processor(images=list(vision_images), return_tensors="np")
    merge = c.processor.image_processor.merge_size**2
    counts = [int(np.prod(g)) // merge for g in f["image_grid_thw"]]
    ids, tags = build_presentation(c.tokenizer, edit_prompt(prompt), [Ref("image")] * len(counts), counts, [], [])
    return ids, tags, {"pixel_values": f["pixel_values"], "image_grid_thw": f["image_grid_thw"]}


def text2image_presentation(c: Components, prompt: str):
    ids = c.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return ids, [1] * len(ids), {}


# -- sampling --------------------------------------------------------------------------------------------------------


def schedules(sigmas=None) -> Schedules:
    """Both modalities on the same explicit grid, used verbatim (the scheduler's `shift_provided_sigmas: false`)."""
    s = np.asarray(SAMPLE_SIGMAS if sigmas is None else sigmas, dtype=np.float64).reshape(-1).astype(np.float32)
    if s.size < 2 or not (s[1:] < s[:-1]).all() or s[-1] != 0.0:
        raise ValueError(f"sigmas must be at least two strictly decreasing values ending at 0.0, got {s.tolist()}")
    return Schedules(s, s.copy(), two_time=False)


def encode_pictures(c: Components, reference_images) -> list:
    """`PH3ImageEditVaeEncoderStep`: H3's conditioning recipe per picture, a fresh seed-42 generator each."""
    out = []
    for im in reference_images:
        fresh = Noise(ENCODE_SEED)

        def noise_cl(shape_cl, fresh=fresh):  # torch draws (B, C, F, H, W); hand back channels-last
            b, f, h, w, ch = shape_cl
            return mx.array(fresh((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))

        px = mx.array(np.asarray(im)[None, None])  # (1, 1, H, W, 3) uint8
        out.append(encode_condition(c.vae, px, PIXEL_MEAN, PIXEL_STD, c.vae_latents_mean, c.vae_latents_std, noise_cl))
    return out


def decode(c: Components, video_rows: mx.array, ncv: int, lh: int, lw: int) -> np.ndarray:
    """Final packed rows -> float32 (H, W, 3) in [0, 1]: conditioning rows dropped, latents de-normalised, decoded,
    ImageNet's normalisation undone, clamped."""
    lat = ops.unpatchify(video_rows[ncv:], c.dit_cfg["in_channels"], 1, lh, lw, tuple(c.dit_cfg["patch_size"]))
    lat = lat.transpose(0, 2, 3, 4, 1) * mx.array(c.vae_latents_std) + mx.array(c.vae_latents_mean)
    px = c.vae.decode(lat)[0, 0].astype(mx.float32)
    img = mx.clip(px * mx.array(PIXEL_STD) + mx.array(PIXEL_MEAN), 0.0, 1.0)
    mx.eval(img)
    return np.array(img)


def check_rows(c: Components, video_rows, ncv: int, lh: int, lw: int) -> None:
    """Saved final rows (for a re-decode) must fit the latent grid they claim and be finite."""
    _, ph, pw = c.dit_cfg["patch_size"]
    width = c.dit_cfg["in_channels"] * int(np.prod(c.dit_cfg["patch_size"]))
    rows = np.asarray(video_rows)
    if rows.ndim != 2 or rows.shape[1] != width:
        raise ValueError(f"video rows must be (n, {width}), got {rows.shape}")
    if min(ncv, lh, lw) < 0 or lh % ph or lw % pw or rows.shape[0] - ncv != (lh // ph) * (lw // pw):
        raise ValueError(f"{rows.shape[0]} rows do not hold {ncv} conditioning rows plus a {lh}x{lw} latent image")
    if not np.isfinite(rows).all():
        raise FloatingPointError("non-finite values in the saved rows")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """`VaeImageProcessor.numpy_to_pil`'s rounding."""
    return (image * 255).round().astype(np.uint8)


def generate(c: Components, prompt: str, images=None, height=None, width=None, seed=0, sigmas=None, on_step=None,
             debug: dict | None = None, check_finite: bool = False, on_denoised=None, free_transformer: bool = False,
             on_stage=None, modulation_cache: bool = False, quantize: int | None = None) -> np.ndarray:
    """One P-H3-Image request: text-to-image without `images`, an edit with them. Returns float32 (H, W, 3) in
    [0, 1]. The prompt is rendered as given (run the prompt upsampler first to expand a short request).

    `debug` collects intermediates under diffusers' state names; `on_denoised(video_rows, meta)` runs before decoding;
    `free_transformer` makes the components single-use: the connector is dropped once the prompt is encoded, the VAE
    encoder once the pictures are, and `c.dit_params` before decoding. `on_stage(name)`: text, pictures, layout,
    quantize, denoise, decode.

    `quantize` (8 or 4) quantizes the transformer's attention and feed-forward matrices (`load.quantize_transformer`)
    once the text and pictures are encoded. With the staged text encoder the run script uses (it frees itself), the
    text model and the quantized matrices are then never resident together; an encoder the caller keeps loaded in
    `c.encode_states` stays loaded.

    `modulation_cache` computes every block's AdaLN modulation up front, for each of the run's 8 calls
    (`ops.ModulationCache`), instead of projecting it inside the call: the same tables, bit for bit. With
    `free_transformer` the 13B `adaln_proj` parameters (~26 GB in bf16) are dropped from `c.dit_params` block by block
    as the tables are built; with lazily loaded weights and no other reference to them, at most one block's projection
    is then resident beside the rest of the transformer (which, quantized, is already loaded)."""
    dbg = debug if debug is not None else {}
    stage = on_stage or (lambda name: None)
    m = c.canvas_multiple
    edit = bool(images)
    if edit:
        height, width, ref_images, vision_images = prepare_edit(c, images, height, width)
        ids, tags, vision = edit_presentation(c, prompt, vision_images)
    else:
        height, width = text2image_size(height, width, m)
        ids, tags, vision = text2image_presentation(c, prompt)
    stage("text")
    states = c.encode_states(ids, vision)
    text = connect(c.connector, states, c.text_dtype if c.text_dtype is not None else states[0].dtype)
    if free_transformer:
        mx.eval(text)
        c.connector = {}
    dbg.update(prompt_embeds=text, text_token_tags=np.asarray(tags), token_ids=ids, height=height, width=width)
    if check_finite:
        assert_finite("prompt embeddings", text)

    cond = []
    if edit:
        stage("pictures")
        cond = encode_pictures(c, ref_images)
        if free_transformer:
            mx.eval(cond)
            c.vae.drop_encoder()
            mx.clear_cache()
        dbg.update(condition_latents=cond)
        if check_finite:
            for i, x in enumerate(cond):
                assert_finite(f"picture latents {i}", x)

    stage("layout")
    ratio = c.vae.cfg.spatial_ratio
    lh, lw = height // ratio, width // ratio
    patch = tuple(c.dit_cfg["patch_size"])
    #@@start layout
    if edit:
        layout = build_ref2va_packed_sequence(np.asarray(tags), [Ref("image")] * len(cond),
                                              [tuple(x.shape[1:4]) for x in cond], [], 1, lh, lw, NUM_AUDIO_LATENTS,
                                              patch, AUDIO_CHANNELS)
    else:
        layout = build_packed_sequence(np.asarray(tags), 1, lh, lw, NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)
    #@@end layout

    #@@start noise-draws
    gen = Noise(seed)
    rows = []
    for x in cond:  # one draw per picture, packed order, then held at t = 0.999 (float32, as the pipeline mixes it)
        b, f, h, w, ch = x.shape
        nz = mx.array(gen((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
        t32 = np.float32(NOISE_AUG)
        noised = mx.array(t32) * x + mx.array(np.float32(1) - t32) * nz
        rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
    rows.append(ops.patchify(mx.array(gen((1, c.dit_cfg["in_channels"], 1, lh, lw))), patch))
    video_rows = mx.concatenate(rows)
    audio_rows = mx.array(gen((NUM_AUDIO_LATENTS * AUDIO_CHANNELS, c.dit_cfg["audio_in_channels"])))
    #@@end noise-draws
    dbg.update(layout=layout, initial_video_rows=video_rows, initial_audio_rows=audio_rows)

    if quantize:
        stage("quantize")
        load.quantize_transformer(c.dit_params, quantize)
    stage("denoise")
    sched = schedules(sigmas if sigmas is not None else c.settings.sample_sigmas)
    cache = None
    #@@start modulation-cache
    if modulation_cache:
        cache = ops.ModulationCache(c.dit_params, c.dit_cfg, schedule_calls(layout, sched), drop=free_transformer)
        if free_transformer:
            mx.clear_cache()
        dbg.update(modulation_timesteps=cache.t)
    #@@end modulation-cache
    v, _ = denoise(c.dit_params, c.dit_cfg, layout, text, video_rows, audio_rows, sched, on_step=on_step,
                   check_finite=check_finite, cache=cache)
    dbg.update(final_video_rows=v)
    meta = dict(ncv=layout.num_condition_video_rows, lh=lh, lw=lw)
    if on_denoised:
        on_denoised(v, meta)
    if free_transformer:
        c.dit_params.clear()
        cache = None  # its tables too, before the decoder runs
        mx.clear_cache()
    stage("decode")
    image = decode(c, v, **meta)
    if check_finite:
        assert_finite("decoded image", mx.array(image))
    return image
