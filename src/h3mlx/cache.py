"""Encoded training examples kept on disk: encode once, train many times.

An entry is what `train.build_example` needs for one example: the picture's clean latents, each reference picture's
latents, the prompt's text rows and their tags. It is two files in a cache directory, `<key>.safetensors` and
`<key>.json`, and no pickle anywhere.

The key is the sha256 of the *request*: the inputs' own digests, every choice made in encoding them (sizes, the seed
of each posterior sample, the float16 rounding, the sizing rule for references, the dtypes), the encoders' identities,
and the versions of this package's source and of the libraries the encode runs through. Change any of those and the
key changes, so an entry made another way is not found. The record repeats the whole request, and a reader refuses
an entry whose record is not the request it was asked for, or whose tensors are not the ones the record describes.

The encoders need weights. `load_encoders` loads them and `encode` runs them; everything else here does not.
Everything that turns a request into an entry's numbers is in this package, under the request's code digest: how a
picture file is opened and oriented, how the encoders are loaded, how they are run. A caller supplies the request's
values and nothing else.
"""

import importlib.metadata as metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from . import files
from . import image as I
from . import load as loaders  # this module has a `load` of its own: an entry's

FORMAT = "h3mlx-training-example-1"
SIZING = ("pipeline", "recipe")  # how a reference picture is sized: diffusers' pipeline, or the training recipe
RESIZE = "lanczos"  # the one resampling filter implemented: PIL's, as the pipeline resizes
DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
# the libraries the encode path imports and computes with: mlx-vlm runs the text encoder, torchvision is the
# processor's image backend, torch draws the posterior's noise, Pillow opens and resizes the pictures
PACKAGES = ("mlx", "mlx-vlm", "numpy", "torch", "torchvision", "pillow", "transformers", "tokenizers", "safetensors")
LOCK = "cache.lock"  # the cache directory's lock file: held while an entry is published
ENCODERS = ("vae", "text_encoder", "text_connector", "tokenizer", "processor")
SETTINGS = ("reference_min_pixels", "reference_max_pixels", "vision_pixels", "canvas_multiple")


class CacheMismatch(ValueError):
    """An entry is there, and it is not the one asked for or not whole."""


# --- versions and identities -----------------------------------------------------------------------------------------


def code_digest(package: Path | None = None) -> str:
    """A digest of this package's source: every `.py` file's name and bytes. Any change to the code gives another
    digest, so an entry is never read by code other than the code that wrote it."""
    package = Path(__file__).resolve().parent if package is None else Path(package)
    sources = sorted(p for p in package.rglob("*.py"))
    if not sources:
        raise ValueError(f"no source files in {package}")
    return files.sha256_json([[str(p.relative_to(package)), files.sha256_file(p)] for p in sources])


def versions() -> dict:
    """What the encode's numbers depend on besides the weights: this package's source, and the libraries that draw
    the posterior's noise, resize the pictures, tokenize the prompt and run the arithmetic."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"h3mlx": code_digest(), "packages": packages}


def identity(directory) -> str:
    """A digest that identifies one component of a release without reading its weights: every file's name, with the
    file's sha256, except that a `.safetensors` file is taken by its name and size alone.

    * It notices: a different release, a changed config, tokenizer or processor file, a file added, removed or
      renamed, a weight file of another size.
    * It does not notice: a weight file replaced by a different one of the same name and size (hashing 66 GB at every
      start is the price of noticing that, and it is not paid here).
    * It ignores, on purpose: hidden files, and everything under a hidden directory. A download tool keeps its own
      bookkeeping there (`.cache/huggingface/...` inside a folder it filled), which changes without the component
      changing."""
    directory = Path(directory)
    listed = sorted(p for p in directory.rglob("*")
                    if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts))
    if not listed:
        raise ValueError(f"nothing to identify in {directory}")
    return files.sha256_json([[str(p.relative_to(directory)), p.stat().st_size if p.suffix == ".safetensors"
                               else files.sha256_file(p)] for p in listed])


# --- sizing a reference picture ----------------------------------------------------------------------------------------


def scale_down_to_area(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
    """(width, height) at the picture's own aspect ratio, scaled down to `pixels` when it has more and never scaled
    up, each side rounded to `multiple`. The recipe's rule for a still reference (`ostris/ai-toolkit` at d6a18ad7,
    `extensions_built_in/diffusion_models/minimax_h3/src/packing.py:112-124`)."""
    scale = min(1.0, math.sqrt(pixels / float(width * height)))
    return (max(multiple, round(width * scale / multiple) * multiple),
            max(multiple, round(height * scale / multiple) * multiple))


def reference_sizes(sizing: str, width: int, height: int, *, target_height: int, target_width: int,
                    min_pixels: int, max_pixels: int, vision_pixels: int, multiple: int) -> tuple[tuple, tuple]:
    """((width, height) the VAE encodes a reference at, (width, height) the text encoder sees it at).

    * `"pipeline"`: what diffusers' edit pipeline does at sampling. For the VAE the picture keeps its own size,
      brought to the nearer bound when its area is outside [`min_pixels`, `max_pixels`]; the text encoder sees it at
      about `vision_pixels` (`image.size_in_area_range`, `image.size_at_area`). The target's size plays no part.
    * `"recipe"`: what the training recipe does. For the VAE the picture is scaled down to the *target's* pixel area
      when it has more, and never up; the text encoder sees it scaled down to `vision_pixels` (in whole thousands,
      as the recipe writes it) when it has more, and never up (`p_h3_image.py:304-311` of the same commit).

    The two differ whenever a reference is larger than the target, or smaller than the pipeline's lower bound."""
    if sizing == "pipeline":
        return (I.size_in_area_range(width, height, min_pixels, max_pixels, multiple),
                I.size_at_area(width, height, vision_pixels, multiple))
    if sizing == "recipe":
        return (scale_down_to_area(width, height, target_height * target_width, multiple),
                scale_down_to_area(width, height, 1000 * (vision_pixels // 1000), multiple))
    raise ValueError(f"reference sizing is one of {SIZING}, got {sizing!r}")


# --- the request, and its key ------------------------------------------------------------------------------------------


def _digest(what: str, value) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise ValueError(f"{what} must be a sha256 in hex, got {value!r}")
    return value


def _seed(what: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError(f"{what} must be an integer seed in [0, 2**63), got {value!r}")
    return value


def _flag(what: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{what} must be True or False, got {value!r}")
    return value


def request(*, picture: str, references, prompt: str, height: int, width: int, latent_seed: int,
            latent_float16_round: bool, reference_sizing: str, reference_seed: int, reference_float16_round: bool,
            vae_dtype: str, text_dtype: str, settings: dict, encoders: dict) -> dict:
    """What one entry is asked to be. Every argument is required and none has a default.

    `picture` and each of `references` are sha256 digests of the input files; `prompt` is the text as given.
    `height` and `width` are the size the picture is trained at. `latent_seed` and `latent_float16_round` are the
    seed of the picture's posterior sample and whether it is rounded through float16 before normalising; the
    `reference_...` pair is the same for the reference pictures, and `reference_sizing` is one of `SIZING`.
    `settings` holds the release's `reference_min_pixels`, `reference_max_pixels` and `vision_pixels` and the
    `canvas_multiple`; `encoders` holds an identity (`identity`) for each of `ENCODERS`.

    An example without references does not depend on the reference choices: its request leaves them out, and they
    may be given as None."""
    references = [_digest(f"references[{i}]", r) for i, r in enumerate(references)]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("the prompt must be a string with something in it")
    if set(settings) != set(SETTINGS) or not all(isinstance(settings[k], int) and settings[k] > 0 for k in SETTINGS):
        raise ValueError(f"settings must hold positive integers for exactly {SETTINGS}, got {settings!r}")
    multiple = settings["canvas_multiple"]
    for what, side in (("height", height), ("width", width)):
        if isinstance(side, bool) or not isinstance(side, int) or side <= 0 or side % multiple:
            raise ValueError(f"{what} must be a positive multiple of {multiple}, got {side!r}")
    for what, dtype in (("vae_dtype", vae_dtype), ("text_dtype", text_dtype)):
        if dtype not in DTYPES:
            raise ValueError(f"{what} is one of {sorted(DTYPES)}, got {dtype!r}")
    if set(encoders) != set(ENCODERS) or not all(isinstance(v, str) and v for v in encoders.values()):
        raise ValueError(f"encoders must name an identity for exactly {ENCODERS}, got {sorted(encoders)}")
    reference = None
    if references:
        if reference_sizing not in SIZING:
            raise ValueError(f"reference sizing is one of {SIZING}, got {reference_sizing!r}")
        _seed("reference_seed", reference_seed)
        _flag("reference_float16_round", reference_float16_round)
        reference = {"sizing": reference_sizing, "seed": reference_seed, "float16_round": reference_float16_round,
                     "resize": RESIZE, "min_pixels": settings["reference_min_pixels"],
                     "max_pixels": settings["reference_max_pixels"], "vision_pixels": settings["vision_pixels"]}
    return {
        "format": FORMAT,
        "inputs": {"picture": _digest("picture", picture), "references": references, "prompt": prompt},
        "target": {"height": height, "width": width, "seed": _seed("latent_seed", latent_seed),
                   "float16_round": _flag("latent_float16_round", latent_float16_round), "resize": RESIZE},
        "references": reference,
        "text": {"dtype": text_dtype, "workflow": "edit" if references else "text2image"},
        "vae": {"dtype": vae_dtype, "canvas_multiple": multiple},
        "encoders": {k: encoders[k] for k in ENCODERS},
        "versions": versions(),
    }


def key(request: dict) -> str:
    return files.sha256_json(request)


def differences(asked, found, path="") -> list[str]:
    """Where two JSON values differ, as dotted paths with both values."""
    if isinstance(asked, dict) and isinstance(found, dict):
        out = []
        for k in sorted(set(asked) | set(found)):
            where = f"{path}.{k}" if path else str(k)
            if k not in asked or k not in found:
                out.append(f"{where}: {'missing from the entry' if k in asked else 'not asked for'}")
            else:
                out += differences(asked[k], found[k], where)
        return out
    return [] if asked == found and type(asked) is type(found) else [f"{path}: asked {asked!r}, the entry has {found!r}"]


# --- entries -----------------------------------------------------------------------------------------------------------


@dataclass
class Entry:
    """One example as read from the cache: `train.build_example`'s inputs."""

    latents: mx.array  # (1, 1, h, w, channels) float32, normalised
    condition_latents: list  # one (1, 1, h_i, w_i, channels) float32 per reference, in order
    text: mx.array  # (1, text rows, text dim), in the request's text dtype
    text_token_tags: np.ndarray  # (text rows,) int64: 1 text, 0 a vision token
    record: dict


def _names(request: dict) -> list[str]:
    count = len(request["inputs"]["references"])
    return ["latents", "text", "text_token_tags"] + [f"condition_latents.{i}" for i in range(count)]


def _described(arrays: dict) -> dict:
    return {name: {"dtype": str(a.dtype).rsplit(".", 1)[-1], "shape": list(a.shape)} for name, a in sorted(arrays.items())}


def check_tensors(request: dict, tensors: dict) -> dict:
    """The arrays of an entry for this request, as `mx.array`s, or a ValueError: exactly the names the request has,
    latents as (1, 1, h, w, channels) float32 with the picture's in the stated height-to-width ratio, text rows as
    (1, n, dim) in the stated dtype, one tag of 0 or 1 per text row, and nothing that is not finite."""
    names = _names(request)
    if sorted(tensors) != sorted(names):
        raise ValueError(f"an entry for this request has exactly {sorted(names)}, got {sorted(tensors)}")
    arrays = {k: (v if isinstance(v, mx.array) else mx.array(np.asarray(v))) for k, v in tensors.items()}
    for name in names:
        if not name.startswith(("latents", "condition_latents")):
            continue
        a = arrays[name]
        if a.ndim != 5 or a.shape[:2] != (1, 1) or 0 in a.shape or a.dtype != mx.float32:
            raise ValueError(f"{name} must be float32 (1, 1, height, width, channels), got {a.dtype} {tuple(a.shape)}")
    latents, target = arrays["latents"], request["target"]
    if latents.shape[2] * target["width"] != latents.shape[3] * target["height"]:
        raise ValueError(f"latents of {latents.shape[2]}x{latents.shape[3]} (height x width) are not a picture of "
                         f"{target['height']}x{target['width']}")
    text, tags = arrays["text"], arrays["text_token_tags"]
    if text.ndim != 3 or text.shape[0] != 1 or 0 in text.shape or text.dtype != DTYPES[request["text"]["dtype"]]:
        raise ValueError(f"text must be {request['text']['dtype']} (1, rows, dim), got {text.dtype} {tuple(text.shape)}")
    if tags.dtype != mx.int64 or tuple(tags.shape) != (text.shape[1],) or not np.isin(np.array(tags), (0, 1)).all():
        raise ValueError(f"text_token_tags must be int64 ({text.shape[1]},) of 0s and 1s, got {tags.dtype} {tuple(tags.shape)}")
    for name in names:
        if name != "text_token_tags" and not bool(mx.all(mx.isfinite(arrays[name].astype(mx.float32)))):
            raise ValueError(f"{name} holds values that are not finite")
    return arrays


def write(directory, request: dict, tensors: dict, wait: float = 30.0) -> str:
    """Write one entry and return its key.

    The tensors are written under a temporary name first. Then, holding the cache directory's lock (`files.exclusive`
    on `cache.lock`, waiting up to `wait` seconds for it), the entry is published: the tensors renamed into place,
    then the record written, atomically, last. So an entry is there only once it is whole, and one writer publishes
    at a time.

    * An entry is written once. A key that already has a record is refused (`FileExistsError`), before the lock and
      again under it, in case another writer published meanwhile.
    * A writer that fails removes what it alone put there, and nothing else: its temporary file, and the tensors it
      renamed into place under the lock, where no record existed and no other writer could be. An entry that has
      its record is never touched.
    * When the lock stays held for `wait` seconds the write is refused with `files.Busy`, which says who holds it."""
    directory = Path(directory)
    arrays = check_tensors(request, tensors)
    name = key(request)
    directory.mkdir(parents=True, exist_ok=True)
    record_path, tensors_path = directory / f"{name}.json", directory / f"{name}.safetensors"
    if os.path.lexists(record_path):
        raise FileExistsError(f"the cache already has an entry {name}")
    staged = files.temporary(tensors_path)
    try:
        mx.save_safetensors(str(staged), arrays)
        files.sync_file(staged)
        record = {"format": FORMAT, "key": name, "request": request,
                  "tensors": {"file": tensors_path.name, "sha256": files.sha256_file(staged),
                              "arrays": _described(arrays)}}
        with files.exclusive(directory / LOCK, f"publishing cache entry {name[:12]}", wait=wait):
            if os.path.lexists(record_path):
                raise FileExistsError(f"the cache already has an entry {name}")
            os.replace(staged, tensors_path)  # over nothing, or over tensors a dead writer left without a record
            try:
                files.atomic_json(record_path, record)
            except BaseException:
                tensors_path.unlink(missing_ok=True)
                raise
    finally:
        staged.unlink(missing_ok=True)
    return name


def load(record_path, request: dict) -> Entry:
    """The entry whose record is at `record_path`, if it is the one `request` asks for and whole; otherwise a
    `CacheMismatch` that says what differs."""
    record_path = Path(record_path)
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError) as e:
        raise CacheMismatch(f"{record_path.name}: the record cannot be read ({e})") from e
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise CacheMismatch(f"{record_path.name}: not a {FORMAT} record")
    apart = differences(request, record.get("request"))
    if apart:
        raise CacheMismatch(f"{record_path.name} is not the entry asked for: " + "; ".join(apart[:6])
                            + (f"; and {len(apart) - 6} more" if len(apart) > 6 else ""))
    described = record.get("tensors") or {}
    if record.get("key") != key(request) or described.get("file") != f"{key(request)}.safetensors":
        raise CacheMismatch(f"{record_path.name}: the record's key is not its request's")
    tensors_path = record_path.with_name(described["file"])
    if not tensors_path.is_file():
        raise CacheMismatch(f"{record_path.name}: its tensors, {tensors_path.name}, are missing")
    if files.sha256_file(tensors_path) != described.get("sha256"):
        raise CacheMismatch(f"{tensors_path.name} is not the file its record describes (sha256 differs)")
    try:
        arrays = mx.load(str(tensors_path), format="safetensors")
    except Exception as e:  # MLX raises its own types for a file that is not safetensors
        raise CacheMismatch(f"{tensors_path.name} cannot be read as safetensors ({e})") from e
    if _described(arrays) != described.get("arrays") or sorted(arrays) != sorted(_names(request)):
        raise CacheMismatch(f"{tensors_path.name}: its arrays are not the ones its record lists")
    count = len(request["inputs"]["references"])
    return Entry(latents=arrays["latents"], condition_latents=[arrays[f"condition_latents.{i}"] for i in range(count)],
                 text=arrays["text"], text_token_tags=np.array(arrays["text_token_tags"]), record=record)


def read(directory, request: dict) -> Entry | None:
    """The entry for `request` in `directory`, or None when there is none. An entry that is there under the
    request's key and is not what was asked, or not whole, is an error (`CacheMismatch`), never a miss."""
    record_path = Path(directory) / f"{key(request)}.json"
    if not os.path.lexists(record_path):
        return None
    return load(record_path, request)


# --- encoding (needs weights) ------------------------------------------------------------------------------------------


def oriented(picture):
    """The picture as RGB with its EXIF orientation applied. Every picture an entry is made from goes through this,
    whoever opened the file, so a photograph that says "rotate me" is encoded the way it is looked at, and once
    applied the orientation is no longer pending: doing it twice changes nothing."""
    return ImageOps.exif_transpose(picture).convert("RGB")


def load_picture(path):
    """A picture file as `encode` takes it: opened, oriented, RGB."""
    with Image.open(path) as opened:
        return oriented(opened)


def picture_size(path) -> tuple[int, int]:
    """(width, height) of a picture file as it will be encoded, that is after its EXIF orientation."""
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).size


def release_configs(weights) -> tuple[dict, dict, I.Settings]:
    """(transformer config, VAE config, pipeline settings) of a release folder: three JSON files, no weights."""
    weights = Path(weights)
    vae = json.loads((weights / "vae" / "config.json").read_text())
    transformer = {k: v for k, v in json.loads((weights / "transformer" / "config.json").read_text()).items()
                   if not k.startswith("_")}
    transformer["patch_size"] = tuple(transformer["patch_size"])
    return transformer, vae, I.Settings.from_index(json.loads((weights / "modular_model_index.json").read_text()))


def load_encoders(weights, vae_dtype: str):
    """`image.Components` for `encode`, from a release folder: the VAE (loaded with `vae_dtype`'s policy), the text
    encoder as far as the connector's hidden states need, the connector, the tokenizer and the processor. This is
    the one place the encoders are loaded for the cache, so how they are loaded is part of the code digest. It needs
    the weights; the tokenizer and the processor are read from their local folders."""
    from transformers import AutoTokenizer, Qwen3VLProcessor

    weights = Path(weights)
    transformer, _, settings = release_configs(weights)
    indices = tuple(json.loads((weights / "text_connector" / "config.json").read_text())["hidden_state_indices"])
    _, encode_states = loaders.load_text_encoder_states(weights / "text_encoder", indices)
    vae, mean, std = loaders.load_vae(weights / "vae", dtype=DTYPES[vae_dtype])
    return I.Components(
        tokenizer=AutoTokenizer.from_pretrained(str(weights / "tokenizer")),
        processor=Qwen3VLProcessor.from_pretrained(str(weights / "processor")), encode_states=encode_states,
        connector=loaders.load_safetensors(weights / "text_connector"), vae=vae, vae_latents_mean=mean,
        vae_latents_std=std, dit_params={}, dit_cfg=transformer, settings=settings)


def target_picture(picture, height: int, width: int, multiple: int):
    """The picture, oriented and RGB (`oriented`), at (width, height), resized with Lanczos when it is another size.
    Its aspect ratio must be the stated one to within one `multiple` a side: a picture of another shape is refused,
    not stretched or cropped."""
    picture = oriented(picture)
    own_width, own_height = picture.size
    if abs(own_width * height / own_height - width) > multiple or abs(own_height * width / own_width - height) > multiple:
        raise ValueError(f"a {own_width}x{own_height} picture does not have the shape of the stated {width}x{height} "
                         "(width x height): crop it first")
    return I.resize(picture, (width, height))


def encode(components, request: dict, picture, references) -> dict:
    """The tensors of the entry `request` asks for, from the pictures themselves (PIL images, the ones whose digests
    the request names; each is oriented by its EXIF tag here, `oriented`, so it does not matter whether the caller
    did) and `components` (`image.Components`: the VAE, the text encoder's `encode_states`, the connector, the
    tokenizer and the processor, loaded as the request says: `load_encoders`).

    * The picture: resized to the stated size (`target_picture`), encoded, one posterior sample from a generator
      seeded with the request's seed, rounded through float16 or not, normalised.
    * Each reference: sized by the request's rule (`reference_sizes`), encoded the same way with the references'
      seed and rounding.
    * The text: the prompt as written for text-to-image; for an edit, the pipeline's presentation (a vision block per
      reference, then the instruction with its "Edit <Picture 1>. " prefix), through the text encoder and the
      connector, cast to the request's text dtype."""
    references = [oriented(r) for r in references]
    if len(references) != len(request["inputs"]["references"]):
        raise ValueError(f"the request names {len(request['inputs']['references'])} references, got {len(references)}")
    if len(references) > I.MAX_EDIT_IMAGES:
        raise ValueError(f"an edit takes at most {I.MAX_EDIT_IMAGES} pictures, got {len(references)}")
    multiple, target = request["vae"]["canvas_multiple"], request["target"]
    if components.canvas_multiple != multiple:
        raise ValueError(f"the request was made for a canvas multiple of {multiple}; these components have "
                         f"{components.canvas_multiple}")
    resized = target_picture(picture, target["height"], target["width"], multiple)
    out = {"latents": I.encode_pictures(components, [resized], float16_round=target["float16_round"],
                                        seed=target["seed"])[0]}
    prompt = request["inputs"]["prompt"]
    if references:
        choice = request["references"]
        sizes = [reference_sizes(choice["sizing"], *r.size, target_height=target["height"], target_width=target["width"],
                                 min_pixels=choice["min_pixels"], max_pixels=choice["max_pixels"],
                                 vision_pixels=choice["vision_pixels"], multiple=multiple) for r in references]
        latents = I.encode_pictures(components, [I.resize(r, vae_size) for r, (vae_size, _) in zip(references, sizes)],
                                    float16_round=choice["float16_round"], seed=choice["seed"])
        out.update({f"condition_latents.{i}": x for i, x in enumerate(latents)})
        ids, tags, vision = I.edit_presentation(components, prompt, [I.resize(r, vision_size)
                                                                     for r, (_, vision_size) in zip(references, sizes)])
    else:
        ids, tags, vision = I.text2image_presentation(components, prompt)
    states = components.encode_states(ids, vision)
    out["text"] = I.connect(components.connector, states, DTYPES[request["text"]["dtype"]])
    out["text_token_tags"] = mx.array(np.asarray(tags, dtype=np.int64))
    return out
