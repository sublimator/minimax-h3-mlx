"""Train a LoRA adapter for P-H3-Image on a list of pictures. Nothing has a default that decides how the model is
trained: the shift, the grid, the density, the weighting, the learning rate, the clip, the rank and the alpha are all
required, and a missing one is an error that names it. Arguments can be kept in a file and passed as `@file`.

    uv run python scripts/train_lora.py @run-001.args --plan-only
    uv run python scripts/train_lora.py @run-001.args
    uv run python scripts/train_lora.py @run-001.args --resume

The dataset is a JSON file: `{"examples": [{"picture": "a.png", "prompt": "...", "height": 1024, "width": 1024,
"references": ["b.png"]}, ...]}`, paths relative to the file. `references` are an edit example's input pictures, in
order; without them an example is text-to-image.

Order: read the configs and the dataset -> look each example up in the cache -> `--plan-only` prints the plan and
stops here, having opened no weights -> encode the examples the cache lacks (the VAE and the text encoder, freed
after) -> the transformer -> the loop. `run.json` in the run directory is rewritten after every step; checkpoints go
to `checkpoints/step-NNNNNNNN/`; the finished adapter is `adapter.safetensors`, in diffusers' format.

Exit codes: 0 the run reached `--max-steps`; 3 the wall clock or the memory limit stopped the session (resume to go
on); 2 the arguments, or a `--resume` that does not match the run directory.

This script downloads nothing and reads only the folders it is given.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from h3mlx import cache, files, load, lora, trainer  # noqa: E402
from h3mlx import image as I  # noqa: E402
from h3mlx.vae import VaeConfig  # noqa: E402

EXAMPLE_KEYS = {"picture", "prompt", "height", "width", "references"}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], fromfile_prefix_chars="@", allow_abbrev=False)
    need = dict(required=True)
    where = ap.add_argument_group("where")
    where.add_argument("--weights", type=Path, **need, help="the P-H3-Image release (diffusers layout)")
    where.add_argument("--dataset", type=Path, **need, help="the JSON list of examples")
    where.add_argument("--cache-dir", type=Path, **need, help="encoded examples (h3mlx.cache)")
    where.add_argument("--run-dir", type=Path, **need, help="the run's record, checkpoints and adapter")
    adapter = ap.add_argument_group("the adapter")
    adapter.add_argument("--rank", type=int, **need)
    adapter.add_argument("--alpha", type=float, **need)
    adapter.add_argument("--targets", type=lambda s: [t for t in s.split(",") if t], **need,
                         help=f"comma-separated, from: {', '.join(lora.BLOCK_TARGETS)}")
    adapter.add_argument("--refiner", action=argparse.BooleanOptionalAction, **need,
                         help="also on the token refiner's blocks, or not")
    adapter.add_argument("--assistant", type=Path, help="a frozen adapter kept active while training (optional)")
    hyper = ap.add_argument_group("hyper-parameters")
    hyper.add_argument("--learning-rate", type=float, **need)
    hyper.add_argument("--betas", type=float, nargs=2, metavar=("BETA1", "BETA2"), **need)
    hyper.add_argument("--eps", type=float, **need)
    hyper.add_argument("--weight-decay", type=float, **need)
    hyper.add_argument("--max-grad-norm", type=float, **need, help="the clip on the gradient's global norm")
    hyper.add_argument("--shift", type=float, **need, help="the noise grid's shift")
    hyper.add_argument("--grid", type=int, **need, help="how many noise levels the grid has")
    hyper.add_argument("--density", choices=trainer.DENSITY, **need, help="how a level is drawn from the grid")
    hyper.add_argument("--logit-mean", type=float, help="required by --density logit_normal")
    hyper.add_argument("--logit-std", type=float, help="required by --density logit_normal")
    hyper.add_argument("--mode-scale", type=float, help="required by --density mode")
    hyper.add_argument("--weighting", choices=trainer.WEIGHTING, **need, help="the loss weight of a level")
    hyper.add_argument("--condition-t", type=float, **need, help="the timestep reference pictures are held at")
    hyper.add_argument("--seed", type=int, **need)
    hyper.add_argument("--backward", choices=sorted(trainer.BACKWARD), default="blockwise",
                       help="how the gradient is taken with bounded memory (default: blockwise, the one that "
                            "measured flat in the number of blocks)")
    encode = ap.add_argument_group("how examples are encoded (part of each cache key)")
    encode.add_argument("--latent-seed", type=int, **need, help="seeds each picture's posterior sample, with its digest")
    encode.add_argument("--latent-float16-round", action=argparse.BooleanOptionalAction, **need)
    encode.add_argument("--vae-dtype", choices=["float32", "bfloat16"], **need)
    encode.add_argument("--text-dtype", choices=sorted(cache.DTYPES), **need)
    encode.add_argument("--reference-sizing", choices=cache.SIZING, help="required when an example has references")
    encode.add_argument("--reference-seed", type=int, help="required when an example has references")
    encode.add_argument("--reference-float16-round", action=argparse.BooleanOptionalAction,
                        help="required when an example has references")
    limit = ap.add_argument_group("limits (each stops the run)")
    limit.add_argument("--max-steps", type=int, **need, help="the run is finished at this step")
    limit.add_argument("--max-seconds", type=float, **need, help="wall clock for this session")
    limit.add_argument("--max-peak-memory-gb", type=float, **need, help="against MLX's peak-memory counter")
    limit.add_argument("--checkpoint-every", type=int, **need, help="steps between checkpoints")
    ap.add_argument("--plan-only", action="store_true", help="print what a run would do; open no weights, write nothing")
    ap.add_argument("--resume", action="store_true", help="go on with the run in --run-dir")
    return ap


def read_dataset(path: Path) -> list[dict]:
    """The examples, checked, with their paths made absolute."""
    try:
        listed = json.loads(path.read_text())["examples"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path}: not a dataset file with an \"examples\" list ({type(e).__name__}: {e})") from e
    if not isinstance(listed, list) or not listed:
        raise ValueError(f"{path}: \"examples\" must be a non-empty list")
    out = []
    for i, item in enumerate(listed):
        if not isinstance(item, dict) or not EXAMPLE_KEYS - {"references"} <= set(item) or set(item) - EXAMPLE_KEYS:
            raise ValueError(f"{path}: example {i} must have picture, prompt, height and width, and may have references; "
                             f"got {sorted(item) if isinstance(item, dict) else type(item).__name__}")
        references = item.get("references", [])
        if not isinstance(references, list) or not all(isinstance(r, str) for r in [item["picture"], *references]):
            raise ValueError(f"{path}: example {i}: picture is a path and references a list of paths")
        picture, *references = [(path.parent / p).resolve() for p in [item["picture"], *references]]
        for p in [picture, *references]:
            if not p.is_file():
                raise ValueError(f"{path}: example {i}: no such file {p}")
        out.append({"picture": picture, "references": references, "prompt": item["prompt"], "height": item["height"],
                    "width": item["width"]})
    return out


def release(weights: Path) -> tuple[dict, dict, I.Settings]:
    """(transformer config, VAE config, pipeline settings): three JSON files, no weights."""
    vcfg = json.loads((weights / "vae" / "config.json").read_text())
    tcfg = {k: v for k, v in json.loads((weights / "transformer" / "config.json").read_text()).items()
            if not k.startswith("_")}
    tcfg["patch_size"] = tuple(tcfg["patch_size"])
    return tcfg, vcfg, I.Settings.from_index(json.loads((weights / "modular_model_index.json").read_text()))


def picture_seed(seed: int, digest: str) -> int:
    """A posterior-sample seed of a picture's own, from `--latent-seed` and the picture's digest: two pictures of one
    size do not get the same noise, and a picture's entry does not depend on where it sits in the dataset."""
    return int(np.random.SeedSequence([seed, int(digest[:15], 16)]).generate_state(1, dtype=np.uint64)[0] >> 1)


def load_picture(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def requests(args, examples: list[dict], weights: Path, settings: I.Settings, multiple: int) -> list[dict]:
    """One cache request per example: the files' digests, the choices on the command line and the encoders'
    identities. Reads the pictures' bytes and the release's small files; no weights."""
    encoders = {name: cache.identity(weights / name) for name in cache.ENCODERS}
    numbers = {"reference_min_pixels": settings.reference_min_pixels, "reference_max_pixels": settings.reference_max_pixels,
               "vision_pixels": settings.vision_pixels, "canvas_multiple": multiple}
    out = []
    for example in examples:
        digest = files.sha256_file(example["picture"])
        out.append(cache.request(
            picture=digest, references=[files.sha256_file(r) for r in example["references"]], prompt=example["prompt"],
            height=example["height"], width=example["width"], latent_seed=picture_seed(args.latent_seed, digest),
            latent_float16_round=args.latent_float16_round,
            reference_sizing=args.reference_sizing, reference_seed=args.reference_seed,
            reference_float16_round=args.reference_float16_round, vae_dtype=args.vae_dtype,
            text_dtype=args.text_dtype, settings=numbers, encoders=encoders))
    return out


def describe(examples, asked, cache_dir: Path, cfg: dict, ratio: int, multiple: int) -> list[dict]:
    """Each example for the plan: its key, whether the cache has it, its sizes and its rows. From the pictures'
    headers and the cache's records: no pixels decoded for size, no tensors read."""
    out = []
    for example, request in zip(examples, asked):
        key = cache.key(request)
        record_path = cache_dir / f"{key}.json"
        text_rows = None
        if record_path.is_file():
            text_rows = json.loads(record_path.read_text())["tensors"]["arrays"]["text"]["shape"][1]
        sizes = []
        if request["references"] is not None:
            choice = request["references"]
            for reference in example["references"]:
                with Image.open(reference) as opened:
                    own = ImageOps.exif_transpose(opened).size
                sizes.append(cache.reference_sizes(
                    choice["sizing"], *own, target_height=example["height"], target_width=example["width"],
                    min_pixels=choice["min_pixels"], max_pixels=choice["max_pixels"],
                    vision_pixels=choice["vision_pixels"], multiple=multiple)[0])
        out.append({"picture": example["picture"].name, "height": example["height"], "width": example["width"],
                    "references": [{"file": r.name, "encoded_at": list(s)} for r, s in zip(example["references"], sizes)],
                    "key": key, "cached": record_path.is_file(),
                    **trainer.example_rows(cfg, ratio, example["height"], example["width"], sizes, text_rows)})
    return out


def encode_missing(weights: Path, cache_dir: Path, todo: list, vae_dtype: str, log) -> None:
    """Encode the examples the cache lacks and write their entries: the VAE and the text encoder are loaded once,
    used for every example, and dropped before the transformer is loaded."""
    from run_p_h3_image import config_components

    indices = tuple(json.loads((weights / "text_connector" / "config.json").read_text())["hidden_state_indices"])
    model, encode_states = load.load_text_encoder_states(weights / "text_encoder", indices)
    c = config_components(weights, encode_states)
    c.vae, _, _ = load.load_vae(weights / "vae", dtype=cache.DTYPES[vae_dtype])
    c.connector = load.load_safetensors(weights / "text_connector")
    for example, request in todo:
        began = time.time()
        tensors = cache.encode(c, request, load_picture(example["picture"]), [load_picture(r) for r in example["references"]])
        key = cache.write(cache_dir, request, tensors)
        log(json.dumps({"event": "encoded", "picture": example["picture"].name, "key": key,
                        "text_rows": int(tensors["text"].shape[1]), "seconds": round(time.time() - began, 2)}))
    del model, encode_states, c
    gc.collect()
    mx.clear_cache()


def main(argv=None, clock=time.monotonic) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    needs = {"logit_normal": ("logit_mean", "logit_std"), "mode": ("mode_scale",)}.get(args.density, ())
    missing = [f"--{n.replace('_', '-')}" for n in needs if getattr(args, n) is None]
    if missing:
        ap.error(f"--density {args.density} needs {' and '.join(missing)}")
    if args.plan_only and args.resume:
        ap.error("--plan-only and --resume do not go together")
    weights, cache_dir, run_dir = (p.expanduser().resolve() for p in (args.weights, args.cache_dir, args.run_dir))
    try:
        examples = read_dataset(args.dataset.expanduser().resolve())
        if any(e["references"] for e in examples):
            unset = [f"--{n.replace('_', '-')}" for n in ("reference_sizing", "reference_seed", "reference_float16_round")
                     if getattr(args, n) is None]
            if unset:
                ap.error(f"the dataset has reference pictures: {', '.join(unset)} required (--reference-float16-round "
                         "or --no-reference-float16-round)")
        cfg, vcfg, settings = release(weights)
        ratio = VaeConfig.from_diffusers(vcfg).spatial_ratio
        multiple = ratio * cfg["patch_size"][2]
        asked = requests(args, examples, weights, settings, multiple)
        traj = trainer.trajectory(
            cfg=cfg, transformer=cache.identity(weights / "transformer"), dataset=[cache.key(r) for r in asked],
            rank=args.rank, alpha=args.alpha, targets=args.targets, refiner=args.refiner,
            assistant=files.sha256_file(args.assistant.expanduser()) if args.assistant else None,
            learning_rate=args.learning_rate, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm, shift=args.shift, grid=args.grid, density=args.density,
            logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale, weighting=args.weighting,
            condition_t=args.condition_t, seed=args.seed, backward=args.backward)
        lim = trainer.limits(max_steps=args.max_steps, max_seconds=args.max_seconds,
                             max_peak_memory_gb=args.max_peak_memory_gb, checkpoint_every=args.checkpoint_every)
        headers = [files.safetensors_header(p) for p in load._shards(weights / "transformer")]
        block = next((h["transformer_blocks.0.attn.to_q.weight"]["dtype"] for h in headers
                      if "transformer_blocks.0.attn.to_q.weight" in h), None)
        plan = trainer.plan(traj, lim, cfg, describe(examples, asked, cache_dir, cfg, ratio, multiple), run_dir=run_dir,
                            cache_dir=cache_dir, weights=trainer.weights_summary(headers), block_dtype=block)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    print(json.dumps({"plan": plan}, indent=1), flush=True)
    if args.plan_only:
        return 0

    log = lambda line: print(line, flush=True)  # noqa: E731
    todo = [(e, r) for e, r in zip(examples, asked) if cache.read(cache_dir, r) is None]
    if todo:
        encode_missing(weights, cache_dir, todo, args.vae_dtype, log)
    entries = [cache.read(cache_dir, r) for r in asked]
    assistant = args.assistant.expanduser() if args.assistant else None
    try:
        return trainer.run(run_dir, traj, lim, cfg, entries,
                           load_weights=lambda: load.load_transformer(weights / "transformer")[0],
                           load_assistant=(lambda: lora.load(assistant)) if assistant else None,
                           resume=args.resume, plan=plan, command=sys.argv[1:] if argv is None else [str(a) for a in argv],
                           clock=clock, log=log)
    except trainer.Refused as e:
        ap.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
