"""One P-H3-Image request (text-to-image, or an edit with --image) on the released weights.

    uv run python scripts/run_p_h3_image.py --weights P-H3-Image --prompt-file prompt.txt --out out/img-001
    uv run python scripts/run_p_h3_image.py --weights P-H3-Image --prompt-file edit.txt --image photo.png \\
        --image hat.png --out out/edit-001

The prompt is rendered as written unless `--upsample`: P-H3-Image wants a full, specific prompt (its card: say what
the image is, one scene, quote every word of text, no hype words). An edit instruction gets "Edit <Picture 1>. " in
front unless it starts with it. `--upsample` first runs the release's prompt upsampler (Qwen3.5-9B, ~19 GB, freed
before the image model loads) on the request, as the diffusers pipeline does by default; the prompt it wrote is saved
as `prompt.txt` and logged.

Order: text encoder (Qwen3-VL-4B, ~9 GB, freed after) -> pictures through the VAE encoder (edits) -> 8 transformer
calls (66 GB, read on first use) -> save `final_rows.npz`, free the transformer -> decode -> `image.png`. `run.json`
logs every stage. `--plan-only` reads configs, the tokenizer and the processor only and prints the sizes (with
`--upsample`, of the request before it is expanded). `--decode-from <final_rows.npz>` re-decodes saved latents on their
own: no prompt or sampling options, the rows checked, and the request that made them (saved in the file) logged as the
source.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from h3mlx import image as I  # noqa: E402
from h3mlx import load  # noqa: E402
from h3mlx import upsampler as U  # noqa: E402
from h3mlx.packing import Ref, build_packed_sequence, build_ref2va_packed_sequence  # noqa: E402
from h3mlx.vae import VaeConfig, VideoVAE  # noqa: E402
from run_ref2va import dit_tflop_per_call, mem, provenance, sha256  # noqa: E402


def configs(weights: Path):
    """(transformer config, vae config, pipeline settings): configs only."""
    vcfg = json.loads((weights / "vae" / "config.json").read_text())
    tcfg = {k: v for k, v in json.loads((weights / "transformer" / "config.json").read_text()).items()
            if not k.startswith("_")}
    tcfg["patch_size"] = tuple(tcfg["patch_size"])
    settings = I.Settings.from_index(json.loads((weights / "modular_model_index.json").read_text()))
    return tcfg, vcfg, settings


def config_components(weights: Path, encode_states=None, text: bool = True) -> I.Components:
    """Configs, and the tokenizer and processor unless `text=False`: enough for the plan; no weight file is opened."""
    tcfg, vcfg, settings = configs(weights)
    tokenizer = processor = None
    if text:
        from transformers import AutoTokenizer, Qwen3VLProcessor

        tokenizer = AutoTokenizer.from_pretrained(str(weights / "tokenizer"))
        processor = Qwen3VLProcessor.from_pretrained(str(weights / "processor"))
    return I.Components(
        tokenizer=tokenizer, processor=processor, encode_states=encode_states, connector={},
        vae=VideoVAE({}, VaeConfig.from_diffusers(vcfg)), vae_latents_mean=tuple(vcfg["latents_mean"]),
        vae_latents_std=tuple(vcfg["latents_std"]), dit_params={}, dit_cfg=tcfg, settings=settings)


def upsampler_aspect(height, width) -> str:
    """The aspect label the pipeline hands the upsampler: from the requested size (default 2912 square), before the
    canvas is rounded to multiples of 32, as `PH3ImageText2ImagePromptUpsampleStep` does."""
    return U.aspect_ratio_label(width or I.DEFAULT_SIZE, height or I.DEFAULT_SIZE)


def plan(c: I.Components, prompt: str, images, height, width, sigmas) -> dict:
    m, ratio = c.canvas_multiple, c.vae.cfg.spatial_ratio
    patch = tuple(c.dit_cfg["patch_size"])
    if images:
        height, width, refs, vis = I.prepare_edit(c, images, height, width)
        ids, tags, vision = I.edit_presentation(c, prompt, vis)
        shapes = [(1, r.size[1] // ratio, r.size[0] // ratio) for r in refs]
        layout = build_ref2va_packed_sequence(np.asarray(tags), [Ref("image")] * len(refs), shapes, [], 1,
                                              height // ratio, width // ratio, I.NUM_AUDIO_LATENTS, patch, 2)
        extra = {"picture_sizes": [list(r.size) for r in refs], "vision_sizes": [list(v.size) for v in vis],
                 "vision_tokens": int(sum(t == 0 for t in tags))}
    else:
        height, width = I.text2image_size(height, width, m)
        ids, tags, _ = I.text2image_presentation(c, prompt)
        layout = build_packed_sequence(np.asarray(tags), 1, height // ratio, width // ratio, I.NUM_AUDIO_LATENTS, patch, 2)
        extra = {}
    sch = I.schedules(sigmas if sigmas is not None else c.settings.sample_sigmas)
    return {"workflow": "edit" if images else "text2image", "height": height, "width": width,
            "megapixels": round(height * width / 1e6, 2), "text_rows": len(ids), **extra,
            "picture_rows": layout.num_condition_video_rows, "sequence_rows": int(layout.position_ids.shape[0]),
            "sigmas": [float(s) for s in sch.video_sigmas], "model_calls": int(len(sch.video_timesteps))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True, help="the P-H3-Image release (diffusers layout)")
    ap.add_argument("--prompt-file", type=Path, help="the request (required unless --decode-from)")
    ap.add_argument("--image", type=Path, action="append", default=[],
                    help="a picture to edit; repeat for references (Picture 2, 3, ...; at most 8 in all)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, help="default 2912 (text-to-image), Picture 1's size >= 4 MP (edit)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--seed", type=int, help="default 0")
    ap.add_argument("--sigmas", type=str, help="comma-separated grid ending in 0 (default: the checkpoint's 9 points)")
    ap.add_argument("--vae-dtype", choices=["float32", "bfloat16"], default="float32",
                    help="float32 (as h3mlx decodes H3), or bfloat16 with the pipeline's float32 modules kept")
    ap.add_argument("--upsample", action="store_true", help="expand the request with the release's prompt upsampler")
    ap.add_argument("--modulation-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="compute the AdaLN modulation once for the run's timesteps and drop adaln_proj (~26 GB); "
                         "on by default")
    ap.add_argument("--max-new-tokens", type=int, help="upsampler budget, reasoning included (default 1024, edits 1536)")
    ap.add_argument("--plan-only", action="store_true", help="configs, tokenizer and processor only; no weights")
    ap.add_argument("--decode-from", type=Path,
                    help="decode a saved final_rows.npz (no prompt, pictures, sampling or upsampling options)")
    ap.add_argument("--memory-limit-gb", type=float, default=150.0)
    ap.add_argument("--cache-limit-gb", type=float, default=4.0)
    args = ap.parse_args(argv)
    if args.image and (args.height is None) != (args.width is None):
        ap.error("an edit takes --height and --width together, or neither (text-to-image defaults each to 2912)")
    if args.decode_from:
        given = [n for n, v in (("--prompt-file", args.prompt_file), ("--image", args.image), ("--sigmas", args.sigmas),
                                ("--upsample", args.upsample), ("--height", args.height), ("--width", args.width), ("--plan-only", args.plan_only),
                                ("--seed", args.seed is not None), ("--max-new-tokens", args.max_new_tokens is not None))
                 if v]
        if given:
            ap.error(f"--decode-from decodes the saved rows as they are; drop {', '.join(given)}")
    elif args.prompt_file is None:
        ap.error("--prompt-file is required (unless --decode-from)")
    if args.seed is None and not args.decode_from:
        args.seed = 0
    if len(args.image) > I.MAX_EDIT_IMAGES:
        ap.error(f"at most {I.MAX_EDIT_IMAGES} --image, got {len(args.image)}")
    sigmas = [float(x) for x in args.sigmas.split(",")] if args.sigmas else None
    if sigmas is not None:
        try:
            I.schedules(sigmas)
        except ValueError as e:
            ap.error(f"--sigmas: {e}")

    t0 = time.time()
    mx.set_memory_limit(int(args.memory_limit_gb * 1e9))
    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    weights = args.weights.expanduser()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = {"argv": sys.argv if argv is None else list(argv), "pid": os.getpid(), "weights": str(weights),
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "provenance": provenance(), "status": "running", "events": []}

    def save_log():
        tmp = out / "run.json.tmp"
        tmp.write_text(json.dumps(log, indent=1, default=str, allow_nan=False))
        os.replace(tmp, out / "run.json")

    last = {"t": 0.0}

    def event(name, **kw):
        now = round(time.time() - t0, 1)
        e = {"t": now, "dt": round(now - last["t"], 1), "event": name, **mem(), **kw}
        if name.startswith("stage_") or name in ("text_done", "denoised", "decoded", "written"):
            e["stage_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
            mx.reset_peak_memory()
        last["t"] = now
        log["events"].append(e)
        print(json.dumps(e), flush=True)
        save_log()

    def write(img, prompt_text):
        if not np.isfinite(img).all():
            raise FloatingPointError("non-finite values in the decoded image")
        event("decoded", shape=list(img.shape), mean=float(img.mean()), std=float(img.std()))
        Image.fromarray(I.to_uint8(img)).save(out / "image.png")
        if prompt_text is not None:
            (out / "prompt.txt").write_text(prompt_text + "\n")
        log["status"] = "done"
        event("written", png=str(out / "image.png"))

    vae_dtype = mx.float32 if args.vae_dtype == "float32" else mx.bfloat16
    try:
        if args.decode_from:
            return replay(args.decode_from.expanduser(), weights, vae_dtype, log, event, save_log, write)

        prompt = args.prompt_file.expanduser().read_text().strip()
        paths = [p.expanduser() for p in args.image]
        images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths]
        log.update(prompt_sha256=sha256(args.prompt_file.expanduser()), prompt_chars=len(prompt), seed=args.seed,
                   images=[{"path": str(p), "sha256": sha256(p), "size": list(im.size)} for p, im in zip(paths, images)])
        indices = tuple(json.loads((weights / "text_connector" / "config.json").read_text())["hidden_state_indices"])
        staged = load.staged_text_encoder_states(weights / "text_encoder", indices)

        def encode_states(ids, vision):
            h = staged(ids, vision)
            event("text_done", text_rows=len(ids), states=len(h))
            return h

        c = config_components(weights, encode_states)
        request = prompt
        if args.upsample and not args.plan_only:
            import gc

            event("upsample_start", request_chars=len(prompt))
            log["request"] = prompt
            aspect = None if images else upsampler_aspect(args.height, args.width)
            model, proc = U.load_upsampler(weights / "prompt_upsampler", weights / "prompt_upsampler_processor")
            answer = {}
            try:
                prompt = U.upsample(model, proc, prompt, images=images or None, aspect_ratio=aspect or "1:1",
                                    max_new_tokens=args.max_new_tokens,
                                    on_answer=lambda a, n: answer.update(text=a, tokens=n))
            finally:
                log["upsampler_answer"] = answer.get("text")
                del model, proc
                gc.collect()
                mx.clear_cache()
            log["upsampled_prompt"] = prompt
            event("upsampled", tokens=answer.get("tokens"), prompt_chars=len(prompt), aspect_ratio=aspect)
        log["plan"] = plan(c, prompt, images, args.height, args.width, sigmas)
        if args.upsample and args.plan_only:
            log["plan"]["text_basis"] = ("the request as given: --upsample replaces it with a longer prompt before "
                                         "encoding, so the run's text and sequence rows may differ")
        event("plan", **log["plan"])
        if args.plan_only:
            log["status"] = "planned"
            save_log()
            return 0

        c.vae, _, _ = load.load_vae(weights / "vae", dtype=vae_dtype)
        c.connector = load.load_safetensors(weights / "text_connector")
        c.dit_params, _ = load.load_transformer(weights / "transformer")
        per_call = dit_tflop_per_call(log["plan"]["sequence_rows"], c.dit_cfg)
        clock = {"prev": None}

        def stage(name):
            if name == "denoise":
                clock["denoise_start"] = time.time()
            event(f"stage_{name}")

        def on_step(i, t):
            now = time.time()
            step_s = round(now - (clock["prev"] or clock["denoise_start"]), 2)
            clock["prev"] = now
            event("step", i=i, timestep=float(t), step_s=step_s, tflops=float(f"{per_call / step_s:.3g}"))

        def on_denoised(v, meta):
            # what produced these rows, so a re-decode can say so instead of describing its own command line
            source = {"workflow": log["plan"]["workflow"], "prompt": prompt,
                      "request": request if args.upsample else None, "seed": args.seed,
                      "height": log["plan"]["height"], "width": log["plan"]["width"], "sigmas": log["plan"]["sigmas"],
                      "images": log["images"], "weights": str(weights), "commit": log["provenance"]["commit"]}
            np.savez(out / "final_rows.npz", video_rows=np.array(v.astype(mx.float32)), source=np.array(json.dumps(source)),
                     **meta)
            event("denoised", saved="final_rows.npz", **meta)

        event("start")
        img = I.generate(c, prompt, images=images or None, height=args.height, width=args.width, seed=args.seed,
                         sigmas=sigmas, on_step=on_step, check_finite=True, on_denoised=on_denoised,
                         free_transformer=True, on_stage=stage, modulation_cache=args.modulation_cache)
        write(img, prompt)
        steps = [e for e in log["events"] if e["event"] == "step"]
        log["summary"] = {"total_s": log["events"][-1]["t"],
                          "first_step_s": steps[0]["step_s"] if steps else None,
                          "step_s_mean_after_first": round(sum(s["step_s"] for s in steps[1:]) / max(1, len(steps) - 1), 2)
                          if len(steps) > 1 else None,
                          "peak_rss_gb": max(e["max_rss_gb"] for e in log["events"])}
        save_log()
        print(json.dumps({"summary": log["summary"]}), flush=True)
        return 0
    except KeyboardInterrupt:
        log["status"] = "interrupted"
        event("interrupted")
        raise
    except BaseException as e:
        log["status"] = "error"
        event("error", error=f"{type(e).__name__}: {e}")
        raise


def replay(path: Path, weights: Path, vae_dtype, log, event, save_log, write) -> int:
    """`--decode-from`: decode saved rows. The plan and the prompt written are the saved run's (from the `source`
    record `final_rows.npz` carries), never this command line's; the rows must fit their latent grid and be finite."""
    z = np.load(path)
    missing = sorted({"video_rows", "ncv", "lh", "lw"} - set(z.files))
    if missing:
        raise ValueError(f"{path.name} lacks {missing}")
    ncv, lh, lw = int(z["ncv"]), int(z["lh"]), int(z["lw"])
    c = config_components(weights, text=False)
    rows = z["video_rows"]
    I.check_rows(c, rows, ncv, lh, lw)
    source = json.loads(str(z["source"])) if "source" in z.files else None
    ratio = c.vae.cfg.spatial_ratio
    log["plan"] = {"workflow": "decode", "decode_from": str(path), "decode_from_sha256": sha256(path),
                   "height": lh * ratio, "width": lw * ratio, "source": source}
    event("plan", **{k: v for k, v in log["plan"].items() if k != "source"})
    c.vae, _, _ = load.load_vae(weights / "vae", dtype=vae_dtype)
    event("stage_decode")
    img = I.decode(c, mx.array(rows), ncv, lh, lw)
    write(img, source.get("prompt") if source else None)
    save_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
