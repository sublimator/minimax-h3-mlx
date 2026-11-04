"""One MiniMax H3 ref2va request on the released weights, staged so the big models are never needed at once.

    uv run python scripts/run_ref2va.py --weights MiniMax-H3 --image him.png --prompt-file prompt.txt \\
        --height 480 --width 864 --frames 124 --steps 50 --seed 0 --out out/run-001

Order (so the 51 GB text encoder and the 66 GB transformer are not evaluated together):
  0. check the writer (ffmpeg, ffprobe) with a tiny clip, before any model loads;
  1. normalise the references, build the Qwen3-VL presentation (CPU);
  2. load the text encoder, encode, free it (`load.staged_text_encoder`);
  3. encode the references with the video / audio VAEs;
  4. denoise on `transformer_ref` (loaded lazily: its weights are read on first use, after step 2);
  5. save the final latents (`final_rows.npz`), free the transformer, decode, write `video.mp4` (with sound),
     `audio.wav`; `run.json` logs every stage.

`--plan-only` reads only configs, the tokenizer and the processor (no weight files) and prints the sizes the run would
use. `--decode-from <final_rows.npz>` skips steps 1-4 and decodes saved latents. Writes only under --out.
`--hyperflow <file.safetensors>` runs Video Rebirth's 8-step LoRA (its own sigma grid: 8 calls; `--steps` refused).
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3mlx import hyperflow, load, references as R  # noqa: E402
from h3mlx.denoise import Schedules  # noqa: E402
from h3mlx.packing import Ref, build_ref2va_packed_sequence  # noqa: E402
from h3mlx.pipeline import Components, ImageRef, decode, ref2va, setup, vision_features  # noqa: E402
from h3mlx.presentation import build_presentation  # noqa: E402
from h3mlx.vae import VaeConfig, VideoVAE  # noqa: E402


def current_rss_gb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()
    return round(int(out or 0) * 1024 / 1e9, 2)


def pressure_level() -> int | None:
    """macOS kernel memory pressure: 1 normal, 2 warning, 4 critical (None elsewhere)."""
    p = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True)
    return int(p.stdout.strip()) if p.returncode == 0 and p.stdout.strip().isdigit() else None


def mem() -> dict:
    return {"mlx_active_gb": round(mx.get_active_memory() / 1e9, 2), "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 2),
            "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "rss_gb": current_rss_gb(),
            "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
            "pressure": pressure_level()}


def dit_tflop_per_call(rows: int, cfg: dict) -> float:
    """Matmul FLOPs of one transformer forward: projections + SwiGLU (2*S*params) and dense attention (4*S^2*inner),
    times the block count; refiner, AdaLN and norms are left out (small)."""
    h, inner, ffn = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["ffn_dim"]
    per_block = 2 * rows * (4 * h * inner + 3 * h * ffn) + 4 * rows * rows * inner
    return per_block * cfg["num_layers"] / 1e12


def lora_tflop_per_call(rows: int, cfg: dict, rank: int) -> float:
    """HyperFlow's unmerged LoRA on q, k, v, out, ff in, ff out: 2 * S * rank * (in + out) each, per block."""
    h, inner, ffn = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["ffn_dim"]
    per_block = 2 * rows * rank * (4 * (h + inner) + (h + 2 * ffn) + (ffn + h))
    return per_block * cfg["num_layers"] / 1e12


def summarize(events: list) -> dict:
    """Stage durations and peaks from the event log (seconds, GB)."""
    t = {}
    for e in events:
        t.setdefault(e["event"], e["t"])
    steps = [e for e in events if e["event"] == "step"]
    out = {"total_s": events[-1]["t"] if events else None}
    for name, a, b in (("text_s", "text_start", "text_done"), ("references_s", "stage_references", "stage_layout"),
                       ("decode_s", "stage_decode", "decoded"), ("write_s", "decoded", "written")):
        if a in t and b in t:
            out[name] = round(t[b] - t[a], 1)
    if steps and "stage_denoise" in t:
        out["denoise_s"] = round(steps[-1]["t"] - t["stage_denoise"], 1)
        out["first_step_s"] = steps[0]["step_s"]  # includes reading the transformer's weights on first use
        rest = [s["step_s"] for s in steps[1:]]
        if rest:
            out["step_s_mean"] = round(sum(rest) / len(rest), 2)
            out["step_s_min"], out["step_s_max"] = min(rest), max(rest)
            out["tflops_mean"] = round(sum(s["tflops"] for s in steps[1:]) / len(rest), 1)
    out["stage_peak_gb"] = {e["event"]: e["stage_peak_gb"] for e in events if e.get("closes_stage")}
    out["process_peak_rss_gb"] = max((e["max_rss_gb"] for e in events), default=None)
    out["worst_pressure"] = max((e["pressure"] or 0 for e in events), default=None)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:  # in 16 MB chunks: the HyperFlow file is 2.8 GB
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    pkgs = {}
    for name in ("mlx", "mlx-vlm", "torch", "transformers", "numpy", "pillow"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {"commit": commit, "dirty": dirty, "python": sys.version.split()[0], "packages": pkgs}


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0).T * 32767.0).round().astype("<i2")  # (samples, channels)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def write_mp4(path: Path, video: np.ndarray, wav: Path, fps: int = 24) -> None:
    frames = (np.clip(video, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    f, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-crf", "16",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def check_writer(out: Path) -> dict:
    """A 24-frame 64x64 clip with one second of silence through the same writer, then ffprobe. Fails fast."""
    probe_dir = out / "_writer_check"
    probe_dir.mkdir(parents=True, exist_ok=True)
    write_wav(probe_dir / "a.wav", np.zeros((2, 32000), np.float32), 32000)
    write_mp4(probe_dir / "v.mp4", np.full((24, 64, 64, 3), 0.5, np.float32), probe_dir / "a.wav")
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,nb_frames", "-of",
                        "json", str(probe_dir / "v.mp4")], capture_output=True, text=True, check=True)
    streams = {s["codec_type"]: s for s in json.loads(p.stdout)["streams"]}
    if int(streams["video"]["nb_frames"]) != 24 or "audio" not in streams:
        raise RuntimeError(f"writer check failed: {streams}")
    return {k: v.get("codec_name") for k, v in streams.items()}


def config_components(args, weights: Path, encode_text=None) -> Components:
    """Tokenizer, processor and configs only: enough for the plan; no weight file is opened."""
    from transformers import AutoTokenizer, Qwen3VLProcessor

    vcfg = json.loads((weights / "vae" / "config.json").read_text())
    acfg = json.loads((weights / "audio_vae" / "config.json").read_text())
    tcfg = {k: v for k, v in json.loads((weights / "transformer_ref" / "config.json").read_text()).items()
            if not k.startswith("_")}
    tcfg["patch_size"] = tuple(tcfg["patch_size"])
    c = Components(tokenizer=AutoTokenizer.from_pretrained(str(weights / "tokenizer")),
                   processor=Qwen3VLProcessor.from_pretrained(str(weights / "processor")), encode_text=encode_text,
                   vae=VideoVAE({}, VaeConfig.from_diffusers(vcfg)), vae_latents_mean=tuple(vcfg["latents_mean"]),
                   vae_latents_std=tuple(vcfg["latents_std"]), audio_vae=None,
                   audio_latents_mean=tuple(acfg["latents_mean"]), audio_latents_std=tuple(acfg["latents_std"]),
                   dit_params={}, dit_cfg=tcfg)
    if args.ref_short_edge:
        c.reference_image_short_edge = args.ref_short_edge
    if args.canvas_short_edge:
        c.canvas_short_edge = args.canvas_short_edge
    return c


def load_weights(c: Components, weights: Path, hyperflow_path: Path | None = None):
    """Attach the weights lazily (mx.load reads data on first use). The audio VAE is small and loaded eagerly. With
    `hyperflow_path`, the LoRA is attached to transformer_ref's params (2.8 GB more, read on first use too)."""
    c.vae, _, _ = load.load_vae(weights / "vae")
    c.audio_vae, _, _ = load.load_audio_vae(weights / "audio_vae")
    c.dit_params, _ = load.load_transformer(weights / "transformer_ref")
    if hyperflow_path is not None:
        c.dit_params, c.dit_cfg, _ = hyperflow.load(hyperflow_path, c.dit_params, c.dit_cfg, "transformer_ref")


def plan(c: Components, prompt: str, image: Image.Image, args) -> dict:
    """Everything up to the layout, without the models: sizes the run would use."""
    h, w, n, refs = setup(c, [ImageRef(image)], args.height, args.width, args.frames)
    vision, ic, vc, vts = vision_features(c, refs)
    ids, tags = build_presentation(c.tokenizer, prompt, [Ref(r.kind, r.has_audio) for r in refs], ic, vc, vts)
    ratio = c.vae.cfg.spatial_ratio
    ref_shapes = [(1, r.image.size[1] // ratio, r.image.size[0] // ratio) for r in refs]
    lf, lh, lw = R.video_latent_num_frames(n), h // ratio, w // ratio
    na = R.audio_latent_num_frames(n)
    layout = build_ref2va_packed_sequence(np.asarray(tags), [Ref(r.kind, r.has_audio) for r in refs], ref_shapes, [],
                                          lf, lh, lw, na, tuple(c.dit_cfg["patch_size"]), 2)
    if args.hyperflow:
        meta, shapes = hyperflow.read_header(args.hyperflow)
        sch = meta.schedules()
    else:
        sch = Schedules.make(args.steps)
    extra = {}
    if args.hyperflow:
        rank = hyperflow.check_file(meta, shapes, c.dit_cfg, "transformer_ref")  # before the hash and any weights
        extra["hyperflow"] = {"file": str(args.hyperflow), "version": meta.version, "gate": meta.gate,
                              "rank": rank, "sigmas": list(meta.sigmas),
                              "base_model_revision": meta.base_model_revision}
    return {**extra, "height": h, "width": w, "frames": n, "seconds": round(n / R.FPS, 3), "latent_frames": lf,
            "reference_size": list(refs[0].image.size), "text_rows": len(ids), "vision_tokens": int(sum(ic)),
            "reference_video_rows": layout.num_condition_video_rows, "target_audio_rows": na * 2,
            "sequence_rows": int(layout.position_ids.shape[0]), "grid_points": int(len(sch.video_sigmas)),
            "model_calls": int(len(sch.video_timesteps))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True, help="the release's root diffusers layout")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--steps", type=int, help="sigma grid points incl. the final 0 (N-1 model calls); default 50")
    ap.add_argument("--hyperflow", type=Path, help="HyperFlow LoRA .safetensors: its 9-point grid, 8 model calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="configs, tokenizer and processor only; no weights")
    ap.add_argument("--check-headers", action="store_true",
                    help="read only the weight files' headers and check names, shapes and dtypes against the code")
    ap.add_argument("--decode-from", type=Path, help="decode a saved final_rows.npz instead of running the model")
    ap.add_argument("--memory-limit-gb", type=float, default=110.0,
                    help="MLX memory limit (a guideline: allocations wait and drop cache before failing)")
    ap.add_argument("--cache-limit-gb", type=float, default=4.0, help="cap on memory MLX keeps cached after use")
    ap.add_argument("--ref-short-edge", type=int, default=0, help="override 2048 (tests only)")
    ap.add_argument("--canvas-short-edge", type=int, default=0, help="override 768 (tests only)")
    args = ap.parse_args(argv)
    if args.hyperflow is not None:
        if args.steps is not None:
            ap.error("--hyperflow runs the grid stored in its file; drop --steps")
        args.hyperflow = args.hyperflow.expanduser()
        hyperflow.read_header(args.hyperflow)  # fail now, not after the text encoder, if it is not a HyperFlow file
    elif args.steps is None:
        args.steps = 50

    t0 = time.time()
    previous_limits = {"memory_limit_gb": round(mx.set_memory_limit(int(args.memory_limit_gb * 1e9)) / 1e9, 2),
                       "cache_limit_gb": round(mx.set_cache_limit(int(args.cache_limit_gb * 1e9)) / 1e9, 2)}
    weights = args.weights.expanduser()
    prompt_path, image_path = args.prompt_file.expanduser(), args.image.expanduser()
    prompt = prompt_path.read_text().strip()
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")  # as diffusers' load_image: phone photos
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = {"argv": sys.argv if argv is None else list(argv), "pid": os.getpid(), "weights": str(weights),
           "image": str(image_path), "image_sha256": sha256(image_path), "image_size": list(image.size),
           "prompt_sha256": sha256(prompt_path), "prompt_chars": len(prompt), "seed": args.seed,
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "provenance": provenance(), "limits": {"memory_limit_gb": args.memory_limit_gb,
                                                  "cache_limit_gb": args.cache_limit_gb, "previous": previous_limits},
           "status": "running", "events": []}

    def save_log():
        tmp = out / "run.json.tmp"
        tmp.write_text(json.dumps(log, indent=1))
        os.replace(tmp, out / "run.json")  # atomic: a reader never sees a half-written file

    last = {"t": 0.0}
    stage_closers = {"text_done", "stage_layout", "denoised", "decoded", "written"}

    def event(name, **kw):
        now = round(time.time() - t0, 1)
        e = {"t": now, "dt": round(now - last["t"], 1), "event": name, **mem(), **kw}
        if name in stage_closers or name.startswith("stage_") or name == "text_start":
            # the peak since the previous boundary, then start a fresh window for the next stage
            e["stage_peak_gb"], e["closes_stage"] = round(mx.get_peak_memory() / 1e9, 2), True
            mx.reset_peak_memory()
        last["t"] = now
        log["events"].append(e)
        print(json.dumps(e), flush=True)
        save_log()

    try:
        if not args.plan_only:
            event("writer_ok", codecs=check_writer(out))
        staged = load.staged_text_encoder(weights / "text_encoder", keep_layers=50)

        def encode_text(ids, vision):
            event("text_start", text_rows=len(ids))
            h = staged(ids, vision)
            event("text_done", hidden=list(h.shape))
            return h

        if args.check_headers:
            event("headers_ok", **load.check_headers(weights))
        c = config_components(args, weights, encode_text)
        log["plan"] = plan(c, prompt, image, args)
        event("plan", **log["plan"])
        if args.plan_only:
            log["status"] = "planned"
            save_log()
            return 0

        #@@start hyperflow-file
        if args.hyperflow and not args.decode_from:  # a re-decode never reads the adapter
            hf = args.hyperflow
            got = sha256(hf)
            want = hyperflow.KNOWN_SHA256.get(hf.name)
            event("hyperflow_file", file=hf.name, bytes=hf.stat().st_size, sha256=got, known_sha256=want)
            if want is not None and got != want:
                raise ValueError(f"{hf.name}: sha256 {got} is not the released {want}")
        #@@end hyperflow-file
        load_weights(c, weights, None if args.decode_from else args.hyperflow)
        if args.decode_from:
            z = np.load(args.decode_from.expanduser())
            meta = {k: int(z[k]) for k in ("ncv", "nca", "lf", "lh", "lw", "na")}
            c.dit_params.clear()
            video, audio = decode(c, mx.array(z["video_rows"]), mx.array(z["audio_rows"]), **meta)
            sr = c.audio_sampling_rate
        else:
            per_call = dit_tflop_per_call(log["plan"]["sequence_rows"], c.dit_cfg)
            if args.hyperflow:
                per_call += lora_tflop_per_call(log["plan"]["sequence_rows"], c.dit_cfg, log["plan"]["hyperflow"]["rank"])
            sched = hyperflow.read_header(args.hyperflow)[0].schedules() if args.hyperflow else None
            n_calls = log["plan"]["model_calls"]
            clock = {"prev": None}

            def on_step(i, t):
                now = time.time()
                step_s = round(now - (clock["prev"] or clock["denoise_start"]), 2)
                clock["prev"] = now
                done = [e["step_s"] for e in log["events"] if e["event"] == "step"][1:] + ([step_s] if i else [])
                eta = (sum(done) / len(done)) * (n_calls - 1 - i) if done else None
                event("step", i=i, timestep=float(t), step_s=step_s, tflops=round(per_call / step_s, 1),
                      eta_min=None if eta is None else round(eta / 60, 1))

            def stage(name):
                if name == "denoise":
                    clock["denoise_start"] = time.time()
                event(f"stage_{name}")

            def on_denoised(v, a, meta):
                np.savez(out / "final_rows.npz", video_rows=np.array(v.astype(mx.float32)),
                         audio_rows=np.array(a.astype(mx.float32)), **meta)
                event("denoised", saved="final_rows.npz", **meta)

            event("start")
            video, audio, sr = ref2va(c, prompt, [ImageRef(image)], num_frames=args.frames, height=args.height,
                                      width=args.width, num_inference_steps=args.steps, seed=args.seed,
                                      on_step=on_step, check_finite=True, on_denoised=on_denoised,
                                      free_transformer=True, on_stage=stage, schedules=sched)
        event("decoded", video_shape=list(video.shape), audio_shape=list(audio.shape),
              video_mean=float(video.mean()), video_std=float(video.std()),
              audio_rms=float(np.sqrt((audio ** 2).mean())))
        want = round(video.shape[0] / R.FPS * sr)  # decoded audio is 800 * latents samples; match the video exactly
        audio = audio[:, :want] if audio.shape[1] >= want else np.pad(audio, [(0, 0), (0, want - audio.shape[1])])
        write_wav(out / "audio.wav", audio, sr)
        write_mp4(out / "video.mp4", video, out / "audio.wav")
        log["status"] = "done"
        event("written", mp4=str(out / "video.mp4"))
        log["summary"] = summarize(log["events"])
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


if __name__ == "__main__":
    raise SystemExit(main())
