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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3mlx import load, references as R  # noqa: E402
from h3mlx.denoise import Schedules  # noqa: E402
from h3mlx.packing import Ref, build_ref2va_packed_sequence  # noqa: E402
from h3mlx.pipeline import Components, ImageRef, decode, ref2va, setup, vision_features  # noqa: E402
from h3mlx.presentation import build_presentation  # noqa: E402
from h3mlx.vae import VaeConfig, VideoVAE  # noqa: E402


def current_rss_gb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()
    return round(int(out or 0) * 1024 / 1e9, 2)


def mem() -> dict:
    return {"mlx_active_gb": round(mx.get_active_memory() / 1e9, 2), "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 2),
            "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "rss_gb": current_rss_gb(),
            "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_weights(c: Components, weights: Path) -> None:
    """Attach the weights lazily (mx.load reads data on first use). The audio VAE is small and loaded eagerly."""
    c.vae, _, _ = load.load_vae(weights / "vae")
    c.audio_vae, _, _ = load.load_audio_vae(weights / "audio_vae")
    c.dit_params, _ = load.load_transformer(weights / "transformer_ref")


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
    sch = Schedules.make(args.steps)
    return {"height": h, "width": w, "frames": n, "seconds": round(n / R.FPS, 3), "latent_frames": lf,
            "reference_size": list(refs[0].image.size), "text_rows": len(ids), "vision_tokens": int(sum(ic)),
            "reference_video_rows": layout.num_condition_video_rows, "target_audio_rows": na * 2,
            "sequence_rows": int(layout.position_ids.shape[0]), "grid_points": args.steps,
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
    ap.add_argument("--steps", type=int, default=50, help="sigma grid points incl. the final 0 (N-1 model calls)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="configs, tokenizer and processor only; no weights")
    ap.add_argument("--decode-from", type=Path, help="decode a saved final_rows.npz instead of running the model")
    ap.add_argument("--ref-short-edge", type=int, default=0, help="override 2048 (tests only)")
    ap.add_argument("--canvas-short-edge", type=int, default=0, help="override 768 (tests only)")
    args = ap.parse_args(argv)

    t0 = time.time()
    weights = args.weights.expanduser()
    prompt_path, image_path = args.prompt_file.expanduser(), args.image.expanduser()
    prompt = prompt_path.read_text().strip()
    image = Image.open(image_path).convert("RGB")
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = {"argv": sys.argv if argv is None else list(argv), "pid": os.getpid(), "weights": str(weights),
           "image": str(image_path), "image_sha256": sha256(image_path), "image_size": list(image.size),
           "prompt_sha256": sha256(prompt_path), "prompt_chars": len(prompt), "seed": args.seed,
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "provenance": provenance(), "status": "running", "events": []}

    def save_log():
        tmp = out / "run.json.tmp"
        tmp.write_text(json.dumps(log, indent=1))
        os.replace(tmp, out / "run.json")  # atomic: a reader never sees a half-written file

    def event(name, **kw):
        e = {"t": round(time.time() - t0, 1), "event": name, **mem(), **kw}
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

        c = config_components(args, weights, encode_text)
        log["plan"] = plan(c, prompt, image, args)
        event("plan", **log["plan"])
        if args.plan_only:
            log["status"] = "planned"
            save_log()
            return 0

        load_weights(c, weights)
        if args.decode_from:
            z = np.load(args.decode_from.expanduser())
            meta = {k: int(z[k]) for k in ("ncv", "nca", "lf", "lh", "lw", "na")}
            c.dit_params.clear()
            video, audio = decode(c, mx.array(z["video_rows"]), mx.array(z["audio_rows"]), **meta)
            sr = c.audio_sampling_rate
        else:
            def on_step(i, t):
                event("step", i=i, timestep=float(t))

            def on_denoised(v, a, meta):
                np.savez(out / "final_rows.npz", video_rows=np.array(v.astype(mx.float32)),
                         audio_rows=np.array(a.astype(mx.float32)), **meta)
                event("denoised", saved="final_rows.npz", **meta)

            event("start")
            video, audio, sr = ref2va(c, prompt, [ImageRef(image)], num_frames=args.frames, height=args.height,
                                      width=args.width, num_inference_steps=args.steps, seed=args.seed,
                                      on_step=on_step, check_finite=True, on_denoised=on_denoised,
                                      free_transformer=True)
        event("decoded", video_shape=list(video.shape), audio_shape=list(audio.shape),
              video_mean=float(video.mean()), video_std=float(video.std()),
              audio_rms=float(np.sqrt((audio ** 2).mean())))
        want = round(video.shape[0] / R.FPS * sr)  # decoded audio is 800 * latents samples; match the video exactly
        audio = audio[:, :want] if audio.shape[1] >= want else np.pad(audio, [(0, 0), (0, want - audio.shape[1])])
        write_wav(out / "audio.wav", audio, sr)
        write_mp4(out / "video.mp4", video, out / "audio.wav")
        log["status"] = "done"
        event("written", mp4=str(out / "video.mp4"))
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
