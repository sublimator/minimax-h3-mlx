"""One MiniMax H3 ref2va request on the released weights, staged so the big models are never needed at once.

    uv run python scripts/run_ref2va.py --weights MiniMax-H3 --image him.png --prompt-file prompt.txt \\
        --height 480 --width 864 --frames 124 --steps 50 --seed 0 --out out/run-001

Order (so the 51 GB text encoder and the 66 GB transformer are not evaluated together):
  1. normalise the references, build the Qwen3-VL presentation (CPU);
  2. load the text encoder, encode, free it (`load.staged_text_encoder`);
  3. encode the references with the video / audio VAEs;
  4. denoise on `transformer_ref` (loaded lazily: its weights are read on first use, after step 2);
  5. decode video and audio, write `video.mp4` (with sound), `audio.wav`, `run.json`.

`--plan-only` stops before loading any weights and prints the sizes (rows, steps) that the run would use.
Writes only under --out. Needs ffmpeg on PATH for the mp4.
"""

import argparse
import json
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
from h3mlx.pipeline import Components, ImageRef, ref2va, setup, vision_features  # noqa: E402
from h3mlx.presentation import build_presentation  # noqa: E402


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS reports bytes


def mem() -> dict:
    return {"mlx_active_gb": round(mx.get_active_memory() / 1e9, 2), "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "max_rss_gb": round(rss_gb(), 2)}


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


def components(args, weights: Path, encode_text) -> Components:
    tok_dir, proc_dir = weights / "tokenizer", weights / "processor"
    from transformers import AutoTokenizer, Qwen3VLProcessor

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    processor = Qwen3VLProcessor.from_pretrained(str(proc_dir))
    vae, vmean, vstd = load.load_vae(weights / "vae")
    avae, amean, astd = load.load_audio_vae(weights / "audio_vae")
    dit, dit_cfg = load.load_transformer(weights / "transformer_ref")  # lazy: read on first use in the denoise
    c = Components(tokenizer=tokenizer, processor=processor, encode_text=encode_text, vae=vae, vae_latents_mean=vmean,
                   vae_latents_std=vstd, audio_vae=avae, audio_latents_mean=amean, audio_latents_std=astd,
                   dit_params=dit, dit_cfg=dit_cfg)
    if args.ref_short_edge:
        c.reference_image_short_edge = args.ref_short_edge
    if args.canvas_short_edge:
        c.canvas_short_edge = args.canvas_short_edge
    return c


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
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--ref-short-edge", type=int, default=0, help="override 2048 (tests only)")
    ap.add_argument("--canvas-short-edge", type=int, default=0, help="override 768 (tests only)")
    args = ap.parse_args(argv)

    t0 = time.time()
    weights = args.weights.expanduser()
    prompt = args.prompt_file.expanduser().read_text().strip()
    image = Image.open(args.image.expanduser()).convert("RGB")
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = {"argv": sys.argv if argv is None else argv, "weights": str(weights), "image": str(args.image),
           "image_size": list(image.size), "prompt_chars": len(prompt), "seed": args.seed, "events": []}

    def event(name, **kw):
        e = {"t": round(time.time() - t0, 1), "event": name, **mem(), **kw}
        log["events"].append(e)
        print(json.dumps(e), flush=True)
        (out / "run.json").write_text(json.dumps(log, indent=1))

    staged = load.staged_text_encoder(weights / "text_encoder", keep_layers=50)

    def encode_text(ids, vision):
        event("text_start", text_rows=len(ids))
        h = staged(ids, vision)
        event("text_done", hidden=list(h.shape))
        return h

    c = components(args, weights, encode_text)
    log["plan"] = plan(c, prompt, image, args)
    event("plan", **log["plan"])
    if args.plan_only:
        return 0

    def on_step(i, t):
        event("step", i=i, timestep=float(t))

    event("start")
    video, audio, sr = ref2va(c, prompt, [ImageRef(image)], num_frames=args.frames, height=args.height,
                              width=args.width, num_inference_steps=args.steps, seed=args.seed, on_step=on_step)
    event("decoded", video_shape=list(video.shape), audio_shape=list(audio.shape))
    want = round(video.shape[0] / R.FPS * sr)  # the decoded audio runs 800 * latents samples; match the video exactly
    audio = audio[:, :want] if audio.shape[1] >= want else np.pad(audio, [(0, 0), (0, want - audio.shape[1])])
    write_wav(out / "audio.wav", audio, sr)
    write_mp4(out / "video.mp4", video, out / "audio.wav")
    event("written", mp4=str(out / "video.mp4"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
