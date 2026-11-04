"""The real-run script end to end on tiny random models saved in the release's own folder layout: the same loaders,
staging, pipeline and writers as on the real run, only smaller."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

HF = Path(__file__).resolve().parents[2] / "ref-hf"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    if not (HF / "processor").exists() or shutil.which("ffmpeg") is None:
        pytest.skip("needs the released tokenizer/processor and ffmpeg")
    from test_pipeline_e2e import build

    t = build()
    d = tmp_path_factory.mktemp("MiniMax-H3")
    t["qwen"].save_pretrained(d / "text_encoder")
    t["dit"].save_pretrained(d / "transformer_ref", max_shard_size="200KB")
    t["vae"].save_pretrained(d / "vae")
    t["audio"].save_pretrained(d / "audio_vae")
    shutil.copytree(HF / "tokenizer", d / "tokenizer")
    shutil.copytree(HF / "processor", d / "processor")
    return d


def _args(release, tmp_path, out):
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (80, 64, 3), dtype=np.uint8)).save(tmp_path / "ref.png")
    (tmp_path / "prompt.txt").write_text("<Picture 1> walks past the stall, smiling. Audio: rain.")
    return ["--weights", str(release), "--image", str(tmp_path / "ref.png"), "--prompt-file", str(tmp_path / "prompt.txt"),
            "--out", str(out), "--height", "64", "--width", "64", "--frames", "124", "--steps", "3",
            "--ref-short-edge", "64", "--canvas-short-edge", "64"]


def _script():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_ref2va

    return run_ref2va


def _probe(mp4):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames",
                        "-of", "json", str(mp4)], capture_output=True, text=True, check=True)
    return {s["codec_type"]: s for s in json.loads(p.stdout)["streams"]}


def test_plan_only_needs_no_weight_files(release, tmp_path):
    """--plan-only on a tree with configs, tokenizer and processor but not a single .safetensors file."""
    bare = tmp_path / "bare"
    for sub in ("transformer_ref", "vae", "audio_vae", "text_encoder"):
        (bare / sub).mkdir(parents=True)
        shutil.copy(release / sub / "config.json", bare / sub / "config.json")
    shutil.copytree(release / "tokenizer", bare / "tokenizer")
    shutil.copytree(release / "processor", bare / "processor")
    assert not list(bare.rglob("*.safetensors"))
    out = tmp_path / "plan"
    assert _script().main(_args(bare, tmp_path, out) + ["--plan-only"]) == 0
    log = json.loads((out / "run.json").read_text())
    assert log["status"] == "planned" and log["plan"]["model_calls"] == 2 and log["plan"]["reference_size"] == [64, 80]
    assert not (out / "video.mp4").exists()


def test_run_script_writes_a_clip_and_can_redecode(release, tmp_path):
    run = _script()
    out = tmp_path / "out"
    assert run.main(_args(release, tmp_path, out)) == 0
    log = json.loads((out / "run.json").read_text())
    names = [e["event"] for e in log["events"]]
    assert log["status"] == "done"
    assert names[0] == "writer_ok"
    assert names.index("text_done") < names.index("stage_references") < names.index("stage_denoise") \
        < names.index("step") < names.index("denoised") < names.index("stage_decode") < names.index("written")
    assert log["limits"]["memory_limit_gb"] == 110.0 and "pressure" in log["events"][0]
    assert all("mlx_cache_gb" in e and "rss_gb" in e for e in log["events"])
    assert log["provenance"]["packages"]["mlx"] and len(log["image_sha256"]) == 64
    streams = _probe(out / "video.mp4")
    assert streams["video"]["width"] == 64 and int(streams["video"]["nb_frames"]) == 124 and "audio" in streams
    assert (out / "final_rows.npz").exists()

    again = tmp_path / "again"
    assert run.main(_args(release, tmp_path, again) + ["--decode-from", str(out / "final_rows.npz")]) == 0
    assert int(_probe(again / "video.mp4")["video"]["nb_frames"]) == 124
    a = json.loads((out / "run.json").read_text())["events"]
    b = json.loads((again / "run.json").read_text())["events"]
    da, db = next(e for e in a if e["event"] == "decoded"), next(e for e in b if e["event"] == "decoded")
    assert da["video_mean"] == db["video_mean"] and da["audio_rms"] == db["audio_rms"]
    assert torch is not None  # torch must be importable at runtime (noise, resampling)


def test_nan_stops_the_denoise_at_step_0():
    import mlx.core as mx
    from conftest import state_mx
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from test_denoise import tiny_layout
    from test_dit_tiny import TINY

    from h3mlx.denoise import Schedules, denoise

    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    p = state_mx(model)
    p["proj_out.bias"] = p["proj_out.bias"] * float("nan")
    lay = tiny_layout()
    args = (p, dict(model.config), lay, mx.zeros((1, len(lay.text_indices), 20)),
            mx.zeros((len(lay.video_indices), 16)), mx.zeros((len(lay.audio_indices), 6)), Schedules.make(5))
    denoise(*args)  # silently produces NaNs without the check
    with pytest.raises(FloatingPointError, match="step 0 video rows"):
        denoise(*args, check_finite=True)
