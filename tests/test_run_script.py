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


def test_run_script_writes_a_clip(release, tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_ref2va

    Image.fromarray(np.random.default_rng(1).integers(0, 256, (80, 64, 3), dtype=np.uint8)).save(tmp_path / "ref.png")
    (tmp_path / "prompt.txt").write_text("<Picture 1> walks past the stall, smiling. Audio: rain.")
    out = tmp_path / "out"
    common = ["--weights", str(release), "--image", str(tmp_path / "ref.png"), "--prompt-file",
              str(tmp_path / "prompt.txt"), "--out", str(out), "--height", "64", "--width", "64", "--frames", "124",
              "--steps", "3", "--ref-short-edge", "64", "--canvas-short-edge", "64"]
    assert run_ref2va.main(common + ["--plan-only"]) == 0
    plan = json.loads((out / "run.json").read_text())["plan"]
    assert plan["model_calls"] == 2 and plan["frames"] == 124 and plan["reference_size"] == [64, 80]
    assert not (out / "video.mp4").exists()

    assert run_ref2va.main(common) == 0
    log = json.loads((out / "run.json").read_text())
    names = [e["event"] for e in log["events"]]
    assert names.index("text_done") < names.index("step") < names.index("written")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames",
                            "-of", "json", str(out / "video.mp4")], capture_output=True, text=True, check=True)
    streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
    assert streams["video"]["width"] == 64 and streams["video"]["height"] == 64
    assert int(streams["video"]["nb_frames"]) == 124 and "audio" in streams
    assert torch is not None  # torch must be importable at runtime (noise, resampling)
