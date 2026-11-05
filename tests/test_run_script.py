"""The real-run script end to end on tiny random models saved in the release's own folder layout: the same loaders,
staging, pipeline and writers as on the real run, only smaller."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import RELEASE
from PIL import Image

HF = RELEASE
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    if not (HF / "processor").exists() or shutil.which("ffmpeg") is None:
        pytest.skip("needs the release fixtures (scripts/fetch_test_fixtures.py) and ffmpeg")
    from test_pipeline_e2e import build

    t = build()
    d = tmp_path_factory.mktemp("MiniMax-H3")
    t["qwen"].save_pretrained(d / "text_encoder")
    from test_dit_mixed_precision import to_mixed

    to_mixed(t["dit"]).save_pretrained(d / "transformer_ref", max_shard_size="200KB")  # the release's precision split
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
    steps = [e for e in log["events"] if e["event"] == "step"]
    assert all(e["step_s"] > 0 and e["tflops"] > 0 for e in steps) and steps[-1]["eta_min"] == 0.0
    summ = log["summary"]
    for key in ("total_s", "text_s", "references_s", "denoise_s", "first_step_s", "step_s_mean", "decode_s", "write_s",
                "tflops_mean", "process_peak_rss_gb", "worst_pressure"):
        assert summ[key] is not None, key
    assert {"text_done", "stage_denoise", "denoised", "decoded"} <= set(summ["stage_peak_gb"])
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


def test_exif_rotated_photo_is_upright_like_diffusers(release, tmp_path):
    """A JPEG with EXIF Orientation=6 (rotate 90): diffusers' load_image applies exif_transpose; so must we."""
    from diffusers.utils import load_image

    img = Image.fromarray(np.random.default_rng(2).integers(0, 256, (48, 64, 3), dtype=np.uint8))  # 64 wide, 48 tall
    exif = Image.Exif()
    exif[0x0112] = 6
    img.save(tmp_path / "phone.jpg", exif=exif.tobytes())
    want = load_image(str(tmp_path / "phone.jpg"))
    args = _args(release, tmp_path, tmp_path / "plan")
    args[args.index("--image") + 1] = str(tmp_path / "phone.jpg")
    assert _script().main(args + ["--plan-only"]) == 0
    log = json.loads((tmp_path / "plan" / "run.json").read_text())
    assert want.size == (48, 64) and log["image_size"] == [48, 64]


def test_check_headers_passes_on_a_release_layout_and_catches_a_wrong_dtype(release, tmp_path):
    from safetensors.torch import load_file, save_file

    from h3mlx import load

    report = load.check_headers(release, keep_layers=50)
    assert report["transformer_ref"]["tensors"] > 0 and report["text_encoder"]["kept_tensors"] > 0
    broken = tmp_path / "broken"
    shutil.copytree(release, broken)
    shard = sorted((broken / "transformer_ref").glob("*.safetensors"))[0]
    sd = load_file(str(shard))
    k = next(k for k in sd if k.startswith("transformer_blocks"))
    sd[k] = sd[k].to(torch.float32) if sd[k].dtype != torch.float32 else sd[k].to(torch.bfloat16)
    save_file(sd, str(shard))
    with pytest.raises(ValueError, match="expected"):
        load.check_headers(broken, keep_layers=50)


def _hyperflow_file(release, path):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from safetensors.torch import save_file
    from test_hyperflow import make_file

    model = MiniMaxH3Transformer3DModel.from_pretrained(release / "transformer_ref")
    tensors, md = make_file(model, dict(model.config), scale=0.05)
    save_file(tensors, str(path), metadata=md)
    return path


def test_run_script_with_hyperflow(release, tmp_path):
    """--hyperflow: the file's 9-point grid (8 calls), the plan and the file's hash logged, a clip written."""
    run = _script()
    hf = _hyperflow_file(release, tmp_path / "hf_test.safetensors")
    args = _args(release, tmp_path, tmp_path / "out")
    i = args.index("--steps")
    del args[i : i + 2]
    with pytest.raises(SystemExit):
        run.main(args + ["--steps", "3", "--hyperflow", str(hf)])  # the grid comes from the file
    assert run.main(args + ["--hyperflow", str(hf)]) == 0
    log = json.loads((tmp_path / "out" / "run.json").read_text())
    assert log["status"] == "done"
    assert log["plan"]["model_calls"] == 8 and log["plan"]["grid_points"] == 9
    assert log["plan"]["hyperflow"]["gate"] == 0.25 and log["plan"]["hyperflow"]["rank"] == 4
    f = next(e for e in log["events"] if e["event"] == "hyperflow_file")
    assert len(f["sha256"]) == 64 and f["known_sha256"] is None
    steps = [e for e in log["events"] if e["event"] == "step"]
    assert [e["i"] for e in steps] == list(range(8))
    assert int(_probe(tmp_path / "out" / "video.mp4")["video"]["nb_frames"]) == 124

    again = tmp_path / "again"  # a re-decode with --hyperflow neither hashes nor loads the adapter
    redo = _args(release, tmp_path, again)
    i = redo.index("--steps")
    del redo[i : i + 2]
    assert run.main(redo + ["--hyperflow", str(hf), "--decode-from", str(tmp_path / "out" / "final_rows.npz")]) == 0
    events = [e["event"] for e in json.loads((again / "run.json").read_text())["events"]]
    assert "hyperflow_file" not in events and "step" not in events and "decoded" in events


def test_run_script_refuses_a_released_name_with_other_bytes(release, tmp_path):
    run = _script()
    hf = _hyperflow_file(release, tmp_path / "minimax_h3_hyperflow_8step_v1.0.safetensors")
    args = _args(release, tmp_path, tmp_path / "out")
    i = args.index("--steps")
    del args[i : i + 2]
    with pytest.raises(ValueError, match="not the released"):
        run.main(args + ["--hyperflow", str(hf)])
    assert json.loads((tmp_path / "out" / "run.json").read_text())["status"] == "error"


def test_plan_only_checks_the_hyperflow_file(release, tmp_path):
    """A file the port can't run fails in the plan step (headers only), before the hash or any weights."""
    import json as _json

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from safetensors.torch import save_file
    from test_hyperflow import make_file

    model = MiniMaxH3Transformer3DModel.from_pretrained(release / "transformer_ref")
    tensors, md = make_file(model, dict(model.config), scale=0.05)
    tensors["transformer.proj_in.lora_A.weight"] = torch.zeros(4, model.proj_in.in_features)
    tensors["transformer.proj_in.lora_B.weight"] = torch.zeros(model.proj_in.out_features, 4)
    bad = tmp_path / "bad.safetensors"
    save_file(tensors, str(bad), metadata=md)
    args = _args(release, tmp_path, tmp_path / "plan")
    i = args.index("--steps")
    del args[i : i + 2]
    with pytest.raises(ValueError, match="not run through ops.plinear"):
        _script().main(args + ["--hyperflow", str(bad), "--plan-only"])
    log = _json.loads((tmp_path / "plan" / "run.json").read_text())
    assert log["status"] == "error" and not any(e["event"] == "hyperflow_file" for e in log["events"])
