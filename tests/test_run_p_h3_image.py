"""`scripts/run_p_h3_image.py` on a tiny release saved in diffusers' own layout: text-to-image and edit through the
command line against diffusers' `PH3ImageBlocks`, the re-decode path (faithful, and refusing bad rows), the VAE's
bfloat16 loading policy against `from_pretrained`, the upsampler's aspect label, and early refusals."""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import RELEASE
from PIL import Image

from h3mlx import image as I
from h3mlx import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p_h3_image as R  # noqa: E402

pytestmark = pytest.mark.skipif(not (RELEASE / "processor").exists(),
                                reason="release fixtures not present (scripts/fetch_test_fixtures.py)")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Tiny random components (as test_p_h3_image) saved as a P-H3-Image release, and the torch modules."""
    import test_p_h3_image as T

    t = T.build()
    t["vae"].enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256, tile_sample_min_overlap_height=64,
                           tile_sample_min_overlap_width=64)  # the defaults, which the release layout gives ours too
    rel = tmp_path_factory.mktemp("release")
    for name, module in (("transformer", t["dit"]), ("vae", t["vae"]), ("text_connector", t["conn"]),
                         ("text_encoder", t["qwen"])):
        module.save_pretrained(str(rel / name))
    for name in ("tokenizer", "processor"):
        shutil.copytree(RELEASE / name, rel / name)
    (rel / "modular_model_index.json").write_text(json.dumps(
        {"_class_name": "PH3ImageModularPipeline", **T.E2E, "sample_sigmas": list(I.SAMPLE_SIGMAS)}))
    rng = np.random.default_rng(11)
    pics = []
    for i, (h, w) in enumerate(((40, 48), (56, 32))):
        p = rel.parent / f"pic{i}.png"
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(p)
        pics.append(p)
    return t, rel, pics


def _prompt(tmp_path, text):
    p = tmp_path / "prompt.txt"
    p.write_text(text)
    return p


def _run(argv):
    return R.main([str(a) for a in argv])


def _log(out):
    return json.loads((out / "run.json").read_text())


T2I = "A risograph print of a lighthouse at dusk, two inks, no other text."
EDIT = "give the lighthouse in picture 1 the stripes from picture 2"


@pytest.mark.parametrize("workflow", ["text2image", "edit"])
def test_cli_matches_diffusers(tiny, tmp_path, workflow):
    import test_p_h3_image as T

    t, rel, pics = tiny
    out = tmp_path / "out"
    if workflow == "edit":
        argv = ["--weights", rel, "--prompt-file", _prompt(tmp_path, EDIT), "--out", out, "--seed", 7,
                "--image", pics[0], "--image", pics[1]]
        images = [Image.open(p).convert("RGB") for p in pics]
        want, *_ = T.run_diffusers(t, EDIT, images, None, None, seed=7)
    else:
        argv = ["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--out", out, "--seed", 7,
                "--height", 40, "--width", 56]
        want, *_ = T.run_diffusers(t, T2I, None, 40, 56, seed=7)
    assert _run(argv) == 0
    got = np.asarray(Image.open(out / "image.png"), dtype=np.int16)
    exp = I.to_uint8(want).astype(np.int16)
    assert got.shape == exp.shape and np.abs(got - exp).max() <= 1
    log = _log(out)
    assert log["status"] == "done" and log["plan"]["workflow"] == workflow
    assert len([e for e in log["events"] if e["event"] == "step"]) == 8
    src = json.loads(str(np.load(out / "final_rows.npz")["source"]))
    assert src["prompt"] == (EDIT if workflow == "edit" else T2I) and src["seed"] == 7


def test_cli_redecode_is_faithful_and_says_where_it_came_from(tiny, tmp_path):
    t, rel, _ = tiny
    first = tmp_path / "first"
    assert _run(["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--out", first, "--seed", 3,
                 "--height", 40, "--width", 56]) == 0
    again = tmp_path / "again"
    assert _run(["--weights", rel, "--out", again, "--decode-from", first / "final_rows.npz"]) == 0
    assert np.array_equal(np.asarray(Image.open(again / "image.png")), np.asarray(Image.open(first / "image.png")))
    log = _log(again)
    assert log["plan"]["workflow"] == "decode" and (log["plan"]["height"], log["plan"]["width"]) == (40, 56)
    assert log["plan"]["source"]["prompt"] == T2I and log["plan"]["source"]["seed"] == 3
    assert len(log["plan"]["decode_from_sha256"]) == 64 and "prompt_sha256" not in log and "seed" not in log
    assert (again / "prompt.txt").read_text().strip() == T2I


def _saved(tiny, tmp_path, **change):
    t, rel, _ = tiny
    first = tmp_path / "first"
    if not (first / "final_rows.npz").exists():
        assert _run(["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--out", first, "--height", 40,
                     "--width", 56]) == 0
    z = dict(np.load(first / "final_rows.npz"))
    z.update(change)
    p = tmp_path / "changed.npz"
    np.savez(p, **z)
    return rel, p


def test_cli_redecode_refuses_nonfinite_rows(tiny, tmp_path):
    rel, p = _saved(tiny, tmp_path)
    z = dict(np.load(p))
    z["video_rows"] = np.full_like(z["video_rows"], np.nan)
    np.savez(p, **z)
    out = tmp_path / "nan"
    with pytest.raises(FloatingPointError):
        _run(["--weights", rel, "--out", out, "--decode-from", p])
    assert _log(out)["status"] == "error" and not (out / "image.png").exists()


def test_cli_redecode_refuses_rows_that_do_not_fit(tiny, tmp_path):
    rel, p = _saved(tiny, tmp_path, lh=np.array(4))
    out = tmp_path / "misfit"
    with pytest.raises(ValueError):
        _run(["--weights", rel, "--out", out, "--decode-from", p])
    assert _log(out)["status"] == "error" and not (out / "image.png").exists()


@pytest.mark.parametrize("extra", [["--prompt-file", "p.txt"], ["--image", "a.png"], ["--sigmas", "1,0"],
                                   ["--upsample"], ["--height", "8", "--width", "8"]])
def test_cli_redecode_refuses_generation_options(tiny, tmp_path, extra):
    _, rel, _ = tiny
    with pytest.raises(SystemExit):
        _run(["--weights", rel, "--out", tmp_path / "x", "--decode-from", tmp_path / "rows.npz", *extra])


def test_cli_refuses_early(tiny, tmp_path):
    """Nine pictures or a bad grid fail at the argument check, before any model (the upsampler included) loads."""
    _, rel, pics = tiny
    p = _prompt(tmp_path, EDIT)
    for extra in (["--image", pics[0]] * 9, ["--sigmas", "1,0.5,0.6,0"]):
        with pytest.raises(SystemExit):
            _run(["--weights", rel, "--prompt-file", p, "--out", tmp_path / "early", "--upsample", *extra])
        assert not (tmp_path / "early" / "run.json").exists()


def test_upsampler_aspect_is_the_requested_size():
    from diffusers.modular_pipelines.p_h3_image.prompt_upsampler import aspect_ratio_label

    assert R.upsampler_aspect(1000, 1150) == aspect_ratio_label(1150, 1000) == "1:1"  # rounded first: 4:3
    assert R.upsampler_aspect(None, None) == aspect_ratio_label(2912, 2912)
    assert R.upsampler_aspect(2176, 3872) == "16:9"


def test_plan_only_with_upsample_says_it_is_the_request(tiny, tmp_path):
    _, rel, _ = tiny
    out = tmp_path / "plan"
    assert _run(["--weights", rel, "--prompt-file", _prompt(tmp_path, "a lighthouse"), "--out", out, "--plan-only",
                 "--upsample", "--height", 40, "--width", 56]) == 0
    log = _log(out)
    assert log["status"] == "planned" and "upsampler" not in json.dumps(log["events"])
    assert "before encoding" in log["plan"]["text_basis"]


def test_cli_unusable_upsampler_answer_stops_before_the_image_model(tiny, tmp_path):
    """A tiny random Qwen3.5 never closes its reasoning: the run must stop with the answer logged and no denoise."""
    from test_upsampler import IDS, PROC, TEXT, VISION
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    _, rel, _ = tiny
    rel2 = tmp_path / "release-upsampler"
    shutil.copytree(rel, rel2)
    torch.manual_seed(13)
    Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=TEXT, vision_config=VISION, **IDS)).eval().save_pretrained(
        str(rel2 / "prompt_upsampler"))
    shutil.copytree(PROC, rel2 / "prompt_upsampler_processor")
    out = tmp_path / "ups"
    with pytest.raises(RuntimeError, match="upsampler"):
        _run(["--weights", rel2, "--prompt-file", _prompt(tmp_path, "a lighthouse"), "--out", out, "--upsample",
              "--max-new-tokens", 6, "--height", 40, "--width", 56])
    log = _log(out)
    names = [e["event"] for e in log["events"]]
    assert log["status"] == "error" and "upsample_start" in names and "start" not in names
    assert log["upsampler_answer"] is not None and log["request"] == "a lighthouse"


# -- the VAE's dtype policy ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ["AutoencoderKLPH3Image", "AutoencoderKLMiniMaxH3"])
def test_vae_bf16_loading_matches_from_pretrained(tmp_path, cls):
    import diffusers
    import test_p_h3_image as T

    ref = getattr(diffusers, cls)(**T.VAE)
    ref.save_pretrained(str(tmp_path / "vae"))
    up = getattr(diffusers, cls).from_pretrained(str(tmp_path / "vae"), torch_dtype=torch.bfloat16)
    want = {k: str(v.dtype).replace("torch.", "") for k, v in up.state_dict().items()}
    vae, _, _ = load.load_vae(tmp_path / "vae", dtype=mx.bfloat16)
    got = {k: str(v.dtype).replace("mlx.core.", "") for k, v in vae.p.items()}
    assert got == want
    assert "float32" in set(got.values()) and "bfloat16" in set(got.values())
    stored, _, _ = load.load_vae(tmp_path / "vae")  # dtype=None: as stored (H3's video path relies on this)
    assert {str(v.dtype) for v in stored.p.values()} == {"mlx.core.float32"}


def test_image_decode_bf16_policy_matches_pipeline(tmp_path):
    """The bf16 decode against diffusers loaded the same way: same policy, so only bf16 rounding differs. This bounds
    the error; it does not pin the policy (casting proj_out to bf16 too stays inside it): the dtype test above does."""
    import test_p_h3_image as T
    from diffusers import AutoencoderKLPH3Image

    T.randomize(AutoencoderKLPH3Image(**T.VAE), 7).save_pretrained(str(tmp_path / "vae"))
    up = AutoencoderKLPH3Image.from_pretrained(str(tmp_path / "vae"), torch_dtype=torch.bfloat16).eval()
    vae, _, _ = load.load_vae(tmp_path / "vae", dtype=mx.bfloat16)
    z = torch.randn(1, 4, 1, 10, 12)
    with torch.no_grad():
        want = up.decode(z).sample.float()
    got = vae.decode(T.cl(z)).astype(mx.float32)
    err = np.abs(np.array(T.back(got)) - want.numpy())
    print(f"bf16 decode max |diff| {err.max():.3e}, mean {err.mean():.3e}, std {want.std():.3f}")
    assert err.mean() < 0.02 * float(want.std()) and err.max() < 0.25 * float(want.std())
