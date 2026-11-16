"""`scripts/run_p_h3_image.py` on a tiny release saved in diffusers' own layout: text-to-image and edit through the
command line against diffusers' `PH3ImageBlocks`, the re-decode path (faithful, and refusing bad rows), the VAE's
bfloat16 loading policy against `from_pretrained`, the upsampler's aspect label, early refusals, and `--lora`: a
trained adapter against diffusers with the same file loaded, its scale, the plan and the record."""

import hashlib
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
from h3mlx import load, lora

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p_h3_image as R  # noqa: E402

pytestmark = pytest.mark.skipif(not (RELEASE / "processor").exists(),
                                reason="release fixtures not present (scripts/fetch_test_fixtures.py)")


def _release(tmp_path_factory, **dims):
    """Tiny random components (as test_p_h3_image) saved as a P-H3-Image release, and the torch modules."""
    import test_p_h3_image as T

    t = T.build(**dims)
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


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _release(tmp_path_factory)


@pytest.fixture(scope="module")
def tiny_q(tmp_path_factory):
    """The same, with widths (64, 128) that quantize whole in groups of 64."""
    return _release(tmp_path_factory, hidden_size=64, ffn_dim=128)


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
    assert src["quantize"] is None and src["modulation_cache"] is True


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
    assert log["settings"]["seed"] is None  # a replay has no seed of its own
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
                                   ["--upsample"], ["--height", "8", "--width", "8"], ["--seed", "99"],
                                   ["--seed", "0"], ["--max-new-tokens", "6"], ["--quantize", "8"],
                                   ["--lora", "a.safetensors", "--lora-scale", "1"], ["--lora-scale", "1"]])
def test_cli_redecode_refuses_generation_options(tiny, tmp_path, extra):
    _, rel, _ = tiny
    with pytest.raises(SystemExit):
        _run(["--weights", rel, "--out", tmp_path / "x", "--decode-from", tmp_path / "rows.npz", *extra])


def test_cli_quantize_refuses_on_the_shapes_before_the_text_encoder(tiny, tmp_path):
    """The tiny release's widths aren't multiples of 64, so `--quantize` can't quantize them whole: the run fails
    right after the weights are opened (shapes only), never after encoding, and never with matrices left in bf16."""
    _, rel, _ = tiny
    out = tmp_path / "q"
    with pytest.raises(ValueError, match="groups of 64"):
        _run(["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--out", out, "--quantize", "8"])
    log = _log(out)
    assert log["status"] == "error" and not any(e["event"].startswith(("text", "stage_")) for e in log["events"])


@pytest.mark.parametrize("bits", [8, 4])
def test_cli_quantized_run_quantizes_after_encoding_and_records_it(tiny_q, tmp_path, bits):
    """A quantized generation end to end: every target quantized, at its own stage after the text encoder (the
    order is what keeps the text encoder and the quantized matrices apart; at this size the bytes prove nothing),
    the saved rows say so, and the picture is the unquantized one's within quantization error (measured 0.01 and
    0.12 of 255)."""
    _, rel, _ = tiny_q
    common = ["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--seed", 7, "--height", 40, "--width", 56]
    assert _run([*common, "--out", tmp_path / "plain"]) == 0
    out = tmp_path / f"q{bits}"
    assert _run([*common, "--out", out, "--quantize", bits]) == 0
    log = _log(out)
    names = [e["event"] for e in log["events"]]
    assert names.index("text_done") < names.index("stage_quantize") < names.index("quantized") < names.index("stage_denoise")
    q = next(e for e in log["events"] if e["event"] == "quantized")
    assert q["bits"] == bits and q["weights"] == 6 * 3  # six matrices in each of 2 blocks and 1 refiner block
    src = json.loads(str(np.load(out / "final_rows.npz")["source"]))
    assert src["quantize"] == bits and src["modulation_cache"] is True
    got = np.asarray(Image.open(out / "image.png"), dtype=np.float32)
    plain = np.asarray(Image.open(tmp_path / "plain" / "image.png"), dtype=np.float32)
    assert got.shape == plain.shape == (40, 56, 3)
    print(f"{bits}-bit vs unquantized: mean |diff| {np.abs(got - plain).mean():.2f} / 255")
    assert np.abs(got - plain).mean() < (0.05 if bits == 8 else 0.5)


def test_cli_refuses_early(tiny, tmp_path):
    """Nine pictures or a bad grid fail at the argument check, before any model (the upsampler included) loads."""
    _, rel, pics = tiny
    p = _prompt(tmp_path, EDIT)
    for extra in (["--image", pics[0]] * 9, ["--sigmas", "1,0.5,0.6,0"]):
        with pytest.raises(SystemExit):
            _run(["--weights", rel, "--prompt-file", p, "--out", tmp_path / "early", "--upsample", *extra])
        assert not (tmp_path / "early" / "run.json").exists()


def test_cli_sizes(tiny, tmp_path):
    """Text-to-image takes one side alone (the other defaults to 2912); an edit takes both or neither."""
    _, rel, pics = tiny
    out = tmp_path / "one-side"
    assert _run(["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--out", out, "--plan-only",
                 "--height", 1000]) == 0
    assert (_log(out)["plan"]["height"], _log(out)["plan"]["width"]) == (1000, 2912)  # the tiny release's multiple is 8
    with pytest.raises(SystemExit):
        _run(["--weights", rel, "--prompt-file", _prompt(tmp_path, EDIT), "--out", tmp_path / "e", "--plan-only",
              "--image", pics[0], "--height", 64])


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


# -- a trained adapter: --lora and --lora-scale ------------------------------------------------------------------------


def _adapter(folder, cfg, modules=None, std=0.2, name="adapter.safetensors"):
    """A random adapter (rank 4, alpha 6: scaling 1.5) written as the trainer writes its own, with `lora.save`."""
    from test_lora import random_adapter

    modules = lora.block_modules(cfg) if modules is None else modules
    lora.save(folder / name, random_adapter(cfg, {m: 4 for m in modules}, {m: 6.0 for m in modules}, seed=5, std=std))
    return folder / name


@pytest.fixture(scope="module")
def tiny_lora(tmp_path_factory):
    """Its own tiny release, an adapter file on every block Linear, diffusers' picture without the adapter, and from
    here on the adapter loaded into the torch transformer (`load_lora_adapter`, under the name "default")."""
    import test_p_h3_image as T

    t, rel, _ = _release(tmp_path_factory)
    path = _adapter(tmp_path_factory.mktemp("lora"), R.configs(rel)[0])
    with torch.no_grad():
        without, *_ = T.run_diffusers(t, T2I, None, 40, 56, seed=7)
    t["dit"].load_lora_adapter(str(path.parent), weight_name=path.name, adapter_name="default")
    return t, rel, path, without


def _t2i(rel, tmp_path, out, *extra):
    return ["--weights", rel, "--prompt-file", _prompt(tmp_path, T2I), "--out", out, "--seed", 7, "--height", 40,
            "--width", 56, *extra]


def _rows(out):
    return np.load(out / "final_rows.npz")["video_rows"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("scale", [1.0, 0.6, -1.3])
def test_cli_lora_matches_diffusers(tiny_lora, tmp_path, scale):
    """`--lora --lora-scale` against diffusers with the same file loaded and the same weight set: the picture to one
    level of 255, as without an adapter. Guards: the adapter moves diffusers' picture by many levels, and so does
    the scale."""
    import test_p_h3_image as T

    t, rel, path, without = tiny_lora
    with torch.no_grad():
        trained, *_ = T.run_diffusers(t, T2I, None, 40, 56, seed=7)
        t["dit"].set_adapters("default", weights=scale)
        try:
            want, *_ = T.run_diffusers(t, T2I, None, 40, 56, seed=7)
        finally:
            t["dit"].set_adapters("default", weights=1.0)
    out = tmp_path / "out"
    assert _run(_t2i(rel, tmp_path, out, "--lora", path, "--lora-scale", scale)) == 0
    got = np.asarray(Image.open(out / "image.png"), dtype=np.int16)
    exp, plain, as_trained = (I.to_uint8(a).astype(np.int16) for a in (want, without, trained))
    print(f"x {scale}: max level diff {np.abs(got - exp).max()}; the adapter moves diffusers' picture by "
          f"{np.abs(exp - plain).max()} levels, the scale by {np.abs(exp - as_trained).max()}")
    assert got.shape == exp.shape and np.abs(got - exp).max() <= 1
    assert np.abs(exp - plain).max() >= 20
    assert scale == 1.0 or np.abs(exp - as_trained).max() >= 20


def test_cli_lora_scale_zero_is_the_run_without_one(tiny_lora, tmp_path):
    """Scale 0: the same rows and the same PNG as the run without `--lora`, bit for bit, with the adapter still
    checked, hashed and recorded as not applied. A run without `--lora` records that it had none."""
    _, rel, path, _ = tiny_lora
    plain, zero = tmp_path / "plain", tmp_path / "zero"
    assert _run(_t2i(rel, tmp_path, plain)) == 0
    assert _run(_t2i(rel, tmp_path, zero, "--lora", path, "--lora-scale", 0)) == 0
    assert np.array_equal(_rows(zero).view(np.uint32), _rows(plain).view(np.uint32))
    assert (zero / "image.png").read_bytes() == (plain / "image.png").read_bytes()
    log = _log(zero)
    assert log["plan"]["lora"]["applied"] is False and log["plan"]["lora"]["sha256"] == _sha(path)
    assert log["plan"]["lora"]["scalings"] == [0.0]
    (event,) = [e for e in log["events"] if e["event"] == "lora"]
    assert event["applied"] is False and event["scale"] == 0
    source = json.loads(str(np.load(zero / "final_rows.npz")["source"]))
    assert source["lora"] == {"path": str(path), "sha256": _sha(path), "scale": 0.0}
    assert json.loads(str(np.load(plain / "final_rows.npz")["source"]))["lora"] is None
    assert "lora" not in _log(plain)["plan"] and not any(e["event"] == "lora" for e in _log(plain)["events"])


@pytest.mark.parametrize("where", ["blocks", "every Linear"])
def test_cli_lora_gives_the_same_rows_with_the_modulation_cache_on_and_off(tiny_lora, tmp_path, where):
    """The cache computes each call's AdaLN tables up front and drops the projections. An adapter on the block
    Linears is not in those tables at all; one on every Linear is, through the time embedder and the projections,
    and the cache carries it. Either way the rows are the uncached run's, bit for bit. Guard: the adapter counts."""
    _, rel, path, _ = tiny_lora
    if where == "every Linear":
        cfg = R.configs(rel)[0]
        path = _adapter(tmp_path, cfg, sorted(lora.wired_shapes(cfg)), std=0.1)
    on, off, plain = tmp_path / "on", tmp_path / "off", tmp_path / "plain"
    assert _run(_t2i(rel, tmp_path, on, "--lora", path, "--lora-scale", 0.6)) == 0
    assert _run(_t2i(rel, tmp_path, off, "--lora", path, "--lora-scale", 0.6, "--no-modulation-cache")) == 0
    assert _run(_t2i(rel, tmp_path, plain)) == 0
    assert np.array_equal(_rows(on).view(np.uint32), _rows(off).view(np.uint32))
    assert np.abs(_rows(on) - _rows(plain)).max() > 0.01
    sources = [json.loads(str(np.load(o / "final_rows.npz")["source"])) for o in (on, off)]
    assert [s["modulation_cache"] for s in sources] == [True, False]


def test_cli_lora_plan_only_names_the_adapter_and_opens_no_weights(tiny_lora, tmp_path, monkeypatch):
    """The plan says which file (its hash and size), the scale, the Linears by kind with the blocks they cover, the
    rank, the alpha, the scaling the forward multiplies by and the parameter count, from the adapter file's header:
    every loader of tensors is made to raise."""
    _, rel, path, _ = tiny_lora

    def refuse(*a, **k):
        raise AssertionError("weights were opened")

    for owner, name in ((mx, "load"), (lora, "load"), (load, "load_transformer"), (load, "load_vae"),
                        (load, "load_safetensors"), (lora, "for_sampling")):
        monkeypatch.setattr(owner, name, refuse)
    out = tmp_path / "plan"
    assert _run(_t2i(rel, tmp_path, out, "--plan-only", "--lora", path, "--lora-scale", 0.8)) == 0
    log = _log(out)
    cfg = R.configs(rel)[0]
    shapes = lora.wired_shapes(cfg)
    kinds = {f"{blocks}.*.{target}": covered
             for blocks, covered in (("transformer_blocks", "0-1"), ("token_refiner.refiner_blocks", "0"))
             for target in lora.BLOCK_TARGETS}
    assert log["status"] == "planned" and log["plan"]["lora"] == {
        "path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size, "scale": 0.8, "applied": True,
        "modules": 18, "module_kinds": kinds, "ranks": [4], "alphas": [6.0], "scalings": [0.8 * 6.0 / 4],
        "parameters": sum(4 * (shapes[m][0] + shapes[m][1]) for m in lora.block_modules(cfg))}
    assert not (out / "image.png").exists() and not any(e["event"] == "lora" for e in log["events"])


def test_cli_lora_record_and_redecode(tiny_lora, tmp_path):
    """What a run with an adapter leaves: the plan's entry, the arguments, an event when the adapter went on (before
    anything else runs), and in `final_rows.npz` the file's hash and the scale, which a re-decode reports as its
    source."""
    _, rel, path, _ = tiny_lora
    out = tmp_path / "out"
    assert _run(_t2i(rel, tmp_path, out, "--lora", path, "--lora-scale", 0.6)) == 0
    log = _log(out)
    assert log["status"] == "done" and log["settings"]["lora"] == str(path) and log["settings"]["lora_scale"] == 0.6
    assert log["plan"]["lora"]["sha256"] == _sha(path) and log["plan"]["lora"]["scale"] == 0.6
    names = [e["event"] for e in log["events"]]
    assert names.index("plan") < names.index("lora") < names.index("start") < names.index("text_done")
    event = log["events"][names.index("lora")]
    assert (event["applied"], event["scale"], event["modules"], event["sha256"]) == (True, 0.6, 18, _sha(path))
    recorded = {"path": str(path), "sha256": _sha(path), "scale": 0.6}
    assert json.loads(str(np.load(out / "final_rows.npz")["source"]))["lora"] == recorded
    again = tmp_path / "again"
    assert _run(["--weights", rel, "--out", again, "--decode-from", out / "final_rows.npz"]) == 0
    assert _log(again)["plan"]["source"]["lora"] == recorded
    assert np.array_equal(np.asarray(Image.open(again / "image.png")), np.asarray(Image.open(out / "image.png")))


def test_cli_lora_refuses_at_the_arguments(tiny_lora, tmp_path, capsys):
    """No default for the scale, and nothing that is not an adapter for this transformer: each is an argument error
    that names the trouble, before any model loads and before `run.json` exists."""
    from safetensors.torch import save_file

    t, rel, path, _ = tiny_lora
    cfg = R.configs(rel)[0]
    wide = _adapter(tmp_path, dict(cfg, hidden_size=64), name="wide.safetensors")
    unknown = tmp_path / "unknown.safetensors"
    save_file({"transformer.nowhere.lora_A.weight": torch.zeros(2, 4), "transformer.nowhere.lora_B.weight": torch.zeros(4, 2)},
              str(unknown))
    not_one = tmp_path / "text.safetensors"
    not_one.write_text("not a safetensors file")
    base = tmp_path / "base.safetensors"
    save_file({"proj_in.weight": torch.zeros(2, 2)}, str(base))
    out = tmp_path / "early"
    for extra, said in (
            (["--lora", path], "--lora needs --lora-scale"),
            (["--lora-scale", 1.0], "give the adapter with --lora"),
            (["--lora", path, "--lora-scale", "nan"], "scale must be a finite number"),
            (["--lora", path, "--lora-scale", "inf"], "scale must be a finite number"),
            (["--lora", tmp_path / "absent.safetensors", "--lora-scale", 1.0], "No such file"),
            (["--lora", not_one, "--lora-scale", 1.0], "not a safetensors file"),
            (["--lora", base, "--lora-scale", 1.0], "only <module>.lora_A/B.weight keys"),
            (["--lora", wide, "--lora-scale", 1.0], "do not fit the Linear"),
            (["--lora", unknown, "--lora-scale", 1.0], "not Linears of this transformer"),
            (["--lora", wide, "--lora-scale", 0], "do not fit the Linear")):  # scale 0 excuses nothing
        with pytest.raises(SystemExit):
            _run(_t2i(rel, tmp_path, out, *extra))
        assert said in capsys.readouterr().err, extra
        assert not out.exists()


def test_cli_lora_refuses_a_broken_adapter_before_the_text_encoder(tiny_lora, tmp_path, monkeypatch):
    """What the header cannot show is refused when the file is read, which is before the text encoder runs: a NaN
    in a matrix, and a file that is not the one the plan hashed."""
    _, rel, path, _ = tiny_lora
    adapter = lora.load(path)
    key = sorted(adapter.tensors)[5]
    poisoned = np.array(adapter.tensors[key])
    poisoned[0, 0] = np.nan
    bad = tmp_path / "nan.safetensors"
    lora.save(bad, lora.Adapter(dict(adapter.tensors, **{key: mx.array(poisoned)}), dict(adapter.alpha)))
    out = tmp_path / "nan"
    with pytest.raises(ValueError, match="hold a NaN or an infinity"):
        _run(_t2i(rel, tmp_path, out, "--lora", bad, "--lora-scale", 1.0))
    log = _log(out)
    assert log["status"] == "error" and not any(e["event"] in ("lora", "start", "text_done") for e in log["events"])

    swapped = tmp_path / "swapped.safetensors"
    shutil.copy(path, swapped)
    read = lora.load

    def load_then_swap(p, *a, **k):
        loaded = read(p, *a, **k)
        mx.eval(loaded.tensors)
        shutil.copy(bad, p)
        return loaded

    monkeypatch.setattr(lora, "load", load_then_swap)
    out = tmp_path / "swapped"
    with pytest.raises(RuntimeError, match="changed while it was being read"):
        _run(_t2i(rel, tmp_path, out, "--lora", swapped, "--lora-scale", 1.0))
    assert _log(out)["status"] == "error" and not any(e["event"] in ("lora", "start") for e in _log(out)["events"])


@pytest.mark.parametrize("bits", [8, 4])
def test_cli_lora_with_quantize_quantizes_the_base_and_keeps_the_adapter(tiny_q, tmp_path, bits):
    """`--quantize` with `--lora`: the adapter goes on first, the six base matrices of each block are quantized as
    they are without one, and the adapter stays dense beside them. The picture is the unquantized run's with the same
    adapter, within the quantization error measured without one, and far from the quantized run's without it."""
    _, rel, _ = tiny_q
    path = _adapter(tmp_path, R.configs(rel)[0])
    runs = {"dense": ["--lora", path, "--lora-scale", 0.7], "both": ["--lora", path, "--lora-scale", 0.7, "--quantize", bits],
            "quantized": ["--quantize", bits]}
    for name, extra in runs.items():
        assert _run(_t2i(rel, tmp_path, tmp_path / name, *extra)) == 0
    log = _log(tmp_path / "both")
    names = [e["event"] for e in log["events"]]
    assert names.index("lora") < names.index("text_done") < names.index("stage_quantize") < names.index("quantized")
    assert log["events"][names.index("quantized")]["weights"] == 6 * 3
    source = json.loads(str(np.load(tmp_path / "both" / "final_rows.npz")["source"]))
    assert source["quantize"] == bits and source["lora"] == {"path": str(path), "sha256": _sha(path), "scale": 0.7}
    dense, both, quantized = (np.asarray(Image.open(tmp_path / n / "image.png"), dtype=np.float32) for n in runs)
    error, effect = np.abs(both - dense).mean(), np.abs(both - quantized).mean()
    print(f"{bits}-bit with the adapter: mean |diff| {error:.2f} / 255 from the unquantized run with it, "
          f"{effect:.2f} from the quantized run without it")
    assert error < (0.05 if bits == 8 else 0.5) and effect > 2.0


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
