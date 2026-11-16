"""`scripts/train_lora.py` and `h3mlx.trainer` on the tiny transformer: the arguments, the plan, a short real loop,
its limits, its record, its checkpoints and resuming.

The base is a tiny random transformer saved as a release folder. The cache is filled with synthetic tensors for the
requests the script itself computes, so no encoder is needed and none is loaded; the last test, which needs the
release's processor, encodes from pictures with the tiny encoders instead.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx, to_np
from PIL import Image
from test_dit_mixed_precision import to_mixed
from test_dit_tiny import TINY
from test_hyperflow import randomised
from test_p_h3_image import NEEDS_PROCESSOR, VAE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_lora as T  # noqa: E402

from h3mlx import cache, files, load, lora, train, trainer  # noqa: E402
from h3mlx.pipeline import Noise  # noqa: E402
from h3mlx.vae import VaeConfig  # noqa: E402

RATIO, MULTIPLE = 4, 8  # the tiny VAE's spatial ratio, and with the transformer's patch of 2 the canvas multiple
SETTINGS = dict(reference_min_pixels=32 * 32, reference_max_pixels=64 * 64, canvas_min_pixels=48 * 48, vision_pixels=48 * 48)
EXAMPLES = [
    {"picture": "a.png", "prompt": "a dog on a beach", "height": 48, "width": 40},
    {"picture": "b.png", "prompt": "make it night", "height": 40, "width": 56, "references": ["r.png"]},
]
PICTURES = {"a.png": (48, 40), "b.png": (40, 56), "r.png": (64, 96)}  # (height, width)
# every argument the script requires, as the tests give it; the three reference choices are required by EXAMPLES[1]
REQUIRED = {
    "--rank": ["4"], "--alpha": ["8"], "--targets": ["attn.to_q,attn.to_v,ff.net.2"], "--refiner": [],
    "--learning-rate": ["1e-3"], "--betas": ["0.9", "0.999"], "--eps": ["1e-8"], "--weight-decay": ["0.01"],
    "--max-grad-norm": ["0.08"], "--shift": ["3.0"], "--grid": ["1000"], "--density": ["uniform"],
    "--weighting": ["none"], "--condition-t": ["0.999"], "--seed": ["5"], "--latent-seed": ["11"],
    "--no-latent-float16-round": [], "--vae-dtype": ["float32"], "--text-dtype": ["float32"],
    "--max-steps": ["12"], "--max-seconds": ["600"], "--max-peak-memory-gb": ["64"], "--checkpoint-every": ["5"],
}
REFERENCE = {"--reference-sizing": ["recipe"], "--reference-seed": ["42"], "--no-reference-float16-round": []}


def make_release(folder: Path, mixed: bool = False) -> Path:
    """A release folder with what the script reads: the tiny transformer's weights and config, the VAE's config, the
    pipeline's settings, and a small file in each encoder's folder (their identities are part of the cache keys)."""
    model = randomised(TINY, 0)
    (to_mixed(model) if mixed else model).save_pretrained(str(folder / "transformer"))
    (folder / "vae").mkdir()
    (folder / "vae" / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLPH3Image", **VAE}))
    (folder / "modular_model_index.json").write_text(json.dumps(SETTINGS))
    for name in ("text_encoder", "text_connector", "tokenizer", "processor"):
        (folder / name).mkdir()
        (folder / name / "config.json").write_text(json.dumps({"stands for": name}))
    return folder


def make_dataset(folder: Path, examples=EXAMPLES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, (h, w) in PICTURES.items():
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(folder / name)
    (folder / "dataset.json").write_text(json.dumps({"examples": examples}))
    return folder / "dataset.json"


def arguments(release, dataset, cache_dir, run_dir, without=(), **changes) -> list[str]:
    """The command line: every required argument, with `changes` replacing values (`{"--seed": ["6"]}`; a value of
    None drops the flag) and `without` dropping flags."""
    given = {"--weights": [str(release)], "--dataset": [str(dataset)], "--cache-dir": [str(cache_dir)],
             "--run-dir": [str(run_dir)], **REQUIRED, **REFERENCE}
    given.update({k.replace("_", "-") if k.startswith("--") else k: v for k, v in changes.items()})
    out = []
    for flag, values in given.items():
        if flag not in without and values is not None:
            out += [flag, *values]
    return out


def fill_cache(argv, text_rows=9, seed=0) -> list[dict]:
    """Synthetic entries for exactly the requests this command line makes. Returns the requests."""
    args = T.parser().parse_args(argv)
    examples = T.read_dataset(Path(args.dataset))
    cfg, vcfg, settings = cache.release_configs(Path(args.weights))
    asked = T.requests(args, examples, Path(args.weights), settings, MULTIPLE)
    g = torch.Generator().manual_seed(seed)
    for example, request in zip(examples, asked):
        if cache.read(args.cache_dir, request) is not None:
            continue
        tensors = {"latents": mx.array(torch.randn(1, 1, example["height"] // RATIO, example["width"] // RATIO, 4, generator=g).numpy())}
        for i, reference in enumerate(example["references"]):
            choice = request["references"]
            (w, h), _ = cache.reference_sizes(choice["sizing"], *Image.open(reference).size, target_height=example["height"],
                                              target_width=example["width"], min_pixels=choice["min_pixels"],
                                              max_pixels=choice["max_pixels"], vision_pixels=choice["vision_pixels"],
                                              multiple=MULTIPLE)
            tensors[f"condition_latents.{i}"] = mx.array(torch.randn(1, 1, h // RATIO, w // RATIO, 4, generator=g).numpy())
        tensors["text"] = mx.array(torch.randn(1, text_rows, TINY["text_dim"], generator=g).numpy()).astype(cache.DTYPES[args.text_dtype])
        tensors["text_token_tags"] = mx.array(np.array([1] * text_rows, dtype=np.int64))
        cache.write(args.cache_dir, request, tensors)
    return asked


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return make_release(tmp_path_factory.mktemp("release"))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture
def work(release, data, tmp_path):
    """(argv builder, run directory, cache directory) with the cache filled for the default arguments."""
    def argv(run="run", without=(), **changes):
        return arguments(release, data, tmp_path / "cache", tmp_path / run, without, **changes)

    fill_cache(argv())
    return argv, tmp_path / "run", tmp_path / "cache"


def snapshot(run_dir) -> dict:
    """Every file under a run directory with its bytes, the lock file aside (it is rewritten by whoever holds it)."""
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(Path(run_dir).rglob("*"))
            if p.is_file() and p.name != trainer.LOCK}


def refuse_heavy(monkeypatch):
    """The loaders of the base weights and of the encoders raise, and so does the encode. (A run's own checkpoint,
    a few small files, may still be read: that is how a resume checks it.)"""
    def refuse(*a, **kw):
        raise AssertionError("something heavy was loaded")

    for owner, name in ((load, "load_safetensors"), (load, "load_transformer"), (load, "load_vae"),
                        (load, "load_text_encoder_states"), (load, "load_text_encoder"), (cache, "load_encoders"),
                        (cache, "encode"), (T, "encode_missing")):
        monkeypatch.setattr(owner, name, refuse)


def refuse_to_load(monkeypatch):
    """Every way the script and the package have of opening weights raises."""
    def refuse(*a, **kw):
        raise AssertionError("weights opened")

    for owner, name in ((mx, "load"), (load, "load_safetensors"), (load, "load_transformer"), (load, "load_vae"),
                        (load, "load_text_encoder_states"), (load, "load_text_encoder"), (lora, "load"), (cache, "encode"),
                        (cache, "load"), (T, "encode_missing")):
        monkeypatch.setattr(owner, name, refuse)


def record_of(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run.json").read_text())


def drawn(record) -> list:
    """The parts of a record's losses that a run determines (not the seconds, not the memory)."""
    return [(x["step"], x["example"], x["sigma"], x["weight"], x["loss"], x["grad_norm"]) for x in record["losses"]]


def adapter_bytes(run_dir) -> dict:
    return {k: to_np(v).tobytes() for k, v in mx.load(str(Path(run_dir) / "adapter.safetensors")).items()}


def run(argv, clock=time.monotonic) -> int:
    return T.main(argv, clock=clock)


# --- the arguments -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--weights", "--dataset", "--cache-dir", "--run-dir", *REQUIRED])
def test_a_missing_argument_is_an_error_that_names_it(work, capsys, flag):
    """Nothing that decides how the model is trained has a default: the shift, the grid, the density, the weighting,
    the learning rate, the clip, the rank, the alpha, the seed, the limits and the encode choices. Leave one out and
    the command stops with the argument's name, before anything is read or written."""
    argv, run_dir, _ = work
    with pytest.raises(SystemExit) as stop:
        run(argv(without=(flag,)))
    assert stop.value.code == 2
    named = {"--refiner": "--refiner/--no-refiner", "--no-latent-float16-round": "--latent-float16-round/--no-latent-float16-round"}
    assert f"the following arguments are required: {named.get(flag, flag)}" in capsys.readouterr().err
    assert not run_dir.exists()


def test_the_script_turns_the_hub_off_before_it_imports_anything():
    """The script must not download. Its first statement after the docstring sets `HF_HUB_OFFLINE`, ahead of every
    import that could bring in a library that reads it, and a process that runs the script has the hub off."""
    source = (ROOT / "scripts" / "train_lora.py").read_text()
    body = source.split('"""', 2)[2]
    statements = [line for line in body.splitlines() if line.strip() and not line.startswith("#")]
    assert statements[:2] == ["import os", 'os.environ["HF_HUB_OFFLINE"] = "1"']
    probe = ("import runpy, sys, os; sys.argv = ['train_lora.py', '--help']\n"
             "try:\n    runpy.run_path(sys.argv.pop(0) and %r, run_name='__main__')\n"
             "except SystemExit:\n    pass\n"
             "import huggingface_hub.constants as c; print('offline', os.environ['HF_HUB_OFFLINE'], c.HF_HUB_OFFLINE)" % str(ROOT / "scripts" / "train_lora.py"))
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env).stdout
    assert out.strip().endswith("offline 1 True"), out[-300:]


def test_only_the_backward_way_has_a_default():
    """Every option of the parser is required, but for: the optional assistant, the density's own parameters and the
    reference choices (required when they apply, checked below), the two modes, and how the gradient is taken, which
    defaults to block by block and changes no weights."""
    optional = {a.dest: a.default for a in T.parser()._actions if a.option_strings and not a.required and a.dest != "help"}
    assert optional == {"assistant": None, "logit_mean": None, "logit_std": None, "mode_scale": None,
                        "reference_sizing": None, "reference_seed": None, "reference_float16_round": None,
                        "plan_only": False, "resume": False, "backward": "blockwise"}


@pytest.mark.parametrize("changes,without,message", [
    ({"--density": ["logit_normal"]}, (), "--density logit_normal needs --logit-mean and --logit-std"),
    ({"--density": ["logit_normal"], "--logit-mean": ["0.0"]}, (), "--density logit_normal needs --logit-std"),
    ({"--density": ["mode"]}, (), "--density mode needs --mode-scale"),
    ({"--mode-scale": ["1.29"]}, (), "mode_scale is not used by density 'uniform'"),
    ({}, ("--reference-sizing",), "the dataset has reference pictures: --reference-sizing required"),
    ({}, ("--reference-seed", "--no-reference-float16-round"), "--reference-seed, --reference-float16-round required"),
    ({"--targets": ["attn.to_q,proj_in"]}, (), "targets must be some of"),
    ({"--rank": ["0"]}, (), "rank must be an integer of at least 1"),
    ({"--alpha": ["0"]}, (), "alpha"),
    ({"--learning-rate": ["0"]}, (), "learning_rate must be a finite number above 0"),
    ({"--learning-rate": ["nan"]}, (), "learning_rate must be a finite number"),
    ({"--betas": ["0.9", "1.0"]}, (), "betas must be two numbers in"),
    ({"--max-grad-norm": ["inf"]}, (), "max_grad_norm must be a finite number"),
    ({"--shift": ["0"]}, (), "shift must be positive and finite"),
    ({"--grid": ["0"]}, (), "grid must be an integer of at least 1"),
    ({"--condition-t": ["1.5"]}, (), "condition_t must be a number in"),
    ({"--seed": ["-1"]}, (), "seed must be an integer in"),
    ({"--max-steps": ["0"]}, (), "max_steps must be an integer of at least 1"),
    ({"--max-seconds": ["0"]}, (), "max_seconds must be a finite number above 0"),
    ({"--max-peak-memory-gb": ["inf"]}, (), "max_peak_memory_gb must be a finite number"),
    ({"--checkpoint-every": ["0"]}, (), "checkpoint_every must be an integer of at least 1"),
])
def test_arguments_that_do_not_make_a_run_are_refused(work, capsys, changes, without, message):
    argv, run_dir, _ = work
    with pytest.raises(SystemExit) as stop:
        run(argv(without=without, **changes))
    assert stop.value.code == 2 and message in capsys.readouterr().err and not run_dir.exists()


@pytest.mark.parametrize("content,message", [
    ("not json", "not a dataset file"), ('{"examples": []}', "must be a non-empty list"),
    ('{"pictures": []}', "not a dataset file"),
    ('{"examples": [{"picture": "a.png", "prompt": "x", "height": 48}]}', "example 0 must have picture, prompt, height and width"),
    ('{"examples": [{"picture": "a.png", "prompt": "x", "height": 48, "width": 40, "caption": "y"}]}', "example 0 must have"),
    ('{"examples": [{"picture": "nowhere.png", "prompt": "x", "height": 48, "width": 40}]}', "no such file"),
    ('{"examples": [{"picture": "a.png", "prompt": "x", "height": 50, "width": 40}]}', "height must be a positive multiple of 8"),
    ('{"examples": [{"picture": "a.png", "prompt": "", "height": 48, "width": 40}]}', "the prompt must be a string"),
])
def test_a_dataset_that_is_not_one_is_refused(work, data, tmp_path, capsys, content, message):
    argv, run_dir, _ = work
    bad = data.parent / f"bad-{abs(hash(content))}.json"
    bad.write_text(content)
    with pytest.raises(SystemExit) as stop:
        run(argv(**{"--dataset": [str(bad)]}))
    assert stop.value.code == 2 and message in capsys.readouterr().err and not run_dir.exists()


# --- the plan ----------------------------------------------------------------------------------------------------------


def test_plan_only_opens_no_weights_and_writes_nothing(work, release, data, tmp_path, capsys, monkeypatch):
    """`--plan-only` with every loader made to raise: it prints the plan and exits 0, the run directory is not
    created, and the cache is as it was. What it prints is checked against a real run of the same command line: the
    adapter's modules and parameter count, the adapter's size, the base weights' size, each example's rows, which
    examples the cache holds, the steps and the checkpoints."""
    argv, run_dir, cache_dir = work
    before = sorted(p.name for p in cache_dir.iterdir())
    with_a_miss = argv(**{"--dataset": [str(make_dataset(tmp_path / "more", EXAMPLES + [
        {"picture": "a.png", "prompt": "a cat this time", "height": 48, "width": 40}]))]})
    with monkeypatch.context() as patch:
        refuse_to_load(patch)
        assert run(with_a_miss + ["--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert not run_dir.exists() and sorted(p.name for p in cache_dir.iterdir()) == before

    assert plan["examples"]["count"] == 3 and plan["examples"]["to_encode"] == 1
    assert [e["cached"] for e in plan["examples"]["list"]] == [True, True, False]
    assert plan["steps"] == {"max_steps": 12, "passes_over_the_examples": 4.0, "checkpoint_every": 5, "checkpoints_at": [5, 10, 12]}
    assert plan["limits"] == {"max_steps": 12, "max_seconds": 600.0, "max_peak_memory_gb": 64.0, "checkpoint_every": 5}
    assert plan["hyper"]["shift"] == 3.0 and plan["hyper"]["grid"] == 1000 and plan["hyper"]["backward"] == "blockwise"
    assert plan["writes"]["record"] == str(run_dir / "run.json") and plan["cache"]["directory"] == str(cache_dir)
    assert plan["writes"]["final_adapter"] == str(run_dir / "adapter.safetensors")

    # against the real thing: run the two cached examples
    assert run(argv()) == 0
    capsys.readouterr()
    trained = mx.load(str(run_dir / "adapter.safetensors"))
    modules = {k.split(".lora_")[0] for k in trained}
    adapter = plan["adapter"]
    assert adapter["modules"] == len(modules) == 3 * (TINY["num_layers"] + TINY["num_refiner_layers"])
    assert adapter["rank"] == 4 and adapter["scaling"] == 2.0 and adapter["parameters"] == sum(v.size for v in trained.values())
    assert "transformer." + adapter["first_modules"][0] in modules and "transformer." + adapter["last_module"] in modules
    assert plan["sizes"]["adapter_bytes"] == sum(v.nbytes for v in trained.values())
    optimizer = mx.load(str(run_dir / "checkpoints" / "step-00000012" / "optimizer.safetensors"))
    assert plan["sizes"]["optimizer_state_bytes"] == sum(v.nbytes for k, v in optimizer.items() if k != "step")
    base = mx.load(str(release / "transformer" / "diffusion_pytorch_model.safetensors"))
    assert plan["sizes"]["base_weights_bytes"] == sum(v.nbytes for v in base.values())
    assert plan["sizes"]["base_weights_bytes_by_dtype"] == {"F32": plan["sizes"]["base_weights_bytes"]}
    asked = fill_cache(argv())
    for described, request in zip(plan["examples"]["list"], asked):
        entry = cache.read(cache_dir, request)
        example = train.build_example(TINY, entry.text_token_tags, entry.latents, 0.5, Noise(0), condition_t=0.999,
                                      condition_latents=entry.condition_latents)
        assert described["key"] == cache.key(request)
        assert described["sequence_rows"] == len(example.layout.token_tags)
        assert described["condition_rows"] == example.num_condition_rows and described["text_rows"] == 9
        assert described["video_rows"] == example.target.shape[0] and described["audio_rows"] == example.audio_rows.shape[0]
    missing = plan["examples"]["list"][2]
    assert missing["text_rows"] is None and missing["sequence_rows"] is None and missing["sequence_rows_without_text"] == 30 + 4
    largest = max(e["sequence_rows"] for e in plan["examples"]["list"][:2])
    assert plan["sizes"]["largest_sequence_rows"] == largest and plan["sizes"]["largest_sequence_is_without_text"] is True
    assert plan["sizes"]["saved_block_inputs_bytes"] == TINY["num_layers"] * largest * TINY["hidden_size"] * 4
    assert record_of(run_dir)["plan"]["adapter"] == {**adapter}


# --- a short real loop -------------------------------------------------------------------------------------------------


def fixed_loss(release, cache_dir, request, adapter_file=None) -> float:
    """One example's loss at three fixed noise levels and fixed noise, with the trained adapter or without."""
    base, cfg = load.load_transformer(release / "transformer")
    entry = cache.read(cache_dir, request)
    spec = lora.load(adapter_file) if adapter_file else lora.init(cfg, 4, 8.0, trainer.adapter_modules(cfg, ["attn.to_q"], False))
    frozen, trainable = lora.split(lora.attach(base, cfg, spec, keep_dtype=True, name="x"), "x")
    total = 0.0
    for i, sigma in enumerate((0.25, 0.5, 0.75)):
        example = train.build_example(cfg, entry.text_token_tags, entry.latents, sigma, Noise(100 + i), condition_t=0.999,
                                      condition_latents=entry.condition_latents)
        total += float(train.example_loss(trainable, frozen, cfg, example, entry.text))
    return total / 3


def test_the_loop_overfits_one_example(release, tmp_path, capsys):
    """One synthetic example, 150 steps at a learning rate of 2e-3 on the six matrices of each block: the example's
    loss at three fixed noise levels, with fixed noise, falls to under 0.8 of what the base model gives, and the
    record's own losses fall too (first 30 steps against the last 30). The adapter it writes is a LoRA file
    `lora.load` reads, with the rank and alpha asked for."""
    dataset = make_dataset(tmp_path / "one", EXAMPLES[:1])
    argv = arguments(release, dataset, tmp_path / "cache", tmp_path / "run", without=tuple(REFERENCE), **{
        "--targets": [",".join(lora.BLOCK_TARGETS)], "--learning-rate": ["2e-3"], "--max-steps": ["150"],
        "--checkpoint-every": ["150"], "--max-grad-norm": ["1.0"]})
    (request,) = fill_cache(argv)
    assert run(argv) == 0
    record = record_of(tmp_path / "run")
    assert record["status"] == "finished" and record["step"] == 150 and len(record["losses"]) == 150
    losses = [x["loss"] for x in record["losses"]]
    before, after = fixed_loss(release, tmp_path / "cache", request), fixed_loss(release, tmp_path / "cache", request, tmp_path / "run" / "adapter.safetensors")
    print(f"fixed loss {before:.4f} -> {after:.4f}; recorded losses, first 30 {np.mean(losses[:30]):.4f}, last 30 {np.mean(losses[-30:]):.4f}")
    assert after < 0.8 * before and np.mean(losses[-30:]) < 0.9 * np.mean(losses[:30])
    adapter = lora.load(tmp_path / "run" / "adapter.safetensors")
    assert sorted(adapter.modules) == sorted(trainer.adapter_modules(TINY, lora.BLOCK_TARGETS, True))
    assert all(adapter.rank(m) == 4 and adapter.alpha[m] == 8 for m in adapter.modules)
    assert all(float(mx.abs(v).max()) > 0 and v.dtype == mx.float32 for v in adapter.tensors.values())
    assert record["final_adapter"]["sha256"] == files.sha256_file(tmp_path / "run" / "adapter.safetensors")


def test_the_loop_is_the_pieces_in_order(release, data, tmp_path, capsys):
    """Six steps of the script, with every hyper-parameter at a value of its own, against the same six steps written
    out here from the pieces each tested elsewhere: the step's seeds, the level from the density and the grid, the
    example builder, the plain gradient (`train.loss_and_gradients`, which is torch's), AdamW and the clip. The
    trained matrices agree to 1e-7, and the recorded losses and norms are the hand loop's. A hyper-parameter that did
    not reach the loop, or reached the wrong place, would show here."""
    hyper = {"--learning-rate": ["3e-4"], "--betas": ["0.8", "0.95"], "--eps": ["1e-6"], "--weight-decay": ["0.1"],
             "--max-grad-norm": ["0.07"], "--shift": ["5.0"], "--grid": ["500"], "--density": ["mode"], "--mode-scale": ["1.29"],
             "--weighting": ["sigma_sqrt"], "--condition-t": ["0.9"], "--seed": ["9"], "--max-steps": ["6"], "--rank": ["3"], "--alpha": ["6"]}
    argv = arguments(release, data, tmp_path / "cache", tmp_path / "run", **hyper)
    asked = fill_cache(argv)
    assert run(argv) == 0
    record = record_of(tmp_path / "run")

    base, cfg = load.load_transformer(release / "transformer")
    entries = [cache.read(tmp_path / "cache", request) for request in asked]
    modules = trainer.adapter_modules(cfg, ["attn.to_q", "attn.to_v", "ff.net.2"], True)
    adapter = lora.init(cfg, 3, 6.0, modules, seed=trainer.step_seed(9, 0, "init"))
    frozen, trainable = lora.split(lora.attach(base, cfg, adapter, keep_dtype=True, name="hand"), "hand")
    optimizer = train.adamw(learning_rate=3e-4, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)
    grid = train.training_sigmas(500, 5.0)
    for step in range(6):
        entry = entries[trainer.example_at(9, step, 2)]
        u = train.density_for_timestep_sampling("mode", 1, mode_scale=1.29, generator=torch.Generator().manual_seed(trainer.step_seed(9, step, "level")))
        sigma = train.sample_sigmas(u, grid)[0][0]
        weight = float(train.loss_weighting_for_sd3("sigma_sqrt", np.float32([sigma]))[0])
        example = train.build_example(cfg, entry.text_token_tags, entry.latents, sigma, Noise(trainer.step_seed(9, step, "noise")),
                                      condition_t=0.9, condition_latents=entry.condition_latents)
        loss, grads = train.loss_and_gradients(trainable, frozen, cfg, example, entry.text, weight)
        trainable, norm = train.optimizer_step(optimizer, trainable, grads, max_grad_norm=0.07)
        logged = record["losses"][step]
        assert logged["sigma"] == float(sigma) and abs(logged["weight"] - weight) < 1e-9
        assert abs(logged["loss"] - float(loss)) <= 1e-6 * float(loss) and abs(logged["grad_norm"] - norm) <= 1e-6 * norm
    assert any(x["grad_norm"] > 0.07 for x in record["losses"]) and any(x["grad_norm"] < 0.07 for x in record["losses"])
    trained = lora.load(tmp_path / "run" / "adapter.safetensors")
    by_hand = lora.file_tensors(trainable, "hand")
    assert set(trained.tensors) == set(by_hand)
    worst = max(float(mx.abs(trained.tensors[k] - by_hand[k]).max()) for k in by_hand)
    moved = max(float(mx.abs(by_hand[k] - adapter.tensors[k]).max()) for k in by_hand)
    assert worst < 1e-7 and moved > 1e-3, (worst, moved)


def test_the_record_holds_the_run(work, capsys):
    """`run.json` after a finished run: every argument, the versions, the seed, the step reached and a loss per step
    with the example, the level, the gradient's norm, the seconds and MLX's peak memory. The examples come in a new
    order each pass and each is used once a pass; a step's level and noise are the step's own."""
    argv, run_dir, cache_dir = work
    assert run(argv()) == 0
    record = record_of(run_dir)
    assert record["format"] == trainer.RECORD and record["status"] == "finished" and record["stop_reason"] == "max_steps"
    assert record["step"] == 12 and record["seed"] == 5 and record["checkpoints"] == [5, 10, 12]
    assert record["versions"] == cache.versions() == record["trajectory"]["versions"]
    hyper = record["trajectory"]["hyper"]
    assert hyper == {"learning_rate": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "max_grad_norm": 0.08,
                     "shift": 3.0, "grid": 1000, "density": {"scheme": "uniform", "logit_mean": None, "logit_std": None, "mode_scale": None},
                     "weighting": "none", "condition_t": 0.999, "seed": 5, "backward": "blockwise", "audio_level": "the picture's"}
    assert record["trajectory"]["adapter"] == {"name": "trained", "rank": 4, "alpha": 8.0, "targets": ["attn.to_q", "attn.to_v", "ff.net.2"],
                                               "refiner": True, "modules": 9, "parameters": record["plan"]["adapter"]["parameters"]}
    assert record["trajectory"]["dataset"] == record["plan"]["cache"]["keys"] and len(record["trajectory"]["dataset"]) == 2
    assert record["limits"] == {"max_steps": 12, "max_seconds": 600.0, "max_peak_memory_gb": 64.0, "checkpoint_every": 5}
    (session,) = record["sessions"]
    assert session["command"] == argv() and session["from_step"] == 0 and session["to_step"] == 12 and not session["resumed"]
    assert set(record["losses"][0]) == {"step", "example", "sigma", "weight", "loss", "grad_norm", "seconds", "peak_memory_gb"}
    assert [x["step"] for x in record["losses"]] == list(range(1, 13))
    grid = train.training_sigmas(1000, 3.0)
    for x in record["losses"]:
        assert np.float32(x["sigma"]) in grid and x["loss"] > 0 and x["grad_norm"] > 0 and x["weight"] is None
    order = [x["example"] for x in record["losses"]]
    assert all(sorted(order[i:i + 2]) == [0, 1] for i in range(0, 12, 2)), "each example once a pass"
    assert len({tuple(order[i:i + 2]) for i in range(0, 12, 2)}) == 2, "and not in the same order every pass"
    assert len({x["sigma"] for x in record["losses"]}) == 12
    clipped = [x["grad_norm"] > 0.08 for x in record["losses"]]
    assert any(clipped) and not all(clipped), "the clip is on for some of these steps"


def test_weighting_and_density_reach_the_loop(work, capsys):
    """Another density and a weighting change the levels drawn and the loss weights, and so the weights trained."""
    argv, run_dir, _ = work
    assert run(argv()) == 0
    changed = argv(run="other", **{"--density": ["logit_normal"], "--logit-mean": ["0.0"], "--logit-std": ["1.0"], "--weighting": ["cosmap"]})
    assert run(changed) == 0
    plain, other = record_of(run_dir), record_of(run_dir.parent / "other")
    assert [x["sigma"] for x in plain["losses"]] != [x["sigma"] for x in other["losses"]]
    for x in other["losses"]:
        s = np.float32(x["sigma"])
        assert abs(x["weight"] - float(train.loss_weighting_for_sd3("cosmap", np.float32([s]))[0])) < 1e-6
    assert adapter_bytes(run_dir) != adapter_bytes(run_dir.parent / "other")


def test_the_two_bounded_gradients_train_the_same_weights(work, capsys):
    argv, run_dir, _ = work
    assert run(argv()) == 0 and run(argv(run="checkpointed", **{"--backward": ["checkpointed"]})) == 0
    a, b = mx.load(str(run_dir / "adapter.safetensors")), mx.load(str(run_dir.parent / "checkpointed" / "adapter.safetensors"))
    assert set(a) == set(b)
    worst = max(float(mx.abs(a[k] - b[k]).max()) for k in a)
    assert worst < 1e-6, worst
    assert record_of(run_dir.parent / "checkpointed")["trajectory"]["hyper"]["backward"] == "checkpointed"


def test_a_frozen_assistant_is_used_and_not_saved(work, release, tmp_path, capsys):
    """With `--assistant`, a second adapter is attached under its own name, frozen: the trained adapter comes out
    different, the file written holds the trained adapter alone, and the record names the assistant's digest."""
    argv, run_dir, _ = work
    cfg = load.load_transformer(release / "transformer")[1]
    g = torch.Generator().manual_seed(3)
    modules = lora.block_modules(cfg)
    shapes = lora.wired_shapes(cfg)
    tensors = {}
    for m in modules:
        tensors[m + ".lora_A.weight"] = mx.array((torch.randn(4, shapes[m][1], generator=g) * 0.2).numpy())
        tensors[m + ".lora_B.weight"] = mx.array((torch.randn(shapes[m][0], 4, generator=g) * 0.2).numpy())
    lora.save(tmp_path / "assistant.safetensors", lora.Adapter(tensors, {m: 4 for m in modules}))
    assert run(argv()) == 0 and run(argv(run="assisted", **{"--assistant": [str(tmp_path / "assistant.safetensors")]})) == 0
    plain, assisted = adapter_bytes(run_dir), adapter_bytes(tmp_path / "assisted")
    assert set(plain) == set(assisted) and plain != assisted
    assert record_of(tmp_path / "assisted")["trajectory"]["assistant"] == files.sha256_file(tmp_path / "assistant.safetensors")
    assert record_of(run_dir)["trajectory"]["assistant"] is None
    with pytest.raises(SystemExit):  # the assistant is part of the trajectory: a resume without it is another run
        run(argv(run="assisted") + ["--resume"])
    assert "assistant: asked None, the record has" in capsys.readouterr().err


def test_a_bf16_base_trains(tmp_path, data, capsys):
    """The setting a real run has: the base in the release's mixed precision, the trained matrices float32."""
    mixed = make_release(tmp_path / "mixed", mixed=True)
    argv = arguments(mixed, data, tmp_path / "cache", tmp_path / "run")
    fill_cache(argv)
    assert run(argv + ["--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert set(plan["sizes"]["base_weights_bytes_by_dtype"]) == {"BF16", "F32"}
    largest = plan["sizes"]["largest_sequence_rows"]
    assert plan["sizes"]["saved_block_inputs_bytes"] == TINY["num_layers"] * largest * TINY["hidden_size"] * 2
    assert run(argv) == 0
    trained = mx.load(str(tmp_path / "run" / "adapter.safetensors"))
    assert all(v.dtype == mx.float32 and float(mx.abs(v).max()) > 0 for v in trained.values())


# --- limits ------------------------------------------------------------------------------------------------------------


class Clock:
    """A clock that moves only when read: one second a reading."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 1.0
        return self.now


def test_the_wall_clock_stops_the_session_and_the_run_resumes(work, capsys):
    """With a clock that advances a second every time it is read and a limit of 20 seconds, the session ends before
    its twelfth step: exit code 3, the record says the wall clock stopped it and at which step, a checkpoint is
    written there, and no final adapter. Resumed with room, the run finishes with the weights of a run that was never
    stopped."""
    argv, run_dir, _ = work
    assert run(argv(run="straight")) == 0
    assert run(argv(**{"--max-seconds": ["20"]}), clock=Clock()) == trainer.STOPPED == 3
    record = record_of(run_dir)
    reached = record["step"]
    assert record["status"] == "stopped" and record["stop_reason"] == "wall_clock" and 0 < reached < 12
    assert "over the limit of 20.0 s" in record["stop_detail"] and len(record["losses"]) == reached
    assert record["checkpoints"][-1] == reached and (run_dir / "checkpoints" / f"step-{reached:08d}" / "checkpoint.json").is_file()
    assert not (run_dir / "adapter.safetensors").exists()
    assert run(argv() + ["--resume"]) == 0
    resumed = record_of(run_dir)
    assert resumed["status"] == "finished" and resumed["step"] == 12 and len(resumed["sessions"]) == 2
    assert resumed["sessions"][1]["from_step"] == reached and resumed["sessions"][1]["resumed"] is True
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(resumed) == drawn(record_of(run_dir.parent / "straight"))


def test_the_memory_limit_stops_the_session(work, capsys, monkeypatch):
    """The limit is compared with MLX's own peak-memory counter. A limit below what loading takes stops the run before
    its first step. A counter that goes over the limit in the middle of the fourth step's backward pass (after an
    evaluated block) ends the session at step 3: the step in progress is dropped, a checkpoint is written at 3, the
    exit code is 3, and the weights at that checkpoint are the uninterrupted run's after three steps."""
    argv, run_dir, _ = work
    assert run(argv(run="tight", **{"--max-peak-memory-gb": ["1e-9"]})) == 3
    tight = record_of(run_dir.parent / "tight")
    assert tight["status"] == "stopped" and tight["stop_reason"] == "peak_memory" and tight["step"] == 0
    assert "MLX's peak memory is" in tight["stop_detail"] and tight["checkpoints"] == [] and tight["losses"] == []

    assert run(argv(run="three", **{"--max-steps": ["3"]})) == 0
    readings = {"count": 0}
    real = mx.get_peak_memory
    stages_a_step = 2 * TINY["num_layers"] + 2  # the block-by-block gradient looks after each of these, then the loop once

    def counter():
        readings["count"] += 1
        over = readings["count"] > 1 + 3 * (stages_a_step + 1) + 2  # the start, three whole steps, two stages of the fourth
        return int(65e9) if over else real()

    monkeypatch.setattr(mx, "get_peak_memory", counter)
    assert run(argv()) == 3
    monkeypatch.undo()
    record = record_of(run_dir)
    assert record["status"] == "stopped" and record["stop_reason"] == "peak_memory" and record["step"] == 3
    assert "65.000 GB, over the limit of 64.0 GB" in record["stop_detail"] and record["checkpoints"] == [3]
    stopped_at = mx.load(str(run_dir / "checkpoints" / "step-00000003" / "adapter.safetensors"))
    three = mx.load(str(run_dir.parent / "three" / "adapter.safetensors"))
    assert all(np.array_equal(to_np(stopped_at[k]), to_np(three[k])) for k in three)


def test_max_steps_finishes_the_run_and_a_resume_can_extend_it(work, capsys):
    """`--max-steps` is where a run is finished (exit code 0). It is a limit, not part of the trajectory: resuming a
    finished run with a higher one goes on, to the weights of a run given the higher one from the start."""
    argv, run_dir, _ = work
    assert run(argv(run="straight")) == 0
    assert run(argv(**{"--max-steps": ["7"]})) == 0
    first = record_of(run_dir)
    assert first["status"] == "finished" and first["step"] == 7 and first["checkpoints"] == [5, 7]
    assert run(argv() + ["--resume"]) == 0
    assert record_of(run_dir)["step"] == 12 and record_of(run_dir)["checkpoints"] == [5, 7, 10, 12]
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert run(argv() + ["--resume"]) == 0, "resuming a finished run at its last step does nothing and says finished"
    assert record_of(run_dir)["step"] == 12 and adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")


# --- stopping in the middle, and resuming --------------------------------------------------------------------------------


def test_a_run_that_dies_resumes_to_the_uninterrupted_runs_weights(work, capsys, monkeypatch):
    """A run that fails in its eighth step has a checkpoint at 5 and a record that reached 7. Resumed, it goes back
    to the checkpoint, says which steps it discards, and ends with the same adapter, byte for byte, the same
    optimizer state and the same loss at every step as a run that was never interrupted."""
    argv, run_dir, _ = work
    assert run(argv(run="straight")) == 0
    calls = {"count": 0}
    real = train.optimizer_step

    def dies(*a, **kw):
        calls["count"] += 1
        if calls["count"] == 8:
            raise RuntimeError("the power went")
        return real(*a, **kw)

    monkeypatch.setattr(train, "optimizer_step", dies)
    with pytest.raises(RuntimeError, match="the power went"):
        run(argv())
    monkeypatch.undo()
    broken = record_of(run_dir)
    assert broken["status"] == "error" and "the power went" in broken["stop_reason"] and broken["step"] == 7
    assert broken["checkpoints"] == [5] and not (run_dir / "adapter.safetensors").exists()

    assert run(argv() + ["--resume"]) == 0
    resumed, straight = record_of(run_dir), record_of(run_dir.parent / "straight")
    assert resumed["sessions"][1]["from_step"] == 5 and resumed["sessions"][1]["discarded_steps"] == 2
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(resumed) == drawn(straight) and resumed["checkpoints"] == [5, 10, 12]
    for step in (10, 12):
        ours = mx.load(str(run_dir / "checkpoints" / f"step-{step:08d}" / "optimizer.safetensors"))
        theirs = mx.load(str(run_dir.parent / "straight" / "checkpoints" / f"step-{step:08d}" / "optimizer.safetensors"))
        assert set(ours) == set(theirs) and all(np.array_equal(to_np(ours[k]), to_np(theirs[k])) for k in ours)
        assert int(ours["step"].item()) == step


def test_a_killed_process_resumes_to_the_same_weights(work, capsys):
    """The same with a real kill: the script as a process of its own, sent SIGKILL once its record shows it is under
    way, at whatever point that lands (in a step, in a checkpoint's write, in the record's). Resumed, it ends with the
    uninterrupted run's adapter, byte for byte. Leftover temporary files are not taken for checkpoints."""
    argv, run_dir, _ = work
    long = {"--max-steps": ["400"], "--checkpoint-every": ["7"]}
    assert run(argv(run="straight", **long)) == 0
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "train_lora.py"), *argv(**long)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    reached = 0
    while time.time() < deadline and process.poll() is None:
        try:
            reached = record_of(run_dir)["step"]
        except (OSError, ValueError):
            reached = 0  # not there yet; a record is never half written, so a ValueError would be a failure of that
        if reached >= 20:
            break
        time.sleep(0.005)
    process.send_signal(signal.SIGKILL)
    assert process.wait() == -signal.SIGKILL, "the process was still training when it was killed"
    killed = record_of(run_dir)
    assert killed["status"] == "running" and 20 <= killed["step"] < 400
    assert not (run_dir / "adapter.safetensors").exists()

    assert run(argv(**long) + ["--resume"]) == 0
    resumed = record_of(run_dir)
    assert resumed["status"] == "finished" and resumed["step"] == 400
    start = resumed["sessions"][1]["from_step"]
    assert start % 7 == 0 and 0 < start <= killed["step"] and resumed["sessions"][1]["discarded_steps"] == killed["step"] - start
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(resumed) == drawn(record_of(run_dir.parent / "straight"))
    assert not [p for p in (run_dir / "checkpoints").iterdir() if p.name.startswith(files.TMP)]


@pytest.mark.parametrize("what,changes,message", [
    ("the learning rate", {"--learning-rate": ["2e-3"]}, "hyper.learning_rate: asked 0.002, the record has 0.001"),
    ("the seed", {"--seed": ["6"]}, "hyper.seed: asked 6, the record has 5"),
    ("the rank", {"--rank": ["2"]}, "adapter.rank: asked 2, the record has 4"),
    ("the clip", {"--max-grad-norm": ["1.0"]}, "hyper.max_grad_norm: asked 1.0, the record has 0.08"),
    ("the shift", {"--shift": ["12.0"]}, "hyper.shift: asked 12.0, the record has 3.0"),
    ("the density", {"--density": ["mode"], "--mode-scale": ["1.29"]}, "hyper.density.scheme: asked 'mode'"),
    ("the targets", {"--targets": ["attn.to_q"]}, "adapter.targets: asked"),
    ("the way back", {"--backward": ["checkpointed"]}, "hyper.backward: asked 'checkpointed', the record has 'blockwise'"),
    ("an encode choice", {"--latent-seed": ["12"]}, "dataset: asked"),
])
def test_a_resume_that_is_another_run_is_refused(work, capsys, what, changes, message):
    """`--resume` goes on with a run only if everything its weights depend on is what the record says: the
    hyper-parameters, the adapter, the seed, the examples (their cache keys, so every encode choice too). The
    refusal names what differs, and the run directory is left as it was."""
    argv, run_dir, _ = work
    assert run(argv(**{"--max-steps": ["6"]})) == 0
    fill_cache(argv(**changes))
    capsys.readouterr()
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit) as stop:
        run(argv(**changes) + ["--resume"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "is another run" in err and message in err
    assert {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == before


def test_a_resume_under_other_code_or_into_nothing_is_refused(work, capsys, monkeypatch):
    argv, run_dir, cache_dir = work
    with pytest.raises(SystemExit):
        run(argv() + ["--resume"])
    assert "no readable run record to resume from" in capsys.readouterr().err and not (run_dir / "run.json").exists()
    assert run(argv(**{"--max-steps": ["6"]})) == 0
    with pytest.raises(SystemExit):  # a run directory in use, without --resume
        run(argv())
    assert "already holds run.json, adapter.safetensors, checkpoints" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(argv() + ["--resume", "--plan-only"])
    assert "--plan-only and --resume do not go together" in capsys.readouterr().err

    # another version of the code: the examples' keys change with it (so the cache must be filled again), and the
    # record's versions no longer match
    with monkeypatch.context() as patch:
        patch.setattr(cache, "code_digest", lambda package=None: "f" * 64)
        fill_cache(argv())
        with pytest.raises(SystemExit):
            run(argv() + ["--resume"])
    err = capsys.readouterr().err
    assert "is another run" in err and "versions.h3mlx: asked 'ffff" in err

    (run_dir / "run.json").write_text(json.dumps({**record_of(run_dir), "format": "something else"}))
    with pytest.raises(SystemExit):
        run(argv() + ["--resume"])
    assert "is not a h3mlx-training-run-1 record" in capsys.readouterr().err


@pytest.mark.parametrize("damage,message", [
    ("a byte of the adapter", "adapter.safetensors is missing or is not the file the checkpoint describes"),
    ("the optimizer file gone", "optimizer.safetensors is missing or is not the file the checkpoint describes"),
    ("the checkpoint's record gone", "the checkpoint's record cannot be read"),
    ("another run's checkpoint", "belongs to another run or another step"),
    ("a renamed step", "belongs to another run or another step"),
    ("fewer losses than steps", "10 steps but 9 losses"),
])
def test_a_damaged_checkpoint_is_refused_not_skipped(work, capsys, monkeypatch, damage, message):
    """The latest checkpoint is the one a resume uses. If it is not whole, or not this run's, the resume stops and
    says so; it does not quietly fall back to an older one. The refusal comes before anything heavy and changes
    nothing: no weights are loaded, no example is encoded, and the run directory, a finished run's record included,
    is byte for byte what it was."""
    argv, run_dir, _ = work
    assert run(argv(**{"--max-steps": ["10"]})) == 0
    assert record_of(run_dir)["status"] == "finished"
    latest = run_dir / "checkpoints" / "step-00000010"
    described = json.loads((latest / "checkpoint.json").read_text())
    if damage == "a byte of the adapter":
        data = bytearray((latest / "adapter.safetensors").read_bytes())
        data[-1] ^= 1
        (latest / "adapter.safetensors").write_bytes(bytes(data))
    elif damage == "the optimizer file gone":
        (latest / "optimizer.safetensors").unlink()
    elif damage == "the checkpoint's record gone":
        (latest / "checkpoint.json").unlink()
    elif damage == "another run's checkpoint":
        (latest / "checkpoint.json").write_text(json.dumps({**described, "trajectory": "0" * 64}))
    elif damage == "a renamed step":
        latest.rename(run_dir / "checkpoints" / "step-00000011")
    elif damage == "fewer losses than steps":
        (latest / "checkpoint.json").write_text(json.dumps({**described, "losses": described["losses"][:9]}))
    before = snapshot(run_dir)
    refuse_heavy(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        run(argv() + ["--resume"])
    assert stop.value.code == 2 and message in capsys.readouterr().err
    assert snapshot(run_dir) == before


def test_a_resume_goes_on_from_the_records_latest_checkpoint_or_not_at_all(work, capsys, monkeypatch):
    """The record lists the checkpoints it wrote. A resume that finds less on disk than the record's latest is
    refused: with the newest checkpoint folder gone it does not roll back to an older one, and with the whole
    checkpoints folder gone it does not start again from step 0. Nothing is written by either refusal. More on disk
    than the record lists is the trace of a run stopped between a checkpoint and its record, and is resumed from."""
    argv, run_dir, _ = work
    assert run(argv(run="straight")) == 0 and run(argv(**{"--max-steps": ["11"]})) == 0
    assert record_of(run_dir)["checkpoints"] == [5, 10, 11]
    aside = run_dir.parent / "aside"
    shutil.move(run_dir / "checkpoints" / "step-00000011", aside)
    before = snapshot(run_dir)
    with pytest.raises(SystemExit) as stop:
        run(argv() + ["--resume"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "lists a checkpoint at step 11" in err and "is step 10" in err
    assert snapshot(run_dir) == before

    shutil.move(run_dir / "checkpoints", run_dir.parent / "all-aside")
    before = snapshot(run_dir)
    with pytest.raises(SystemExit) as stop:
        run(argv() + ["--resume"])
    err = capsys.readouterr().err
    assert stop.value.code == 2 and "lists a checkpoint at step 11" in err and "is none" in err
    assert snapshot(run_dir) == before
    shutil.move(run_dir.parent / "all-aside", run_dir / "checkpoints")
    shutil.move(aside, run_dir / "checkpoints" / "step-00000011")

    # the other way round: the record behind the disk (a run stopped after its checkpoint's rename, before its record)
    record = record_of(run_dir)
    behind = {**record, "status": "running", "step": 9, "losses": record["losses"][:9], "checkpoints": [5]}
    (run_dir / "run.json").write_text(json.dumps(behind))
    (run_dir / "adapter.safetensors").unlink()
    assert run(argv() + ["--resume"]) == 0
    resumed = record_of(run_dir)
    assert resumed["sessions"][-1]["from_step"] == 11 and resumed["step"] == 12 and len(resumed["losses"]) == 12
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(resumed) == drawn(record_of(run_dir.parent / "straight"))

    # and a run that died before its first checkpoint goes on from its start, which is where its record says its
    # last checkpoint is
    calls = {"count": 0}
    real = train.optimizer_step

    def dies(*a, **kw):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("the power went")
        return real(*a, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(train, "optimizer_step", dies)
        with pytest.raises(RuntimeError):
            run(argv(run="young", **{"--checkpoint-every": ["50"]}))
    young = record_of(run_dir.parent / "young")
    assert young["step"] == 2 and young["checkpoints"] == [] and not (run_dir.parent / "young" / "checkpoints").exists()
    assert run(argv(run="young", **{"--checkpoint-every": ["50"]}) + ["--resume"]) == 0
    again = record_of(run_dir.parent / "young")
    assert again["sessions"][1]["from_step"] == 0 and again["sessions"][1]["discarded_steps"] == 2
    assert adapter_bytes(run_dir.parent / "young") == adapter_bytes(run_dir.parent / "straight")


def test_a_resume_past_its_max_steps_is_refused(work, capsys):
    """A run at step 12 resumed with a `--max-steps` of 5 has nothing to do. It is refused, and nothing is written:
    not a record saying step 12 beside a limit of 5, and not the adapter again."""
    argv, run_dir, _ = work
    assert run(argv()) == 0
    before = snapshot(run_dir)
    with pytest.raises(SystemExit) as stop:
        run(argv(**{"--max-steps": ["5"]}) + ["--resume"])
    assert stop.value.code == 2 and "the run is at step 12, past a max_steps of 5" in capsys.readouterr().err
    assert snapshot(run_dir) == before


def test_a_refused_resume_encodes_nothing(release, data, tmp_path, capsys, monkeypatch):
    """The run directory is checked before the cache is filled: a `--resume` into a directory with no run, or with
    another run's record, is refused with the encoders never loaded, even when every example is missing from the
    cache."""
    filled = arguments(release, data, tmp_path / "cache", tmp_path / "run", **{"--max-steps": ["3"]})
    fill_cache(filled)
    assert run(filled) == 0
    capsys.readouterr()
    refuse_heavy(monkeypatch)
    empty = tmp_path / "empty-cache"
    for run_dir, changes, message in ((tmp_path / "no-such-run", {}, "no readable run record to resume from"),
                                      (tmp_path / "run", {"--latent-seed": ["12"]}, "is another run")):
        with pytest.raises(SystemExit) as stop:
            run(arguments(release, data, empty, run_dir, **{"--max-steps": ["3"], **changes}) + ["--resume"])
        assert stop.value.code == 2 and message in capsys.readouterr().err
    assert not empty.exists() and not (tmp_path / "no-such-run").exists()


# --- who owns a run directory ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("there", ["adapter.safetensors", "checkpoints", "run.json"])
def test_a_new_run_overwrites_nothing(work, capsys, there):
    """A new run writes `run.json`, `adapter.safetensors` and `checkpoints/` in its directory. If any of the three is
    already there the run is refused before it reads a weight, whether or not there is a record: a directory holding
    only someone's adapter is not an empty run directory. Other files are not the run's and do not stop it."""
    argv, run_dir, _ = work
    run_dir.mkdir()
    (run_dir / "notes.txt").write_text("the arguments I used")
    assert trainer.OUTPUTS == ("run.json", "adapter.safetensors", "checkpoints")
    if there == "checkpoints":
        (run_dir / there).mkdir()
    else:
        (run_dir / there).write_bytes(b"someone's file, not this run's")
    before = snapshot(run_dir)
    with pytest.raises(SystemExit) as stop:
        run(argv())
    err = capsys.readouterr().err
    assert stop.value.code == 2 and f"already holds {there}" in err and "overwrites none of them" in err
    assert snapshot(run_dir) == before
    (run_dir / there).rmdir() if there == "checkpoints" else (run_dir / there).unlink()
    assert run(argv()) == 0 and (run_dir / "notes.txt").read_text() == "the arguments I used"


def test_one_session_in_a_run_directory_at_a_time(work, capsys, monkeypatch):
    """While one session has a run directory, a second is refused and told who holds it, and it touches nothing:
    in particular not the first one's checkpoint that is written and about to be renamed into place. Here the second
    session is started at exactly that moment, from inside the first one's rename; the first then finishes as if
    nothing had happened. Once it is over the directory can be taken again."""
    argv, run_dir, _ = work
    assert run(argv(run="straight", **{"--max-steps": ["2"], "--checkpoint-every": ["1"]})) == 0
    assert run(argv(**{"--max-steps": ["1"], "--checkpoint-every": ["1"]})) == 0
    real = os.rename
    seen = {}

    def rename_while_a_second_session_starts(src, dst):
        if Path(dst).name == "step-00000002" and not seen:
            seen["temporary"] = Path(src).is_dir()
            seen["holder"] = files.lock_holder(run_dir / trainer.LOCK)
            with pytest.raises(SystemExit) as stop:
                run(argv(**{"--max-steps": ["2"], "--checkpoint-every": ["1"]}) + ["--resume"])
            seen["second"] = (stop.value.code, capsys.readouterr().err)
            seen["temporary_after"] = Path(src).is_dir()
        return real(src, dst)

    monkeypatch.setattr(os, "rename", rename_while_a_second_session_starts)
    assert run(argv(**{"--max-steps": ["2"], "--checkpoint-every": ["1"]}) + ["--resume"]) == 0
    monkeypatch.undo()
    code, err = seen["second"]
    assert code == 2 and "the run directory is in use" in err and f"held by process {os.getpid()} (training in run" in err
    assert seen["temporary"] and seen["temporary_after"], "the first session's checkpoint in progress was left alone"
    assert seen["holder"]["pid"] == os.getpid() and seen["holder"]["what"] == "training in run"
    assert record_of(run_dir)["status"] == "finished" and record_of(run_dir)["checkpoints"] == [1, 2]
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert files.lock_holder(run_dir / trainer.LOCK) is None, "released: the lock file says nobody holds it"
    assert run(argv(**{"--max-steps": ["2"], "--checkpoint-every": ["1"]}) + ["--resume"]) == 0

    # held by something else entirely: a new run is refused too, and leaves no outputs
    fresh = run_dir.parent / "fresh"
    fresh.mkdir()
    with files.exclusive(fresh / trainer.LOCK, "something else"):
        with pytest.raises(SystemExit):
            run(argv(run="fresh"))
        assert "held by process" in capsys.readouterr().err and "(something else" not in ""
    assert sorted(p.name for p in fresh.iterdir()) == [trainer.LOCK]


def test_a_second_process_is_refused_and_sigterm_stops_the_first_cleanly(work, capsys):
    """Two real processes. The script runs as a process of its own; once its record shows it is under way, a resume
    of the same run from this process is refused, naming the other process's id, and changes nothing. The first
    process is then sent SIGTERM: it stops at the next block or step, writes a checkpoint at its last completed step,
    records that it was terminated and exits with code 3. With it gone the lock is free, and the run resumes to the
    uninterrupted run's adapter, byte for byte."""
    argv, run_dir, _ = work
    long = {"--max-steps": ["400"], "--checkpoint-every": ["50"]}
    assert run(argv(run="straight", **long)) == 0
    capsys.readouterr()
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "train_lora.py"), *argv(**long)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 120
        while time.time() < deadline and process.poll() is None:
            try:
                if record_of(run_dir)["step"] >= 20:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.005)
        assert process.poll() is None, "the first process is still training"
        with pytest.raises(SystemExit) as stop:
            run(argv(**long) + ["--resume"])
        err = capsys.readouterr().err
        assert stop.value.code == 2 and f"held by process {process.pid} (training in run" in err
        assert process.poll() is None, "and it was not disturbed"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == trainer.STOPPED == 3
    finally:
        if process.poll() is None:
            process.kill()
    stopped = record_of(run_dir)
    reached = stopped["step"]
    assert stopped["status"] == "stopped" and stopped["stop_reason"] == "terminated" and 20 <= reached < 400
    assert "SIGTERM" in stopped["stop_detail"] and stopped["checkpoints"][-1] == reached and len(stopped["losses"]) == reached
    assert (run_dir / "checkpoints" / f"step-{reached:08d}" / "checkpoint.json").is_file()
    assert not (run_dir / "adapter.safetensors").exists() and stopped["sessions"][0]["stopped_by"] == "terminated"

    assert run(argv(**long) + ["--resume"]) == 0
    resumed = record_of(run_dir)
    assert resumed["step"] == 400 and resumed["sessions"][1]["from_step"] == reached and resumed["sessions"][1]["discarded_steps"] == 0
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(resumed) == drawn(record_of(run_dir.parent / "straight"))


def test_sigterm_stops_at_the_next_boundary_with_a_checkpoint(work, capsys, monkeypatch):
    """SIGTERM delivered to this process during the fourth step's update: the step that was under way when it came
    is completed (the signal is looked at after an evaluated block and after a step, not in the middle of an
    update), the session ends there with a checkpoint at step 4, status `stopped`, reason `terminated`, exit code 3,
    and SIGTERM's handler is what it was before the session. A second signal, in the middle of a backward pass,
    drops that step. Resumed, the run ends where an uninterrupted one does."""
    argv, run_dir, _ = work
    assert run(argv(run="straight")) == 0
    before = signal.getsignal(signal.SIGTERM)
    calls = {"count": 0}
    real = train.optimizer_step

    def signalled(*a, **kw):
        calls["count"] += 1
        if calls["count"] == 4:
            os.kill(os.getpid(), signal.SIGTERM)
        return real(*a, **kw)

    monkeypatch.setattr(train, "optimizer_step", signalled)
    assert run(argv()) == 3
    monkeypatch.undo()
    assert signal.getsignal(signal.SIGTERM) is before
    record = record_of(run_dir)
    assert record["status"] == "stopped" and record["stop_reason"] == "terminated" and record["step"] == 4
    assert record["checkpoints"] == [4] and not (run_dir / "adapter.safetensors").exists()

    # in the middle of the sixth step's backward pass: that step is dropped, the session ends at 5
    stages = {"count": 0}
    real_blockwise = trainer.BACKWARD["blockwise"]

    def signalled_mid_step(*a, after_stage=None, **kw):
        stages["count"] += 1

        def stage():
            if stages["count"] == 2 and not stages.get("sent"):
                stages["sent"] = True
                os.kill(os.getpid(), signal.SIGTERM)
            after_stage()

        return real_blockwise(*a, after_stage=stage, **kw)

    monkeypatch.setitem(trainer.BACKWARD, "blockwise", signalled_mid_step)
    assert run(argv() + ["--resume"]) == 3
    monkeypatch.undo()
    record = record_of(run_dir)
    assert record["step"] == 5 and record["checkpoints"] == [4, 5] and record["stop_reason"] == "terminated"
    assert run(argv() + ["--resume"]) == 0
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(record_of(run_dir)) == drawn(record_of(run_dir.parent / "straight"))


def test_every_safetensors_dtype_has_a_size():
    """The plan sums the base weights' bytes from their headers. Every dtype a safetensors header can name is sized,
    the unsigned ones included (the optimizer's own step count is a U64), and one it does not know is an error with
    the tensor's name, which the script reports as it reports any bad argument."""
    names = ["BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "F64", "I64", "U64"]
    sizes = [1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 8, 8, 8]
    header = {f"t{i}": {"dtype": name, "shape": [3, 2], "data_offsets": [0, 0]} for i, name in enumerate(names)}
    summary = trainer.weights_summary([{**header, "__metadata__": {"format": "pt"}}])
    assert summary["tensors"] == len(names) and summary["bytes"] == 6 * sum(sizes)
    assert summary["bytes_by_dtype"] == {name: 6 * size for name, size in sorted(zip(names, sizes))}
    for name, dtype in (("U64", mx.uint64), ("U32", mx.uint32), ("U16", mx.uint16), ("I16", mx.int16), ("BOOL", mx.bool_)):
        assert trainer.BYTES[name] == mx.zeros((1,), dtype=dtype).nbytes, name
    with pytest.raises(ValueError, match="odd.weight has a dtype this summary does not know the size of: 'Q7'"):
        trainer.weights_summary([{"odd.weight": {"dtype": "Q7", "shape": [1], "data_offsets": [0, 1]}}])


def test_checkpoints_are_whole_or_absent(work, capsys, monkeypatch):
    """A checkpoint is a directory written under a temporary name and renamed. A write that fails leaves no
    `step-` directory and no temporary one; a temporary directory left by a killed writer is not a checkpoint and is
    cleared when the next session starts."""
    argv, run_dir, _ = work
    real = os.rename

    def no_rename(src, dst):
        if Path(dst).name == "step-00000010":
            raise OSError("no rename today")
        return real(src, dst)

    monkeypatch.setattr(os, "rename", no_rename)
    with pytest.raises(OSError, match="no rename today"):
        run(argv())
    monkeypatch.undo()
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-00000005"]
    assert record_of(run_dir)["status"] == "error" and record_of(run_dir)["checkpoints"] == [5]

    leftover = run_dir / "checkpoints" / f"{files.TMP}999-deadbeef-step-00000009"
    leftover.mkdir()
    (leftover / "adapter.safetensors").write_bytes(b"half")
    assert trainer.latest_checkpoint(run_dir).name == "step-00000005"

    # a writer that dies between the checkpoint's two files, with no chance to clean up (a kill): what it leaves is
    # under a temporary name, so the half-written step 10 is not there to be resumed from
    def dies(*a, **kw):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(train, "optimizer_state", dies)
        patch.setattr(trainer.shutil, "rmtree", lambda *a, **kw: None)
        with pytest.raises(KeyboardInterrupt):
            run(argv() + ["--resume"])
    names = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert [n for n in names if not n.startswith(files.TMP)] == ["step-00000005"] and len(names) == 3
    assert any(n.endswith("step-00000010") and n.startswith(files.TMP) for n in names)
    assert record_of(run_dir)["status"] == "interrupted"
    assert run(argv() + ["--resume"]) == 0
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-00000005", "step-00000010", "step-00000012"]
    described = json.loads((run_dir / "checkpoints" / "step-00000010" / "checkpoint.json").read_text())
    assert described["format"] == trainer.CHECKPOINT and described["step"] == 10 and len(described["losses"]) == 10
    assert described["rng"] == {"scheme": "seeded per step", "seed": 5, "next_step": 10}
    assert set(described["files"]) == {"adapter.safetensors", "optimizer.safetensors"}
    assert described["trajectory"] == files.sha256_json(record_of(run_dir)["trajectory"])


@pytest.mark.parametrize("what", ["a checkpoint", "the record", "the final adapter"])
def test_a_failure_after_a_rename_removes_nothing(work, capsys, monkeypatch, what):
    """The same boundary as in the cache, at the run's three atomic writes: the rename has happened, and flushing the
    directory afterwards fails. Nothing is cleaned up, because what was renamed is in place and whole: the session
    ends with `files.NotDurable`, the checkpoint (or the record, or the adapter) is there, a checkpoint that is there
    is in the record's list, and the run resumes to the uninterrupted run's weights."""
    argv, run_dir, _ = work
    assert run(argv(run="straight")) == 0
    real = files.sync_directory
    target = {"a checkpoint": run_dir / "checkpoints" / "step-00000010", "the record": run_dir / "run.json",
              "the final adapter": run_dir / "adapter.safetensors"}[what]
    failed = []

    def fail_once_it_is_in_place(path):
        ready = target.exists() and (what != "the record" or json.loads(target.read_text()).get("step") == 7)
        if Path(path) == target.parent and ready and not failed:
            failed.append(True)
            raise OSError("the directory would not flush")
        return real(path)

    monkeypatch.setattr(files, "sync_directory", fail_once_it_is_in_place)
    with pytest.raises(files.NotDurable, match=f"{target.name} is written and in place"):
        run(argv())
    monkeypatch.undo()
    assert failed == [True]
    record = record_of(run_dir)
    assert record["status"] == "error" and "NotDurable" in record["stop_reason"]
    if what == "a checkpoint":
        assert record["step"] == 10 and record["checkpoints"] == [5, 10]
        assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-00000005", "step-00000010"]
        step, matrices, _, losses = trainer.read_checkpoint(target, record["trajectory"], TINY)
        assert step == 10 and len(losses) == 10 and len(matrices) == 18
    elif what == "the record":
        assert record["step"] == 7 and len(record["losses"]) == 7 and record["checkpoints"] == [5]
    else:
        assert record["step"] == 12 and record["checkpoints"] == [5, 10, 12]
        assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight"), "the adapter is in place and whole"
    assert run(argv() + ["--resume"]) == 0
    assert record_of(run_dir)["status"] == "finished"
    assert adapter_bytes(run_dir) == adapter_bytes(run_dir.parent / "straight")
    assert drawn(record_of(run_dir)) == drawn(record_of(run_dir.parent / "straight"))


# --- what a step draws -------------------------------------------------------------------------------------------------


def test_every_draw_is_a_function_of_the_seed_and_the_step():
    """There is no generator carried from step to step: a step's example, level and noise come from seeds made of
    the run's seed, the step and the stream. The values are pinned, because a run resumed after this function changed
    would draw something else (the code digest in the record refuses that resume; this says why it would be wrong)."""
    assert trainer.step_seed(5, 0, "noise") == 7760099336911183838 and trainer.step_seed(5, 1, "noise") == 326124782927953987
    assert trainer.step_seed(5, 0, "level") == 5900253131158821752 and trainer.step_seed(6, 0, "noise") == 3232754514343299279
    assert [trainer.example_at(5, i, 7) for i in range(14)] == [4, 0, 2, 3, 1, 5, 6, 3, 0, 5, 1, 6, 4, 2]
    seeds = {trainer.step_seed(seed, step, stream) for seed in (0, 1, 5) for step in range(50) for stream in trainer.STREAMS}
    assert len(seeds) == 3 * 50 * len(trainer.STREAMS) and all(0 <= s < 2**63 for s in seeds)
    for count in (1, 2, 7):
        for epoch in range(4):
            assert sorted(trainer.example_at(5, epoch * count + i, count) for i in range(count)) == list(range(count))
    assert [trainer.example_at(5, i, 7) for i in range(7)] != [trainer.example_at(5, 7 + i, 7) for i in range(7)]
    assert [trainer.example_at(5, i, 7) for i in range(7)] != [trainer.example_at(6, i, 7) for i in range(7)]
    # the adapter's start and each picture's posterior seed come from seeds too
    spec = {"adapter": {"rank": 2, "alpha": 2.0, "targets": ["attn.to_q"], "refiner": False}, "hyper": {"seed": 5}}
    first, again = trainer.start_adapter(TINY, spec), trainer.start_adapter(TINY, spec)
    other = trainer.start_adapter(TINY, {**spec, "hyper": {"seed": 6}})
    key = "transformer_blocks.0.attn.to_q.lora_A.weight"
    assert np.array_equal(to_np(first.tensors[key]), to_np(again.tensors[key]))
    assert not np.array_equal(to_np(first.tensors[key]), to_np(other.tensors[key]))
    assert not to_np(first.tensors[key.replace("lora_A", "lora_B")]).any(), "PEFT's start: lora_B zero"
    a, b = files.sha256_json("one picture"), files.sha256_json("another")
    assert T.picture_seed(11, a) == T.picture_seed(11, a) != T.picture_seed(11, b)
    assert T.picture_seed(12, a) != T.picture_seed(11, a) and 0 <= T.picture_seed(11, a) < 2**63
    hyper = {"seed": 5, "weighting": "sigma_sqrt", "density": {"scheme": "mode", "logit_mean": None, "logit_std": None, "mode_scale": 1.29}}
    grid = train.training_sigmas(1000, 3.0)
    levels = [trainer.noise_level(hyper, grid, step) for step in range(20)]
    assert levels == [trainer.noise_level(hyper, grid, step) for step in range(20)] and len({s for s, _ in levels}) == 20
    assert all(abs(w - 1 / float(s) ** 2) < 1e-3 * w for s, w in levels)
    assert trainer.noise_level({**hyper, "weighting": "none"}, grid, 3) == (levels[3][0], None)


# --- from pictures, with the tiny encoders -----------------------------------------------------------------------------


@NEEDS_PROCESSOR
def test_a_run_from_pictures_encodes_what_the_cache_lacks(tmp_path, capsys, monkeypatch):
    """The whole path on a tiny release that has every component (it needs the release's tokenizer and processor
    files): an empty cache, a text-to-image example and an edit example. The plan says two examples are to be
    encoded; the run loads the tiny VAE and text encoder, writes two entries and trains. A second run finds both
    entries and loads no encoder."""
    import test_run_p_h3_image as R

    class Factory:
        def mktemp(self, name):
            folder = tmp_path / name
            folder.mkdir()
            return folder

    _, rel, pictures = R._release(Factory())
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"examples": [
        {"picture": str(pictures[0]), "prompt": "A risograph print of a lighthouse at dusk.", "height": 40, "width": 48},
        {"picture": str(pictures[1]), "prompt": "give it the stripes from picture 1", "height": 56, "width": 32,
         "references": [str(pictures[0])]}]}))
    argv = arguments(rel, dataset, tmp_path / "cache", tmp_path / "run", **{"--max-steps": ["3"], "--checkpoint-every": ["3"]})
    assert run(argv + ["--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["examples"]["to_encode"] == 2 and not (tmp_path / "cache").exists()
    assert run(argv) == 0
    out = capsys.readouterr().out
    assert out.count('"event": "encoded"') == 2
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2 == len(list((tmp_path / "cache").glob("*.safetensors")))
    record = record_of(tmp_path / "run")
    assert record["status"] == "finished" and record["step"] == 3 and all(np.isfinite(x["loss"]) for x in record["losses"])

    monkeypatch.setattr(T, "encode_missing", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("encoded again")))
    again = arguments(rel, dataset, tmp_path / "cache", tmp_path / "again", **{"--max-steps": ["3"], "--checkpoint-every": ["3"]})
    assert run(again) == 0
    assert '"event": "encoded"' not in capsys.readouterr().out
    assert adapter_bytes(tmp_path / "again") == adapter_bytes(tmp_path / "run")
    assert record_of(tmp_path / "again")["plan"]["examples"]["to_encode"] == 0
    assert all(e["text_rows"] for e in record_of(tmp_path / "again")["plan"]["examples"]["list"])
