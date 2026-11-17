"""One LoRA training run on cached examples: what it would do (the plan), what it did (the record), where it can go
on from (checkpoints), what stops it (limits), and the loop itself.

The loop is the pieces of `train` and `backward` in order: pick an example, draw a noise level and the noise, build
the example, take the loss and its gradients, clip and step. Everything a step draws is a function of the run's seed
and the step's number (`step_seed`), so there is no generator to carry from step to step, and a run resumed at step k
draws what an uninterrupted run draws at step k.

Two groups of arguments. The *trajectory* is everything the weights depend on: the model, the examples, the adapter,
the hyper-parameters, the seed, the versions. A run can only be resumed with the same trajectory. The *limits* say
when to stop and how often to checkpoint, and may change from one session to the next.

A run directory belongs to one session at a time (`open_run`), and every check that can refuse a run is made there,
before anything heavy: nothing here loads a model either, the caller hands in functions that do.
"""

import contextlib
import json
import math
import os
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import backward, cache, files, lora, ops, train
from .image import AUDIO_CHANNELS, NUM_AUDIO_LATENTS
from .pipeline import Noise

RECORD = "h3mlx-training-run-1"
CHECKPOINT = "h3mlx-training-checkpoint-1"
TRAINED, ASSISTANT = "trained", "assistant"  # the two adapters' names in the weights
BACKWARD = {"blockwise": backward.loss_and_gradients_blockwise, "checkpointed": backward.loss_and_gradients_checkpointed}
DENSITY = ("uniform", "logit_normal", "mode")
WEIGHTING = ("none", "sigma_sqrt", "cosmap")
STREAMS = {"init": 0, "order": 1, "level": 2, "noise": 3}
FINISHED, STOPPED = 0, 3  # exit codes: the run reached its last step; a limit or a signal stopped it before that
OUTPUTS = ("run.json", "adapter.safetensors", "checkpoints")  # what a run writes in its directory
LOCK = "run.lock"  # the run directory's lock file: held by the one session training in it
# bytes per element of every dtype a safetensors header can name
BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
         "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8, "C64": 8}


class Refused(ValueError):
    """The run cannot start or go on as asked: a run directory in use, or a resume that does not match."""


class LimitReached(Exception):
    """A limit or a signal stopped the session. `reason` is `"wall_clock"`, `"peak_memory"` or `"terminated"`."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


# --- what each step draws -----------------------------------------------------------------------------------------------


def step_seed(seed: int, step: int, stream: str) -> int:
    """A seed for one of step `step`'s draws (`stream`: the example order, the noise level, the noise; `"init"` at
    step 0 is the adapter's initialisation), from the run's seed. numpy's `SeedSequence` mixes the three numbers, so
    neighbouring steps and streams share nothing. 63 bits, which is what a torch generator takes."""
    return int(np.random.SeedSequence([seed, step, STREAMS[stream]]).generate_state(1, dtype=np.uint64)[0] >> 1)


def example_at(seed: int, step: int, count: int) -> int:
    """Which example step `step` trains on: the examples in a new random order for every pass over them."""
    epoch, position = divmod(step, count)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, STREAMS["order"]])).permutation(count)
    return int(order[position])


def noise_level(hyper: dict, grid: np.ndarray, step: int) -> tuple[np.float32, float | None]:
    """(sigma, loss weight or None) for step `step`: one draw of the density from the step's own generator, looked up
    in the grid, and the weighting of that level."""
    import torch

    density = hyper["density"]
    generator = torch.Generator().manual_seed(step_seed(hyper["seed"], step, "level"))
    u = train.density_for_timestep_sampling(density["scheme"], 1, density["logit_mean"], density["logit_std"],
                                            density["mode_scale"], generator=generator)
    sigma = train.sample_sigmas(u, grid)[0][0]
    if hyper["weighting"] == "none":
        return sigma, None
    return sigma, float(train.loss_weighting_for_sd3(hyper["weighting"], np.float32([sigma]))[0])


# --- the arguments, validated ------------------------------------------------------------------------------------------


def _number(what: str, value, low: float = 0.0, strict: bool = True) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or not (value > low if strict else value >= low):
        raise ValueError(f"{what} must be a finite number {'above' if strict else 'of at least'} {low}, got {value!r}")
    return value


def _count(what: str, value, low: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{what} must be an integer of at least {low}, got {value!r}")
    return value


def adapter_modules(cfg: dict, targets, refiner: bool) -> list[str]:
    """The Linears that get the trained adapter: the named attention and feed-forward matrices (`lora.BLOCK_TARGETS`)
    of every transformer block and, with `refiner`, of every token-refiner block."""
    targets = list(targets)
    unknown = [t for t in targets if t not in lora.BLOCK_TARGETS]
    if not targets or unknown or len(set(targets)) != len(targets):
        raise ValueError(f"targets must be some of {lora.BLOCK_TARGETS}, each once; got {targets}")
    modules = lora.block_modules(cfg, targets)
    return modules if refiner else [m for m in modules if not m.startswith("token_refiner.")]


def parameter_count(cfg: dict, modules, rank: int) -> int:
    shapes = lora.wired_shapes(cfg)
    return sum(rank * (shapes[m][0] + shapes[m][1]) for m in modules)


def trajectory(*, cfg: dict, transformer: str, dataset, rank: int, alpha: float, targets, refiner: bool,
               assistant: str | None, learning_rate: float, betas, eps: float, weight_decay: float,
               max_grad_norm: float, shift: float, grid: int, density: str, logit_mean, logit_std, mode_scale,
               weighting: str, condition_t: float, seed: int, backward: str) -> dict:
    """Everything the trained weights depend on, checked, as one JSON value. No argument has a default.

    `transformer` is the base weights' identity (`cache.identity`), `dataset` the examples' cache keys in order,
    `assistant` the sha256 of a frozen adapter's file or None. The density's own parameters are required for its
    scheme and must be None otherwise: `logit_mean` and `logit_std` for `"logit_normal"`, `mode_scale` for `"mode"`."""
    dataset = list(dataset)
    if not dataset or not all(isinstance(k, str) and len(k) == 64 for k in dataset):
        raise ValueError("the dataset is a non-empty list of cache keys")
    _count("rank", rank)
    lora.check_alpha("alpha", alpha)
    if not isinstance(refiner, bool):
        raise ValueError(f"refiner must be True or False, got {refiner!r}")
    modules = adapter_modules(cfg, targets, refiner)
    betas = list(betas)
    if len(betas) != 2 or not all(isinstance(b, float) and 0 <= b < 1 for b in betas):
        raise ValueError(f"betas must be two numbers in [0, 1), got {betas}")
    if not ops.usable_shift(shift):
        raise ValueError(f"shift must be positive and finite, got {shift!r}")
    train.training_sigmas(_count("grid", grid), shift)
    if density not in DENSITY:
        raise ValueError(f"density is one of {DENSITY}, got {density!r}")
    wanted = {"uniform": (), "logit_normal": ("logit_mean", "logit_std"), "mode": ("mode_scale",)}[density]
    for name, value in (("logit_mean", logit_mean), ("logit_std", logit_std), ("mode_scale", mode_scale)):
        if name in wanted:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"density {density!r} needs {name} as a finite number, got {value!r}")
        elif value is not None:
            raise ValueError(f"{name} is not used by density {density!r}: leave it out")
    if density == "logit_normal":
        _number("logit_std", logit_std)
    if weighting not in WEIGHTING:
        raise ValueError(f"weighting is one of {WEIGHTING}, got {weighting!r}")
    if not isinstance(condition_t, float) or not 0 <= condition_t <= 1:
        raise ValueError(f"condition_t must be a number in [0, 1], got {condition_t!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
    if backward not in BACKWARD:
        raise ValueError(f"backward is one of {sorted(BACKWARD)}, got {backward!r}")
    if assistant is not None and not (isinstance(assistant, str) and len(assistant) == 64):
        raise ValueError(f"assistant is the sha256 of the frozen adapter's file, or None; got {assistant!r}")
    return {
        "model": {"transformer": transformer, "config": json.loads(files.canonical_json(cfg))},
        "dataset": dataset,
        "adapter": {"name": TRAINED, "rank": rank, "alpha": alpha, "targets": list(targets), "refiner": refiner,
                    "modules": len(modules), "parameters": parameter_count(cfg, modules, rank)},
        "assistant": assistant,
        "hyper": {"learning_rate": _number("learning_rate", learning_rate), "betas": betas, "eps": _number("eps", eps),
                  "weight_decay": _number("weight_decay", weight_decay, strict=False),
                  "max_grad_norm": _number("max_grad_norm", max_grad_norm), "shift": shift, "grid": grid,
                  "density": {"scheme": density, "logit_mean": logit_mean, "logit_std": logit_std, "mode_scale": mode_scale},
                  "weighting": weighting, "condition_t": condition_t, "seed": seed, "backward": backward,
                  "audio_level": "the picture's"},
        "versions": cache.versions(),
    }


def limits(*, max_steps: int, max_seconds: float, max_peak_memory_gb: float, checkpoint_every: int) -> dict:
    """When a session stops, and how often it checkpoints. Each is required, and each stops the run: reaching
    `max_steps` finishes it, and the wall clock or MLX's peak-memory counter going over its limit ends the session
    with the run resumable. No argument has a default."""
    return {"max_steps": _count("max_steps", max_steps), "max_seconds": _number("max_seconds", max_seconds),
            "max_peak_memory_gb": _number("max_peak_memory_gb", max_peak_memory_gb),
            "checkpoint_every": _count("checkpoint_every", checkpoint_every)}


# --- the plan ----------------------------------------------------------------------------------------------------------


def example_rows(cfg: dict, vae_ratio: int, height: int, width: int, reference_sizes=(), text_rows=None) -> dict:
    """The rows one example puts in the packed sequence. `reference_sizes` are (width, height) in pixels; `text_rows`
    is None while the prompt has not been encoded."""
    _, ph, pw = cfg["patch_size"]
    per = lambda w, h: (h // vae_ratio // ph) * (w // vae_ratio // pw)  # noqa: E731
    rows = {"video_rows": per(width, height), "condition_rows": sum(per(w, h) for w, h in reference_sizes),
            "audio_rows": NUM_AUDIO_LATENTS * AUDIO_CHANNELS, "text_rows": text_rows}
    known = rows["video_rows"] + rows["condition_rows"] + rows["audio_rows"]
    rows["sequence_rows"] = None if text_rows is None else known + text_rows
    rows["sequence_rows_without_text"] = known
    return rows


def weights_summary(headers) -> dict:
    """From the safetensors headers of the base weights (no tensor data): how many tensors, their bytes, and which
    dtypes hold them."""
    by_dtype: dict = {}
    count = 0
    for header in headers:
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            count += 1
            if entry["dtype"] not in BYTES:
                raise ValueError(f"{name} has a dtype this summary does not know the size of: {entry['dtype']!r}")
            size = BYTES[entry["dtype"]] * math.prod(entry["shape"])
            by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + size
    return {"tensors": count, "bytes": sum(by_dtype.values()), "bytes_by_dtype": dict(sorted(by_dtype.items()))}


def plan(traj: dict, lim: dict, cfg: dict, examples: list, *, run_dir, cache_dir, weights: dict | None = None,
         block_dtype: str | None = None) -> dict:
    """What a run would do, from the arguments and the configs alone.

    `examples` holds one dict per example: its cache key, whether the cache has it, its sizes and its rows
    (`example_rows`). `weights` is `weights_summary` of the base weights' headers, and `block_dtype` the dtype the
    headers give a block's matrices; both may be None, and the estimates that need them are then left out."""
    adapter, hyper = traj["adapter"], traj["hyper"]
    modules = adapter_modules(cfg, adapter["targets"], adapter["refiner"])
    count = len(examples)
    largest = max(e["sequence_rows"] or e["sequence_rows_without_text"] for e in examples)
    steps = lim["max_steps"]
    checkpoints = sorted(set(range(lim["checkpoint_every"], steps + 1, lim["checkpoint_every"])) | {steps})
    sizes = {"adapter_bytes": 4 * adapter["parameters"], "optimizer_state_bytes": 8 * adapter["parameters"],
             "checkpoint_bytes": 12 * adapter["parameters"], "checkpoints_bytes": 12 * adapter["parameters"] * len(checkpoints),
             "largest_sequence_rows": largest,
             "largest_sequence_is_without_text": any(e["sequence_rows"] is None for e in examples)}
    if weights is not None:
        sizes["base_weights_bytes"] = weights["bytes"]
        sizes["base_weights_bytes_by_dtype"] = weights["bytes_by_dtype"]
    if block_dtype is not None and hyper["backward"] == "blockwise":
        sizes["saved_block_inputs_bytes"] = cfg["num_layers"] * largest * cfg["hidden_size"] * BYTES[block_dtype]
    run_dir = Path(run_dir)
    return {
        "examples": {"count": count, "to_encode": len({e["key"] for e in examples if not e["cached"]}), "list": examples},
        "steps": {"max_steps": steps, "passes_over_the_examples": round(steps / count, 3),
                  "checkpoint_every": lim["checkpoint_every"], "checkpoints_at": checkpoints},
        "adapter": {**adapter, "scaling": adapter["alpha"] / adapter["rank"], "first_modules": modules[:3],
                    "last_module": modules[-1]},
        "assistant": traj["assistant"],
        "sizes": sizes,
        "hyper": hyper,
        "limits": lim,
        "cache": {"directory": str(cache_dir), "keys": [e["key"] for e in examples]},
        "writes": {"run_directory": str(run_dir), "record": str(run_dir / "run.json"),
                   "checkpoints": str(run_dir / "checkpoints" / "step-NNNNNNNN"),
                   "final_adapter": str(run_dir / "adapter.safetensors")},
    }


# --- the record --------------------------------------------------------------------------------------------------------


def save_record(run_dir: Path, record: dict) -> None:
    files.atomic_json(Path(run_dir) / "run.json", record)


def read_record(run_dir: Path) -> dict:
    path = Path(run_dir) / "run.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: no readable run record to resume from ({e})") from e
    if not isinstance(record, dict) or record.get("format") != RECORD:
        raise Refused(f"{path} is not a {RECORD} record")
    return record


# --- checkpoints -------------------------------------------------------------------------------------------------------


def checkpoint_path(run_dir, step: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"step-{step:08d}"


def write_checkpoint(run_dir, step: int, traj: dict, trainable: dict, optimizer, losses: list) -> Path:
    """One checkpoint: the adapter as a diffusers LoRA file, the optimizer's state, and a record with the step, the
    losses so far and each file's digest. It is written as a directory under a temporary name and renamed into place,
    so a checkpoint directory is there only when it is whole. A failure before the rename removes the temporary
    directory. After the rename nothing is removed: if flushing the checkpoints directory then fails, the error is
    `files.NotDurable` and the checkpoint stands."""
    final = checkpoint_path(run_dir, step)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = files.temporary(final)
    adapter = lora.Adapter(lora.file_tensors(trainable, TRAINED),
                           {m: traj["adapter"]["alpha"] for m in {lora.RUNTIME_RE.match(k)["module"] for k in trainable}})
    try:
        tmp.mkdir()
        lora.save(tmp / "adapter.safetensors", adapter)
        mx.save_safetensors(str(tmp / "optimizer.safetensors"), train.optimizer_state(optimizer, trainable))
        described = {"format": CHECKPOINT, "step": step, "trajectory": files.sha256_json(traj),
                     "rng": {"scheme": "seeded per step", "seed": traj["hyper"]["seed"], "next_step": step},
                     "files": {name: files.sha256_file(tmp / name) for name in ("adapter.safetensors", "optimizer.safetensors")},
                     "losses": losses}
        (tmp / "checkpoint.json").write_text(json.dumps(described, indent=1, sort_keys=True, allow_nan=False) + "\n")
        for name in ("adapter.safetensors", "optimizer.safetensors", "checkpoint.json"):
            files.sync_file(tmp / name)
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    files.flush_renamed(final)  # after the rename: a failure here is `files.NotDurable`, and the checkpoint stands
    return final


def latest_checkpoint(run_dir) -> Path | None:
    """The checkpoint directory with the highest step, or None. Directories under temporary names are a killed
    writer's and are not checkpoints."""
    parent = Path(run_dir) / "checkpoints"
    found = sorted(p for p in parent.glob("step-*") if p.is_dir() and p.name[5:].isdigit()) if parent.is_dir() else []
    return found[-1] if found else None


def read_checkpoint(path: Path, traj: dict, cfg: dict) -> tuple[int, dict, dict, list]:
    """(step, trainable matrices, optimizer state, losses) from a checkpoint directory, if it is whole and belongs to
    this trajectory; otherwise `Refused`. Nothing falls back to an older checkpoint: a damaged one is to be looked
    at, not skipped."""
    try:
        described = json.loads((path / "checkpoint.json").read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: the checkpoint's record cannot be read ({e})") from e
    if not isinstance(described, dict) or described.get("format") != CHECKPOINT:
        raise Refused(f"{path} is not a {CHECKPOINT} checkpoint")
    step = described.get("step")
    if step != int(path.name[5:]) or described.get("trajectory") != files.sha256_json(traj):
        raise Refused(f"{path} belongs to another run or another step (its record says step {step})")
    for name, digest in described.get("files", {}).items():
        if not (path / name).is_file() or files.sha256_file(path / name) != digest:
            raise Refused(f"{path / name} is missing or is not the file the checkpoint describes")
    if set(described.get("files", {})) != {"adapter.safetensors", "optimizer.safetensors"}:
        raise Refused(f"{path}: the checkpoint does not list its two files")
    losses = described.get("losses")
    if not isinstance(losses, list) or len(losses) != step:
        raise Refused(f"{path}: {step} steps but {len(losses) if isinstance(losses, list) else 'no'} losses")
    adapter = lora.load(path / "adapter.safetensors")
    spec = traj["adapter"]
    modules = adapter_modules(cfg, spec["targets"], spec["refiner"])
    lora.check(adapter, cfg)
    if sorted(adapter.modules) != sorted(modules) or any(adapter.rank(m) != spec["rank"] or adapter.alpha[m] != spec["alpha"]
                                                          for m in modules):
        raise Refused(f"{path}: the saved adapter is not this run's (modules, rank or alpha differ)")
    trainable = {k: v.astype(mx.float32) for k, v in lora.runtime_tensors(adapter, TRAINED).items()}
    saved = mx.load(str(path / "optimizer.safetensors"), format="safetensors")
    return step, trainable, saved, losses


# --- the run -----------------------------------------------------------------------------------------------------------


def start_adapter(cfg: dict, traj: dict) -> lora.Adapter:
    """The adapter a run starts from: PEFT's initialisation (`lora.init`), seeded from the run's seed."""
    spec = traj["adapter"]
    return lora.init(cfg, spec["rank"], spec["alpha"], adapter_modules(cfg, spec["targets"], spec["refiner"]),
                     seed=step_seed(traj["hyper"]["seed"], 0, "init"))


@dataclass
class Session:
    """A run directory held by this process and checked, ready to train in (`open_run`)."""

    run_dir: Path
    trajectory: dict
    limits: dict
    cfg: dict
    resume: bool
    record: dict  # the run's record: as read on a resume, new otherwise. Nothing has been written yet
    step: int  # the step training goes on from
    matrices: dict | None  # the checkpoint's trainable matrices, or None when there is no checkpoint
    optimizer: dict | None  # the checkpoint's optimizer state
    losses: list  # the losses up to `step`


def _differences(traj: dict, record: dict) -> list[str]:
    return [a.replace("the entry has", "the record has").replace("missing from the entry", "missing from the record")
            for a in cache.differences(traj, record.get("trajectory"))]


@contextlib.contextmanager
def open_run(run_dir, traj: dict, lim: dict, cfg: dict, *, resume: bool):
    """Take a run directory for this process and check that the run can start or go on in it. Yields a `Session`.

    Everything here is cheap and reads only the run directory: no weights are loaded, no example is encoded, and
    nothing in the directory is changed but its lock file. So a caller does it first, and a run that is refused
    costs nothing and leaves the directory as it was.

    * **One session at a time.** The directory's lock file, `run.lock`, is held for as long as the session is open
      (`files.exclusive`: the lock is the process's, so a session that is killed does not leave it held). A second
      session on the same directory is refused, and told which process holds it.
    * **A new run** needs a directory that holds none of a run's outputs (`OUTPUTS`: `run.json`,
      `adapter.safetensors`, `checkpoints`), whether or not there is a record. Nothing is overwritten.
    * **A resume** needs the record, with exactly this trajectory; the latest checkpoint on disk, which must not be
      older than the latest one the record lists, and must be whole and this run's (`read_checkpoint`); and a
      `max_steps` that the run has not already passed. A run whose record lists no checkpoint and that has none on
      disk goes on from step 0, which is where its record says its last checkpoint is."""
    run_dir = Path(run_dir)
    if resume and not run_dir.is_dir():
        raise Refused(f"{run_dir / 'run.json'}: no readable run record to resume from (no such directory)")
    run_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(files.exclusive(run_dir / LOCK, f"training in {run_dir.name}"))
        except files.Busy as busy:
            raise Refused(f"the run directory is in use: {busy}. One session trains in a run directory at a time") from None
        if not resume:
            there = [name for name in OUTPUTS if os.path.lexists(run_dir / name)]
            if there:
                raise Refused(f"{run_dir} already holds {', '.join(there)}: a new run writes {', '.join(OUTPUTS)} "
                              "and overwrites none of them. Pass resume to go on with the run that is there, or "
                              "use another directory")
            record = {"format": RECORD, "trajectory": traj, "seed": traj["hyper"]["seed"],
                      "versions": traj["versions"], "sessions": [], "step": 0, "losses": [], "checkpoints": []}
            yield Session(run_dir, traj, lim, cfg, False, record, 0, None, None, [])
            return
        record = read_record(run_dir)
        apart = _differences(traj, record)
        if apart:
            raise Refused(f"{run_dir / 'run.json'} is another run: " + "; ".join(apart[:6])
                          + (f"; and {len(apart) - 6} more" if len(apart) > 6 else ""))
        listed = max(record.get("checkpoints") or [0])
        checkpoint = latest_checkpoint(run_dir)
        found = int(checkpoint.name[5:]) if checkpoint is not None else 0
        if found < listed:
            raise Refused(f"{run_dir / 'run.json'} lists a checkpoint at step {listed}, and the latest in "
                          f"{run_dir / 'checkpoints'} is " + (f"step {found}" if found else "none")
                          + ": the run is not gone back to an older checkpoint, or to its start, without being told to")
        step, matrices, saved, losses = (0, None, None, []) if checkpoint is None else read_checkpoint(checkpoint, traj, cfg)
        if step > lim["max_steps"]:
            raise Refused(f"the run is at step {step}, past a max_steps of {lim['max_steps']}: nothing to do, and "
                          "nothing is written")
        yield Session(run_dir, traj, lim, cfg, True, record, step, matrices, saved, losses)


def train_session(session: Session, entries: list, load_weights, load_assistant=None, *, plan: dict | None = None,
                  command=None, clock=time.monotonic, log=print) -> int:
    """Train in an open session until a limit or a signal stops it. Returns `FINISHED` (0) when the run has reached
    `max_steps`, `STOPPED` (3) when the wall clock, the memory limit or a termination signal ended the session first.

    `entries` are the examples' cache entries in the trajectory's order. `load_weights()` returns the base weights
    and `load_assistant()` the frozen adapter (`lora.Adapter`) or None; they are called once, after the record is
    written. `clock` is a monotonic clock in seconds.

    * The record (`run.json`) is rewritten atomically after every step. Checkpoints are written every
      `checkpoint_every` steps and when the session ends, each atomically.
    * The memory limit is read from MLX's peak-memory counter (reset when the session starts): after every evaluated
      block of the block-by-block gradient, and after every step. Over it, the step in progress is dropped and the
      session ends at the last completed step.
    * SIGTERM is taken as a request to stop: at the next of the same places the step in progress is dropped, a
      checkpoint is written at the last completed step, the record says `terminated`, and the function returns. The
      signal's previous handler is put back when the session ends."""
    run_dir, traj, lim, cfg, record = session.run_dir, session.trajectory, session.limits, session.cfg, session.record
    hyper, spec = traj["hyper"], traj["adapter"]
    if len(entries) != len(traj["dataset"]):
        raise ValueError(f"{len(traj['dataset'])} examples in the trajectory, {len(entries)} entries")
    for stale in (run_dir / "checkpoints").glob(f"{files.TMP}*") if (run_dir / "checkpoints").is_dir() else ():
        shutil.rmtree(stale, ignore_errors=True)  # a dead writer's: this session holds the directory

    started = clock()
    step, losses = session.step, session.losses
    discarded = record["step"] - step
    note = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": command, "limits": lim,
            "resumed": session.resume, "from_step": step, "discarded_steps": max(discarded, 0), "to_step": None,
            "stopped_by": None}
    record.update(status="running", stop_reason=None, limits=lim, plan=plan if plan is not None else record.get("plan"),
                  step=step, losses=list(losses), checkpoints=[c for c in record["checkpoints"] if c <= step])
    record.pop("stop_detail", None)
    record["sessions"].append(note)
    mx.reset_peak_memory()
    asked_to_stop = []
    in_main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, lambda number, frame: asked_to_stop.append(number)) if in_main else None

    try:
        save_record(run_dir, record)
        base = load_weights()
        assistant = load_assistant() if load_assistant is not None else None
        if (assistant is None) != (traj["assistant"] is None):
            raise ValueError("the trajectory and the assistant handed in disagree about whether there is one")
        params = base if assistant is None else lora.attach(base, cfg, assistant, name=ASSISTANT)
        optimizer = train.adamw(learning_rate=hyper["learning_rate"], betas=tuple(hyper["betas"]), eps=hyper["eps"],
                                weight_decay=hyper["weight_decay"])
        if session.matrices is None:
            adapter = start_adapter(cfg, traj)
        else:
            adapter = lora.Adapter(lora.file_tensors(session.matrices, TRAINED),
                                   {m: spec["alpha"] for m in adapter_modules(cfg, spec["targets"], spec["refiner"])})
        frozen, trainable = lora.split(lora.attach(params, cfg, adapter, keep_dtype=True, name=TRAINED), TRAINED)
        if session.optimizer is not None:
            train.restore_optimizer(optimizer, trainable, session.optimizer)
        log(json.dumps({"event": "start", "step": step, "max_steps": lim["max_steps"], "resumed": session.resume,
                        "examples": len(entries), "trainable_matrices": len(trainable)}))

        grid = train.training_sigmas(hyper["grid"], hyper["shift"])
        gradient = BACKWARD[hyper["backward"]]
        limit_bytes = lim["max_peak_memory_gb"] * 1e9
        saved_at = step

        def check():
            if asked_to_stop:
                raise LimitReached("terminated", "a termination signal (SIGTERM) asked the session to stop")
            peak = mx.get_peak_memory()
            if peak > limit_bytes:
                raise LimitReached("peak_memory", f"MLX's peak memory is {peak / 1e9:.3f} GB, over the limit of "
                                                  f"{lim['max_peak_memory_gb']} GB")

        def checkpoint_now():
            nonlocal saved_at
            if step > saved_at:
                try:
                    write_checkpoint(run_dir, step, traj, trainable, optimizer, record["losses"])
                except files.NotDurable:  # it is there and whole, and the record says so; the session still ends
                    record["checkpoints"].append(step)
                    saved_at = step
                    raise
                record["checkpoints"].append(step)
                saved_at = step

        stopped = None
        try:
            check()
            while step < lim["max_steps"]:
                if clock() - started > lim["max_seconds"]:
                    raise LimitReached("wall_clock", f"{clock() - started:.1f} s have passed, over the limit of "
                                                     f"{lim['max_seconds']} s")
                began = clock()
                index = example_at(hyper["seed"], step, len(entries))
                entry = entries[index]
                sigma, weight = noise_level(hyper, grid, step)
                example = train.build_example(cfg, entry.text_token_tags, entry.latents, sigma,
                                              Noise(step_seed(hyper["seed"], step, "noise")),
                                              condition_t=hyper["condition_t"], condition_latents=entry.condition_latents)
                extra = {"after_stage": check} if hyper["backward"] == "blockwise" else {}
                loss, grads = gradient(trainable, frozen, cfg, example, entry.text, weight, **extra)
                loss = float(loss)
                if not math.isfinite(loss):
                    raise FloatingPointError(f"the loss at step {step + 1} is {loss}")
                trainable, norm = train.optimizer_step(optimizer, trainable, grads, max_grad_norm=hyper["max_grad_norm"])
                step += 1
                record["losses"].append({"step": step, "example": index, "sigma": float(sigma), "weight": weight,
                                         "loss": loss, "grad_norm": norm, "seconds": round(clock() - began, 4),
                                         "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4)})
                record["step"] = step
                if step % lim["checkpoint_every"] == 0:
                    checkpoint_now()
                save_record(run_dir, record)
                log(json.dumps({"event": "step", **record["losses"][-1]}))
                check()
        except LimitReached as limit:
            stopped = limit
        checkpoint_now()
        if stopped is None:
            adapter = lora.Adapter(lora.file_tensors(trainable, TRAINED), dict(adapter.alpha))
            files.atomic_write(run_dir / "adapter.safetensors", lambda tmp: lora.save(tmp, adapter))
            record.update(status="finished", stop_reason="max_steps",
                          final_adapter={"file": "adapter.safetensors", "sha256": files.sha256_file(run_dir / "adapter.safetensors")})
        else:
            record.update(status="stopped", stop_reason=stopped.reason, stop_detail=stopped.detail)
        note.update(to_step=step, stopped_by=record["stop_reason"], seconds=round(clock() - started, 3),
                    peak_memory_gb=round(mx.get_peak_memory() / 1e9, 4))
        save_record(run_dir, record)
        log(json.dumps({"event": record["status"], "step": step, "reason": record["stop_reason"]}))
        return FINISHED if stopped is None else STOPPED
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                      stop_reason=f"{type(error).__name__}: {error}")
        note.update(stopped_by=record["stop_reason"])
        save_record(run_dir, record)
        raise
    finally:
        if in_main:
            signal.signal(signal.SIGTERM, previous)


def run(run_dir, traj: dict, lim: dict, cfg: dict, entries: list, load_weights, load_assistant=None, *,
        resume: bool, plan: dict | None = None, command=None, clock=time.monotonic, log=print) -> int:
    """`open_run`, then `train_session` in it: for a caller with nothing to do in between. (The script encodes the
    examples its cache lacks in between, once the run has been checked.)"""
    with open_run(run_dir, traj, lim, cfg, resume=resume) as session:
        return train_session(session, entries, load_weights, load_assistant, plan=plan, command=command, clock=clock,
                             log=log)
