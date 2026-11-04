"""HyperFlow, Video Rebirth's 8-step LoRA for MiniMax H3, on our MLX transformer.

Ported from the loader at github.com/Video-Rebirth/hyperflow `1dd2f342` (Apache-2.0: `lora.py`, `embedder.py`,
`schedule.py`, `blocks.py`); weights `VideoRebirthAI/hyperflow` at `5bde94d7` (`minimax_h3_hyperflow_8step_v1.0`,
MiniMax H3 Community License). What it changes, and nothing else:

* a PEFT LoRA (rank 256, alpha 256) on every block's and refiner block's to_q/to_k/to_v/to_out.0/ff.net.0.proj/ff.net.2
  and on the time embedder's two linears;
* a second time embedder for the step's endpoint r (`endpoint_time_embedder`: the base embedder's weights with its own
  LoRA), blended as emb(t) + gate * (emb_r(r) - emb(t)) (`ops.two_time_embedding`);
* a fixed raw sigma grid from the file (9 points, 8 model calls), shifted 12 (video) / 3 (audio) as the release's
  schedulers shift theirs; conditioning rows keep r = t (`denoise.row_time_pairs`).

The adapter never touches `adaln_proj`, which is why the one file also serves `transformer_ref` (ref2va).
"""

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from .denoise import Schedules

KEY_RE = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")
REQUIRED = ("hyperflow", "hyperflow_version", "hyperflow_gate", "lora_alpha", "base_model")
# sha256 of released files (VideoRebirthAI/hyperflow `hyperflow.json` at 5bde94d7)
KNOWN_SHA256 = {"minimax_h3_hyperflow_8step_v1.0.safetensors":
                "9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447"}
DEFAULT_SIGMAS_8STEP = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
ENDPOINT = "endpoint_time_embedder."
BASE_TIME = "time_embedder."


@dataclass(frozen=True)
class HyperFlowMeta:
    """The safetensors header of a HyperFlow file, parsed and checked (their `HyperFlowMetadata`)."""

    version: str
    gate: float
    lora_alpha: float
    base_model: str
    base_model_revision: str | None = None
    lora_rank: int | None = None
    sigmas: tuple = DEFAULT_SIGMAS_8STEP
    video_shift: float | None = None
    audio_shift: float | None = None
    compatible_subfolders: tuple = ()
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_header(cls, md: dict | None) -> "HyperFlowMeta":
        md = dict(md or {})
        missing = [k for k in REQUIRED if k not in md]
        if missing or md.get("hyperflow", "").lower() != "true":
            raise ValueError(f"not a HyperFlow file: the header needs {list(REQUIRED)} with hyperflow == 'true'; "
                             f"missing {missing or ['hyperflow=true']}")

        def lst(key, cast):
            if key not in md:
                return None
            v = json.loads(md[key])
            if not isinstance(v, list):
                raise ValueError(f"header {key}: expected a JSON list, got {md[key]!r}")
            return tuple(cast(x) for x in v)

        return cls(
            version=md["hyperflow_version"], gate=float(md["hyperflow_gate"]), lora_alpha=float(md["lora_alpha"]),
            base_model=md["base_model"], base_model_revision=md.get("base_model_revision"),
            lora_rank=int(md["lora_rank"]) if "lora_rank" in md else None,
            sigmas=lst("hyperflow_sigmas", float) or DEFAULT_SIGMAS_8STEP,
            video_shift=float(md["hyperflow_video_shift"]) if "hyperflow_video_shift" in md else None,
            audio_shift=float(md["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in md else None,
            compatible_subfolders=lst("compatible_subfolders", str) or (), raw=md,
        )

    @property
    def model_calls(self) -> int:
        return len(self.sigmas) - 1

    def schedules(self, video_shift: float = 12.0, audio_shift: float = 3.0) -> Schedules:
        """The file's grid through the pipeline's shifts (the release's 12 / 3); warns if the file was trained with
        others, as HyperFlow does."""
        trained = (self.video_shift, self.audio_shift)
        if None not in trained and trained != (video_shift, audio_shift):
            warnings.warn(f"HyperFlow was trained with shifts {trained}, the pipeline uses {(video_shift, audio_shift)}")
        return Schedules.from_grid(self.sigmas, video_shift, audio_shift, two_time=True)


def plan_from_keys(keys) -> dict:
    """{module: {"A": key, "B": key}} in the file's own (public) module names; anything else is an error."""
    plan = {}
    for key in keys:
        m = KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unexpected key {key!r}: only transformer.<module>.lora_A/B.weight keys are allowed")
        plan.setdefault(m["module"], {})[m["matrix"]] = key
    incomplete = [k for k, v in plan.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
    return plan


def _base_name(module: str) -> str:
    """The transformer Linear a module's LoRA sits on: the endpoint embedder is a copy of `time_embedder`."""
    return BASE_TIME + module[len(ENDPOINT):] if module.startswith(ENDPOINT) else module


def check_plan(plan: dict, shapes: dict, base_shapes: dict, meta: HyperFlowMeta) -> int:
    """Every LoRA lands on a Linear weight of the right shape, one rank throughout, scaling 1. `shapes` maps the file's
    keys to their shapes, `base_shapes` our parameter names to theirs. Returns the rank."""
    no_base = [m for m in plan if _base_name(m) + ".weight" not in base_shapes]
    if no_base:
        raise KeyError(f"{len(no_base)} LoRA target(s) have no Linear weight in this transformer, e.g. {no_base[:5]}")
    ranks = set()
    for module, keys in plan.items():
        a, b = tuple(shapes[keys["A"]]), tuple(shapes[keys["B"]])
        out_f, in_f = tuple(base_shapes[_base_name(module) + ".weight"])
        if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
            raise ValueError(f"{module}: lora_A {a} / lora_B {b} do not fit the Linear ({out_f}, {in_f})")
        ranks.add(a[0])
    if len(ranks) != 1:
        raise ValueError(f"all LoRA targets must share one rank, found {sorted(ranks)}")
    rank = ranks.pop()
    if meta.lora_rank is not None and meta.lora_rank != rank:
        raise ValueError(f"header says lora_rank={meta.lora_rank} but the tensors have rank {rank}")
    if meta.lora_alpha / rank != 1.0:
        raise NotImplementedError(f"LoRA scaling alpha/rank = {meta.lora_alpha}/{rank} != 1 (ops.plinear assumes 1)")
    return rank


def attach(params: dict, cfg: dict, tensors: dict, meta: HyperFlowMeta, subfolder: str = "transformer_ref"):
    """Return (params, cfg) with HyperFlow attached; the inputs are not modified. Adapter tensors are cast to their
    base weight's dtype, as PEFT's injection does (bf16 on the blocks, f32 on the time embedder)."""
    if meta.compatible_subfolders and subfolder not in meta.compatible_subfolders:
        raise ValueError(f"this HyperFlow file lists {meta.compatible_subfolders}, not {subfolder}")
    if ENDPOINT + "linear_1.weight" in params:
        raise ValueError("HyperFlow is already attached to these parameters")
    plan = plan_from_keys(tensors)
    check_plan(plan, {k: v.shape for k, v in tensors.items()}, {k: v.shape for k, v in params.items()}, meta)
    out = dict(params)
    for name in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"):
        out[ENDPOINT + name] = params[BASE_TIME + name]  # a copy of the base embedder (shared arrays, no new memory)
    for module, keys in plan.items():
        dtype = params[_base_name(module) + ".weight"].dtype
        out[module + ".lora_A.weight"] = tensors[keys["A"]].astype(dtype)
        out[module + ".lora_B.weight"] = tensors[keys["B"]].astype(dtype)
    return out, {**cfg, "hyperflow_gate": float(meta.gate)}


def read_header(path) -> tuple[HyperFlowMeta, dict]:
    """(meta, {key: shape}) from the safetensors header alone, no tensor data."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    md = header.pop("__metadata__", None)
    return HyperFlowMeta.from_header(md), {k: tuple(v["shape"]) for k, v in header.items()}


def load(path, params: dict, cfg: dict, subfolder: str = "transformer_ref"):
    """Load a HyperFlow .safetensors onto our transformer params: (params, cfg, meta)."""
    path = Path(path)
    tensors, md = mx.load(str(path), return_metadata=True)
    meta = HyperFlowMeta.from_header(md)
    params, cfg = attach(params, cfg, tensors, meta, subfolder)
    return params, cfg, meta
