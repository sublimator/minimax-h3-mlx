"""LoRA adapters for the MLX transformer: PEFT's scaling and initialisation, attaching, merging, and the files.

Ported from PEFT 0.21.2 (`tuners/lora/layer.py`: the scaling in `update_layer`, `reset_lora_parameters`,
`Linear.get_delta_weight`, `Linear.merge`, `Linear.forward`; `tuners/lora/model.py` and `utils/other.py`: the rank and
alpha patterns) and from diffusers main at `d3d8af89` (`loaders/peft.py`: `load_lora_adapter`, `save_lora_adapter`;
`utils/peft_utils.py`: `get_peft_kwargs`; `loaders/lora_pipeline.py`: `MiniMaxH3LoraLoaderMixin.lora_state_dict`;
`loaders/lora_conversion_utils.py`: `_convert_non_diffusers_minimax_h3_lora_to_diffusers`).

An adapter is two matrices per Linear, `lora_A` (rank, in) and `lora_B` (out, rank), and an alpha; the Linear then
computes `W x + (alpha / rank) * B (A x)` (`ops.plinear`). HyperFlow's own loader stays in `hyperflow.py`: its file
also carries a second time embedder and a sigma grid, and its scaling is 1.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import hyperflow, ops

METADATA_KEY = "lora_adapter_metadata"  # diffusers' LORA_ADAPTER_METADATA_KEY (loaders/lora_base.py)
PREFIXES = ("transformer", "transformer_ref")  # the component names a diffusers pipeline file puts before each key
KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")
BLOCK_TARGETS = hyperflow.BLOCK_TARGETS
SCALING = ".lora_scaling"
ORIGINAL_PREFIXES = ("diffusion_model.", "blocks.", "token_refiner.blocks.", "final_layer.", "lora_unet_")
UNSUPPORTED = ("use_dora", "use_rslora", "lora_bias", "fan_in_fan_out")


@dataclass
class Adapter:
    """LoRA factors for some of the transformer's Linears, in PEFT's parametrisation.

    `tensors` maps `<module>.lora_A.weight` to (rank, in) and `<module>.lora_B.weight` to (out, rank), with diffusers
    module names and no component prefix, so the dict merges straight into the transformer's weights. `alpha` maps each
    module to its lora_alpha. A module's rank is its `lora_A`'s first axis; ranks and alphas may differ per module."""

    tensors: dict
    alpha: dict

    @property
    def modules(self) -> list[str]:
        return sorted(self.alpha)

    def rank(self, module: str) -> int:
        return int(self.tensors[module + ".lora_A.weight"].shape[0])

    def scaling(self, module: str) -> float:
        """`lora_alpha / r`, as PEFT's `update_layer` sets it (without rsLoRA)."""
        return self.alpha[module] / self.rank(module)


# --- where a LoRA can act ------------------------------------------------------------------------------------------


def wired_shapes(cfg: dict) -> dict:
    """{module: (out, in)} for every Linear the forward runs through `ops.plinear`: the six matrices of each block and
    refiner block, and the time embedder's two. (`hyperflow.wired_shapes` without HyperFlow's endpoint embedder.)"""
    return {m: s for m, s in hyperflow.wired_shapes(cfg).items() if not m.startswith(hyperflow.ENDPOINT)}


def block_modules(cfg: dict, targets=BLOCK_TARGETS) -> list[str]:
    """The attention and feed-forward Linears of every block and refiner block, by module name."""
    out = [f"transformer_blocks.{i}.{t}" for i in range(cfg["num_layers"]) for t in targets]
    return out + [f"token_refiner.refiner_blocks.{i}.{t}" for i in range(cfg["num_refiner_layers"]) for t in targets]


def check(adapter: Adapter, cfg: dict) -> None:
    """Every module is a Linear the forward applies a LoRA to (PEFT would wrap any `nn.Linear` a file names; a LoRA on
    `proj_in` or `adaln_proj` would be silently ignored here, so it is refused), and each pair fits it."""
    if not adapter.alpha:
        raise ValueError("no LoRA tensors in this adapter")
    wired = wired_shapes(cfg)
    unwired = sorted(m for m in adapter.alpha if m not in wired)
    if unwired:
        raise ValueError(f"{len(unwired)} LoRA target(s) are not run through ops.plinear here (the port would ignore "
                         f"them), e.g. {unwired[:5]}")
    for module in adapter.modules:
        a = tuple(adapter.tensors[module + ".lora_A.weight"].shape)
        b = tuple(adapter.tensors[module + ".lora_B.weight"].shape)
        out_f, in_f = wired[module]
        if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
            raise ValueError(f"{module}: lora_A {a} / lora_B {b} do not fit the Linear ({out_f}, {in_f})")
        if a[0] <= 0:
            raise ValueError(f"{module}: LoRA rank must be positive, got {a[0]}")


# --- initialisation (PEFT's `reset_lora_parameters`) -----------------------------------------------------------------


def kaiming_uniform(shape, generator) -> np.ndarray:
    """`nn.init.kaiming_uniform_(weight, a=math.sqrt(5))` for a (rank, in) `lora_A`, drawn from a torch generator so the
    values are torch's: uniform in +-bound, with bound = sqrt(3) * gain / sqrt(fan_in) and gain = sqrt(2 / (1 + a^2)),
    which is 1 / sqrt(in). The bound is computed in the same order as torch computes it, in Python floats."""
    import torch

    fan_in = shape[1]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return torch.empty(tuple(shape), dtype=torch.float32).uniform_(-bound, bound, generator=generator).numpy()


def init(cfg: dict, rank: int, alpha: float | None = None, modules=None, seed: int = 0) -> Adapter:
    """A new adapter as PEFT initialises one (`init_lora_weights=True`): `lora_A` Kaiming-uniform, `lora_B` zero, so it
    starts as a no-op. Both are float32 whatever the base weights' dtype (a bf16 parameter is too coarse to train).
    `alpha` defaults to `rank` (scaling 1). `modules` defaults to `block_modules(cfg)`; they are drawn in sorted order
    from one torch generator seeded with `seed` (PEFT draws in the model's module order and from the global generator,
    so the same seed does not give PEFT's values)."""
    import torch

    if rank <= 0:
        raise ValueError(f"`rank` should be a positive integer value but the value passed is {rank}")
    wired = wired_shapes(cfg)
    modules = sorted(block_modules(cfg) if modules is None else modules)
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for module in modules:
        if module not in wired:
            raise ValueError(f"{module} is not a Linear the forward runs through ops.plinear")
        out_f, in_f = wired[module]
        tensors[module + ".lora_A.weight"] = mx.array(kaiming_uniform((rank, in_f), generator))
        tensors[module + ".lora_B.weight"] = mx.zeros((out_f, rank), dtype=mx.float32)
    return Adapter(tensors, {m: rank if alpha is None else alpha for m in modules})


# --- attaching and merging -------------------------------------------------------------------------------------------


def attach(params: dict, cfg: dict, adapter: Adapter, keep_dtype: bool = False) -> dict:
    """The transformer's weights with the adapter beside them, for `ops.plinear`; `params` is not modified.

    By default each matrix is cast to its base weight's dtype, as PEFT's injection does (`load_lora_adapter`, and
    `hyperflow.attach`). `keep_dtype=True` leaves the adapter as it is: float32 factors over a bf16 base, what
    `cast_training_params` gives a trainer (and what diffusers' `low_cpu_mem_usage=True` loading gives a float32 file).
    A module whose scaling is not 1 also gets `<module>.lora_scaling`."""
    check(adapter, cfg)
    taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
    if taken:
        raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: one adapter per Linear")
    out = dict(params)
    for module in adapter.modules:
        dtype = params[module + ".weight"].dtype
        for matrix in ("A", "B"):
            key = f"{module}.lora_{matrix}.weight"
            out[key] = adapter.tensors[key] if keep_dtype else adapter.tensors[key].astype(dtype)
        scaling = adapter.scaling(module)
        if scaling != 1.0:
            out[module + SCALING] = mx.array(scaling, dtype=mx.float32)
    return out


def delta_weight(a: mx.array, b: mx.array, scaling: float) -> mx.array:
    """PEFT's `Linear.get_delta_weight` on CPU: `(B @ A) * scaling`, with half-precision factors multiplied in float32
    and the product cast back to their dtype."""
    dtype = b.dtype
    half = dtype in (mx.float16, mx.bfloat16)
    if half:
        a, b = a.astype(mx.float32), b.astype(mx.float32)
    out = (b @ a) * scaling
    return out.astype(dtype) if half else out


def merge(params: dict, cfg: dict, adapter: Adapter) -> dict:
    """The transformer's plain weights with the adapter added into them (PEFT's `Linear.merge`: `weight +=
    delta_weight`, in the weight's dtype); `params` is not modified, and the result has its keys and nothing of the
    adapter left to apply.

    Attaching and merging are alternatives here. Weights that already carry a LoRA on one of the adapter's Linears are
    refused: the attached factors would stay in the result, and `plinear` would apply them on top of the merged ones.
    (A PEFT layer holds both states itself, and its forward skips an adapter once it is merged.) Not for quantized
    weights."""
    check(adapter, cfg)
    taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
    if taken:
        raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: merge into the plain "
                         "weights, not into attached ones (the adapter would be applied twice)")
    out = dict(params)
    for module in adapter.modules:
        w = params[module + ".weight"]
        if isinstance(w, ops.QuantizedWeight):
            raise ValueError(f"{module}: can't merge a LoRA into a quantized weight; merge first, then quantize")
        a, b = adapter.tensors[module + ".lora_A.weight"], adapter.tensors[module + ".lora_B.weight"]
        out[module + ".weight"] = (w + delta_weight(a, b, adapter.scaling(module))).astype(w.dtype)
    return out


# --- adapters in the original checkpoint's key names ---------------------------------------------------------------

_FLATTENED = [  # musubi-tuner's `lora_unet_` names, every `.` collapsed to `_`
    (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
    (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
    (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
    (r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj", r"token_refiner.blocks.\1.attn.\2_proj"),
    (r"token_refiner_blocks_(\d+)_mlp_fc([12])", r"token_refiner.blocks.\1.mlp.fc\2"),
    (r"(video|audio)_patch_proj", r"\1_patch_proj"),
    (r"condition_proj", "condition_proj"),
    (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
    (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
    (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
]
_STANDALONE = {  # the projections outside the block stack
    "video_patch_proj": "proj_in",
    "audio_patch_proj": "audio_proj_in",
    "condition_proj": "context_embedder",
    "time_embedder.proj_in": "time_embedder.linear_1",
    "time_embedder.proj_out": "time_embedder.linear_2",
    "final_layer.adaln_proj.linear": "norm_out.linear",
    "final_layer.video_out": "proj_out",
    "final_layer.audio_out": "audio_proj_out",
}
_BLOCKS = [  # (key pattern, original prefix, diffusers prefix)
    (r"blocks\.(\d+)\.", "blocks", "transformer_blocks"),
    (r"token_refiner\.blocks\.(\d+)\.", "token_refiner.blocks", "token_refiner.refiner_blocks"),
]
HEAD_DIM = 128  # every released partition's head width, which diffusers' converter assumes for a DiffSynth file


def is_original_layout(keys) -> bool:
    """Whether any key carries one of the original checkpoint's module names, none of which a diffusers name shares
    (`MiniMaxH3LoraLoaderMixin.lora_state_dict`'s `is_non_diffusers_format`)."""
    return any(k.startswith(ORIGINAL_PREFIXES) for k in keys)


def convert_original(state_dict: dict, head_dim: int = HEAD_DIM) -> dict:
    """diffusers' `_convert_non_diffusers_minimax_h3_lora_to_diffusers`: a LoRA trained against the original
    checkpoint's module names, onto `transformer.<diffusers module>.lora_A/B.weight` keys.

    * `diffusion_model.` prefix (ai-toolkit) or none; musubi-tuner's flattened `lora_unet_` names; DiffSynth-Studio's
      `.default.` infix.
    * kohya's `lora_down` / `lora_up`, with each module's `.alpha` folded into the two matrices (so the result has
      scaling 1).
    * the fused `attn.qkv_proj` split into `to_q` / `to_k` / `to_v`: `lora_A` shared, `lora_B` cut in thirds, after
      de-interleaving a DiffSynth file's per-head rows. `attn.out_proj` becomes `to_out.0`.
    * `mlp.fc1` becomes `ff.net.0.proj` with the two halves of its output rows swapped (the original fuses
      [gate; value], diffusers' SwiGLU [value; gate]); `mlp.fc2` becomes `ff.net.2`.
    * the patch, condition, timestep and final projections onto their diffusers names.

    Anything left over is an error. The input dict is not modified. Two differences from diffusers, both on purpose:
    an `.alpha` that is zero, negative or not finite is refused (their fold loops about a thousand times to all-zero
    matrices on zero, never returns on a negative one, and writes NaNs on NaN), and `head_dim`, the attention head
    width a DiffSynth file's rows are de-interleaved by, is an argument (theirs is fixed at 128, its default here)."""
    sd = {k.removeprefix("diffusion_model."): v for k, v in state_dict.items()}

    # DiffSynth-Studio keeps the raw checkpoint's per-head-interleaved fused QKV; its `.default.` infix is the tell
    is_diffsynth = any(".lora_A.default.weight" in k or ".lora_B.default.weight" in k for k in sd)
    if is_diffsynth:
        sd = {k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): v for k, v in sd.items()}

    if any(k.startswith("lora_unet_") for k in sd):
        unflattened = {}
        for key, value in sd.items():
            module, _, suffix = key.removeprefix("lora_unet_").partition(".")
            dotted = None
            for pattern, replacement in _FLATTENED:
                if re.fullmatch(pattern, module):
                    dotted = re.sub(pattern, replacement, module)
                    break
            if dotted is None:
                raise ValueError(f"`{key}` does not name a MiniMax-H3 module in musubi-tuner's flattened `lora_unet_` layout.")
            unflattened[f"{dotted}.{suffix}"] = value
        sd = unflattened

    is_kohya = any(".lora_down.weight" in k for k in sd)
    down_suffix = ".lora_down.weight" if is_kohya else ".lora_A.weight"
    up_suffix = ".lora_up.weight" if is_kohya else ".lora_B.weight"

    def pull(base):
        """Pop the (lora_A, lora_B) pair of a module with any `.alpha` folded in, or None if absent."""
        if base + down_suffix not in sd:
            return None
        down = sd.pop(base + down_suffix)
        up = sd.pop(base + up_suffix)
        #@@start alpha-fold
        alpha = sd.pop(base + ".alpha", None)
        if alpha is not None:
            if not alpha.item() > 0 or math.isinf(alpha.item()):
                raise ValueError(f"`{base}.alpha` is {alpha.item()}: a LoRA alpha must be positive and finite")
            # the forward scales by alpha / rank; split that factor between down and up
            scale_down, scale_up = alpha.item() / down.shape[0], 1.0
            while scale_down * 2 < scale_up:
                scale_down *= 2
                scale_up /= 2
            down, up = ops.times_scalar(down, scale_down), ops.times_scalar(up, scale_up)
        #@@end alpha-fold
        return down, up

    out = {}
    for source, target in _STANDALONE.items():
        pair = pull(source)
        if pair is not None:
            out[f"{target}.lora_A.weight"], out[f"{target}.lora_B.weight"] = pair

    for pattern, source_prefix, target_prefix in _BLOCKS:
        num_layers = 0
        for key in sd:
            match = re.match(pattern, key)
            if match:
                num_layers = max(num_layers, int(match.group(1)) + 1)

        for i in range(num_layers):
            source, target = f"{source_prefix}.{i}", f"{target_prefix}.{i}"

            #@@start qkv-split
            qkv = pull(f"{source}.attn.qkv_proj")
            if qkv is not None:
                down, up = qkv
                if up.shape[0] % 3 != 0:
                    raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not divisible "
                                     "by 3. This is not a fused MiniMax-H3 QKV projection.")
                if is_diffsynth:
                    # [head0: q k v, head1: q k v, ...] -> [q_all; k_all; v_all]
                    if up.shape[0] % (3 * head_dim) != 0:
                        raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not a "
                                         f"multiple of 3 * {head_dim}. This is not a per-head-interleaved MiniMax-H3 "
                                         "QKV projection.")
                    num_heads = up.shape[0] // (3 * head_dim)
                    grouped = up.reshape(num_heads, 3 * head_dim, up.shape[1])
                    up = mx.concatenate([s.reshape(num_heads * head_dim, up.shape[1])
                                         for s in mx.split(grouped, 3, axis=1)], axis=0)
                for proj, up_proj in zip(("to_q", "to_k", "to_v"), mx.split(up, 3, axis=0)):
                    out[f"{target}.attn.{proj}.lora_A.weight"] = down
                    out[f"{target}.attn.{proj}.lora_B.weight"] = up_proj
            #@@end qkv-split

            #@@start fc1-swap
            fc1 = pull(f"{source}.mlp.fc1")
            if fc1 is not None:
                down, up = fc1
                if up.shape[0] % 2 != 0:
                    raise ValueError(f"`{source}.mlp.fc1` has {up.shape[0]} output rows, which is not even. This is "
                                     "not a fused MiniMax-H3 SwiGLU projection.")
                up_gate, up_value = mx.split(up, 2, axis=0)
                out[f"{target}.ff.net.0.proj.lora_A.weight"] = down
                out[f"{target}.ff.net.0.proj.lora_B.weight"] = mx.concatenate([up_value, up_gate], axis=0)
            #@@end fc1-swap

            for source_module, target_module in (("attn.out_proj", "attn.to_out.0"), ("mlp.fc2", "ff.net.2"),
                                                 ("adaln_proj.linear", "adaln_proj.linear")):
                pair = pull(f"{source}.{source_module}")
                if pair is not None:
                    out[f"{target}.{target_module}.lora_A.weight"], out[f"{target}.{target_module}.lora_B.weight"] = pair

    if sd:
        raise ValueError(f"`state_dict` should be empty at this point but has {sorted(sd)}. This may be an unsupported "
                         "MiniMax-H3 LoRA layout.")
    return {f"transformer.{k}": v for k, v in out.items()}


# --- files -------------------------------------------------------------------------------------------------------------


def pattern_key(pattern_keys, key_to_match: str) -> str:
    """PEFT's `get_pattern_key`: the first `rank_pattern` / `alpha_pattern` key that matches the module name as a
    regular expression on its last dotted parts, else the name itself."""
    for key in pattern_keys:
        if re.match(rf"(.*\.)?({key})$", key_to_match):
            return key
    return key_to_match


def _component(tensors: dict, prefix: str | None) -> tuple[dict, str | None]:
    """The keys of one component with its prefix removed (`load_lora_adapter(prefix=...)`'s filter), and that prefix.
    With no `prefix` given the file must hold one component, prefixed or not."""
    found = [p for p in PREFIXES if any(k.startswith(p + ".") for k in tensors)]
    if prefix is None:
        if len(found) > 1:
            raise ValueError(f"this file holds LoRAs for {found}: say which with `prefix=`")
        prefix = found[0] if found else None
        other = () if prefix is None else [k for k in tensors if not k.startswith(prefix + ".")]
        if other:
            raise ValueError(f"keys with and without the `{prefix}.` prefix in one file, e.g. {sorted(other)[:3]}")
    if prefix is None:
        return dict(tensors), None
    return {k.removeprefix(prefix + "."): v for k, v in tensors.items() if k.startswith(prefix + ".")}, prefix


def read(tensors: dict, header: dict | None = None, prefix: str | None = None, head_dim: int = HEAD_DIM) -> Adapter:
    """An adapter from a LoRA file's tensors and its safetensors header (`__metadata__`), as diffusers would load it.

    Keys: `transformer.<module>.lora_A/B.weight` (what `save_lora_weights` writes and `load_lora_adapter` reads by
    default), the same without the prefix (what `save_lora_adapter` writes), PEFT's own state dict (a `.default.`
    infix), or the original checkpoint's names (`convert_original`, which folds any `.alpha` tensors into the matrices;
    `head_dim` is passed to it).

    Alpha, per module: from the `lora_adapter_metadata` diffusers writes (`r`, `lora_alpha`, `rank_pattern`,
    `alpha_pattern`, resolved as PEFT resolves them); else from a plain `alpha` entry in the header, taken as one
    network alpha for every module; else the module's own rank (scaling 1), which is `get_peft_kwargs`' rule. DoRA,
    rsLoRA, a LoRA bias and `fan_in_fan_out` are refused, not ignored."""
    header = dict(header or {})
    tensors = dict(tensors)
    if any("dora_scale" in k or "lora_magnitude_vector" in k for k in tensors):
        raise NotImplementedError("DoRA adapters are not supported")  # diffusers drops `dora_scale` with a warning
    has_alpha_tensors = any(k.endswith(".alpha") for k in tensors)
    if is_original_layout(tensors):
        tensors = convert_original(tensors, head_dim)
    elif any(".default.weight" in k for k in tensors) and not any(k.startswith(tuple(p + "." for p in PREFIXES)) for k in tensors):
        tensors = {k.replace(".default.", "."): v for k, v in tensors.items()}  # a PEFT state dict, adapter "default"
    tensors, prefix = _component(tensors, prefix)

    by_module = {}
    for key in tensors:
        m = KEY_RE.match(key)
        if m is None:
            raise ValueError(f"unexpected key {key!r}: only <module>.lora_A/B.weight keys are supported")
        by_module.setdefault(m["module"], set()).add(m["matrix"])
    incomplete = sorted(k for k, v in by_module.items() if v != {"A", "B"})
    if incomplete:
        raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
    if not by_module:
        raise ValueError("no LoRA tensors" + (f" under the `{prefix}.` prefix" if prefix else "") + " in this file")
    ranks = {m: int(tensors[m + ".lora_B.weight"].shape[1]) for m in by_module}

    config = json.loads(header[METADATA_KEY]) if header.get(METADATA_KEY) else None
    if config is not None and prefix is not None:
        config = {k.removeprefix(prefix + "."): v for k, v in config.items() if k.startswith(prefix + ".")}
    if config is not None:
        if "r" not in config or "lora_alpha" not in config:
            raise ValueError(f"the file's {METADATA_KEY} has no `r` / `lora_alpha`" + (f" under `{prefix}.`" if prefix else ""))
        on = [k for k in UNSUPPORTED if config.get(k)]
        if on:
            raise NotImplementedError(f"this adapter was saved with {on}, which the port does not implement")
        rank_pattern, alpha_pattern = config.get("rank_pattern") or {}, config.get("alpha_pattern") or {}
        alpha = {}
        for module, rank in ranks.items():
            r = rank_pattern.get(pattern_key(rank_pattern, module), config["r"])
            if r != rank:
                raise ValueError(f"{module}: the metadata says rank {r} but lora_B has rank {rank}")
            alpha[module] = alpha_pattern.get(pattern_key(alpha_pattern, module), config["lora_alpha"])
        return Adapter(tensors, alpha)

    network_alpha = None
    if "alpha" in header and not has_alpha_tensors:
        try:
            network_alpha = float(header["alpha"])
        except ValueError:  # diffusers warns and loads with alpha == rank
            network_alpha = None
    return Adapter(tensors, {m: (rank if network_alpha is None else network_alpha) for m, rank in ranks.items()})


def load(path, prefix: str | None = None, head_dim: int = HEAD_DIM) -> Adapter:
    """`read` on a .safetensors file (the tensors stay lazy until used)."""
    tensors, header = mx.load(str(path), return_metadata=True)
    return read(tensors, header, prefix, head_dim)


def _number(x):
    return int(x) if float(x).is_integer() else float(x)


def config(adapter: Adapter) -> dict:
    """The `LoraConfig` arguments that describe the adapter, as `get_peft_kwargs` derives them: the most common rank and
    alpha, and a `^`-anchored pattern entry for each module that differs. The other entries state what the port
    implements (no dropout, DoRA, rsLoRA or LoRA bias). A subset of what `LoraConfig.to_dict()` holds, so it loads
    with `LoraConfig(**config)` on PEFT versions that add or drop other fields."""
    ranks = {m: adapter.rank(m) for m in adapter.modules}
    alphas = {m: _number(adapter.alpha[m]) for m in adapter.modules}
    r = Counter(ranks.values()).most_common()[0][0]
    lora_alpha = Counter(alphas.values()).most_common()[0][0]
    return {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": {f"^{m}": v for m, v in ranks.items() if v != r},
        "alpha_pattern": {f"^{m}": v for m, v in alphas.items() if v != lora_alpha},
        "target_modules": adapter.modules,
        "bias": "none",
        "lora_dropout": 0.0,
        "use_dora": False,
        "use_rslora": False,
        "lora_bias": False,
        "fan_in_fan_out": False,
    }


def save(path, adapter: Adapter, prefix: str | None = "transformer") -> None:
    """Write the adapter as diffusers does: `<prefix>.<module>.lora_A/B.weight` tensors and, in the safetensors header,
    `format: pt` plus the `LoraConfig` arguments as JSON under `lora_adapter_metadata` (each key prefixed too). That is
    `save_lora_weights`' file, which `load_lora_adapter` reads by default. `prefix=None` writes `save_lora_adapter`'s:
    no prefixes, read back with `load_lora_adapter(..., prefix=None)`. Tensors keep their dtype."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"expected a .safetensors path, got {path}")
    if not adapter.alpha:
        raise ValueError("no LoRA tensors in this adapter")
    lead = "" if prefix is None else prefix + "."
    tensors = {lead + k: v for k, v in adapter.tensors.items()}
    meta = {lead + k: v for k, v in config(adapter).items()}
    mx.save_safetensors(str(path), tensors, metadata={"format": "pt", METADATA_KEY: json.dumps(meta, indent=2, sort_keys=True)})
