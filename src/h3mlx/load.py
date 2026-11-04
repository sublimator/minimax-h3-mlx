"""Load the released MiniMax H3 components (HF `MiniMaxAI/MiniMax-H3`, root diffusers layout) into h3mlx.

- `transformer_ref/` (or `transformer/`): diffusers names load as-is into `ops.dit_forward`; dtypes as stored (the
  release is bf16 with f32 proj_in/out, audio heads, time_embedder).
- `vae/`: 5-D conv weights moved to MLX layout (O, kt, kh, kw, I).
- `audio_vae/`: weight-norm pairs folded, conv layouts converted, float32.
- `text_encoder/`: Qwen3-VL in mlx-vlm, truncated to the first `keep_layers` decoder layers (MiniMax reads the state
  after layer 50, pre-norm), no lm_head; layers past the cut and the head are never read from disk.

Safetensors are opened lazily (`mx.load`), so only what is kept is materialised.
"""

import gc
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import vae_ops as V
from .audio_vae import AudioVAE, fold_weight_norm, to_mlx_layout
from .vae import VaeConfig, VideoVAE


def _shards(d: Path) -> list[Path]:
    idx = sorted(d.glob("*.safetensors.index.json"))
    if idx:
        files = sorted(set(json.loads(idx[0].read_text())["weight_map"].values()))
        return [d / f for f in files]
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {d}")
    return files


def load_safetensors(d: Path, keep=None) -> dict:
    out = {}
    for f in _shards(Path(d)):
        for k, v in mx.load(str(f)).items():
            if keep is None or keep(k):
                out[k] = v
    return out


def _config(d: Path) -> dict:
    return json.loads((Path(d) / "config.json").read_text())


def load_transformer(d) -> tuple[dict, dict]:
    cfg = {k: v for k, v in _config(d).items() if not k.startswith("_")}
    cfg["patch_size"] = tuple(cfg["patch_size"])
    return load_safetensors(d), cfg


def load_vae(d, **tiling) -> tuple[VideoVAE, tuple, tuple]:
    cfg = _config(d)
    # contiguous once: a transposed view would be copied again inside every tile's graph (121-209 tiles per reference)
    p = {k: (mx.contiguous(V.conv3d_weight(v)) if v.ndim == 5 else v) for k, v in load_safetensors(d).items()}
    return VideoVAE(p, VaeConfig.from_diffusers(cfg, **tiling)), tuple(cfg["latents_mean"]), tuple(cfg["latents_std"])


def load_audio_vae(d) -> tuple[AudioVAE, tuple, tuple]:
    cfg = _config(d)
    cfg = {k: (tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in v) if isinstance(v, (list, tuple)) else v)
           for k, v in cfg.items()}
    sd = {k: np.array(v.astype(mx.float32)) for k, v in load_safetensors(d).items()}
    return AudioVAE(to_mlx_layout(fold_weight_norm(sd)), cfg), tuple(cfg["latents_mean"]), tuple(cfg["latents_std"])


def qwen_params(raw: dict) -> dict:
    """A Qwen3-VL config.json (the release's transformers-4.57 spelling with `rope_scaling` + `rope_theta`, or
    transformers 5's `rope_parameters`) -> the dict mlx-vlm's ModelConfig.from_dict takes."""
    params = dict(raw)
    tc = dict(raw["text_config"])
    if "rope_parameters" in tc and "rope_scaling" not in tc:
        rp = dict(tc.pop("rope_parameters"))
        tc.setdefault("rope_theta", rp.pop("rope_theta", 5e6))
        tc["rope_scaling"] = rp
    tc.setdefault("model_type", "qwen3_vl_text")
    params["text_config"] = tc
    params["vision_config"] = dict(raw["vision_config"], model_type="qwen3_vl")
    params.setdefault("vocab_size", tc["vocab_size"])
    return params


_EVAL_EACH: dict = {}


def eval_each_layer(layers) -> None:
    """Make every decoder layer evaluate its output, so the 50-layer pass is 50 small GPU graphs instead of one. A single
    long graph at 7,000+ tokens tripped macOS's GPU watchdog (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
    in a pre-flight probe. The layer keeps its class's attributes and parameters: only `__call__` changes."""
    for layer in layers:
        base = type(layer)
        if base in _EVAL_EACH.values():
            continue
        if base not in _EVAL_EACH:
            def call(self, *a, _base=base, **k):
                out = _base.__call__(self, *a, **k)
                mx.eval(out)
                return out
            _EVAL_EACH[base] = type(base.__name__ + "EvalEach", (base,), {"__call__": call})
        layer.__class__ = _EVAL_EACH[base]


def load_text_encoder(d, keep_layers: int = 50, dtype=None):
    """mlx-vlm Qwen3-VL from the release's `text_encoder/`, with only decoder layers [0, keep_layers) and no lm_head.
    Returns (model, encode(token_ids, vision_inputs) -> (1, n, hidden) after `keep_layers` layers, pre-norm)."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_vl import Model, ModelConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    from mlx_vlm.utils import sanitize_weights

    config = ModelConfig.from_dict(qwen_params(_config(d)))
    n_layers = config.text_config.num_hidden_layers
    if not 0 < keep_layers <= n_layers:
        raise ValueError(f"keep_layers must be in (0, {n_layers}], got {keep_layers}")

    config.text_config.num_hidden_layers = keep_layers  # never build the layers past the cut
    model = Model(config)
    model.language_model.pop("lm_head", None)  # unused; would be a random 5120 x 151936 matrix
    layer_re = re.compile(r"language_model\.layers\.(\d+)\.")

    def keep(k):
        if "lm_head" in k:
            return False
        m = layer_re.search(k)
        return m is None or int(m.group(1)) < keep_layers

    w = load_safetensors(d, keep)
    w = sanitize_weights(VisionModel, sanitize_weights(model, w), config.vision_config)
    if dtype is not None:
        w = {k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v for k, v in w.items()}
    model.load_weights(list(w.items()), strict=False)
    want = {k for k, _ in tree_flatten(model.parameters()) if "lm_head" not in k}
    missing = want - set(w)
    if missing:
        raise ValueError(f"text encoder weights missing: {sorted(missing)[:6]} (+{max(0, len(missing) - 6)})")
    eval_each_layer(model.language_model.model.layers)

    def encode(token_ids, vision_inputs):
        kw = {k: mx.array(np.asarray(v)) for k, v in (vision_inputs or {}).items()}
        return model.hidden_state_at_layer(mx.array([token_ids]), keep_layers, **kw)

    return model, encode


def staged_text_encoder(d, keep_layers: int = 50, dtype=None):
    """An `encode(token_ids, vision_inputs)` that loads the text encoder, runs it, and drops its own references before
    returning (~51 GB bf16 for 50 layers of the release). It does not unload anything else: keep the transformer
    unevaluated (lazy `mx.load`) until text encoding is done if the two must not be resident together."""

    def encode(token_ids, vision_inputs):
        model, run = load_text_encoder(d, keep_layers, dtype)
        out = run(token_ids, vision_inputs)
        mx.eval(out)
        del model, run
        gc.collect()  # a reference cycle would otherwise keep ~51 GB alive into the denoise
        mx.clear_cache()
        return out

    return encode


KEEP_F32 = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")


def check_headers(weights: Path, keep_layers: int = 50) -> dict:
    """Read only the safetensors headers (mx.load is lazy: shapes and dtypes, no data) and check them against what the
    code will build: transformer_ref's keys against its index and its dtypes against the release's precision split;
    the text encoder's kept parameters against the mlx-vlm model built from its config (built lazily, never evaluated);
    the VAEs' conv weights are 5-D. Raises ValueError listing what is wrong."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_vl import Model, ModelConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    from mlx_vlm.utils import sanitize_weights

    weights = Path(weights)
    problems, report = [], {}

    t = load_safetensors(weights / "transformer_ref")
    index = json.loads(next((weights / "transformer_ref").glob("*.index.json")).read_text())["weight_map"]
    if set(t) != set(index):
        problems.append(f"transformer_ref keys differ from its index: {len(set(t) ^ set(index))}")
    for k, v in t.items():
        want = mx.float32 if k.split(".")[0] in KEEP_F32 else mx.bfloat16
        if v.dtype != want:
            problems.append(f"transformer_ref {k}: {v.dtype}, expected {want}")
    cfg = _config(weights / "transformer_ref")
    h, inner = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"]
    for k, shape in {"transformer_blocks.0.attn.to_q.weight": (inner, h), "proj_in.weight": (h, cfg["in_channels"] * 4),
                     "transformer_blocks.49.ff.net.0.proj.weight": (2 * cfg["ffn_dim"], h)}.items():
        if k in t and tuple(t[k].shape) != shape:
            problems.append(f"transformer_ref {k}: shape {tuple(t[k].shape)}, expected {shape}")
    report["transformer_ref"] = {"tensors": len(t), "gb": round(sum(v.nbytes for v in t.values()) / 1e9, 2)}

    config = ModelConfig.from_dict(qwen_params(_config(weights / "text_encoder")))
    config.text_config.num_hidden_layers = keep_layers
    model = Model(config)  # parameters are created lazily and never evaluated here
    model.language_model.pop("lm_head", None)
    layer_re = re.compile(r"language_model\.layers\.(\d+)\.")
    w = load_safetensors(weights / "text_encoder",
                         lambda k: "lm_head" not in k and not ((m := layer_re.search(k)) and int(m.group(1)) >= keep_layers))
    w = sanitize_weights(VisionModel, sanitize_weights(model, w), config.vision_config)
    want = dict(tree_flatten(model.parameters()))
    missing = sorted(set(want) - set(w))
    extra = sorted(k for k in set(w) - set(want) if "rotary" not in k)
    bad_shape = [k for k in want if k in w and tuple(w[k].shape) != tuple(want[k].shape)]
    for name, items in (("missing", missing), ("unexpected", extra), ("wrong shape", bad_shape)):
        if items:
            problems.append(f"text_encoder {name}: {items[:5]} (+{max(0, len(items) - 5)})")
    report["text_encoder"] = {"kept_tensors": len(want), "gb": round(sum(w[k].nbytes for k in want if k in w) / 1e9, 2),
                              "dtypes": sorted({str(w[k].dtype) for k in want if k in w})}

    for sub in ("vae", "audio_vae"):
        sd = load_safetensors(weights / sub)
        report[sub] = {"tensors": len(sd), "gb": round(sum(v.nbytes for v in sd.values()) / 1e9, 2),
                       "dtypes": sorted({str(v.dtype) for v in sd.values()})}
    if not any(v.ndim == 5 for v in load_safetensors(weights / "vae").values()):
        problems.append("vae: no 5-D conv weights")
    if problems:
        raise ValueError("weight headers: " + "; ".join(problems))
    return report
