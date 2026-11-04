"""Load the released MiniMax H3 components (HF `MiniMaxAI/MiniMax-H3`, root diffusers layout) into h3mlx.

- `transformer_ref/` (or `transformer/`): diffusers names load as-is into `ops.dit_forward`; dtypes as stored (the
  release is bf16 with f32 proj_in/out, audio heads, time_embedder).
- `vae/`: 5-D conv weights moved to MLX layout (O, kt, kh, kw, I).
- `audio_vae/`: weight-norm pairs folded, conv layouts converted, float32.
- `text_encoder/`: Qwen3-VL in mlx-vlm, truncated to the first `keep_layers` decoder layers (MiniMax reads the state
  after layer 50, pre-norm), no lm_head; layers past the cut and the head are never read from disk.

Safetensors are opened lazily (`mx.load`), so only what is kept is materialised.
"""

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
    p = {k: (V.conv3d_weight(v) if v.ndim == 5 else v) for k, v in load_safetensors(d).items()}
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

    def encode(token_ids, vision_inputs):
        kw = {k: mx.array(np.asarray(v)) for k, v in (vision_inputs or {}).items()}
        return model.hidden_state_at_layer(mx.array([token_ids]), keep_layers, **kw)

    return model, encode


def staged_text_encoder(d, keep_layers: int = 50, dtype=None):
    """An `encode(token_ids, vision_inputs)` that loads the text encoder, runs it, and frees it before returning, so
    it is never resident together with the transformer (~51 GB bf16 for 50 layers of the release)."""

    def encode(token_ids, vision_inputs):
        model, run = load_text_encoder(d, keep_layers, dtype)
        out = run(token_ids, vision_inputs)
        mx.eval(out)
        del model, run
        mx.clear_cache()
        return out

    return encode
