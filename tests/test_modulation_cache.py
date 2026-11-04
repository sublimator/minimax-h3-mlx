"""The AdaLN modulation cache gives the same forward and the same denoise run, with the adaln weights dropped."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx, to_mx
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from test_denoise import tiny_layout
from test_dit_mixed_precision import to_mixed
from test_dit_tiny import TINY, packed_layout

from h3mlx import ops
from h3mlx.denoise import Schedules, denoise, schedule_timesteps


@pytest.mark.parametrize("mixed", [False, True])
def test_forward_with_cache_equals_without(mixed):
    torch.manual_seed(4)
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    if mixed:
        model = to_mixed(model)
    p, cfg = state_mx(model), dict(model.config)
    lay = {k: to_mx(v) for k, v in packed_layout().items()}
    x = (mx.random.normal((1, 12, 16)), mx.random.normal((1, 7, 6)), mx.random.normal((1, 5, 20)))
    want = ops.dit_forward(p, cfg, *x, **lay)
    cache = ops.ModulationCache(p, cfg, np.array([1.0, 0.37, 0.05, 0.6, 0.999], dtype=np.float32))  # a superset
    got = ops.dit_forward(ops.ModulationCache.drop_weights(p, cfg), cfg, *x, **lay, cache=cache)
    for g, w in zip(got, want):
        assert float(mx.abs(g.astype(mx.float32) - w.astype(mx.float32)).max()) == 0.0


def test_cache_drops_the_adaln_weights_and_rejects_unknown_t():
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    p, cfg = state_mx(model), dict(model.config)
    small = ops.ModulationCache.drop_weights(p, cfg)
    assert not any("adaln_proj" in k or k.startswith("norm_out.linear") for k in small)
    cache = ops.ModulationCache(p, cfg, [0.5])
    with pytest.raises(KeyError):
        cache.rows(mx.array([0.25]), mx.array([0]))


def test_denoise_with_cache_equals_without():
    torch.manual_seed(8)
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    p, cfg = state_mx(model), dict(model.config)
    lay = tiny_layout()
    v0 = mx.random.normal((len(lay.video_indices), 16))
    a0 = mx.random.normal((len(lay.audio_indices), 6))
    text = mx.random.normal((1, len(lay.text_indices), 20))
    sch = Schedules.make(5)
    want = denoise(p, cfg, lay, text, v0, a0, sch)
    cache = ops.ModulationCache(p, cfg, schedule_timesteps(lay, sch))
    got = denoise(ops.ModulationCache.drop_weights(p, cfg), cfg, lay, text, v0, a0, sch, cache=cache)
    for g, w in zip(got, want):
        assert float(mx.abs(g - w).max()) == 0.0
