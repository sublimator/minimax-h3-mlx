"""The AdaLN modulation cache: exactly equal to the uncached path on the small fixtures below; in general equal in
maths and within torch-bf16 accuracy (the error-budget test), with the adaln weights dropped."""

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


def test_cache_error_budget_over_a_50_point_schedule():
    """Not bit-identical in general (review D2: the cache multiplies all timesteps at once, a step only
    its own; bf16 rounding then differs, e.g. step 6 here). What must hold: at every step, cached and uncached MLX are as
    close to an f32 torch run as torch's own bf16 run is (ratio <= 1.25; observed <= 1.06)."""
    from test_dit_mixed_precision import rel_err

    from h3mlx.denoise import row_timesteps

    torch.manual_seed(43)
    cfg = dict(TINY, hidden_size=96, attention_head_dim=128, ffn_dim=256, freq_dim=64, time_embed_hidden_dim=384,
               time_embed_dim=192, rope_freq_dim=16)
    f32 = MiniMaxH3Transformer3DModel(**cfg).eval()
    mixed = to_mixed(f32)
    p, c = state_mx(mixed), dict(mixed.config)
    lay, sch = tiny_layout(), Schedules.make(50)
    cache = ops.ModulationCache(p, c, schedule_timesteps(lay, sch))
    small = ops.ModulationCache.drop_weights(p, c)
    xt = (torch.randn(1, len(lay.video_indices), 16), torch.randn(1, len(lay.audio_indices), 6),
          torch.randn(1, len(lay.text_indices), 20))
    x = tuple(to_mx(t) for t in xt)
    names = ("token_tags", "video_indices", "audio_indices", "text_indices")
    kw = {k: mx.array(getattr(lay, k)) for k in names} | {"position_ids": mx.array(lay.position_ids.astype(np.float32))}
    tk = {k: torch.from_numpy(getattr(lay, k)) for k in names} | {"position_ids": torch.from_numpy(lay.position_ids)}
    for t, ta in zip(sch.video_timesteps, sch.audio_timesteps):
        u, inv = row_timesteps(lay, float(t), float(ta), max(float(t), 0.999), 1.0)
        with torch.no_grad():
            truth = f32(*xt, torch.from_numpy(u), torch.from_numpy(inv), **tk, return_dict=False)
            tb = mixed(*xt, torch.from_numpy(u), torch.from_numpy(inv), **tk, return_dict=False)
        step = dict(kw, timestep=mx.array(u), timestep_indices=mx.array(inv))
        unc = ops.dit_forward(p, c, *x, **step)
        cac = ops.dit_forward(small, c, *x, **step, cache=cache)
        for j in range(2):
            budget = 1.25 * rel_err(tb[j], truth[j]) + 1e-4
            assert rel_err(unc[j], truth[j]) <= budget and rel_err(cac[j], truth[j]) <= budget, (float(t), j)
