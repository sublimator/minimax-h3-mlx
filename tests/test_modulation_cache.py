"""The AdaLN modulation cache: bit-identical to the uncached path, since each call's tables are computed from exactly
the timesteps that call passes (one product over the union of all timesteps rounded differently: the error-budget
test's larger model showed it), with the adaln weights dropped."""

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
from h3mlx.denoise import Schedules, denoise, schedule_calls


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
    calls = [np.float32([0.6, 0.999]), np.array(lay["timestep"]), np.float32([0.5])]  # this call among others
    cache = ops.ModulationCache(p, cfg, calls)
    got = ops.dit_forward(ops.ModulationCache.drop_weights(p, cfg), cfg, *x, **lay, cache=cache)
    for g, w in zip(got, want):
        assert float(mx.abs(g.astype(mx.float32) - w.astype(mx.float32)).max()) == 0.0


def test_cache_drops_the_adaln_weights_and_rejects_unknown_calls():
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    p, cfg = state_mx(model), dict(model.config)
    small = ops.ModulationCache.drop_weights(p, cfg)
    assert not any("adaln_proj" in k or k.startswith("norm_out.linear") for k in small)
    cache = ops.ModulationCache(p, cfg, [np.float32([0.5, 1.0]), np.float32([0.5, 1.0]), np.float32([0.25])])
    assert len(cache.blocks) == 2 and list(cache.t) == [0.25, 0.5, 1.0]  # a repeated call is built once
    cache.call(mx.array([0.5, 1.0]))
    for unknown in ([0.5], [1.0, 0.5], [0.5, 1.0, 0.25]):  # a call is its exact timestep array, order included
        with pytest.raises(KeyError):
            cache.call(mx.array(unknown))


def test_cache_keys_are_one_row_of_float32_whatever_the_caller_passes():
    """`denoise` passes float32 arrays; a list, a float64 array or an mx.array of the same values is the same call and
    embeds the same float32 values; a bf16 mx.array is read too (0.5 and 1.0 are exact in bf16); anything but one row
    is refused."""
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    p, cfg = state_mx(model), dict(model.config)
    a = ops.ModulationCache(p, cfg, [[0.6, 0.999], [0.5, 1.0]])
    b = ops.ModulationCache(p, cfg, [np.float32([0.6, 0.999]), np.float64([0.5, 1.0])])

    def tables(cache, t):
        blocks, out = cache.call(t)
        return [x for tab in blocks for x in tab] + list(out)

    for t in ([0.6, 0.999], np.array([0.6, 0.999]), mx.array(np.float32([0.6, 0.999])),
              mx.array([0.5, 1.0]).astype(mx.bfloat16)):
        for x, y in zip(tables(a, t), tables(b, t)):
            assert bool(mx.array_equal(x, y))
    with pytest.raises(ValueError, match="one row"):
        ops.ModulationCache(p, cfg, [np.float32([[0.6, 0.999]])])
    with pytest.raises(ValueError, match="one row"):
        a.call(mx.array([[0.6, 0.999]]))


def test_cache_with_drop_pops_the_same_weights_in_place_and_builds_the_same_tables():
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    p, cfg = state_mx(model), dict(model.config)
    calls = [np.float32([1.0, 0.5]), np.float32([0.999])]
    kept = ops.ModulationCache(dict(p), cfg, calls)
    q = dict(p)
    dropped = ops.ModulationCache(q, cfg, calls, drop=True)
    assert set(q) == set(ops.ModulationCache.drop_weights(p, cfg))
    for c in range(len(calls)):
        for a, b in zip(kept.blocks[c] + [kept.out[c]], dropped.blocks[c] + [dropped.out[c]]):
            for x, y in zip(a, b):
                assert bool(mx.array_equal(x, y))


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
    cache = ops.ModulationCache(p, cfg, schedule_calls(lay, sch))
    got = denoise(ops.ModulationCache.drop_weights(p, cfg), cfg, lay, text, v0, a0, sch, cache=cache)
    for g, w in zip(got, want):
        assert float(mx.abs(g - w).max()) == 0.0


def test_cache_is_bit_identical_over_a_50_point_schedule():
    """A larger mixed-precision model where one product over the union of the timesteps rounded differently from the
    step's own (review D2, e.g. step 6). Per-call tables must make every step bit-identical; the uncached path must
    stay as close to an f32 torch run as torch's own bf16 run is (ratio <= 1.25; observed <= 1.06)."""
    from test_dit_mixed_precision import rel_err

    from h3mlx.denoise import row_timesteps

    torch.manual_seed(43)
    cfg = dict(TINY, hidden_size=96, attention_head_dim=128, ffn_dim=256, freq_dim=64, time_embed_hidden_dim=384,
               time_embed_dim=192, rope_freq_dim=16)
    f32 = MiniMaxH3Transformer3DModel(**cfg).eval()
    mixed = to_mixed(f32)
    p, c = state_mx(mixed), dict(mixed.config)
    lay, sch = tiny_layout(), Schedules.make(50)
    cache = ops.ModulationCache(p, c, schedule_calls(lay, sch))
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
            assert bool(mx.array_equal(cac[j], unc[j])), (float(t), j)
            assert rel_err(unc[j], truth[j]) <= 1.25 * rel_err(tb[j], truth[j]) + 1e-4, (float(t), j)
