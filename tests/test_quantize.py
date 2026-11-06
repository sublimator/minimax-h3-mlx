"""Quantized transformer weights (ours, not the reference's: diffusers runs the transformer in bf16): the wrapper against
a dequantized reference, which weights `quantize_transformer` replaces, and the tiny DiT's error at 8 and 4 bits."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx, to_mx
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from test_dit_mixed_precision import to_mixed
from test_dit_tiny import TINY, packed_layout

from h3mlx import load, ops

GROUP = 32  # MLX groups are 32, 64 or 128 wide; the release's input widths are multiples of 64
QTINY = dict(TINY, hidden_size=64, ffn_dim=128)  # input widths 64, 64 and 128: whole groups


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_weight_is_its_dequantized_matrix(bits):
    w = mx.random.normal((96, 128)).astype(mx.bfloat16)
    q = ops.QuantizedWeight(w, bits, 64)
    assert q.shape == (96, 128) and q.dtype == mx.bfloat16 and q.nbytes < w.nbytes
    x = mx.random.normal((5, 128)).astype(mx.bfloat16)
    want = x @ mx.dequantize(q.wq, q.scales, q.biases, group_size=64, bits=bits).T
    got = ops.linear(x, q)
    assert got.dtype == mx.bfloat16
    g, w32 = got.astype(mx.float32), want.astype(mx.float32)
    # bf16 accumulation order: a few bf16 steps at the size of the row's largest output
    assert bool(mx.all(mx.abs(g - w32) <= 2 ** -6 * mx.abs(w32).max(axis=-1, keepdims=True)))
    assert ops.linear(x.astype(mx.float32), q).dtype == mx.float32  # promotes like `x @ w.T`


def _model(mixed):
    torch.manual_seed(4)
    m = MiniMaxH3Transformer3DModel(**QTINY).eval()
    return to_mixed(m) if mixed else m


def test_quantize_transformer_replaces_only_the_block_linears():
    model = _model(False)
    p, cfg = state_mx(model), dict(model.config)
    before = dict(p)
    out = load.quantize_transformer(p, 8, GROUP)
    replaced = sorted(k for k, v in out.items() if isinstance(v, ops.QuantizedWeight))
    per_block = 6  # to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2
    assert len(replaced) == per_block * (cfg["num_layers"] + cfg["num_refiner_layers"])
    assert all(load.QUANT_TARGETS.search(k) for k in replaced)
    assert all(out[k] is before[k] for k in out if k not in replaced)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("bits,limit", [(8, 0.02), (4, 0.25)])
def test_quantized_forward_stays_close(mixed, bits, limit):
    """Relative error of the whole tiny forward against the unquantized one; also with the modulation cache."""
    model = _model(mixed)
    p, cfg = state_mx(model), dict(model.config)
    lay = {k: to_mx(v) for k, v in packed_layout().items()}
    x = (mx.random.normal((1, 12, 16)), mx.random.normal((1, 7, 6)), mx.random.normal((1, 5, 20)))
    want = ops.dit_forward(p, cfg, *x, **lay)
    qp = load.quantize_transformer(dict(p), bits, GROUP)
    got = ops.dit_forward(qp, cfg, *x, **lay)
    cache = ops.ModulationCache(qp, cfg, np.array([1.0, 0.37, 0.05, 0.6, 0.999], dtype=np.float32))
    cached = ops.dit_forward(ops.ModulationCache.drop_weights(qp, cfg), cfg, *x, **lay, cache=cache)
    for g, c, w in zip(got, cached, want):
        g, c, w = (np.array(a.astype(mx.float32)) for a in (g, c, w))
        rel = np.linalg.norm(g - w) / np.linalg.norm(w)
        print(f"bits {bits} mixed {mixed}: relative error {rel:.4f}")
        assert np.isfinite(g).all() and rel < limit
        assert np.abs(c - g).max() <= 1e-2 * np.abs(g).max()  # the cache changes only the AdaLN arithmetic's order
