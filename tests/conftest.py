"""Helpers: torch <-> mlx, and a closeness check that reports the worst error."""

import mlx.core as mx
import numpy as np
import pytest
import torch


def to_mx(t: torch.Tensor) -> mx.array:
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    return mx.array(t.detach().numpy())


def to_np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().float().numpy()
    if isinstance(a, mx.array):
        return np.array(a.astype(mx.float32))
    return np.asarray(a)


def state_mx(module: torch.nn.Module) -> dict:
    return {k: to_mx(v) for k, v in module.state_dict().items()}


def assert_close(got, want, atol: float, rtol: float = 0.0, what: str = ""):
    g, w = to_np(got), to_np(want)
    assert g.shape == w.shape, f"{what}: shape {g.shape} != {w.shape}"
    err = np.abs(g.astype(np.float64) - w.astype(np.float64))
    bound = atol + rtol * np.abs(w.astype(np.float64))
    worst = float(err.max()) if err.size else 0.0
    assert (err <= bound).all(), f"{what}: max abs err {worst:.3e} (atol {atol:.1e}, rtol {rtol:.1e})"
    return worst


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    mx.random.seed(0)
