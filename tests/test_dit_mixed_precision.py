"""Mixed precision as the checkpoint ships it (block stack bf16; proj_in/out, audio heads, time_embedder, rope f32).

bf16 can't match bit for bit across frameworks, so both bf16 runs are scored against the same f32 run: the MLX error
must not be much worse than torch's own bf16 error.
"""

import copy

import mlx.core as mx
import numpy as np
import torch
from conftest import state_mx, to_mx, to_np
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from test_dit_tiny import TINY, packed_layout

from h3mlx import ops

KEEP_F32 = MiniMaxH3Transformer3DModel._keep_in_fp32_modules


def to_mixed(model):
    m = copy.deepcopy(model)
    for name, prm in m.named_parameters():
        if not any(k in name for k in KEEP_F32):
            prm.data = prm.data.to(torch.bfloat16)
    return m


def rel_err(got, want):
    g, w = to_np(got).astype(np.float64), to_np(want).astype(np.float64)
    return float(np.linalg.norm(g - w) / np.linalg.norm(w))


def test_mixed_precision_tracks_torch():
    torch.manual_seed(1)
    cfg = dict(TINY, num_layers=4)
    model = MiniMaxH3Transformer3DModel(**cfg).eval()
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if "norm" in name and prm.ndim == 1:
                prm.normal_()
    mixed = to_mixed(model)
    assert mixed.proj_in.weight.dtype == torch.float32 and mixed.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    lay = packed_layout(n_text=9, n_video=40, n_audio=11)
    video, audio, text = torch.randn(1, 40, 16), torch.randn(1, 11, 6), torch.randn(1, 9, 20)
    with torch.no_grad():
        truth = model(video, audio, text, **lay, return_dict=False)
        t_bf = mixed(video, audio, text, **lay, return_dict=False)
    lay_mx = {k: to_mx(v) for k, v in lay.items()}
    m_bf = ops.dit_forward(state_mx(mixed), dict(mixed.config), to_mx(video), to_mx(audio), to_mx(text), **lay_mx)
    for i, name in enumerate(("video", "audio")):
        e_torch, e_mlx = rel_err(t_bf[i], truth[i]), rel_err(m_bf[i], truth[i])
        print(f"{name}: torch bf16 rel err {e_torch:.4f}, mlx bf16 rel err {e_mlx:.4f}")
        assert m_bf[i].dtype == mx.float32, "heads are f32, so the outputs are f32"
        assert e_mlx <= 1.5 * e_torch + 1e-3, f"{name}: mlx {e_mlx:.4f} vs torch {e_torch:.4f}"
