"""Scratch probe behind docs/porting-hyperflow.md section 8: MLX vs torch bf16 relative error (each against one f32
torch run), with HyperFlow and without, over 8 seeds (model seed s, file seed s + 50, inputs seed 100 + s), 4 layers of
the tiny config. Run from the repo root: python bench/hyperflow_bf16_seeds.py"""
import sys
sys.path.insert(0, "tests")
import torch
from conftest import to_mx
import test_hyperflow as T
from test_dit_mixed_precision import to_mixed, rel_err
from h3mlx import ops
rows = []
for seed in range(8):
    gen = torch.Generator().manual_seed(100 + seed)
    cfg = dict(T.TINY, num_layers=4)
    model = T.randomised(cfg, seed)
    tensors, meta = T.make_file(model, cfg, seed=seed + 50)
    lay, endpoint = T.pair_layout(gen)
    video, audio, text = torch.randn(1, 12, 16, generator=gen), torch.randn(1, 7, 6, generator=gen), torch.randn(1, 5, 20, generator=gen)
    truth_m = T.torch_hyperflow(model, {k: v.float() for k, v in tensors.items()}, meta)
    mixed = to_mixed(model)
    bf_m = T.torch_hyperflow(mixed, tensors, meta)
    truth_m.time_embedder.r = bf_m.time_embedder.r = endpoint
    with torch.no_grad():
        truth = truth_m(video, audio, text, **lay, return_dict=False)
        t_bf = bf_m(video, audio, text, **lay, return_dict=False)
        base_bf = mixed(video, audio, text, **lay, return_dict=False)
        base_truth = model(video, audio, text, **lay, return_dict=False)
    params, mcfg = T.mlx_hyperflow(mixed, tensors, meta)
    lay_mx = {k: to_mx(v) for k, v in lay.items()}
    m_bf = ops.dit_forward(params, mcfg, to_mx(video), to_mx(audio), to_mx(text), **lay_mx, endpoint=to_mx(endpoint))
    from conftest import state_mx
    m_base = ops.dit_forward(state_mx(mixed), dict(mixed.config), to_mx(video), to_mx(audio), to_mx(text), **lay_mx)
    r = [rel_err(m_bf[i], truth[i]) / rel_err(t_bf[i], truth[i]) for i in (0, 1)]
    rb = [rel_err(m_base[i], base_truth[i]) / rel_err(base_bf[i], base_truth[i]) for i in (0, 1)]
    print(f"seed {seed}: hyperflow mlx/torch video {r[0]:.2f} audio {r[1]:.2f} | base model video {rb[0]:.2f} audio {rb[1]:.2f}")
