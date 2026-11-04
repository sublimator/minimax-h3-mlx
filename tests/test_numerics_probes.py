"""Independent numerical probes; five intentional failures document parity/validation gaps."""

import mlx.core as mx
import pytest
import torch
from conftest import assert_close, state_mx, to_mx
from diffusers.models.transformers import transformer_minimax_h3 as ref
from diffusers.modular_pipelines.minimax_h3 import before_denoise as bd
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

from h3mlx import ops


def test_bf16_silu_matches_single_round_reference():
    # Pointwise, not a GEMM/backend reduction tolerance issue: ref computes activation
    # at opmath precision and rounds once, while bf16 sigmoid * bf16 x rounds twice.
    x = torch.tensor([4.59375], dtype=torch.bfloat16)
    want = torch.nn.functional.silu(x)
    assert float(want[0]) == 4.5625
    assert_close(ops.silu(to_mx(x)), want, atol=0, what="bf16 silu")


def test_mixed_adaln_keeps_fp32_activation_before_bf16_projection():
    m = ref.MiniMaxH3AdaLayerNormModulation(24, 16).bfloat16()
    temb = torch.randn(3, 24) * 3
    with torch.no_grad():
        want = m(temb)
        wrong = m(temb.bfloat16())
    assert any(not torch.equal(a, b) for a, b in zip(want, wrong))
    p = state_mx(m)
    got = ops.adaln_modulation(to_mx(temb), p["linear.weight"], p["linear.bias"], 16)
    for g, w in zip(got, want):
        assert g.dtype == mx.bfloat16
        assert_close(g, w, atol=0.032, rtol=0.008)


def test_partial_rope_bf16_keeps_tail_exact():
    x = torch.randn(2, 11, 3, 128).bfloat16()
    rot = ref.MiniMaxH3RotaryPosEmbed(16)(torch.randn(11, 3) * 9)
    got = ops.apply_rotary(to_mx(x), to_mx(rot[0]), to_mx(rot[1]))
    assert got.dtype == mx.bfloat16
    assert_close(got, ref._apply_rotary_emb(x, *rot), atol=0)
    assert_close(got[..., 96:], x[..., 96:], atol=0)


def test_rectangular_spatiotemporal_patch_order_matches_reference():
    x = torch.arange(2 * 3 * 4 * 6 * 8).reshape(2, 3, 4, 6, 8).float()
    patch = (2, 3, 2)
    got = ops.patchify(to_mx(x), patch)
    assert_close(got, bd.patchify_video_latents(x, patch), atol=0)
    assert_close(ops.unpatchify(got, 3, 4, 6, 8, patch), x, atol=0)


def test_attention_additive_mask_matches_torch():
    q, k, v = (torch.randn(2, 5, 3, 32) for _ in range(3))
    mask = torch.zeros(1, 1, 5, 5)
    mask[..., 3:] = -float("inf")
    want = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    assert_close(ops.attention(to_mx(q), to_mx(k), to_mx(v), to_mx(mask)), want, atol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scheduler_all_steps_including_terminal(dtype):
    scheduler = MiniMaxH3Scheduler(shift=3.0)
    scheduler.set_timesteps(21)
    x = torch.randn(2, 7).to(dtype)
    for i, t in enumerate(scheduler.timesteps):
        velocity = torch.randn_like(x)
        want = scheduler.step(velocity, t, x).prev_sample
        got = ops.flow_step(
            to_mx(velocity), float(t), to_mx(x),
            float(scheduler.sigmas[i]), float(scheduler.sigmas[i + 1]),
        )
        assert_close(got, want, atol=1e-6 if dtype == torch.float32 else 0.016, rtol=0.008)
        x = want


def test_mixed_rmsnorm_output_dtype_matches_torch():
    x = torch.randn(2, 7, 32).bfloat16()
    norm = torch.nn.RMSNorm(32, eps=1e-5)  # fp32 affine weight
    want = norm(x)
    assert want.dtype == torch.bfloat16
    got = ops.rms_norm(to_mx(x), to_mx(norm.weight), 1e-5)
    assert got.dtype == mx.bfloat16


def test_bf16_parameter_conversion_detaches_like_float32_branch():
    weight = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    assert to_mx(weight).dtype == mx.bfloat16


@pytest.mark.parametrize("steps,shift", [(1, 12.0), (5, 0.0)])
def test_scheduler_rejects_invalid_grid_like_reference(steps, shift):
    with pytest.raises(ValueError):
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps)
    with pytest.raises(ValueError):
        ops.flow_sigmas(steps, shift)
