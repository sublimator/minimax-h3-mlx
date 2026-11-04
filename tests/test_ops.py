"""Op tests: each MLX op against its diffusers 0.42.1 counterpart on small random tensors, no weights."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, state_mx, to_mx
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers import transformer_minimax_h3 as ref
from diffusers.modular_pipelines.minimax_h3 import before_denoise as bd
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

from h3mlx import ops

F32 = dict(atol=2e-5, rtol=2e-5)


# --- rope ---------------------------------------------------------------------------------------------------------


def test_rope_inv_freq_matches():
    r = ref.MiniMaxH3RotaryPosEmbed(16, 1e4)
    assert_close(ops.rope_inv_freq(16, 1e4), r.inv_freq, atol=0, what="inv_freq")


@pytest.mark.parametrize("freq_dim", [16, 4])
def test_rope_cos_sin_matches(freq_dim):
    r = ref.MiniMaxH3RotaryPosEmbed(freq_dim, 1e4)
    pos = torch.randn(37, 3, dtype=torch.float64) * 20  # rotary positions are float64, non-integer in the reference
    cos_t, sin_t = r(pos)
    cos_m, sin_m = ops.rope_cos_sin(to_mx(pos.float()), ops.rope_inv_freq(freq_dim, 1e4))
    assert cos_m.shape == (37, 6 * freq_dim)
    assert_close(cos_m, cos_t, atol=1e-5, what="cos")
    assert_close(sin_m, sin_t, atol=1e-5, what="sin")


def test_apply_rotary_matches_real_dims():
    x = torch.randn(2, 37, 4, 128)
    cos, sin = ref.MiniMaxH3RotaryPosEmbed(16, 1e4)(torch.randn(37, 3) * 20)
    want = ref._apply_rotary_emb(x, cos, sin)
    got = ops.apply_rotary(to_mx(x), to_mx(cos), to_mx(sin))
    assert_close(got, want, **F32, what="rotary")
    assert_close(got[..., 96:], x[..., 96:], atol=0, what="pass-through channels")


# --- norms and AdaLN ----------------------------------------------------------------------------------------------


def test_rms_norm_matches():
    n = torch.nn.RMSNorm(64, eps=1e-5)
    torch.nn.init.normal_(n.weight)
    x = torch.randn(2, 11, 64) * 3
    assert_close(ops.rms_norm(to_mx(x), to_mx(n.weight), 1e-5), n(x), **F32, what="rmsnorm")


def test_adaln_modulation_and_row_addressing():
    hidden, temb_dim, num_t, seq = 16, 24, 2, 9
    m = ref.MiniMaxH3AdaLayerNormModulation(time_embed_dim=temb_dim, hidden_size=hidden)
    temb = torch.randn(num_t, temb_dim)
    want = m(temb)
    got = ops.adaln_modulation(to_mx(temb), to_mx(m.linear.weight), to_mx(m.linear.bias), hidden)
    assert len(got) == 6
    for i, (g, w) in enumerate(zip(got, want)):
        assert g.shape == (num_t * 3, hidden)
        assert_close(g, w, **F32, what=f"chunk {i}")
    # rows: adaln_indices = timestep_indices * 3 + token_tags
    tidx = torch.randint(0, num_t, (seq,))
    tags = torch.randint(0, 3, (seq,))
    rows = tidx * 3 + tags
    x = torch.randn(1, seq, hidden)
    want_mod = x * (1.0 + want[1].index_select(0, rows)) + want[0].index_select(0, rows)
    got_mod = ops.modulate(to_mx(x), got[0], got[1], to_mx(rows))
    assert_close(got_mod, want_mod, **F32, what="modulate")


def test_norm_out_matches():
    hidden, temb_dim, num_t, seq = 16, 24, 3, 10
    m = ref.MiniMaxH3AdaLayerNormOut(hidden_size=hidden, time_embed_dim=temb_dim, eps=1e-5)
    torch.nn.init.normal_(m.norm.weight)
    x, temb = torch.randn(2, seq, hidden), torch.randn(num_t, temb_dim)
    tidx = torch.randint(0, num_t, (seq,))
    want = m(x, temb, tidx)
    p = state_mx(m)
    got = ops.norm_out(to_mx(x), to_mx(temb), p["norm.weight"], p["linear.weight"], p["linear.bias"], to_mx(tidx), 1e-5)
    assert_close(got, want, **F32, what="norm_out")


# --- attention ----------------------------------------------------------------------------------------------------


def test_attention_unmasked_matches_torch_sdpa():
    q, k, v = (torch.randn(2, 50, 4, 32) for _ in range(3))
    want = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    got = ops.attention(to_mx(q), to_mx(k), to_mx(v))
    assert_close(got, want, **F32, what="sdpa")


def test_attention_padded_to_64_with_mask_equals_unpadded():
    """The reference pads the packed sequence to a multiple of 64 and masks the tail; diffusers runs unpadded.
    Both must give the same rows for the used part."""
    used, total = 50, 64
    q, k, v = (torch.randn(1, used, 4, 32) for _ in range(3))
    pad = lambda t: torch.cat([t, torch.randn(1, total - used, 4, 32) * 5], dim=1)  # noqa: E731  junk in the tail
    unpadded = ops.attention(to_mx(q), to_mx(k), to_mx(v))
    padded = ops.attention(to_mx(pad(q)), to_mx(pad(k)), to_mx(pad(v)), mask=ops.key_padding_mask(used, total))
    assert_close(padded[:, :used], unpadded, **F32, what="padded+mask vs unpadded")
    unmasked = ops.attention(to_mx(pad(q)), to_mx(pad(k)), to_mx(pad(v)))
    assert float(mx.abs(unmasked[:, :used] - unpadded).max()) > 1e-3, "junk tail must matter without the mask"


def test_attention_module_with_qk_norm_and_rope():
    hidden, heads, dim_head, freq_dim, seq = 48, 2, 32, 4, 13
    a = ref.MiniMaxH3Attention(hidden_size=hidden, heads=heads, dim_head=dim_head, qk_norm_eps=1e-5)
    torch.nn.init.normal_(a.norm_q.weight)
    torch.nn.init.normal_(a.norm_k.weight)
    x = torch.randn(2, seq, hidden)
    rot = ref.MiniMaxH3RotaryPosEmbed(freq_dim, 1e4)(torch.randn(seq, 3) * 10)
    want = a(x, rot)
    got = ops.attn_block(to_mx(x), state_mx(a), heads, 1e-5, (to_mx(rot[0]), to_mx(rot[1])))
    assert_close(got, want, **F32, what="attention module")


def test_swiglu_ff_matches():
    ff = FeedForward(32, inner_dim=80, activation_fn="swiglu", bias=False)
    x = torch.randn(2, 7, 32)
    assert_close(ops.swiglu_ff(to_mx(x), state_mx(ff)), ff(x), **F32, what="swiglu ff")


# --- timestep embedding -------------------------------------------------------------------------------------------


def test_timestep_embedding_matches():
    proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    emb = TimestepEmbedding(in_channels=256, time_embed_dim=64, out_dim=32)
    t = torch.tensor([0.0, 0.031, 0.5, 0.97, 1.0])  # unscaled, in [0, 1]
    assert_close(ops.timestep_sinusoid(to_mx(t), 256), proj(t), atol=1e-6, what="sinusoid")
    assert_close(ops.timestep_embedding(to_mx(t), state_mx(emb), 256), emb(proj(t)), **F32, what="time embedder")


# --- patchify and position grids ----------------------------------------------------------------------------------


def test_patchify_matches_and_unpatchify_inverts():
    lat = torch.randn(2, 24, 5, 6, 8)
    want = bd.patchify_video_latents(lat, (1, 2, 2))
    got = ops.patchify(to_mx(lat), (1, 2, 2))
    assert got.shape == (2 * 5 * 3 * 4, 96)
    assert_close(got, want, atol=0, what="patchify")
    assert_close(ops.unpatchify(got, 24, 5, 6, 8), lat, atol=0, what="unpatchify roundtrip")


@pytest.mark.parametrize("lh,lw", [(30, 54), (36, 64), (44, 44), (60, 34)])
def test_position_grids_match_exactly(lh, lw):
    sqrt_area = np.sqrt(lh * lw)
    assert np.array_equal(ops.spatial_position_grid(lw, 2, sqrt_area), bd._spatial_position_grid(lw, 2, sqrt_area).numpy())
    assert np.array_equal(ops.temporal_position_grid(31, 3.5), bd._temporal_position_grid(31, 3.5).numpy())
    g, wg = ops.frame_position_grid(lh, lw, 2, 2)
    g_ref, wg_ref = bd._frame_position_grid(lh, lw, 2, 2)
    assert np.array_equal(g, g_ref.numpy()) and np.array_equal(wg, wg_ref.numpy())


# --- scheduler ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,shift", [(5, 12.0), (9, 12.0), (21, 12.0), (21, 3.0), (50, 12.0), (2, 3.0)])
def test_flow_sigmas_match(n, shift):
    s = MiniMaxH3Scheduler(shift=shift)
    s.set_timesteps(n)
    assert np.array_equal(ops.flow_sigmas(n, shift), s.sigmas.numpy()), "sigma grid must match bit for bit"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flow_step_matches(dtype):
    s = MiniMaxH3Scheduler(shift=12.0)
    s.set_timesteps(9)
    sig = s.sigmas.numpy()
    x = torch.randn(3, 96).to(dtype)
    v = torch.randn(3, 96).to(dtype)
    for i, t in enumerate(s.timesteps[:3]):
        want = s.step(v, t, x).prev_sample
        got = ops.flow_step(to_mx(v), float(t), to_mx(x), float(sig[i]), float(sig[i + 1]))
        tol = dict(atol=1e-6) if dtype == torch.float32 else dict(atol=1.6e-2, rtol=8e-3)  # one bf16 ulp
        assert_close(got, want, **tol, what=f"step {i} {dtype}")
        x = want


# --- one transformer block ----------------------------------------------------------------------------------------


def test_one_transformer_block_matches():
    hidden, heads, dim_head, ffn, temb_dim, freq_dim, seq, num_t = 48, 2, 32, 96, 24, 4, 17, 2
    blk = ref.MiniMaxH3TransformerBlock(hidden, heads, dim_head, ffn, temb_dim, 1e-5, 1e-5)
    for n in (blk.norm1, blk.norm2, blk.attn.norm_q, blk.attn.norm_k):
        torch.nn.init.normal_(n.weight)
    x, temb = torch.randn(2, seq, hidden), torch.randn(num_t, temb_dim)
    rows = torch.randint(0, num_t, (seq,)) * 3 + torch.randint(0, 3, (seq,))
    rot = ref.MiniMaxH3RotaryPosEmbed(freq_dim, 1e4)(torch.randn(seq, 3) * 10)
    want = blk(x, temb, rows, rot)
    got = ops.dit_block(to_mx(x), to_mx(temb), to_mx(rows), (to_mx(rot[0]), to_mx(rot[1])), state_mx(blk), heads, hidden)
    assert_close(got, want, atol=1e-4, rtol=1e-4, what="block")


def test_torch_linspace_f32_bit_exact():
    for n in range(2, 600):
        assert np.array_equal(ops.torch_linspace_f32(1.0, 0.0, n), torch.linspace(1.0, 0.0, n).numpy()), n
