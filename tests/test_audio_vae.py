"""The audio VAE against diffusers `AutoencoderKLMiniMaxH3Audio` at a tiny config with random weights (weight-norm
folded, conv layouts converted, Kaiser filters read from the checkpoint buffers)."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

from h3mlx.audio_vae import AudioVAE, fold_weight_norm, to_mlx_layout

TINY = dict(encoder_dim=4, encoder_rates=(2, 4), latent_dim=16, latent_channels=4, num_attention_heads=2,
            decoder_dim=16, decoder_rates=(4, 2), decoder_kernel_sizes=(8, 4), resblock_kernel_sizes=(3, 7),
            resblock_dilation_sizes=((1, 3), (1, 3)))


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(11)
    ref = AutoencoderKLMiniMaxH3Audio(**TINY).eval()
    with torch.no_grad():
        for name, prm in ref.named_parameters():
            if name.endswith(("alpha", "beta")):
                prm.copy_(torch.rand_like(prm) * 0.8 + 0.2 if prm.ndim == 3 else torch.randn_like(prm) * 0.3)
            elif name.endswith(("q_bias", "v_bias")) or ("norm" in name and prm.ndim == 1):
                prm.normal_(0.0 if "bias" in name else 1.0, 0.2)
    sd = {k: v.detach().numpy() for k, v in ref.state_dict().items()}
    assert any(k.endswith("weight_g") for k in sd) and any(k.endswith("filter") for k in sd)
    cfg = {k: (tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in v) if isinstance(v, (list, tuple)) else v)
           for k, v in dict(ref.config).items()}
    return ref, AudioVAE(to_mlx_layout(fold_weight_norm(sd)), cfg)


def test_weight_norm_fold_matches_torch(pair):
    ref, _ = pair
    folded = fold_weight_norm({k: v.detach().numpy() for k, v in ref.state_dict().items()})
    for name, mod in ref.named_modules():
        if hasattr(mod, "weight_v"):
            np.testing.assert_allclose(folded[f"{name}.weight"], mod.weight.detach().numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("samples", [256, 243])  # 243 is not a multiple of the hop: exercises the right pad
def test_encode_mean_matches(pair, samples):
    ref, ours = pair
    wave = torch.randn(2, 1, samples) * 0.3
    with torch.no_grad():
        want = ref.encode(wave).latent_dist.mode()  # (B, C, frames)
    got = ours.encode_mean(mx.array(wave.numpy().transpose(0, 2, 1)))
    assert_close(got.transpose(0, 2, 1), want, atol=2e-4, rtol=2e-4, what="posterior mean")


@pytest.mark.parametrize("frames", [5, 12])
def test_decode_matches(pair, frames):
    ref, ours = pair
    lat = torch.randn(2, 4, frames)
    with torch.no_grad():
        want = ref.decode(lat).sample  # (B, 1, frames * 8)
    got = ours.decode(mx.array(lat.numpy().transpose(0, 2, 1)))
    assert got.shape == (2, frames * 8, 1)
    assert_close(got.transpose(0, 2, 1), want, atol=2e-4, rtol=2e-4, what="waveform")
