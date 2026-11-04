"""The VAE wrapper (tiling, chunking, single-frame branch, decode cross-fade, conditioning recipe) against diffusers
`AutoencoderKLMiniMaxH3` at a tiny config with random weights."""

import pytest
import torch
from conftest import assert_close, to_mx
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

from h3mlx import vae_ops as V
from h3mlx.vae import VaeConfig, VideoVAE, encode_condition, split_tiles

TINY = dict(latent_channels=4, block_out_channels=(8, 8, 8), layers_per_block=1, spatial_downsample_factors=(2, 2, 1),
            temporal_downsample_factors=(1, 2, 2), norm_num_groups=4, decoder_num_layers=1,
            decoder_num_attention_heads=2, decoder_attention_head_dim=16, decoder_num_register_tokens=2,
            latents_mean=tuple(0.1 * i for i in range(4)), latents_std=tuple(1.0 + 0.2 * i for i in range(4)))
TILE = dict(tile_h=32, tile_w=32, tile_overlap_h=8, tile_overlap_w=8)
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def cl(t):
    return to_mx(t.permute(0, 2, 3, 4, 1).contiguous())


def back(a):
    return a.transpose(0, 4, 1, 2, 3)


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(3)
    ref = AutoencoderKLMiniMaxH3(**TINY).eval()
    with torch.no_grad():
        for name, prm in ref.named_parameters():
            if name.endswith(("scale1", "scale2", "register_tokens")) or "norm" in name:
                prm.normal_()
    ref.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32, tile_sample_min_overlap_height=8,
                      tile_sample_min_overlap_width=8)
    p = {k: (V.conv3d_weight(to_mx(v)) if v.ndim == 5 else to_mx(v)) for k, v in ref.state_dict().items()}
    ours = VideoVAE(p, VaeConfig.from_diffusers(dict(ref.config), **TILE))
    return ref, ours


@pytest.mark.parametrize("length,tile,ov", [(48, 32, 8), (40, 32, 8), (32, 32, 8), (100, 32, 8), (1080, 256, 64)])
def test_split_tiles_matches(pair, length, tile, ov):
    ref, _ = pair
    assert tuple(split_tiles(length, tile, ov, 4)) == tuple(ref._split_tiles(length, tile, ov))


def test_single_frame_uses_spatial_encoder_alone(pair):
    ref, ours = pair
    x = torch.randn(1, 3, 1, 48, 40)
    with torch.no_grad():
        want = ref._encode(x)
    got = ours.encode_moments(cl(x))
    assert got.shape[1] == 1, "a still must give one latent frame, not the 17-copy temporal path"
    assert_close(back(got), want, atol=3e-4, rtol=3e-4, what="still moments (tiled)")


def test_video_chunking_matches(pair):
    ref, ours = pair
    x = torch.randn(1, 3, 22, 32, 32)  # 17 * 1 + 5 frames -> 5 * 1 + 2 latent frames
    with torch.no_grad():
        want = ref._encode(x)
    got = ours.encode_moments(cl(x))
    assert got.shape[1] == 7
    assert_close(back(got), want, atol=3e-4, rtol=3e-4, what="video moments")


@pytest.mark.parametrize("frames", [7, 9, 12, 13])  # 9 and 13 take the padding branch
def test_decode_matches(pair, frames):
    ref, ours = pair
    z = torch.randn(1, 4, frames, 10, 12)
    with torch.no_grad():
        want = ref._decode(z)
    got = ours.decode(cl(z))
    assert_close(back(got), want, atol=3e-4, rtol=3e-4, what=f"decode {frames} latent frames (tiled, chunked)")


def test_encode_condition_recipe_matches(pair):
    ref, ours = pair
    pixels = torch.randint(0, 256, (1, 3, 1, 48, 40), dtype=torch.uint8)
    want = encode_vae_condition(ref, pixels, PIXEL_MEAN, PIXEL_STD, encode_seed=42)

    def torch_noise(shape_cl):  # torch draws in (B, C, F, H, W) order; hand it back channels-last
        b, f, h, w, c = shape_cl
        n = torch.randn((b, c, f, h, w), generator=torch.Generator().manual_seed(42))
        return cl(n)

    got = encode_condition(ours, cl(pixels), PIXEL_MEAN, PIXEL_STD, TINY["latents_mean"], TINY["latents_std"], torch_noise)
    assert_close(back(got), want, atol=2e-3, rtol=2e-3, what="conditioning latents (after the fp16 round trip)")
