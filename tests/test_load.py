"""Loaders against checkpoints saved the way the release is (diffusers/transformers `save_pretrained`, safetensors,
mixed bf16/f32 transformer, weight-norm audio VAE), at tiny sizes."""

import copy

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, fixture, state_mx, to_mx

from h3mlx import load, ops


def test_transformer_loads_mixed_precision_like_the_release(tmp_path):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from test_dit_tiny import TINY, packed_layout

    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    mixed = copy.deepcopy(model)
    for name, prm in mixed.named_parameters():
        if not any(k in name for k in MiniMaxH3Transformer3DModel._keep_in_fp32_modules):
            prm.data = prm.data.to(torch.bfloat16)
    mixed.save_pretrained(tmp_path / "transformer_ref", max_shard_size="40KB")  # several shards + index
    assert (tmp_path / "transformer_ref" / "diffusion_pytorch_model.safetensors.index.json").exists()
    p, cfg = load.load_transformer(tmp_path / "transformer_ref")
    assert p["proj_in.weight"].dtype == mx.float32 and p["transformer_blocks.0.attn.to_q.weight"].dtype == mx.bfloat16
    lay = {k: to_mx(v) for k, v in packed_layout().items()}
    x = (mx.random.normal((1, 12, 16)), mx.random.normal((1, 7, 6)), mx.random.normal((1, 5, 20)))
    got = ops.dit_forward(p, cfg, *x, **lay)
    want = ops.dit_forward(state_mx(mixed), dict(mixed.config), *x, **lay)
    assert_close(got[0], want[0], atol=0, what="loaded == in-memory (same bf16 weights)")


def test_vae_loads(tmp_path):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from test_vae import TINY

    ref = AutoencoderKLMiniMaxH3(**TINY).eval()
    ref.save_pretrained(tmp_path / "vae")
    vae, mean, std = load.load_vae(tmp_path / "vae")
    assert mean == tuple(TINY["latents_mean"]) and std == tuple(TINY["latents_std"])
    x = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        want = ref._encode(x)
    got = vae.encode_moments(to_mx(x.permute(0, 2, 3, 4, 1).contiguous()))
    assert_close(got.transpose(0, 4, 1, 2, 3), want, atol=3e-4, rtol=3e-4, what="loaded vae encode")


def test_audio_vae_loads(tmp_path):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    from test_audio_vae import TINY

    ref = AutoencoderKLMiniMaxH3Audio(**TINY, latents_mean=[0.0] * 4, latents_std=[1.0] * 4).eval()
    with torch.no_grad():
        for name, prm in ref.named_parameters():
            if name.endswith(("alpha", "beta")):
                prm.copy_(torch.rand_like(prm) * 0.5 + 0.3)
    ref.save_pretrained(tmp_path / "audio_vae")
    avae, _, _ = load.load_audio_vae(tmp_path / "audio_vae")
    lat = torch.randn(2, 4, 6)
    with torch.no_grad():
        want = ref.decode(lat).sample
    got = avae.decode(mx.array(lat.numpy().transpose(0, 2, 1)))
    assert_close(got.transpose(0, 2, 1), want, atol=2e-4, rtol=2e-4, what="loaded audio decode")


@pytest.mark.parametrize("keep", [3])
def test_text_encoder_loads_truncated(tmp_path, keep):
    from test_text_encoder_tiny import IDS, ROPE, TEXT, VISION
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(2)
    hf = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=dict(TEXT, rope_parameters=dict(ROPE, rope_theta=5e6)), vision_config=VISION, **IDS)
    ).eval().float()
    hf.save_pretrained(tmp_path / "text_encoder")
    model, encode = load.load_text_encoder(tmp_path / "text_encoder", keep_layers=keep)
    assert len(model.language_model.model.layers) == keep
    ids = [151643, 9707, 11, 1879, 0, 220, 33975]
    with torch.no_grad():
        want = hf.model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states[keep]
    got = encode(ids, None)
    assert_close(got, want, atol=1e-4, rtol=1e-4, what=f"hidden_states[{keep}] from a truncated load")
    np.testing.assert_equal(got.shape, tuple(want.shape))


def test_release_text_encoder_config_parses():
    """The release's own text_encoder/config.json (a release fixture, HF @ 42ed227) gives the architecture MiniMax reads."""
    import json
    
    from mlx_vlm.models.qwen3_vl import ModelConfig

    raw = json.loads(fixture("MiniMax-H3", "text_encoder", "config.json").read_text())
    c = ModelConfig.from_dict(load.qwen_params(raw))
    t, v = c.text_config, c.vision_config
    assert (t.num_hidden_layers, t.hidden_size, t.num_attention_heads, t.num_key_value_heads, t.head_dim) == (64, 5120, 64, 8, 128)
    assert t.rope_theta == 5000000 and t.rope_scaling["mrope_section"] == [24, 20, 20] and t.rope_scaling["mrope_interleaved"]
    assert (v.depth, v.hidden_size, v.out_hidden_size, v.patch_size, v.deepstack_visual_indexes) == (27, 1152, 5120, 16, [8, 16, 24])
    assert (c.image_token_id, c.video_token_id) == (151655, 151656)



def test_eval_each_layer_keeps_outputs_and_attributes(tmp_path):
    """The per-layer evaluation must not change the hidden state, and mlx-vlm's own `layers[0].self_attn` lookup must
    still work."""
    from test_text_encoder_tiny import IDS, ROPE, TEXT, VISION
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(9)
    hf = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=dict(TEXT, rope_parameters=dict(ROPE, rope_theta=5e6)), vision_config=VISION, **IDS)
    ).eval().float()
    hf.save_pretrained(tmp_path / "te")
    model, encode = load.load_text_encoder(tmp_path / "te", keep_layers=3)
    layers = model.language_model.model.layers
    assert all(type(layer).__name__.endswith("EvalEach") for layer in layers)
    assert layers[0].self_attn is not None
    ids = [151643, 9707, 11, 1879, 0, 220]
    with torch.no_grad():
        want = hf.model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states[3]
    assert_close(encode(ids, None), want, atol=1e-4, rtol=1e-4, what="hidden state with per-layer eval")
