"""ref2va end to end against diffusers' own `MiniMaxH3Blocks` pipeline, with tiny random components and the released
tokenizer/processor: same seed, same references, 3 steps; compares the decoded video and audio.

Tiny: Qwen3-VL with 52 layers (MiniMax reads hidden_states[50]) of width 64, a 2-layer transformer_ref, a 3-level VAE
(ratio 4 spatial, 4 temporal), a hop-8 audio VAE. Reference images at a 64 px short edge, a 64x64 canvas, 124 frames.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx
from PIL import Image

HF = Path(__file__).resolve().parents[2] / "ref-hf"


def build():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from test_audio_vae import TINY as AUDIO
    from test_text_encoder_tiny import IDS, ROPE, TEXT, VISION
    from test_vae import TINY as VAE

    torch.manual_seed(21)
    text_cfg = dict(TEXT, num_hidden_layers=52)
    qwen = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=dict(text_cfg, rope_parameters=dict(ROPE, rope_theta=5e6)), vision_config=VISION, **IDS)
    ).eval().float()
    vae = AutoencoderKLMiniMaxH3(**VAE).eval()
    audio = AutoencoderKLMiniMaxH3Audio(**AUDIO, latents_mean=[0.05 * i for i in range(4)],
                                       latents_std=[1.0 + 0.1 * i for i in range(4)]).eval()
    dit = MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1, ffn_dim=96,
        in_channels=4, audio_in_channels=4, patch_size=(1, 2, 2), text_dim=64, freq_dim=32, time_embed_hidden_dim=40,
        time_embed_dim=24, rope_freq_dim=4,
    ).eval()
    with torch.no_grad():
        for m in (vae, audio, dit):
            for name, prm in m.named_parameters():
                if name.endswith(("scale1", "scale2", "register_tokens")) or ("norm" in name and prm.ndim == 1):
                    prm.normal_(0.0 if name.endswith(("scale1", "scale2")) else 1.0, 0.2)
    tok = AutoTokenizer.from_pretrained(str(HF / "tokenizer"))
    proc = Qwen3VLProcessor.from_pretrained(str(HF / "processor"))
    return dict(qwen=qwen, vae=vae, audio=audio, dit=dit, tok=tok, proc=proc,
                schedulers=(MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)), text_cfg=text_cfg)


def references(rng):
    image = Image.fromarray(rng.integers(0, 256, (80, 64, 3), dtype=np.uint8))
    frames = rng.integers(0, 256, (22, 64, 64, 3), dtype=np.uint8)
    wave = (rng.standard_normal((2, 3200)) * 0.2).astype(np.float32)
    return image, frames, wave


def run_diffusers(t, prompt, image, frames, wave, steps, seed):
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference

    pipe = MiniMaxH3Blocks().init_pipeline()
    pipe.update_components(text_encoder=t["qwen"], tokenizer=t["tok"], processor=t["proc"], vae=t["vae"],
                           audio_vae=t["audio"], transformer_ref=t["dit"], scheduler=t["schedulers"][0],
                           audio_scheduler=t["schedulers"][1])
    pipe.register_to_config(reference_image_short_edge=64, canvas_short_edge=64, canvas_max_pixels=64 * 112)
    refs = [MiniMaxH3ImageReference(image=image),
            MiniMaxH3VideoReference(frames=frames, fps=24.0, audio=torch.from_numpy(wave), sample_rate=32000)]
    out = pipe(prompt=prompt, references=refs, height=64, width=64, num_frames=124, num_inference_steps=steps,
               generator=torch.Generator().manual_seed(seed), output_type="np", output=["videos", "audio"])
    return out


def run_ours(t, prompt, image, frames, wave, steps, seed):
    from mlx_vlm.models.qwen3_vl import Model, ModelConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    from mlx_vlm.utils import sanitize_weights
    from test_text_encoder_tiny import IDS, ROPE, VISION

    from h3mlx import vae_ops as V
    from h3mlx.audio_vae import AudioVAE, fold_weight_norm, to_mlx_layout
    from h3mlx.pipeline import AudioRef, Components, ImageRef, VideoRef, ref2va  # noqa: F401
    from h3mlx.vae import VaeConfig, VideoVAE

    params = dict(model_type="qwen3_vl", **IDS, vocab_size=151936,
                  text_config=dict(t["text_cfg"], model_type="qwen3_vl_text", rope_theta=5e6, rope_scaling=dict(ROPE)),
                  vision_config=dict(VISION, model_type="qwen3_vl"))
    cfg = ModelConfig.from_dict(params)
    qm = Model(cfg)
    w = {k: mx.array(v.detach().float().numpy()) for k, v in t["qwen"].state_dict().items()}
    w = sanitize_weights(VisionModel, sanitize_weights(qm, w), cfg.vision_config)
    qm.load_weights(list(w.items()), strict=False)

    def encode_text(ids, vision):
        kw = {k: mx.array(np.asarray(v)) for k, v in vision.items()}
        return qm.hidden_state_at_layer(mx.array([ids]), 50, **kw)

    vae_p = {k: (V.conv3d_weight(mx.array(v.numpy())) if v.ndim == 5 else mx.array(v.numpy()))
             for k, v in t["vae"].state_dict().items()}
    vcfg = dict(t["vae"].config)
    vae = VideoVAE(vae_p, VaeConfig.from_diffusers(vcfg))
    acfg = {k: (tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in v) if isinstance(v, (list, tuple)) else v)
            for k, v in dict(t["audio"].config).items()}
    audio = AudioVAE(to_mlx_layout(fold_weight_norm({k: v.numpy() for k, v in t["audio"].state_dict().items()})), acfg)
    c = Components(tokenizer=t["tok"], processor=t["proc"], encode_text=encode_text, vae=vae,
                   vae_latents_mean=tuple(vcfg["latents_mean"]), vae_latents_std=tuple(vcfg["latents_std"]),
                   audio_vae=audio, audio_latents_mean=tuple(acfg["latents_mean"]),
                   audio_latents_std=tuple(acfg["latents_std"]), dit_params=state_mx(t["dit"]),
                   dit_cfg=dict(t["dit"].config), reference_image_short_edge=64, canvas_short_edge=64,
                   canvas_max_pixels=64 * 112)
    refs = [ImageRef(image), VideoRef(frames, 24.0, wave, 32000)]
    return ref2va(c, prompt, refs, num_frames=124, height=64, width=64, num_inference_steps=steps, seed=seed)


@pytest.mark.skipif(not (HF / "processor").exists(), reason="released tokenizer/processor not present")
def test_ref2va_end_to_end_matches_diffusers():
    t = build()
    image, frames, wave = references(np.random.default_rng(3))
    prompt = "<Picture 1> walks past the stall from <Video 1>, smiling. Audio: rain."
    with torch.no_grad():
        want = run_diffusers(t, prompt, image, frames, wave, steps=3, seed=7)
    got_video, got_audio, sr = run_ours(t, prompt, image, frames, wave, steps=3, seed=7)
    want_video = np.asarray(want["videos"])[0]
    want_audio = want["audio"].numpy()[0]
    print("video", got_video.shape, want_video.shape, "audio", got_audio.shape, want_audio.shape)
    assert got_video.shape == want_video.shape and got_audio.shape == want_audio.shape
    ev, ea = np.abs(got_video - want_video).max(), np.abs(got_audio - want_audio).max()
    print(f"max |video diff| {ev:.2e}, max |audio diff| {ea:.2e}; video std {want_video.std():.3f}, audio std {want_audio.std():.3f}")
    assert ev < 2e-3 and ea < 2e-3
