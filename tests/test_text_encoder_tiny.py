"""The Qwen3-VL conditioner in MLX (mlx-vlm 0.7.11 `hidden_state_at_layer`) against transformers' Qwen3-VL run the
way diffusers 0.42.1 runs it (`encoders.get_qwen3vl_prompt_embeds`: `text_encoder.model`, `mm_token_type_ids`,
`hidden_states[layer]`), on a tiny random config that keeps the released features: interleaved mrope, deepstack,
GQA, a picture and a video in one ref2va presentation built with the released tokenizer and processor."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

from h3mlx.packing import Ref
from h3mlx.presentation import build_presentation

HF = Path(__file__).resolve().parents[2] / "ref-hf"
LAYER = 3

TEXT = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, max_position_embeddings=4096, hidden_act="silu",
            attention_bias=False)
ROPE = dict(rope_type="default", mrope_section=[4, 2, 2], mrope_interleaved=True)
VISION = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3, patch_size=16,
              spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=64,
              deepstack_visual_indexes=[0, 2], hidden_act="gelu_pytorch_tanh")
IDS = dict(image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653)


@pytest.fixture(scope="module")
def setup():
    if not (HF / "processor").exists():
        pytest.skip("released processor not present")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    torch.manual_seed(5)
    cfg = Qwen3VLConfig(text_config=dict(TEXT, rope_parameters=dict(ROPE, rope_theta=5e6)), vision_config=VISION, **IDS)
    hf = Qwen3VLForConditionalGeneration(cfg).eval().float()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.normal_(1.0, 0.3)
    proc = Qwen3VLProcessor.from_pretrained(str(HF / "processor"))
    return hf, proc


def mlx_model(hf):
    from mlx_vlm.models.qwen3_vl import Model, ModelConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel
    from mlx_vlm.utils import sanitize_weights

    params = dict(model_type="qwen3_vl", **IDS, vocab_size=151936,
                  text_config=dict(TEXT, model_type="qwen3_vl_text", rope_theta=5e6, rope_scaling=dict(ROPE)),
                  vision_config=dict(VISION, model_type="qwen3_vl"))
    config = ModelConfig.from_dict(params)
    model = Model(config)
    w = {k: mx.array(v.detach().float().numpy()) for k, v in hf.state_dict().items()}
    w = sanitize_weights(model, w)
    w = sanitize_weights(VisionModel, w, config.vision_config)
    model.load_weights(list(w.items()), strict=False)
    missing = {k for k, _ in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())} - set(w)
    assert not missing, f"mlx-vlm params not loaded: {sorted(missing)[:8]}"
    return model


def presentation(proc):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (96, 64, 3), dtype=np.uint8)
    video = rng.integers(0, 256, (6, 64, 64, 3), dtype=np.uint8)
    img = proc.image_processor(images=[image], return_tensors="pt")
    vid = proc.video_processor(videos=[video], do_sample_frames=False, return_tensors="pt")
    merge = proc.image_processor.merge_size ** 2
    img_counts = [int(g.prod()) // merge for g in img["image_grid_thw"]]
    vid_counts = [int(g[1]) * int(g[2]) // merge for g in vid["video_grid_thw"]]
    ts = [0.2, 1.2, 2.2][: int(vid["video_grid_thw"][0][0])]  # labels only; the frames are already "sampled"
    refs = [Ref("image"), Ref("video", True)]
    ids, tags = build_presentation(proc.tokenizer, "She turns and smiles. Audio: rain.", refs, img_counts, vid_counts, [ts])
    vision = {"pixel_values": img["pixel_values"], "image_grid_thw": img["image_grid_thw"],
              "pixel_values_videos": vid["pixel_values_videos"], "video_grid_thw": vid["video_grid_thw"]}
    return ids, tags, vision


def test_hidden_state_at_layer_matches_transformers(setup):
    hf, proc = setup
    ids, _, vision = presentation(proc)
    with torch.no_grad():
        want = get_qwen3vl_prompt_embeds(hf, proc, ids, vision, text_encoder_layer=LAYER, dtype=torch.float32)
    model = mlx_model(hf)
    got = model.hidden_state_at_layer(
        mx.array([ids]), LAYER,
        pixel_values=mx.array(vision["pixel_values"].numpy()),
        image_grid_thw=mx.array(vision["image_grid_thw"].numpy()),
        pixel_values_videos=mx.array(vision["pixel_values_videos"].numpy()),
        video_grid_thw=mx.array(vision["video_grid_thw"].numpy()),
    )
    assert got.shape == tuple(want.shape)
    n_vis = sum(i in (IDS["image_token_id"], IDS["video_token_id"]) for i in ids)
    print(f"tokens {len(ids)} (vision pads {n_vis}), hidden {got.shape[-1]}")
    assert_close(got, want, atol=1e-3, rtol=1e-3, what=f"hidden_states[{LAYER}]")


def test_wrong_layer_or_dropped_video_grid_differs(setup):
    """Guards: reading one layer early must not pass; and the port's text encoder calls get_rope_index with
    video_grid_thw=None (fine for fl2va), which for a ref2va presentation with a video raises in mlx-vlm 0.7.11."""
    hf, proc = setup
    ids, _, vision = presentation(proc)
    with torch.no_grad():
        want = get_qwen3vl_prompt_embeds(hf, proc, ids, vision, text_encoder_layer=LAYER, dtype=torch.float32)
    model = mlx_model(hf)
    kw = dict(pixel_values=mx.array(vision["pixel_values"].numpy()), image_grid_thw=mx.array(vision["image_grid_thw"].numpy()),
              pixel_values_videos=mx.array(vision["pixel_values_videos"].numpy()),
              video_grid_thw=mx.array(vision["video_grid_thw"].numpy()))
    early = model.hidden_state_at_layer(mx.array([ids]), LAYER - 1, **kw)
    assert float(mx.abs(early - mx.array(want.numpy())).max()) > 1e-2

    # mlx-vlm 0.7.11 fails loudly rather than silently mis-placing video tokens
    with pytest.raises(TypeError):
        model.language_model.get_rope_index(mx.array([ids]), kw["image_grid_thw"], None, None)
