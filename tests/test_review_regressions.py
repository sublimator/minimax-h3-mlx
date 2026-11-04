"""Regressions from a code review: input validation, the text encoder's dropped
head, staged loading, bf16 ViT weights, one-step schedules."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from PIL import Image

from h3mlx import load, vae_ops as V
from h3mlx.denoise import Schedules
from h3mlx.pipeline import AudioRef, ImageRef, VideoRef, setup


class FakeVae:
    class cfg:
        spatial_ratio = 16


class C:  # just what setup() reads
    vae = FakeVae()
    dit_cfg = {"patch_size": (1, 2, 2)}
    canvas_short_edge, canvas_max_pixels = 768, 768 * 1344
    reference_image_short_edge = 2048
    audio_sampling_rate = 32000

    @property
    def canvas_multiple(self):
        return 32


IMG = ImageRef(Image.new("RGB", (64, 48)))


@pytest.mark.parametrize("h,w", [(None, 1024), (768, None), (720, 1280), (480, 860)])
def test_setup_rejects_bad_canvas(h, w):
    with pytest.raises(ValueError):
        setup(C(), [IMG], h, w, 124)


@pytest.mark.parametrize("refs", [[], [AudioRef(np.zeros((2, 32000), np.float32), 32000)], [IMG] * 10,
                                  [VideoRef(np.zeros((30, 32, 32, 3), np.uint8))] * 4, [object()]])
def test_setup_rejects_bad_references(refs):
    with pytest.raises(ValueError):
        setup(C(), refs, None, None, 124)


def test_setup_accepts_a_good_request():
    h, w, n, refs = setup(C(), [IMG], 480, 864, 120)
    assert (h, w, n) == (480, 864, 124) and refs[0].image.size == (2720, 2048)


def test_one_step_schedule_raises():  # review H3R1
    with pytest.raises(ValueError):
        Schedules.make(1)


def test_vit_decoder_accepts_bf16_weights():  # review item 5
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoViTDecoder3d

    dec = MiniMaxH3VideoViTDecoder3d(in_channels=6, out_channels=3, patch_size=4, patch_size_t=2, num_layers=1,
                                     num_attention_heads=2, attention_head_dim=16, num_register_tokens=2).eval()
    p = {k: mx.array(v.numpy()).astype(mx.bfloat16) for k, v in dec.state_dict().items()}
    out = V.vit_decoder(mx.random.normal((1, 2, 3, 3, 6)), p, num_layers=1, heads=2, patch_size=4, patch_size_t=2,
                        out_channels=3, rope_dim=12)
    mx.eval(out)
    assert out.shape == (1, 4, 12, 12, 3)


@pytest.fixture(scope="module")
def saved_qwen(tmp_path_factory):
    from test_text_encoder_tiny import IDS, ROPE, TEXT, VISION
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(3)
    hf = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(text_config=dict(TEXT, rope_parameters=dict(ROPE, rope_theta=5e6)), vision_config=VISION, **IDS)
    ).eval().float()
    d = tmp_path_factory.mktemp("qwen") / "text_encoder"
    hf.save_pretrained(d)
    return d


def test_text_encoder_has_no_head_and_only_kept_layers(saved_qwen):  # review H3R2
    from mlx.utils import tree_flatten

    model, _ = load.load_text_encoder(saved_qwen, keep_layers=2)
    names = [k for k, _ in tree_flatten(model.parameters())]
    assert not any("lm_head" in k for k in names)
    assert len(model.language_model.model.layers) == 2
    assert not any(".layers.2." in k or ".layers.3." in k for k in names if "language_model" in k)


def test_staged_encoder_matches_resident(saved_qwen):
    ids = [151643, 9707, 11, 1879]
    _, resident = load.load_text_encoder(saved_qwen, keep_layers=3)
    staged = load.staged_text_encoder(saved_qwen, keep_layers=3)
    assert float(mx.abs(staged(ids, None) - resident(ids, None)).max()) == 0.0
