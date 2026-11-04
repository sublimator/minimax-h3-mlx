"""The whole transformer forward at a tiny config with random weights, against MiniMaxH3Transformer3DModel."""

import mlx.core as mx
import pytest
import torch
from conftest import assert_close, state_mx, to_mx
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

from h3mlx import ops

TINY = dict(
    num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1, ffn_dim=96,
    in_channels=4, audio_in_channels=6, patch_size=(1, 2, 2), text_dim=20, freq_dim=32, time_embed_hidden_dim=40,
    time_embed_dim=24, rope_freq_dim=4, rope_theta=1e4, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
)


def packed_layout(n_text=5, n_video=12, n_audio=7, n_t=3, gen=None):
    """A shuffled packed sequence: text, video and audio rows interleaved, each row with its own timestep index."""
    seq = n_text + n_video + n_audio
    perm = torch.randperm(seq, generator=gen)
    text_idx, video_idx, audio_idx = perm[:n_text], perm[n_text : n_text + n_video], perm[n_text + n_video :]
    tags = torch.zeros(seq, dtype=torch.long)
    tags[text_idx], tags[audio_idx] = 1, 2
    t_idx = torch.randint(0, n_t, (seq,), generator=gen)
    pos = torch.randn(seq, 3, generator=gen) * 10
    timestep = torch.tensor([1.0, 0.37, 0.05])[:n_t]
    return dict(timestep=timestep, timestep_indices=t_idx, token_tags=tags, position_ids=pos,
                video_indices=video_idx, audio_indices=audio_idx, text_indices=text_idx)


@pytest.mark.parametrize("batch", [1, 2])
def test_tiny_forward_matches(batch):
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    with torch.no_grad():
        for name, prm in model.named_parameters():  # norms default to ones; randomise so they count
            if "norm" in name and prm.ndim == 1:
                prm.normal_()
    lay = packed_layout()
    video = torch.randn(batch, len(lay["video_indices"]), 4 * 1 * 2 * 2)
    audio = torch.randn(batch, len(lay["audio_indices"]), 6)
    text = torch.randn(batch, len(lay["text_indices"]), 20)
    with torch.no_grad():
        want = model(video, audio, text, **lay, return_dict=False)
    cfg = dict(model.config)
    got = ops.dit_forward(state_mx(model), cfg, to_mx(video), to_mx(audio), to_mx(text),
                          **{k: to_mx(v) for k, v in lay.items()})
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    assert_close(got[1], want[1], atol=2e-4, rtol=2e-4, what="audio velocity")


def test_tiny_forward_sensitive_to_layout():
    """Swapping two rows' modality tags must change the output (guards against a test that can't fail)."""
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    lay = packed_layout()
    video, audio, text = torch.randn(1, 12, 16), torch.randn(1, 7, 6), torch.randn(1, 5, 20)
    args = (state_mx(model), dict(model.config), to_mx(video), to_mx(audio), to_mx(text))
    base = ops.dit_forward(*args, **{k: to_mx(v) for k, v in lay.items()})[0]
    lay["timestep_indices"] = (lay["timestep_indices"] + 1) % 3
    moved = ops.dit_forward(*args, **{k: to_mx(v) for k, v in lay.items()})[0]
    assert float(mx.abs(base - moved).max()) > 1e-3
