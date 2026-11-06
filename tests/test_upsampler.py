"""P-H3-Image's prompt upsampler: the request text, chat template and answer handling against diffusers'
`upsample_prompt` (with a stand-in model that records its inputs and returns a fixed answer), and a tiny random
Qwen3.5 in transformers against the same weights in mlx-vlm (prefill logits and greedy tokens, with two pictures)."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import FIXTURES, assert_close
from PIL import Image

from h3mlx import upsampler as U

PROC = FIXTURES / "P-H3-Image" / "prompt_upsampler_processor"
pytestmark = pytest.mark.skipif(not PROC.exists(), reason="release fixtures not present (scripts/fetch_test_fixtures.py)")

IDS = dict(image_token_id=248056, video_token_id=248057, vision_start_token_id=248053, vision_end_token_id=248054)
TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, full_attention_interval=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, attn_output_gate=True, linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_value_head_dim=64, rms_norm_eps=1e-6, vocab_size=248320,
            max_position_embeddings=4096, mtp_num_hidden_layers=0, tie_word_embeddings=False,
            rope_parameters=dict(mrope_interleaved=True, mrope_section=[2, 1, 1], partial_rotary_factor=0.25,
                                 rope_theta=1e7, rope_type="default"))
# mlx-vlm's gated-delta kernel splits a key head over 32 lanes (`n_per_t = Dk / 32`): key heads under 32 do not build.
VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16,
              spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64, deepstack_visual_indexes=[],
              hidden_act="gelu_pytorch_tanh", in_channels=3)


def pictures(seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((96, 64), (80, 120))]


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(PROC))


class _Recorder:
    """Stands in for the model inside diffusers' `upsample_prompt`: records the inputs, answers with fixed text."""

    device = "cpu"

    def __init__(self, answer_ids):
        self.answer_ids, self.inputs = answer_ids, None

    def generate(self, **kw):
        self.inputs = kw
        return torch.cat([kw["input_ids"], torch.tensor([self.answer_ids])], dim=1)


T2I_ANSWER = "<think>\nweighing the light\n</think>\n\nA documentary photograph of a fishmonger laughing at dawn."
EDIT_ANSWER = ("<think>\nok\n</think>\n\nEdit <Picture 1>. Put the straw hat from Picture 2 on the woman. Keep her face, "
               "pose, the market and the light exactly as they are.")


@pytest.mark.parametrize("workflow", ["text2image", "edit"])
def test_requests_and_answers_match_diffusers(processor, workflow, monkeypatch):
    from diffusers.modular_pipelines.p_h3_image import prompt_upsampler as D

    tok = processor.tokenizer
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    answer = T2I_ANSWER if workflow == "text2image" else EDIT_ANSWER
    ids = tok(answer, add_special_tokens=False)["input_ids"] + [end_id]
    images = pictures() if workflow == "edit" else None
    rec = _Recorder(ids)
    want = D.upsample_prompt(rec, processor, "  a fishmonger at dawn ", aspect_ratio="16:9",
                             images=[D.prepare_upsampler_image(im) for im in images] if images else None,
                             max_new_tokens=99)

    seen = {}

    def fake_generate(model, inputs, end, max_new):
        seen.update(inputs=inputs, end=end, max_new=max_new)
        return ids

    monkeypatch.setattr(U, "generate_ids", fake_generate)
    got = U.upsample(None, processor, "  a fishmonger at dawn ", images=images, aspect_ratio="16:9", max_new_tokens=99)
    assert got == want
    assert seen["end"] == end_id and seen["max_new"] == 99
    assert np.array_equal(seen["inputs"]["input_ids"], rec.inputs["input_ids"].numpy())
    if images:
        assert np.array_equal(seen["inputs"]["pixel_values"], rec.inputs["pixel_values"].numpy())
        assert np.array_equal(seen["inputs"]["image_grid_thw"], rec.inputs["image_grid_thw"].numpy())


def test_helpers_match_diffusers():
    from diffusers.modular_pipelines.p_h3_image import prompt_upsampler as D

    for w, h in ((2912, 2912), (3872, 2176), (2176, 3872), (3360, 2496), (1000, 1001), (5000, 1000)):
        assert U.aspect_ratio_label(w, h) == D.aspect_ratio_label(w, h)
    brief = ("subject_definitions:\n<Subject 1>: the woman\nsummary:\nx\nretention_analysis:\n"
             "<Picture 1> (woman): fully_preserved - face\nsummary2\ndetailed_description:\n[keyframe completion] y\n"
             "overall_soundscape:\nnone\nnon_diegetic_music:\nnone")
    for text in (EDIT_ANSWER.split("</think>")[1], "Edit <Picture 1>. Make it red.", brief, "a plain prompt"):
        assert U.is_edit_prompt(text) == D.is_edit_prompt(text)
    for im in pictures(3) + [Image.new("RGB", (4000, 3000))]:
        assert np.array_equal(np.asarray(U.prepare_image(im)), np.asarray(D.prepare_upsampler_image(im)))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, processor):
    from mlx_vlm.utils import load_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(13)
    hf = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=TEXT, vision_config=VISION, **IDS)).eval().float()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.normal_(0.0 if "linear_attn.norm" not in name else 1.0, 0.3)
    d = tmp_path_factory.mktemp("qwen35")
    hf.save_pretrained(str(d))
    return hf, load_model(d)


def _inputs(processor, images):
    text = U.chat_text(processor.tokenizer, "a fishmonger at dawn", len(images), "1:1")
    return processor(text=[text], images=[U.prepare_image(im) for im in images] or None, return_tensors="pt")


@pytest.mark.parametrize("n_images", [0, 2])
def test_tiny_prefill_logits_match_transformers(tiny, processor, n_images):
    hf, model = tiny
    inputs = _inputs(processor, pictures()[:n_images])
    with torch.no_grad():
        want = hf(**inputs).logits[0, -4:]
    kw = {k: mx.array(v.numpy()) for k, v in inputs.items() if k in ("image_grid_thw",)}
    out = model(mx.array(inputs["input_ids"].numpy()),
                mx.array(inputs["pixel_values"].numpy()) if "pixel_values" in inputs else None,
                mask=mx.array(inputs["attention_mask"].numpy()), **kw)
    got = (out.logits if hasattr(out, "logits") else out)[0, -4:]
    assert_close(got, want, atol=2e-3, rtol=2e-3, what=f"last 4 prefill logits, {n_images} pictures")


@pytest.mark.parametrize("n_images", [0, 2])
def test_tiny_greedy_tokens_match_transformers(tiny, processor, n_images):
    hf, model = tiny
    tok = processor.tokenizer
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    inputs = _inputs(processor, pictures()[:n_images])
    with torch.no_grad():
        want = hf.generate(**inputs, max_new_tokens=12, do_sample=False, eos_token_id=end_id,
                           pad_token_id=tok.pad_token_id)[0, inputs["input_ids"].shape[1]:].tolist()
    got = U.generate_ids(model, {k: v.numpy() for k, v in inputs.items()}, end_id, 12)
    assert got == want, f"greedy tokens differ: {got} vs {want}"
