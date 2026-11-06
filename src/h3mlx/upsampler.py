"""P-H3-Image's prompt upsampler (a Qwen3.5-9B fine-tune) in MLX via mlx-vlm, mirroring diffusers'
`modular_pipelines/p_h3_image/prompt_upsampler.py`: the same request text, chat template and processor inputs, one
greedy generation, the prompt taken from after `</think>`, and the same validity checks.

Greedy decoding in MLX and in torch can part ways once two logits are within rounding of each other, so the prompt
it writes may differ from the torch pipeline's in places; the checks below decide whether it is usable.
"""

import math
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .image import MAX_EDIT_IMAGES, size_at_area

ASPECT_RATIOS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
UPSAMPLER_IMAGE_PIXELS = 1_000_000
T2I_MAX_NEW_TOKENS = 1024
EDIT_MAX_NEW_TOKENS = 1536
EDIT_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
                 "non_diegetic_music")
_SECTION_RE = re.compile(r"(?m)^(" + "|".join(EDIT_SECTIONS) + r"):[ \t]*$")
_RETENTION_RE = re.compile(
    r"^<(Picture|Subject) (\d+)>(?: \(([^)]*)\))?:\s*(fully_preserved|partially_preserved|attribute_transfer|"
    r"weak_reference)\s*[-–—]\s*(.*)$"
)


def aspect_ratio_label(width: int, height: int) -> str:
    return min(ASPECT_RATIOS, key=lambda label: abs(math.log(width / height / ASPECT_RATIOS[label])))


def is_edit_prompt(prompt: str) -> bool:
    """A brief (the six sections in order, `[keyframe completion]` once, every retention line naming a picture or
    subject and how it is kept) or a one-paragraph instruct prompt starting `"Edit <Picture 1>."` that says what it
    keeps."""
    text = prompt.strip()
    names = _SECTION_RE.findall(text)
    if names:
        parts = _SECTION_RE.split(text)
        body = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        retention = [line.strip() for line in body.get("retention_analysis", "").splitlines() if line.strip()]
        return (tuple(names) == EDIT_SECTIONS and text.count("[keyframe completion]") == 1 and bool(retention)
                and all(_RETENTION_RE.match(line) for line in retention))
    return text.startswith("Edit <Picture 1>.") and "Keep" in text and "\n" not in text


def prepare_image(image: Image.Image) -> Image.Image:
    """An image as the upsampler sees it: RGB at about one megapixel, sides multiples of 32 (Lanczos)."""
    image = image.convert("RGB")
    size = size_at_area(*image.size, UPSAMPLER_IMAGE_PIXELS, 32)
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def chat_text(tokenizer, request: str, n_images: int = 0, aspect_ratio: str = "1:1") -> str:
    if n_images:
        content = []
        for i in range(n_images):
            content += [{"type": "text", "text": f"<Picture {i + 1}>: "}, {"type": "image"}]
        content.append({"type": "text", "text": f"Edit <Picture 1>. {request.strip()}"})
    else:
        content = f"Aspect ratio: {aspect_ratio}\nRequest: {request.strip()}"
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def load_upsampler(model_dir, processor_dir):
    """(mlx-vlm model, transformers processor). The weights are read lazily on first use."""
    from mlx_vlm.utils import load_model
    from transformers import AutoProcessor

    return load_model(Path(model_dir)), AutoProcessor.from_pretrained(str(processor_dir))


def generate_ids(model, inputs: dict, end_id: int, max_new_tokens: int) -> list[int]:
    """Greedy tokens after the prompt, up to and including `end_id` or `max_new_tokens`."""
    from mlx_vlm.generate.ar import generate_step

    kw = {k: mx.array(np.asarray(v)) for k, v in inputs.items() if k not in ("input_ids", "attention_mask",
                                                                               "pixel_values", "mm_token_type_ids")}
    pv = inputs.get("pixel_values")
    out = []
    for tok, _ in generate_step(mx.array(np.asarray(inputs["input_ids"])), model,
                                None if pv is None else mx.array(np.asarray(pv)),
                                mx.array(np.asarray(inputs["attention_mask"])), max_tokens=max_new_tokens,
                                temperature=0.0, **kw):
        out.append(int(tok))
        if out[-1] == end_id:
            break
    return out


def upsample(model, processor, request: str, images=None, aspect_ratio: str = "1:1", max_new_tokens: int | None = None,
             on_answer=None) -> str:
    """`upsample_prompt`: a text-to-image prompt for `aspect_ratio`, or an edit prompt written from `images` (the image
    to edit first; prepared here). Raises if the answer was cut off or is not of the expected form."""
    if not isinstance(request, str):
        raise ValueError(f"the upsampler takes one request at a time, got {type(request).__name__}")
    images = [prepare_image(im) for im in (images or [])]
    if len(images) > MAX_EDIT_IMAGES:
        raise ValueError(f"the upsampler takes at most {MAX_EDIT_IMAGES} images, got {len(images)}")
    if max_new_tokens is None:
        max_new_tokens = EDIT_MAX_NEW_TOKENS if images else T2I_MAX_NEW_TOKENS
    tok = processor.tokenizer
    text = chat_text(tok, request, len(images), aspect_ratio)
    inputs = processor(text=[text], images=images or None, return_tensors="np")
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    new_ids = generate_ids(model, dict(inputs), end_id, max_new_tokens)
    ended = end_id in new_ids
    answer = tok.decode(new_ids[: new_ids.index(end_id)] if ended else new_ids, skip_special_tokens=False)
    if on_answer:
        on_answer(answer, len(new_ids))
    prompt = answer.partition("</think>")[2].strip()
    valid = ended and answer.count("</think>") == 1 and len(prompt.split()) >= 5
    if images:
        valid = valid and is_edit_prompt(prompt)
    if not valid:
        raise RuntimeError("the prompt upsampler did not return a usable prompt "
                           f"({'cut off at max_new_tokens' if not ended else 'not of the expected form'}):\n{answer}")
    return prompt
