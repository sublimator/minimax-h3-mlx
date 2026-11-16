"""Adapters in the original checkpoint's key names, converted, against diffusers' converter.

Random adapters are written in each producer's layout (ai-toolkit, bare, kohya, musubi-tuner, DiffSynth-Studio). The
oracle is `_convert_non_diffusers_minimax_h3_lora_to_diffusers` for the dicts, and for the forward
`MiniMaxH3LoraLoaderMixin.lora_state_dict` followed by `load_lora_adapter` on the tiny transformer.
"""

import copy

import numpy as np
import pytest
import torch
from conftest import assert_close, state_mx, to_mx, to_np
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers
from diffusers.loaders.lora_pipeline import MiniMaxH3LoraLoaderMixin
from safetensors.torch import save_file
from test_dit_mixed_precision import rel_err
from test_dit_tiny import TINY
from test_hyperflow import randomised
from test_lora import FILE, GUARD, inputs, mlx_forward, peft_layers, to_mx_half, torch_forward

from h3mlx import lora

LAYOUTS = ("ai-toolkit", "bare", "kohya", "musubi", "diffsynth")
WIDE = dict(TINY, attention_head_dim=128)  # the head width diffusers' converter assumes for a DiffSynth file
RANK = 4
ALPHAS = (4.0, 1.0, 3.0, 16.0, 0.0625)  # scalings 1, 1/4 (one turn of the fold's loop), 3/4, 4, 1/64 (three turns)


def original_shapes(cfg, wired_only=True):
    """{module: (out, in)} in the original checkpoint's names, for the Linears our forward applies a LoRA to, or for
    every module the converter knows."""
    h, inner, ffn = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["ffn_dim"]
    per_block = {"attn.qkv_proj": (3 * inner, h), "attn.out_proj": (h, inner), "mlp.fc1": (2 * ffn, h), "mlp.fc2": (h, ffn)}
    out = {}
    for prefix, n in (("blocks", cfg["num_layers"]), ("token_refiner.blocks", cfg["num_refiner_layers"])):
        for i in range(n):
            out.update({f"{prefix}.{i}.{k}": v for k, v in per_block.items()})
            if prefix == "blocks" and not wired_only:
                out[f"blocks.{i}.adaln_proj.linear"] = (18 * h, cfg["time_embed_dim"])
    out["time_embedder.proj_in"] = (cfg["time_embed_hidden_dim"], cfg["freq_dim"])
    out["time_embedder.proj_out"] = (cfg["time_embed_dim"], cfg["time_embed_hidden_dim"])
    if not wired_only:
        out.update({"video_patch_proj": (h, 16), "audio_patch_proj": (h, 6), "condition_proj": (h, cfg["text_dim"]),
                    "final_layer.adaln_proj.linear": (2 * h, cfg["time_embed_dim"]), "final_layer.video_out": (16, h),
                    "final_layer.audio_out": (6, h)})
    return out


def original_file(cfg, layout, wired_only=True, dtype=torch.float32, seed=6, std=0.3, rank=RANK):
    """A random adapter in one producer's key layout (torch tensors)."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for n, (module, (out_f, in_f)) in enumerate(original_shapes(cfg, wired_only).items()):
        down = (torch.randn(rank, in_f, generator=g) * std).to(dtype)
        up = (torch.randn(out_f, rank, generator=g) * std).to(dtype)
        alpha = torch.tensor(ALPHAS[n % len(ALPHAS)], dtype=dtype)
        if layout == "ai-toolkit":
            sd[f"diffusion_model.{module}.lora_A.weight"], sd[f"diffusion_model.{module}.lora_B.weight"] = down, up
        elif layout == "bare":
            sd[f"{module}.lora_A.weight"], sd[f"{module}.lora_B.weight"] = down, up
        elif layout == "diffsynth":
            sd[f"{module}.lora_A.default.weight"], sd[f"{module}.lora_B.default.weight"] = down, up
        else:
            name = module if layout == "kohya" else "lora_unet_" + module.replace(".", "_")
            sd[f"{name}.lora_down.weight"], sd[f"{name}.lora_up.weight"], sd[f"{name}.alpha"] = down, up, alpha
    return sd


def as_mx(sd):
    return {k: to_mx_half(v) for k, v in sd.items()}


def assert_same_dicts(got, want):
    assert list(got) == list(want), "the same keys in the same order"
    for k, w in want.items():
        assert str(got[k].dtype).split(".")[-1] == str(w.dtype).split(".")[-1], k
        assert got[k].shape == tuple(w.shape) and np.array_equal(to_np(got[k]), to_np(w)), k


# --- the dicts ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank", [4, 6])
def test_converts_to_diffusers_dicts(layout, dtype, rank):
    """Every module the converter knows (the AdaLN, patch, condition and final projections too), each layout, three
    dtypes: the same keys in the same order and the same tensors, bit for bit. At rank 6 the folded factors (alpha / 6
    and its doublings) are not half-precision values, so a fold that rounds the factor to the tensor's dtype first, as
    MLX's own `bf16 * float` does, fails here; at rank 4 every factor is a power of two or 3/4 and it would pass."""
    sd = original_file(WIDE, layout, wired_only=False, dtype=dtype, rank=rank)
    assert lora.is_original_layout(sd)
    want = _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(sd))
    before = dict(as_mx(sd))
    got = lora.convert_original(before)
    assert_same_dicts(got, want)
    assert len(got) == 2 * (len(original_shapes(WIDE, wired_only=False)) + 2 * (WIDE["num_layers"] + WIDE["num_refiner_layers"]))
    assert set(before) == set(sd), "the input dict is not modified"


def test_layout_detection():
    for layout in LAYOUTS:
        assert lora.is_original_layout(original_file(WIDE, layout))
    names = [f"{p}{m}.lora_A.weight" for m in lora.block_modules(TINY) + ["time_embedder.linear_1"] for p in ("", "transformer.")]
    assert not lora.is_original_layout(names), "diffusers names, the refiner's included, are left alone"


def test_what_the_conversion_does():
    """The parts of the mapping, checked without diffusers: one lora_A shared by q, k and v; lora_B cut in thirds;
    fc1's halves swapped; a kohya alpha split between the two matrices so that their product is scaled by alpha/rank."""
    sd = original_file(TINY, "kohya")
    got = lora.convert_original(as_mx(sd))
    t, inner, ffn = "transformer.transformer_blocks.1.", 64, TINY["ffn_dim"]
    up = to_np(sd["blocks.1.attn.qkv_proj.lora_up.weight"])
    down = to_np(sd["blocks.1.attn.qkv_proj.lora_down.weight"])
    alpha = float(sd["blocks.1.attn.qkv_proj.alpha"])
    assert alpha != RANK
    for i, proj in enumerate(("to_q", "to_k", "to_v")):
        a, b = to_np(got[f"{t}attn.{proj}.lora_A.weight"]), to_np(got[f"{t}attn.{proj}.lora_B.weight"])
        assert_close(b @ a, (alpha / RANK) * (up[i * inner:(i + 1) * inner] @ down), atol=1e-6, rtol=1e-5, what=proj)
    fc1 = to_np(sd["blocks.1.mlp.fc1.lora_up.weight"])
    scale = float(sd["blocks.1.mlp.fc1.alpha"]) / RANK
    b, a = to_np(got[t + "ff.net.0.proj.lora_B.weight"]), to_np(got[t + "ff.net.0.proj.lora_A.weight"])
    swapped = np.concatenate([fc1[ffn:], fc1[:ffn]])
    assert_close(b @ a, scale * (swapped @ to_np(sd["blocks.1.mlp.fc1.lora_down.weight"])), atol=1e-6, rtol=1e-5, what="fc1")
    assert rel_err(b @ a, scale * (fc1 @ to_np(sd["blocks.1.mlp.fc1.lora_down.weight"]))) > 0.5, "unswapped is far off"


def deinterleaved(up, heads, head_dim):
    """[head0: q k v, head1: q k v, ...] -> [q of every head; k of every head; v of every head], by index."""
    return up.reshape(heads, 3, head_dim, up.shape[1]).transpose(1, 0, 2, 3).reshape(3 * heads * head_dim, up.shape[1])


@pytest.mark.parametrize("cfg", [WIDE, TINY], ids=["head 128", "head 32"])
def test_diffsynth_rows_by_head_width(cfg):
    """A DiffSynth file's fused QKV rows are per-head. At diffusers' fixed head width (128) the index formula above is
    their converter's result; at the tiny model's width (32), where theirs can't run, ours follows the same formula
    when given the width."""
    heads, head_dim = cfg["num_attention_heads"], cfg["attention_head_dim"]
    sd = original_file(cfg, "diffsynth")
    up = sd["blocks.0.attn.qkv_proj.lora_B.default.weight"]
    want = np.split(deinterleaved(up.numpy(), heads, head_dim), 3)
    if head_dim == 128:
        theirs = _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(sd))
        for proj, w in zip(("to_q", "to_k", "to_v"), want):
            assert np.array_equal(theirs[f"transformer.transformer_blocks.0.attn.{proj}.lora_B.weight"].numpy(), w)
    else:
        with pytest.raises(ValueError, match="not a multiple of 3 \\* 128"):
            _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(sd))
        with pytest.raises(ValueError, match="not a multiple of 3 \\* 128"):
            lora.convert_original(as_mx(sd))  # the default is theirs
    got = lora.convert_original(as_mx(sd), head_dim=head_dim)
    for proj, w in zip(("to_q", "to_k", "to_v"), want):
        assert np.array_equal(to_np(got[f"transformer.transformer_blocks.0.attn.{proj}.lora_B.weight"]), w)
    assert not np.array_equal(want[0], up.numpy()[: heads * head_dim]), "the guard: the rows really are reordered"


# --- the forward -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_converted_adapter_gives_the_same_forward(layout):
    cfg = WIDE if layout == "diffsynth" else TINY
    model = randomised(cfg, 2)
    sd = original_file(cfg, layout)
    state_dict, metadata = MiniMaxH3LoraLoaderMixin.lora_state_dict(dict(sd), return_lora_metadata=True)
    assert metadata is None
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(state_dict, metadata=metadata)
    (name,) = loaded.peft_config
    layers = peft_layers(loaded)
    assert {layer.scaling[name] for layer in layers.values()} == {1.0}

    adapter = lora.read(as_mx(sd))
    assert adapter.modules == sorted(layers)
    # a folded alpha is spent: the converted adapter's scaling is 1 and nothing applies alpha / rank a second time
    assert all(adapter.scaling(m) == 1.0 for m in adapter.modules)
    params = lora.attach(state_mx(model), dict(cfg), adapter)
    assert not any(k.endswith(lora.SCALING) for k in params)
    lay, args = inputs()
    want = torch_forward(loaded, lay, args)
    got = mlx_forward(params, dict(cfg), lay, args)
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    assert_close(got[1], want[1], atol=2e-4, rtol=2e-4, what="audio velocity")
    assert rel_err(want[0], torch_forward(model, lay, args)[0]) > GUARD

    # guards, each a wrong reading of the same file that must move the output
    def forward_of(tensors):
        return mlx_forward(lora.attach(state_mx(model), dict(cfg), lora.read(tensors)), dict(cfg), lay, args)[0]

    if layout in ("kohya", "musubi"):
        no_alpha = {k: v for k, v in as_mx(sd).items() if not k.endswith(".alpha")}
        assert rel_err(forward_of(no_alpha), want[0]) > GUARD, "the alphas count"
        twice = lora.Adapter(adapter.tensors, {m: ALPHAS[1] for m in adapter.modules})
        again = mlx_forward(lora.attach(state_mx(model), dict(cfg), twice), dict(cfg), lay, args)[0]
        assert rel_err(again, want[0]) > GUARD, "alpha / rank applied on top of the fold is visible"
    if layout == "diffsynth":
        plain = {k.replace(".default.", "."): v for k, v in as_mx(sd).items()}  # read as a bare file: no de-interleave
        assert rel_err(forward_of(plain), want[0]) > GUARD, "the per-head row order counts"


@pytest.mark.parametrize("layout,entry,scaling", [("ai-toolkit", "8", 2.0), ("kohya", "8", 1.0)])
def test_header_alpha_with_a_converted_file(tmp_path, layout, entry, scaling):
    """An `alpha` entry in the safetensors header is one network alpha for an alpha-less file (scaling alpha / rank),
    and is ignored when the file has `.alpha` tensors, which the fold has already applied."""
    model = randomised(TINY, 2)
    sd = original_file(TINY, layout)
    save_file(sd, str(tmp_path / FILE), metadata={"alpha": entry})
    state_dict, metadata = MiniMaxH3LoraLoaderMixin.lora_state_dict(str(tmp_path), weight_name=FILE, return_lora_metadata=True)
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(state_dict, metadata=metadata)
    (name,) = loaded.peft_config
    assert {layer.scaling[name] for layer in peft_layers(loaded).values()} == {scaling}
    adapter = lora.load(tmp_path / FILE)
    assert all(adapter.scaling(m) == scaling for m in adapter.modules)
    lay, args = inputs()
    got = mlx_forward(lora.attach(state_mx(model), dict(TINY), adapter), dict(TINY), lay, args)
    assert_close(got[0], torch_forward(loaded, lay, args)[0], atol=2e-4, rtol=2e-4, what="video velocity")


# --- what is refused ---------------------------------------------------------------------------------------------------


def test_refusals_shared_with_diffusers():
    """Inputs both converters refuse, with the same error."""
    base = original_file(TINY, "bare")
    cases = {
        "should be empty": {**base, "blocks.0.attn.rel_pos.lora_A.weight": torch.zeros(4, 4)},
        "not divisible by 3": {"blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(4, 48), "blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(64, 4)},
        "not even": {"blocks.0.mlp.fc1.lora_A.weight": torch.zeros(4, 48), "blocks.0.mlp.fc1.lora_B.weight": torch.zeros(193, 4)},
        "does not name a MiniMax-H3 module": {"lora_unet_blocks_0_attn_proj.lora_down.weight": torch.zeros(4, 48),
                                               "lora_unet_blocks_0_attn_proj.lora_up.weight": torch.zeros(48, 4)},
    }
    for message, sd in cases.items():
        with pytest.raises(ValueError, match=message):
            _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(sd))
        with pytest.raises(ValueError, match=message):
            lora.convert_original(as_mx(sd))


def fold_turns(alpha, rank, cap=5000):
    """The loop of diffusers' alpha fold (`pull` in lora_conversion_utils.py), transcribed, with a cap on its turns."""
    scale_down, scale_up, turns = alpha / rank, 1.0, 0
    while scale_down * 2 < scale_up and turns < cap:
        scale_down *= 2
        scale_up /= 2
        turns += 1
    return turns


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_alpha_is_refused(alpha):
    """Ours alone. diffusers' fold halves `scale_up` while `scale_down * 2 < scale_up`. On alpha 0 that runs until
    `scale_up` underflows, 1075 turns, and both matrices come out zero (their converter is run on it below). On a
    negative alpha it never returns, so their converter is not run on that input: the transcribed loop, capped, stands
    in for it."""
    assert [fold_turns(a, RANK) for a in ALPHAS] == [0, 1, 0, 0, 3]
    assert (fold_turns(0.0, RANK), fold_turns(-1.0, RANK)) == (1075, 5000)
    sd = {"blocks.0.attn.out_proj.lora_down.weight": torch.randn(4, 64), "blocks.0.attn.out_proj.lora_up.weight": torch.randn(48, 4),
          "blocks.0.attn.out_proj.alpha": torch.tensor(alpha)}
    with pytest.raises(ValueError, match="positive and finite"):
        lora.convert_original(as_mx(sd))
    with pytest.raises(ValueError, match="positive and finite"):
        lora.read(as_mx(sd))
    if alpha == 0.0:
        theirs = _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(sd))
        assert all(not v.any() for v in theirs.values()), "diffusers: an adapter of zeros"


def test_unwired_converted_modules_are_refused_at_attach():
    """The converter maps the AdaLN, patch, condition and final projections, as diffusers' does, but our forward runs
    no LoRA there: attaching refuses them rather than dropping them."""
    model = randomised(TINY, 2)
    adapter = lora.read(as_mx(original_file(TINY, "ai-toolkit", wired_only=False)))
    assert "transformer_blocks.0.adaln_proj.linear" in adapter.modules and "proj_in" in adapter.modules
    with pytest.raises(ValueError, match="not run through ops.plinear"):
        lora.attach(state_mx(model), dict(TINY), adapter)
    wired = lora.wired_shapes(TINY)
    kept = lora.Adapter({k: v for k, v in adapter.tensors.items() if k.rsplit(".lora_", 1)[0] in wired},
                        {m: a for m, a in adapter.alpha.items() if m in wired})
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in lora.attach(state_mx(model), dict(TINY), kept)
