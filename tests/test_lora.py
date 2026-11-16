"""LoRA adapters on our MLX transformer, against PEFT and diffusers.

The oracle is the tiny diffusers transformer with PEFT's own layers (`add_adapter`, `load_lora_adapter`, `fuse_lora`)
and diffusers' own files (`save_lora_adapter`, `MiniMaxH3LoraLoaderMixin.save_lora_weights` / `lora_state_dict`). Each
side loads the other's file; nothing here compares our loader with our own writer except the round trip.
"""

import copy
import json
import math

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, state_mx, to_mx, to_np
from diffusers.loaders.lora_pipeline import MiniMaxH3LoraLoaderMixin
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from peft.utils import get_peft_model_state_dict
from peft.utils.other import get_pattern_key
from safetensors import safe_open
from safetensors.torch import save_file
from test_dit_mixed_precision import rel_err, to_mixed
from test_dit_tiny import TINY, packed_layout
from test_hyperflow import make_file, randomised

from h3mlx import hyperflow, load, lora, ops

FILE = "pytorch_lora_weights.safetensors"  # diffusers' LORA_WEIGHT_NAME_SAFE
MODULES = lora.block_modules(TINY) + ["time_embedder.linear_1", "time_embedder.linear_2"]


# --- helpers -----------------------------------------------------------------------------------------------------------


GUARD = 0.01  # a guard's change must be this large, relative: 50 times the forward's tolerance


def inputs(seed=3):
    gen = torch.Generator().manual_seed(seed)
    lay = packed_layout(gen=gen)
    video, audio, text = torch.randn(1, 12, 16, generator=gen), torch.randn(1, 7, 6, generator=gen), torch.randn(1, 5, 20, generator=gen)
    return lay, (video, audio, text)


def torch_forward(model, lay, args):
    with torch.no_grad():
        return model(*args, **lay, return_dict=False)


def mlx_forward(params, cfg, lay, args):
    return ops.dit_forward(params, cfg, *(to_mx(a) for a in args), **{k: to_mx(v) for k, v in lay.items()})


def randomise_b(model, seed=1, std=0.3):
    """PEFT starts lora_B at zero, where an adapter does nothing: fill it so the adapter counts."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if "lora_B" in name:
                prm.copy_((torch.randn(prm.shape, generator=g) * std).to(prm.dtype))
    return model


def with_peft_adapter(model, seed=1, **config):
    m = copy.deepcopy(model)
    m.add_adapter(LoraConfig(target_modules=list(MODULES), init_lora_weights=True, **config))
    return randomise_b(m, seed)


def peft_layers(model):
    return {n: mod for n, mod in model.named_modules() if isinstance(mod, PeftLinear)}


def random_adapter(cfg, ranks, alphas, seed=2, std=0.3, dtype=torch.float32):
    """An adapter with a rank and an alpha per module, both matrices random."""
    g = torch.Generator().manual_seed(seed)
    shapes = lora.wired_shapes(cfg)
    tensors = {}
    for m, r in ranks.items():
        out_f, in_f = shapes[m]
        tensors[m + ".lora_A.weight"] = to_mx_half((torch.randn(r, in_f, generator=g) * std).to(dtype))
        tensors[m + ".lora_B.weight"] = to_mx_half((torch.randn(out_f, r, generator=g) * std).to(dtype))
    return lora.Adapter(tensors, dict(alphas))


def write_diffusers_file(model, how, folder, adapter_name="default"):
    """The two files diffusers writes: the model's own (`save_lora_adapter`, no key prefix) and the pipeline's
    (`save_lora_weights`, `transformer.` before every key and every metadata entry)."""
    if how == "save_lora_adapter":
        model.save_lora_adapter(str(folder), adapter_name=adapter_name)
    else:
        MiniMaxH3LoraLoaderMixin.save_lora_weights(
            str(folder), transformer_lora_layers=get_peft_model_state_dict(model, adapter_name=adapter_name),
            transformer_lora_adapter_metadata=model.peft_config[adapter_name].to_dict())
    return folder / FILE


def header_of(path):
    with safe_open(str(path), "pt") as f:
        return f.metadata(), list(f.keys())


# --- the scaling, one Linear -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("r,alpha", [(4, 4), (4, 10), (3, 1), (8, 5.6)])
def test_plinear_scaling_matches_peft_linear(r, alpha):
    base = torch.nn.Linear(12, 20, bias=True)
    layer = randomise_b(PeftLinear(copy.deepcopy(base), "default", LoraConfig(r=r, lora_alpha=alpha), r=r, lora_alpha=alpha))
    assert layer.scaling["default"] == alpha / r
    x = torch.randn(5, 12)
    with torch.no_grad():
        want = layer(x)
    p = {"l.weight": to_mx(base.weight), "l.bias": to_mx(base.bias),
         "l.lora_A.weight": to_mx(layer.lora_A["default"].weight), "l.lora_B.weight": to_mx(layer.lora_B["default"].weight)}
    unscaled = ops.plinear(to_mx(x), p, "l", bias=True)
    if alpha != r:
        p["l" + lora.SCALING] = mx.array(alpha / r, dtype=mx.float32)
    got = ops.plinear(to_mx(x), p, "l", bias=True)
    assert_close(got, want, atol=2e-6, rtol=2e-6, what="base + scaling * B(A(x))")
    if alpha != r:  # guard: leaving the scaling out must be visible
        assert rel_err(unscaled, want) > GUARD


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scaling", [0.7, 1 / 3, 2.5])
def test_half_precision_scaling_rounds_as_torch(dtype, scaling):
    """Selection matrices for A and B and a zero base weight make every other step exact, so the output is the scaled
    delta alone, bit for bit: torch multiplies a half tensor by a Python float in float32 and rounds once. MLX's own
    `bf16 * 0.7` multiplies by 0.69921875 (the scalar rounded first) and fails this."""
    r, n = 4, 6
    base = torch.nn.Linear(n, n, bias=False).to(dtype)
    layer = PeftLinear(base, "default", LoraConfig(r=r, lora_alpha=scaling * r), r=r, lora_alpha=scaling * r)
    with torch.no_grad():
        base.weight.zero_()
        layer.lora_A["default"].weight.copy_(torch.eye(n)[:r])
        layer.lora_B["default"].weight.copy_(torch.eye(n)[:, :r])
    assert layer.lora_A["default"].weight.dtype == dtype
    x = torch.randn(4000, n).to(dtype)
    with torch.no_grad():
        want = layer(x)
    p = {"l.weight": to_mx_half(base.weight), "l.lora_A.weight": to_mx_half(layer.lora_A["default"].weight),
         "l.lora_B.weight": to_mx_half(layer.lora_B["default"].weight),
         "l" + lora.SCALING: mx.array(layer.scaling["default"], dtype=mx.float32)}
    got = ops.plinear(to_mx_half(x), p, "l")
    assert got.dtype == p["l.weight"].dtype
    assert np.array_equal(to_np(got), to_np(want))
    weak = to_mx_half(x)[:, :r] * scaling  # what MLX's weak scalar does
    if scaling != 2.5:
        assert not np.array_equal(to_np(weak), to_np(want)[:, :r]), "the guard: the naive product must differ"


def to_mx_half(t):
    return to_mx(t) if t.dtype != torch.float16 else mx.array(t.detach().float().numpy()).astype(mx.float16)


# --- files, both ways --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["save_lora_adapter", "save_lora_weights"])
@pytest.mark.parametrize("r,alpha", [(4, 10), (6, 2)])
def test_diffusers_file_loads_in_mlx(tmp_path, how, r, alpha):
    model = randomised(TINY, 2)
    peft_model = with_peft_adapter(model, r=r, lora_alpha=alpha)
    path = write_diffusers_file(peft_model, how, tmp_path)
    _, keys = header_of(path)
    assert all(k.startswith("transformer.") for k in keys) == (how == "save_lora_weights")

    adapter = lora.load(path)
    assert adapter.modules == sorted(MODULES)
    assert all(adapter.rank(m) == r and adapter.scaling(m) == alpha / r for m in adapter.modules)
    lay, args = inputs()
    want = torch_forward(peft_model, lay, args)
    cfg = dict(model.config)
    got = mlx_forward(lora.attach(state_mx(model), cfg, adapter), cfg, lay, args)
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    assert_close(got[1], want[1], atol=2e-4, rtol=2e-4, what="audio velocity")
    # guards: the adapter moves the output, and so does its scaling
    assert rel_err(want[0], torch_forward(model, lay, args)[0]) > GUARD
    unscaled = lora.Adapter(adapter.tensors, {m: r for m in adapter.modules})
    assert rel_err(mlx_forward(lora.attach(state_mx(model), cfg, unscaled), cfg, lay, args)[0], want[0]) > GUARD


@pytest.mark.parametrize("prefix", ["transformer", None])
@pytest.mark.parametrize("rank,alpha", [(4, 10), (6, 2)])
def test_mlx_file_loads_in_diffusers(tmp_path, prefix, rank, alpha):
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    adapter = random_adapter(cfg, {m: rank for m in MODULES}, {m: alpha for m in MODULES})
    lora.save(tmp_path / FILE, adapter, prefix=prefix)

    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(str(tmp_path), weight_name=FILE, prefix=prefix)
    layers = peft_layers(loaded)
    assert sorted(layers) == adapter.modules, "PEFT wrapped exactly the modules in the file"
    (name,) = loaded.peft_config
    assert (loaded.peft_config[name].r, loaded.peft_config[name].lora_alpha) == (rank, alpha)
    for m, layer in layers.items():
        assert (layer.r[name], layer.scaling[name]) == (rank, alpha / rank), m
        assert torch.equal(layer.lora_B[name].weight, torch.from_numpy(to_np(adapter.tensors[m + ".lora_B.weight"])))

    lay, args = inputs()
    want = torch_forward(loaded, lay, args)
    got = mlx_forward(lora.attach(state_mx(model), cfg, adapter), cfg, lay, args)
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    assert_close(got[1], want[1], atol=2e-4, rtol=2e-4, what="audio velocity")
    assert rel_err(want[0], torch_forward(model, lay, args)[0]) > GUARD


def test_round_trip_and_header(tmp_path):
    cfg = dict(TINY)
    adapter = random_adapter(cfg, {m: 4 for m in MODULES}, {m: 10 for m in MODULES})
    lora.save(tmp_path / "a.safetensors", adapter)
    back = lora.load(tmp_path / "a.safetensors")
    assert back.alpha == adapter.alpha and set(back.tensors) == set(adapter.tensors)
    for k, v in adapter.tensors.items():
        assert back.tensors[k].dtype == mx.float32 and np.array_equal(to_np(back.tensors[k]), to_np(v)), k
    header, keys = header_of(tmp_path / "a.safetensors")
    assert header["format"] == "pt" and set(header) == {"format", lora.METADATA_KEY}
    assert set(keys) == {"transformer." + k for k in adapter.tensors}
    # the JSON is written as diffusers writes it (`json.dumps(..., indent=2, sort_keys=True)`), every key prefixed
    want = json.dumps({"transformer." + k: v for k, v in lora.config(adapter).items()}, indent=2, sort_keys=True)
    assert header[lora.METADATA_KEY] == want


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("prefix", ["transformer", None])
def test_half_precision_adapter_keeps_its_dtype_in_the_file(tmp_path, dtype, prefix):
    """`save` writes the tensors in the dtype they have: a bf16 or float16 adapter is that on disk (read back here with
    safetensors and with diffusers, neither through our loader), and `load` returns the same dtype and bits. A writer
    that upcast to float32 would pass every float32 test."""
    adapter = random_adapter(dict(TINY), {m: 4 for m in MODULES}, {m: 10 for m in MODULES}, dtype=dtype)
    path = tmp_path / FILE
    lora.save(path, adapter, prefix=prefix)
    lead = "" if prefix is None else prefix + "."
    with safe_open(str(path), "pt") as f:
        stored = {k: f.get_tensor(k) for k in f.keys()}
    through_diffusers = MiniMaxH3LoraLoaderMixin.lora_state_dict(str(tmp_path), weight_name=FILE)
    assert set(stored) == set(through_diffusers) == {lead + k for k in adapter.tensors}
    back = lora.load(path)
    for k, v in adapter.tensors.items():
        assert stored[lead + k].dtype == through_diffusers[lead + k].dtype == dtype, k
        assert np.array_equal(to_np(stored[lead + k]), to_np(v)), k
        assert back.tensors[k].dtype == v.dtype and v.dtype != mx.float32, k
        assert np.array_equal(to_np(back.tensors[k]), to_np(v)), k


def test_config_is_a_subset_of_pefts():
    """What we write under `lora_adapter_metadata` is what `LoraConfig.to_dict()` holds for the same adapter, entry by
    entry, minus the entries we leave to PEFT's defaults; and PEFT builds the same config back from it."""
    adapter = random_adapter(dict(TINY), {m: 4 for m in MODULES}, {m: 10 for m in MODULES})
    ours = lora.config(adapter)
    theirs = LoraConfig(r=4, lora_alpha=10, target_modules=list(MODULES)).to_dict()
    assert set(ours) <= set(theirs)
    for k, v in ours.items():
        t = sorted(theirs[k]) if isinstance(theirs[k], (set, list)) else theirs[k]
        assert v == t, k
    rebuilt = LoraConfig(**ours).to_dict()
    assert {k: (sorted(v) if isinstance(v, (set, list)) else v) for k, v in rebuilt.items()} == \
           {k: (sorted(v) if isinstance(v, (set, list)) else v) for k, v in theirs.items()}


# --- ranks and alphas that differ per module ---------------------------------------------------------------------------


@pytest.mark.parametrize("patterns,key", [
    (["attn.to_q", "to_q"], "transformer_blocks.0.attn.to_q"),  # the first match wins
    (["to_q", "attn.to_q"], "transformer_blocks.0.attn.to_q"),
    (["^transformer_blocks.0.attn.to_q"], "transformer_blocks.0.attn.to_q"),  # anchored: the full name only
    (["^transformer_blocks.0.attn.to_q"], "token_refiner.refiner_blocks.0.attn.to_q"),
    (["blocks.0.attn.to_q"], "transformer_blocks.0.attn.to_q"),  # not a whole dotted part: no match
    (["refiner_blocks.0.attn.to_q"], "token_refiner.refiner_blocks.0.attn.to_q"),
    (["ff.net.2"], "transformer_blocks.1.ff.net.2"),
    ([r"transformer_blocks\.\d+\.ff\.net\.2"], "transformer_blocks.1.ff.net.2"),
    ([], "time_embedder.linear_1"),
])
def test_pattern_key_is_pefts(patterns, key):
    assert lora.pattern_key(patterns, key) == get_pattern_key(patterns, key)


PATTERNS = dict(r=4, lora_alpha=8,
                rank_pattern={"^transformer_blocks.0.attn.to_q": 2, "ff.net.2": 6},
                alpha_pattern={"attn.to_k": 3, "^transformer_blocks.1.attn.to_v": 5, "ff.net.2": 9})


@pytest.mark.parametrize("how", ["save_lora_adapter", "save_lora_weights"])
def test_rank_and_alpha_patterns_from_diffusers(tmp_path, how):
    model = randomised(TINY, 2)
    peft_model = with_peft_adapter(model, **PATTERNS)
    layers = peft_layers(peft_model)
    assert len({(layer.r["default"], layer.lora_alpha["default"]) for layer in layers.values()}) == 5, "a real mix"
    adapter = lora.load(write_diffusers_file(peft_model, how, tmp_path))
    for m, layer in layers.items():
        assert adapter.rank(m) == layer.r["default"] and adapter.alpha[m] == layer.lora_alpha["default"], m
        assert adapter.scaling(m) == layer.scaling["default"], m
    lay, args = inputs()
    want = torch_forward(peft_model, lay, args)
    cfg = dict(model.config)
    got = mlx_forward(lora.attach(state_mx(model), cfg, adapter), cfg, lay, args)
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    flat = lora.Adapter(adapter.tensors, {m: 8 for m in adapter.modules})  # guard: the alpha pattern ignored
    assert rel_err(mlx_forward(lora.attach(state_mx(model), cfg, flat), cfg, lay, args)[0], want[0]) > GUARD


def test_mixed_ranks_and_alphas_to_diffusers(tmp_path):
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    ranks = {m: (2 if "to_q" in m else 6 if "ff.net.2" in m else 4) for m in MODULES}
    alphas = {m: (3 if "to_k" in m else 2.5 if m == "transformer_blocks.1.attn.to_v" else 8) for m in MODULES}
    adapter = random_adapter(cfg, ranks, alphas)
    made = lora.config(adapter)
    assert (made["r"], made["lora_alpha"]) == (4, 8)
    assert made["rank_pattern"]["^transformer_blocks.0.attn.to_q"] == 2 and "^time_embedder.linear_1" not in made["rank_pattern"]
    assert made["alpha_pattern"]["^transformer_blocks.1.attn.to_v"] == 2.5
    lora.save(tmp_path / FILE, adapter)
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(str(tmp_path), weight_name=FILE)
    layers = peft_layers(loaded)
    (name,) = loaded.peft_config
    assert sorted(layers) == adapter.modules
    for m, layer in layers.items():
        assert (layer.r[name], layer.lora_alpha[name], layer.scaling[name]) == (ranks[m], alphas[m], adapter.scaling(m)), m
    lay, args = inputs()
    want = torch_forward(loaded, lay, args)
    got = mlx_forward(lora.attach(state_mx(model), cfg, adapter), cfg, lay, args)
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    assert lora.load(tmp_path / FILE).alpha == adapter.alpha


def test_no_metadata_is_scaling_one(tmp_path):
    """A file with no `lora_adapter_metadata` and two ranks: diffusers (`get_peft_kwargs`) gives every module
    alpha == its own rank. Ours must too, and not, say, the first rank for all."""
    model = randomised(TINY, 2)
    peft_model = with_peft_adapter(model, r=4, lora_alpha=4, rank_pattern={"ff.net.2": 6})
    sd = {"transformer." + k: v for k, v in get_peft_model_state_dict(peft_model).items()}
    save_file(sd, str(tmp_path / FILE))
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(str(tmp_path), weight_name=FILE)
    (name,) = loaded.peft_config
    layers = peft_layers(loaded)
    assert {layer.scaling[name] for layer in layers.values()} == {1.0} and {layer.r[name] for layer in layers.values()} == {4, 6}
    adapter = lora.load(tmp_path / FILE)
    assert all(adapter.scaling(m) == 1.0 for m in adapter.modules)
    assert {adapter.rank(m) for m in adapter.modules} == {4, 6}
    cfg = dict(model.config)
    attached = lora.attach(state_mx(model), cfg, adapter)
    assert not any(k.endswith(lora.SCALING) for k in attached), "scaling 1 adds no key: plinear's HyperFlow path"
    lay, args = inputs()
    assert_close(mlx_forward(attached, cfg, lay, args)[0], torch_forward(loaded, lay, args)[0], atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("entry,scalings", [("2", {0.5, 2 / 6}), ("1.5", {1.5 / 4, 1.5 / 6}), ("sixteen", {1.0})])
def test_header_alpha_is_one_network_alpha(tmp_path, entry, scalings):
    """An alpha-less file whose safetensors header records `alpha`: `MiniMaxH3LoraLoaderMixin.lora_state_dict` scales
    every module by alpha / its rank, and ignores an entry that is not a number."""
    model = randomised(TINY, 2)
    peft_model = with_peft_adapter(model, r=4, lora_alpha=4, rank_pattern={"ff.net.2": 6})
    sd = {"transformer." + k: v for k, v in get_peft_model_state_dict(peft_model).items()}
    save_file(sd, str(tmp_path / FILE), metadata={"alpha": entry})
    state_dict, metadata = MiniMaxH3LoraLoaderMixin.lora_state_dict(str(tmp_path), weight_name=FILE, return_lora_metadata=True)
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(state_dict, metadata=metadata)
    (name,) = loaded.peft_config
    layers = peft_layers(loaded)
    assert {layer.scaling[name] for layer in layers.values()} == scalings
    adapter = lora.load(tmp_path / FILE)
    for m, layer in layers.items():
        assert adapter.scaling(m) == layer.scaling[name], m


def test_peft_state_dict_names(tmp_path):
    """PEFT's own state dict (`.default.` after lora_A / lora_B, no component prefix): diffusers' mixin renames it, and
    so do we, to the same keys."""
    model = randomised(TINY, 2)
    peft_model = with_peft_adapter(model, r=4, lora_alpha=4)
    dump = {k: v for k, v in peft_model.state_dict().items() if ".lora_" in k}
    assert all(".default.weight" in k for k in dump)
    want = MiniMaxH3LoraLoaderMixin.lora_state_dict(dict(dump))
    adapter = lora.read({k: to_mx(v) for k, v in dump.items()})
    assert {"transformer." + k for k in adapter.tensors} == set(want)
    for k, v in adapter.tensors.items():
        assert np.array_equal(to_np(v), to_np(want["transformer." + k])), k


def test_transformer_ref_prefix(tmp_path):
    adapter = random_adapter(dict(TINY), {m: 4 for m in MODULES[:6]}, {m: 4 for m in MODULES[:6]})
    lora.save(tmp_path / "ref.safetensors", adapter, prefix="transformer_ref")
    assert lora.load(tmp_path / "ref.safetensors").modules == adapter.modules  # one component: found without being told
    both = {**{"transformer." + k: v for k, v in adapter.tensors.items()},
            **{"transformer_ref." + k: v * 2 for k, v in adapter.tensors.items()}}
    with pytest.raises(ValueError, match="say which"):
        lora.read(both)
    ref = lora.read(both, prefix="transformer_ref")
    k = MODULES[0] + ".lora_B.weight"
    assert np.array_equal(to_np(ref.tensors[k]), to_np(adapter.tensors[k] * 2))


def test_prefix_reads_one_component_and_ignores_the_others():
    """A file with a transformer adapter and a text encoder's DoRA keys. `load_lora_adapter(prefix="transformer")`
    filters by the prefix and never sees the rest; ours the same, where it used to refuse the whole file for the other
    component's DoRA. With no prefix a file of several components is refused. DoRA in the component that is read is
    still refused (diffusers loads it as DoRA)."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    mine = MODULES[:6]
    adapter = random_adapter(cfg, {m: 4 for m in mine}, {m: 4 for m in mine})
    file = {"transformer." + k: torch.from_numpy(to_np(v)) for k, v in adapter.tensors.items()}
    file.update({"text_encoder.layers.0.q_proj.lora_A.weight": torch.zeros(4, 32),
                 "text_encoder.layers.0.q_proj.lora_B.weight": torch.zeros(32, 4),
                 "text_encoder.layers.0.q_proj.lora_magnitude_vector": torch.ones(32)})
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter(dict(file), prefix="transformer")
    as_mx = {k: to_mx(v) for k, v in file.items()}
    got = lora.read(as_mx, prefix="transformer")
    assert got.modules == sorted(peft_layers(loaded)) == sorted(mine)
    lay, args = inputs()
    out = mlx_forward(lora.attach(state_mx(model), cfg, got), cfg, lay, args)
    assert_close(out[0], torch_forward(loaded, lay, args)[0], atol=2e-4, rtol=2e-4, what="video velocity")
    with pytest.raises(ValueError, match="with and without"):
        lora.read(as_mx)

    module = mine[0]
    dora = {k: v for k, v in file.items() if k.startswith("transformer.")}
    dora[f"transformer.{module}.lora_magnitude_vector"] = torch.ones(lora.wired_shapes(cfg)[module][0])
    as_dora = copy.deepcopy(model)
    as_dora.load_lora_adapter(dict(dora), prefix="transformer")
    (name,) = as_dora.peft_config
    assert peft_layers(as_dora)[module].use_dora[name], "diffusers: a DoRA layer"
    for prefix in ("transformer", None):
        with pytest.raises(NotImplementedError, match="DoRA"):
            lora.read({k: to_mx(v) for k, v in dora.items()}, prefix=prefix)


# --- initialisation ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 48), (16, 300), (7, 13)])
def test_kaiming_uniform_is_torchs(shape):
    want = torch.nn.init.kaiming_uniform_(torch.empty(shape), a=math.sqrt(5), generator=torch.Generator().manual_seed(9))
    got = lora.kaiming_uniform(shape, torch.Generator().manual_seed(9))
    assert got.dtype == np.float32 and np.array_equal(got, want.numpy())
    bound = 1 / math.sqrt(shape[1])  # what the gain and the sqrt(3) come to; a fan taken from the rank would break it
    assert np.abs(got).max() <= bound * (1 + 1e-6)
    if shape[0] * shape[1] > 1000:
        assert np.abs(got).max() > 0.99 * bound


@pytest.mark.parametrize("out_f,in_f,r", [(48, 64, 4), (96, 48, 7), (20, 300, 16)])
def test_init_draw_is_pefts(out_f, in_f, r):
    """A PEFT layer built after `torch.manual_seed(s)`: its `update_layer` makes two `nn.Linear`s, whose constructors
    draw lora_A's (r * in) and lora_B's (out * r) default values, and then `reset_lora_parameters` redraws lora_A and
    zeroes lora_B. Skip the first two draws on a generator seeded the same and ours is PEFT's lora_A, bit for bit."""
    base = torch.nn.Linear(in_f, out_f, bias=False)
    torch.manual_seed(11)
    layer = PeftLinear(base, "default", LoraConfig(r=r, lora_alpha=3), r=r, lora_alpha=3)
    g = torch.Generator().manual_seed(11)
    torch.empty(r * in_f + out_f * r).uniform_(generator=g)
    assert np.array_equal(lora.kaiming_uniform((r, in_f), g), layer.lora_A["default"].weight.detach().numpy())
    assert not layer.lora_B["default"].weight.any()


def test_init_is_a_float32_no_op():
    model = to_mixed(randomised(TINY, 2))
    cfg = dict(model.config)
    adapter = lora.init(cfg, rank=4, alpha=10, seed=5)
    assert adapter.modules == sorted(lora.block_modules(cfg)), "attention and feed-forward only: no time embedder"
    assert len(adapter.modules) == 6 * (cfg["num_layers"] + cfg["num_refiner_layers"])
    shapes = lora.wired_shapes(cfg)
    for m in adapter.modules:
        a, b = adapter.tensors[m + ".lora_A.weight"], adapter.tensors[m + ".lora_B.weight"]
        assert a.dtype == b.dtype == mx.float32, "trainable weights are float32 over a bf16 base"
        assert a.shape == (4, shapes[m][1]) and b.shape == (shapes[m][0], 4)
        assert not bool(mx.any(b != 0)) and float(mx.abs(a).max()) <= 1 / math.sqrt(shapes[m][1]) * (1 + 1e-6)
        assert adapter.scaling(m) == 2.5
    # one generator, sorted module order
    g = torch.Generator().manual_seed(5)
    for m in adapter.modules:
        assert np.array_equal(to_np(adapter.tensors[m + ".lora_A.weight"]), lora.kaiming_uniform((4, shapes[m][1]), g)), m
    assert not np.array_equal(to_np(lora.init(cfg, 4, seed=6).tensors[adapter.modules[0] + ".lora_A.weight"]),
                              to_np(adapter.tensors[adapter.modules[0] + ".lora_A.weight"]))
    assert lora.init(cfg, 4).alpha[adapter.modules[0]] == 4, "alpha defaults to the rank"
    # lora_B is zero: the forward is the base model's, bit for bit, in float32 and in the release's mixed precision
    lay, args = inputs()
    base = state_mx(model)
    plain = mlx_forward(base, cfg, lay, args)
    for keep in (False, True):
        with_init = mlx_forward(lora.attach(base, cfg, adapter, keep_dtype=keep), cfg, lay, args)
        assert np.array_equal(to_np(with_init[0]), to_np(plain[0])) and np.array_equal(to_np(with_init[1]), to_np(plain[1]))


# --- dtypes ------------------------------------------------------------------------------------------------------------


def test_adapter_dtypes_follow_peft(tmp_path):
    """Over the release's mixed precision (bf16 blocks, float32 time embedder). Attached the default way, each matrix
    takes its base weight's dtype, as PEFT's injection gives it. `keep_dtype=True` leaves float32 factors over the bf16
    blocks, as `cast_training_params` leaves a trainer's; both forwards stay as close to the float32 run as torch's."""
    cfg = dict(TINY, num_layers=4)
    model = randomised(cfg, 4)
    mixed = to_mixed(model)
    modules = lora.block_modules(cfg) + ["time_embedder.linear_1", "time_embedder.linear_2"]
    adapter = random_adapter(cfg, {m: 4 for m in modules}, {m: 10 for m in modules}, std=0.2)
    lora.save(tmp_path / FILE, adapter)
    lay, args = inputs(5)
    truth_m = copy.deepcopy(model)
    truth_m.load_lora_adapter(str(tmp_path), weight_name=FILE)
    truth = torch_forward(truth_m, lay, args)

    injected = copy.deepcopy(mixed)
    injected.load_lora_adapter(str(tmp_path), weight_name=FILE)
    (name,) = injected.peft_config
    trainer = copy.deepcopy(injected)
    cast_training_params(trainer, dtype=torch.float32)
    base = state_mx(mixed)
    block, time = "transformer_blocks.0.attn.to_q", "time_embedder.linear_1"
    for torch_model, keep, block_dtype in ((injected, False, (torch.bfloat16, mx.bfloat16)), (trainer, True, (torch.float32, mx.float32))):
        layers = peft_layers(torch_model)
        assert layers[block].lora_A[name].weight.dtype == block_dtype[0] and layers[block].base_layer.weight.dtype == torch.bfloat16
        assert layers[time].lora_A[name].weight.dtype == torch.float32
        params = lora.attach(base, cfg, adapter, keep_dtype=keep)
        assert params[block + ".lora_A.weight"].dtype == block_dtype[1] and params[block + ".weight"].dtype == mx.bfloat16
        assert params[time + ".lora_A.weight"].dtype == mx.float32
        t_out = torch_forward(torch_model, lay, args)
        m_out = mlx_forward(params, cfg, lay, args)
        for i, what in enumerate(("video", "audio")):
            assert m_out[i].dtype == mx.float32
            e_torch, e_mlx = rel_err(t_out[i], truth[i]), rel_err(m_out[i], truth[i])
            print(f"keep_dtype={keep} {what}: torch rel err {e_torch:.4f}, mlx rel err {e_mlx:.4f}")
            assert e_mlx <= 1.5 * e_torch + 1e-3, f"{what}: mlx {e_mlx:.4f} vs torch {e_torch:.4f}"


# --- merging -----------------------------------------------------------------------------------------------------------


def test_merged_equals_unmerged(tmp_path):
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    adapter = random_adapter(cfg, {m: 4 for m in MODULES}, {m: 10 for m in MODULES})
    lora.save(tmp_path / FILE, adapter)
    peft_model = copy.deepcopy(model)
    peft_model.load_lora_adapter(str(tmp_path), weight_name=FILE)
    lay, args = inputs()
    want = torch_forward(peft_model, lay, args)
    base = state_mx(model)
    merged = lora.merge(base, cfg, adapter)
    assert set(merged) == set(base), "merged weights carry no LoRA keys"
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" not in base, "merge does not modify its input"
    got_merged = mlx_forward(merged, cfg, lay, args)
    got_unmerged = mlx_forward(lora.attach(base, cfg, adapter), cfg, lay, args)
    for i, what in enumerate(("video", "audio")):
        assert_close(got_merged[i], got_unmerged[i], atol=2e-4, rtol=2e-4, what=f"{what}: merged against unmerged")
        assert_close(got_merged[i], want[i], atol=2e-4, rtol=2e-4, what=f"{what}: merged against PEFT unmerged")
    assert rel_err(got_merged[0], mlx_forward(base, cfg, lay, args)[0]) > GUARD
    # PEFT's own merge: the same weights, and the same forward
    peft_model.fuse_lora()
    layers = peft_layers(peft_model)
    assert all(layer.merged for layer in layers.values())
    for m, layer in layers.items():
        assert_close(merged[m + ".weight"], layer.base_layer.weight, atol=5e-6, rtol=5e-6, what=m)
    assert_close(got_merged[0], torch_forward(peft_model, lay, args)[0], atol=2e-4, rtol=2e-4, what="against PEFT merged")


def test_merge_takes_plain_weights_only():
    """Attaching and merging are alternatives. Weights that already carry the adapter are refused: merging into them
    would leave its factors in place, and the forward would apply them on top of the merged weights. The order that
    works is to merge into the plain weights and attach whatever else afterwards."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = random_adapter(cfg, {m: 4 for m in MODULES}, {m: 10 for m in MODULES})
    other = random_adapter(cfg, {m: 3 for m in MODULES}, {m: 6 for m in MODULES}, seed=9)
    attached = lora.attach(base, cfg, adapter)
    with pytest.raises(ValueError, match="already carry a LoRA.*applied twice"):
        lora.merge(attached, cfg, adapter)
    with pytest.raises(ValueError, match="already carry a LoRA"):  # another adapter's factors on the same Linears
        lora.merge(attached, cfg, other)
    hf_tensors, hf_md = make_file(model, TINY)
    hf_params, hf_cfg = hyperflow.attach(base, cfg, {k: to_mx(v) for k, v in hf_tensors.items()},
                                         hyperflow.HyperFlowMeta.from_header(hf_md))
    with pytest.raises(ValueError, match="already carry a LoRA"):
        lora.merge(hf_params, hf_cfg, adapter)

    lay, args = inputs()
    once = mlx_forward(attached, cfg, lay, args)[0]
    merged = lora.merge(base, cfg, adapter)
    assert_close(mlx_forward(merged, cfg, lay, args)[0], once, atol=2e-4, rtol=2e-4, what="merged into plain weights")
    # what the refusal prevents: the adapter on top of its own merge is the adapter applied twice, far from once
    twice = mlx_forward(lora.attach(merged, cfg, adapter), cfg, lay, args)[0]
    assert rel_err(twice, once) > GUARD
    # merge, then attach another adapter: the same as merging both
    both = mlx_forward(lora.attach(merged, cfg, other), cfg, lay, args)[0]
    assert_close(both, mlx_forward(lora.merge(merged, cfg, other), cfg, lay, args)[0], atol=2e-4, rtol=2e-4, what="merged + attached")
    assert rel_err(both, once) > GUARD


@pytest.mark.parametrize("base_dtype,lora_dtype", [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16),
                                                   (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("r,alpha", [(4, 4), (4, 10), (3, 1)])
def test_delta_weight_and_merge_are_pefts(base_dtype, lora_dtype, r, alpha):
    """`get_delta_weight` and `merge` on one PEFT layer, for float32, bf16 and a float32 adapter over a bf16 weight.
    The float32 products can differ in their last bit (matmul order), so bf16 results are compared to one bf16 step."""
    base = torch.nn.Linear(24, 40, bias=False).to(base_dtype)
    base.weight.requires_grad_(False)  # frozen, as in a trainer: only the adapter is cast below
    layer = randomise_b(PeftLinear(base, "default", LoraConfig(r=r, lora_alpha=alpha), r=r, lora_alpha=alpha))
    if lora_dtype != base_dtype:
        cast_training_params(layer, dtype=lora_dtype)
    a_t, b_t = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    assert a_t.dtype == b_t.dtype == lora_dtype
    w0 = to_mx(base.weight)
    a, b = to_mx(a_t), to_mx(b_t)
    want_delta = layer.get_delta_weight("default")
    got_delta = lora.delta_weight(a, b, alpha / r)
    tol = dict(atol=1e-6, rtol=1e-6) if lora_dtype == torch.float32 else dict(atol=1e-4, rtol=2 ** -7)
    assert str(got_delta.dtype).split(".")[-1] == str(want_delta.dtype).split(".")[-1]
    assert_close(got_delta, want_delta, what="delta weight", **tol)
    layer.merge()
    got = (w0 + got_delta).astype(w0.dtype)
    tol = dict(atol=1e-6, rtol=1e-6) if base_dtype == torch.float32 else dict(atol=1e-4, rtol=2 ** -7)
    assert got.dtype == w0.dtype
    assert_close(got, base.weight, what="merged weight", **tol)
    if alpha != r:  # guard: an unscaled delta is outside the tolerance
        wrong = (w0 + lora.delta_weight(a, b, 1.0)).astype(w0.dtype)
        assert rel_err(wrong, base.weight) > GUARD


# --- what is refused ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -4.0, float("nan"), float("inf")])
def test_alpha_must_be_positive_and_finite(tmp_path, bad):
    """PEFT takes any number as alpha: its layer's scaling is then zero, negative, NaN or infinite, and its config
    writes NaN or Infinity into the header, which a strict JSON reader refuses. Ours refuses such an alpha wherever
    one comes in (`init`, `read` from the metadata or the header) or goes out (`attach`, `merge`, `save`)."""
    layer = PeftLinear(torch.nn.Linear(4, 4), "default", LoraConfig(r=4, lora_alpha=bad), r=4, lora_alpha=bad)
    scaling = layer.scaling["default"]
    assert scaling == bad / 4 or (math.isnan(scaling) and math.isnan(bad)), "PEFT accepts it"
    if not math.isfinite(bad):
        written = json.dumps(LoraConfig(r=4, lora_alpha=bad).to_dict()["lora_alpha"])  # what diffusers' save writes
        with pytest.raises(ValueError, match="not JSON"):
            json.loads(written, parse_constant=not_json)

    model = randomised(TINY, 0)
    cfg, base = dict(model.config), state_mx(model)
    good = random_adapter(cfg, {m: 4 for m in MODULES}, {m: 10 for m in MODULES})
    one_bad = lora.Adapter(good.tensors, {**good.alpha, MODULES[3]: bad})
    with pytest.raises(ValueError, match="positive, finite"):
        lora.init(cfg, rank=4, alpha=bad)
    for use in (lambda: lora.check(one_bad, cfg), lambda: lora.attach(base, cfg, one_bad), lambda: lora.merge(base, cfg, one_bad),
                lambda: lora.save(tmp_path / "bad.safetensors", one_bad)):
        with pytest.raises(ValueError, match=f"{MODULES[3]}.*positive, finite"):
            use()
    assert not (tmp_path / "bad.safetensors").exists(), "nothing written"
    described = lora.config(good)
    for header in ({"alpha": str(bad)}, {lora.METADATA_KEY: json.dumps({**described, "lora_alpha": bad})},
                   {lora.METADATA_KEY: json.dumps({**described, "alpha_pattern": {"attn.to_q": bad}})}):
        with pytest.raises(ValueError, match="positive, finite"):
            lora.read(good.tensors, header)


def not_json(token):
    raise ValueError(f"{token} is not JSON")


def test_saved_metadata_is_strict_json(tmp_path, monkeypatch):
    adapter = random_adapter(dict(TINY), {m: 4 for m in MODULES}, {m: (2.5 if "to_v" in m else 10) for m in MODULES})
    lora.save(tmp_path / FILE, adapter)
    header, _ = header_of(tmp_path / FILE)
    parsed = json.loads(header[lora.METADATA_KEY], parse_constant=not_json)
    assert parsed["transformer.lora_alpha"] == 10 and set(parsed["transformer.alpha_pattern"].values()) == {2.5}
    # the writer itself refuses a token that is not JSON, even with the alpha check out of the way
    monkeypatch.setattr(lora, "check_alpha", lambda what, alpha: None)
    with pytest.raises(ValueError, match="JSON"):
        lora.save(tmp_path / "nan.safetensors", lora.Adapter(adapter.tensors, {m: float("nan") for m in MODULES}))
    assert not (tmp_path / "nan.safetensors").exists()


def test_an_adapter_must_match_its_own_modules(tmp_path):
    """An `Adapter` is a dict of tensors and a dict of alphas. If they disagree (a stray tensor, half a pair, a pair
    with no alpha, an alpha with no pair), `check` says so, and with it `attach`, `merge` and `save`: `save` used to
    write such a file, which `load` then refused, and a missing matrix was a bare KeyError."""
    model = randomised(TINY, 0)
    cfg, base = dict(model.config), state_mx(model)
    good = random_adapter(cfg, {m: 4 for m in MODULES[:4]}, {m: 4 for m in MODULES[:4]})
    first, other = MODULES[0], MODULES[5]
    a, b = first + ".lora_A.weight", first + ".lora_B.weight"
    wrong = {
        "a stray tensor": lora.Adapter({**good.tensors, "stray": mx.zeros((2, 2))}, dict(good.alpha)),
        "no lora_B": lora.Adapter({k: v for k, v in good.tensors.items() if k != b}, dict(good.alpha)),
        "no lora_A": lora.Adapter({k: v for k, v in good.tensors.items() if k != a}, dict(good.alpha)),
        "a pair without an alpha": lora.Adapter(dict(good.tensors), {m: v for m, v in good.alpha.items() if m != first}),
        "an alpha without a pair": lora.Adapter(dict(good.tensors), {**good.alpha, other: 4}),
    }
    for what, adapter in wrong.items():
        for n, use in enumerate((lambda: lora.check_keys(adapter), lambda: lora.check(adapter, cfg),
                                 lambda: lora.attach(base, cfg, adapter), lambda: lora.merge(base, cfg, adapter),
                                 lambda: lora.save(tmp_path / f"{n}.safetensors", adapter))):
            with pytest.raises(ValueError, match="do not match its modules"):
                use()
    assert not list(tmp_path.iterdir()), "nothing written"
    lora.save(tmp_path / "good.safetensors", good)
    assert lora.load(tmp_path / "good.safetensors").modules == good.modules


def test_an_empty_prefix_is_refused(tmp_path):
    """`prefix=""` is not `prefix=None`. diffusers takes it as a real prefix: its filter keeps keys that start with a
    dot, so it loads nothing from an ordinary file, and written that way every key gets a leading dot. Ours refuses."""
    model = randomised(TINY, 0)
    adapter = random_adapter(dict(TINY), {m: 4 for m in MODULES[:4]}, {m: 4 for m in MODULES[:4]})
    plain = {k: torch.from_numpy(to_np(v)) for k, v in adapter.tensors.items()}
    theirs = copy.deepcopy(model)
    theirs.load_lora_adapter(dict(plain), prefix="")
    assert not peft_layers(theirs), "diffusers: nothing loaded"
    with pytest.raises(ValueError, match="empty prefix"):
        lora.save(tmp_path / "a.safetensors", adapter, prefix="")
    assert not (tmp_path / "a.safetensors").exists()
    with pytest.raises(ValueError, match="empty prefix"):
        lora.read(adapter.tensors, prefix="")


@pytest.mark.parametrize("bias", ["all", "lora_only"])
def test_a_bias_setting_other_than_none_is_refused(bias):
    """With `bias="all"` or `"lora_only"` PEFT also trains base-layer biases and writes them with the adapter (shown:
    the state dict then holds `.bias` tensors). Metadata that says so describes more than the two matrices the port
    reads. diffusers loads such metadata without a word; ours refuses it, with the other settings it does not
    implement. `"none"`, and no entry at all, are fine."""
    model = randomised(TINY, 0)
    modules = ["time_embedder.linear_1", "transformer_blocks.0.attn.to_q"]
    theirs = copy.deepcopy(model)
    theirs.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=modules, bias=bias))
    written = get_peft_model_state_dict(theirs)
    assert any(k.endswith(".bias") for k in written), "PEFT saves biases with this setting"
    adapter = random_adapter(dict(TINY), {m: 4 for m in modules}, {m: 4 for m in modules})
    described = lora.config(adapter)
    loaded = copy.deepcopy(model)
    loaded.load_lora_adapter({k: torch.from_numpy(to_np(v)) for k, v in adapter.tensors.items()}, prefix=None,
                             metadata={**described, "bias": bias})
    assert sorted(peft_layers(loaded)) == sorted(modules), "diffusers: loaded, no complaint"
    with pytest.raises(NotImplementedError, match=f"bias='{bias}'"):
        lora.read(adapter.tensors, {lora.METADATA_KEY: json.dumps({**described, "bias": bias})})
    for fine in ({**described, "bias": "none"}, {k: v for k, v in described.items() if k != "bias"}):
        assert lora.read(adapter.tensors, {lora.METADATA_KEY: json.dumps(fine)}).modules == sorted(modules)


def test_refusals(tmp_path):
    model = randomised(TINY, 0)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = random_adapter(cfg, {m: 4 for m in MODULES}, {m: 10 for m in MODULES})
    t = adapter.tensors
    attached = lora.attach(base, cfg, adapter)
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" not in base, "attach does not modify its input"
    assert attached["transformer_blocks.0.attn.to_q" + lora.SCALING].item() == 2.5
    with pytest.raises(ValueError, match="already carry a LoRA"):
        lora.attach(attached, cfg, adapter)
    hf_tensors, hf_md = make_file(model, TINY)
    hf_params, hf_cfg = hyperflow.attach(base, cfg, {k: to_mx(v) for k, v in hf_tensors.items()},
                                         hyperflow.HyperFlowMeta.from_header(hf_md))
    with pytest.raises(ValueError, match="already carry a LoRA"):  # an unnamed adapter shares its Linears with none
        lora.attach(hf_params, hf_cfg, adapter)

    # a target that is not one of the transformer's Linears: the forward would silently skip a LoRA there
    for module, shape_a, shape_b in (("transformer_blocks.0.norm1", (4, 48), (48, 4)), ("nowhere.linear", (4, 16), (48, 4)),
                                     ("endpoint_time_embedder.linear_1", (4, 32), (40, 4))):
        extra = lora.read({**t, module + ".lora_A.weight": mx.zeros(shape_a), module + ".lora_B.weight": mx.zeros(shape_b)})
        with pytest.raises(ValueError, match="not Linears of this transformer"):
            lora.attach(base, cfg, extra)
        with pytest.raises(ValueError, match="not Linears of this transformer"):
            lora.merge(base, cfg, extra)
    with pytest.raises(ValueError, match="do not fit"):
        lora.attach(base, cfg, lora.read({**t, "transformer_blocks.0.attn.to_q.lora_A.weight": mx.zeros((4, 7))}))
    with pytest.raises(ValueError, match="do not fit"):  # lora_A and lora_B of different ranks
        lora.attach(base, cfg, lora.read({**t, "transformer_blocks.0.attn.to_q.lora_A.weight": mx.zeros((5, 48))}))
    with pytest.raises(ValueError, match="no LoRA tensors"):
        lora.attach(base, cfg, lora.Adapter({}, {}))

    # keys
    with pytest.raises(ValueError, match="only one of"):
        lora.read({k: v for k, v in t.items() if k != "transformer_blocks.0.attn.to_q.lora_B.weight"})
    with pytest.raises(ValueError, match="unexpected key"):
        lora.read({**t, "transformer_blocks.0.attn.to_q.lora_B.bias": mx.zeros((64,))})
    with pytest.raises(ValueError, match="unexpected key"):
        lora.read({**t, "transformer_blocks.0.attn.to_q.alpha": mx.array(4.0)})
    with pytest.raises(NotImplementedError, match="DoRA"):
        lora.read({**t, "transformer_blocks.0.attn.to_q.lora_magnitude_vector": mx.zeros((64,))})
    with pytest.raises(ValueError, match="with and without"):
        lora.read({**{"transformer." + k: v for k, v in t.items()}, "text_encoder.x.lora_A.weight": mx.zeros((1, 1))})
    with pytest.raises(ValueError, match="no LoRA tensors"):
        lora.read({})

    # metadata
    good = lora.config(adapter)
    for flag in ("use_dora", "use_rslora", "lora_bias", "fan_in_fan_out"):
        with pytest.raises(NotImplementedError, match=flag):
            lora.read(t, {lora.METADATA_KEY: json.dumps({**good, flag: True})})
    with pytest.raises(ValueError, match="says rank 8"):
        lora.read(t, {lora.METADATA_KEY: json.dumps({**good, "r": 8})})
    with pytest.raises(ValueError, match="no `r`"):  # prefixed tensors, metadata without the prefix
        lora.read({"transformer." + k: v for k, v in t.items()}, {lora.METADATA_KEY: json.dumps(good)})
    assert lora.read(t, {lora.METADATA_KEY: json.dumps(good)}).alpha == adapter.alpha

    # saving, init, merging
    with pytest.raises(ValueError, match=".safetensors"):
        lora.save(tmp_path / "a.bin", adapter)
    with pytest.raises(ValueError, match="no LoRA tensors"):
        lora.save(tmp_path / "a.safetensors", lora.Adapter({}, {}))
    with pytest.raises(ValueError, match="positive"):
        lora.init(cfg, rank=0)
    with pytest.raises(ValueError, match="not a Linear"):
        lora.init(cfg, rank=4, modules=["transformer_blocks.0.norm1"])
    quantized = load.quantize_transformer(state_mx(MiniMaxH3(dict(TINY, hidden_size=64, ffn_dim=128))), 8, 32)
    q_cfg = dict(TINY, hidden_size=64, ffn_dim=128)
    q_adapter = lora.init(q_cfg, rank=4)
    with pytest.raises(ValueError, match="quantized"):
        lora.merge(quantized, q_cfg, q_adapter)
    on_q = lora.attach(quantized, q_cfg, q_adapter)  # attaching to quantized weights is fine: the adapter stays dense
    assert on_q["transformer_blocks.0.attn.to_q.lora_A.weight"].dtype == mx.float32


def MiniMaxH3(cfg):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    torch.manual_seed(4)
    return MiniMaxH3Transformer3DModel(**cfg).eval()
