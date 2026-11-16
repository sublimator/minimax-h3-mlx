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
        tensors[m + ".lora_A.weight"] = to_mx((torch.randn(r, in_f, generator=g) * std).to(dtype))
        tensors[m + ".lora_B.weight"] = to_mx((torch.randn(out_f, r, generator=g) * std).to(dtype))
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
    with pytest.raises(ValueError, match="already carry a LoRA"):  # two adapters on one Linear: not yet
        lora.attach(hf_params, hf_cfg, adapter)

    # a LoRA the forward would silently skip
    for module, shape_a, shape_b in (("proj_in", (4, 16), (48, 4)), ("transformer_blocks.0.adaln_proj.linear", (4, 24), (864, 4)),
                                     ("endpoint_time_embedder.linear_1", (4, 32), (40, 4))):
        extra = lora.read({**t, module + ".lora_A.weight": mx.zeros(shape_a), module + ".lora_B.weight": mx.zeros(shape_b)})
        with pytest.raises(ValueError, match="not run through ops.plinear"):
            lora.attach(base, cfg, extra)
        with pytest.raises(ValueError, match="not run through ops.plinear"):
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
        lora.init(cfg, rank=4, modules=["proj_in"])
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
