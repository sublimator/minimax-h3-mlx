"""A trained adapter at sampling: a scale on `lora.attach`, `lora.for_sampling`, `lora.from_header`, `lora.summary`.

The oracle for the scale is the tiny diffusers transformer with the adapter loaded (`load_lora_adapter`) and scaled the
two ways diffusers offers for this model: `set_adapters(name, weights=)`, which is PEFT's `set_scale`, and the forward's
own `attention_kwargs={"scale": ...}`. One Linear and the whole transformer are also held to the algebra, bit for bit.
The whole pipeline with an adapter is in test_p_h3_image.py, and the command line in test_run_p_h3_image.py.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, state_mx, to_mx, to_np
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from safetensors.torch import save_file
from test_dit_mixed_precision import rel_err
from test_dit_tiny import TINY
from test_hyperflow import randomised
from test_lora import FILE, GUARD, MODULES, inputs, mlx_forward, peft_layers, random_adapter, torch_forward
from test_lora_convert import LAYOUTS, WIDE, original_file

from h3mlx import load, lora, ops

SCALES = [0.6, -1.3, 2.0, 1 / 3]
ONE = "transformer_blocks.0.attn.to_q"


def mixed(cfg, modules=MODULES, std=0.3, seed=2):
    """Two ranks and two alphas, so the scalings differ from Linear to Linear (1.5 and 2.5 / 3)."""
    ranks = {m: (4, 3)[i % 2] for i, m in enumerate(modules)}
    alphas = {m: (6, 2.5)[i % 2] for i, m in enumerate(modules)}
    return random_adapter(cfg, ranks, alphas, seed=seed, std=std)


def one(x, params):
    """The Linear `ONE` (an attention query projection: no bias) on `x`."""
    return ops.plinear(x, params, ONE)


def bits(a):
    a = to_np(a)
    return a.view({2: np.uint16, 4: np.uint32}[a.dtype.itemsize])


def same_bits(a, b):
    return a.dtype == b.dtype and np.array_equal(bits(a), bits(b))


# --- the scale ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", SCALES)
def test_scale_is_diffusers_adapter_weight(tmp_path, scale):
    """The adapter at a scale against diffusers with the same file loaded and `set_adapters(name, weights=scale)`:
    every Linear's scaling is PEFT's number, and the forward is its forward. The forward's own
    `attention_kwargs={"scale": ...}` gives the same thing by the other route. Guard: the unscaled adapter is far."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    adapter = mixed(cfg)
    lora.save(tmp_path / FILE, adapter)
    theirs = copy.deepcopy(model)
    theirs.load_lora_adapter(str(tmp_path), weight_name=FILE, adapter_name="default")
    lay, args = inputs()
    unscaled = torch_forward(theirs, lay, args)
    with torch.no_grad():
        by_argument = theirs(*args, **lay, attention_kwargs={"scale": scale}, return_dict=False)
    theirs.set_adapters("default", weights=scale)
    want = torch_forward(theirs, lay, args)

    params = lora.attach(state_mx(model), cfg, lora.load(tmp_path / FILE), scale=scale)
    layers = peft_layers(theirs)
    assert set(layers) == set(adapter.modules)
    for module, layer in layers.items():
        assert layer.scaling["default"] == scale * adapter.alpha[module] / adapter.rank(module)
        ours = params.get(module + lora.SCALING)
        assert (1.0 if ours is None else ours.item()) == float(np.float32(layer.scaling["default"])), module
    got = mlx_forward(params, cfg, lay, args)
    for g, w, b, u, what in zip(got, want, by_argument, unscaled, ("video velocity", "audio velocity")):
        assert_close(g, w, atol=2e-4, rtol=2e-4, what=what)
        assert_close(g, b, atol=2e-4, rtol=2e-4, what=what + " (attention_kwargs)")
        assert rel_err(u, w) > GUARD, "the scale counts"


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("scale", SCALES)
def test_scale_on_one_linear_is_the_delta_times_the_scale(dtype, scale):
    """One Linear, bit for bit: the output is the base Linear's plus B(A x) times float32(scale * alpha / rank), the
    product rounded as torch rounds a tensor times a Python float. In float32 it is also PEFT's layer after
    `set_scale`."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = {k: v.astype(dtype) for k, v in state_mx(model).items()}
    r, alpha = 4, 6
    adapter = random_adapter(cfg, {ONE: r}, {ONE: alpha})
    params = lora.attach(base, cfg, adapter, scale=scale)
    a, b = params[ONE + ".lora_A.weight"], params[ONE + ".lora_B.weight"]
    assert a.dtype == b.dtype == dtype
    x = mx.array(np.random.default_rng(4).standard_normal((5, cfg["hidden_size"])).astype(np.float32)).astype(dtype)
    y = one(x, base)
    delta = ops.linear(ops.linear(x, a), b)
    want = y + ops.times_scalar(delta, mx.array(scale * alpha / r, dtype=mx.float32))
    got = one(x, params)
    assert same_bits(got, want)
    assert rel_err(got, one(x, lora.attach(base, cfg, adapter))) > GUARD, "the scale counts"

    if dtype == mx.float32:
        out_f, in_f = base[ONE + ".weight"].shape
        linear = torch.nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(to_np(base[ONE + ".weight"])))
        layer = PeftLinear(linear, "default", LoraConfig(r=r, lora_alpha=alpha), r=r, lora_alpha=alpha)
        with torch.no_grad():
            layer.lora_A["default"].weight.copy_(torch.from_numpy(to_np(a)))
            layer.lora_B["default"].weight.copy_(torch.from_numpy(to_np(b)))
            layer.set_scale("default", scale)
            theirs = layer(torch.from_numpy(to_np(x)))
        assert_close(got, theirs, atol=2e-6, rtol=2e-6, what="PEFT's layer after set_scale")


@pytest.mark.parametrize("scale", [0.5, 4.0, -2.0])
def test_a_power_of_two_scale_is_the_adapter_with_lora_b_scaled(scale):
    """The whole transformer, bit for bit, with an adapter on every Linear it has. Multiplying by a power of two is
    exact, so the adapter at `scale` must be the adapter whose `lora_B` is `scale` times as large, at scale 1: the
    scale reaches every Linear, as a factor on the delta and on nothing else."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = mixed(cfg, sorted(lora.wired_shapes(cfg)), std=0.2)
    larger = lora.Adapter({k: (v * scale if ".lora_B." in k else v) for k, v in adapter.tensors.items()}, dict(adapter.alpha))
    lay, args = inputs()
    got = mlx_forward(lora.attach(base, cfg, adapter, scale=scale), cfg, lay, args)
    want = mlx_forward(lora.attach(base, cfg, larger), cfg, lay, args)
    plain = mlx_forward(lora.attach(base, cfg, adapter), cfg, lay, args)
    mx.eval(got, want, plain)
    for g, w, p in zip(got, want, plain):
        assert same_bits(g, w)
        assert rel_err(g, p) > GUARD, "the scale counts"


def test_scale_one_changes_nothing_and_a_named_adapter_takes_a_scale():
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = mixed(cfg)
    plain, one = lora.attach(base, cfg, adapter), lora.attach(base, cfg, adapter, scale=1.0)
    assert list(plain) == list(one)
    assert all(same_bits(plain[k], one[k]) for k in plain if isinstance(plain[k], mx.array))
    assert any(k.endswith(lora.SCALING) for k in plain)  # there are scalings to compare

    named = lora.attach(base, cfg, adapter, name="subject", scale=0.7)
    for module in adapter.modules:
        want = np.float32(0.7 * adapter.alpha[module] / adapter.rank(module))
        assert named[f"{module}{lora.SCALING}.subject"].item() == float(want)

    for bad in (float("nan"), float("inf"), -float("inf"), True, "1", None):
        with pytest.raises(ValueError, match="scale must be a finite number"):
            lora.attach(base, cfg, adapter, scale=bad)


# --- the weights a sampler runs ------------------------------------------------------------------------------------------


def test_for_sampling_attaches_and_leaves_the_base_alone():
    """The sampler's weights are the base's own arrays with the adapter's matrices beside them (cast to the base's
    dtype, scaled as `attach` scales), and the forward is `attach`'s: nothing is merged."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = {k: v.astype(mx.bfloat16) for k, v in state_mx(model).items()}
    before = dict(base)
    adapter = mixed(cfg)
    got = lora.for_sampling(base, cfg, adapter, 0.6)
    assert base.keys() == before.keys() and all(base[k] is before[k] for k in before), "the weights passed in are not modified"
    assert all(got[k] is base[k] for k in base), "the base weights are the same arrays: not merged into"
    want = lora.attach(base, cfg, adapter, scale=0.6)
    assert list(got) == list(want)
    added = [k for k in got if k not in base]
    assert len(added) == 3 * len(adapter.modules)  # two matrices and a scaling each
    assert all(same_bits(got[k], want[k]) for k in added)
    assert all(got[k].dtype == mx.bfloat16 for k in added if ".lora_A." in k or ".lora_B." in k)


def test_merging_into_bf16_loses_a_small_adapter_and_attaching_keeps_it():
    """Why the sampler attaches. A bf16 weight has eight significant bits, so a delta under half a step of every
    weight rounds away when it is merged: the merged weights are the base's, bit for bit, and the adapter is gone.
    Attached, the same adapter still moves the Linear's output, by `in * a * b` here (ones in, a rank-one delta)."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = {k: v.astype(mx.bfloat16) for k, v in state_mx(model).items()}
    out_f, in_f = base[ONE + ".weight"].shape
    rng = np.random.default_rng(0)
    # every weight in [2^-6, 2^-5): one bf16 step there is 2^-13, half of it 6.1e-5
    w = rng.uniform(2.0**-6, 2.0**-5, (out_f, in_f)) * rng.choice([-1.0, 1.0], (out_f, in_f))
    base[ONE + ".weight"] = mx.array(w.astype(np.float32)).astype(mx.bfloat16)
    a, b = 2.0**-7, 2.0**-8  # a * b = 3.05e-5: under half a step of every weight
    adapter = lora.Adapter({ONE + ".lora_A.weight": mx.full((1, in_f), a), ONE + ".lora_B.weight": mx.full((out_f, 1), b)},
                           {ONE: 1})
    merged = lora.merge(base, cfg, adapter)
    assert same_bits(merged[ONE + ".weight"], base[ONE + ".weight"]), "the whole delta rounds away"

    x = mx.ones((3, in_f), dtype=mx.bfloat16)
    plain = to_np(one(x, base).astype(mx.float32))
    sampled = to_np(one(x, lora.for_sampling(base, cfg, adapter, 1.0)).astype(mx.float32))
    assert np.array_equal(to_np(one(x, merged).astype(mx.float32)), plain)
    moved = np.abs(sampled - plain)
    print(f"merged: 0 of {out_f * in_f} weights changed; attached: the output moves by {moved.mean():.2e} (in * a * b = {in_f * a * b:.2e})")
    assert in_f * a * b * 0.7 < moved.mean() < in_f * a * b * 1.3


def test_scale_zero_is_the_weights_themselves():
    """At 0 the sampler runs the weights it was given: the same dict, so the same picture bit for bit. (diffusers at
    weight 0 keeps the adapter and adds a zero delta, which `attach(scale=0)` reproduces: the same numbers here.)"""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = mixed(cfg)
    for zero in (0, 0.0, -0.0):
        assert lora.for_sampling(base, cfg, adapter, zero) is base
    kept = lora.attach(base, cfg, adapter, scale=0.0)
    assert all(kept[m + lora.SCALING].item() == 0 for m in adapter.modules)
    lay, args = inputs()
    for g, w in zip(mlx_forward(kept, cfg, lay, args), mlx_forward(base, cfg, lay, args)):
        assert np.array_equal(to_np(g), to_np(w))


def test_an_adapter_attached_at_zero_is_not_bit_for_bit_nothing():
    """Why 0 attaches nothing instead of scaling by zero: adding a zero delta turns a negative zero in the Linear's
    output into a positive one. (On the CPU stream, where a product of a negative zero comes out negative.)"""
    with mx.stream(mx.cpu):
        p = {"l.weight": mx.array([[1.0]])}
        x = mx.array([[-0.0]])
        with_zero = dict(p, **{"l.lora_A.weight": mx.array([[-1.0]]), "l.lora_B.weight": mx.array([[1.0]]),
                               "l" + lora.SCALING: mx.array(0.0)})
        plain, added = to_np(ops.plinear(x, p, "l")), to_np(ops.plinear(x, with_zero, "l"))
    assert plain == added == 0 and np.signbit(plain).all() and not np.signbit(added).any()


def test_for_sampling_refuses():
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = mixed(cfg)
    for bad in (float("nan"), float("inf"), True, "1", None):
        with pytest.raises(ValueError, match="scale must be a finite number"):
            lora.for_sampling(base, cfg, adapter, bad)

    key = adapter.modules[3] + ".lora_B.weight"
    for value in (float("nan"), float("inf")):
        tensors = dict(adapter.tensors)
        poisoned = np.array(tensors[key])
        poisoned[1, 2] = value
        tensors[key] = mx.array(poisoned)
        for scale in (1.0, 0):  # a broken file is refused whatever the scale
            with pytest.raises(ValueError, match=f"1 of the adapter's {len(tensors)} matrices hold a NaN or an infinity.*{key}"):
                lora.for_sampling(base, cfg, lora.Adapter(tensors, dict(adapter.alpha)), scale)

    wide = dict(cfg, hidden_size=64)
    misfit = random_adapter(wide, {ONE: 4}, {ONE: 4})
    for scale in (1.0, 0):
        with pytest.raises(ValueError, match="do not fit the Linear"):
            lora.for_sampling(base, cfg, misfit, scale)
    q, v = "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_v"
    only_q, only_v = (random_adapter(cfg, {m: 1}, {m: 1}) for m in (q, v))
    for scale in (1.0, 0):  # at 0 too: the weights handed back would run the adapter already on them
        with pytest.raises(ValueError, match="already carry a LoRA"):
            lora.for_sampling(lora.attach(base, cfg, adapter), cfg, adapter, scale)
        for carried in (lora.attach(base, cfg, only_q), lora.attach(base, cfg, only_q, name="first")):
            with pytest.raises(ValueError, match="already carry a LoRA"):  # on another Linear than the new adapter's
                lora.for_sampling(carried, cfg, only_v, scale)
    assert lora.carried_lora_keys(base) == []


def test_a_scale_finite_in_python_and_not_in_float32_is_refused():
    """A finite scale whose scaling, `scale * alpha / rank`, overflows float32 (the dtype the forward multiplies by)
    would put an infinity into every Linear the adapter is on. Refused at `attach`, so `for_sampling` refuses it too;
    the largest float32 scaling is still taken."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    base = state_mx(model)
    adapter = mixed(cfg)
    for bad in (1e40, -1e40, 1e308):
        with pytest.raises(ValueError, match="not finite in float32"):
            lora.attach(base, cfg, adapter, scale=bad)
        with pytest.raises(ValueError, match="not finite in float32"):
            lora.for_sampling(base, cfg, adapter, bad)
    biggest = float(np.finfo(np.float32).max) / 2.5  # the largest alpha / rank here is 1.5: still finite
    got = lora.attach(base, cfg, adapter, scale=biggest)
    assert all(np.isfinite(got[m + lora.SCALING].item()) for m in adapter.modules)


def test_factors_finite_in_the_file_and_not_in_the_base_dtype_are_refused():
    """The file's matrices are checked as they are, then again once cast to their base weight's dtype: a float32 factor
    of 1e5 is finite and is an infinity in float16 (largest 65,504). bf16 has float32's range, so the same file over a
    bf16 base is taken."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    adapter = mixed(cfg)
    key = adapter.modules[3] + ".lora_B.weight"
    tensors = dict(adapter.tensors)
    big = np.array(tensors[key])
    big[1, 2] = 1e5
    tensors[key] = mx.array(big)
    wide = lora.Adapter(tensors, dict(adapter.alpha))
    half = {k: v.astype(mx.float16) for k, v in state_mx(model).items()}
    refused = f"1 of the adapter's {len(tensors)} matrices are finite in the file and not in their base weight's dtype"
    with pytest.raises(ValueError, match=f"{refused}.*{key}"):
        lora.for_sampling(half, cfg, wide, 1.0)
    bf16 = {k: v.astype(mx.bfloat16) for k, v in state_mx(model).items()}
    got = lora.for_sampling(bf16, cfg, wide, 1.0)
    assert np.isfinite(to_np(got[key].astype(mx.float32))).all()


def test_a_quantized_base_keeps_the_adapter_dense():
    """`--quantize` after `for_sampling`: the base matrices are quantized and nothing of the adapter is. One Linear is
    then the quantized base's output plus the dense delta, bit for bit; the whole forward is the forward over the
    matrices the quantized weights stand for, with the same adapter; and the adapter still counts."""
    cfg = dict(TINY, hidden_size=64, ffn_dim=128)
    model = randomised(cfg, 2)
    cfg = dict(model.config)
    adapter = mixed(cfg, lora.block_modules(cfg))
    attached = lora.for_sampling(state_mx(model), cfg, adapter, 0.7)
    targets = load.check_quantizable(attached)
    assert len(targets) == len(lora.block_modules(cfg)) and not any(".lora_" in k for k in targets)
    before = dict(attached)
    quantized = load.quantize_transformer(attached, 8)
    assert all(isinstance(quantized[k], ops.QuantizedWeight) for k in targets)
    others = [k for k in quantized if k not in targets]
    assert all(quantized[k] is before[k] for k in others), "everything else, the adapter included, is as it was"
    assert sum(".lora_" in k for k in others) == 3 * len(adapter.modules)

    x = mx.array(np.random.default_rng(4).standard_normal((5, 64)).astype(np.float32))
    a, b, s = (quantized[ONE + k] for k in (".lora_A.weight", ".lora_B.weight", lora.SCALING))
    base_only = {k: v for k, v in quantized.items() if ".lora_" not in k}
    want = one(x, base_only) + ops.times_scalar(ops.linear(ops.linear(x, a), b), s)
    assert same_bits(one(x, quantized), want)

    dense = {k: (mx.dequantize(v.wq, v.scales, v.biases, group_size=v.group_size, bits=v.bits)
                 if isinstance(v, ops.QuantizedWeight) else v) for k, v in quantized.items()}
    lay, args = inputs()
    video, audio, text = args
    args = (torch.randn(1, 12, 16), audio, text)
    got = mlx_forward(quantized, cfg, lay, args)[0]
    assert_close(got, mlx_forward(dense, cfg, lay, args)[0], atol=2e-4, rtol=2e-4, what="the dequantized base with the adapter")
    assert rel_err(got, mlx_forward(base_only, cfg, lay, args)[0]) > GUARD, "the adapter counts over a quantized base"


# --- an adapter from its file's header -----------------------------------------------------------------------------------


def files_of_every_kind(tmp_path):
    """{name: path}: every kind of file `lora.load` reads, written by the tools that write them."""
    cfg = dict(WIDE)
    adapter = mixed(cfg)
    out = {}
    for name, prefix in (("diffusers", "transformer"), ("transformer_ref", "transformer_ref"), ("no-prefix", None)):
        out[name] = tmp_path / f"{name}.safetensors"
        lora.save(out[name], adapter, prefix=prefix)
    plain = {"transformer." + k: torch.from_numpy(to_np(v)) for k, v in adapter.tensors.items()}
    out["no-metadata"] = tmp_path / "no-metadata.safetensors"
    save_file(plain, str(out["no-metadata"]))
    out["header-alpha"] = tmp_path / "header-alpha.safetensors"
    save_file(plain, str(out["header-alpha"]), metadata={"alpha": "8"})
    for layout in LAYOUTS:
        out[layout] = tmp_path / f"{layout}.safetensors"
        save_file(original_file(cfg, layout), str(out[layout]))
    out["original-header-alpha"] = tmp_path / "original-header-alpha.safetensors"
    save_file(original_file(cfg, "ai-toolkit"), str(out["original-header-alpha"]), metadata={"alpha": "8"})
    return cfg, out


def test_from_header_is_the_loaded_adapter_and_reads_no_tensor(tmp_path, monkeypatch):
    """For every kind of file the loader reads: the header alone gives the loaded adapter's modules, shapes and alphas,
    with the tensor loader made to raise."""
    cfg, paths = files_of_every_kind(tmp_path)
    loaded = {name: lora.load(path) for name, path in paths.items()}

    def refuse(*a, **k):
        raise AssertionError("a tensor file was opened")

    monkeypatch.setattr(mx, "load", refuse)
    for name, path in paths.items():
        got, want = lora.from_header(path), loaded[name]
        assert got.modules == want.modules and got.alpha == want.alpha, name
        assert {k: tuple(v.shape) for k, v in got.tensors.items()} == {k: tuple(v.shape) for k, v in want.tensors.items()}, name
        lora.check(got, cfg)
        assert lora.summary(got) == lora.summary(want), name
    assert len({tuple(a.alpha.values()) for a in loaded.values()}) > 2  # the kinds differ in their alphas


def test_from_header_refuses_what_is_not_a_lora_file(tmp_path):
    model = randomised(TINY, 2)
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"abc")
    with pytest.raises(ValueError, match="not a safetensors file"):
        lora.from_header(short)
    junk = tmp_path / "junk.safetensors"
    for header in (b"[1,2,3] ", b'{"a": 1}'):  # JSON of the stated length, but no table of tensors
        junk.write_bytes(len(header).to_bytes(8, "little") + header)
        with pytest.raises(ValueError, match="not a safetensors header"):
            lora.from_header(junk)
    weights = tmp_path / "weights.safetensors"
    save_file({k: v for k, v in model.state_dict().items() if "to_q" in k}, str(weights))
    with pytest.raises(ValueError, match="only <module>.lora_A/B.weight keys"):
        lora.from_header(weights)
    adapter = mixed(dict(model.config))
    half = {"transformer." + k: torch.from_numpy(to_np(v)) for k, v in adapter.tensors.items() if ".lora_A." in k}
    save_file(half, str(tmp_path / "half.safetensors"))
    with pytest.raises(ValueError, match="only one of lora_A / lora_B"):
        lora.from_header(tmp_path / "half.safetensors")
    with pytest.raises(FileNotFoundError):
        lora.from_header(tmp_path / "absent.safetensors")


def test_from_header_refuses_a_malformed_header_as_a_value_error(tmp_path):
    """The sampler turns a ValueError from the header into an argument error, before a model loads. Each of these
    used to escape as another exception (a TypeError, MLX's `std::bad_cast`, an AttributeError) or to be read as if
    whole (a header shorter than it says)."""
    def write(name, header: str, length=None):
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes((len(header.encode()) if length is None else length).to_bytes(8, "little") + header.encode())
        return path

    def entry(key, shape):
        return f'"{key}": {{"dtype": "F32", "shape": {shape}, "data_offsets": [0, 0]}}'

    a, b = entry("t.lora_A.weight", "[4, 8]"), entry("t.lora_B.weight", "[8, 4]")
    strings, shape = "__metadata__ is not a map of strings to strings", "shape of non-negative integers"
    cases = {
        "truncated": (write("truncated", "{" + a + "}", length=5000), "a header of 5000 bytes"),
        "metadata a list": (write("meta-list", '{"__metadata__": [1, 2], ' + a + "}"), strings),
        "metadata not strings": (write("meta-int", '{"__metadata__": {"alpha": 8}, ' + a + "}"), strings),
        "shape of text": (write("shape-text", "{" + entry("t.lora_A.weight", '["x", 4]') + "}"), shape),
        "shape of bools": (write("shape-bool", "{" + entry("t.lora_A.weight", "[true, 4]") + "}"), shape),
        "negative shape": (write("shape-neg", "{" + entry("t.lora_A.weight", "[-1, 4]") + "}"), shape),
        "lora metadata a list": (write("lora-meta", '{"__metadata__": {"lora_adapter_metadata": "[1, 2]"}, '
                                       + a + ", " + b + "}"), "is not a JSON object"),
    }
    for what, (path, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            lora.from_header(path)


def test_the_sampler_reads_an_original_file_at_the_transformers_head_width(tmp_path):
    """A DiffSynth-Studio file fuses q, k and v per head; reading it takes the head width, and the sampler's plan now
    passes the transformer's. At TINY's 2 x 32 the default of 128 refused a file that fits; at 4 x 32 (128 wide in all)
    it would have split q, k and v at the wrong stride with no error. The scripts' loads pass it as well."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_p_h3_image as R

    for cfg in (dict(TINY), dict(TINY, num_attention_heads=4, attention_head_dim=32)):
        path = tmp_path / f"diffsynth-{cfg['num_attention_heads']}.safetensors"
        save_file(original_file(cfg, "diffsynth"), str(path))
        plan = R.adapter_plan(path, 1.0, cfg)
        right = lora.load(path, head_dim=cfg["attention_head_dim"])
        assert plan["modules"] == len(right.modules)
        assert lora.summary(lora.from_header(path, head_dim=cfg["attention_head_dim"])) == lora.summary(right)
    for script in ("run_p_h3_image.py", "train_lora.py"):
        source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
        loads = [line for line in source.splitlines() if "lora.load(" in line or "lora.from_header(" in line]
        assert loads and all("head_dim=" in line for line in loads), (script, loads)


def test_the_plan_refuses_a_scale_that_overflows_float32(tmp_path):
    """The plan's summary refuses what `attach` would, so `--plan-only` does not pass a run that would fail."""
    model = randomised(TINY, 2)
    adapter = mixed(dict(model.config))
    with pytest.raises(ValueError, match="not finite in float32"):
        lora.summary(adapter, 1e40)


def test_summary_names_the_linears_the_ranks_and_the_scalings():
    cfg = dict(TINY, num_layers=6)
    ranks = {f"transformer_blocks.{i}.attn.to_q": 4 for i in (5, 0, 1, 2)}
    ranks.update({"transformer_blocks.3.attn.to_out.0": 3, "token_refiner.refiner_blocks.0.ff.net.0.proj": 4, "proj_in": 3,
                  "time_embedder.linear_1": 4})
    alphas = {m: (6.0 if r == 4 else 3) for m, r in ranks.items()}
    adapter = random_adapter(cfg, ranks, alphas)
    shapes = lora.wired_shapes(cfg)
    want = {
        "modules": 8,
        "module_kinds": {"proj_in": None, "time_embedder.linear_1": None,
                         "token_refiner.refiner_blocks.*.ff.net.0.proj": "0",
                         "transformer_blocks.*.attn.to_out.0": "3", "transformer_blocks.*.attn.to_q": "0-2,5"},
        "ranks": [3, 4], "alphas": [3.0, 6.0], "scalings": [1.0, 1.5],
        "parameters": sum(r * (shapes[m][0] + shapes[m][1]) for m, r in ranks.items()),
    }
    assert lora.summary(adapter) == want
    assert lora.summary(adapter, 0.5) == dict(want, scalings=[0.5, 0.75])
    assert lora.summary(adapter, -2) == dict(want, scalings=[-3.0, -2.0])
    assert lora.summary(adapter, 0)["scalings"] == [0.0]
    with pytest.raises(ValueError, match="scale must be a finite number"):
        lora.summary(adapter, float("nan"))
