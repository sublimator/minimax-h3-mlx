"""Two named adapters on the same Linears, a frozen one and a trainable one, against PEFT.

The oracle is the tiny diffusers transformer with both adapters loaded under names (`load_lora_adapter(adapter_name=)`)
and switched with `set_adapters`, PEFT's own layer for the rounding of a stacked forward, and torch autograd for which
weights a gradient reaches. The public P-H3-Image training adapter is checked by its header only.
"""

import copy
import json

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, fixture, state_mx, to_mx, to_np
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLinear
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open
from test_dit_mixed_precision import rel_err
from test_dit_tiny import TINY
from test_hyperflow import make_file, randomised
from test_lora import FILE, GUARD, inputs, mlx_forward, peft_layers, random_adapter, to_mx_half, torch_forward

from h3mlx import hyperflow, lora, ops

# the frozen one sits where the public training adapter does (attention and feed-forward); the trainable one shares
# most of those Linears, leaves some to the assistant alone, and has some of its own
ASSISTANT_MODULES = lora.block_modules(TINY)
SUBJECT_MODULES = [m for m in lora.block_modules(TINY) if "to_k" not in m] + [
    "transformer_blocks.0.adaln_proj.linear", "proj_in", "time_embedder.linear_2"]
STATES = [("assistant", "subject"), ("subject", "assistant"), ("assistant",), ("subject",), ()]


def two_adapters(cfg=TINY, dtype=torch.float32):
    assistant = random_adapter(cfg, {m: 4 for m in ASSISTANT_MODULES}, {m: 4 for m in ASSISTANT_MODULES}, seed=2, dtype=dtype)
    subject = random_adapter(cfg, {m: 3 for m in SUBJECT_MODULES}, {m: 6 for m in SUBJECT_MODULES}, seed=7)
    return assistant, subject


def peft_stack(model, tmp_path, assistant, subject):
    """Both adapters loaded into a copy of the model under their names, from files we wrote."""
    m = copy.deepcopy(model)
    for name, adapter in (("assistant", assistant), ("subject", subject)):
        (tmp_path / name).mkdir(exist_ok=True)
        lora.save(tmp_path / name / FILE, adapter)
        m.load_lora_adapter(str(tmp_path / name), weight_name=FILE, adapter_name=name)
    return m


def mlx_stack(model, assistant, subject, keep_subject_dtype=False):
    params = lora.attach(state_mx(model), dict(model.config), assistant, name="assistant")
    return lora.attach(params, dict(model.config), subject, keep_dtype=keep_subject_dtype, name="subject")


def activate(peft_model, names):
    if names:
        peft_model.set_adapters(list(names))
    else:
        peft_model.disable_lora()
    return peft_model


# --- names ---------------------------------------------------------------------------------------------------------------


def test_runtime_keys_are_pefts(tmp_path):
    """In the weights a named adapter's matrices carry PEFT's runtime names, `<module>.lora_A.<name>.weight`: the keys
    of PEFT's own state dict with two adapters loaded. What we add beside them is ours: a scaling where it is not 1,
    and each Linear's list of active adapters."""
    model = randomised(TINY, 2)
    assistant, subject = two_adapters()
    theirs = {k for k in peft_stack(model, tmp_path, assistant, subject).state_dict() if "lora_" in k}
    params = mlx_stack(model, assistant, subject)
    ours = {k for k in params if lora.RUNTIME_RE.match(k)}
    assert ours == theirs and len(ours) == 2 * (len(ASSISTANT_MODULES) + len(SUBJECT_MODULES))
    extra = {k for k in params if ".lora_" in k} - ours
    assert extra == {m + ".lora_scaling.subject" for m in SUBJECT_MODULES} | {
        m + lora.ADAPTERS for m in set(ASSISTANT_MODULES) | set(SUBJECT_MODULES)}
    assert lora.adapter_names(params) == ["assistant", "subject"]
    assert params["transformer_blocks.0.attn.to_q" + lora.ADAPTERS] == ("assistant", "subject"), "the order of attaching"
    assert params["transformer_blocks.0.attn.to_k" + lora.ADAPTERS] == ("assistant",)
    assert params["proj_in" + lora.ADAPTERS] == ("subject",)
    assert lora.runtime_key("a.b", "A") == "a.b.lora_A.weight" and lora.runtime_key("a.b", "B", "n") == "a.b.lora_B.n.weight"


def test_file_keys_carry_no_adapter_name(tmp_path):
    """A file has no adapter names: PEFT strips them when it saves (`get_peft_model_state_dict`, `save_lora_adapter`),
    and `Adapter.tensors` and `lora.save` use those names. `runtime_tensors` and `file_tensors` go between the two."""
    model = randomised(TINY, 2)
    assistant, subject = two_adapters()
    stacked = peft_stack(model, tmp_path, assistant, subject)
    for name, adapter in (("assistant", assistant), ("subject", subject)):
        assert set(get_peft_model_state_dict(stacked, adapter_name=name)) == set(adapter.tensors)
        (tmp_path / "theirs").mkdir(exist_ok=True)
        stacked.save_lora_adapter(str(tmp_path / "theirs"), adapter_name=name, weight_name=name + ".safetensors")
        with safe_open(str(tmp_path / "theirs" / (name + ".safetensors")), "pt") as f:
            assert set(f.keys()) == set(adapter.tensors)
        lora.save(tmp_path / "ours.safetensors", adapter, prefix=None)
        with safe_open(str(tmp_path / "ours.safetensors"), "pt") as f:
            assert set(f.keys()) == set(adapter.tensors)
        assert not any(name in k for k in adapter.tensors)
    params = mlx_stack(model, assistant, subject, keep_subject_dtype=True)
    frozen, trainable = lora.split(params, "subject")
    assert set(trainable) == set(lora.runtime_tensors(subject, "subject"))
    back = lora.file_tensors(trainable, "subject")
    assert set(back) == set(subject.tensors) and all(back[k] is subject.tensors[k] for k in back)
    with pytest.raises(ValueError, match="not a matrix of the adapter"):
        lora.file_tensors(trainable, "assistant")


# --- the forward ---------------------------------------------------------------------------------------------------------


def test_stacked_forward_matches_peft_in_every_state(tmp_path):
    """Both adapters active in either order, each alone, and neither: `set_active` against `set_adapters`. The
    assistant off and the subject on is how a trained adapter is sampled."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    assistant, subject = two_adapters()
    stacked = peft_stack(model, tmp_path, assistant, subject)
    params = mlx_stack(model, assistant, subject)
    lay, args = inputs()
    outputs = {}
    for names in STATES:
        want = torch_forward(activate(stacked, names), lay, args)
        active = lora.set_active(params, names)
        assert active["transformer_blocks.0.attn.to_q" + lora.ADAPTERS] == names
        got = mlx_forward(active, cfg, lay, args)
        assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what=f"video velocity, active {names}")
        assert_close(got[1], want[1], atol=2e-4, rtol=2e-4, what=f"audio velocity, active {names}")
        outputs[names] = got[0]
    assert params["transformer_blocks.0.attn.to_q" + lora.ADAPTERS] == ("assistant", "subject"), "set_active copies"
    # guards: the states are different models (the two orders are the same sum in float32), and neither is the base
    plain = mlx_forward(state_mx(model), cfg, lay, args)[0]
    assert rel_err(outputs[()], plain) == 0.0
    distinct = [("assistant", "subject"), ("assistant",), ("subject",), ()]
    for i, a in enumerate(distinct):
        for b in distinct[i + 1:]:
            assert rel_err(outputs[a], outputs[b]) > GUARD, (a, b)
    # each alone is that adapter attached on its own
    for name, adapter in (("assistant", assistant), ("subject", subject)):
        alone = mlx_forward(lora.attach(state_mx(model), cfg, adapter), cfg, lay, args)[0]
        assert_close(outputs[(name,)], alone, atol=1e-5, rtol=1e-5, what=f"{name} alone")


def peft_two(base, first, second, active):
    """One PEFT layer with two adapters, each (name, A, B, alpha, dtype), and `active` as its adapter order."""
    name, a, b, alpha, dtype = first
    layer = PeftLinear(base, name, LoraConfig(r=a.shape[0], lora_alpha=alpha), r=a.shape[0], lora_alpha=alpha)
    name2, a2, b2, alpha2, dtype2 = second
    layer.update_layer(name2, a2.shape[0], lora_alpha=alpha2, config=LoraConfig(r=a2.shape[0], lora_alpha=alpha2))
    with torch.no_grad():
        for n, wa, wb, dt in ((name, a, b, dtype), (name2, a2, b2, dtype2)):
            layer.lora_A[n].to(dt)
            layer.lora_B[n].to(dt)
            layer.lora_A[n].weight.copy_(wa)
            layer.lora_B[n].weight.copy_(wb)
            assert layer.lora_A[n].weight.dtype == dt
    layer.set_adapter(list(active))
    return layer


@pytest.mark.parametrize("base_dtype,first_dtype", [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16)])
def test_stacked_rounding_is_pefts(base_dtype, first_dtype):
    """One Linear, an identity base and selection matrices for both adapters, so every product is exact and what is
    left is PEFT's order of casts and rounded sums, bit for bit, in both adapter orders.

    * bf16 base, a bf16 adapter and a float32 one (a frozen adapter beside a trainable one): which sum is rounded to
      bf16 on the way depends on the order.
    * float32 base and input, a bf16 adapter then a float32 one: PEFT casts the input inside its loop and keeps the
      cast, so the second adapter sees the input rounded to bf16. Casting from the original input instead fails here."""
    n, r = 8, 4
    eye = torch.eye(n)
    first = ("assistant", eye[:r], eye[:, :r], 0.7 * r, first_dtype)
    second = ("subject", eye[2:2 + r], eye[:, :r], 1.3 * r, torch.float32)
    base = torch.nn.Linear(n, n, bias=False).to(base_dtype)
    with torch.no_grad():
        base.weight.copy_(eye)
    x = torch.randn(4000, n).to(base_dtype)
    results = {}
    for active in (("assistant", "subject"), ("subject", "assistant")):
        layer = peft_two(copy.deepcopy(base), first, second, active)
        with torch.no_grad():
            want = layer(x)
        p = {"l.weight": to_mx_half(base.weight), "l" + lora.ADAPTERS: active}
        for name, a, b, alpha, dtype in (first, second):
            p[f"l.lora_A.{name}.weight"] = to_mx_half(a.to(dtype))
            p[f"l.lora_B.{name}.weight"] = to_mx_half(b.to(dtype))
            p[f"l.lora_scaling.{name}"] = mx.array(layer.scaling[name], dtype=mx.float32)
        got = ops.plinear(to_mx_half(x), p, "l")
        assert got.dtype == p["l.weight"].dtype and np.array_equal(to_np(got), to_np(want)), active
        results[active] = to_np(want)
    assert not np.array_equal(*results.values()), "the guard: in PEFT itself the order changes the bits"


@pytest.mark.parametrize("named", [False, True])
def test_modulation_cache_carries_a_lora_on_the_adaln_projections(named):
    """A LoRA on a block's `adaln_proj.linear` or on `norm_out.linear` changes the AdaLN tables, so a modulation cache
    built from those weights must hold the adapted tables: a cached forward is the uncached one bit for bit. When the
    cache drops a projection, the LoRA on it goes too, and LoRAs elsewhere stay."""
    model = randomised(TINY, 2)
    cfg, base = dict(model.config), state_mx(model)
    modules = ["transformer_blocks.0.adaln_proj.linear", "transformer_blocks.1.adaln_proj.linear", "norm_out.linear",
               "transformer_blocks.0.attn.to_q"]
    adapter = random_adapter(cfg, {m: 3 for m in modules}, {m: 6 for m in modules}, seed=4)
    params = lora.attach(base, cfg, adapter, name="subject" if named else None)
    lay, args = inputs()
    m_args, m_lay = [to_mx(a) for a in args], {k: to_mx(v) for k, v in lay.items()}
    uncached = ops.dit_forward(params, cfg, *m_args, **m_lay)

    dropped = dict(params)
    cache = ops.ModulationCache(dropped, cfg, [lay["timestep"].numpy()], drop=True)
    gone = [k for k in params if k not in dropped]
    assert all("adaln_proj.linear." in k or k.startswith("norm_out.linear.") for k in gone)
    assert not any("adaln_proj.linear" in k or k.startswith("norm_out.linear") for k in dropped), "weights, biases and LoRAs"
    assert any(".lora_" in k for k in gone) and any(k.startswith("transformer_blocks.0.attn.to_q.lora_") for k in dropped)
    assert set(ops.ModulationCache.drop_weights(params, cfg)) == set(dropped)
    cached = ops.dit_forward(dropped, cfg, *m_args, **m_lay, cache=cache)
    for got, want in zip(cached, uncached):
        assert np.array_equal(to_np(got), to_np(want))
    # guards: the AdaLN LoRAs count, in the cache as in the forward
    plain_cache = ops.ModulationCache(base, cfg, [lay["timestep"].numpy()])
    without = ops.dit_forward(dropped, cfg, *m_args, **m_lay, cache=plain_cache)
    assert rel_err(without[0], uncached[0]) > GUARD


# --- which weights a gradient reaches ------------------------------------------------------------------------------------


def test_gradients_reach_the_trainable_factors_only(tmp_path):
    """The frozen adapter and the base are constants: `split` hands a trainer the trainable adapter's two matrices per
    Linear and nothing else, and the gradient with respect to them is torch's, where the assistant's and the base's
    parameters are frozen and end with no gradient. The trainable gradient still runs through the assistant's deltas
    in the layers downstream: cutting the gradient there would change these numbers."""
    model = randomised(TINY, 2)
    cfg = dict(model.config)
    assistant, subject = two_adapters()
    stacked = activate(peft_stack(model, tmp_path, assistant, subject), ("assistant", "subject"))
    for name, prm in stacked.named_parameters():
        prm.requires_grad_(".lora_" in name and ".subject." in name)
    lay, args = inputs()
    video, audio = stacked(*args, **lay, return_dict=False)
    ((video ** 2).mean() + (audio ** 2).mean()).backward()
    theirs = {n: p.grad for n, p in stacked.named_parameters()}
    assert all(g is None for n, g in theirs.items() if ".subject." not in n), "torch: nothing for the assistant or the base"

    params = mlx_stack(model, assistant, subject, keep_subject_dtype=True)
    frozen, trainable = lora.split(params, "subject")
    assert set(trainable) == {k for k in theirs if ".subject." in k}, "the subject's matrices, PEFT's names"
    assert not any(".subject." in k and k.endswith(".weight") for k in frozen) and len(frozen) + len(trainable) == len(params)
    assert all(k in frozen for k in params if ".assistant." in k) and "transformer_blocks.0.attn.to_q.weight" in frozen
    m_args = [to_mx(a) for a in args]
    m_lay = {k: to_mx(v) for k, v in lay.items()}

    def objective(constants):
        def of(weights):  # the two groups meet inside the function that is differentiated
            v, a = ops.dit_forward({**constants, **weights}, cfg, *m_args, **m_lay)
            return (v ** 2).mean() + (a ** 2).mean()
        return of

    value, grads = mx.value_and_grad(objective(frozen))(trainable)
    assert set(grads) == set(trainable), "a gradient for each trainable matrix and for nothing else"
    worst = 0.0
    for k, g in grads.items():
        assert float(mx.abs(g).max()) > 0, k
        worst = max(worst, rel_err(g, theirs[k]))
    print(f"worst relative error of a gradient tensor against torch: {worst:.2e}")
    assert worst < 1e-4
    # the guard: without the assistant the same gradients are other numbers, so its deltas are on the path
    alone, alone_trainable = lora.split(lora.set_active(params, ("subject",)), "subject")
    _, without = mx.value_and_grad(objective(alone))(alone_trainable)
    assert max(rel_err(without[k], grads[k]) for k in grads) > GUARD


# --- what is refused -----------------------------------------------------------------------------------------------------


def test_stacking_refusals():
    model = randomised(TINY, 0)
    cfg, base = dict(model.config), state_mx(model)
    assistant, subject = two_adapters()
    named = lora.attach(base, cfg, assistant, name="assistant")
    assert "transformer_blocks.0.attn.to_q" + lora.ADAPTERS not in base, "attach does not modify its input"
    with pytest.raises(ValueError, match="name 'assistant' is already in use"):  # the same name twice
        lora.attach(named, cfg, subject, name="assistant")
    elsewhere = lora.Adapter({k: v for k, v in subject.tensors.items() if k.startswith("proj_in.")}, {"proj_in": 6})
    assert "proj_in" not in assistant.modules  # diffusers refuses a name in use whichever modules the second one has
    with pytest.raises(ValueError, match="name 'assistant' is already in use"):
        lora.attach(named, cfg, elsewhere, name="assistant")
    with pytest.raises(ValueError, match="already carry a LoRA"):  # an unnamed adapter shares its Linears with none
        lora.attach(named, cfg, subject)
    with pytest.raises(ValueError, match="unnamed LoRA"):  # nor a named one with an unnamed one
        lora.attach(lora.attach(base, cfg, assistant), cfg, subject, name="subject")
    hf_tensors, hf_md = make_file(model, TINY)
    hf_params, hf_cfg = hyperflow.attach(base, cfg, {k: to_mx(v) for k, v in hf_tensors.items()},
                                         hyperflow.HyperFlowMeta.from_header(hf_md))
    with pytest.raises(ValueError, match="unnamed LoRA"):  # HyperFlow's adapter is an unnamed one
        lora.attach(hf_params, hf_cfg, subject, name="subject")
    for bad in ("", "a.b", "weight.x"):
        with pytest.raises(ValueError, match="without dots"):
            lora.attach(base, cfg, subject, name=bad)
    with pytest.raises(ValueError, match="already carry a LoRA"):  # merging is for plain weights
        lora.merge(named, cfg, subject)
    both = lora.attach(named, cfg, subject, name="subject")
    for bad in (("nobody",), ("assistant", "nobody"), ("subject", "subject")):
        with pytest.raises(ValueError, match="not attached, or are named twice"):
            lora.set_active(both, bad)
    with pytest.raises(ValueError, match="no adapter named 'nobody'"):
        lora.split(both, "nobody")
    with pytest.raises(ValueError, match="no adapter named"):  # an unnamed adapter has no group to split off
        lora.split(lora.attach(base, cfg, assistant), "assistant")
    # an adapter switched off stays in the weights, and comes back
    off = lora.set_active(both, ("subject",))
    assert lora.adapter_names(off) == ["assistant", "subject"]
    assert lora.set_active(off, ("assistant", "subject"))["proj_in" + lora.ADAPTERS] == ("subject",)
    assert off["transformer_blocks.0.attn.to_k" + lora.ADAPTERS] == ()


def test_the_frozen_half_of_a_split_is_not_a_model():
    """`split` leaves the trainable adapter's scalings and its place in each Linear's list in the frozen half, so that
    the two halves merge back into the model. On its own that half is refused: by the forward, and by `attach` under
    the same name, which would otherwise list the adapter twice on each Linear and apply it twice. `join` is the way
    back, and a changed adapter goes in as the trainable half."""
    model = randomised(TINY, 0)
    cfg = dict(model.config)
    assistant, subject = two_adapters()
    both = mlx_stack(model, assistant, subject, keep_subject_dtype=True)
    frozen, trainable = lora.split(both, "subject")
    lay, args = inputs()
    with pytest.raises(ValueError, match="'subject' is listed as active but its matrices are not"):
        mlx_forward(frozen, cfg, lay, args)
    with pytest.raises(ValueError, match="name 'subject' is already in use"):
        lora.attach(frozen, cfg, subject, keep_dtype=True, name="subject")
    # each trace the name leaves is enough: a place in a list alone, and a scaling alone
    listed = {k: v for k, v in frozen.items() if not k.endswith(lora.SCALING + ".subject")}
    scaled = {k: (tuple(n for n in v if n != "subject") if k.endswith(lora.ADAPTERS) else v) for k, v in frozen.items()}
    assert len(listed) < len(frozen) and scaled != frozen
    for half in (listed, scaled):
        with pytest.raises(ValueError, match="name 'subject' is already in use"):
            lora.attach(half, cfg, subject, keep_dtype=True, name="subject")
    lora.attach(frozen, cfg, subject, keep_dtype=True, name="another")  # a name that is free is still fine

    joined = lora.join(frozen, trainable)
    assert joined.keys() == both.keys() and all(joined[k] is both[k] for k in both)
    want = mlx_forward(both, cfg, lay, args)
    for got, ref in zip(mlx_forward(joined, cfg, lay, args), want):
        assert np.array_equal(to_np(got), to_np(ref))
    # an updated adapter: the same keys with new values, joined, is applied once
    doubled = {k: v * 2 if ".lora_B." in k else v for k, v in trainable.items()}
    twice_as_strong = lora.Adapter({k: v * 2 if ".lora_B." in k else v for k, v in subject.tensors.items()}, subject.alpha)
    ref = mlx_forward(mlx_stack(model, assistant, twice_as_strong, keep_subject_dtype=True), cfg, lay, args)
    for got, r, before in zip(mlx_forward(lora.join(frozen, doubled), cfg, lay, args), ref, want):
        assert np.array_equal(to_np(got), to_np(r)) and rel_err(got, before) > GUARD


def test_check_trainable():
    """What a loss checks about its two groups (`lora.check_trainable`, and `join` through it). Each refused case is
    one the forward would run without reading the matrices, so their gradients would be exactly zero."""
    model = randomised(TINY, 0)
    cfg, base = dict(model.config), state_mx(model)
    assistant, subject = two_adapters()
    with_assistant = lora.attach(base, cfg, assistant, name="assistant")
    both = lora.attach(with_assistant, cfg, subject, keep_dtype=True, name="subject")
    frozen, trainable = lora.split(both, "subject")
    lora.check_trainable(trainable, frozen)
    with pytest.raises(ValueError, match="`trainable` is empty"):
        lora.check_trainable({}, frozen)
    with pytest.raises(ValueError, match="a file's keys are not read by the forward"):  # an Adapter's own dict
        lora.check_trainable(subject.tensors, with_assistant)
    with pytest.raises(ValueError, match="'subject' is not active on"):  # runtime names, but never attached
        lora.check_trainable(lora.runtime_tensors(subject, "subject"), with_assistant)
    with pytest.raises(ValueError, match="'subject' is not active on"):  # attached, then switched off
        lora.check_trainable(*reversed(lora.split(lora.set_active(both, ("assistant",)), "subject")))
    with pytest.raises(ValueError, match="'subject' is not active on"):  # and with no adapter in the weights at all
        lora.check_trainable(trainable, base)
    one = next(k for k in trainable if ".lora_B." in k)
    with pytest.raises(ValueError, match="one of lora_A / lora_B without the other"):
        lora.check_trainable({k: v for k, v in trainable.items() if k != one}, frozen)
    with pytest.raises(ValueError, match="is not an attached adapter's matrix"):
        lora.check_trainable({**trainable, "proj_in.weight.copy": base["proj_in.weight"]}, frozen)
    with pytest.raises(ValueError, match="'subject' is not active on"):
        lora.join(with_assistant, lora.runtime_tensors(subject, "subject"))


def test_the_two_groups_share_nothing_and_make_a_whole_model():
    """Two more things `check_trainable` (and so `join` and every loss) refuses. A key in both groups: merged, the
    trainable one wins, so the frozen assistant's matrices handed in beside the trained ones would be trained too.
    And a trainable group that holds only some of the split adapter's modules: the rest are still listed in the
    frozen half without their matrices, which the forward would only find out part of the way through."""
    model = randomised(TINY, 0)
    cfg = dict(model.config)
    assistant, subject = two_adapters()
    both = mlx_stack(model, assistant, subject, keep_subject_dtype=True)
    frozen, trainable = lora.split(both, "subject")
    assistants = {k: v for k, v in frozen.items() if ".assistant." in k and k.endswith(".weight")}
    for function in (lora.check_trainable, lora.join):
        with pytest.raises(ValueError, match=r"36 key\(s\) are in both groups"):
            function(*((frozen, {**trainable, **assistants}) if function is lora.join else ({**trainable, **assistants}, frozen)))
        with pytest.raises(ValueError, match="are in both groups"):  # the whole model as the frozen half
            function(*((both, trainable) if function is lora.join else (trainable, both)))
    one_module = "transformer_blocks.0.attn.to_q"
    partial = {k: v for k, v in trainable.items() if not k.startswith(one_module + ".")}
    assert len(partial) == len(trainable) - 2
    with pytest.raises(ValueError, match=f"'subject' is listed on 1 Linear.s. whose matrices are in neither group, e.g. '{one_module}'"):
        lora.check_trainable(partial, frozen)
    with pytest.raises(ValueError, match="whose matrices are in neither group"):
        lora.join(frozen, partial)
    # the rest of the adapter moved to the frozen side is a whole model again: train some modules, freeze the others
    moved = {k: v for k, v in trainable.items() if k.startswith(one_module + ".")}
    lora.check_trainable(partial, {**frozen, **moved})
    lay, args = inputs()
    want = mlx_forward(both, cfg, lay, args)
    for got, ref in zip(mlx_forward(lora.join({**frozen, **moved}, partial), cfg, lay, args), want):
        assert np.array_equal(to_np(got), to_np(ref))


def test_check_config():
    """`lora.check_config`: the weights have the config's blocks, and each trainable matrix sits on one of the
    config's Linears in that Linear's shape."""
    model = randomised(TINY, 0)
    cfg = dict(model.config)
    assistant, subject = two_adapters()
    frozen, trainable = lora.split(mlx_stack(model, assistant, subject, keep_subject_dtype=True), "subject")
    lora.check_config(trainable, frozen, cfg)
    for change, message in ((dict(num_layers=1), "the config has 1 transformer_blocks"), (dict(num_layers=3), "the config has 3 transformer_blocks"),
                            (dict(num_refiner_layers=0), "the config has 0 token_refiner.refiner_blocks"),
                            (dict(num_refiner_layers=2), "the config has 2 token_refiner.refiner_blocks")):
        with pytest.raises(ValueError, match=message):
            lora.check_config(trainable, frozen, dict(cfg, **change))
    key_a, key_b = "transformer_blocks.0.attn.to_q.lora_A.subject.weight", "transformer_blocks.0.attn.to_q.lora_B.subject.weight"
    elsewhere = {"transformer_blocks.0.attn.nowhere.lora_A.subject.weight": trainable[key_a]}
    with pytest.raises(ValueError, match="is not on a Linear of the transformer this config describes"):
        lora.check_config({**trainable, **elsewhere}, frozen, cfg)
    for bad, message in (({key_a: trainable[key_a][:, :-1]}, "does not fit a Linear of"), ({key_b: trainable[key_b][:-1]}, "does not fit a Linear of"),
                         ({key_a: trainable[key_a][:-1]}, "have different ranks"), ({key_a: trainable[key_a][0]}, "does not fit a Linear of")):
        with pytest.raises(ValueError, match=message):
            lora.check_config({**trainable, **bad}, frozen, cfg)
    wider = dict(cfg, hidden_size=64)  # the right names, another model's widths
    with pytest.raises(ValueError, match="does not fit a Linear of"):
        lora.check_config(trainable, frozen, wider)


# --- the public training adapter, by its header ----------------------------------------------------------------------------


def test_training_adapter_header_fits_p_h3_image():
    """The safetensors header of the public P-H3-Image training adapter (keys, shapes and dtypes; a release fixture,
    no tensor data) against the released transformer's config.

    What it shows: the file is in the original checkpoint's key layout, its names convert, every converted module is a
    Linear of the transformer, and each pair has the shape that Linear needs. What it cannot show: anything about the
    values. Whether the adapter fits these weights, or does what its name says, is not tested by a header."""
    header = json.loads(fixture("p_h3_image_training_adapter_v1.header.json").read_text())
    metadata = header.pop("__metadata__")
    assert len(header) == 416 and {v["dtype"] for v in header.values()} == {"BF16"}
    assert all(k.startswith("diffusion_model.") for k in header) and lora.is_original_layout(header)
    assert "alpha" not in metadata and lora.METADATA_KEY not in metadata and not any(k.endswith(".alpha") for k in header)
    cfg = json.loads(fixture("P-H3-Image", "transformer", "config.json").read_text())
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if not k.startswith("_")}

    # shapes only: these zeros are lazy and are never evaluated
    adapter = lora.read({k: mx.zeros(tuple(v["shape"]), dtype=mx.bfloat16) for k, v in header.items()}, metadata)
    lora.check(adapter, cfg)
    assert adapter.modules == sorted(lora.block_modules(cfg)) and len(adapter.modules) == 6 * (50 + 2)
    assert all(adapter.rank(m) == 16 and adapter.scaling(m) == 1.0 for m in adapter.modules), "no alpha anywhere: scaling 1"
    assert all(v.dtype == mx.bfloat16 for v in adapter.tensors.values())
    assert not any("adaln_proj" in m or "time_embedder" in m for m in adapter.modules)
    # diffusers' converter gives the same keys and shapes from the same header
    theirs = _convert_non_diffusers_minimax_h3_lora_to_diffusers(
        {k: torch.zeros(tuple(v["shape"]), dtype=torch.bfloat16) for k, v in header.items()})
    assert {k: tuple(v.shape) for k, v in theirs.items()} == {"transformer." + k: tuple(v.shape) for k, v in adapter.tensors.items()}
    # and it stacks under a name beside a trainable adapter on the same Linears (shapes again; no weights exist here)
    shapes = lora.wired_shapes(cfg)
    base = {m + ".weight": mx.zeros(shapes[m], dtype=mx.bfloat16) for m in shapes}
    stacked = lora.attach(lora.attach(base, cfg, adapter, name="assistant"), cfg, lora.init(cfg, rank=4), keep_dtype=True, name="subject")
    assert stacked["transformer_blocks.49.ff.net.2" + lora.ADAPTERS] == ("assistant", "subject")
