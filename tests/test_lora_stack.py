"""Two named adapters on the same Linears, a frozen one and a trainable one, against PEFT.

The oracle is the tiny diffusers transformer with both adapters loaded under names (`load_lora_adapter(adapter_name=)`)
and switched with `set_adapters`, PEFT's own layer for the rounding of a stacked forward, and torch autograd for which
weights a gradient reaches.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, state_mx, to_mx, to_np
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
    with pytest.raises(ValueError, match="named 'assistant'"):  # the same name twice
        lora.attach(named, cfg, subject, name="assistant")
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
