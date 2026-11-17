"""The backward pass of the two fast ops the differentiated forward calls, on half-precision inputs.

`ops.attention` (the sampling forward's) and `ops.attention_for_training` against torch's
`scaled_dot_product_attention`, at ordinary and at large attention scores; and `ops.rms_norm` against torch's
`rms_norm` at extreme inputs. Every input is rounded to its half-precision dtype before anything sees it, so each
side gets the same numbers and the reference is torch in float32 on them. `bench/attention_bf16_backward.py` prints
the full tables, and a float64 model of what MLX's backward does.

The size that matters is the unscaled score, q . k before the scale: MLX 0.32.4's fused backward holds it in the
inputs' dtype (`ops.attention_for_training`). At a head width of 128 it is 11.3 times the logit. Several tests here
pin what MLX 0.32.4 does on the default device; the port asks for that version or a newer one.
"""

import inspect
import math

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from h3mlx import ops

ROWS = 160
ON_GPU = mx.default_device() == mx.gpu
NEEDS_GPU = pytest.mark.skipif(not ON_GPU, reason="the fused attention kernel is the default (GPU) device's")
DEVICES = pytest.mark.parametrize("device", [mx.default_device(), mx.cpu], ids=["default device", "cpu stream"])


def to_bf16(a, dtype=mx.bfloat16):
    return np.array(mx.array(a).astype(dtype).astype(mx.float32))


def inputs(std, width=128, heads=2, rows=ROWS, seed=0, dtype=mx.bfloat16):
    """q, k, v and a cotangent, (1, rows, heads, width) as `ops.attention` takes them, float32 arrays of values
    `dtype` holds. q and k are standard normal times `std`, and one key in each head is copied onto a query, so that
    pair's score, the largest, is |k|^2: about 128 * std^2 at a width of 128, and its logit about 11 * std^2."""
    rng = np.random.default_rng(seed)
    q, k, v, ct = (rng.standard_normal((1, rows, heads, width)).astype(np.float32) for _ in range(4))
    q, k = q * np.float32(std), k * np.float32(std)
    for h in range(heads):
        q[0, 7 + h, h] = k[0, 100 + h, h]
    return tuple(to_bf16(t, dtype) for t in (q, k, v, ct))


def largest_score(q, k):
    return float(np.abs(np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64))).max())


def torch_gradients(q, k, v, ct, dtype=torch.float32):
    q, k, v = (torch.from_numpy(t).to(dtype).transpose(1, 2).requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q, k, v).backward(torch.from_numpy(ct).to(dtype).transpose(1, 2))
    return [t.grad.transpose(1, 2).float().numpy() for t in (q, k, v)]


def gradients(attend, q, k, v, ct, dtype=mx.bfloat16, device=None):
    with mx.stream(device or mx.default_device()):
        (out,), grads = mx.vjp(attend, [mx.array(t).astype(dtype) for t in (q, k, v)], [mx.array(ct).astype(dtype)])
        mx.eval(out, grads)
    assert out.dtype == dtype and all(g.dtype == dtype for g in grads)
    return [np.array(g.astype(mx.float32)) for g in grads]


def error(got, want):
    return float(np.linalg.norm(got.astype(np.float64) - want.astype(np.float64)) / np.linalg.norm(want.astype(np.float64)))


def norm(a):
    return float(np.linalg.norm(a.astype(np.float64)))


# --- attention -----------------------------------------------------------------------------------------------------------


@DEVICES
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("std", [1, 10, 30])
def test_training_attention_backward_is_torchs_at_every_logit_scale(std, dtype, device):
    """`ops.attention_for_training` on half-precision q, k and v, at largest scores of about 140, 13,000 and 119,000
    (logits of 12, 1,200 and 10,500): each of the three gradients is torch's float32 gradient on the same numbers,
    within the rounding of a half-precision result (measured: 0.0017 in bf16 and 0.0002 in float16, at every scale;
    torch's own bf16 backward is within 0.011 of it too)."""
    q, k, v, ct = inputs(std, dtype=dtype)
    want = torch_gradients(q, k, v, ct)
    got = gradients(ops.attention_for_training, q, k, v, ct, dtype, device)
    score = largest_score(q, k)
    print(f"{dtype}, largest score {score:,.0f} (logit {score / math.sqrt(q.shape[-1]):,.0f}): ours "
          + ", ".join(f"{error(g, w):.4f}" for g, w in zip(got, want)))
    for g, w in zip(got, want):
        assert error(g, w) < 5e-3
    if dtype == mx.bfloat16:
        for t, w in zip(torch_gradients(q, k, v, ct, torch.bfloat16), want):
            assert error(t, w) < 2e-2


@NEEDS_GPU
def test_the_sampling_attentions_backward_in_bf16_goes_wrong_as_logits_grow():
    """What `attention_for_training` is for, on MLX 0.32.4. `ops.attention` is the fast call on the bf16 inputs
    themselves. At small scores its backward is fine. At a largest score near 119,000 (a logit of 10,500; a bf16
    score is off by up to 22.6 in the logit there), on the default device and at the release's head width, each
    gradient is more than a hundred times the size of torch's (measured here, at 160 rows: 330 times for q and k,
    4,500 for v; the bench's 605 rows give a million and more). On the CPU stream, where MLX differentiates the op
    written out, it stays the right size and is wrong for q and k all the same (measured 1.9 and 2.1, relative).
    None of that shows in the forward, which `test_training_attention_forward_is_the_sampling_attentions` pins."""
    q, k, v, ct = inputs(1)
    for g, w in zip(gradients(ops.attention, q, k, v, ct), torch_gradients(q, k, v, ct)):
        assert error(g, w) < 2e-2
    q, k, v, ct = inputs(30)
    want = torch_gradients(q, k, v, ct)
    assert 65_536 < largest_score(q, k) < 131_072  # a bf16 step of 512 there: up to 22.6 in the logit
    on_gpu, on_cpu = gradients(ops.attention, q, k, v, ct), gradients(ops.attention, q, k, v, ct, device=mx.cpu)
    print("bf16 call, default device: " + ", ".join(f"{norm(g) / norm(w):.3g}" for g, w in zip(on_gpu, want))
          + " times torch's norms; cpu stream, relative error " + ", ".join(f"{error(g, w):.3g}" for g, w in zip(on_cpu, want)))
    assert all(norm(g) > 100 * norm(w) for g, w in zip(on_gpu, want))
    assert all(norm(g) < 10 * norm(w) for g, w in zip(on_cpu, want)) and error(on_cpu[0], want[0]) > 0.5


def test_the_sampling_forward_keeps_the_half_precision_call():
    """Sampling has no backward pass and does not pay for the float32 copies: every function on the way to the
    attention takes `ops.attention` unless it is told otherwise, and the sampling forward runs with
    `attention_for_training` made to raise."""
    for function in (ops.attn_block, ops.dit_block, ops.token_refiner, ops.dit_inputs):
        assert inspect.signature(function).parameters["attend"].default is ops.attention, function.__name__
    import test_train_gradients as G
    from test_dit_mixed_precision import to_mixed

    s = G.setup(1, 0.5)
    frozen, trainable = G.mlx_groups(s, base=to_mixed(s["model"]))

    def refuse(*a, **k):
        raise AssertionError("the sampling forward called the training attention")

    with G.swapped("attention_for_training", refuse):
        loss = G.sampling_forward_loss({**frozen, **trainable}, s["cfg"], s["example"], mx.array(s["text"].numpy()))
        mx.eval(loss)
        with pytest.raises(AssertionError, match="training attention"):
            mx.eval(G.train.example_loss(trainable, frozen, s["cfg"], s["example"], mx.array(s["text"].numpy())))
    assert np.isfinite(float(loss))


@NEEDS_GPU
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("std", [1, 30])
def test_training_attention_forward_is_the_sampling_attentions(width, std):
    """On the default device, at the head widths MLX 0.32.4's fused kernels take (the release's is 128), the bf16
    call already computes in float32 and rounds: `attention_for_training` returns the same bits, at small scores and
    at large ones. So the training forward's values are the sampling forward's there. (On 0.32.3 and 0.31.2, which
    the port does not support, the bf16 call computes in bf16 at these widths too.)"""
    q, k, v, _ = (mx.array(t).astype(mx.bfloat16) for t in inputs(std, width))
    plain, training = ops.attention(q, k, v), ops.attention_for_training(q, k, v)
    assert plain.dtype == training.dtype == mx.bfloat16
    assert np.array_equal(np.array(plain.astype(mx.float32)), np.array(training.astype(mx.float32)))


@DEVICES
def test_where_the_bf16_call_computes_in_bf16_the_training_attention_is_close_not_equal(device):
    """At a head width the fused kernels do not take (32, the tiny test models'), and anywhere on the CPU stream,
    MLX computes the bf16 call in bf16. The float32 call rounded to bf16 is then the more exact of the two, not the
    same bits: within 2% of it, as a whole, at small scores."""
    q, k, v, _ = inputs(1, width=32)
    with mx.stream(device):
        args = [mx.array(t).astype(mx.bfloat16) for t in (q, k, v)]
        plain = np.array(ops.attention(*args).astype(mx.float32))
        training = np.array(ops.attention_for_training(*args).astype(mx.float32))
        exact = np.array(ops.attention(*(a.astype(mx.float32) for a in args)))
    print(f"head width 32, {device}: {np.mean(plain == training):.1%} of entries equal; against float32, the bf16 "
          f"call {error(plain, exact):.4f}, the training attention {error(training, exact):.4f}")
    assert error(training, plain) < 2e-2
    assert error(training, exact) <= error(plain, exact)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_training_attention_dtypes(dtype):
    """float32 inputs are not touched: the same call, the same bits forward and backward. Half-precision inputs come
    back in their own dtype, and so do their gradients."""
    q, k, v, ct = inputs(1, width=32)
    args = [mx.array(t).astype(dtype) for t in (q, k, v)]
    out = ops.attention_for_training(*args)
    assert out.dtype == dtype and out.shape == args[0].shape
    got = gradients(ops.attention_for_training, q, k, v, ct, dtype)
    if dtype == mx.float32:
        assert np.array_equal(np.array(out), np.array(ops.attention(*args)))
        for g, w in zip(got, gradients(ops.attention, q, k, v, ct, dtype)):
            assert np.array_equal(g, w)
    for g, w in zip(got, torch_gradients(q, k, v, ct)):
        assert error(g, w) < (1e-5 if dtype == mx.float32 else 5e-3)


def test_training_attention_takes_a_mask():
    """The model passes none, but the argument is `attention`'s: a boolean mask over the keys, in bf16, against torch."""
    q, k, v, ct = inputs(5, width=32)
    mask = ops.key_padding_mask(100, ROWS)
    (out,), grads = mx.vjp(lambda q, k, v: ops.attention_for_training(q, k, v, mask),
                           [mx.array(t).astype(mx.bfloat16) for t in (q, k, v)], [mx.array(ct).astype(mx.bfloat16)])
    tq, tk, tv = (torch.from_numpy(t).transpose(1, 2).requires_grad_(True) for t in (q, k, v))
    keep = torch.arange(ROWS) < 100
    want = F.scaled_dot_product_attention(tq, tk, tv, attn_mask=keep[None, None, None, :])
    want.backward(torch.from_numpy(ct).transpose(1, 2))
    assert error(np.array(out.astype(mx.float32)), want.detach().transpose(1, 2).numpy()) < 5e-3
    for g, t in zip(grads, (tq, tk, tv)):
        assert error(np.array(g.astype(mx.float32)), t.grad.transpose(1, 2).numpy()) < 5e-3
    assert not np.array(grads[1].astype(mx.float32))[0, 100:].any(), "no gradient for a key nobody may see"


# --- the other fast op: the RMS norm ---------------------------------------------------------------------------------------


@DEVICES
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("what,scale,gain", [("unit rows", 1.0, 1.0), ("rows of 1e-3", 1e-3, 1.0), ("rows of 1e-6", 1e-6, 1.0),
                                              ("one row exactly zero", 1.0, 1.0), ("rows of 1e6", 1e6, 1.0),
                                              ("affine weights of 40", 1.0, 40.0), ("rows of 1e-4, weights of 40", 1e-4, 40.0),
                                              ("rows from 1e-6 to 1e4", None, 5.0)])
def test_rms_norm_backward_in_bf16_is_torchs(what, scale, gain, eps, device):
    """`ops.rms_norm` is `mx.fast.rms_norm`, on bf16 rows in the blocks. Its backward was tested where a wrong one
    would hurt: rows far below the epsilon (where the derivative is largest, `weight / sqrt(eps)`), a row of
    zeros, very large rows, large affine weights, and rows of every size at once. In each case the gradients for
    the input and for the weight are torch's float32 gradients within bf16's rounding (under 0.005; torch's own bf16
    backward is at 0.003). Nothing to fix here."""
    rng = np.random.default_rng(0)
    shape = (1, 300, 2, 128)
    x = rng.standard_normal(shape).astype(np.float32)
    x *= (10.0 ** rng.uniform(-6, 4, shape[:-1] + (1,))).astype(np.float32) if scale is None else np.float32(scale)
    if what == "one row exactly zero":
        x[0, 3] = 0
    x, w, ct = to_bf16(x), to_bf16((rng.standard_normal(shape[-1]) * gain).astype(np.float32)), to_bf16(rng.standard_normal(shape).astype(np.float32))
    tx, tw = torch.from_numpy(x).requires_grad_(True), torch.from_numpy(w).requires_grad_(True)
    F.rms_norm(tx, (shape[-1],), tw, eps).backward(torch.from_numpy(ct))
    with mx.stream(device):
        _, (gx, gw) = mx.vjp(lambda x, w: ops.rms_norm(x, w, eps), [mx.array(x).astype(mx.bfloat16), mx.array(w).astype(mx.bfloat16)],
                             [mx.array(ct).astype(mx.bfloat16)])
        mx.eval(gx, gw)
    assert gx.dtype == gw.dtype == mx.bfloat16
    e_x, e_w = error(np.array(gx.astype(mx.float32)), tx.grad.numpy()), error(np.array(gw.astype(mx.float32)), tw.grad.numpy())
    assert e_x < 5e-3 and e_w < 6e-3, (e_x, e_w)
