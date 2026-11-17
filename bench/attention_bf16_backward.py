"""Measurement behind docs/porting-lora-training.md's section on the attention's backward pass: the gradient of
`mx.fast.scaled_dot_product_attention` on bf16 inputs as the attention scores grow, against the same function on
float32 inputs and against torch's `scaled_dot_product_attention` in float32 and in bf16. Not a test: the tests are
tests/test_attention_backward.py. What is wrong is MLX's fused backward for this op, new in 0.32.4, which keeps the
unscaled scores q.k in the inputs' dtype before the exponential; so the size that matters is the largest unscaled
score, and the tables give it beside the logit (the score times the scale).

Seven tables.

1. The op, two heads of 605 rows by 128 (the release's head width), one key in each head aligned with a query so
   that the largest score is known, q and k drawn at a growing standard deviation. Every input is rounded to bf16
   first, so each variant gets the same numbers and only the arithmetic differs. Norms of the three gradients and
   their relative error against torch in float32, for: torch in bf16; MLX's fast call in float32 and in bf16 on the
   default device and on the CPU stream; the attention written out by hand (scores, softmax, product) in bf16; and
   `ops.attention_for_training`, which runs the fast call on bf16 inputs in float32. One more row is not a variant
   of the arithmetic: torch in float32 on the inputs before they were rounded, which shows how far rounding q and
   k to bf16 moves the gradient on its own.
2. The upstream backward's arithmetic, modelled in float64: exact scores from the bf16 inputs, an exact
   log-sum-exp, the scores rounded to bf16, then `exp(score * scale - lse)`. The largest score, a bf16 step there,
   the logit error that step allows, the largest row sum of the probabilities so rebuilt (it should be 1), and the
   model's three gradients beside MLX's own bf16 ones.
3. The attention written out by hand in bf16 with four spellings of the softmax, at the two largest scales.
4. Which head widths take the fused path: for each width, the share of the bf16 forward's entries that are the
   float32 forward's rounded to bf16, and the bf16 gradient's norm against float32's. Then the message MLX gives
   at a width of 192, where the float32 gradient's kernel does not load: it names the kernel.
5. `mx.fast.rms_norm`, the other fast op in the differentiated forward, at extreme inputs: its bf16 backward
   against torch's in float32 and in bf16.
6. What `ops.attention_for_training` costs at a real size, 56 heads of 600 rows by 128: time of one forward and
   backward, and MLX's peak memory over it, against the bf16 call. Each of the two is measured in a process of
   its own, five times.
7. The same inside a step: the block-by-block gradient of a transformer with the release's widths (hidden 5,376,
   56 heads of 128, feed-forward 14,336) and two blocks of random weights in the release's mixed precision, a
   rank-16 float32 adapter on the six block matrices, one 768 by 768 text-to-image example. Seconds for the
   gradient and MLX's peak memory over it, with `ops.attention_for_training` and with the bf16 call in its place,
   each in a process of its own.

Run from the repo root: python bench/attention_bf16_backward.py [--skip-step]"""
import json
import math
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from h3mlx import ops

HEADS, ROWS, WIDTH = 2, 605, 128


def to_bf16(a):
    return np.array(mx.array(a).astype(mx.bfloat16).astype(mx.float32))


def inputs(std, heads=HEADS, rows=ROWS, width=WIDTH, seed=0, rounded=True):
    """q, k, v and a cotangent as float32 numpy arrays, (1, heads, rows, width): standard normal, q and k times
    `std`, with one key per head copied onto a query, so that pair's logit is |k|^2 / sqrt(width). `rounded`: every
    value is one bf16 holds."""
    rng = np.random.default_rng(seed)
    q, k, v, ct = (rng.standard_normal((1, heads, rows, width)).astype(np.float32) for _ in range(4))
    q, k = q * np.float32(std), k * np.float32(std)
    for h in range(heads):
        q[0, h, (7 + h) % rows] = k[0, h, (300 + h) % rows]
    return tuple(to_bf16(t) for t in (q, k, v, ct)) if rounded else (q, k, v, ct)


def error(got, want):
    return float(np.linalg.norm(got.astype(np.float64) - want.astype(np.float64)) / np.linalg.norm(want.astype(np.float64)))


def norm(a):
    return float(np.linalg.norm(a.astype(np.float64)))


def torch_gradients(q, k, v, ct, dtype):
    q, k, v = (torch.from_numpy(t).to(dtype).requires_grad_(True) for t in (q, k, v))
    F.scaled_dot_product_attention(q, k, v).backward(torch.from_numpy(ct).to(dtype))
    return [t.grad.float().numpy() for t in (q, k, v)]


def fast(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(q.shape[-1]))


def by_hand(q, k, v):
    scores = (q * (1 / math.sqrt(q.shape[-1]))) @ k.transpose(0, 1, 3, 2)
    return mx.softmax(scores, axis=-1, precise=True) @ v


def for_training(q, k, v):
    """`ops.attention_for_training` on (batch, heads, rows, width) inputs (it takes rows before heads)."""
    return ops.attention_for_training(*(t.transpose(0, 2, 1, 3) for t in (q, k, v))).transpose(0, 2, 1, 3)


def mlx_gradients(f, q, k, v, ct, dtype, device=None):
    with mx.stream(device or mx.default_device()):
        _, grads = mx.vjp(f, [mx.array(t).astype(dtype) for t in (q, k, v)], [mx.array(ct).astype(dtype)])
        mx.eval(grads)
    return [np.array(g.astype(mx.float32)) for g in grads]


def largest_score(q, k):
    return float(np.abs(np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), k.astype(np.float64))).max())


def bf16_step(value):
    """The spacing of bf16 numbers at `value`: eight significant bits."""
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def op_table():
    print("| q, k std | largest score | largest logit | which | dQ | dK | dV | error of dQ | of dK | of dV |")
    print("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for std in (1, 5, 10, 15, 20, 30, 40):
        q, k, v, ct = inputs(std)
        score = largest_score(q, k)
        logit = score / math.sqrt(WIDTH)
        reference = torch_gradients(q, k, v, ct, torch.float32)
        rows = [("torch, float32", reference), ("torch, bf16", torch_gradients(q, k, v, ct, torch.bfloat16)),
                ("fast, float32", mlx_gradients(fast, q, k, v, ct, mx.float32)),
                ("fast, bf16", mlx_gradients(fast, q, k, v, ct, mx.bfloat16)),
                ("fast, bf16, CPU stream", mlx_gradients(fast, q, k, v, ct, mx.bfloat16, mx.cpu)),
                ("by hand, bf16", mlx_gradients(by_hand, q, k, v, ct, mx.bfloat16)),
                ("for training, bf16", mlx_gradients(for_training, q, k, v, ct, mx.bfloat16)),
                ("torch, float32, inputs not rounded", torch_gradients(*inputs(std, rounded=False), torch.float32))]
        for name, grads in rows:
            errors = ["" if grads is reference else f"{error(g, r):.2g}" for g, r in zip(grads, reference)]
            print(f"| {std} | {score:,.0f} | {logit:,.0f} | {name} | " + " | ".join(f"{norm(g):.3g}" for g in grads) + " | " + " | ".join(errors) + " |")


def model_table():
    """The backward MLX 0.32.4 added for this op (upstream `sdpa_vjp_blocked` and the kernel `sdpa_vjp_ds`), in
    float64: the scores `S = Q K^T` held in the inputs' dtype, the probabilities rebuilt as
    `exp(S * scale - lse)` from the forward's log-sum-exp, `dS = P * (dP - delta) * scale` with `dP = dO V^T` and
    `delta` the row's `o . dO`, then `dQ = dS K`, `dK = dS^T Q`, `dV = P^T dO`. Rounding S, dP, dS and P to bf16 is
    all that separates this from the exact gradient."""
    print("| q, k std | largest score | bf16 step there | logit error up to | largest row sum of P | dQ, dK, dV: exact | the model | MLX's fast call, bf16 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    scale = 1 / math.sqrt(WIDTH)
    for std in (5, 15, 30, 40):
        q, k, v, ct = (t.astype(np.float64) for t in inputs(std))
        scores = q @ np.swapaxes(k, -1, -2)
        top = (scores * scale).max(-1, keepdims=True)
        lse = np.log(np.exp(scores * scale - top).sum(-1, keepdims=True)) + top
        d_p = ct @ np.swapaxes(v, -1, -2)

        def backward(rounded):
            r = (lambda a: to_bf16(a.astype(np.float32)).astype(np.float64)) if rounded else (lambda a: a)
            with np.errstate(over="ignore", invalid="ignore"):
                p = np.exp(r(scores) * scale - lse)
                delta = ((np.exp(scores * scale - lse) @ v) * ct).sum(-1, keepdims=True)  # o . dO, from the forward's output
                d_s = r(p * (r(d_p) - delta) * scale)
                p = r(p)
                return p, [d_s @ k, np.swapaxes(d_s, -1, -2) @ q, np.swapaxes(p, -1, -2) @ ct]

        _, exact = backward(False)
        p, model = backward(True)
        ours = mlx_gradients(fast, *(t.astype(np.float32) for t in (q, k, v, ct)), mx.bfloat16)
        score = float(np.abs(scores).max())
        cells = [", ".join(f"{norm(g):.4g}" for g in grads) for grads in (exact, model, ours)]
        print(f"| {std} | {score:,.0f} | {bf16_step(score):,.0f} | {bf16_step(score) / 2 * scale:.1f} | {p.sum(-1).max():.3g} | " + " | ".join(cells) + " |")


SOFTMAX = {
    "mx.softmax": lambda s: mx.softmax(s, axis=-1),
    "mx.softmax, precise": lambda s: mx.softmax(s, axis=-1, precise=True),
    "exp(s - logsumexp(s))": lambda s: mx.exp(s - mx.logsumexp(s, axis=-1, keepdims=True)),
    "exp(s - max) / sum": lambda s: (lambda e: e / e.sum(axis=-1, keepdims=True))(mx.exp(s - mx.max(s, axis=-1, keepdims=True))),
}


def spellings_table():
    print("| largest logit | softmax, in bf16 | dQ | dK | dV | error of dQ | of dK | of dV |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for std in (30, 40):
        q, k, v, ct = inputs(std)
        logit = largest_score(q, k) / math.sqrt(WIDTH)
        reference = torch_gradients(q, k, v, ct, torch.float32)
        for name, softmax in SOFTMAX.items():
            def attend(q, k, v, softmax=softmax):
                return softmax((q * (1 / math.sqrt(q.shape[-1]))) @ k.transpose(0, 1, 3, 2)) @ v

            grads = mlx_gradients(attend, q, k, v, ct, mx.bfloat16)
            print(f"| {logit:,.0f} | {name} | " + " | ".join(f"{norm(g):.3g}" for g in grads) + " | "
                  + " | ".join(f"{error(g, r):.2g}" for g, r in zip(grads, reference)) + " |")
        print(f"| {logit:,.0f} | (torch, float32) | " + " | ".join(f"{norm(g):.3g}" for g in reference) + " | | | |")


def width_table():
    print("| head width | bf16 forward that is float32's, rounded | bf16 dV | float32 dV |")
    print("| ---: | ---: | ---: | ---: |")
    for width in (32, 48, 63, 64, 65, 72, 80, 96, 100, 127, 128, 129, 160):
        q, k, v, ct = inputs(30, rows=96, width=width)
        q16, k16, v16 = (mx.array(t).astype(mx.bfloat16) for t in (q, k, v))
        through = fast(q16.astype(mx.float32), k16.astype(mx.float32), v16.astype(mx.float32)).astype(mx.bfloat16)
        same = np.mean(np.array(fast(q16, k16, v16).astype(mx.float32)) == np.array(through.astype(mx.float32)))
        half = mlx_gradients(fast, q, k, v, ct, mx.bfloat16)[2]
        full = mlx_gradients(for_training, q, k, v, ct, mx.bfloat16)[2]
        print(f"| {width} | {same:.1%} | {norm(half):.3g} | {norm(full):.3g} |")


def kernel_name():
    """At a head width of 192 the fused kernel for float32 inputs fails to load when a gradient is taken, and the
    error names it. (The gradient of the bf16 call runs there.)"""
    q, k, v, ct = inputs(1, rows=96, width=192)
    for name, dtype in (("float32", mx.float32), ("bf16", mx.bfloat16)):
        try:
            mlx_gradients(fast, q, k, v, ct, dtype)
        except RuntimeError as e:
            print(f"a gradient of the {name} call at a head width of 192: " + str(e).splitlines()[0])
        else:
            print(f"a gradient of the {name} call at a head width of 192 ran")


def rms_norm_table():
    print("| input | eps | torch bf16: dx, dw | fast bf16: dx, dw | fast bf16 on the CPU stream: dx, dw |")
    print("| --- | ---: | ---: | ---: | ---: |")
    rng = np.random.default_rng(0)
    shape = (1, 600, 2, 128)
    for eps in (1e-5, 1e-6):
        for what, scale, gain in (("unit rows", 1.0, 1.0), ("rows of 1e-3", 1e-3, 1.0), ("rows of 1e-6", 1e-6, 1.0),
                                  ("one row exactly zero", 1.0, 1.0), ("rows of 1e6", 1e6, 1.0), ("affine weights of 40", 1.0, 40.0),
                                  ("rows of 1e-4, weights of 40", 1e-4, 40.0), ("rows from 1e-6 to 1e4", None, 5.0)):
            x = rng.standard_normal(shape).astype(np.float32)
            x *= (10.0 ** rng.uniform(-6, 4, shape[:-1] + (1,))).astype(np.float32) if scale is None else np.float32(scale)
            if what == "one row exactly zero":
                x[0, 3] = 0
            w = (rng.standard_normal(shape[-1]) * gain).astype(np.float32)
            ct = rng.standard_normal(shape).astype(np.float32)

            def theirs(dtype):
                x_, w_ = (torch.from_numpy(t).to(dtype).requires_grad_(True) for t in (x, w))
                F.rms_norm(x_, (shape[-1],), w_, eps).backward(torch.from_numpy(ct).to(dtype))
                return x_.grad.float().numpy(), w_.grad.float().numpy()

            def ours(device):
                with mx.stream(device):
                    _, grads = mx.vjp(lambda x, w: ops.rms_norm(x, w, eps), [mx.array(x).astype(mx.bfloat16), mx.array(w).astype(mx.bfloat16)],
                                      [mx.array(ct).astype(mx.bfloat16)])
                    mx.eval(grads)
                return [np.array(g.astype(mx.float32)) for g in grads]

            reference = theirs(torch.float32)
            cells = [", ".join(f"{error(g, r):.1e}" for g, r in zip(grads, reference))
                     for grads in (theirs(torch.bfloat16), ours(mx.default_device()), ours(mx.cpu))]
            print(f"| {what} | {eps:g} | " + " | ".join(cells) + " |")


COST = {"the bf16 call": fast, "`attention_for_training`": for_training}


def cost_cell(name, heads=56, rows=600, width=128, repeats=10):
    """One measurement, printed as JSON: the peak of MLX's active memory over one forward and backward (over what
    was active before it, the inputs), and the mean time of ten."""
    f = COST[name]
    q, k, v, ct = inputs(1, heads, rows, width)
    args = [mx.array(t).astype(mx.bfloat16) for t in (q, k, v)]
    cotangent = [mx.array(ct).astype(mx.bfloat16)]
    mx.eval(args, cotangent)
    before = mx.get_active_memory()  # the inputs
    mx.eval(mx.vjp(f, args, cotangent)[1])  # the kernels are built and the buffers have been seen once
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(mx.vjp(f, args, cotangent)[1])
    peak = mx.get_peak_memory() - before
    began = time.perf_counter()
    for _ in range(repeats):
        mx.eval(mx.vjp(f, args, cotangent)[1])
    print(json.dumps({"peak_mb": peak / 1e6, "ms": 1000 * (time.perf_counter() - began) / repeats}))


def cost_table(heads=56, rows=600, width=128):
    elements = heads * rows * width
    print(f"q, k and v are {elements:,} elements each: {2 * elements / 1e6:.1f} MB in bf16, {4 * elements / 1e6:.1f} MB in float32; "
          f"the scores are {heads * rows * rows:,}, {4 * heads * rows * rows / 1e6:.1f} MB in float32")
    print("| attention | forward and backward, five runs | peak memory over it, five runs |")
    print("| --- | ---: | ---: |")
    for name in COST:
        runs = [json.loads(subprocess.run([sys.executable, __file__, "--cost-cell", name], capture_output=True, text=True,
                                          check=True).stdout.strip().splitlines()[-1]) for _ in range(5)]
        print(f"| {name} | " + ", ".join(f"{r['ms']:.1f}" for r in runs) + " ms | " + ", ".join(f"{r['peak_mb']:.0f}" for r in runs) + " MB |")


REAL = dict(num_attention_heads=56, attention_head_dim=128, hidden_size=5376, ffn_dim=14336, num_layers=2,
            num_refiner_layers=1, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=5120, freq_dim=256,
            time_embed_hidden_dim=5376, time_embed_dim=2688, rope_freq_dim=16, rope_theta=10000.0, norm_eps=1e-5,
            qk_norm_eps=1e-5, final_norm_eps=1e-5)  # the release's transformer config, with 2 blocks for its 50
STEP = {"`attention_for_training`": None, "the bf16 call": "attention"}


def step_cell(name, repeats=3):
    """One measurement, printed as JSON: the block-by-block gradient of a two-block transformer of the release's
    widths on one 768 by 768 example (576 picture rows, 64 text rows and the audio rows), seconds and peak."""
    from h3mlx import backward, lora, train
    from h3mlx.pipeline import Noise

    sys.path.insert(0, "tests")
    from conftest import state_mx
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from test_dit_mixed_precision import to_mixed

    cfg = dict(REAL)
    torch.manual_seed(0)
    base = state_mx(to_mixed(MiniMaxH3Transformer3DModel(**cfg).eval()))
    g = torch.Generator().manual_seed(1)
    fresh = lora.init(cfg, rank=16, modules=[m for m in lora.block_modules(cfg) if m.startswith("transformer_blocks.")], seed=0)
    tensors = {key: (value if ".lora_A." in key else mx.array((torch.randn(value.shape, generator=g) * 0.01).numpy()))
               for key, value in fresh.tensors.items()}
    frozen, trainable = lora.split(lora.attach(base, cfg, lora.Adapter(tensors, fresh.alpha), keep_dtype=True, name="trained"), "trained")
    latents = mx.array(torch.randn(1, 1, 48, 48, cfg["in_channels"], generator=g).numpy())
    example = train.build_example(cfg, np.array([1] * 64), latents, 0.9, Noise(0), condition_t=0.999)
    text = mx.array(torch.randn(1, 64, cfg["text_dim"], generator=g).numpy())
    mx.eval(frozen, trainable, example.video_rows, example.audio_rows, example.target, text)
    if STEP[name]:
        ops.attention_for_training = getattr(ops, STEP[name])

    def gradient():
        loss, grads = backward.loss_and_gradients_blockwise(trainable, frozen, cfg, example, text)
        mx.eval(loss, grads)
        return float(loss), math.sqrt(sum(float((g * g).sum()) for g in grads.values()))

    loss, size = gradient()  # the weights are read and the kernels built
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    began = time.perf_counter()
    for _ in range(repeats):
        gradient()
    print(json.dumps({"rows": len(example.layout.token_tags), "seconds": (time.perf_counter() - began) / repeats,
                      "peak_mb": (mx.get_peak_memory() - before) / 1e6, "loss": loss, "norm": size}))


def step_table():
    print("| attention | rows | block-by-block gradient, two blocks, three runs | peak memory over it | loss | gradient norm |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in STEP:
        runs = [json.loads(subprocess.run([sys.executable, __file__, "--step-cell", name], capture_output=True, text=True,
                                          check=True).stdout.strip().splitlines()[-1]) for _ in range(3)]
        print(f"| {name} | {runs[0]['rows']} | " + ", ".join(f"{r['seconds']:.3f}" for r in runs) + " s | "
              + ", ".join(f"{r['peak_mb']:.0f}" for r in runs) + f" MB | {runs[0]['loss']:.4f} | {runs[0]['norm']:.4g} |")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cost-cell"]:
        cost_cell(sys.argv[2])
        raise SystemExit(0)
    if sys.argv[1:2] == ["--step-cell"]:
        step_cell(sys.argv[2])
        raise SystemExit(0)
    print(f"mlx {mx.__version__}, torch {torch.__version__}, default device {mx.default_device()}\n")
    op_table()
    print()
    model_table()
    print()
    spellings_table()
    print()
    width_table()
    kernel_name()
    print()
    rms_norm_table()
    print()
    cost_table()
    if "--skip-step" not in sys.argv:
        print()
        step_table()
