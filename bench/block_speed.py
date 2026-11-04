"""Time one MiniMax H3 DiT block at real size with random bf16 weights (no MiniMax weights involved).

    uv run python bench/block_speed.py [seq_len]

Default seq: 480p 5 s (864x480, 124 frames -> 37 latent frames x 27x15 rows) + 207x2 audio rows + 300 text rows.
"""

import sys
import time

import mlx.core as mx

from h3mlx import ops

H, HEADS, DH, FFN, TEMB = 5376, 56, 128, 14336, 2688
seq = int(sys.argv[1]) if len(sys.argv) > 1 else 37 * 405 + 414 + 300
dt = mx.bfloat16
inner = HEADS * DH


def w(o, i):
    return (mx.random.normal((o, i)) * (i**-0.5)).astype(dt)


p = {
    "norm1.weight": mx.ones((H,), dtype=dt), "norm2.weight": mx.ones((H,), dtype=dt),
    "attn.to_q.weight": w(inner, H), "attn.to_k.weight": w(inner, H), "attn.to_v.weight": w(inner, H),
    "attn.norm_q.weight": mx.ones((DH,), dtype=dt), "attn.norm_k.weight": mx.ones((DH,), dtype=dt),
    "attn.to_out.0.weight": w(H, inner),
    "ff.net.0.proj.weight": w(2 * FFN, H), "ff.net.2.weight": w(H, FFN),
    "adaln_proj.linear.weight": w(6 * H * 3, TEMB), "adaln_proj.linear.bias": mx.zeros((6 * H * 3,), dtype=dt),
}
mx.eval(p)
nparams = sum(v.size for v in p.values())
x = mx.random.normal((1, seq, H)).astype(dt)
temb = mx.random.normal((2, TEMB))
rows = mx.random.randint(0, 6, (seq,))
rot = ops.rope_cos_sin(mx.random.normal((seq, 3)) * 20, ops.rope_inv_freq(16, 1e4))
mx.eval(x, temb, rows, *rot)

times = []
for i in range(4):
    t0 = time.perf_counter()
    y = ops.dit_block(x, temb, rows, rot, p, HEADS, H)
    mx.eval(y)
    times.append(time.perf_counter() - t0)
best = min(times[1:])
print(f"seq {seq}, block params {nparams / 1e6:.0f}M bf16, peak mem {mx.get_peak_memory() / 1e9:.1f} GB")
print(f"one block: {best:.2f} s (runs {', '.join(f'{t:.2f}' for t in times)})")
print(f"one forward (50 blocks): ~{50 * best:.0f} s; Turbo4 (4 forwards) ~{200 * best / 60:.1f} min; "
      f"20 steps ~{1000 * best / 60:.0f} min  [DiT only: no text encoder, no VAE]")
