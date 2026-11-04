"""One HyperFlow-sized projection (ff.net.0.proj: 5376 -> 28672, rank 256, bf16) timed warm, as base, with the
unmerged LoRA (ops.plinear) and with the LoRA merged into the weight. Usage: python bench/lora_speed.py <rows> <mode>,
mode in base | lora | merged. Results at 52,487 rows on an M2 Max are in docs/porting-hyperflow.md section 7."""
import sys, time, gc, json
import mlx.core as mx
from h3mlx import ops
N = int(sys.argv[1]); mode = sys.argv[2]
I, O, R = 5376, 28672, 256
mx.set_cache_limit(0); mx.random.seed(77)
x = mx.random.normal((1, N, I)).astype(mx.bfloat16)
W = (mx.random.normal((O, I)) * 0.02).astype(mx.bfloat16)
A = (mx.random.normal((R, I)) * 0.02).astype(mx.bfloat16)
B = (mx.random.normal((O, R)) * 0.02).astype(mx.bfloat16)
p = {"ff.weight": W}
if mode == "lora":
    p.update({"ff.lora_A.weight": A, "ff.lora_B.weight": B})
if mode == "merged":
    p = {"ff.weight": (W.astype(mx.float32) + B.astype(mx.float32) @ A.astype(mx.float32)).astype(mx.bfloat16)}
mx.eval(x, p)
times = []
for rep in range(4):
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
    before = mx.get_active_memory(); t0 = time.perf_counter()
    y = ops.plinear(x, p, "ff"); mx.eval(y)
    times.append(round(time.perf_counter() - t0, 3)); peak = mx.get_peak_memory() - before
    del y
print(json.dumps({"N": N, "mode": mode, "times_s": times, "warm_mean_s": round(sum(times[1:]) / 3, 3), "peak_over_inputs_gb": round(peak / 1e9, 2)}))
