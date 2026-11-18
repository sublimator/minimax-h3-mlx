<!--
rendered_from: porting-lora-training.md.j2
rendered_at: 2026-11-18T15:07:36Z
branch: main
commit: 2bc9db8
commit_message: fix: train: encode an example listed twice only once
-->

---

<sub>Last updated: 2026-11-18 | branch: main | commit: 2bc9db8 (fix: train: encode an example listed twice only once)</sub>

---

# Porting LoRA training to MLX, part 1: PEFT and diffusers → `h3mlx`

This guide covers the first pieces a LoRA trainer for the MLX transformer needs, each ported op by op and tested
against the PyTorch original on small random tensors:

1. **LoRA adapters:** PEFT's scaling and initialisation, attaching and merging, and the files diffusers reads and
   writes (§1).
2. **A converter** for adapters saved under the original checkpoint's key names (§2).
3. **The forward process, the training target and the loss** (§3).
4. **Sampling the noise level and weighting the loss** (§4).
5. **Two named adapters on one `Linear`,** a frozen one beside a trainable one (§5).
6. **One training example:** a picture with or without reference pictures, as rows, audio rows and row timesteps (§6).
7. **The example's loss and its gradient** on the transformer (§7).
8. **A backward pass that does not hold every block at once,** two ways (§8).
9. **The optimizer step** (§9).
10. **A cache of encoded examples,** and how reference pictures are sized (§10).
11. **The training script:** `scripts/train_lora.py`, its plan, limits, record and checkpoints (§11).
12. **Sampling with a trained adapter:** `scripts/run_p_h3_image.py --lora`, attached at a scale (§12).

In each section the original comes first, then the port, then what changed and which test ties them together. It
assumes the main guide (`docs/porting.md`) for the transformer and the scheduler, and the HyperFlow guide
(`docs/porting-hyperflow.md`) for how a LoRA reaches the forward (`ops.plinear`).

**The tests here train no real model.** There is a training script (§11), and in the tests it runs on a tiny random
transformer: it overfits one synthetic example there, stops at its limits and resumes to the same weights.
The tests are weight-free. Runs on the released weights have since been made with the script: the first found a
wrong gradient that no tiny model had shown (§7.1, the attention's backward pass on MLX 0.32.4), and §14 gives their
measurements, what they showed about likeness, and what is still not done.

**Sources, pinned.**

| what | where | pin |
| --- | --- | --- |
| LoRA layer, scaling, init, merge (Apache-2.0) | `huggingface/peft`, `src/peft/tuners/lora/` | tag `v0.21.2` (`eaa35c7`), the version the tests import |
| LoRA files, the original-layout converter, the scheduler, `training_utils`, the P-H3-Image pipeline's layout and row steps (Apache-2.0) | `huggingface/diffusers` main | `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1`, the commit the tests import |
| a training recipe for this model (MIT; read, not run, not quoted) | `ostris/ai-toolkit`, `extensions_built_in/diffusion_models/minimax_h3/` | `d6a18ad72da9abca3ffb67ea8e42c25764eee61a` |
| the fused attention's backward pass (MIT; cited by file and function in §7.1, not quoted) | `ml-explore/mlx`, `mlx/fast.cpp`, `mlx/backend/metal/` | tag `v0.32.4`, the version the port locks |

**The oracle, and what it is not.** The tests compare against PEFT, diffusers and torch run on tiny models and random
tensors. diffusers' example training scripts are not part of the checkout these guides read. For the training-time
choices that diffusers has no code for, §6 follows a third-party trainer's model files as a named recipe. That trainer
is not the model maker's, and it is not an oracle here: it was read, not run, and nothing is compared with it
numerically. So this guide claims parity with the library functions it quotes and with nothing else: where a trainer
has to make a choice (the shift, the number of noise levels, the density, the weighting, the conditioning level), the
port takes an argument and has no default.

| piece | ours | original | tests |
| --- | --- | --- | --- |
| scaling, init, attach, merge, files | `src/h3mlx/lora.py`, `ops.plinear` | PEFT `layer.py`; diffusers `loaders/peft.py`, `utils/peft_utils.py`, `loaders/lora_pipeline.py` | `tests/test_lora.py` (72) |
| original-layout converter | `lora.convert_original` | diffusers `loaders/lora_conversion_utils.py` | `tests/test_lora_convert.py` (53) |
| forward process, target, loss | `src/h3mlx/train.py` | diffusers `scheduling_minimax_h3.py`; torch `mse_loss` | `tests/test_train.py` (47) |
| noise-level sampling, loss weighting | `src/h3mlx/train.py` | diffusers `training_utils.py`, its two schedulers | `tests/test_train_sampling.py` (37) |
| named adapters on one `Linear` | `src/h3mlx/lora.py`, `ops.plinear` | PEFT `layer.py`; diffusers `load_lora_adapter`, `set_adapters` | `tests/test_lora_stack.py` (14, one of them on a release fixture) |
| one training example | `train.build_example` | diffusers `p_h3_image/before_denoise.py`, `scheduling_minimax_h3.py`; the recipe | `tests/test_train_example.py` (33) |
| the example's loss and gradient | `train.example_loss`, `train.loss_and_gradients`, `ops.attention_for_training` | torch autograd on diffusers' transformer with PEFT's layers; torch's `scaled_dot_product_attention` and `rms_norm` | `tests/test_train_gradients.py` (46), `tests/test_attention_backward.py` (56) |
| a bounded backward pass | `src/h3mlx/backward.py` | the plain gradient of the row above | `tests/test_backward.py` (43) |
| the optimizer step | `train.adamw`, `train.optimizer_step` | `torch.optim.AdamW`, `clip_grad_norm_` | `tests/test_train_optimizer.py` (32) |
| the cache of encoded examples | `src/h3mlx/cache.py`, `src/h3mlx/files.py` | diffusers `p_h3_image/before_encoder.py` for one sizing rule; the recipe for the other | `tests/test_cache.py` (99, three of them on release fixtures) |
| the training script | `scripts/train_lora.py`, `src/h3mlx/trainer.py` | none: the pieces above, in order | `tests/test_train_lora.py` (104, one of them on release fixtures) |
| a trained adapter at sampling | `lora.for_sampling`, `lora.from_header`, `lora.summary`, `scripts/run_p_h3_image.py` | PEFT `set_scale`; diffusers `load_lora_adapter`, `set_adapters`, `apply_lora_scale`, and its P-H3-Image pipeline | `tests/test_lora_sampling.py` (25); on release fixtures, 2 in `tests/test_p_h3_image.py` and 14 in `tests/test_run_p_h3_image.py` |

## 1. LoRA adapters

An adapter is two matrices per `Linear`, `lora_A` (rank × in) and `lora_B` (out × rank), and a number, alpha. The
layer then computes `W x + (alpha / rank) · B (A x)`.

### 1.1 The scaling

PEFT sets the scaling when it builds the layer, and applies it in the forward:

📍 [`peft:src/peft/tuners/lora/layer.py:278-281`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L278-L281)
```python
 278         if use_rslora:
 279             self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
 280         else:
 281             self.scaling[adapter_name] = lora_alpha / r
```
📍 [`peft:src/peft/tuners/lora/layer.py:1088-1113`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L1088-L1113)
```python
1088             result = self.base_layer(x, *args, **kwargs)
1089             torch_result_dtype = result.dtype
1090
1091             lora_A_keys = self.lora_A.keys()
1092             for active_adapter in self.active_adapters:
1093                 if active_adapter not in lora_A_keys:
1094                     continue
1095
1096                 lora_A = self.lora_A[active_adapter]
1097                 lora_B = self.lora_B[active_adapter]
1098                 dropout = self.lora_dropout[active_adapter]
1099                 scaling = self.scaling[active_adapter]
1100                 x = self._cast_input_dtype(x, lora_A.weight.dtype)
1101                 if active_adapter not in self.lora_variant:  # vanilla LoRA
1102                     result = result + lora_B(lora_A(dropout(x))) * scaling
1103                 else:
1104                     result = self.lora_variant[active_adapter].forward(
1105                         self,
1106                         active_adapter=active_adapter,
1107                         x=x,
1108                         result=result,
1109                         **variant_kwargs,
1110                         **kwargs,
1111                     )
1112
1113             result = result.to(torch_result_dtype)
```

`scaling` is a Python float. The HyperFlow port left the multiplication out, because HyperFlow's alpha equals its
rank and `hyperflow.attach` refuses anything else. A general adapter needs it, so `plinear` now multiplies when the
weights carry a scaling, and is unchanged when they don't:

📍 [`src/h3mlx/ops.py:100-105`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L100-L105)
```python
 100 def times_scalar(x: mx.array, s) -> mx.array:
 101     """torch's `tensor * python_float`. A half-precision tensor is multiplied in float32 by the float32 scalar and
 102     rounded once; MLX alone would first round the scalar to the tensor's dtype (0.7 becomes 0.69921875 in bf16)."""
 103     if x.dtype in (mx.float16, mx.bfloat16):
 104         return (x.astype(mx.float32) * s).astype(x.dtype)
 105     return x * s
```
📍 [`src/h3mlx/ops.py:108-146`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L108-L146)
```python
 108 def plinear(x: mx.array, p: dict, name: str, bias: bool = False) -> mx.array:
 109     """The Linear `name` of state dict `p`, plus its LoRA when `name.lora_A.weight` / `name.lora_B.weight` are present.
 110
 111     PEFT's lora.Linear: the input is cast to the adapter's dtype, the two products round in that dtype, the delta is
 112     multiplied by the scaling (alpha / rank), and the sum is cast back to the base result's dtype. The scaling is the
 113     0-d float32 array `name.lora_scaling`, which `lora.attach` adds when it is not 1; without it the `* scaling` is
 114     exact and left out (HyperFlow's path: `hyperflow.attach` refuses any other scaling).
 115
 116     Named adapters: `name.lora_adapters`, when present, is the tuple of the adapters active on this Linear, in order,
 117     and each one's matrices are under PEFT's runtime names, `name.lora_A.<adapter>.weight` and
 118     `name.lora_B.<adapter>.weight`, with an optional `name.lora_scaling.<adapter>`. PEFT's loop is followed to the
 119     letter: the input is cast to each adapter's dtype in turn and that cast carries over to the next adapter, each
 120     scaled delta is added to the running result, and the result is cast back once at the end. An adapter that is
 121     listed without its matrices (the frozen half of `lora.split` on its own) is refused."""
 122     y = linear(x, p[name + ".weight"], p[name + ".bias"] if bias else None)
 123     adapters = p.get(name + ".lora_adapters")
 124     if adapters is not None:
 125         out = y
 126         for adapter in adapters:
 127             a, b = p.get(f"{name}.lora_A.{adapter}.weight"), p.get(f"{name}.lora_B.{adapter}.weight")
 128             if a is None or b is None:
 129                 raise ValueError(f"{name}: the adapter {adapter!r} is listed as active but its matrices are not in "
 130                                  "these weights. The frozen half of `lora.split` is not a model on its own: put the "
 131                                  "halves back together with `lora.join`")
 132             x = x.astype(a.dtype)
 133             delta = linear(linear(x, a), b)
 134             scaling = p.get(f"{name}.lora_scaling.{adapter}")
 135             if scaling is not None:
 136                 delta = times_scalar(delta, scaling)
 137             out = out + delta
 138         return out.astype(y.dtype)
 139     a = p.get(name + ".lora_A.weight")
 140     if a is None:
 141         return y
 142     delta = linear(linear(x.astype(a.dtype), a), p[name + ".lora_B.weight"])
 143     scaling = p.get(name + ".lora_scaling")
 144     if scaling is not None:
 145         delta = times_scalar(delta, scaling)
 146     return (y + delta).astype(y.dtype)
```

- **Where the scaling lives.** `lora.attach` adds `<module>.lora_scaling`, a 0-d float32 array, beside the two
  matrices, and only when the scaling is not 1. Without it the old expression runs, so HyperFlow's path and its tests
  are untouched.
- **How torch rounds it.** `tensor * python_float` on a bf16 or float16 tensor multiplies in float32 by the float32
  scalar and rounds once. MLX's own `bf16_array * 0.7` first rounds the scalar to bf16, which makes it 0.69921875: a
  relative error of 1.1e-3 in every adapter with such a scaling. `times_scalar` does what torch does. On float32
  tensors the two agree anyway.

`test_plinear_scaling_matches_peft_linear` checks one `Linear` against PEFT's layer at four (rank, alpha) pairs, and
that leaving the scaling out is visible. `test_half_precision_scaling_rounds_as_torch` isolates the rounding: with
selection matrices for A and B and a zero base weight every other step is exact, so the output is the scaled delta
alone and must equal PEFT's bit for bit in bf16 and float16; it also checks that MLX's plain product does not.

### 1.2 The initialisation

📍 [`peft:src/peft/tuners/lora/layer.py:332-345`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L332-L345)
```python
 332     def reset_lora_parameters(self, adapter_name, init_lora_weights):
 333         if init_lora_weights is not False:
 334             if adapter_name in self.lora_A.keys():
 335                 if init_lora_weights is True:
 336                     # initialize A the same way as the default for nn.Linear and B to zero
 337                     # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
 338                     nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
 339                 elif init_lora_weights.lower() == "gaussian":
 340                     nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
 341                 else:
 342                     raise ValueError(f"Unknown initialization {init_lora_weights=}")
 343                 nn.init.zeros_(self.lora_B[adapter_name].weight)
 344                 if self.lora_bias[adapter_name]:
 345                     nn.init.zeros_(self.lora_B[adapter_name].bias)
```

`kaiming_uniform_(a=sqrt(5))` is a uniform draw in ±bound with bound = √3 · gain / √fan_in and gain = √(2 / (1 + a²)),
which comes to 1 / √in. `lora_B` is zero, so a new adapter does nothing until it is trained.

📍 [`src/h3mlx/lora.py:138-148`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L138-L148)
```python
 138 def kaiming_uniform(shape, generator) -> np.ndarray:
 139     """`nn.init.kaiming_uniform_(weight, a=math.sqrt(5))` for a (rank, in) `lora_A`, drawn from a torch generator so the
 140     values are torch's: uniform in +-bound, with bound = sqrt(3) * gain / sqrt(fan_in) and gain = sqrt(2 / (1 + a^2)),
 141     which is 1 / sqrt(in). The bound is computed in the same order as torch computes it, in Python floats."""
 142     import torch
 143
 144     fan_in = shape[1]
 145     gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
 146     std = gain / math.sqrt(fan_in)
 147     bound = math.sqrt(3.0) * std
 148     return torch.empty(tuple(shape), dtype=torch.float32).uniform_(-bound, bound, generator=generator).numpy()
```
📍 [`src/h3mlx/lora.py:151-173`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L151-L173)
```python
 151 def init(cfg: dict, rank: int, alpha: float | None = None, modules=None, seed: int = 0) -> Adapter:
 152     """A new adapter as PEFT initialises one (`init_lora_weights=True`): `lora_A` Kaiming-uniform, `lora_B` zero, so it
 153     starts as a no-op. Both are float32 whatever the base weights' dtype (a bf16 parameter is too coarse to train).
 154     `alpha` defaults to `rank` (scaling 1). `modules` defaults to `block_modules(cfg)`; they are drawn in sorted order
 155     from one torch generator seeded with `seed` (PEFT draws in the model's module order and from the global generator,
 156     so the same seed does not give PEFT's values)."""
 157     import torch
 158
 159     if rank <= 0:
 160         raise ValueError(f"`rank` should be a positive integer value but the value passed is {rank}")
 161     if alpha is not None:
 162         check_alpha("init", alpha)
 163     wired = wired_shapes(cfg)
 164     modules = sorted(block_modules(cfg) if modules is None else modules)
 165     generator = torch.Generator().manual_seed(seed)
 166     tensors = {}
 167     for module in modules:
 168         if module not in wired:
 169             raise ValueError(f"{module} is not a Linear of this transformer")
 170         out_f, in_f = wired[module]
 171         tensors[module + ".lora_A.weight"] = mx.array(kaiming_uniform((rank, in_f), generator))
 172         tensors[module + ".lora_B.weight"] = mx.zeros((out_f, rank), dtype=mx.float32)
 173     return Adapter(tensors, {m: rank if alpha is None else alpha for m in modules})
```

- **The draw is torch's.** The values come from a torch generator, as the sampling noise does, and the bound is
  computed in the same order in Python floats, so one matrix equals `torch.nn.init.kaiming_uniform_` bit for bit
  (`test_kaiming_uniform_is_torchs`).
- **Against PEFT itself.** A PEFT layer built after `torch.manual_seed(s)` first constructs two `nn.Linear`s, whose
  own constructors draw `lora_A`'s and `lora_B`'s default values, and only then redraws `lora_A`. Skip those two draws
  on a generator seeded the same and ours is PEFT's `lora_A` bit for bit (`test_init_draw_is_pefts`).
- **A whole adapter is not PEFT's for the same seed.** `init` draws the modules in sorted order from its own
  generator. PEFT draws in the model's module order from the global generator, with the two extra draws per layer.
  Tests that need equal adapters on both sides copy the tensors across.
- **float32, always.** The matrices are float32 whatever the base weights are. `alpha` defaults to the rank.
- **Which modules.** The default is the attention and feed-forward matrices of every block and refiner block
  (`block_modules`). Any other `Linear` of the transformer can be named explicitly (§1.3).

`test_init_is_a_float32_no_op` checks the shapes, the dtype over a bf16 base, the bound, the zero `lora_B`, the
order of the draws, and that the forward with a fresh adapter is the base model's bit for bit.

### 1.3 Attaching, and the dtype

📍 [`src/h3mlx/lora.py:38-63`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L38-L63)
```python
  38 @dataclass
  39 class Adapter:
  40     """LoRA factors for some of the transformer's Linears, in PEFT's parametrisation.
  41
  42     `tensors` maps `<module>.lora_A.weight` to (rank, in) and `<module>.lora_B.weight` to (out, rank), with diffusers
  43     module names and no component prefix: a file's names. `alpha` maps each module to its lora_alpha. A module's rank
  44     is its `lora_A`'s first axis; ranks and alphas may differ per module.
  45
  46     `attach` is how an adapter gets beside the transformer's weights: it adds the scalings and, for a named adapter,
  47     PEFT's runtime names and each Linear's list of active adapters. `tensors` on its own is not something to merge
  48     into the weights or to hand a loss: that would be right only for an unnamed adapter whose every scaling is 1, on
  49     Linears that carry no named adapter; beside a named adapter the forward would not read these keys at all."""
  50
  51     tensors: dict
  52     alpha: dict
  53
  54     @property
  55     def modules(self) -> list[str]:
  56         return sorted(self.alpha)
  57
  58     def rank(self, module: str) -> int:
  59         return int(self.tensors[module + ".lora_A.weight"].shape[0])
  60
  61     def scaling(self, module: str) -> float:
  62         """`lora_alpha / r`, as PEFT's `update_layer` sets it (without rsLoRA)."""
  63         return self.alpha[module] / self.rank(module)
```
📍 [`src/h3mlx/lora.py:69-82`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L69-L82)
```python
  69 def wired_shapes(cfg: dict) -> dict:
  70     """{module: (out, in)} for every Linear of the transformer, all of which the forward runs through `ops.plinear`:
  71     the six matrices of each block and refiner block, the time embedder's two, each block's AdaLN projection and the
  72     final one, and the input and output projections. These are the `nn.Linear`s of diffusers' model, the modules PEFT
  73     can put a LoRA on."""
  74     out = {m: s for m, s in hyperflow.wired_shapes(cfg).items() if not m.startswith(hyperflow.ENDPOINT)}
  75     hidden, temb = cfg["hidden_size"], cfg["time_embed_dim"]
  76     patch_dim = cfg["in_channels"] * math.prod(cfg["patch_size"])
  77     out.update({f"transformer_blocks.{i}.adaln_proj.linear": (6 * hidden * ops.MODALITY_NUM, temb)
  78                 for i in range(cfg["num_layers"])})
  79     out.update({"proj_in": (hidden, patch_dim), "audio_proj_in": (hidden, cfg["audio_in_channels"]),
  80                 "context_embedder": (hidden, cfg["text_dim"]), "norm_out.linear": (2 * hidden, temb),
  81                 "proj_out": (patch_dim, hidden), "audio_proj_out": (cfg["audio_in_channels"], hidden)})
  82     return out
```
📍 [`src/h3mlx/lora.py:114-132`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L114-L132)
```python
 114 def check(adapter: Adapter, cfg: dict) -> None:
 115     """Every module is a Linear of the transformer (the forward applies a LoRA on each of them, and on nothing else: a
 116     target that is not one would be silently ignored, so it is refused), the tensors are exactly those modules' pairs,
 117     each pair fits its Linear, and each alpha is a positive, finite number."""
 118     check_keys(adapter)
 119     wired = wired_shapes(cfg)
 120     unwired = sorted(m for m in adapter.alpha if m not in wired)
 121     if unwired:
 122         raise ValueError(f"{len(unwired)} LoRA target(s) are not Linears of this transformer (the forward would ignore "
 123                          f"them), e.g. {unwired[:5]}")
 124     for module in adapter.modules:
 125         a = tuple(adapter.tensors[module + ".lora_A.weight"].shape)
 126         b = tuple(adapter.tensors[module + ".lora_B.weight"].shape)
 127         out_f, in_f = wired[module]
 128         if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
 129             raise ValueError(f"{module}: lora_A {a} / lora_B {b} do not fit the Linear ({out_f}, {in_f})")
 130         if a[0] <= 0:
 131             raise ValueError(f"{module}: LoRA rank must be positive, got {a[0]}")
 132         check_alpha(module, adapter.alpha[module])
```
📍 [`src/h3mlx/lora.py:224-274`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L224-L274)
```python
 224 def attach(params: dict, cfg: dict, adapter: Adapter, keep_dtype: bool = False, name: str | None = None,
 225            scale: float = 1.0) -> dict:
 226     """The transformer's weights with the adapter beside them, for `ops.plinear`; `params` is not modified.
 227
 228     By default each matrix is cast to its base weight's dtype, as PEFT's injection does (`load_lora_adapter`, and
 229     `hyperflow.attach`). `keep_dtype=True` leaves the adapter as it is: float32 factors over a bf16 base, what
 230     `cast_training_params` gives a trainer (and what diffusers' `low_cpu_mem_usage=True` loading gives a float32 file).
 231     A module whose scaling is not 1 also gets a `lora_scaling` entry, in float32; a scaling that is finite as a Python
 232     float and not in float32 (a scale of 1e40) is refused.
 233
 234     `scale` is the adapter's strength, 1.0 being as trained: each module's scaling becomes `scale * alpha / rank`,
 235     computed in that order, which is PEFT's `set_scale` and so what diffusers' `set_adapters(name, weights=scale)`
 236     leaves on every layer. Any finite number: a negative one subtracts the adapter's delta, and zero keeps the adapter
 237     attached with a delta of zero (`for_sampling` leaves it off instead).
 238
 239     Without a `name` the adapter is the only one its Linears can carry. With one it is stacked: its matrices go under
 240     PEFT's runtime names, and the name is appended to each Linear's `lora_adapters`, so adapters apply in the order
 241     they were attached (`set_active` changes that). A named and an unnamed adapter can't share a Linear, and a name is
 242     used once: diffusers' `load_lora_adapter` refuses a name already in use too. That includes a name these weights
 243     only list or carry a scaling for, which is what the frozen half of `split` looks like (`join` is the way back
 244     from it); attaching onto that half would name the adapter twice on each Linear and apply it twice."""
 245     check(adapter, cfg)
 246     check_scalings(adapter, scale)
 247     if name is None:
 248         taken = [m for m in adapter.modules if _carries_lora(params, m)]
 249         if taken:
 250             raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: one unnamed adapter per "
 251                              "Linear; give each a name to stack them")
 252     else:
 253         if not name or "." in name:
 254             raise ValueError(f"an adapter name is a non-empty word without dots, got {name!r}")
 255         taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
 256         if taken:
 257             raise ValueError(f"{len(taken)} Linear(s) already carry an unnamed LoRA, e.g. {taken[:3]}")
 258         used = _named_parts(params, name)
 259         if used:
 260             raise ValueError(f"the adapter name {name!r} is already in use in these weights ({len(used)} keys, e.g. "
 261                              f"{used[:2]}). If they are the frozen half of `split`, `join` it with its trainable half; "
 262                              "a changed adapter goes in as that trainable half, not through a second `attach`")
 263     out = dict(params)
 264     for module in adapter.modules:
 265         dtype = params[module + ".weight"].dtype
 266         for matrix in ("A", "B"):
 267             tensor = adapter.tensors[f"{module}.lora_{matrix}.weight"]
 268             out[runtime_key(module, matrix, name)] = tensor if keep_dtype else tensor.astype(dtype)
 269         scaling = adapter.scaling(module) if scale == 1.0 else scale * adapter.alpha[module] / adapter.rank(module)
 270         if scaling != 1.0:
 271             out[module + SCALING + ("" if name is None else "." + name)] = mx.array(scaling, dtype=mx.float32)
 272         if name is not None:
 273             out[module + ADAPTERS] = params.get(module + ADAPTERS, ()) + (name,)
 274     return out
```

- **Any `Linear` of the transformer.** PEFT wraps any `nn.Linear` a file names, and diffusers loads a LoRA onto each:
  the six matrices of a block or refiner block, the time embedder's two, a block's AdaLN projection
  (`adaln_proj.linear`), the final one (`norm_out.linear`), and the input and output projections. The forward now runs
  every one of them through `plinear`, so an adapter on any of them acts as it does in diffusers. A first version
  served the attention, feed-forward and time-embedder linears only and refused the rest; published MiniMax H3
  adapters carry LoRAs on the AdaLN projections (the mixin's comment in §1.5), so that was a gap.
- **The modulation cache follows.** The cache (main guide, §2.6) computes each block's AdaLN tables up front and can
  drop the projections. Its tables come from the same `plinear` call, so they include a LoRA on a projection, and
  dropping a projection drops the LoRA on it. `test_modulation_cache_carries_a_lora_on_the_adaln_projections` checks a
  cached forward against the uncached one bit for bit.
- **A target that is not a `Linear` of the model is refused.** The forward would ignore it.
- **One unnamed adapter per `Linear`.** Attaching one to weights that already carry a LoRA, HyperFlow's included, is
  refused. Named adapters stack (§5).
- **An adapter must match itself.** `Adapter` is two plain dicts, tensors and alphas. `check_keys` refuses a stray
  tensor, half a pair, a pair without an alpha and an alpha without a pair, for `attach`, `merge` and `save` alike
  (`test_an_adapter_must_match_its_own_modules`).
- **A positive, finite alpha. This differs from PEFT on purpose.** PEFT takes any number: zero leaves an adapter that
  does nothing and cannot train, a negative one flips the delta's sign, NaN and infinity poison the forward. `check`
  refuses them, and so do `init`, `read` and `save` (§1.5). `test_alpha_must_be_positive_and_finite` shows PEFT's
  layer taking each of the four and ours refusing it at every entry and exit.

📍 [`src/h3mlx/lora.py:91-100`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L91-L100)
```python
  91 def check_alpha(what: str, alpha) -> None:
  92     """An alpha is a positive, finite number. PEFT takes any: zero leaves an adapter that does nothing and cannot
  93     train, a negative one flips the delta's sign, and NaN or infinity poison the forward and are written to the file's
  94     header as `NaN` / `Infinity`, which is not JSON. Refused wherever an alpha comes in or goes out."""
  95     try:
  96         ok = math.isfinite(alpha) and alpha > 0
  97     except TypeError:
  98         ok = False
  99     if not ok:
 100         raise ValueError(f"{what}: lora_alpha is {alpha!r}; it must be a positive, finite number")
```
📍 [`src/h3mlx/lora.py:103-111`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L103-L111)
```python
 103 def check_keys(adapter: Adapter) -> None:
 104     """The tensors are the two matrices of each module that has an alpha, and nothing else: no stray tensor, no half
 105     pair, no pair without an alpha. An `Adapter` is two plain dicts, so nothing else keeps them in step."""
 106     if not adapter.alpha:
 107         raise ValueError("no LoRA tensors in this adapter")
 108     want = {f"{module}.lora_{matrix}.weight" for module in adapter.alpha for matrix in "AB"}
 109     missing, extra = sorted(want - set(adapter.tensors)), sorted(set(adapter.tensors) - want)
 110     if missing or extra:
 111         raise ValueError(f"the adapter's tensors do not match its modules: missing {missing[:4]}, unexpected {extra[:4]}")
```
- **The dtype has two answers in the original, and so here.** PEFT creates an adapter in its base layer's dtype, and
  `load_lora_adapter` copies a file's values into it: bf16 on the blocks, float32 on the time embedder. That is the
  default (and what `hyperflow.attach` does). A trainer then casts the trainable parameters up:

📍 [`diffusers:src/diffusers/training_utils.py:322-336`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/training_utils.py#L322-L336)
```python
 322 def cast_training_params(model: torch.nn.Module | list[torch.nn.Module], dtype=torch.float32):
 323     """
 324     Casts the training parameters of the model to the specified data type.
 325
 326     Args:
 327         model: The PyTorch model whose parameters will be cast.
 328         dtype: The data type to which the model parameters will be cast.
 329     """
 330     if not isinstance(model, list):
 331         model = [model]
 332     for m in model:
 333         for param in m.parameters():
 334             # only upcast trainable parameters into fp32
 335             if param.requires_grad:
 336                 param.data = param.to(dtype)
```

  `keep_dtype=True` is that state: float32 factors over a bf16 base. `plinear` needs no change for it. The input is
  cast to the adapter's dtype, the delta is float32, the sum with the bf16 base result promotes to float32 and is cast
  back, which is PEFT's forward line for line.

`test_adapter_dtypes_follow_peft` asserts the dtypes on both sides in both states, on a model in the release's mixed
precision, and holds each forward to the bound the base model's mixed-precision test uses (MLX's error against the
float32 run at most 1.5× torch's own, plus 1e-3). Measured relative errors, torch then MLX:

| adapter | video | audio |
| --- | --- | --- |
| cast to the base dtype (bf16 blocks) | 0.0117, 0.0109 | 0.0083, 0.0096 |
| kept float32 over bf16 blocks | 0.0099, 0.0102 | 0.0099, 0.0067 |

### 1.4 Merging

📍 [`peft:src/peft/tuners/lora/layer.py:1044-1072`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L1044-L1072)
```python
1044     def get_delta_weight(self, adapter) -> torch.Tensor:
1045         """
1046         Compute the delta weight for the given adapter.
1047
1048         Args:
1049             adapter (str):
1050                 The name of the adapter for which the delta weight should be computed.
1051         """
1052         device = self.lora_B[adapter].weight.device
1053         dtype = self.lora_B[adapter].weight.dtype
1054
1055         # In case users wants to merge the adapter weights that are in
1056         # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
1057         # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
1058         cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
1059
1060         weight_A = self.lora_A[adapter].weight
1061         weight_B = self.lora_B[adapter].weight
1062
1063         if cast_to_fp32:
1064             weight_A = weight_A.float()
1065             weight_B = weight_B.float()
1066
1067         output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
1068
1069         if cast_to_fp32:
1070             output_tensor = output_tensor.to(dtype=dtype)
1071
1072         return output_tensor
```
📍 [`peft:src/peft/tuners/lora/layer.py:1005-1008`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L1005-L1008)
```python
1005                 else:
1006                     if active_adapter not in self.lora_variant:  # vanilla LoRA
1007                         delta_weight = self.get_delta_weight(active_adapter)
1008                         base_layer.weight.data += delta_weight
```

📍 [`src/h3mlx/lora.py:408-416`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L408-L416)
```python
 408 def delta_weight(a: mx.array, b: mx.array, scaling: float) -> mx.array:
 409     """PEFT's `Linear.get_delta_weight` on CPU: `(B @ A) * scaling`, with half-precision factors multiplied in float32
 410     and the product cast back to their dtype."""
 411     dtype = b.dtype
 412     half = dtype in (mx.float16, mx.bfloat16)
 413     if half:
 414         a, b = a.astype(mx.float32), b.astype(mx.float32)
 415     out = (b @ a) * scaling
 416     return out.astype(dtype) if half else out
```
📍 [`src/h3mlx/lora.py:419-440`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L419-L440)
```python
 419 def merge(params: dict, cfg: dict, adapter: Adapter) -> dict:
 420     """The transformer's plain weights with the adapter added into them (PEFT's `Linear.merge`: `weight +=
 421     delta_weight`, in the weight's dtype); `params` is not modified, and the result has its keys and nothing of the
 422     adapter left to apply.
 423
 424     Attaching and merging are alternatives here. Weights that already carry a LoRA, named or not, on one of the
 425     adapter's Linears are refused: the attached factors would stay in the result, and `plinear` would apply them on top
 426     of the merged ones. (A PEFT layer holds both states itself, and its forward skips an adapter once it is merged.)
 427     Not for quantized weights."""
 428     check(adapter, cfg)
 429     taken = [m for m in adapter.modules if _carries_lora(params, m)]
 430     if taken:
 431         raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: merge into the plain "
 432                          "weights, not into attached ones (the adapter would be applied twice)")
 433     out = dict(params)
 434     for module in adapter.modules:
 435         w = params[module + ".weight"]
 436         if isinstance(w, ops.QuantizedWeight):
 437             raise ValueError(f"{module}: can't merge a LoRA into a quantized weight; merge first, then quantize")
 438         a, b = adapter.tensors[module + ".lora_A.weight"], adapter.tensors[module + ".lora_B.weight"]
 439         out[module + ".weight"] = (w + delta_weight(a, b, adapter.scaling(module))).astype(w.dtype)
 440     return out
```

`delta_weight` takes PEFT's CPU path: half-precision factors are multiplied in float32 and the product is cast back.
`merge` takes the plain weights and returns new ones with the same keys, and refuses a quantized weight.

- **Attaching and merging are alternatives. This differs from PEFT in shape.** A PEFT layer holds both states: the
  adapter is attached first, and once merged its forward skips it. Here the two are separate weight dicts, so `merge`
  refuses weights that already carry a LoRA on one of the adapter's linears. Merging into them would leave the
  attached factors in the result, and `plinear` would apply the adapter a second time on top of the merged weights.
  The order that works is to merge into the plain weights and attach whatever else afterwards.

`test_merged_equals_unmerged` checks the merged forward against the unmerged one, against PEFT's unmerged forward, and
against PEFT's own merge (`fuse_lora`), weights and forward. `test_merge_takes_plain_weights_only` checks the refusal
(for the adapter's own factors, another adapter's, and HyperFlow's), that the adapter on top of its own merge is far
from the adapter applied once, and that merge-then-attach equals merging both. `test_delta_weight_and_merge_are_pefts`
checks one layer for float32, for bf16, and for a float32 adapter over a bf16 weight; the bf16 results are held to one
bf16 step, because the float32 product's last bit depends on the matmul's order.

The HyperFlow guide's reasons for not merging at inference still stand (bf16 rounding differs from the unmerged path).
Merging is here for a trained adapter's user, not for the run scripts.

### 1.5 The files

diffusers writes two layouts, and reads both:

| written by | keys | `lora_adapter_metadata` keys | read by |
| --- | --- | --- | --- |
| `save_lora_adapter` (the model) | `<module>.lora_A.weight` | `r`, `lora_alpha`, … | `load_lora_adapter(..., prefix=None)` |
| `save_lora_weights` (the pipeline mixin) | `transformer.<module>.lora_A.weight` | `transformer.r`, `transformer.lora_alpha`, … | `load_lora_adapter(...)`, whose default prefix is `transformer` |

The second is also the layout of HyperFlow's file. The metadata is PEFT's whole `LoraConfig` as JSON, in the
safetensors header beside `format: pt`:

📍 [`diffusers:src/diffusers/loaders/peft.py:425-444`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/peft.py#L425-L444)
```python
 425         lora_adapter_metadata = self.peft_config[adapter_name].to_dict()
 426
 427         lora_layers_to_save = get_peft_model_state_dict(
 428             self.to(dtype=torch.float32 if upcast_before_saving else None), adapter_name=adapter_name
 429         )
 430         if os.path.isfile(save_directory):
 431             raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
 432
 433         if safe_serialization:
 434
 435             def save_function(weights, filename):
 436                 # Inject framework format.
 437                 metadata = {"format": "pt"}
 438                 if lora_adapter_metadata is not None:
 439                     for key, value in lora_adapter_metadata.items():
 440                         if isinstance(value, set):
 441                             lora_adapter_metadata[key] = list(value)
 442                     metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)
 443
 444                 return safetensors.torch.save_file(weights, filename, metadata=metadata)
```
📍 [`diffusers:src/diffusers/loaders/lora_pipeline.py:7446-7487`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_pipeline.py#L7446-L7487)
```python
7446     @classmethod
7447     def save_lora_weights(
7448         cls,
7449         save_directory: str | os.PathLike,
7450         transformer_lora_layers: dict[str, torch.nn.Module | torch.Tensor] = None,
7451         transformer_ref_lora_layers: dict[str, torch.nn.Module | torch.Tensor] = None,
7452         is_main_process: bool = True,
7453         weight_name: str = None,
7454         save_function: Callable = None,
7455         safe_serialization: bool = True,
7456         transformer_lora_adapter_metadata: dict | None = None,
7457         transformer_ref_lora_adapter_metadata: dict | None = None,
7458     ):
7459         r"""
7460         Save the LoRA layers of one or both MiniMax-H3 partitions. Which partition a LoRA belongs to is not recoverable
7461         from its keys, so this is the only way to publish an H3 LoRA that records it. See
7462         [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for more information.
7463         """
7464         lora_layers = {}
7465         lora_metadata = {}
7466
7467         if transformer_lora_layers:
7468             lora_layers[cls.transformer_name] = transformer_lora_layers
7469             lora_metadata[cls.transformer_name] = transformer_lora_adapter_metadata
7470         if transformer_ref_lora_layers:
7471             lora_layers[cls.transformer_ref_name] = transformer_ref_lora_layers
7472             lora_metadata[cls.transformer_ref_name] = transformer_ref_lora_adapter_metadata
7473
7474         if not lora_layers:
7475             raise ValueError(
7476                 "You must pass at least one of `transformer_lora_layers` or `transformer_ref_lora_layers`."
7477             )
7478
7479         cls._save_lora_weights(
7480             save_directory=save_directory,
7481             lora_layers=lora_layers,
7482             lora_metadata=lora_metadata,
7483             is_main_process=is_main_process,
7484             weight_name=weight_name,
7485             save_function=save_function,
7486             safe_serialization=safe_serialization,
7487         )
```

On loading, diffusers filters the keys and the metadata by the prefix, reads each module's rank off `lora_B`, and
builds a `LoraConfig` from the metadata if there is any, or else derives one:

📍 [`diffusers:src/diffusers/loaders/peft.py:209-212`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/peft.py#L209-L212)
```python
 209         if prefix is not None:
 210             state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
 211             if metadata is not None:
 212                 metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}
```
📍 [`diffusers:src/diffusers/loaders/peft.py:250-275`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/peft.py#L250-L275)
```python
 250                 rank = {}
 251                 for key, val in state_dict.items():
 252                     # Cannot figure out rank from lora layers that don't have at least 2 dimensions.
 253                     # Bias layers in LoRA only have a single dimension
 254                     if "lora_B" in key and val.ndim > 1:
 255                         # Check out https://github.com/huggingface/peft/pull/2419 for the `^` symbol.
 256                         # We may run into some ambiguous configuration values when a model has module
 257                         # names, sharing a common prefix (`proj_out.weight` and `blocks.transformer.proj_out.weight`,
 258                         # for example) and they have different LoRA ranks.
 259                         rank[f"^{key}"] = val.shape[1]
 260
 261                 if network_alphas is not None and len(network_alphas) >= 1:
 262                     alpha_keys = [k for k in network_alphas.keys() if k.startswith(f"{prefix}.")]
 263                     network_alphas = {
 264                         k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys
 265                     }
 266
 267                 # create LoraConfig
 268                 adapter_config = _create_lora_config(
 269                     state_dict,
 270                     network_alphas,
 271                     metadata,
 272                     rank,
 273                     model_state_dict=self.state_dict(),
 274                     adapter_name=adapter_name,
 275                 )
```
📍 [`diffusers:src/diffusers/utils/peft_utils.py:491-511`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/utils/peft_utils.py#L491-L511)
```python
 491 def _create_lora_config(
 492     state_dict, network_alphas, metadata, rank_pattern_dict, is_unet=True, model_state_dict=None, adapter_name=None
 493 ):
 494     from peft import LoraConfig
 495
 496     if metadata is not None:
 497         lora_config_kwargs = metadata
 498     else:
 499         lora_config_kwargs = get_peft_kwargs(
 500             rank_pattern_dict,
 501             network_alpha_dict=network_alphas,
 502             peft_state_dict=state_dict,
 503             is_unet=is_unet,
 504             model_state_dict=model_state_dict,
 505             adapter_name=adapter_name,
 506         )
 507
 508     try:
 509         return LoraConfig(**lora_config_kwargs)
 510     except TypeError as e:
 511         raise TypeError("`LoraConfig` class could not be instantiated.") from e
```
📍 [`diffusers:src/diffusers/utils/peft_utils.py:153-208`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/utils/peft_utils.py#L153-L208)
```python
 153 def get_peft_kwargs(
 154     rank_dict, network_alpha_dict, peft_state_dict, is_unet=True, model_state_dict=None, adapter_name=None
 155 ):
 156     rank_pattern = {}
 157     alpha_pattern = {}
 158     r = lora_alpha = list(rank_dict.values())[0]
 159     has_alphas = network_alpha_dict is not None and len(network_alpha_dict) > 0
 160
 161     if len(set(rank_dict.values())) > 1:
 162         # get the rank occurring the most number of times
 163         r = collections.Counter(rank_dict.values()).most_common()[0][0]
 164
 165         # for modules with rank different from the most occurring rank, add it to the `rank_pattern`
 166         rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
 167         rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}
 168
 169         if not has_alphas:
 170             # No alpha data in the checkpoint: the diffusers/PEFT convention is
 171             # `W_eff = W + lora_B @ lora_A`, i.e. alpha == rank per module (scale 1.0).
 172             # Mirror the ranks into the alphas so every module keeps scale 1.0.
 173             lora_alpha = r
 174             alpha_pattern = dict(rank_pattern)
 175
 176     if network_alpha_dict is not None and len(network_alpha_dict) > 0:
 177         if len(set(network_alpha_dict.values())) > 1:
 178             # get the alpha occurring the most number of times
 179             lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]
 180
 181             # for modules with alpha different from the most occurring alpha, add it to the `alpha_pattern`
 182             alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
 183             if is_unet:
 184                 alpha_pattern = {
 185                     ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
 186                     for k, v in alpha_pattern.items()
 187                 }
 188             else:
 189                 alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
 190         else:
 191             lora_alpha = set(network_alpha_dict.values()).pop()
 192
 193     target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})
 194     use_dora = any("lora_magnitude_vector" in k for k in peft_state_dict)
 195     # for now we know that the "bias" keys are only associated with `lora_B`.
 196     lora_bias = any("lora_B" in k and k.endswith(".bias") for k in peft_state_dict)
 197
 198     lora_config_kwargs = {
 199         "r": r,
 200         "lora_alpha": lora_alpha,
 201         "rank_pattern": rank_pattern,
 202         "alpha_pattern": alpha_pattern,
 203         "target_modules": target_modules,
 204         "use_dora": use_dora,
 205         "lora_bias": lora_bias,
 206     }
 207
 208     return lora_config_kwargs
```

Two things follow, and the port copies both:

- **Alpha travels only in the metadata.** A file without it loads with alpha equal to each module's own rank, so
  scaling 1, even when the ranks are mixed.
- **Ranks and alphas can differ per module.** PEFT resolves them from `rank_pattern` and `alpha_pattern`:

📍 [`peft:src/peft/tuners/lora/model.py:308-312`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/model.py#L308-L312)
```python
 308         # Regexp matching - Find key which matches current target_name in patterns provided
 309         r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
 310         alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
 311         r = lora_config.rank_pattern.get(r_key, lora_config.r)
 312         alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)
```

  `get_pattern_key` (in PEFT's `utils/other.py`, which the sparse checkout here does not hold) returns the first
  pattern key that matches the module name as the regular expression `(.*\.)?(<key>)$`, or the name itself.
  `test_pattern_key_is_pefts` compares ours with the installed function on nine cases.

The pipeline mixin adds three more rules before any of that, which matter for files not written by diffusers:

📍 [`diffusers:src/diffusers/loaders/lora_pipeline.py:7204-7228`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_pipeline.py#L7204-L7228)
```python
7204         # Read before the conversion below, which folds per-module alphas into the weights and drops the scalars.
7205         has_alpha_tensors = any(k.endswith(".alpha") for k in state_dict)
7206
7207         is_dora_scale_present = any("dora_scale" in k for k in state_dict)
7208         if is_dora_scale_present:
7209             warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
7210             logger.warning(warn_msg)
7211             state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}
7212
7213         # The original checkpoint's module names, none of which a diffusers name shares. Checked before the peft dump
7214         # below, which DiffSynth-Studio's files would otherwise match: they carry `.default.` over these same names.
7215         is_non_diffusers_format = any(
7216             k.startswith(("diffusion_model.", "blocks.", "token_refiner.blocks.", "final_layer.", "lora_unet_"))
7217             for k in state_dict
7218         )
7219         # A peft dump: diffusers module names carrying peft's `.default.` infix and no component prefix. The missing
7220         # prefix is what keeps this from shadowing a file diffusers itself wrote.
7221         is_unprefixed_diffusers_format = any(".default.weight" in k for k in state_dict) and not any(
7222             k.startswith((f"{cls.transformer_name}.", f"{cls.transformer_ref_name}.")) for k in state_dict
7223         )
7224
7225         if is_non_diffusers_format:
7226             state_dict = _convert_non_diffusers_minimax_h3_lora_to_diffusers(state_dict)
7227         elif is_unprefixed_diffusers_format:
7228             state_dict = {f"{cls.transformer_name}.{k.replace('.default.', '.')}": v for k, v in state_dict.items()}
```
📍 [`diffusers:src/diffusers/loaders/lora_pipeline.py:7230-7249`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_pipeline.py#L7230-L7249)
```python
7230         # Published H3 LoRAs are alpha-less and mixed-rank (64 on attention and FFN, 16 on the AdaLN projections), and
7231         # `get_peft_kwargs` would scale one of the two rank groups by `alpha / r`, so `alpha == rank` is pinned below.
7232         #
7233         # An alpha-less file may instead record its training alpha in the `__metadata__`, honored as a uniform network
7234         # alpha (`alpha / rank` per module). Per-module scalars win, the fold above having already applied them.
7235         network_alpha = None
7236         if file_metadata is not None and "alpha" in file_metadata and not has_alpha_tensors:
7237             try:
7238                 network_alpha = float(file_metadata["alpha"])
7239             except ValueError:
7240                 logger.warning(
7241                     f"This LoRA file records `alpha` as {file_metadata['alpha']!r} in its `__metadata__`. MiniMax-H3"
7242                     " reads that entry as the network alpha every module was trained with, and it has to be a number,"
7243                     " so it is being ignored — the adapter is loaded with `alpha == rank` instead."
7244                 )
7245             else:
7246                 logger.info(
7247                     f"Using the network alpha {network_alpha} this LoRA file records in its `__metadata__`; every "
7248                     "module is scaled by `alpha / rank`."
7249                 )
```

Ours is one reader for all of it, and one writer:

📍 [`src/h3mlx/lora.py:611-617`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L611-L617)
```python
 611 def pattern_key(pattern_keys, key_to_match: str) -> str:
 612     """PEFT's `get_pattern_key`: the first `rank_pattern` / `alpha_pattern` key that matches the module name as a
 613     regular expression on its last dotted parts, else the name itself."""
 614     for key in pattern_keys:
 615         if re.match(rf"(.*\.)?({key})$", key_to_match):
 616             return key
 617     return key_to_match
```
📍 [`src/h3mlx/lora.py:620-635`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L620-L635)
```python
 620 def _component(tensors: dict, prefix: str | None) -> tuple[dict, str | None]:
 621     """The keys of one component with its prefix removed (`load_lora_adapter(prefix=...)`'s filter), and that prefix.
 622     With no `prefix` given the file must hold one component, prefixed or not."""
 623     if prefix == "":
 624         raise ValueError("an empty prefix names no component: pass None for a file without prefixes")
 625     found = [p for p in PREFIXES if any(k.startswith(p + ".") for k in tensors)]
 626     if prefix is None:
 627         if len(found) > 1:
 628             raise ValueError(f"this file holds LoRAs for {found}: say which with `prefix=`")
 629         prefix = found[0] if found else None
 630         other = () if prefix is None else [k for k in tensors if not k.startswith(prefix + ".")]
 631         if other:
 632             raise ValueError(f"keys with and without the `{prefix}.` prefix in one file, e.g. {sorted(other)[:3]}")
 633     if prefix is None:
 634         return dict(tensors), None
 635     return {k.removeprefix(prefix + "."): v for k, v in tensors.items() if k.startswith(prefix + ".")}, prefix
```
📍 [`src/h3mlx/lora.py:638-642`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L638-L642)
```python
 638 def _refuse_dora(keys) -> None:
 639     """DoRA is not implemented: refuse its keys (diffusers drops a kohya `dora_scale` with a warning, and loads PEFT's
 640     `lora_magnitude_vector` as DoRA)."""
 641     if any("dora_scale" in k or "lora_magnitude_vector" in k for k in keys):
 642         raise NotImplementedError("DoRA adapters are not supported")
```
📍 [`src/h3mlx/lora.py:645-720`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L645-L720)
```python
 645 def read(tensors: dict, header: dict | None = None, prefix: str | None = None, head_dim: int = HEAD_DIM) -> Adapter:
 646     """An adapter from a LoRA file's tensors and its safetensors header (`__metadata__`), as diffusers would load it.
 647
 648     Keys: `transformer.<module>.lora_A/B.weight` (what `save_lora_weights` writes and `load_lora_adapter` reads by
 649     default), the same without the prefix (what `save_lora_adapter` writes), PEFT's own state dict (a `.default.`
 650     infix), or the original checkpoint's names (`convert_original`, which folds any `.alpha` tensors into the matrices;
 651     `head_dim` is passed to it).
 652
 653     `prefix` picks one component's keys from a file that holds several, as `load_lora_adapter(prefix=...)` does; other
 654     components' keys are then not looked at. Without it the file must hold one component. An original-layout file
 655     names no component and is read whole whatever `prefix` says: the converter labels it `transformer.`, and the
 656     mixin loads those layers into either partition (`load_into_transformer_ref`). The adapter itself belongs to no
 657     partition; the caller attaches it to the weights it means.
 658
 659     Alpha, per module: from the `lora_adapter_metadata` diffusers writes (`r`, `lora_alpha`, `rank_pattern`,
 660     `alpha_pattern`, resolved as PEFT resolves them); else from a plain `alpha` entry in the header, taken as one
 661     network alpha for every module; else the module's own rank (scaling 1), which is `get_peft_kwargs`' rule. DoRA,
 662     rsLoRA, a LoRA bias, a `bias` setting other than none, `fan_in_fan_out` and an alpha that is not positive and
 663     finite are refused, not ignored."""
 664     header = dict(header or {})
 665     tensors = dict(tensors)
 666     has_alpha_tensors = any(k.endswith(".alpha") for k in tensors)
 667     if is_original_layout(tensors):
 668         _refuse_dora(tensors)  # before the converter, which would call a `dora_scale` key a leftover
 669         tensors = convert_original(tensors, head_dim)
 670         prefix = "transformer"  # the converter's label, not a choice of partition
 671     elif any(".default.weight" in k for k in tensors) and not any(k.startswith(tuple(p + "." for p in PREFIXES)) for k in tensors):
 672         tensors = {k.replace(".default.", "."): v for k, v in tensors.items()}  # a PEFT state dict, adapter "default"
 673     tensors, prefix = _component(tensors, prefix)
 674     _refuse_dora(tensors)  # of the component read, not of the others
 675
 676     by_module = {}
 677     for key in tensors:
 678         m = KEY_RE.match(key)
 679         if m is None:
 680             raise ValueError(f"unexpected key {key!r}: only <module>.lora_A/B.weight keys are supported")
 681         by_module.setdefault(m["module"], set()).add(m["matrix"])
 682     incomplete = sorted(k for k, v in by_module.items() if v != {"A", "B"})
 683     if incomplete:
 684         raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
 685     if not by_module:
 686         raise ValueError("no LoRA tensors" + (f" under the `{prefix}.` prefix" if prefix else "") + " in this file")
 687     ranks = {m: int(tensors[m + ".lora_B.weight"].shape[1]) for m in by_module}
 688
 689     config = json.loads(header[METADATA_KEY]) if header.get(METADATA_KEY) else None
 690     if config is not None and not isinstance(config, dict):
 691         raise ValueError(f"{METADATA_KEY} is not a JSON object (got a {type(config).__name__})")
 692     if config is not None and prefix is not None:
 693         config = {k.removeprefix(prefix + "."): v for k, v in config.items() if k.startswith(prefix + ".")}
 694     if config is not None:
 695         if "r" not in config or "lora_alpha" not in config:
 696             raise ValueError(f"the file's {METADATA_KEY} has no `r` / `lora_alpha`" + (f" under `{prefix}.`" if prefix else ""))
 697         on = [k for k in UNSUPPORTED if config.get(k)]
 698         if config.get("bias") not in (None, "none"):  # PEFT then trains and saves base-layer biases as well
 699             on.append(f"bias={config['bias']!r}")
 700         if on:
 701             raise NotImplementedError(f"this adapter was saved with {on}, which the port does not implement")
 702         rank_pattern, alpha_pattern = config.get("rank_pattern") or {}, config.get("alpha_pattern") or {}
 703         alpha = {}
 704         for module, rank in ranks.items():
 705             r = rank_pattern.get(pattern_key(rank_pattern, module), config["r"])
 706             if r != rank:
 707                 raise ValueError(f"{module}: the metadata says rank {r} but lora_B has rank {rank}")
 708             alpha[module] = alpha_pattern.get(pattern_key(alpha_pattern, module), config["lora_alpha"])
 709             check_alpha(f"{module} (the file's {METADATA_KEY})", alpha[module])
 710         return Adapter(tensors, alpha)
 711
 712     network_alpha = None
 713     if "alpha" in header and not has_alpha_tensors:
 714         try:
 715             network_alpha = float(header["alpha"])
 716         except ValueError:  # diffusers warns and loads with alpha == rank
 717             network_alpha = None
 718         else:
 719             check_alpha("the file header's `alpha`", network_alpha)
 720     return Adapter(tensors, {m: (rank if network_alpha is None else network_alpha) for m, rank in ranks.items()})
```
📍 [`src/h3mlx/lora.py:843-864`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L843-L864)
```python
 843 def config(adapter: Adapter) -> dict:
 844     """The `LoraConfig` arguments that describe the adapter, as `get_peft_kwargs` derives them: the most common rank and
 845     alpha, and a `^`-anchored pattern entry for each module that differs. The other entries state what the port
 846     implements (no dropout, DoRA, rsLoRA or LoRA bias). A subset of what `LoraConfig.to_dict()` holds, so it loads
 847     with `LoraConfig(**config)` on PEFT versions that add or drop other fields."""
 848     ranks = {m: adapter.rank(m) for m in adapter.modules}
 849     alphas = {m: _number(adapter.alpha[m]) for m in adapter.modules}
 850     r = Counter(ranks.values()).most_common()[0][0]
 851     lora_alpha = Counter(alphas.values()).most_common()[0][0]
 852     return {
 853         "r": r,
 854         "lora_alpha": lora_alpha,
 855         "rank_pattern": {f"^{m}": v for m, v in ranks.items() if v != r},
 856         "alpha_pattern": {f"^{m}": v for m, v in alphas.items() if v != lora_alpha},
 857         "target_modules": adapter.modules,
 858         "bias": "none",
 859         "lora_dropout": 0.0,
 860         "use_dora": False,
 861         "use_rslora": False,
 862         "lora_bias": False,
 863         "fan_in_fan_out": False,
 864     }
```
📍 [`src/h3mlx/lora.py:867-885`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L867-L885)
```python
 867 def save(path, adapter: Adapter, prefix: str | None = "transformer") -> None:
 868     """Write the adapter as diffusers does: `<prefix>.<module>.lora_A/B.weight` tensors and, in the safetensors header,
 869     `format: pt` plus the `LoraConfig` arguments as JSON under `lora_adapter_metadata` (each key prefixed too). That is
 870     `save_lora_weights`' file, which `load_lora_adapter` reads by default. `prefix=None` writes `save_lora_adapter`'s:
 871     no prefixes, read back with `load_lora_adapter(..., prefix=None)`. Tensors keep their dtype. Nothing is written
 872     for an adapter whose tensors do not match its modules, or with an alpha that is not positive and finite."""
 873     path = Path(path)
 874     if path.suffix != ".safetensors":
 875         raise ValueError(f"expected a .safetensors path, got {path}")
 876     if prefix == "":
 877         raise ValueError("an empty prefix would put a leading dot on every key: pass None for no prefix")
 878     check_keys(adapter)
 879     for module in adapter.modules:
 880         check_alpha(module, adapter.alpha[module])
 881     lead = "" if prefix is None else prefix + "."
 882     tensors = {lead + k: v for k, v in adapter.tensors.items()}
 883     meta = {lead + k: v for k, v in config(adapter).items()}
 884     described = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False)  # strict JSON: no NaN or Infinity tokens
 885     mx.save_safetensors(str(path), tensors, metadata={"format": "pt", METADATA_KEY: described})
```

What differs from the original, on purpose:

- **The metadata we write is a subset.** `config` writes the eleven `LoraConfig` arguments that describe the adapter
  and state what the port implements. PEFT 0.21.2's `to_dict()` has 41 entries, and `LoraConfig(**metadata)` raises on
  one it does not know, so a full dump from one PEFT version can fail to load in another.
  `test_config_is_a_subset_of_pefts` checks every entry we write equals PEFT's for the same adapter, and that PEFT
  builds the same config back from ours.
- **Refused, not ignored:** DoRA (diffusers drops a kohya `dora_scale` with a warning and loads the rest as a plain
  LoRA, and loads PEFT's `lora_magnitude_vector` as DoRA), rsLoRA, a LoRA bias, `fan_in_fan_out`, any key that is not
  a `lora_A` / `lora_B` weight, half a pair, a rank in the metadata that is not the tensors', and a file holding both
  `transformer.` and `transformer_ref.` adapters unless told which.
- **A `bias` setting other than none is refused.** With `bias: all` or `lora_only` PEFT also trains base-layer biases
  and writes them with the adapter, so such metadata describes more than the two matrices the port reads. diffusers
  loads it without a word (`test_a_bias_setting_other_than_none_is_refused`).
- **An empty prefix is refused.** `prefix=""` is not `prefix=None`: diffusers takes it as a prefix of its own, finds no
  key that starts with a dot, and loads nothing; written that way, every key gets a leading dot
  (`test_an_empty_prefix_is_refused`).
- **An alpha that is not positive and finite is refused,** from the metadata's `lora_alpha` and `alpha_pattern` and
  from a header `alpha` alike (§1.3). diffusers takes `float("nan")` from a header and hands it to PEFT.
- **The metadata is strict JSON.** Python's `json.dumps` writes `NaN` and `Infinity` for such numbers, which is what
  diffusers' writer would put in the header for a config holding one, and which strict JSON readers refuse. `save`
  checks the alphas first and writes with `allow_nan=False` (`test_saved_metadata_is_strict_json`).
- **Not read:** PEFT's own folder format (`adapter_config.json` beside `adapter_model.safetensors`), and the older
  diffusers `lora.down` / `lora.up` key style.

How `prefix` works, which is diffusers' filter:

- **A prefix picks one component.** With `prefix="transformer"` only that component's keys are read; another
  component's keys in the same file, DoRA or not, are not looked at, as in `load_lora_adapter(prefix=...)`
  (`test_prefix_reads_one_component_and_ignores_the_others`). Without a prefix the file must hold one component.
- **An original-layout file names no component** and is read whole whatever `prefix` says (§2). The adapter that
  comes back belongs to no partition: the caller attaches it to the weights it means.
- **Tensors keep their dtype in the file.** A bf16 or float16 adapter is written and read back as such
  (`test_half_precision_adapter_keeps_its_dtype_in_the_file`, which also reads the file with safetensors and
  diffusers).

The file tests go both ways. diffusers writes and we read: `test_diffusers_file_loads_in_mlx` (both layouts, two
(rank, alpha) pairs) and `test_rank_and_alpha_patterns_from_diffusers`. We write and diffusers reads:
`test_mlx_file_loads_in_diffusers` (both layouts) and `test_mixed_ranks_and_alphas_to_diffusers`, which also check
that every PEFT layer ends up with our rank, alpha and scaling. Each compares the tiny transformer's forward and has
guards: the adapter must move the output, and so must its scaling.

## 2. Adapters in the original checkpoint's key names

LoRAs trained outside diffusers name the original checkpoint's modules, which differ from diffusers' in names and in
two layouts: the original fuses q, k and v into one projection, and orders the two halves of the feed-forward's first
projection the other way round. diffusers converts such a file on loading. Its docstring lists the producers it
knows:

📍 [`diffusers:src/diffusers/loaders/lora_conversion_utils.py:3342-3360`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_conversion_utils.py#L3342-L3360)
```python
3342 def _convert_non_diffusers_minimax_h3_lora_to_diffusers(state_dict):
3343     """Convert a non-diffusers MiniMax-H3 LoRA state dict onto `MiniMaxH3Transformer3DModel`'s module names.
3344
3345     Every known producer trains against the original checkpoint's module names — ai-toolkit under a `diffusion_model.`
3346     prefix, the reference `generate.py` / ComfyUI checkpoints under no prefix at all, musubi-tuner under a flattened
3347     `lora_unet_` one, DiffSynth-Studio under no prefix but with peft's `.default.` infix — so the prefix is optional
3348     and the module names are what identifies the format. Handles:
3349
3350     - `diffusion_model.` prefix removal, and bare `blocks.` / `token_refiner.` / `final_layer.` keys
3351     - musubi-tuner's flattened `lora_unet_blocks_0_attn_qkv_proj` -> `blocks.0.attn.qkv_proj`
3352     - `lora_down`/`lora_up` (kohya) -> `lora_A`/`lora_B`, with `.alpha` folded into the weights
3353     - fused `attn.qkv_proj` -> split `to_q`/`to_k`/`to_v`; `attn.out_proj` -> `to_out.0`
3354     - `mlp.fc1` -> `ff.net.0.proj` with its two output halves swapped, `mlp.fc2` -> `ff.net.2`
3355     - `blocks.` -> `transformer_blocks.`, `token_refiner.blocks.` -> `token_refiner.refiner_blocks.`, and the
3356       `final_layer.` / patch / condition / timestep projections onto their diffusers names
3357
3358     The result is prefixed with `transformer.`, the partition every published H3 LoRA is trained against;
3359     `MiniMaxH3LoraLoaderMixin.load_lora_weights` is what redirects it to `transformer_ref` when asked.
3360     """
```

`lora.convert_original` is that function, branch for branch. `lora.read` calls it when a key carries one of the
original names (`is_original_layout`, the mixin's test in §1.5).

Such a file names no component. The converter labels every key `transformer.`, and the mixin then loads those layers
into whichever partition the caller asks for:

📍 [`diffusers:src/diffusers/loaders/lora_pipeline.py:7416-7444`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_pipeline.py#L7416-L7444)
```python
7416     @classmethod
7417     def load_lora_into_transformer_ref(
7418         cls,
7419         state_dict,
7420         transformer_ref,
7421         prefix,
7422         adapter_name=None,
7423         _pipeline=None,
7424         low_cpu_mem_usage=False,
7425         hotswap: bool = False,
7426         metadata=None,
7427     ):
7428         """
7429         Load LoRA layers into the `transformer_ref` partition. `prefix` is the component name the keys carry, which is
7430         `transformer_ref` for a file that names the partition and `transformer` for one routed here by
7431         `load_into_transformer_ref=True`. See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more
7432         details.
7433         """
7434         logger.info(f"Loading {prefix}.")
7435         transformer_ref.load_lora_adapter(
7436             state_dict,
7437             prefix=prefix,
7438             network_alphas=None,
7439             adapter_name=adapter_name,
7440             metadata=metadata,
7441             _pipeline=_pipeline,
7442             low_cpu_mem_usage=low_cpu_mem_usage,
7443             hotswap=hotswap,
7444         )
```

`read` does the same from the other end: an original-layout file is read whole whatever `prefix` says, and the caller
attaches the adapter to the weights it means. (A plain filter by `transformer_ref.` finds nothing in a converted file,
in diffusers' `load_lora_adapter` too.) `test_original_layout_file_is_read_whole_for_either_partition` checks the three
prefixes against the mixin's two loaders.

### 2.1 Prefixes and flattened names

📍 [`diffusers:src/diffusers/loaders/lora_conversion_utils.py:3361-3403`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_conversion_utils.py#L3361-L3403)
```python
3361     state_dict = {k.removeprefix("diffusion_model."): v for k, v in state_dict.items()}
3362
3363     # DiffSynth-Studio runs the raw checkpoint's per-head-interleaved fused QKV verbatim, where every other producer
3364     # reorders it to `[q_all; k_all; v_all]` at load time as the reference implementation does. The two are
3365     # indistinguishable by shape; what identifies DiffSynth is that it writes peft's `.default.` infix over these names.
3366     is_diffsynth = any(".lora_A.default.weight" in k or ".lora_B.default.weight" in k for k in state_dict)
3367     if is_diffsynth:
3368         state_dict = {
3369             k.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."): v
3370             for k, v in state_dict.items()
3371         }
3372
3373     # musubi-tuner (kohya sd-scripts) writes one flat module name per key under a `lora_unet_` prefix, with every `.`
3374     # collapsed to `_`. H3's own module names contain underscores (`qkv_proj`, `token_refiner`, `final_layer`,
3375     # `video_out`), so the dot path is recovered by matching the whole flattened name against the module vocabulary
3376     # rather than by splitting on `_`.
3377     if any(k.startswith("lora_unet_") for k in state_dict):
3378         flattened_modules = [
3379             (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
3380             (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
3381             (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
3382             (r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj", r"token_refiner.blocks.\1.attn.\2_proj"),
3383             (r"token_refiner_blocks_(\d+)_mlp_fc([12])", r"token_refiner.blocks.\1.mlp.fc\2"),
3384             (r"(video|audio)_patch_proj", r"\1_patch_proj"),
3385             (r"condition_proj", "condition_proj"),
3386             (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
3387             (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
3388             (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
3389         ]
3390         unflattened = {}
3391         for key, value in state_dict.items():
3392             module, _, suffix = key.removeprefix("lora_unet_").partition(".")
3393             dotted = None
3394             for pattern, replacement in flattened_modules:
3395                 if re.fullmatch(pattern, module):
3396                     dotted = re.sub(pattern, replacement, module)
3397                     break
3398             if dotted is None:
3399                 raise ValueError(
3400                     f"`{key}` does not name a MiniMax-H3 module in musubi-tuner's flattened `lora_unet_` layout."
3401                 )
3402             unflattened[f"{dotted}.{suffix}"] = value
3403         state_dict = unflattened
```

📍 [`src/h3mlx/lora.py:445-456`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L445-L456)
```python
 445 _FLATTENED = [  # musubi-tuner's `lora_unet_` names, every `.` collapsed to `_`
 446     (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
 447     (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
 448     (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
 449     (r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj", r"token_refiner.blocks.\1.attn.\2_proj"),
 450     (r"token_refiner_blocks_(\d+)_mlp_fc([12])", r"token_refiner.blocks.\1.mlp.fc\2"),
 451     (r"(video|audio)_patch_proj", r"\1_patch_proj"),
 452     (r"condition_proj", "condition_proj"),
 453     (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
 454     (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
 455     (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
 456 ]
```

Ours is the same sequence: drop `diffusion_model.`, note and strip DiffSynth's `.default.` infix, and match
musubi-tuner's flattened names against the same ten patterns.

### 2.2 The alpha fold

📍 [`diffusers:src/diffusers/loaders/lora_conversion_utils.py:3405-3424`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_conversion_utils.py#L3405-L3424)
```python
3405     is_kohya = any(".lora_down.weight" in k for k in state_dict)
3406     down_suffix = ".lora_down.weight" if is_kohya else ".lora_A.weight"
3407     up_suffix = ".lora_up.weight" if is_kohya else ".lora_B.weight"
3408
3409     def pull(base):
3410         """Pop the (lora_A, lora_B) pair for a module path with any `.alpha` folded in, or None if absent."""
3411         down_key = base + down_suffix
3412         if down_key not in state_dict:
3413             return None
3414         down = state_dict.pop(down_key)
3415         up = state_dict.pop(base + up_suffix)
3416         alpha = state_dict.pop(base + ".alpha", None)
3417         if alpha is not None:
3418             # LoRA is scaled by `alpha / rank` in the forward pass; split the factor between down and up.
3419             scale_down, scale_up = alpha.item() / down.shape[0], 1.0
3420             while scale_down * 2 < scale_up:
3421                 scale_down *= 2
3422                 scale_up /= 2
3423             down, up = down * scale_down, up * scale_up
3424         return down, up
```

A kohya-style file stores an `.alpha` scalar beside each pair. The converter multiplies `alpha / rank` into the two
matrices, balancing the factor between them, and drops the scalar. The result therefore has scaling 1.

📍 [`src/h3mlx/lora.py:533-542`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L533-L542)
```python
 533         alpha = sd.pop(base + ".alpha", None)
 534         if alpha is not None:
 535             if not alpha.item() > 0 or math.isinf(alpha.item()):
 536                 raise ValueError(f"`{base}.alpha` is {alpha.item()}: a LoRA alpha must be positive and finite")
 537             # the forward scales by alpha / rank; split that factor between down and up
 538             scale_down, scale_up = alpha.item() / down.shape[0], 1.0
 539             while scale_down * 2 < scale_up:
 540                 scale_down *= 2
 541                 scale_up /= 2
 542             down, up = ops.times_scalar(down, scale_down), ops.times_scalar(up, scale_up)
```

- **The products round as torch's.** The fold multiplies each matrix by a Python float, so it uses `times_scalar`
  (§1.1). With bf16 or float16 matrices and a rank that is not a power of two, MLX's plain product gives other bits;
  `test_converts_to_diffusers_dicts` runs rank 6 for that reason.
- **A bad alpha is refused. This differs from diffusers on purpose.** Their loop halves `scale_up` while
  `scale_down * 2 < scale_up`, with no check on alpha. With alpha 0 it runs until `scale_up` underflows, 1,075 turns,
  and both matrices come out zero. With a negative alpha `scale_down` falls to −∞, `scale_up` to 0, and the condition
  stays true: it never returns. A NaN skips the loop and writes NaNs. Ours raises on an alpha that is zero, negative
  or not finite. `test_bad_alpha_is_refused` runs their converter on alpha 0 and shows the zeros; it does not run it
  on a negative alpha, and uses a transcription of the loop with a cap on its turns instead.
- **The alpha is applied once.** A converted adapter has no metadata, so `read` gives every module alpha equal to its
  rank. A plain `alpha` entry in the file's header is one network alpha for a file with no `.alpha` tensors, and is
  ignored when the file has them, as in the mixin (§1.5). `test_converted_adapter_gives_the_same_forward` checks the
  scaling is 1 and that no `lora_scaling` key is attached, and that applying `alpha / rank` again would move the
  output; `test_header_alpha_with_a_converted_file` checks both header cases against diffusers.

### 2.3 The fused QKV projection

📍 [`diffusers:src/diffusers/loaders/lora_conversion_utils.py:3464-3495`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_conversion_utils.py#L3464-L3495)
```python
3464             # Fused qkv -> split to_q / to_k / to_v (shared down/lora_A, chunk up/lora_B in thirds), after the
3465             # per-head de-interleave a DiffSynth file needs.
3466             qkv = pull(f"{source}.attn.qkv_proj")
3467             if qkv is not None:
3468                 down, up = qkv
3469                 if up.shape[0] % 3 != 0:
3470                     raise ValueError(
3471                         f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not divisible by 3. "
3472                         "This is not a fused MiniMax-H3 QKV projection."
3473                     )
3474                 if is_diffsynth:
3475                     # `[head0: q k v, head1: q k v, ...]` -> `[q_all; k_all; v_all]`. Every released MiniMax-H3
3476                     # partition has a 128-wide head, which is what lets the head count be read off the row count.
3477                     head_dim = 128
3478                     if up.shape[0] % (3 * head_dim) != 0:
3479                         raise ValueError(
3480                             f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not a multiple of "
3481                             f"3 * {head_dim}. This is not a per-head-interleaved MiniMax-H3 QKV projection."
3482                         )
3483                     num_heads = up.shape[0] // (3 * head_dim)
3484                     grouped = up.reshape(num_heads, 3 * head_dim, up.shape[1])
3485                     up = torch.cat(
3486                         [
3487                             head_slice.reshape(num_heads * head_dim, up.shape[1])
3488                             for head_slice in grouped.split(head_dim, dim=1)
3489                         ],
3490                         dim=0,
3491                     )
3492                 up_q, up_k, up_v = torch.chunk(up, 3, dim=0)
3493                 for proj, up_proj in (("to_q", up_q), ("to_k", up_k), ("to_v", up_v)):
3494                     converted_state_dict[f"{target}.attn.{proj}.lora_A.weight"] = down.clone()
3495                     converted_state_dict[f"{target}.attn.{proj}.lora_B.weight"] = up_proj.contiguous()
```

📍 [`src/h3mlx/lora.py:563-581`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L563-L581)
```python
 563             qkv = pull(f"{source}.attn.qkv_proj")
 564             if qkv is not None:
 565                 down, up = qkv
 566                 if up.shape[0] % 3 != 0:
 567                     raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not divisible "
 568                                      "by 3. This is not a fused MiniMax-H3 QKV projection.")
 569                 if is_diffsynth:
 570                     # [head0: q k v, head1: q k v, ...] -> [q_all; k_all; v_all]
 571                     if up.shape[0] % (3 * head_dim) != 0:
 572                         raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not a "
 573                                          f"multiple of 3 * {head_dim}. This is not a per-head-interleaved MiniMax-H3 "
 574                                          "QKV projection.")
 575                     num_heads = up.shape[0] // (3 * head_dim)
 576                     grouped = up.reshape(num_heads, 3 * head_dim, up.shape[1])
 577                     up = mx.concatenate([s.reshape(num_heads * head_dim, up.shape[1])
 578                                          for s in mx.split(grouped, 3, axis=1)], axis=0)
 579                 for proj, up_proj in zip(("to_q", "to_k", "to_v"), mx.split(up, 3, axis=0)):
 580                     out[f"{target}.attn.{proj}.lora_A.weight"] = down
 581                     out[f"{target}.attn.{proj}.lora_B.weight"] = up_proj
```

`lora_A` is shared by the three projections and `lora_B` is cut in thirds. A DiffSynth-Studio file keeps the raw
checkpoint's row order, q, k and v interleaved per head, and is reordered first.

- **The head width is an argument. This differs from diffusers on purpose.** Theirs is fixed at 128, the width of
  every released partition, and reads the head count off the row count. Ours takes `head_dim`, 128 by default, so a
  model with another width can be converted by passing its config's `attention_head_dim`.
  `test_diffsynth_rows_by_head_width` ties an index formula to their result at 128, then checks ours against the same
  formula at the tiny model's width of 32, where theirs refuses the file.

### 2.4 The feed-forward halves, and the rest

📍 [`diffusers:src/diffusers/loaders/lora_conversion_utils.py:3497-3532`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_conversion_utils.py#L3497-L3532)
```python
3497             # `fc1` stays fused, as diffusers' `SwiGLU` also fuses its two projections, but the reference computes
3498             # `fc2(silu(gate) * value)` from a fused `[gate; value]` while `SwiGLU` computes `value * silu(gate)` from
3499             # a fused `[value; gate]`, so the two halves swap places. `lora_A` is untouched: the swap is a permutation
3500             # of output rows, so it applies to `lora_B` alone.
3501             fc1 = pull(f"{source}.mlp.fc1")
3502             if fc1 is not None:
3503                 down, up = fc1
3504                 if up.shape[0] % 2 != 0:
3505                     raise ValueError(
3506                         f"`{source}.mlp.fc1` has {up.shape[0]} output rows, which is not even. This is not a fused "
3507                         "MiniMax-H3 SwiGLU projection."
3508                     )
3509                 up_gate, up_value = up.chunk(2, dim=0)
3510                 converted_state_dict[f"{target}.ff.net.0.proj.lora_A.weight"] = down
3511                 converted_state_dict[f"{target}.ff.net.0.proj.lora_B.weight"] = torch.cat(
3512                     [up_value, up_gate], dim=0
3513                 ).contiguous()
3514
3515             for source_module, target_module in (
3516                 ("attn.out_proj", "attn.to_out.0"),
3517                 ("mlp.fc2", "ff.net.2"),
3518                 ("adaln_proj.linear", "adaln_proj.linear"),
3519             ):
3520                 pair = pull(f"{source}.{source_module}")
3521                 if pair is not None:
3522                     down, up = pair
3523                     converted_state_dict[f"{target}.{target_module}.lora_A.weight"] = down
3524                     converted_state_dict[f"{target}.{target_module}.lora_B.weight"] = up
3525
3526     if len(state_dict) > 0:
3527         raise ValueError(
3528             f"`state_dict` should be empty at this point but has {sorted(state_dict.keys())}. "
3529             "This may be an unsupported MiniMax-H3 LoRA layout."
3530         )
3531
3532     return {f"transformer.{k}": v for k, v in converted_state_dict.items()}
```

📍 [`src/h3mlx/lora.py:585-593`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L585-L593)
```python
 585             fc1 = pull(f"{source}.mlp.fc1")
 586             if fc1 is not None:
 587                 down, up = fc1
 588                 if up.shape[0] % 2 != 0:
 589                     raise ValueError(f"`{source}.mlp.fc1` has {up.shape[0]} output rows, which is not even. This is "
 590                                      "not a fused MiniMax-H3 SwiGLU projection.")
 591                 up_gate, up_value = mx.split(up, 2, axis=0)
 592                 out[f"{target}.ff.net.0.proj.lora_A.weight"] = down
 593                 out[f"{target}.ff.net.0.proj.lora_B.weight"] = mx.concatenate([up_value, up_gate], axis=0)
```
📍 [`src/h3mlx/lora.py:457-466`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L457-L466)
```python
 457 _STANDALONE = {  # the projections outside the block stack
 458     "video_patch_proj": "proj_in",
 459     "audio_patch_proj": "audio_proj_in",
 460     "condition_proj": "context_embedder",
 461     "time_embedder.proj_in": "time_embedder.linear_1",
 462     "time_embedder.proj_out": "time_embedder.linear_2",
 463     "final_layer.adaln_proj.linear": "norm_out.linear",
 464     "final_layer.video_out": "proj_out",
 465     "final_layer.audio_out": "audio_proj_out",
 466 }
```

The original computes `fc2(silu(gate) * value)` from a fused `[gate; value]`, diffusers' SwiGLU `value * silu(gate)`
from `[value; gate]`, so `lora_B`'s two halves swap and `lora_A` is untouched. Leftover keys are an error on both
sides.

- **Half a pair names its module. This differs from diffusers on purpose.** A module with its first matrix and no
  second makes their `pull` pop a key that is not there, a bare KeyError. Ours raises a ValueError that says which
  module and which matrix (`test_half_a_pair_names_the_module`). The second matrix alone is a leftover key, refused on
  both sides.
- **A kohya DoRA file is refused. This differs from diffusers on purpose.** Their mixin drops the `dora_scale` keys
  with a warning and loads what is left as a plain LoRA, which is not the adapter that was trained
  (`test_dora_scale_in_an_original_file_is_refused`).

- **Every converted module attaches.** The converter also maps LoRAs on the AdaLN projections and on the patch,
  condition and final projections, as theirs does, and diffusers loads each onto its `Linear`. So does the port
  (§1.3). `test_every_converted_module_attaches_and_gives_the_same_forward` converts an adapter on all 28 linears of
  the tiny model and compares the forward with diffusers', then leaves out each group of those modules in turn and
  checks the output moves.

### 2.5 The tests

`tests/test_lora_convert.py` writes random adapters in five layouts: `diffusion_model.`-prefixed, bare, kohya
(`lora_down` / `lora_up` / `.alpha`), musubi-tuner's flattened names, and DiffSynth's.

| test | pins |
| --- | --- |
| `test_converts_to_diffusers_dicts` (5 layouts × 3 dtypes × 2 ranks) | every module the converter knows: the same keys in the same order and the same tensors as diffusers', bit for bit, in float32, bf16 and float16 |
| `test_what_the_conversion_does` | without diffusers: q, k and v share `lora_A`; `lora_B` in thirds; the swapped halves; `B·A` scaled by `alpha / rank` |
| `test_diffsynth_rows_by_head_width` (×2) | the de-interleave at head width 128 (theirs) and 32 (ours, by formula) |
| `test_converted_adapter_gives_the_same_forward` (×5) | the tiny transformer's forward against the mixin's `lora_state_dict` + `load_lora_adapter`; guards per layout |
| `test_header_alpha_with_a_converted_file` (×2) | a header `alpha` used, and ignored beside `.alpha` tensors |
| `test_original_layout_file_is_read_whole_for_either_partition` (×3) | no prefix, `transformer` and `transformer_ref`: the same adapter, and the forward the mixin's loader for that partition gives |
| `test_every_converted_module_attaches_and_gives_the_same_forward` | an adapter on every `Linear`, the AdaLN and the input and output projections included: diffusers' forward, and each group counts |
| `test_refusals_shared_with_diffusers`, `test_bad_alpha_is_refused` (×4), `test_half_a_pair_names_the_module` (×2), `test_dora_scale_in_an_original_file_is_refused`, `test_layout_detection` | the errors both sides raise; the three ours raises where theirs loops, raises a bare KeyError or drops keys; which keys count as original |

## 3. The forward process, the target and the loss

The scheduler works in `t = 1 − sigma`, with `t = 1` clean. Its own notes state the sign:

📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:20-24`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/schedulers/scheduling_minimax_h3.py#L20-L24)
```python
  20 1. **The velocity sign is reversed.** MiniMax-H3's transformer predicts a *data-ward* velocity, so `x0 = x_t + sigma *
  21    v` instead of diffusers' `x0 = x_t - sigma * v`.
  22 2. **Timesteps are `t = 1 - sigma` in `[0, 1]`**, with `t = 1` meaning *clean*. Flow-match schedulers expose `timesteps
  23    = sigma * num_train_timesteps`, i.e. the opposite direction on a 1000x scale. The transformer's AdaLN consumes the
  24    H3 convention directly.
```

### 3.1 The forward process

📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:226-256`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/schedulers/scheduling_minimax_h3.py#L226-L256)
```python
 226     def scale_noise(
 227         self,
 228         sample: torch.FloatTensor,
 229         timestep: float | torch.FloatTensor,
 230         noise: torch.FloatTensor,
 231     ) -> torch.FloatTensor:
 232         r"""
 233         Rectified-flow forward process, in MiniMax-H3's `t` convention: `x_t = t*x_0 + (1 - t)*noise`.
 234
 235         MiniMax-H3 uses this to noise its conditioning anchors, where `t` is the `noise_aug` level rather than a
 236         schedule entry, so `timestep` is taken at face value and is *not* looked up in `self.timesteps`.
 237
 238         Args:
 239             sample (`torch.FloatTensor`):
 240                 The clean sample `x_0`.
 241             timestep (`float` or `torch.FloatTensor`):
 242                 The target time in `[0, 1]`; `1` returns `sample` unchanged.
 243             noise (`torch.FloatTensor`):
 244                 The noise to mix in.
 245
 246         Returns:
 247             `torch.FloatTensor`: The noised sample.
 248         """
 249         if not isinstance(timestep, torch.Tensor):
 250             # A Python float keeps `1 - t` at double precision: `1 - 0.999` evaluated in float32 comes out 1.3e-5
 251             # relative too small, which the bfloat16 forward amplifies into a visibly different sample.
 252             return timestep * sample + (1.0 - timestep) * noise
 253         timestep = timestep.to(device=sample.device, dtype=sample.dtype)
 254         while timestep.ndim < sample.ndim:
 255             timestep = timestep.unsqueeze(-1)
 256         return timestep * sample + (1.0 - timestep) * noise
```

📍 [`src/h3mlx/train.py:30-54`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L30-L54)
```python
  30 def scale_noise(sample: mx.array, timestep, noise: mx.array) -> mx.array:
  31     """`MiniMaxH3Scheduler.scale_noise`: `x_t = t * x_0 + (1 - t) * noise`, `t` in [0, 1] and 1 clean.
  32
  33     The timestep's kind decides the arithmetic, as in the scheduler:
  34
  35     * A number (a Python float or int, or a numpy scalar) takes `1 - t` in its own precision before it meets the
  36       noise: double for a Python float (in float32, `1 - 0.999` comes out 1.3e-5 relative too small), float32 for an
  37       `np.float32`. Each product is then tensor-times-number.
  38     * An `mx.array` is cast to the sample's dtype and lined up with its leading axes (a 0-d `t`, or one per example),
  39       and `1 - t` is taken in that dtype, which is how the pipelines noise their conditioning rows.
  40     * A numpy array is a TypeError, as it is for the scheduler: wrap it, `mx.array(timesteps)`, to have it treated as
  41       a tensor (the timesteps `1 - sigma` from `sample_sigmas` are numpy).
  42
  43     One difference, on purpose: a timestep with more axes than the sample is refused. The scheduler broadcasts it into
  44     a result larger than the sample."""
  45     if isinstance(timestep, np.ndarray):
  46         raise TypeError("a numpy array of timesteps is not accepted (nor by the scheduler): pass a number, or "
  47                         "mx.array(timesteps) to have them treated as a tensor")
  48     if not isinstance(timestep, mx.array):
  49         return ops.times_scalar(sample, float(timestep)) + ops.times_scalar(noise, float(1.0 - timestep))
  50     if timestep.ndim > sample.ndim:
  51         raise ValueError(f"a timestep of shape {timestep.shape} has more axes than the sample {sample.shape}")
  52     t = timestep.astype(sample.dtype)
  53     t = t.reshape(t.shape + (1,) * (sample.ndim - t.ndim))
  54     return t * sample + (1.0 - t) * noise
```

The two branches are different arithmetic, and the port keeps both:

- **A number** keeps `1 − t` in its own precision until it meets the noise: double for a Python float, float32 for
  an `np.float32`. A numpy scalar is not a tensor to the scheduler, so it takes this branch there and here. In half
  precision each product is torch's `tensor * python_float`, so `times_scalar` again.
- **A tensor** (an `mx.array` here) is cast to the sample's dtype first, so `1 − t` is taken in float32 (or bf16).
  This is the expression `image.generate` already noises its pictures with;
  `test_float_and_array_timesteps_differ_as_in_the_scheduler` checks that, and that the two branches really differ.
- **The same value can give two results.** On a bf16 sample, `np.float32(0.999)` multiplies in float32, while a 0-d
  tensor of 0.999 is first cast to bf16, where it is 1.0 and `1 − t` is 0. The scheduler does both, and so does the
  port (`test_a_number_and_a_tensor_of_the_same_value_differ_in_half_precision`).
- **A numpy array is a TypeError,** 0-d or not, as in the scheduler, whose `timestep * sample` has no path for one.
  Ours says what to do: `sample_sigmas` returns numpy, so its timesteps go in as `mx.array(1 - sigmas)` and are then
  a tensor's (`test_scale_noise_refuses_a_numpy_array_as_the_scheduler_does`).
- **More axes than the sample is refused. This differs from the scheduler on purpose.** Theirs broadcasts such a
  timestep into a result larger than the sample. Several timesteps against one example is ordinary broadcasting on
  both sides and is kept (`test_scale_noise_timestep_axes`).

`test_scale_noise_with_a_float_is_the_schedulers` (5 timesteps × float32, bf16, float16),
`test_scale_noise_with_a_numpy_scalar_is_the_schedulers` (`np.float32`, `np.float64` and `int` × the three dtypes)
and `test_scale_noise_with_an_array_is_the_schedulers` (a 0-d timestep, one per example, one shaped for the rows) are
bit for bit against the scheduler.

### 3.2 The target

diffusers has no function for the training target of this model. It is fixed by the scheduler's step:

📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:313-329`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/schedulers/scheduling_minimax_h3.py#L313-L329)
```python
 313         # x0 from the data-ward velocity. The sigma used here is recovered from the *timestep* the
 314         # transformer was conditioned on, whereas the Euler ratio below uses the sigma grid: for
 315         # sigma < 0.5 the float32 round trip `1 - (1 - sigma)` is not exact, and the reference keeps
 316         # the two sources apart.
 317         if not isinstance(timestep, torch.Tensor):
 318             timestep = torch.tensor(timestep, dtype=sample.dtype)
 319         sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
 320         while sigma_from_timestep.ndim < sample.ndim:
 321             sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
 322         denoised = sample + sigma_from_timestep * model_output
 323
 324         # Euler with eta = 0, written as an x_t / x0 blend and evaluated in float32.
 325         compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
 326         sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
 327         sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
 328         ratio = sigma_next / sigma
 329         prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
```
📍 [`src/h3mlx/ops.py:414-422`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L414-L422)
```python
 414 def flow_step(v: mx.array, t: float, x: mx.array, sigma: float, sigma_next: float) -> mx.array:
 415     """x0 = x + (1 - t) * v (sigma from the timestep, in x's dtype); x_next = r*x + (1 - r)*x0, r = s_next / s, f32."""
 416     st = (mx.array(1.0, dtype=x.dtype) - mx.array(t, dtype=x.dtype)).astype(x.dtype)
 417     x0 = x + st * v
 418     cdt = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
 419     s = mx.array(sigma, dtype=cdt)
 420     sn = mx.array(sigma_next, dtype=cdt)
 421     r = sn / s
 422     return (r * x.astype(cdt) + (1.0 - r) * x0.astype(cdt)).astype(x.dtype)
```

With `x_t = t·x_0 + (1 − t)·noise` and `x_0 = x_t + (1 − t)·v`, the model's output must satisfy
`(1 − t)·v = (1 − t)·(x_0 − noise)`. So the target is `x_0 − noise` at every `t`.

📍 [`src/h3mlx/train.py:57-61`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L57-L61)
```python
  57 def velocity_target(sample: mx.array, noise: mx.array) -> mx.array:
  58     """What the transformer is trained to predict for `scale_noise(sample, t, noise)`, at every `t`: `x_0 - noise`.
  59     From the scheduler's `x_0 = x_t + (1 - t) * v` with `x_t = t * x_0 + (1 - t) * noise`: `(1 - t) * v` must equal
  60     `(1 - t) * (x_0 - noise)`. Not `noise - x_0`."""
  61     return sample - noise
```

Under the usual convention the notes above quote, `x_0 = x_t − sigma·v`, the target is `noise − x_0`: the same
velocity with the other sign, and the wrong one for this model. Three tests fail if the sign is flipped:

- `test_one_euler_step_along_the_target_lands_on_x0` (4 timesteps): noise a sample, take the target as the model's
  output, step to sigma 0 with the scheduler's `step` and with our `flow_step`. Both return `x_0` within 4e-6. The
  test also runs the flipped target and checks it lands at `2·x_t − x_0`, more than a thousand tolerances away.
- `test_target_is_the_velocity_the_step_equation_implies`: solves the step equation for `v` in float64 without our
  function and compares.
- `test_loss_sees_the_sign_of_the_target`: a prediction that satisfies the step equation has a loss below 1e-10
  against our target, and `4·mean((x_0 − noise)²)` against the flipped one.

### 3.3 The loss

📍 [`src/h3mlx/train.py:64-87`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L64-L87)
```python
  64 def flow_match_loss(pred: mx.array, target: mx.array, weight=None) -> mx.array:
  65     """`mean(weight * (pred - target) ** 2)` over every element, in float32 whatever the inputs' dtypes. `weight` is a
  66     number, or an array lined up with the leading axes (one weight per example, from `loss_weighting_for_sd3`).
  67
  68     Pass the rows the model is trained on and nothing else: the generated video rows, picked by the layout as `denoise`
  69     picks them (`pred[:, layout.num_condition_video_rows:]`). Condition rows are inputs. Rows can't be told apart by
  70     their timestep index: at `t >= 0.999` a generated row and a condition row share one.
  71
  72     `pred` and `target` must have the same shape, and each axis of `weight` must be 1 or the size of the leading axis
  73     it lines up with. torch's `mse_loss` broadcasts a target of another shape with a warning; here it is an error,
  74     because a prediction that still holds its condition rows would be averaged against a one-row target without a
  75     word."""
  76     if tuple(pred.shape) != tuple(target.shape):
  77         raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape: pass the same "
  78                          "rows of both")
  79     err = pred.astype(mx.float32) - target.astype(mx.float32)
  80     sq = err * err
  81     if weight is not None:
  82         w = weight.astype(mx.float32) if isinstance(weight, mx.array) else mx.array(weight, dtype=mx.float32)
  83         if w.ndim > sq.ndim or any(a not in (1, b) for a, b in zip(w.shape, sq.shape)):
  84             raise ValueError(f"a weight of shape {tuple(w.shape)} does not line up with the leading axes of "
  85                              f"{tuple(sq.shape)}")
  86         sq = w.reshape(w.shape + (1,) * (sq.ndim - w.ndim)) * sq
  87     return mx.mean(sq)
```

- **The oracle is torch's `mse_loss`,** on float32 casts of both inputs (`test_loss_is_torchs_mse`, including a bf16
  prediction against a bf16 target, where squaring in bf16 would be wrong in the third digit). The weighted form, each
  example's mean of `w · err²` and then the mean over examples, is written out in torch in `test_weighted_loss`; it is
  not taken from a trainer. `test_loss_gradient_is_torchs` compares the gradient with respect to the prediction.
- **Shapes must match. This differs from torch on purpose.** `mse_loss` broadcasts a target of another shape with a
  warning. A prediction that still holds nine condition rows against a one-row target is then averaged over all ten
  rows. Ours raises, and checks the weight the same way: each of its axes must be 1 or the size of the leading axis it
  meets (`test_loss_refuses_shapes_that_do_not_line_up`).
- **Which rows.** The loss averages whatever it is given, so the caller passes the generated video rows and nothing
  else. They are picked by the layout, as `denoise` picks them, and not by timestep. The table of distinct timesteps
  shrinks as `t` rises: on the tests' ref2va layout it is `[0.5, 0.999, 1.0]` at `t = 0.5`, two values at
  `t = 0.9995`, where the generated rows and the condition rows (held at `max(t, 0.999)`) share one, and a single
  value at `t = 1`. Building the example and its rows is a later piece.

## 4. Sampling the noise level and weighting the loss

### 4.1 The density

📍 [`diffusers:src/diffusers/training_utils.py:366-390`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/training_utils.py#L366-L390)
```python
 366 def compute_density_for_timestep_sampling(
 367     weighting_scheme: str,
 368     batch_size: int,
 369     logit_mean: float = None,
 370     logit_std: float = None,
 371     mode_scale: float = None,
 372     device: torch.device | str = "cpu",
 373     generator: torch.Generator | None = None,
 374 ):
 375     """
 376     Compute the density for sampling the timesteps when doing SD3 training.
 377
 378     Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.
 379
 380     SD3 paper reference: https://huggingface.co/papers/2403.03206v1.
 381     """
 382     if weighting_scheme == "logit_normal":
 383         u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
 384         u = torch.nn.functional.sigmoid(u)
 385     elif weighting_scheme == "mode":
 386         u = torch.rand(size=(batch_size,), device=device, generator=generator)
 387         u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
 388     else:
 389         u = torch.rand(size=(batch_size,), device=device, generator=generator)
 390     return u
```

📍 [`src/h3mlx/train.py:444-470`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L444-L470)
```python
 444 def density_for_timestep_sampling(weighting_scheme: str, batch_size: int, logit_mean: float | None = None,
 445                                   logit_std: float | None = None, mode_scale: float | None = None,
 446                                   generator=None) -> np.ndarray:
 447     """diffusers' `compute_density_for_timestep_sampling`: `batch_size` values `u` in [0, 1], float32.
 448
 449     * `"logit_normal"`: the sigmoid of a normal draw (`logit_mean`, `logit_std`).
 450     * `"mode"`: `1 - u - mode_scale * (cos(pi * u / 2) ** 2 - 1 + u)` of a uniform draw.
 451     * anything else: a uniform draw.
 452
 453     The draw is torch's, from `generator` (a `torch.Generator`; None is torch's global one), so it is the same stream
 454     as theirs. The arithmetic after it is numpy float32 in their order: equal to theirs bit for bit except for the
 455     sigmoid's `exp` and the `cos`, which numpy and torch round differently in the last place or two.
 456
 457     A scheme's own parameters are required, as in theirs (TypeError), and must be finite, which theirs does not check
 458     (ValueError here). A negative or NaN `logit_std` is torch's own RuntimeError on both sides."""
 459     import torch
 460
 461     if weighting_scheme == "logit_normal":
 462         _density_parameters(weighting_scheme, logit_mean=logit_mean, logit_std=logit_std)
 463         u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator).numpy()
 464         return (np.float32(1) / (np.float32(1) + np.exp(-u))).astype(np.float32)
 465     if weighting_scheme == "mode":
 466         _density_parameters(weighting_scheme, mode_scale=mode_scale)
 467         u = torch.rand(size=(batch_size,), generator=generator).numpy()
 468         c = np.cos(np.float32(math.pi) * u / np.float32(2))
 469         return (np.float32(1) - u - np.float32(mode_scale) * (c * c - np.float32(1) + u)).astype(np.float32)
 470     return torch.rand(size=(batch_size,), generator=generator).numpy()
```

The draw is torch's own call on a torch generator, so the stream is theirs. The arithmetic after it is numpy float32
in their order. What that gives, over 40 seeds and batch sizes from 1 to 1,000 (`test_density_matches_diffusers`):

| scheme | against torch | why |
| --- | --- | --- |
| uniform (the default branch) | bit for bit | the draw and nothing else |
| `logit_normal` | within 2 float32 steps (the test allows 4) | numpy's `exp` in the sigmoid is not torch's |
| `mode` | within 6 float32 steps (the test allows 12) | numpy's `cos` is not torch's, and the subtraction that follows magnifies it |

`test_density_guards` checks that swapped arguments and a wrong scheme are far outside those allowances, that two
calls on one generator continue its stream, and that no generator means torch's global one.

The parameters, case by case (`test_density_parameters_are_required_and_finite`):

| input | theirs | ours |
| --- | --- | --- |
| a scheme's parameter left out (`mode_scale`, `logit_mean`, `logit_std`) | TypeError | TypeError that names it |
| `mode_scale` or `logit_mean` NaN or infinite; `logit_std` infinite | values that are NaN, infinite, or all 0 and 1, and no error | ValueError. **This differs on purpose.** |
| `logit_std` negative or NaN | `torch.normal`'s RuntimeError | the same error, from the same call |
| `logit_std` zero | 0.5 for every draw | the same |

### 4.2 The loss weighting

📍 [`diffusers:src/diffusers/training_utils.py:393-408`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/training_utils.py#L393-L408)
```python
 393 def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
 394     """
 395     Computes loss weighting scheme for SD3 training.
 396
 397     Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.
 398
 399     SD3 paper reference: https://huggingface.co/papers/2403.03206v1.
 400     """
 401     if weighting_scheme == "sigma_sqrt":
 402         weighting = (sigmas**-2.0).float()
 403     elif weighting_scheme == "cosmap":
 404         bot = 1 - 2 * sigmas + 2 * sigmas**2
 405         weighting = 2 / (math.pi * bot)
 406     else:
 407         weighting = torch.ones_like(sigmas)
 408     return weighting
```

📍 [`src/h3mlx/train.py:473-497`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L473-L497)
```python
 473 def loss_weighting_for_sd3(weighting_scheme: str, sigmas) -> np.ndarray:
 474     """diffusers' `compute_loss_weighting_for_sd3`, on float32 or float64 noise levels (numpy, or an `mx.array`).
 475
 476     * `"sigma_sqrt"`: `sigma ** -2`, which torch computes as `1 / (sigma * sigma)`; float32 (infinite at sigma 0).
 477     * `"cosmap"`: `2 / (pi * (1 - 2 * sigma + 2 * sigma ** 2))`, in the input's dtype.
 478     * anything else: ones.
 479
 480     `sigma` is the noise level in both conventions (1 is pure noise), so nothing changes for this model's `1 - t`.
 481
 482     Half-precision and integer levels are refused. Theirs computes a bf16 tensor's weighting in bf16, three digits of
 483     it; the numpy result here can't hold bf16, and upcasting without saying so would be neither theirs nor exact."""
 484     if isinstance(sigmas, mx.array):
 485         if sigmas.dtype not in (mx.float32, mx.float64):
 486             raise ValueError(f"sigmas must be float32 or float64, got {sigmas.dtype}: cast them first")
 487         sigmas = np.array(sigmas)
 488     s = np.asarray(sigmas)
 489     if s.dtype not in (np.float32, np.float64):
 490         raise ValueError(f"sigmas must be float32 or float64, got {s.dtype}: cast them first")
 491     one, two = s.dtype.type(1), s.dtype.type(2)
 492     if weighting_scheme == "sigma_sqrt":
 493         return (one / (s * s)).astype(np.float32)
 494     if weighting_scheme == "cosmap":
 495         bot = one - two * s + two * (s * s)
 496         return (two / (s.dtype.type(math.pi) * bot)).astype(s.dtype)
 497     return np.ones_like(s)
```

Bit for bit, for float32 and float64 input (`test_loss_weighting_is_diffusers`). Two details carry that: torch
computes `sigma ** -2.0` as `1 / (sigma * sigma)`, which `np.power` does not match in the last bit, and `sigma_sqrt`
returns float32 even from float64 while `cosmap` keeps the input's dtype. `sigma` means the noise level in SD3's
convention and in this model's, so the functions apply as they are.

- **float32 or float64 only. This differs from diffusers on purpose.** Theirs computes in the tensor's own
  precision, so bf16 levels get a bf16 weighting, three digits of it. The numpy result here cannot hold bf16, and
  upcasting quietly would be neither theirs nor exact, so half-precision and integer levels are refused. An
  `mx.array` of float32 is taken like numpy's (`test_loss_weighting_takes_float32_or_float64_only`).

### 4.3 The grid and the lookup

MiniMax H3's scheduler has a sampling grid and no training grid. The port builds the levels to train at from its own
`ops.flow_sigmas`, which is that scheduler's `linspace` branch, bit for bit (main guide):

📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:168-179`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/schedulers/scheduling_minimax_h3.py#L168-L179)
```python
 168         if sigmas is None:
 169             if num_inference_steps is None or num_inference_steps < 2:
 170                 raise ValueError(
 171                     "`set_timesteps` requires either an explicit `sigmas` schedule or "
 172                     f"`num_inference_steps` >= 2, got {num_inference_steps}."
 173                 )
 174
 175             # The rectified-flow sigma range is fixed at [1.0, 0.0].
 176             base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
 177             sigmas = self._shift * base / (1 + (self._shift - 1) * base)
 178             # The shift compresses the grid near sigma = 1; collapse any float32 collisions it creates.
 179             sigmas = torch.unique_consecutive(sigmas)
```
📍 [`src/h3mlx/ops.py:401-411`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L401-L411)
```python
 401 def flow_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
 402     """linspace(1, 0, n) in float32, exponential shift, consecutive duplicates collapsed. Terminal 0 included."""
 403     if num_inference_steps is None or num_inference_steps < 2:
 404         raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
 405     if not usable_shift(shift):
 406         raise ValueError(f"shift must be positive and finite, got {shift}")
 407     base = torch_linspace_f32(1.0, 0.0, num_inference_steps)
 408     shift32 = np.float32(shift)
 409     sig = (shift32 * base / (np.float32(1) + (shift32 - np.float32(1)) * base)).astype(np.float32)
 410     keep = np.concatenate([[True], sig[1:] != sig[:-1]])
 411     return sig[keep]
```

📍 [`src/h3mlx/train.py:500-509`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L500-L509)
```python
 500 def training_sigmas(num_points: int, shift: float) -> np.ndarray:
 501     """`num_points` noise levels to train at, from 1.0 (pure noise) down, float32: the scheduler's own shifted grid
 502     for `num_points + 1` points (`ops.flow_sigmas`: `linspace(1, 0, n)` through `s * sigma / (1 + (s - 1) * sigma)`)
 503     without its last point, sigma 0, where the sample is clean and there is nothing to predict. Both numbers are the
 504     caller's: this module has no default shift or grid size."""
 505     grid = ops.flow_sigmas(num_points + 1, shift)[:-1]
 506     if len(grid) != num_points:
 507         raise ValueError(f"shift {shift} merges neighbouring float32 levels of a {num_points}-point grid "
 508                          f"({len(grid)} distinct): ask for fewer points")
 509     return grid
```
📍 [`src/h3mlx/train.py:512-531`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L512-L531)
```python
 512 def sample_sigmas(u, grid) -> tuple[np.ndarray, np.ndarray]:
 513     """One noise level per `u`: `grid[floor(u * len(grid))]`, the multiplication in float32 (an SD3-style trainer's
 514     `(u * num_train_timesteps).long()` lookup). Returns (sigmas, indices). `u = 1.0` would index one past the end,
 515     which torch refuses; it is given the last level here (the logit-normal sigmoid rounds to 1.0 for a large draw, and
 516     `"mode"` maps a draw of 0 to 1.0). The model's timestep for a level is `1 - sigma` in float32; `scale_noise` takes
 517     those as `mx.array(1 - sigmas)`.
 518
 519     Every `u` must lie in [0, 1], which a NaN does not: cast to an index a NaN is 0 on this hardware, pure noise for
 520     every such draw, here and in the torch lookup. The grid must be one non-empty row of finite levels."""
 521     grid = np.asarray(grid)
 522     if grid.ndim != 1 or grid.size == 0:
 523         raise ValueError(f"the grid must be one non-empty row of noise levels, got shape {grid.shape}")
 524     if not np.isfinite(grid).all():
 525         raise ValueError("the grid holds levels that are not finite")
 526     u = np.asarray(u, dtype=np.float32)
 527     inside = (u >= 0) & (u <= 1)
 528     if not inside.all():
 529         raise ValueError(f"every u must lie in [0, 1] (a NaN does not): got {u[~inside][:4].tolist()}")
 530     indices = np.minimum((u * np.float32(len(grid))).astype(np.int64), len(grid) - 1)
 531     return grid[indices], indices
```

- **The grid** is `flow_sigmas(num_points + 1, shift)` without its terminal 0: `num_points` levels from 1.0 (pure
  noise) down to the shifted `1 / num_points`. At sigma 0 the sample is clean and there is nothing to predict.
  `test_training_sigmas_are_the_schedulers` checks it is `MiniMaxH3Scheduler`'s grid bit for bit at six sizes and
  shifts, and that `1 − sigma` is the scheduler's `timesteps`.
- **Against the generic flow-match scheduler.** `FlowMatchEulerDiscreteScheduler` builds its training sigmas as
  `k / n` shifted by the same formula:

📍 [`diffusers:src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:127-133`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L127-L133)
```python
 127         timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
 128         timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
 129
 130         sigmas = timesteps / num_train_timesteps
 131         if not use_dynamic_shifting:
 132             # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
 133             sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

  They are the same levels by other arithmetic: within 3 float32 steps of ours at 1,000 points and shift 12, and not
  equal. The test asserts both.
- **No defaults.** The shift and the number of points are arguments. This module does not choose them.
- **The lookup** is `grid[floor(u · n)]` with the product in float32. diffusers has no library function for it; the
  tests write it in torch (`(u * n).long()`). The float32 product matters: `float32(0.005) · 1000` rounds up to 5.0,
  where a float64 product gives 4 (`test_sample_sigmas_lookup`).
- **`u = 1.0` is kept on the last level. This differs from the torch lookup,** which indexes one past the end and
  raises. It can happen: the sigmoid rounds to 1.0 for a large normal draw, and `mode` maps a draw of 0 to 1.0.
- **A NaN is refused. This differs from the torch lookup,** where a NaN `u` cast to an index is 0 on this hardware:
  sigma 1.0, pure noise, for every such draw and with no error. A range test written `u < 0 or u > 1` lets a NaN
  through, so ours tests that every `u` is inside. The grid must be one non-empty row
  (`test_sample_sigmas_refuses_nan_and_a_bad_grid`).
- **A million levels don't fit.** At shift 12 the float32 grid merges neighbours somewhere between 100,000 and
  1,000,000 points; `training_sigmas` raises instead of returning fewer levels than asked.
- **The shift must be a positive, finite float32. This differs from the scheduler on purpose.** Its test is
  `shift <= 0`, which NaN and infinity pass; 1e39 is infinite as a float32 and 1e-50 is zero. Each gives the scheduler
  a grid with NaN in it and no error, and gave `ops.flow_sigmas` and `denoise.shift_grid` the same, on the sampling
  path too. Both raise now (`ops.usable_shift`); valid shifts are untouched. `sample_sigmas` also refuses a grid with
  a level that is not finite (`test_a_shift_that_is_not_a_positive_finite_float32_is_refused`).

From a seed to sigmas, against the torch chain (diffusers' density, the scheduler's grid, the lookup):

| density | result |
| --- | --- |
| uniform | the same sigmas, bit for bit (`test_same_seed_same_sigmas_uniform`, 40 seeds × 3 grids) |
| `logit_normal`, `mode` | the same level except when `u` sits within a step or two of a boundary: at most 1 of 40,000 draws moved, to the neighbouring level, in each of five cases (`test_same_seed_nearly_same_sigmas_otherwise` allows 20) |

## 5. Two named adapters on one `Linear`

A trainer may keep a second adapter active beside the one it trains. The recipe of §6 does: it loads an "assistant"
adapter as a live LoRA for training, frozen, and switches it off for sampling (`minimax_h3.py:354-454` of the pinned
trainer). In PEFT that is two named adapters on the same layers.

### 5.1 Names, in the weights and in a file

PEFT keeps each adapter's matrices under the adapter's name, so its state dict has
`<module>.lora_A.<name>.weight` and `<module>.lora_B.<name>.weight`. When it saves one adapter it strips the name: a
file has `<module>.lora_A.weight`, whichever adapter it was. The port uses the same two spellings in the same two
places:

📍 [`src/h3mlx/lora.py:179-182`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L179-L182)
```python
 179 def runtime_key(module: str, matrix: str, name: str | None = None) -> str:
 180     """A matrix's key in the weights dict: `<module>.lora_A.weight`, or for a named adapter PEFT's runtime name,
 181     `<module>.lora_A.<name>.weight`. (In a file neither carries a name: `Adapter.tensors`.)"""
 182     return f"{module}.lora_{matrix}.weight" if name is None else f"{module}.lora_{matrix}.{name}.weight"
```
📍 [`src/h3mlx/lora.py:391-393`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L391-L393)
```python
 391 def runtime_tensors(adapter: Adapter, name: str) -> dict:
 392     """An adapter's tensors under their runtime names for `name` (what `split` returns as `trainable`)."""
 393     return {runtime_key(m, x, name): adapter.tensors[f"{m}.lora_{x}.weight"] for m in adapter.modules for x in "AB"}
```
📍 [`src/h3mlx/lora.py:396-405`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L396-L405)
```python
 396 def file_tensors(tensors: dict, name: str) -> dict:
 397     """The reverse: a named adapter's matrices back under the names they have in a file and in `Adapter.tensors`,
 398     `<module>.lora_A/B.weight`, as PEFT strips the adapter's name when it saves."""
 399     out = {}
 400     for key, value in tensors.items():
 401         m = RUNTIME_RE.match(key)
 402         if m is None or m["name"] != name:
 403             raise ValueError(f"{key!r} is not a matrix of the adapter {name!r}")
 404         out[f"{m['module']}.lora_{m['matrix']}.weight"] = value
 405     return out
```

- **In the weights:** `attach(..., name="subject")` puts the matrices under PEFT's runtime names. Beside them it adds
  two things PEFT keeps as Python state: `<module>.lora_scaling.<name>` where the scaling is not 1, and
  `<module>.lora_adapters`, the tuple of the adapters active on that `Linear`, in order.
- **In a file and in `Adapter.tensors`:** no name.

One difference in which names are allowed: the port takes any non-empty name without a dot. PEFT keeps its adapters
in a `ModuleDict`, which cannot hold a name that is an attribute of `nn.Module`, so `load_lora_adapter(adapter_name="train")`
fails there with `KeyError: "attribute 'train' already exists"`. The port has no guard for those names: a name never
leaves the process (a file carries none), and the list of reserved words belongs to torch.

`test_runtime_keys_are_pefts` compares the weights' LoRA keys with PEFT's state dict for two loaded adapters, and
`test_file_keys_carry_no_adapter_name` compares the file keys with `get_peft_model_state_dict` and with
`save_lora_adapter(adapter_name=...)`. They are separate tests because they are separate claims.

### 5.2 The forward

PEFT's forward (the extract in §1.1) loops over the active adapters in order. Three details of that loop are easy to
get wrong:

- it casts the input to each adapter's dtype **and keeps the cast**, so a later adapter sees the input as the earlier
  one left it;
- it adds each scaled delta to the running result, so the dtype of the running sum, and where it is rounded, depend
  on the order;
- it casts back to the base result's dtype once, at the end.

`plinear` (§1.1) follows it to the letter. An unnamed adapter, HyperFlow's included, takes the old path untouched.

📍 [`src/h3mlx/lora.py:224-274`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L224-L274)
```python
 224 def attach(params: dict, cfg: dict, adapter: Adapter, keep_dtype: bool = False, name: str | None = None,
 225            scale: float = 1.0) -> dict:
 226     """The transformer's weights with the adapter beside them, for `ops.plinear`; `params` is not modified.
 227
 228     By default each matrix is cast to its base weight's dtype, as PEFT's injection does (`load_lora_adapter`, and
 229     `hyperflow.attach`). `keep_dtype=True` leaves the adapter as it is: float32 factors over a bf16 base, what
 230     `cast_training_params` gives a trainer (and what diffusers' `low_cpu_mem_usage=True` loading gives a float32 file).
 231     A module whose scaling is not 1 also gets a `lora_scaling` entry, in float32; a scaling that is finite as a Python
 232     float and not in float32 (a scale of 1e40) is refused.
 233
 234     `scale` is the adapter's strength, 1.0 being as trained: each module's scaling becomes `scale * alpha / rank`,
 235     computed in that order, which is PEFT's `set_scale` and so what diffusers' `set_adapters(name, weights=scale)`
 236     leaves on every layer. Any finite number: a negative one subtracts the adapter's delta, and zero keeps the adapter
 237     attached with a delta of zero (`for_sampling` leaves it off instead).
 238
 239     Without a `name` the adapter is the only one its Linears can carry. With one it is stacked: its matrices go under
 240     PEFT's runtime names, and the name is appended to each Linear's `lora_adapters`, so adapters apply in the order
 241     they were attached (`set_active` changes that). A named and an unnamed adapter can't share a Linear, and a name is
 242     used once: diffusers' `load_lora_adapter` refuses a name already in use too. That includes a name these weights
 243     only list or carry a scaling for, which is what the frozen half of `split` looks like (`join` is the way back
 244     from it); attaching onto that half would name the adapter twice on each Linear and apply it twice."""
 245     check(adapter, cfg)
 246     check_scalings(adapter, scale)
 247     if name is None:
 248         taken = [m for m in adapter.modules if _carries_lora(params, m)]
 249         if taken:
 250             raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: one unnamed adapter per "
 251                              "Linear; give each a name to stack them")
 252     else:
 253         if not name or "." in name:
 254             raise ValueError(f"an adapter name is a non-empty word without dots, got {name!r}")
 255         taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
 256         if taken:
 257             raise ValueError(f"{len(taken)} Linear(s) already carry an unnamed LoRA, e.g. {taken[:3]}")
 258         used = _named_parts(params, name)
 259         if used:
 260             raise ValueError(f"the adapter name {name!r} is already in use in these weights ({len(used)} keys, e.g. "
 261                              f"{used[:2]}). If they are the frozen half of `split`, `join` it with its trainable half; "
 262                              "a changed adapter goes in as that trainable half, not through a second `attach`")
 263     out = dict(params)
 264     for module in adapter.modules:
 265         dtype = params[module + ".weight"].dtype
 266         for matrix in ("A", "B"):
 267             tensor = adapter.tensors[f"{module}.lora_{matrix}.weight"]
 268             out[runtime_key(module, matrix, name)] = tensor if keep_dtype else tensor.astype(dtype)
 269         scaling = adapter.scaling(module) if scale == 1.0 else scale * adapter.alpha[module] / adapter.rank(module)
 270         if scaling != 1.0:
 271             out[module + SCALING + ("" if name is None else "." + name)] = mx.array(scaling, dtype=mx.float32)
 272         if name is not None:
 273             out[module + ADAPTERS] = params.get(module + ADAPTERS, ()) + (name,)
 274     return out
```
📍 [`src/h3mlx/lora.py:282-295`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L282-L295)
```python
 282 def set_active(params: dict, names) -> dict:
 283     """The weights with the adapters `names` active, in that order, and every other named adapter switched off: PEFT's
 284     `set_adapters`. A Linear applies those of `names` it carries. The matrices of an adapter that is off stay in the
 285     dict, unused, so it can be switched back on; `params` is not modified."""
 286     names = tuple(names)
 287     unknown = sorted(set(names) - set(adapter_names(params)))
 288     if unknown or len(set(names)) != len(names):
 289         raise ValueError(f"adapters {unknown or list(names)} are not attached, or are named twice; attached: {adapter_names(params)}")
 290     out = dict(params)
 291     for key in params:
 292         if key.endswith(ADAPTERS):
 293             module = key[: -len(ADAPTERS)]
 294             out[key] = tuple(n for n in names if runtime_key(module, "A", n) in params)
 295     return out
```

- **A name is used once.** `attach` refuses a name already in use anywhere in the weights, as diffusers'
  `load_lora_adapter` does ("already in use in the model").
- **Order.** Adapters apply in the order they were attached. `set_active(params, names)` is PEFT's `set_adapters`: it
  returns weights with those adapters active in that order. The matrices of an adapter that is off stay in the dict,
  as PEFT keeps them.
- **Sampling.** The assistant off and the trained adapter on is `set_active(params, ("subject",))`.

`test_stacked_forward_matches_peft_in_every_state` loads both adapters into diffusers' transformer under names and
compares the forward in five states: both active in either order, each alone, neither. The two adapters overlap on
most linears, and each has some to itself (the trainable one sits on an AdaLN projection and on `proj_in` too).

`test_stacked_rounding_is_pefts` pins the three details bit for bit on one `Linear`, with an identity base and
selection matrices so that every product is exact:

| case | what it pins |
| --- | --- |
| bf16 base, a bf16 adapter and a float32 one, both orders | which sum is rounded to bf16 on the way |
| float32 base and input, a bf16 adapter then a float32 one, both orders | the kept cast: the second adapter sees the input rounded to bf16 |

In PEFT itself the two orders give different bits in both cases; the test checks that too.

### 5.3 Two groups of weights

📍 [`src/h3mlx/lora.py:298-308`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L298-L308)
```python
 298 def split(params: dict, name: str) -> tuple[dict, dict]:
 299     """(frozen, trainable): the weights without the two matrices of the adapter `name`, and those matrices alone under
 300     their runtime names. The two are separate groups: a trainer differentiates with respect to `trainable` only, and
 301     merges them (`{**frozen, **trainable}`) inside the function it differentiates, so the base weights and every other
 302     adapter are constants. The adapter's scalings and its place in each Linear's `lora_adapters` stay in `frozen`,
 303     which is therefore not a model on its own: the forward refuses it, and so does `attach` for the same name. `join`
 304     puts the two back together."""
 305     trainable = {k: v for k, v in params.items() if (m := RUNTIME_RE.match(k)) and m["name"] == name}
 306     if not trainable:
 307         raise ValueError(f"no adapter named {name!r} in these weights; attached: {adapter_names(params)}")
 308     return {k: v for k, v in params.items() if k not in trainable}, trainable
```

`split(params, "subject")` returns the trainable adapter's matrices as one dict and everything else as another: the
base weights, the other adapter, the scalings and the adapter lists. A trainer differentiates with respect to the
first and merges the two inside the function it differentiates (§7). Nothing is marked as frozen: a weight is frozen
by not being in the group that is differentiated.

`test_gradients_reach_the_trainable_factors_only` compares the gradient of a simple objective with torch's, where the
assistant's and the base's parameters have `requires_grad` off and end with no gradient. The worst gradient tensor is
8.6e-6 from torch's, relative. The trainable gradient still runs through the assistant's deltas in the layers
downstream; the test checks that switching the assistant off changes it.

Not from PEFT: PEFT's `set_adapters` turns `requires_grad` on for every adapter it activates, so a trainer has to
freeze the assistant again after activating it. The torch side of these tests does.

**The frozen half is not a model.** `split` leaves the trainable adapter's scalings and its place in each `Linear`'s
list in the frozen half, so that the two halves merge back into the model. Two things follow, and both are refused
rather than left to go wrong quietly (`test_the_frozen_half_of_a_split_is_not_a_model`):

- The forward on the frozen half alone raises a `ValueError` that names the adapter and says what the dict is, where
  a missing key alone would say neither. The check sits in `plinear`'s named-adapter branch only; a forward with no
  adapter does not pass through it.
- `attach` under the same name onto the frozen half is refused: the name is still in use there, by a list entry or a
  scaling. Accepted, it would list the adapter twice on each `Linear` and apply it twice.

📍 [`src/h3mlx/lora.py:383-388`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L383-L388)
```python
 383 def join(frozen: dict, trainable: dict) -> dict:
 384     """The reverse of `split`: the two groups as one dict of weights again, for a forward outside the loss (a
 385     validation sample), for `set_active`, or for `file_tensors` and `save` after training. `trainable` is checked as
 386     a loss checks it (`check_trainable`)."""
 387     check_trainable(trainable, frozen)
 388     return {**frozen, **trainable}
```
📍 [`src/h3mlx/lora.py:311-352`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L311-L352)
```python
 311 def check_trainable(trainable: dict, frozen: dict) -> None:
 312     """Refuses a pair of groups in which `trainable` would not be what the forward trains.
 313
 314     `ops.plinear` reads a named adapter's matrices only where the Linear lists that name as active. Matrices under a
 315     file's names, for a name that was never attached, or for an adapter that `set_active` switched off are merged
 316     into the weights and then ignored: the loss comes out without them and their gradients are exactly zero, with no
 317     error. (A file's names on a Linear with no named adapter are read, but as an unnamed adapter of scaling 1.)
 318
 319     So: `trainable` is not empty; each of its keys is a runtime name, `<module>.lora_A/B.<name>.weight`; that name is
 320     in `frozen`'s list of active adapters for that module; and each module's two matrices come together.
 321
 322     Two more, about the pair as a whole. No key is in both groups: merged, the trainable one would win, and an
 323     adapter thought frozen would be trained. And the two together are a model: every adapter a Linear lists has its
 324     matrices in one group or the other, so a trainable group that holds only some of an adapter's modules is refused
 325     here and not later, inside the forward."""
 326     if not trainable:
 327         raise ValueError("no trainable matrices: `trainable` is empty")
 328     shared = sorted(set(trainable) & set(frozen))
 329     if shared:
 330         raise ValueError(f"{len(shared)} key(s) are in both groups, e.g. {shared[0]!r}: a matrix is trained or frozen, "
 331                          "not both")
 332     seen: dict = {}
 333     for key in trainable:
 334         m = RUNTIME_RE.match(key)
 335         if m is None:
 336             raise ValueError(f"{key!r} is not an attached adapter's matrix (`<module>.lora_A/B.<name>.weight`): "
 337                              "`attach` the adapter under a name and `split` it; a file's keys are not read by the forward")
 338         active = frozen.get(m["module"] + ADAPTERS, ())
 339         if m["name"] not in active:
 340             raise ValueError(f"the adapter {m['name']!r} is not active on {m['module']!r} in the frozen weights (active "
 341                              f"there: {list(active)}): its matrices would not be used and their gradients would be zero")
 342         seen.setdefault((m["module"], m["name"]), set()).add(m["matrix"])
 343     half = sorted(module for (module, _), matrices in seen.items() if matrices != {"A", "B"})
 344     if half:
 345         raise ValueError(f"{len(half)} module(s) have one of lora_A / lora_B without the other, e.g. {half[:3]}")
 346     absent = sorted((name, key[: -len(ADAPTERS)]) for key, active in frozen.items() if key.endswith(ADAPTERS)
 347                     for name in active
 348                     if runtime_key(key[: -len(ADAPTERS)], "A", name) not in frozen and (key[: -len(ADAPTERS)], name) not in seen)
 349     if absent:
 350         raise ValueError(f"the adapter {absent[0][0]!r} is listed on {len(absent)} Linear(s) whose matrices are in "
 351                          f"neither group, e.g. {absent[0][1]!r}: the trainable group must hold every module of the "
 352                          "adapter that was split off")
```

`join` is the way back: the two halves as one dict again, for a forward outside the loss, for `set_active`, or for
saving. An updated adapter goes in as the trainable half, not through a second `attach`.

`check_trainable` is what `join` and every loss check first (§7). A named adapter's matrices are read only where the
`Linear` lists the name as active, so a trainable group that is not that (a file's keys, a name never attached, an
adapter switched off) is merged into the weights and then ignored: the forward runs, and the gradients come back
exactly zero with no error. PEFT has no such state to get into, because there the trainable parameters are the
module's own. `test_check_trainable` covers each case.

It also looks at the pair as a whole (`test_the_two_groups_share_nothing_and_make_a_whole_model`):

- **No key in both groups.** Merged, the trainable one wins, so a frozen adapter's matrices handed in beside the
  trained ones would be trained too.
- **The two make a model.** Every adapter a `Linear` lists has its matrices in one group or the other. A trainable
  group that holds only some of the split adapter's modules is refused here, not part of the way through a forward.
  (The rest of the adapter moved to the frozen side is fine: that is training some modules and freezing the others.)

📍 [`src/h3mlx/lora.py:355-380`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L355-L380)
```python
 355 def check_config(trainable: dict, frozen: dict, cfg: dict) -> None:
 356     """Refuses weights or trainable matrices that are not this config's transformer's.
 357
 358     The forward runs `cfg`'s blocks. Weights with more blocks than that would run as a shorter model, and trainable
 359     matrices on the blocks left out would never be read: their gradients would be exactly zero, with no error. So the
 360     weights have the config's transformer blocks and refiner blocks, the last of each and none beyond; every
 361     trainable matrix sits on a Linear of this config (`wired_shapes`); and its shape is that Linear's, with one rank
 362     for the pair."""
 363     for prefix, count in (("transformer_blocks", cfg["num_layers"]), ("token_refiner.refiner_blocks", cfg["num_refiner_layers"])):
 364         if f"{prefix}.{count}.attn.to_q.weight" in frozen or (count and f"{prefix}.{count - 1}.attn.to_q.weight" not in frozen):
 365             raise ValueError(f"the config has {count} {prefix}, and the weights do not: they are another model's, or "
 366                              "the config is")
 367     wired = wired_shapes(cfg)
 368     ranks: dict = {}
 369     for key, value in trainable.items():
 370         m = RUNTIME_RE.match(key)
 371         if m is None or m["module"] not in wired:
 372             raise ValueError(f"{key!r} is not on a Linear of the transformer this config describes: it would never be "
 373                              "read, and its gradient would be zero")
 374         out_features, in_features = wired[m["module"]]
 375         shape = tuple(value.shape)
 376         rank = shape[0 if m["matrix"] == "A" else -1] if len(shape) == 2 else 0
 377         if rank < 1 or shape != ((rank, in_features) if m["matrix"] == "A" else (out_features, rank)):
 378             raise ValueError(f"{key!r} has shape {shape}, which does not fit a Linear of {out_features} x {in_features}")
 379         if ranks.setdefault((m["module"], m["name"]), rank) != rank:
 380             raise ValueError(f"{m['module']}: lora_A and lora_B of {m['name']!r} have different ranks")
```

`check_config` is the third thing a loss checks: the weights and the trainable matrices are the config's. The forward
runs the config's blocks, so a config of two blocks handed weights and matrices for four ran a two-block model and
returned gradients of exactly zero for blocks 2 and 3. Now the weights must have the config's blocks and no more,
and every trainable matrix must sit on one of the config's Linears, in its shape (`test_check_config`,
`test_a_config_of_fewer_blocks_than_the_weights_is_refused`).

### 5.4 The public training adapter, by its header

The recipe's assistant for P-H3-Image is a public file. Its safetensors header (keys, shapes and dtypes; no tensor
data) is a release fixture, fetched by `scripts/fetch_test_fixtures.py` like HyperFlow's:

- **File:** `ostris/minimax_h3_training_adapter`, `p_h3_image_training_adapter_v1.safetensors`, revision `e4e119ca`.
  The repository states no licence.
- **Header:** 416 tensors, all bf16, rank 16, in the original checkpoint's key layout with the `diffusion_model.`
  prefix, on `qkv_proj`, `out_proj`, `mlp.fc1` and `mlp.fc2` of 50 blocks and 2 refiner blocks. No `.alpha` tensors
  and no `alpha` entry.

`test_training_adapter_header_fits_p_h3_image` converts the header's shapes (§2) and checks them against the released
transformer's config: 312 modules, the attention and feed-forward linears of every block and refiner block, each pair
of the shape its `Linear` needs, scaling 1, and the same keys and shapes as diffusers' converter gives. It also
stacks under a name beside a fresh trainable adapter.

**A header proves nothing about whether the adapter fits the weights.** Shapes and names agree; the values were never
read. Whether it was trained against these weights, and whether training with it attached does what its name says,
can only be found out on real weights.

### 5.5 The tests

| test | pins |
| --- | --- |
| `test_runtime_keys_are_pefts` | the weights' LoRA keys for two named adapters are PEFT's state-dict keys; what is added beside them |
| `test_file_keys_carry_no_adapter_name` | file keys against PEFT's and diffusers' writers; `runtime_tensors` and `file_tensors` |
| `test_stacked_forward_matches_peft_in_every_state` | the tiny transformer's forward in five activation states, within 2e-4 |
| `test_stacked_rounding_is_pefts` (×2) | PEFT's order of casts and sums, bit for bit, in both adapter orders |
| `test_modulation_cache_carries_a_lora_on_the_adaln_projections` (×2) | a cached forward with AdaLN LoRAs, named and unnamed, bit for bit the uncached one; what the cache drops |
| `test_gradients_reach_the_trainable_factors_only` | the trainable group's gradient against torch autograd; none elsewhere |
| `test_stacking_refusals` | a name twice, on the same modules or on others; a named adapter beside an unnamed one (HyperFlow's included); names with dots; merging onto named adapters; unknown names |
| `test_the_frozen_half_of_a_split_is_not_a_model` | the forward and `attach` refuse the frozen half; `join` gives back the same weights and the same forward, bit for bit; an updated adapter joined is applied once |
| `test_check_trainable` | an empty group, a file's keys, a name never attached, an adapter switched off, half a pair, a key that is no adapter's |
| `test_the_two_groups_share_nothing_and_make_a_whole_model` | a key in both groups; an adapter held in part; the rest of it frozen is allowed and is the same forward |
| `test_check_config` | weights with more or fewer blocks than the config; a matrix on no Linear of the config, or in another shape or rank |
| `test_training_adapter_header_fits_p_h3_image` (release fixture) | the public adapter's header against the released config |

## 6. One training example

diffusers has no trainer for this model in the checkout read here, so there is no library code for what a training
step feeds the transformer. The port follows a named recipe for that, **the MiniMax H3 and P-H3-Image model files of
`ostris/ai-toolkit` at commit `d6a18ad7`** (MIT). It is a third-party trainer's recipe. It is not the model maker's:
how Pruna or MiniMax trained is not published. It was read, not run, and none of its code is copied here.

Most of what a training step needs is also a sampling step, and for those diffusers has code that the port already
matches. So the rule in this section is: where diffusers has the step, use the port's tested function and test
against diffusers; where only the recipe has it, follow the recipe and say so.

### 6.1 What the recipe says

| step | in the recipe | diffusers has it? |
| --- | --- | --- |
| the model's timestep is `1 − sigma`, sigma from a 0..1000 scale | `minimax_h3.py:866`, `874-875` | yes: the scheduler's convention |
| audio level: remapped from the video level by the two shifts; for P-H3-Image the audio shift is the video shift, so one level for both | `minimax_h3.py:871-873`; `p_h3_image.py:120-121`; `src/packing.py:646-653` | at sampling both schedulers get the same sigmas |
| a picture has no soundtrack: audio rows are zeros noised at the audio level | `minimax_h3.py:919-923` | no: sampling starts them from noise |
| no loss on audio rows for a single frame | `minimax_h3.py:892-901` | no |
| row timesteps: text and video rows at the picture's, audio rows at the audio's, references at `max(t, level)` | `minimax_h3.py:975-983` | yes: `build_row_timesteps` |
| reference pictures: encoded, mixed with noise at the conditioning level, packed ahead of the picture's rows | `minimax_h3.py:1258-1348` | yes: `PH3ImagePrepareConditionLatentsStep` |
| the reference encode: a seed-42 posterior sample **without** float16 rounding for P-H3-Image | `minimax_h3.py:724-738`; `p_h3_image.py:122-123` | the pipeline rounds |
| the loss is on the generated video rows; the model's output is the velocity `x_0 − noise` | `minimax_h3.py:1025-1031`, `1047-1050` | the sign: yes (§3.2) |
| an assistant adapter, live and frozen during training | `minimax_h3.py:354-454` | no (§5) |

Values read from the recipe's source. None is a default in the port:

| value | where |
| --- | --- |
| training grid: 1,000 levels, shifted by the video shift, no dynamic shifting | `minimax_h3.py:103-107`, `230-233` |
| video shift 12.0, audio shift 3.0 | `src/packing.py:47-48` |
| conditioning level 0.999, encode seed 42 | `src/packing.py:52-53` |
| two audio latents for one frame, two audio channels | `src/packing.py:43`, `91-93` |

**No preset is defined.** A preset would need the recipe's timestep density and loss weighting too. Those are set in
the trainer's loop and job configuration, which were not read, so they are missing, and a preset with holes would be
worse than none.

Not read, and so not claimed: how the trainer mixes the picture with noise and how it samples timesteps (its loop),
the base-class method its audio rows pass through, and its own packing of the sequence.

### 6.2 The builder

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:661-693`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py#L661-L693)
```python
 661     @torch.no_grad()
 662     def __call__(
 663         self, components: PH3ImageModularPipeline, state: PipelineState
 664     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 665         block_state = self.get_block_state(state)
 666         device = components._execution_device
 667         patch_size = components.patch_size
 668
 669         # One draw per condition, in packed order. Each is packed on its own because `ref2va` references are encoded
 670         # at their own resolutions, so their latents do not share a shape.
 671         packed = []
 672         for condition in block_state.condition_latents:
 673             noise = randn_tensor(condition.shape, generator=block_state.generator, device=device, dtype=torch.float32)
 674             # The anchors are not fully clean: the released model noises them to `t = 0.999` and holds them there
 675             # for every step. Mixing before the patchify is the same arithmetic, since patchify only permutes. The
 676             # level goes in as a tensor so `1 - t` is evaluated in float32, the way this port has always mixed it.
 677             noised = components.scheduler.scale_noise(
 678                 condition.to(device), torch.tensor(components.keyframe_noise_aug), noise
 679             )
 680             packed.append(patchify_video_latents(noised, patch_size))
 681         block_state.condition_rows = torch.cat(packed)
 682         # In a hand-assembled chain the canvas reaching the layout is user input, so it can disagree with the
 683         # conditioning that was actually encoded. Left alone the mismatch first surfaces as an `index_copy` shape
 684         # error inside the transformer, 50 layers deep.
 685         if block_state.condition_rows.shape[0] != block_state.num_condition_video_rows:
 686             raise ValueError(
 687                 f"The layout reserved {block_state.num_condition_video_rows} conditioning rows but the encoded "
 688                 f"conditioning latents pack into {block_state.condition_rows.shape[0]}. The canvas the layout was "
 689                 "built from and the one the conditioning was encoded at do not agree."
 690             )
 691
 692         self.set_block_state(state, block_state)
 693         return components, state
```
📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:741-761`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py#L741-L761)
```python
 741     @torch.no_grad()
 742     def __call__(
 743         self, components: PH3ImageModularPipeline, state: PipelineState
 744     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 745         block_state = self.get_block_state(state)
 746         device = components._execution_device
 747
 748         latents = block_state.latents
 749         if latents is None:
 750             shape = (1, components.vae_latent_channels, 1, block_state.latent_height, block_state.latent_width)
 751             latents = randn_tensor(shape, generator=block_state.generator, device=device, dtype=torch.float32)
 752         block_state.latents = patchify_video_latents(latents.to(device, torch.float32), components.patch_size)
 753         block_state.audio_latents = randn_tensor(
 754             (block_state.num_audio_latents * components.audio_channels, components.audio_latent_channels),
 755             generator=block_state.generator,
 756             device=device,
 757             dtype=torch.float32,
 758         )
 759
 760         self.set_block_state(state, block_state)
 761         return components, state
```
📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:939-944`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py#L939-L944)
```python
 939         sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
 940         row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
 941         row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
 942         row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
 943         row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
 944         return torch.unique(row_timesteps, sorted=True, return_inverse=True)
```

📍 [`src/h3mlx/train.py:98-118`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L98-L118)
```python
  98 @dataclass
  99 class Example:
 100     """One picture as the transformer sees it in training (`build_example`)."""
 101
 102     layout: Layout
 103     video_rows: mx.array  # (condition rows + generated rows, patch dim): the references' rows, then the noised picture's
 104     audio_rows: mx.array  # (audio rows, audio channels): silence noised at the audio level
 105     target: mx.array  # (generated rows, patch dim): `velocity_target` of the picture's rows, x_0 - noise
 106     clean_rows: mx.array  # (generated rows, patch dim): the picture's rows before noising
 107     timestep: np.ndarray  # the distinct row timesteps, sorted, float32
 108     timestep_indices: np.ndarray  # each row's index into them
 109     sigma: np.float32  # the picture's noise level; its rows' timestep is 1 - sigma
 110
 111     @property
 112     def num_condition_rows(self) -> int:
 113         return self.layout.num_condition_video_rows
 114
 115     def generated(self, video_pred: mx.array) -> mx.array:
 116         """The transformer's video output (batch, video rows, patch dim) without the condition rows: what the loss is
 117         taken over, against `target`. The rows are told apart by their place in the layout, as `denoise` does it."""
 118         return video_pred[:, self.num_condition_rows:]
```
📍 [`src/h3mlx/train.py:140-212`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L140-L212)
```python
 140 def build_example(cfg: dict, text_token_tags, latents: mx.array, sigma, noise, *, condition_t: float,
 141                   condition_latents=(), audio_sigma=None) -> Example:
 142     """One training example for a single picture with zero or more reference pictures.
 143
 144     `latents` is the picture, normalised, (1, 1, h, w, channels), and `condition_latents` the references in packed
 145     order, each (1, 1, h_i, w_i, channels) at its own size, as the VAE encode returns them. `text_token_tags` has one
 146     tag per prompt row: 1 for text, 0 for a vision token (tagged as video), which are the two the picture pipelines'
 147     encoders give. `sigma` in (0, 1] is the picture's noise level; it must also be large enough that the timestep
 148     `1 - sigma` is below 1 in float32, which is any sigma above 2**-25 (about 3e-8). Nothing is clamped. `noise(shape)`
 149     returns float32 draws (`pipeline.Noise`). `condition_t` is the timestep the references are noised to and held at;
 150     it has no default here (the pipeline and the recipe both use 0.999).
 151
 152     * Layout: text-to-image's without references, the edit's with them (`image.generate`'s two, diffusers').
 153     * Draws, in the pipeline's order: one per reference, the picture's, the audio rows'.
 154     * References: `scale_noise` to `condition_t` given as an array, so `1 - t` is taken in float32 as the pipeline
 155       takes it. (The recipe writes the same mix with a Python float, whose `1 - t` is double: 1.3e-5 relative apart.)
 156     * The picture: `scale_noise` to `t = 1 - sigma` in float32, also as an array, so the factor on the noise is the
 157       `1 - t` that the sampler's step multiplies the velocity by. Its target is `velocity_target`.
 158     * Audio rows (recipe): a picture has no soundtrack, so they are silence noised at the audio level, and no loss is
 159       taken on them. `audio_sigma=None` means the picture's own level, which is the recipe's rule for P-H3-Image. The
 160       recipe multiplies the noise by sigma itself, this by `1 - t`: sigma exactly down to 0.5, and below that sigma
 161       after the float32 rounding of `t = 1 - sigma`, within 6e-8 of it.
 162     * Row timesteps: `denoise.row_timesteps`; the references sit at `max(t, condition_t)`, as in the pipeline and the
 163       recipe."""
 164     from .image import AUDIO_CHANNELS, NUM_AUDIO_LATENTS
 165
 166     sigma32 = np.float32(sigma)
 167     audio32 = sigma32 if audio_sigma is None else np.float32(audio_sigma)
 168     for what, value in (("sigma", sigma32), ("audio_sigma", audio32)):
 169         if not (np.isfinite(value) and 0 < value <= 1):
 170             raise ValueError(f"{what} must lie in (0, 1], got {value}")
 171         if np.float32(1) - value >= 1:
 172             raise ValueError(f"{what} = {value} is too small: the timestep 1 - {what} rounds to 1.0 in float32, so the "
 173                              "rows would not be noised at all. It must be above 2**-25, about 3e-8")
 174     if not (math.isfinite(condition_t) and 0 <= condition_t <= 1):
 175         raise ValueError(f"condition_t must lie in [0, 1], got {condition_t}")
 176     patch = tuple(cfg["patch_size"])
 177     channels = cfg["in_channels"]
 178     condition_latents = list(condition_latents)
 179     for what, x in [("latents", latents)] + [(f"condition_latents[{i}]", x) for i, x in enumerate(condition_latents)]:
 180         if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] != 1 or x.shape[4] != channels or 0 in x.shape[2:4]:
 181             raise ValueError(f"{what} must be (1, 1, height, width, {channels}) with a height and a width, "
 182                              f"got {tuple(x.shape)}")
 183     lh, lw = latents.shape[2], latents.shape[3]
 184
 185     tags = np.asarray(text_token_tags)
 186     if tags.ndim != 1 or tags.size == 0 or not np.issubdtype(tags.dtype, np.integer) or not np.isin(tags, (0, 1)).all():
 187         raise ValueError("text_token_tags must be one integer tag per prompt row, each 1 (text) or 0 (a vision token), "
 188                          f"and at least one row; got {tags.tolist() if tags.size <= 16 else tags.shape}")
 189     if condition_latents:
 190         layout = build_ref2va_packed_sequence(tags, [Ref("image")] * len(condition_latents),
 191                                               [tuple(x.shape[1:4]) for x in condition_latents], [], 1, lh, lw,
 192                                               NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)
 193     else:
 194         layout = build_packed_sequence(tags, 1, lh, lw, NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)
 195
 196     t, t_audio = np.float32(1) - sigma32, np.float32(1) - audio32
 197     rows = []
 198     for x in condition_latents:
 199         b, f, h, w, ch = x.shape
 200         drawn = mx.array(noise((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 201         noised = scale_noise(x.astype(mx.float32), mx.array(np.float32(condition_t)), drawn)
 202         rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
 203     clean = latents.astype(mx.float32).transpose(0, 4, 1, 2, 3)
 204     drawn = mx.array(noise((1, channels, 1, lh, lw)))
 205     rows.append(ops.patchify(scale_noise(clean, mx.array(t), drawn), patch))
 206     audio_noise = mx.array(noise((NUM_AUDIO_LATENTS * AUDIO_CHANNELS, cfg["audio_in_channels"])))
 207     audio_rows = scale_noise(mx.zeros(audio_noise.shape, dtype=mx.float32), mx.array(t_audio), audio_noise)
 208
 209     timestep, timestep_indices = row_timesteps(layout, float(t), float(t_audio), max(float(t), float(condition_t)), 1.0)
 210     return Example(layout=layout, video_rows=mx.concatenate(rows), audio_rows=audio_rows,
 211                    target=ops.patchify(velocity_target(clean, drawn), patch), clean_rows=ops.patchify(clean, patch),
 212                    timestep=timestep, timestep_indices=timestep_indices, sigma=sigma32)
```

Step by step, and whose it is:

- **Layout: diffusers'.** Text-to-image's layout without references and the edit's with them, the two
  `image.generate` builds (P-H3-Image guide). The recipe packs its own sequence, which was not read and is not
  compared.
- **Order of the draws: the pipeline's.** One per reference in packed order, then the picture's, then the audio
  rows'. At sigma 1 the example's generated rows and audio rows are therefore the sampler's starting rows for the
  same seed.
- **References: the pipeline's mix.** `scale_noise` to the conditioning timestep given as a tensor, so `1 − t` is
  taken in float32. The recipe writes the same mix with a Python float, whose `1 − t` is double (§3.1): 1.3e-5
  relative apart on the noise's factor. The level itself is an argument with no default.
- **The picture: the scheduler's forward process** (§3.1) at `t = 1 − sigma` in float32, given as a tensor. The factor
  on the noise is then the `1 − t` that the sampler's step multiplies the velocity by (§3.2). Its target is
  `velocity_target`.
- **Audio rows: the recipe's.** Silence noised at the audio level. `audio_sigma=None` is the picture's level, the
  recipe's rule for P-H3-Image. One difference: the recipe multiplies the noise by sigma, the port by `1 − t`. They
  are the same number down to sigma 0.5; below that they differ by the float32 rounding of `t = 1 − sigma`, at most
  6e-8.
- **Row timesteps: diffusers'.** `denoise.row_timesteps`, with the references at `max(t, level)`.

Three inputs are refused that would otherwise build a wrong example without an error:

- **A sigma whose timestep is 1.** The rows are noised to the float32 timestep `1 − sigma`, and for a sigma of 2⁻²⁵
  (about 3e-8) or less that rounds to exactly 1.0: the picture's rows would be the clean picture, and the audio rows
  exactly zero. So sigma must be above 2⁻²⁵, for the picture and for the audio rows. The recipe clamps its sigma to
  [1e-6, 1] before it takes `1 − sigma` (`minimax_h3.py:866`), so every level the recipe can produce is accepted
  here. The port does not clamp: a level below the bound is an error, and a caller who wants the recipe's floor
  clamps before calling.
- **Latents with no height or no width,** which would put NaN in the position ids.
- **Prompt tags that are not text or vision.** One integer tag per prompt row, 1 for text and 0 for a vision token,
  the two the picture pipelines' encoders produce. An empty prompt, the audio tag and anything past the three
  modalities are refused; MLX's `take` does not report an index that is out of range.

### 6.3 The loss mask

📍 [`src/h3mlx/train.py:121-127`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L121-L127)
```python
 121 def loss_mask(layout: Layout) -> np.ndarray:
 122     """True at the packed-sequence rows a picture's loss covers: the generated video rows. Text rows, audio rows and
 123     condition rows are False. Read off the layout's indices, never off timesteps: at a timestep of the condition level
 124     or above, a generated row and a condition row carry the same value."""
 125     mask = np.zeros(len(layout.token_tags), dtype=bool)
 126     mask[layout.video_indices[layout.num_condition_video_rows:]] = True
 127     return mask
```

The loss covers the generated video rows and nothing else. `loss_mask` is that set over the packed sequence, and
`Example.generated` takes the same rows out of the transformer's video output.

The rows are found by the layout, never by timestep. With the conditioning level at 0.999, at a noise level of 0.001
or less the picture's timestep is at or above it, the references are held at the picture's timestep, and a generated
row and a condition row carry the same value. `test_loss_mask_does_not_depend_on_timesteps` builds that case at sigma 0.0005:
the table of distinct timesteps has one entry, a mask read off timesteps takes the references' rows as rows to train
on, and the layout's mask is unchanged.

### 6.4 The reference pictures' encode

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/encoders.py:61-65`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/encoders.py#L61-L65)
```python
  61     pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
  62     posterior = vae.encode(pixels, return_dict=False)[0]
  63     latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
  64     latents = latents.to(torch.float16).float().cpu()
  65     return (latents - latents_mean) / latents_std
```

📍 [`src/h3mlx/vae.py:253-264`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/vae.py#L253-L264)
```python
 253 def encode_condition(vae: VideoVAE, pixels_u8: mx.array, pixel_mean, pixel_std, latents_mean, latents_std, noise_fn,
 254                      float16_round: bool = True):
 255     """`encode_vae_condition`: uint8 pixels (1, F, H, W, 3) -> ImageNet-normalised -> moments -> posterior sample
 256     (noise from `noise_fn(shape)`, which must reproduce torch.Generator().manual_seed(42) to match the release) ->
 257     float16 round trip -> (latent - mean) / std. `float16_round=False` leaves the round trip out: not what the
 258     pipeline does, but what one training recipe does for its reference pictures (`train.encode_references`)."""
 259     x = (pixels_u8.astype(mx.float32) / 255.0 - mx.array(pixel_mean)) / mx.array(pixel_std)
 260     m = vae.encode_moments(x)
 261     lat = posterior_sample(m, noise_fn(m.shape[:-1] + (m.shape[-1] // 2,)))
 262     if float16_round:
 263         lat = lat.astype(mx.float16).astype(mx.float32)
 264     return (lat - mx.array(latents_mean)) / mx.array(latents_std)
```
📍 [`src/h3mlx/train.py:130-137`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L130-L137)
```python
 130 def encode_references(components, pictures, *, float16_round: bool) -> list:
 131     """The latents of an edit example's reference pictures: `image.encode_pictures` (a seed-42 posterior sample each),
 132     with or without its float16 round trip, which the caller must choose. The pipeline rounds, so `True` is what the
 133     model is given at sampling by diffusers and by this port. The recipe turns the rounding off for P-H3-Image, so
 134     `False` is what that trainer shows the model."""
 135     from .image import encode_pictures
 136
 137     return encode_pictures(components, pictures, float16_round=float16_round)
```

The pipeline rounds the posterior sample through float16 before normalising it; the port's `encode_condition` does
the same, and that is what the model is shown at sampling. The recipe turns the rounding off for P-H3-Image. Neither
is known to be how the model was trained. So `train.encode_references` takes the rounding as a required argument:

- `float16_round=True` is the pipeline's encode, unchanged.
- `float16_round=False` is the posterior sample itself, as the recipe encodes references.

The two differ by float16's rounding, at most 5e-4 relative. A comparison with torch through the tiny VAE is looser
than that, so `test_reference_encode_float16_rounding_is_a_choice` checks the rounding directly: un-normalised, the
rounded latents are float16 values and the others are not.

The builder takes latents, not pictures. Resizing a reference is not part of it. The pipeline keeps a reference at
its own size within an area range (P-H3-Image guide); the recipe scales it down to the target's area
(`src/packing.py:112-124`). Which a trainer does belongs with the latent cache, a later piece.

### 6.5 The tests

The torch side of `tests/test_train_example.py` assembles the same example from diffusers' functions: the two layout
builders, `patchify_video_latents`, the scheduler's `scale_noise` and `build_row_timesteps`, with the noise drawn from
one torch generator in the pipeline's order.

| test | pins |
| --- | --- |
| `test_example_is_diffusers_pieces` (3 reference counts × 4 levels) | layout, rows, audio rows, target and row timesteps, bit for bit; at sigma 1 the sampler's starting rows; the audio rows against sigma times the noise |
| `test_example_guards` | separate draws; references not left clean; an audio level and a conditioning level of their own move what they should and nothing else |
| `test_example_target_steps_back_to_the_picture` (×4) | one Euler step along the example's target lands on the picture's clean rows; along the flipped target it does not |
| `test_loss_mask_is_the_generated_video_rows` (×3) | the mask against one built from diffusers' layout; `Example.generated` |
| `test_loss_mask_does_not_depend_on_timesteps` | a generated row and a condition row with the same timestep value |
| `test_reference_encode_float16_rounding_is_a_choice`, `test_encode_references_needs_the_choice` | both answers reach the encode; no default |
| `test_example_refusals` | no conditioning level; a level outside (0, 1]; latents in the wrong layout, or with no height or width |
| `test_a_sigma_whose_timestep_rounds_to_one_is_refused` | sigma at and below 2⁻²⁵ refused for the picture and the audio rows; just above it, and at the recipe's 1e-6, a real example |
| `test_text_token_tags_are_text_or_vision` (×8) | empty, out of range, the audio tag, floats, booleans, two axes |

## 7. The example's loss and its gradient

📍 [`src/h3mlx/train.py:218-236`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L218-L236)
```python
 218 def check_inputs(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array) -> None:
 219     """What every loss and gradient of an example checks first.
 220
 221     * `trainable` is what the forward will train (`lora.check_trainable`: not empty, runtime names, active in
 222       `frozen`, both matrices, nothing shared with `frozen`, the adapter whole).
 223     * The weights and the matrices are this config's (`lora.check_config`): the weights have the config's blocks and
 224       no more, and every trained matrix sits on a Linear the config has, in that Linear's shape. A config of fewer
 225       blocks than the weights would run a shorter model, and the matrices of the blocks left out would get gradients
 226       of exactly zero.
 227     * `text` is exactly (1, the layout's text rows, text dim). A text of one row would be broadcast over every text
 228       position by the forward's scatter.
 229
 230     Each of these would otherwise go wrong without an error."""
 231     lora.check_trainable(trainable, frozen)
 232     lora.check_config(trainable, frozen, cfg)
 233     want = (1, len(example.layout.text_indices), cfg["text_dim"])
 234     if not isinstance(text, mx.array) or tuple(text.shape) != want:
 235         raise ValueError(f"text must be an array of shape {want} (1, the layout's text rows, text dim), got "
 236                          f"{tuple(text.shape) if hasattr(text, 'shape') else type(text).__name__}")
```
📍 [`src/h3mlx/train.py:290-305`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L290-L305)
```python
 290 def example_loss(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None) -> mx.array:
 291     """One example's loss: the transformer's forward on the example's rows with the weights `{**frozen, **trainable}`
 292     (`ExampleForward`), then `flow_match_loss` of the generated video rows against the example's target. Nothing is
 293     taken from the audio output or from the condition rows.
 294
 295     The weights come in as two arguments, the trainable matrices first (`lora.split`), so the function can be
 296     differentiated with respect to them and to nothing else; `check_inputs` refuses a `trainable` the forward would
 297     not use and a `text` that is not (1, text rows, text dim), the prompt's embeddings. Products with float32 LoRA
 298     matrices are float32 whatever the base's dtype (`ops.plinear`), and the loss is float32."""
 299     check_inputs(trainable, frozen, cfg, example, text)
 300     p = {**frozen, **trainable}
 301     forward = ExampleForward(cfg, example, text, weight)
 302     x, temb = forward.start(p)
 303     for i in range(cfg["num_layers"]):
 304         x = forward.block(x, temb, ops._sub(p, f"transformer_blocks.{i}."))
 305     return forward.finish(x, temb, p)
```
📍 [`src/h3mlx/train.py:308-312`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L308-L312)
```python
 308 def loss_and_gradients(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None):
 309     """(loss, gradients) of `example_loss`. The gradients are a dict with `trainable`'s keys and no others: the base
 310     weights and any other adapter in `frozen` are constants of the differentiation and get none."""
 311     check_inputs(trainable, frozen, cfg, example, text)
 312     return mx.value_and_grad(example_loss)(trainable, frozen, cfg, example, text, weight)
```

The weights come in as two arguments, the trainable group first. `mx.value_and_grad` differentiates with respect to
its first argument, so the gradients are a dict with the trainable group's keys. The base weights and any other
adapter are in the second argument and are constants.

The torch side is diffusers' transformer with PEFT's layers: a frozen assistant and a trainable adapter loaded under
names, the trainable parameters cast to float32 (`cast_training_params`, §1.3), the example assembled from diffusers'
functions (§6.5), `mse_loss` on the generated video rows, `backward`.

**What a loss checks first.** `check_inputs` runs before anything is computed, in the loss, in its gradient and in
both gradients of §8:

- the trainable group is one the forward will read (`lora.check_trainable`, §5.3). Without the check, three easy
  mistakes run to the end and return gradients of exactly zero: an `Adapter`'s own dict passed beside a named
  adapter, runtime names for an adapter that was never attached, and an adapter that `set_active` has switched off
  (`test_the_loss_refuses_matrices_the_forward_would_not_read`, which also shows the zeros through the unchecked
  forward);
- the weights and the matrices are the config's (`lora.check_config`, §5.3);
- the text is exactly (1, the layout's text rows, text dim). The forward scatters the text rows into the packed
  sequence with an assignment, and an assignment broadcasts: one row of text would fill every text position
  (`test_the_loss_refuses_a_text_of_the_wrong_shape`).

### 7.1 The forward that is differentiated

📍 [`src/h3mlx/train.py:239-287`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L239-L287)
```python
 239 class ExampleForward:
 240     """The transformer's forward on one example, for differentiating, in three stages: `start` (the projections, the
 241     token refiner, the scatter and the time embedding), `block` (one transformer block) and `finish` (the final norm,
 242     the heads and the loss). Each stage takes its weights as a dict, so a caller chooses which of them are arguments
 243     of a transform.
 244
 245     These are `ops.dit_forward`'s operations in its order. Three things differ. Nothing is evaluated between the
 246     stages: the sampling forward evaluates after every block, and an array evaluated while it is being traced keeps
 247     its graph. The modulation tables' rows are gathered with `ops.take_rows_for_training`, whose gradient over a
 248     half-precision table is the sum torch's is. And the attention is `ops.attention_for_training`, which runs
 249     half-precision q, k and v through the same call in float32: its backward is then right when attention scores are
 250     large, which the half-precision call's is not.
 251
 252     Where the release runs (MLX's fused kernels on the GPU), none of the three changes a value: the loss is the
 253     sampling forward's bit for bit, in float32 and over a bf16 base (tested at head width 64; the release's is 128).
 254     Where MLX computes a half-precision attention in half precision instead (head width 32, the CPU), the float32
 255     call is the more exact of the two and the losses differ by rounding (within 0.2% on the tiny models)."""
 256
 257     def __init__(self, cfg: dict, example: Example, text: mx.array, weight=None):
 258         lay = example.layout
 259         self.cfg, self.example, self.text, self.weight = cfg, example, text, weight
 260         self.video, self.audio = example.video_rows[None], example.audio_rows[None]
 261         self.timestep, self.t_rows = mx.array(example.timestep), mx.array(example.timestep_indices)
 262         self.video_indices, self.audio_indices = mx.array(lay.video_indices), mx.array(lay.audio_indices)
 263         self.text_indices = mx.array(lay.text_indices)
 264         self.seq = len(lay.token_tags)
 265         self.adaln_indices = self.t_rows * ops.MODALITY_NUM + mx.array(lay.token_tags)
 266         self.rotary = ops.rope_cos_sin(mx.array(lay.position_ids.astype(np.float32)),
 267                                        ops.rope_inv_freq(cfg["rope_freq_dim"], cfg["rope_theta"]))
 268
 269     def start(self, p: dict) -> tuple[mx.array, mx.array]:
 270         """(the packed rows before the first block, the time embedding) from the whole dict of weights."""
 271         x = ops.dit_inputs(p, self.cfg, self.video, self.audio, self.text, self.seq, self.video_indices,
 272                            self.audio_indices, self.text_indices, attend=ops.attention_for_training)
 273         return x, ops.two_time_embedding(self.timestep, None, p, self.cfg)
 274
 275     def block(self, x: mx.array, temb: mx.array, block_p: dict) -> mx.array:
 276         """One block on its own weights (the keys under `transformer_blocks.<i>.`, without that prefix)."""
 277         cfg = self.cfg
 278         return ops.dit_block(x, temb, self.adaln_indices, self.rotary, block_p, cfg["num_attention_heads"],
 279                              cfg["hidden_size"], cfg["norm_eps"], cfg["qk_norm_eps"], take=ops.take_rows_for_training,
 280                              attend=ops.attention_for_training)
 281
 282     def finish(self, x: mx.array, temb: mx.array, p: dict) -> mx.array:
 283         """The loss from the last block's output, from the whole dict of weights."""
 284         shift, scale = ops.norm_out_tables(temb, p)
 285         video_pred, _ = ops.dit_outputs(x, p, self.cfg, shift, scale, self.t_rows, self.video_indices,
 286                                         self.audio_indices, take=ops.take_rows_for_training)
 287         return flow_match_loss(self.example.generated(video_pred), self.example.target[None], self.weight)
```

`example_loss` does not call `ops.dit_forward`. It runs that forward's operations in its order, from the functions
`dit_forward` is itself made of (`ops.dit_inputs`, `ops.dit_block`, `ops.dit_outputs`), in three stages. Three
things differ from the sampling forward, and none changes a value where the release runs:
`test_the_training_forward_is_the_sampling_forward` compares the loss with one taken through `ops.dit_forward`, on
three examples with adapters on every `Linear`. It is the same bit for bit in float32, and over a bf16 base at a head
width of 64 (the release's is 128; the third difference, below, says why the width matters). Over a bf16 base at a
head width of 32, the other tiny models', the two losses are within 0.2% of each other and not the same bits.

**Nothing is evaluated inside it.** The sampling forward calls `mx.eval` after every block, which keeps its lazy
graph one block long. Under autograd that is the wrong thing: an array evaluated while it is being traced keeps the
graph that made it, so every block's intermediates stay alive until the backward pass has used them. §8 measures it.

**The modulation tables' rows are gathered through float32.** Each block reads six tables of one row per
(timestep, modality) pair, and every sequence row takes its row of each:

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:362-366`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/transformers/transformer_minimax_h3.py#L362-L366)
```python
 362         norm_hidden_states = norm_hidden_states * (
 363             1.0 + scale_msa.index_select(0, adaln_indices)
 364         ) + shift_msa.index_select(0, adaln_indices)
 365         attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
 366         hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output
```
📍 [`src/h3mlx/ops.py:166-168`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L166-L168)
```python
 166 def take_rows(table: mx.array, rows: mx.array) -> mx.array:
 167     """table[rows] along the first axis: each sequence row's entry of a (t, modality) table."""
 168     return mx.take(table, rows, axis=0)
```
📍 [`src/h3mlx/ops.py:171-183`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L171-L183)
```python
 171 def take_rows_for_training(table: mx.array, rows: mx.array) -> mx.array:
 172     """`take_rows` for a forward that is differentiated: the same values bit for bit, with a half-precision table
 173     gathered through float32.
 174
 175     The gather's backward adds every sequence row's cotangent into the table row it came from. `mx.take` does that
 176     sum in the table's dtype, and a bf16 running sum stops taking in what is added once it is a few hundred times
 177     the addends: the error grows with the rows per table row, from 1% at a hundred to tens of percent at thousands.
 178     torch's `index_select`, which the reference module gathers with, returns the sum rounded to bf16 once. Gathered
 179     through float32 the sum is taken in float32 and rounded once, as torch's. The cost is a float32 copy of the
 180     gathered rows, so sampling, which has no backward, keeps `take_rows`."""
 181     if table.dtype in (mx.float16, mx.bfloat16):
 182         return mx.take(table.astype(mx.float32), rows, axis=0).astype(table.dtype)
 183     return mx.take(table, rows, axis=0)
```

In the forward the two gathers are the same bits. Their gradients are not. The backward of a gather adds every
sequence row's cotangent into the table row it came from: thousands of rows into six. `mx.take` does that sum in the
table's dtype, and a running bf16 sum, with eight bits, stops taking in what it is given once it is a few hundred
times the size of the addends. torch's `index_select` returns the sum rounded to bf16 once. Gathered through float32,
MLX sums in float32 and rounds once.

Relative error of the table's gradient against the exact sum, bf16, on the CPU stream
(`bench/train_gather_gradient.py`; the rows' factors are standard normal draws around the mean given):

| sequence rows | per table row | mean | `take_rows` | `take_rows_for_training` | torch `index_select` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 100 | 0 | 0.013 | 0.0018 | 0.0018 |
| 6,000 | 1,000 | 0 | 0.036 | 0.0016 | 0.0016 |
| 24,000 | 4,000 | 0 | 0.074 | 0.0017 | 0.0017 |
| 24,000 | 4,000 | 0.5 | 0.49 | 0.0016 | 0.0016 |
| 48,000 | 8,000 | 0 | 0.11 | 0.0016 | 0.0016 |
| 48,000 | 8,000 | 0.5 | 0.74 | 0.0014 | 0.0014 |

- **Which gradients this reaches.** The ones a table's gradient feeds: an adapter on a block's AdaLN projection, on
  the final norm's projection, or on the time embedder. Adapters on the attention and feed-forward linears are not
  downstream of it, and their gradients are identical either way.
- **Against torch, at the op.** `test_the_training_gathers_gradient_is_torchs_index_select` gathers six table rows
  into 6,000: over a bf16 and a float16 table the training gather's gradient is torch's `index_select`'s bit for bit
  (on the CPU stream; on the default device in over 99% of the elements, see below), and the plain gather's is 3%
  from the exact sum with 7% of its elements torch's.
- **On the whole model.** `test_a_tables_gradient_over_a_bf16_base_is_the_rows_sum` needs a reference that does not
  depend on either gather, and makes one: the same example with a timestep entry of its own for every row. Each table
  then has a row per sequence row, used once, so nothing is summed in the gather; the rows' cotangents are summed
  later, in float32, in the product with the float32 matrices. The forward's values and the loss are the same bits.
  On 2,316 rows the gradients with shared tables are within 0.66% of that reference (one rounding of each sum
  apart); with the sampling forward's gather in the training forward they are 20% to 52% away.
- **Why sampling keeps the plain gather.** The training gather makes a float32 copy of the gathered rows, twice their
  size, and sampling has no backward. `dit_block` and `dit_outputs` take the gather as an argument, `take`, and
  `ops.dit_forward` leaves it at `take_rows`: its operations are the ones it had.
- **Order on the GPU.** On the default device MLX adds a gather's cotangents in no fixed order, so a float32 sum can
  differ in its last bit from one run to the next, and a bf16 result that sits on a rounding boundary can then move
  one step. The CPU stream is repeatable. Tests that need exact equality over a bf16 base run there.

**The attention runs in float32.** This one was found by the first run on real weights, not by a test: its loss was
finite and plausible at every step while the gradient's norm ran to millions. A block's attention is one call, in
diffusers and here:

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:199-209`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/transformers/transformer_minimax_h3.py#L199-L209)
```python
 199         hidden_states = dispatch_attention_fn(
 200             query,
 201             key,
 202             value,
 203             attn_mask=attention_mask,
 204             dropout_p=0.0,
 205             is_causal=False,
 206             backend=self._attention_backend,
 207             parallel_config=self._parallel_config,
 208         )
 209         hidden_states = hidden_states.flatten(2, 3).type_as(query)
```
📍 [`src/h3mlx/ops.py:212-220`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L212-L220)
```python
 212 def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
 213     """Full self-attention. q, k, v: (batch, seq, heads, head_dim) as diffusers lays them out; returns the same layout.
 214
 215     `mask` is an additive or boolean mask broadcastable to (batch, heads, seq_q, seq_k); the model itself passes none.
 216     """
 217     scale = 1.0 / math.sqrt(q.shape[-1])
 218     qt, kt, vt = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
 219     o = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale, mask=mask)
 220     return o.transpose(0, 2, 1, 3)
```
📍 [`src/h3mlx/ops.py:223-244`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/ops.py#L223-L244)
```python
 223 def attention_for_training(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
 224     """`attention` for a forward that is differentiated: half-precision q, k and v go through the same call in
 225     float32, and the result is cast back to their dtype.
 226
 227     MLX 0.32.4 gave `mx.fast.scaled_dot_product_attention` a fused backward pass on the GPU. It recomputes the
 228     unscaled scores `q . k` into a buffer of the inputs' dtype and rebuilds the probabilities from them as
 229     `exp(score * scale - logsumexp)`. In bf16 a score keeps eight significant bits, so the logit under the
 230     exponential is off by up to half a bf16 step of the score times the scale: 0.7 at a score of 2,800, 5.7 at
 231     25,000, 22.6 at 100,000, at 128 a head. The probabilities stop summing to 1 and the gradients for q, k and v
 232     come out wrong by that factor, while the forward stays right. A pretrained model has heads with scores that
 233     large. In float16 the unscaled score overflows at 65,504. On float32 inputs the same backward is torch's to
 234     float32 rounding, on that MLX and on earlier ones, whose bf16 backward was bounded and still several times off
 235     for q and k at such scores.
 236
 237     The forward's values do not change where the fused kernels run (MLX 0.32.4, the default device, head widths
 238     such as the release's 128): the bf16 call already returns the float32 result rounded to bf16, bit for bit.
 239     Elsewhere MLX computes the bf16 call in bf16 and this one is the more exact. The cost is float32 copies of q, k
 240     and v and of the result while the call runs, so sampling, which has no backward, keeps `attention`."""
 241     dtype = _promoted(_promoted(q.dtype, k.dtype), v.dtype)
 242     if dtype not in (mx.float16, mx.bfloat16):
 243         return attention(q, k, v, mask)
 244     return attention(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), mask).astype(dtype)
```

In the forward the two are the same bits where the release runs. Their gradients are not, on MLX 0.32.4, which is
the version the port locks and the oldest it accepts. Everything measured below is on that version, on an M2 Max,
unless it says otherwise.

**What MLX does.** MLX 0.32.4 added a fused backward pass for `mx.fast.scaled_dot_product_attention` on the GPU.
In its source at the 0.32.4 tag:

- when a gradient is being traced, the forward also returns each row's log-sum-exp, in float32 (`mlx/fast.cpp`,
  `output_logsumexp`);
- the backward (`mlx/backend/metal/scaled_dot_product_attention.cpp`, `ScaledDotProductAttentionVJP::eval_gpu` and
  `sdpa_vjp_blocked`) recomputes the unscaled scores `S = Q Kᵀ` into a tile of the inputs' dtype
  (`Dtype ctype = q.dtype()`);
- its kernel (`mlx/backend/metal/kernels/scaled_dot_product_attention_vjp.metal`, `sdpa_vjp_ds`) rebuilds each
  probability from that tile as `exp(float(S) * scale − lse)`, and the scores' cotangent from the probability.

With bf16 inputs the unscaled score is therefore rounded to eight significant bits before the exponential. One bf16
step at a score s is 2^(⌊log₂ s⌋ − 7), and the logit under the exponential is off by up to half a step times the
scale. So the size that matters is the unscaled score, q · k before the scale, which at 128 a head is 11.3 times the
logit. The rebuilt probabilities stop summing to 1, and all three gradients carry the error.

`bench/attention_bf16_backward.py` computes exactly that in float64 (exact scores from the bf16 inputs, an exact
log-sum-exp, then the tile's rounding) beside MLX's own bf16 gradients. Two heads of 605 rows at the release's head
width of 128, q and k drawn at a growing standard deviation with one key in each head aligned with a query, and
every input rounded to bf16 first. Norms of the gradients for q, k and v:

| largest score | a bf16 step there | logit error up to | largest row sum of the rebuilt probabilities | exact | the float64 model | MLX's fast call, bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,784 | 16 | 0.7 | 1.4 | 356, 353, 367 | 360, 357, 369 | 360, 357, 369 |
| 25,025 | 128 | 5.7 | 256 | 425, 418, 391 | 712, 693, 3,127 | 713, 693, 3,127 |
| 100,101 | 512 | 22.6 | 4.3e9 | 352, 347, 393 | 7.67e8, 7.67e8, 4.87e10 | 7.73e8, 7.73e8, 4.87e10 |
| 178,170 | 1,024 | 45.3 | 2.2e9 | 580, 580, 393 | 2.10e9, 2.21e9, 3.21e10 | 2.10e9, 2.21e9, 3.21e10 |

The model gives MLX's wrong gradients to two or three digits at every scale, for all three inputs. That is the
cause, and not only a candidate for it.

**Against torch.** The same op against torch's `scaled_dot_product_attention` in float32, each variant given the
same bf16 numbers. Relative error of the gradient for q and for v (the gradient for k follows q's):

| largest score | the logit it is | torch, bf16 | MLX's fast call, float32 | MLX's fast call, bf16 | by hand in bf16, and the fast call on the CPU stream | `attention_for_training`, bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 111 | 10 | 0.003, 0.004 | 8e-7, 6e-7 | 0.005, 0.008 | 0.006, 0.004 | 0.0017, 0.0016 |
| 2,784 | 246 | 0.007, 0.003 | 5e-6, 5e-6 | 0.11, 0.12 | 0.09, 0.03 | 0.0017, 0.0016 |
| 11,136 | 984 | 0.007, 0.002 | 2e-5, 2e-5 | 0.46, 0.54 | 0.35, 0.05 | 0.0017, 0.0015 |
| 25,025 | 2,212 | 0.007, 0.002 | 5e-5, 3e-5 | 1.0, 7.8 | 0.66, 0.07 | 0.0017, 0.0014 |
| 44,543 | 3,937 | 0.006, 0.002 | 6e-5, 6e-5 | 4.6, 16 | 0.68, 0.07 | 0.0017, 0.0014 |
| 100,101 | 8,848 | 0.009, 0.002 | 2e-4, 1e-4 | 2.2e6, 1.2e8 | 2.6, 0.10 | 0.0016, 0.0014 |
| 178,170 | 15,748 | 0.006, 0.002 | 2e-4, 2e-4 | 3.6e6, 8.2e7 | 1.6, 0.10 | 0.0017, 0.0014 |

- **What torch does.** Its bf16 backward stays within 1% of its float32 one at every scale.
- **What MLX does in float32.** The same as torch, to 2e-4 at a score of 178,000.
- **What MLX's fast call does in bf16, on the default device.** It is 11% off at a score of 2,800, half wrong at
  11,000, eight times the true gradient for v at 25,000, and a million times and more from 100,000. A randomly
  initialised model's scores are in the tens. A pretrained one's are not; what the released weights' are has not
  been measured here, where there are no weights.
- **What MLX does in bf16 without the fused backward.** On the CPU stream MLX differentiates the op written out
  (`ScaledDotProductAttentionVJP::use_fallback`), and so does the attention written out by hand: the same numbers as
  each other, the right size at every scale, and wrong for q and k by a third at a score of 11,000 and by more than
  their own size from 100,000. That is also what MLX 0.32.3 and 0.31.2 give on the default device, where the
  backward kernel does not exist (it is not in the 0.32.3 tag): the same script in throwaway environments gives
  1,074, 1,048 and 391 at a score of 100,000, where float32 gives 352, 347 and 393. And it is what 0.32.4 gives with
  its own switch, the environment variable `MLX_SDPA_VJP_FALLBACK`. So no bf16 attention in MLX has a backward that
  holds at these scores, on any of the three versions, and the float32 one does on all three: the bf16 inputs
  widened to float32 give float32's gradients on 0.32.4, 0.32.3 and 0.31.2 alike.
- **float16** has a finer step, eleven bits, and a smaller range: the unscaled score overflows at 65,504.
  `test_training_attention_backward_is_torchs_at_every_logit_scale` holds the training attention to torch in bf16
  and in float16, up to a score of 119,000.
- **What the port does.** `attention_for_training` runs the same call on q, k and v widened to float32 and casts the
  result back. `attn_block`, `dit_block`, `token_refiner` and `dit_inputs` take the attention as an argument,
  `attend`, as they take the gather; the training forward passes this one, for the blocks and for the token
  refiner, and `ops.dit_forward` leaves it at `attention`.

What else is established, and what is not:

- **The forward is right, and at the release's head width it is float32's.** On MLX 0.32.4's default device, at
  head widths of 64, 72, 80, 96 and 128, the bf16 call returns the float32 result rounded to bf16 in every entry
  (the bench's fourth table; the tests pin 64 and 128, at small scores and large). At 32, 48, 63, 65, 100, 127, 129
  and 160, and anywhere on the CPU stream, it does not: there MLX computes the forward in bf16, and differentiates
  the op written out. The tiny test models have a head width of 32 and never ran the fused kernels, forward or
  backward. On 0.32.3 and 0.31.2 the bf16 forward at 128 is computed in bf16 as well (2% to 44% of its entries are
  not the float32 result's).
- **At a head width of 192 the float32 call cannot be differentiated on this machine:** its kernel does not load,
  and MLX's error names it, `steel_attention_float32_…_save_lse_t` (the bench prints the message; the bf16 call's
  gradient runs there). The release's width is 128, where both load.
- **Any one of q, k and v widened is enough on this MLX,** because the call promotes the others. The port widens all
  three and does not rely on that.
- **Not established: what the forward's kernels do on paths not tried.** The forward matched float32 in every entry
  of every case here.

**On the whole model.** `test_gradients_through_sharp_attention_over_a_bf16_base` uses a tiny transformer whose head
width the fused kernels take (64) and multiplies every `norm_q` and `norm_k` weight by 30 or 50 (`sharpened`), which
puts the largest score of every attention, the token refiner's included, between 14,800 and 122,700. Four blocks in
bf16, float32 matrices on the block and refiner linears. The reference is the same forward with the attention written
out by hand in float32, which is another backward pass over the same values:

| gain | seed | largest scores | a bf16 score's logit error, up to | ours against the reference | with the bf16 call in the training forward | ours against the float32 model | torch over the bf16 base against it |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 30 | 0 | 14,808 to 31,331 | 4 to 8 | 0.0048 | 640 | 1.00 | 32.6 |
| 30 | 1 | 17,124 to 44,115 | 8 to 16 | 0.0039 | 1.3e5 | 0.91 | 23.8 |
| 30 | 2 | 15,919 to 37,991 | 4 to 16 | 0.0009 | 2.1e5 | 1.08 | 0.97 |
| 50 | 0 | 41,197 to 86,991 | 16 to 32 | 0.0012 | 1.5e20 | 0.27 | 6.92 |
| 50 | 1 | 47,591 to 122,674 | 16 to 32 | 0.0027 | 5.8e25 | 0.99 | 1.36 |
| 50 | 2 | 44,089 to 105,208 | 16 to 32 | 0.0020 | 3.0e26 | 0.69 | 0.70 |

- **The loss is still the sampling forward's,** bit for bit, in all six.
- **The gradient is the reference's within 0.5%,** where the bf16 call's was 640 to 3e26 times too large. The test
  asks for 2% and for the bf16 call to be at least a hundred times off.
- **In float32 there is an exact reference,** torch and PEFT:
  `test_gradients_through_sharp_attention_in_float32_are_torchs` holds the whole gradient to 5e-3 with the blocks'
  scores in the tens of thousands (measured: 1e-4 to 9e-4).
- **Over a bf16 base, attention this sharp is not the float32 model's.** The last two columns are relative to the
  float32 model's gradient: ours is 0.3 to 1.1 of its size away, and torch's own bf16 gradient 0.7 to 33. Rounding q
  and k to bf16 moves a logit of several thousand by several units, which is enough to change which keys a query
  attends to; in the bench, that rounding alone moves a float32 gradient by 4% at a score of 2,800 and by more than
  its size at 100,000. The fix removes a backward pass that was wrong by orders of magnitude. It does not make the
  gradient of a bf16 model with very sharp attention a smooth thing, in torch or here.
- **The bounded gradients and the script** have the same case:
  `test_bounded_gradients_through_sharp_attention_over_a_bf16_base` (§8.5) and
  `test_a_bf16_base_with_sharp_attention_trains_on_the_right_gradient` (§11.10), where the script's first three
  steps log norms of 0.95, 13 and 0.83 and, with the bf16 call put back, 4.5e6, 7.0e3 and 6.6e4, with losses
  within 0.01 of each other. That is the first real run's picture on a tiny model.
- **The cost of the op,** at 56 heads of 600 rows by 128, where q, k and v are 4.3 million elements each (8.6 MB in
  bf16, 17.2 in float32) and the scores 20 million: one forward and backward of the attention takes 8 ms against 6,
  and MLX's peak over it is 457 MB against 173 (five runs each, each in its own process).
- **The cost in a step.** The bench also takes the block-by-block gradient (§8.2) of a transformer with the
  release's widths (hidden 5,376, 56 heads of 128, feed-forward 14,336) and two blocks of random weights, a rank-16
  adapter on the block matrices, 644 rows. With the training attention it takes 0.40, 0.40 and 0.42 s, and with the
  bf16 call 0.44, 0.40 and 0.40 s: no difference that three runs show. Its peak is 1,793 MB against 1,497, 0.3 GB
  more, and block by block that does not grow with the number of blocks. On the synthetic model of §8.4, whose
  head width of 32 has no fused kernel, the float32 scores are held outright, and its tables were measured again.
- **Why sampling keeps the bf16 call.** It has no backward pass, its forward is already the float32 one, and it
  would pay for the copies. `test_the_sampling_forward_keeps_the_half_precision_call` runs it with the training
  attention made to raise.

**The other fast op is right.** The differentiated forward calls one more of MLX's fused ops on bf16 rows,
`mx.fast.rms_norm` (`ops.rms_norm`): twice on every block's rows, on q and k per head, in the token refiner and
before the heads. Its backward
was tested where a wrong one would hurt: rows far below the epsilon, where the derivative is largest, a row of
exact zeros, rows of 1e6, affine weights of 40, and rows of every size from 1e-6 to 1e4 at once, at epsilons of 1e-5
and 1e-6, on both devices. The gradients for the input and for the weight are within 0.5% of torch's float32
gradients in every case, where torch's own bf16 backward is at 0.3%
(`test_rms_norm_backward_in_bf16_is_torchs`, and the bench's fifth table). Nothing was changed there. The layer
norm and the attention of the VAEs are fused ops too, and they are in the encoders, which are not differentiated.

### 7.2 Float32

`test_loss_and_gradients_are_torchs` runs eight cases: text-to-image and edits with one and two references, four
noise levels, with the assistant beside the trainable adapter and without. The loss agrees to 1e-5 relative, and
every gradient tensor to 1e-4 (the worst measured is 1.2e-5).

At PEFT's initialisation `lora_B` is zero, which is where training starts. Then the loss is the model's without the
adapter, `lora_A`'s gradient is exactly zero on both sides, and `lora_B`'s is torch's
(`test_at_pefts_init_the_gradient_reaches_lora_b_alone`).

### 7.3 A bf16 base with float32 factors

This is the setting the trainer is meant to run in: the base in the release's mixed precision, the assistant in its
base's dtype as loaded, the trainable matrices float32 and their products float32. There is no exact answer in bf16,
so each side is scored against the float32 run, as the forward's mixed-precision tests are. Four layers, adapters on
the attention and feed-forward linears (`test_gradients_over_a_bf16_base_track_torchs`):

| seed | gradient error against float32, torch | MLX | loss error, torch | MLX |
| --- | --- | --- | --- | --- |
| 0 | 0.0114 | 0.0114 | 0.0041 | 0.0048 |
| 1 | 0.0118 | 0.0113 | 0.0005 | 0.0006 |
| 2 | 0.0156 | 0.0145 | 0.0010 | 0.0016 |

The test asks that MLX's gradient error be at most 1.5 times torch's own plus 1e-3, and that both losses be within 1%
of the float32 loss.

One detail of the torch side matters here. Loading an adapter into a bf16 model rounds its matrices to bf16, and
`cast_training_params` widens their dtype without bringing the values back. A trainer's matrices leave that grid at
their first step, so the tests put the float32 values back into the PEFT model after the cast: both sides then hold
the same matrices.

**Float32 products, pinned.** The test above cannot tell float32 products from bf16 ones: with ordinary matrices
either is within bf16's noise of the float32 run. An adapter whose rows cancel can:

📍 [`tests/test_train_gradients.py:100-115`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/tests/test_train_gradients.py#L100-L115)
```python
 100 def cancelling_adapter(cfg, modules, half, delta, seed, std=0.1):
 101     """An adapter that only float32 products get right. lora_A's second half is minus its first half plus
 102     `delta` * N, lora_B's second half repeats its first, and the scaling is 1 / delta: the update is B1 (N x), the
 103     difference of two products `1 / delta` times its size. With delta = 2**-10, float32 keeps 14 bits of it and bf16
 104     none."""
 105     g = torch.Generator().manual_seed(seed)
 106     shapes = lora.wired_shapes(cfg)
 107     tensors, alpha = {}, {}
 108     for m in modules:
 109         out_f, in_f = shapes[m]
 110         a1, n = torch.randn(half, in_f, generator=g) * std, torch.randn(half, in_f, generator=g) * std
 111         b1 = torch.randn(out_f, half, generator=g) * std
 112         tensors[m + ".lora_A.weight"] = to_mx(torch.cat([a1, -a1 + delta * n]))
 113         tensors[m + ".lora_B.weight"] = to_mx(torch.cat([b1, b1], dim=1))
 114         alpha[m] = 2 * half / delta
 115     return lora.Adapter(tensors, alpha)
```

Its update is the small difference of two large products, B₁(N x) computed as (B₁ A₁ x − B₁ A₁ x + δ B₁ N x) / δ
with δ = 2⁻¹⁰. Float32 keeps about 14 bits of that difference; bf16, with 8, keeps none.
`test_products_with_float32_matrices_are_float32_over_a_bf16_base` runs it over the bf16 base against torch and
PEFT, and runs our side once more with the same matrices cast to bf16, which is what products in the base's dtype
would compute:

| seed | gradient error against float32, torch | MLX | MLX with the matrices in bf16 |
| --- | --- | --- | --- |
| 0 | 0.0098 | 0.0095 | 0.61 |
| 1 | 0.0118 | 0.0124 | 0.70 |
| 2 | 0.0109 | 0.0112 | 0.66 |

- **Larger adapters are noisier.** In a two-layer variant with adapter matrices three times larger, the gradient
  errors ran from 1% to 10% on both sides over six seeds, with MLX between 0.5 and 1.5 times torch's (a probe, not a
  test). These are tiny random models. They say nothing about 50 real layers.
- **No bf16 product path.** Running the trainable products in bf16 to save memory is a separate mode, not built.

### 7.4 A quantized base

The base's `Linear` weights can be 8-bit quantized objects (`ops.QuantizedWeight`) inside the frozen group:
`mx.value_and_grad` walks the argument and leaves them alone. `test_gradients_over_a_quantized_base` quantizes 21
weights of a tiny model and compares with the same base holding the matrices those weights stand for: the loss to
1e-5 and every gradient tensor to 1e-4 (8.9e-6 measured). The comparison is against the dequantized matrices, not the
originals, from which the gradient is 1% away.

### 7.5 Who gets a gradient, and the sign

- `test_no_gradient_for_the_base_or_the_assistant`: in torch only the trainable adapter's parameters hold a gradient
  after `backward`. Ours returns the trainable group's keys and nothing else. The check is not empty: differentiated
  on purpose, the assistant's matrices and a base weight have non-zero gradients here, so a function that
  differentiated every weight would return them.
- `test_the_targets_sign_reaches_the_loss_and_the_gradient`: the torch side's target is not `x_0 − noise` but the
  velocity the scheduler's step needs, solved from the step equation in float64 (§3.2). Ours matches it. With the
  example's target negated the loss moves by 8% and the gradient is 1.25 away, relative. (A random model's output is
  nearly uncorrelated with the target, so the loss alone is a weak witness of the sign; the gradient is not.)

One thing about MLX learned here: `mx.value_and_grad` walks every argument it is handed, and it rebuilds the tuples
of adapter names in the dict of constants. The arrays in it are left as the same objects.

### 7.6 The tests

| test | pins |
| --- | --- |
| `test_loss_and_gradients_are_torchs` (×8) | float32 loss and every gradient tensor against torch and PEFT |
| `test_at_pefts_init_the_gradient_reaches_lora_b_alone` | the start of training: zero `lora_B` |
| `test_weight_scales_the_loss_and_the_gradient` | the loss weight |
| `test_no_gradient_for_the_base_or_the_assistant` | gradients for the trainable group only |
| `test_the_targets_sign_reaches_the_loss_and_the_gradient` | the target's sign, against a target solved from the step equation |
| `test_gradients_over_a_bf16_base_track_torchs` (×3) | bf16 base, float32 factors: MLX's error against torch's own |
| `test_products_with_float32_matrices_are_float32_over_a_bf16_base` (×3) | the trainable products are float32: an adapter that bf16 products get 61% to 70% wrong |
| `test_the_training_forward_is_the_sampling_forward` (×9) | the staged forward's loss is `ops.dit_forward`'s: bit for bit in float32 and over a bf16 base at a head width of 64, within 0.2% at 32 |
| `test_gradients_through_sharp_attention_over_a_bf16_base` (×6) | scores in the tens of thousands over a bf16 base: the loss bit for bit, the gradient against a by-hand float32 attention, and the bf16 call's a hundred times off at least |
| `test_gradients_through_sharp_attention_in_float32_are_torchs` (×6) | the same in float32 against torch and PEFT |
| `test_training_attention_backward_is_torchs_at_every_logit_scale` (×12) | the op in bf16 and float16 against torch's float32 gradient, at scores of 140, 13,000 and 119,000, on both devices |
| `test_the_sampling_attentions_backward_in_bf16_goes_wrong_as_logits_grow` | what the bf16 call does at a score of 119,000 on MLX 0.32.4: a hundred times too large on the default device, wrong for q and k on the CPU stream |
| `test_training_attention_forward_is_the_sampling_attentions` (×4), `test_where_the_bf16_call_computes_in_bf16_the_training_attention_is_close_not_equal` (×2) | the forward: the same bits at head widths of 64 and 128, close and more exact at 32 |
| `test_training_attention_dtypes` (×3), `test_training_attention_takes_a_mask` | float32 untouched; half precision in and out; the mask |
| `test_the_sampling_forward_keeps_the_half_precision_call` | sampling does not run the training attention |
| `test_rms_norm_backward_in_bf16_is_torchs` (×32) | the RMS norm's bf16 backward at extreme inputs, on both devices |
| `test_the_training_gathers_gradient_is_torchs_index_select` (×3) | the gather's gradient against torch's, bit for bit in bf16 and float16; the two gathers' forward values the same bits |
| `test_a_tables_gradient_over_a_bf16_base_is_the_rows_sum` | the whole gradient at 2,316 rows against a reference with nothing to sum |
| `test_the_loss_refuses_matrices_the_forward_would_not_read`, `test_the_loss_refuses_a_text_of_the_wrong_shape` | what `check_inputs` refuses, and what each case did without it |
| `test_gradients_over_a_quantized_base` | a quantized base as a constant of the differentiation |

## 8. A backward pass that does not hold every block at once

`loss_and_gradients` builds one graph for the whole forward and differentiates it. Every block's intermediates are
alive until the backward pass reaches that block, so the peak grows with the number of blocks. `h3mlx.backward` has
two other ways to get the same loss and the same gradients. There is no library original for either: the oracle is
§7's gradient, which is itself tested against torch.

### 8.1 Each block checkpointed

📍 [`src/h3mlx/backward.py:41-57`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/backward.py#L41-L57)
```python
  41 def checkpointed_loss(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None) -> mx.array:
  42     """`train.example_loss` with each block run as an `mx.checkpoint` of (the block's input, the time embedding, the
  43     block's trainable matrices). The same operations, so the same loss bit for bit."""
  44     check_inputs(trainable, frozen, cfg, example, text)
  45     p = {**frozen, **trainable}
  46     forward = ExampleForward(cfg, example, text, weight)
  47     x, temb = forward.start(p)
  48     for i in range(cfg["num_layers"]):
  49         prefix = f"{BLOCK}{i}."
  50         names = [k[len(prefix):] for k in trainable if k.startswith(prefix)]
  51         constants = ops._sub(frozen, prefix)
  52
  53         def block(x, temb, *matrices, constants=constants, names=names):
  54             return forward.block(x, temb, {**constants, **dict(zip(names, matrices))})
  55
  56         x = mx.checkpoint(block)(x, temb, *[trainable[prefix + k] for k in names])
  57     return forward.finish(x, temb, p)
```
📍 [`src/h3mlx/backward.py:60-65`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/backward.py#L60-L65)
```python
  60 def loss_and_gradients_checkpointed(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array,
  61                                     weight=None):
  62     """(loss, gradients) as `train.loss_and_gradients` returns them, with each block checkpointed. One lazy graph:
  63     nothing is evaluated here, and the caller's evaluation must not come before the gradients are built."""
  64     check_inputs(trainable, frozen, cfg, example, text)
  65     return mx.value_and_grad(checkpointed_loss)(trainable, frozen, cfg, example, text, weight)
```

`mx.checkpoint` wraps a function so that its intermediates are not kept for the backward pass; they are recomputed
from the function's inputs when the backward pass gets there. Two facts about it decide how it is used here, and each
has a minimal test:

- **A checkpointed function gives no gradient to what it captures.** It differentiates with respect to its arguments.
  An array it reads from the enclosing scope gets a gradient of exactly zero, with no error, even when the outer
  function is differentiated with respect to that array
  (`test_a_checkpointed_function_gives_no_gradient_to_what_it_captures`). So the block's input, the time embedding and
  the block's trainable matrices are arguments. The block's frozen weights are captured on purpose: zero is the
  gradient wanted for them, and they are constants either way.
- **An `mx.eval` inside the loop defeats it.** `test_an_eval_in_the_loop_defeats_the_checkpoint` differentiates a
  chain of sixteen blocks four ways on the CPU stream. Peak memory over the start: plain 86 MB, checkpointed 55 MB;
  with an `mx.eval` after each block, as the sampling forward has, 185 MB plain and 185 MB checkpointed. Evaluated
  mid-trace, each block's output keeps its graph, and the checkpoint has nothing left to drop. This is why §7.1's
  forward evaluates nothing.

### 8.2 Block by block, by hand

📍 [`src/h3mlx/backward.py:68-125`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/backward.py#L68-L125)
```python
  68 def loss_and_gradients_blockwise(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array,
  69                                  weight=None, after_stage=None):
  70     """(loss, gradients) as `train.loss_and_gradients` returns them, one block at a time; evaluated on return.
  71     `after_stage()`, when given, is called after every evaluation (each block on the way up and on the way back):
  72     the place for a caller to look at memory and stop by raising.
  73
  74     Forward: the stages run in order, each evaluated, and each block's input is kept. Backward: the vector-Jacobian
  75     product of the last stage gives the cotangent of the last block's output; each block, last to first, turns the
  76     cotangent of its output into those of its input, of the time embedding and of its own trainable matrices, and is
  77     evaluated before the next; the stage before the blocks takes what is left. One block's graph is alive at a time,
  78     and what grows with depth is the kept inputs, one per block. Each block's forward runs twice (on the way up, and
  79     again inside its product)."""
  80     check_inputs(trainable, frozen, cfg, example, text)
  81     p = {**frozen, **trainable}
  82     forward = ExampleForward(cfg, example, text, weight)
  83     head_keys, block_keys, tail_keys = _stages(trainable, cfg)
  84
  85     def start(*matrices):
  86         return list(forward.start({**p, **dict(zip(head_keys, matrices))}))
  87
  88     def block(i):
  89         prefix = f"{BLOCK}{i}."
  90         constants = ops._sub(frozen, prefix)
  91         names = [k[len(prefix):] for k in block_keys[i]]
  92         return lambda x, temb, *matrices: forward.block(x, temb, {**constants, **dict(zip(names, matrices))})
  93
  94     def finish(x, temb, *matrices):
  95         return forward.finish(x, temb, {**p, **dict(zip(tail_keys, matrices))})
  96
  97     after_stage = after_stage or (lambda: None)
  98     x, temb = start(*[trainable[k] for k in head_keys])
  99     mx.eval(x, temb)
 100     after_stage()
 101     inputs = [x]
 102     for i in range(cfg["num_layers"]):
 103         x = block(i)(inputs[-1], temb, *[trainable[k] for k in block_keys[i]])
 104         mx.eval(x)
 105         after_stage()
 106         inputs.append(x)
 107
 108     (loss,), (g_x, g_temb, *g_tail) = mx.vjp(finish, [inputs.pop(), temb, *[trainable[k] for k in tail_keys]],
 109                                              [mx.array(1.0, dtype=mx.float32)])
 110     mx.eval(loss, g_x, g_tail)
 111     after_stage()
 112     grads = dict(zip(tail_keys, g_tail))
 113     for i in reversed(range(cfg["num_layers"])):
 114         _, (g_x, g_t, *g_block) = mx.vjp(block(i), [inputs.pop(), temb, *[trainable[k] for k in block_keys[i]]], [g_x])
 115         if head_keys:  # the time embedding's cotangent is for the stage before the blocks: left unevaluated without one
 116             g_temb = g_temb + g_t
 117             mx.eval(g_temb)
 118         mx.eval(g_x, g_block)
 119         after_stage()
 120         grads.update(zip(block_keys[i], g_block))
 121     if head_keys:
 122         _, g_head = mx.vjp(start, [trainable[k] for k in head_keys], [g_x, g_temb])
 123         mx.eval(g_head)
 124         grads.update(zip(head_keys, g_head))
 125     return loss, {k: grads[k] for k in trainable}
```

The blocks are walked forward, each evaluated, keeping each block's input. Then the last stage's vector-Jacobian
product gives the cotangent of the last block's output, and each block in turn, last to first, turns the cotangent of
its output into the cotangent of its input, of the time embedding, and of its own trainable matrices, and is
evaluated before the next. The stage before the blocks takes what is left: the cotangent of the first block's input
and the time embedding's, summed over the blocks.

- **What it keeps.** One block's graph at a time, and one saved input per block: rows × hidden values in the
  base's dtype.
- **What it costs.** Each block's forward runs twice, once on the way up and once inside its product.
- **What it skips.** With no trainable matrix before the blocks, nothing needs the time embedding's cotangent, and it
  is never evaluated.
- **A place to stop.** `after_stage`, when given, is called after every evaluation. The training loop looks at MLX's
  memory counter there and stops by raising (§11.7).

### 8.3 Both are the plain gradient

`test_bounded_gradients_equal_the_plain_gradient` compares each way's loss and every gradient tensor with §7's, in
float32 and over a bf16 base with float32 factors, with a trainable adapter alone and beside a frozen assistant, with
an adapter on every `Linear` (so the stages before and after the blocks have trainable matrices too) and with one on
the block linears only. The keys are the trainable group's: nothing for the assistant or the base.

- **Tolerance:** 1e-5 per gradient tensor, relative. Measured: every tensor exactly equal, except those a modulation
  table's gradient feeds (§7.1), which are within 3e-7.
- **One case with more room.** Over a bf16 base on the default device those table-fed tensors are allowed 2e-3, for
  the rounding step §7.1 describes. None moved in the runs measured, and the same cases hold at 1e-5 on the CPU
  stream.

### 8.4 Memory, measured on a synthetic model

`bench/train_backward_memory.py` builds a synthetic transformer (hidden size 96, random weights in the release's
mixed precision, a rank-4 float32 adapter on the block linears, one text-to-image example) and measures the peak of
MLX's active memory during one gradient, over what was active before it, each cell in a process of its own. Four
ways: through the sampling forward with its evals, plain, checkpointed, block by block. On an M2 Max, in MB:

| 596 rows | 4 blocks | 8 blocks | 16 blocks | growth per block |
| --- | ---: | ---: | ---: | ---: |
| default device: through the sampling forward | 69 | 136 | 271 | 16.9 |
| default device: plain | 84 | 131 | 190 | 8.9 |
| default device: checkpointed | 88 | 91 | 115 | 2.2 |
| default device: block by block | 35 | 37 | 37 | 0.2 |
| CPU stream: through the sampling forward | 80 | 159 | 316 | 19.7 |
| CPU stream: plain | 43 | 76 | 139 | 8.0 |
| CPU stream: checkpointed | 23 | 23 | 46 | 1.9 |
| CPU stream: block by block | 21 | 21 | 22 | 0.1 |

| 2,324 rows | 4 blocks | 8 blocks | 16 blocks | growth per block |
| --- | ---: | ---: | ---: | ---: |
| default device: through the sampling forward | 468 | 784 | 1564 | 91.4 |
| default device: plain | 780 | 1032 | 1625 | 70.4 |
| default device: checkpointed | 652 | 653 | 741 | 7.4 |
| default device: block by block | 235 | 237 | 234 | -0.1 |
| CPU stream: through the sampling forward | 437 | 871 | 1739 | 108.5 |
| CPU stream: plain | 362 | 615 | 1118 | 63.0 |
| CPU stream: checkpointed | 182 | 186 | 273 | 7.6 |
| CPU stream: block by block | 175 | 180 | 182 | 0.6 |

What the numbers show, and what they do not:

- **Differentiating the sampling forward as it is grows fastest,** on both devices. Taking the evals out of the
  forward (plain) cuts the growth to between two fifths and about three quarters. The plain gradient starts higher
  than it did, because the training forward's attention is float32 (§7.1) and at this model's head width MLX then
  holds float32 scores, 43 MB a block at 2,324 rows: at 16 blocks and 2,324 rows on the default device it is the
  largest number in its table (1,625 MB against 1,564). The row through the sampling forward is what it was, its
  attention still the bf16 call.
- **Block by block is nearly flat.** Its growth per block is the saved input: 596 rows × 96 × 2 bytes is 0.11 MB, and
  2,324 rows is 0.45 MB.
- **Checkpointing saves less, and less predictably.** On this model it grows more slowly than plain on both devices.
  On the sixteen-block chain of §8.1, which the script also prints, it saves a third on the CPU stream (86 MB plain,
  55 MB checkpointed) and nothing on the default device (195 MB plain, 209 MB checkpointed), where one evaluation
  holds its buffers until the device has run them. What it saves depends on the device and on the graph.
- **Time.** At 16 blocks the two bounded ways took 1.2 to 1.8 times the plain gradient's time on the default device
  and 1.2 to 1.4 times on the CPU stream (one run each; the script prints the seconds).
- **The default device's numbers move between runs,** by tens of MB in some cells: before the attention's change,
  the checkpointed gradient at 4 blocks and 596 rows measured 54 MB in one run and 76 MB in another. The CPU
  stream's repeat.
- **The number is MLX's counter.** `mx.get_peak_memory()` counts active buffers; what MLX holds in its buffer cache
  is not in it.
- **Nothing here predicts 50 real blocks.** The real model has hidden size 5,376, tens of thousands of rows and an
  attention width of its own. Which way fits in memory there, and how long it takes, has to be measured there.

`test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept` asserts directions only, from 4 blocks to 16, as
ratios of growth:

- on the default device, the block-by-block gradient's peak grows by less than a twentieth of the plain one's;
- on the CPU stream, where the numbers repeat: that again; the checkpointed gradient's growth is below half the
  plain one's; and the plain one's is below 0.6 of the growth through the sampling forward.

The checkpointed path is not asserted on the default device.

### 8.5 The tests

| test | pins |
| --- | --- |
| `test_checkpointed_loss_is_the_loss` (×2) | checkpointing changes no value: the loss bit for bit |
| `test_bounded_gradients_equal_the_plain_gradient` (×24) | both ways against the plain gradient: float32 and bf16 base, one and two adapters, adapters on every linear and on the block linears |
| `test_bounded_gradients_through_sharp_attention_over_a_bf16_base` (×4) | attention scores in the tens of thousands over a bf16 base: each way is the plain gradient exactly, and with the bf16 attention call it is a thousand times as large or not finite |
| `test_bounded_gradients_at_other_levels_and_layouts` (×2) | text-to-image and two references, a shared timestep, a loss weight |
| `test_bounded_gradients_over_a_quantized_base` (×2) | quantized weights captured by a checkpoint, and as constants of each product |
| `test_bounded_gradients_refuse_what_the_loss_refuses` (×3) | `check_inputs` before anything is computed |
| `test_a_config_of_fewer_blocks_than_the_weights_is_refused` | weights for four blocks under a config of two: refused by all three gradients, and what they returned unchecked |
| `test_the_blockwise_gradient_reports_after_every_stage` | the hook a trainer's memory limit uses: called after each evaluation, and an exception in it stops the gradient |
| `test_stages_cover_every_trainable_key_once` | which stage of the forward each trainable matrix belongs to |
| `test_a_checkpointed_function_gives_no_gradient_to_what_it_captures` | the first fact about `mx.checkpoint` |
| `test_an_eval_in_the_loop_defeats_the_checkpoint` | the second |
| `test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept` | the directions of the memory measurement |

## 9. The optimizer step

📍 [`src/h3mlx/train.py:322-342`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L322-L342)
```python
 322 def adamw(*, learning_rate: float, betas, eps: float, weight_decay: float) -> optim.AdamW:
 323     """MLX's AdamW set up to compute `torch.optim.AdamW`'s update: with `bias_correction=True`. MLX leaves the bias
 324     correction out by default, and its first step is then `(1 - beta1) / sqrt(1 - beta2)` times torch's, 3.2 times
 325     for betas of (0.9, 0.999). No hyper-parameter has a default here: every one is the caller's to give.
 326
 327     The arguments are checked as torch's constructor checks them, which MLX's does not: a learning rate and a weight
 328     decay of at least 0, betas in [0, 1), none of them NaN. Three things are stricter than torch's: they must be
 329     finite; the learning rate is a number (a schedule is not built); and `eps` must be above 0. torch takes an `eps`
 330     of 0, and so does the arithmetic until a gradient element is exactly zero: the update is then 0 / 0. That is
 331     every element of `lora_A` on the first step from PEFT's initialisation, where `lora_B` is zero."""
 332     for what, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay)):
 333         if not _real(value) or not math.isfinite(value) or value < 0:
 334             raise ValueError(f"{what} must be a finite number of at least 0, got {value!r}")
 335     betas = list(betas)
 336     if len(betas) != 2 or not all(_real(b) and 0 <= b < 1 for b in betas):
 337         raise ValueError(f"betas must be two numbers in [0, 1), got {betas!r}")
 338     if not _real(eps) or not math.isfinite(eps) or eps <= 0:
 339         raise ValueError(f"eps must be a finite number above 0, got {eps!r}: with 0, a gradient element that is "
 340                          "exactly zero is updated by 0 / 0")
 341     return optim.AdamW(learning_rate=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay,
 342                        bias_correction=True)
```
📍 [`src/h3mlx/train.py:345-385`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L345-L385)
```python
 345 def optimizer_step(optimizer, trainable: dict, grads: dict, *, max_grad_norm: float) -> tuple[dict, float]:
 346     """One step: (the updated matrices, the gradients' global norm before clipping), both evaluated.
 347
 348     The gradients are scaled down to a global norm of `max_grad_norm` when theirs is larger, as torch's
 349     `clip_grad_norm_` scales them (`math.inf` clips nothing), and the optimizer applies them. Refused before any
 350     matrix is touched:
 351
 352     * matrices that are not float32. At a learning rate of 1e-4 a bf16 element of magnitude 2**-5 or more does not
 353       move at all, because the update is under half its spacing, and a smaller one moves by whole steps of its
 354       spacing; in float16 the second moment of small gradients underflows and the update is NaN. So the trainable
 355       matrices are float32 over whatever base (`lora.attach(..., keep_dtype=True)`).
 356     * gradients whose keys are not the matrices' keys, or that are not float32 arrays of their matrix's shape. A
 357       gradient of another shape that broadcasts would be applied, and would leave moments of its own shape in the
 358       optimizer. A bf16 gradient would have its norm summed in bf16 by MLX's clip (which widens float16 and not
 359       bf16), a sum that stalls, and would leave bf16 moments;
 360     * a `max_grad_norm` that is not a positive number;
 361     * a gradient norm that is not finite, which torch's clip refuses under `error_if_nonfinite` and otherwise lets
 362       through into the weights."""
 363     if not trainable or set(grads) != set(trainable):
 364         raise ValueError(f"the gradients' keys must be the trainable matrices' keys: {len(grads)} against {len(trainable)}")
 365     narrow = sorted(k for k, v in trainable.items() if v.dtype != mx.float32)
 366     if narrow:
 367         raise ValueError(f"the trainable matrices must be float32; {len(narrow)} are not, e.g. {narrow[0]} "
 368                          f"({trainable[narrow[0]].dtype}): half-precision matrices do not take updates of this size")
 369     wrong = sorted(k for k, v in trainable.items()
 370                    if not isinstance(grads[k], mx.array) or grads[k].dtype != mx.float32 or grads[k].shape != v.shape)
 371     if wrong:
 372         g = grads[wrong[0]]
 373         found = f"{g.dtype} {tuple(g.shape)}" if isinstance(g, mx.array) else type(g).__name__
 374         raise ValueError(f"each gradient must be a float32 array of its matrix's shape; {len(wrong)} are not, e.g. "
 375                          f"{wrong[0]}: {found} for a matrix of {tuple(trainable[wrong[0]].shape)}")
 376     if not isinstance(max_grad_norm, numbers.Real) or isinstance(max_grad_norm, bool) or not max_grad_norm > 0:
 377         raise ValueError(f"max_grad_norm must be a positive number (math.inf to clip nothing), got {max_grad_norm!r}")
 378     clipped, norm = optim.clip_grad_norm(grads, float(max_grad_norm))
 379     norm = float(norm)
 380     if not math.isfinite(norm):
 381         raise ValueError(f"the gradients' norm is {norm}: not applied")
 382     updated = optimizer.apply_gradients(clipped, trainable)
 383     updated = {k: updated[k] for k in trainable}
 384     mx.eval(updated, optimizer.state)
 385     return updated, norm
```

The original is torch's: `torch.optim.AdamW` after `torch.nn.utils.clip_grad_norm_`. The port uses MLX's own
`AdamW` and `clip_grad_norm`, whose formulas are the same ones (decoupled weight decay, then the bias-corrected
update; the gradients scaled by `min(1, max_norm / (norm + 1e-6))`), and checks the result against torch over a run
of steps. **No hyper-parameter has a default.** The learning rate, the betas, epsilon, the weight decay and the bound
on the gradient's norm are keyword arguments the caller must give; the recipe's values for them were not read.

**The arguments are checked as torch checks them.** torch's constructor refuses a negative or NaN learning rate,
weight decay or epsilon, and betas outside [0, 1). MLX's refuses none of these, and a NaN learning rate goes straight
into every weight. `train.adamw` refuses what torch refuses and three things torch takes
(`test_adamw_checks_its_arguments_as_torch_does_and_eps`):

- an infinite value;
- a learning rate that is not a number (MLX takes a schedule there; none is built here);
- **an epsilon of zero.** A gradient element that is exactly zero has both of Adam's moments zero, and its update is
  0 / (0 + eps). With an epsilon of zero that is NaN, in torch and in MLX alike
  (`test_an_epsilon_of_zero_divides_zero_by_zero` runs both). Exact zeros are not rare here: from PEFT's
  initialisation every element of `lora_A` has one on the first step, because `lora_B` is zero.

### 9.1 Twelve steps against torch

`test_steps_are_torchs` runs twelve steps over three examples in turn (text-to-image, one reference and two, at
three noise levels), with a frozen assistant beside the trained adapter, in float32. After every step it compares the
loss (to 1e-5), the gradient's norm before clipping (to 1e-5) and how far the matrices have moved from their start
(to 1e-4, relative, over all of them). Four runs: two settings in which every hyper-parameter differs, from PEFT's
initialisation and from random matrices.

| setting | start | movement from torch's, worst step | steps clipped |
| --- | --- | ---: | ---: |
| rate 1e-3, betas (0.9, 0.999), eps 1e-8, decay 0.01 | PEFT's initialisation | 2.1e-5 | 5 of 12 |
| rate 3e-4, betas (0.8, 0.95), eps 1e-3, decay 0.1 | PEFT's initialisation | 7.8e-7 | 4 of 12 |
| rate 1e-3, betas (0.9, 0.999), eps 1e-8, decay 0.01 | random matrices | 7.8e-6 | 8 of 12 |
| rate 3e-4, betas (0.8, 0.95), eps 1e-3, decay 0.1 | random matrices | 3.6e-6 | 8 of 12 |

- **The clipping is part of what is compared.** Adam's update barely depends on the gradient's scale, so a wrong
  clip applied on every step would hardly show. The bound is therefore set between the norms the three examples
  give, so that some steps are clipped and some are not, and the second setting has an epsilon of the gradients'
  size. `test_clipping_is_torchs` also checks the clipped gradients themselves, through one step of plain gradient
  descent, below the norm, above it and with no bound.
- **It trains.** In all four runs the first example's loss is lower at its fourth visit than at its first, on both
  sides.

**Over a bf16 base** the two sides' gradients are 1% apart (§7.3), and Adam's first steps move every element by the
learning rate in the direction of its gradient's sign, so an element whose gradient is near zero can go the other
way. `test_steps_over_a_bf16_base_stay_near_torchs` asserts little, on purpose: the losses within 0.5% at every step
(6.9e-4 measured) and the movement within 25% of torch's (2% to 12% measured). Step for step, a run over a bf16 base
is not comparable with torch's.

### 9.2 Bias correction

MLX's `AdamW` leaves the bias correction out unless asked: `bias_correction=False` is its default. torch always
applies it. On the first step torch moves every element by the learning rate, and MLX's default moves it by
(1 − β₁) / √(1 − β₂) times that, 3.16 times for betas of (0.9, 0.999).
`test_mlx_adamw_needs_bias_correction_to_be_torchs` measures that ratio and checks five steps of `train.adamw`
against torch's on one matrix.

### 9.3 Why the trained matrices are float32

`optimizer_step` refuses matrices that are not float32. `test_half_precision_matrices_do_not_take_updates_of_1e_4`
shows why, with ten AdamW steps at a learning rate of 1e-4 and a fixed gradient:

| dtype, magnitude of the elements | what happens |
| --- | --- |
| float32 | every element moves by 1e-3, ten steps' worth |
| bf16, 2⁻⁵ (0.031) and up | not one element changes: bf16's spacing there is 2.4e-4 or more, and an update of 1e-4 is under half of it |
| bf16, 2⁻⁶ to 2⁻⁵ | every element moves by one whole spacing a step, 1.2e-4: 22% too far |
| bf16, zero (`lora_B`'s start) | moves by the learning rate |
| float16 | the second moment of a gradient of 1e-2 is 1e-7, below float16's smallest normal number: the update is NaN or infinite for most elements |

So half-precision matrices are not uniformly stuck. Small elements move, coarsely, and stop moving as they grow; in
float16 the optimizer's own state breaks first.

### 9.4 What the step refuses

- matrices that are not float32 (§9.3);
- gradients whose keys are not the matrices' keys;
- **gradients that are not float32 arrays of their matrix's shape.** MLX's optimizer would apply both kinds
  (`test_two_gradients_the_library_would_apply`, `test_why_a_gradient_must_be_float32`):
  - a gradient of another shape broadcasts. One of shape (4, 2, 3) for a matrix of (2, 3) returns a matrix of
    (4, 2, 3) and advances the step count; torch refuses to assign such a gradient at all;
  - a float16 gradient of 0.001 on a float32 matrix, with ordinary settings, has a finite norm and an update of
    minus infinity: its second moment and the epsilon are both zero in float16;
  - a bf16 gradient has its norm summed in bf16 by MLX's clip, which widens float16 gradients and not bf16 ones. The
    sum stalls: 624 matrices' norm comes out as 45.25 for an exact 72.42, so the clip would be that much too weak.
  With float32 gradients the clip's sum is float32, and stays the library's: on the same 624 matrices it is within
  7e-6 of the exact norm, and over the twelve steps of §9.1 it is torch's to 1e-5;
- a bound on the norm that is not a positive number (`math.inf` clips nothing);
- a gradient norm that is not finite. torch's clip raises for one only under `error_if_nonfinite`, and otherwise
  the NaN goes into the weights. Here it is always an error, raised before the optimizer's state is touched.

### 9.5 Saving the optimizer's state

📍 [`src/h3mlx/train.py:388-400`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L388-L400)
```python
 388 def optimizer_state(optimizer, trainable: dict) -> dict:
 389     """The optimizer's state as a flat dict of arrays, for a checkpoint: the step count and, for each trainable
 390     matrix, Adam's two moments, under `m.<key>` and `v.<key>`. The learning rate is not state here: it is constant
 391     and an argument (`adamw`). Before the first step there is no state to save, which is an error."""
 392     state = optimizer.state
 393     missing = sorted(k for k in trainable if k not in state or set(state[k]) != {"m", "v"})
 394     if missing or "step" not in state:
 395         raise ValueError(f"the optimizer has no Adam state for {len(missing)} of {len(trainable)} matrices: it has not "
 396                          "stepped on them yet")
 397     out = {"step": mx.array(int(state["step"].item()), dtype=mx.uint64)}
 398     for key in trainable:
 399         out["m." + key], out["v." + key] = state[key]["m"], state[key]["v"]
 400     return out
```
📍 [`src/h3mlx/train.py:403-425`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/train.py#L403-L425)
```python
 403 def restore_optimizer(optimizer, trainable: dict, saved: dict) -> None:
 404     """Put a saved state (`optimizer_state`) into a new optimizer made with the same hyper-parameters, so that its
 405     next step is the step the saved one would have taken. The saved moments must be exactly the trainable matrices',
 406     in their shapes and float32."""
 407     want = {"step"} | {f"{moment}.{key}" for key in trainable for moment in "mv"}
 408     if set(saved) != want:
 409         raise ValueError(f"the saved optimizer state is not for these matrices: {len(set(saved) ^ want)} names differ, "
 410                          f"e.g. {sorted(set(saved) ^ want)[:2]}")
 411     step = int(np.array(saved["step"]).item())
 412     if step < 1:
 413         raise ValueError(f"a saved optimizer state has taken at least one step, got {step}")
 414     for key, matrix in trainable.items():
 415         for moment in "mv":
 416             got = saved[f"{moment}.{key}"]
 417             if got.shape != matrix.shape or got.dtype != mx.float32:
 418                 raise ValueError(f"{moment}.{key}: saved as {got.dtype} {tuple(got.shape)}, the matrix is float32 "
 419                                  f"{tuple(matrix.shape)}")
 420     optimizer.init(trainable)
 421     state = optimizer.state
 422     state["step"] = mx.array(step, dtype=state["step"].dtype)
 423     for key in trainable:
 424         state[key] = {"m": saved["m." + key], "v": saved["v." + key]}
 425     mx.eval(state)
```

The state is the step count and Adam's two moments for each matrix; the learning rate is an argument, not state.
`test_a_saved_optimizer_goes_on_where_it_stopped` takes six steps in one go, and three steps, a round trip of the
state through a safetensors file into a new optimizer, and three more: the same matrices bit for bit. A new optimizer
without the state does not give them, and neither do the moments with the wrong step count, which sets the bias
correction.

### 9.6 The tests

| test | pins |
| --- | --- |
| `test_steps_are_torchs` (×4) | twelve steps against `torch.optim.AdamW` after `clip_grad_norm_`: losses, norms, and the matrices' movement |
| `test_steps_over_a_bf16_base_stay_near_torchs` | the same loop over a bf16 base, loosely |
| `test_clipping_is_torchs` (×4) | the clipped gradients and the returned norm against torch's |
| `test_mlx_adamw_needs_bias_correction_to_be_torchs` | MLX's default against torch's first step; `train.adamw` against five |
| `test_half_precision_matrices_do_not_take_updates_of_1e_4` | bf16 and float16 under AdamW at 1e-4 |
| `test_optimizer_step_refusals` | what the step refuses, with the optimizer's state untouched |
| `test_two_gradients_the_library_would_apply`, `test_why_a_gradient_must_be_float32` | a gradient of another shape, a float16 one and a bf16 one: what MLX does with each, and that the step refuses them |
| `test_adamw_checks_its_arguments_as_torch_does_and_eps` (×14), `test_an_epsilon_of_zero_divides_zero_by_zero` | torch's refusals, and the three that go further |
| `test_a_saved_optimizer_goes_on_where_it_stopped`, `test_optimizer_state_refusals` | the state through a file and back; a state that is not these matrices' |
| `test_no_hyper_parameter_has_a_default` | every hyper-parameter keyword-only and required |

## 10. A cache of encoded examples

Encoding a picture and a prompt takes the VAE and the text encoder; a training step needs neither. So each example
is encoded once and kept: an entry holds what `build_example` (§6.2) takes, namely the picture's clean latents, each
reference picture's latents, the prompt's text rows and their tags. There is no library original for this piece.
What it takes from elsewhere are two rules for sizing a reference picture: diffusers' and the recipe's.

### 10.1 The request is the key

📍 [`src/h3mlx/cache.py:149-196`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L149-L196)
```python
 149 def request(*, picture: str, references, prompt: str, height: int, width: int, latent_seed: int,
 150             latent_float16_round: bool, reference_sizing: str, reference_seed: int, reference_float16_round: bool,
 151             vae_dtype: str, text_dtype: str, settings: dict, encoders: dict) -> dict:
 152     """What one entry is asked to be. Every argument is required and none has a default.
 153
 154     `picture` and each of `references` are sha256 digests of the input files; `prompt` is the text as given.
 155     `height` and `width` are the size the picture is trained at. `latent_seed` and `latent_float16_round` are the
 156     seed of the picture's posterior sample and whether it is rounded through float16 before normalising; the
 157     `reference_...` pair is the same for the reference pictures, and `reference_sizing` is one of `SIZING`.
 158     `settings` holds the release's `reference_min_pixels`, `reference_max_pixels` and `vision_pixels` and the
 159     `canvas_multiple`; `encoders` holds an identity (`identity`) for each of `ENCODERS`.
 160
 161     An example without references does not depend on the reference choices: its request leaves them out, and they
 162     may be given as None."""
 163     references = [_digest(f"references[{i}]", r) for i, r in enumerate(references)]
 164     if not isinstance(prompt, str) or not prompt.strip():
 165         raise ValueError("the prompt must be a string with something in it")
 166     if set(settings) != set(SETTINGS) or not all(isinstance(settings[k], int) and settings[k] > 0 for k in SETTINGS):
 167         raise ValueError(f"settings must hold positive integers for exactly {SETTINGS}, got {settings!r}")
 168     multiple = settings["canvas_multiple"]
 169     for what, side in (("height", height), ("width", width)):
 170         if isinstance(side, bool) or not isinstance(side, int) or side <= 0 or side % multiple:
 171             raise ValueError(f"{what} must be a positive multiple of {multiple}, got {side!r}")
 172     for what, dtype in (("vae_dtype", vae_dtype), ("text_dtype", text_dtype)):
 173         if dtype not in DTYPES:
 174             raise ValueError(f"{what} is one of {sorted(DTYPES)}, got {dtype!r}")
 175     if set(encoders) != set(ENCODERS) or not all(isinstance(v, str) and v for v in encoders.values()):
 176         raise ValueError(f"encoders must name an identity for exactly {ENCODERS}, got {sorted(encoders)}")
 177     reference = None
 178     if references:
 179         if reference_sizing not in SIZING:
 180             raise ValueError(f"reference sizing is one of {SIZING}, got {reference_sizing!r}")
 181         _seed("reference_seed", reference_seed)
 182         _flag("reference_float16_round", reference_float16_round)
 183         reference = {"sizing": reference_sizing, "seed": reference_seed, "float16_round": reference_float16_round,
 184                      "resize": RESIZE, "min_pixels": settings["reference_min_pixels"],
 185                      "max_pixels": settings["reference_max_pixels"], "vision_pixels": settings["vision_pixels"]}
 186     return {
 187         "format": FORMAT,
 188         "inputs": {"picture": _digest("picture", picture), "references": references, "prompt": prompt},
 189         "target": {"height": height, "width": width, "seed": _seed("latent_seed", latent_seed),
 190                    "float16_round": _flag("latent_float16_round", latent_float16_round), "resize": RESIZE},
 191         "references": reference,
 192         "text": {"dtype": text_dtype, "workflow": "edit" if references else "text2image"},
 193         "vae": {"dtype": vae_dtype, "canvas_multiple": multiple},
 194         "encoders": {k: encoders[k] for k in ENCODERS},
 195         "versions": versions(),
 196     }
```
📍 [`src/h3mlx/cache.py:199-200`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L199-L200)
```python
 199 def key(request: dict) -> str:
 200     return files.sha256_json(request)
```

An entry is asked for by a *request*: one JSON value with everything the entry's numbers depend on. Its sha256 is the
entry's key, and the two files are named by it. So an entry made any other way is under another name:

| part of the request | what it holds |
| --- | --- |
| inputs | the sha256 of the picture's file and of each reference's, in order; the prompt as given |
| target | the size the picture is trained at, the seed of its posterior sample, whether the sample is rounded through float16, the resampling filter |
| references | the sizing rule (§10.3), the seed and the rounding of their posterior samples, the release's pixel bounds; absent for an example without references, which then does not depend on any of it |
| text | the dtype the rows are kept in, and whether the prompt is presented as text-to-image or as an edit |
| VAE | the dtype policy it was loaded with, and the canvas multiple |
| encoders | an identity for each of the VAE, the text encoder, the connector, the tokenizer and the processor |
| versions | a digest of this package's source, and the versions of MLX, mlx-vlm, numpy, torch, torchvision, Pillow, transformers, tokenizers and safetensors |

No argument of `request` has a default.

- **The code version** is `code_digest`: the name and sha256 of every `.py` file of the package. It is deliberately
  wide. Any change to the package, a comment included, makes every entry a miss, and encoding again is cheap beside
  training on entries made by other code.
- **The digest covers the package and not the scripts,** and that is enough because a script cannot change what an
  entry holds for a given request. How a picture file is opened and oriented (`load_picture`), how the encoders are
  loaded (`load_encoders`) and how they are run (`encode`) are all in the package. What the training script does
  decide, each picture's posterior seed, the sizes and the choices, goes into the request and so into the key
  (`test_what_turns_a_request_into_an_entry_is_in_the_package`).
- **The libraries** are the ones the encode path computes with: mlx-vlm runs the text encoder, torchvision is the
  processor's image backend, torch draws the posterior's noise, Pillow opens and resizes the pictures, the tokenizer
  tokenizes.
- **An encoder's identity** is `identity`: each small file by its sha256, each `.safetensors` file by its name and
  size.
  - It notices a different release, a changed config, tokenizer or processor file, a file added, removed or renamed,
    and a weight file of another size.
  - **It does not read the weights,** and so it does not notice a weight file replaced by a different one of the same
    name and size. Hashing the weights at every start would; that is not done.
  - It ignores hidden files and everything under a hidden directory. A download tool keeps bookkeeping of its own
    there, inside the folders it fills, and that changes without the component changing: counted, it would make
    every entry a miss and every resume a refusal for no reason.

📍 [`src/h3mlx/cache.py:53-60`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L53-L60)
```python
  53 def code_digest(package: Path | None = None) -> str:
  54     """A digest of this package's source: every `.py` file's name and bytes. Any change to the code gives another
  55     digest, so an entry is never read by code other than the code that wrote it."""
  56     package = Path(__file__).resolve().parent if package is None else Path(package)
  57     sources = sorted(p for p in package.rglob("*.py"))
  58     if not sources:
  59         raise ValueError(f"no source files in {package}")
  60     return files.sha256_json([[str(p.relative_to(package)), files.sha256_file(p)] for p in sources])
```
📍 [`src/h3mlx/cache.py:75-92`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L75-L92)
```python
  75 def identity(directory) -> str:
  76     """A digest that identifies one component of a release without reading its weights: every file's name, with the
  77     file's sha256, except that a `.safetensors` file is taken by its name and size alone.
  78
  79     * It notices: a different release, a changed config, tokenizer or processor file, a file added, removed or
  80       renamed, a weight file of another size.
  81     * It does not notice: a weight file replaced by a different one of the same name and size (hashing 66 GB at every
  82       start is the price of noticing that, and it is not paid here).
  83     * It ignores, on purpose: hidden files, and everything under a hidden directory. A download tool keeps its own
  84       bookkeeping there (`.cache/huggingface/...` inside a folder it filled), which changes without the component
  85       changing."""
  86     directory = Path(directory)
  87     listed = sorted(p for p in directory.rglob("*")
  88                     if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts))
  89     if not listed:
  90         raise ValueError(f"nothing to identify in {directory}")
  91     return files.sha256_json([[str(p.relative_to(directory)), p.stat().st_size if p.suffix == ".safetensors"
  92                                else files.sha256_file(p)] for p in listed])
```

### 10.2 Two files, written whole, read with suspicion

📍 [`src/h3mlx/cache.py:269-317`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L269-L317)
```python
 269 def write(directory, request: dict, tensors: dict, wait: float = 30.0) -> str:
 270     """Write one entry and return its key.
 271
 272     The tensors are written under a temporary name first. Then, holding the cache directory's lock (`files.exclusive`
 273     on `cache.lock`, waiting up to `wait` seconds for it), the entry is published: the tensors renamed into place,
 274     then the record written, atomically, last. So an entry is there only once it is whole, and one writer publishes
 275     at a time.
 276
 277     * An entry is written once. A key that already has a record is refused (`FileExistsError`), before the lock and
 278       again under it, in case another writer published meanwhile.
 279     * A writer that fails removes what it alone put there, and nothing else: its temporary file, and the tensors it
 280       renamed into place under the lock, where no record existed and no other writer could be. An entry that has
 281       its record is never touched.
 282     * Once the record's rename has happened the entry is published, and it stands. The one thing that can still
 283       fail is flushing the directory afterwards (`files.NotDurable`): that error is raised as it is and nothing is
 284       removed. Readers do not take the lock, so an entry that has been visible cannot be taken back without a
 285       reader seeing half of it; and what a cache loses when a write turns out not to have been durable is one
 286       entry, which is encoded again. The tensors' own rename is flushed before the record is written, so a record
 287       that is on disk has its tensors on disk.
 288     * When the lock stays held for `wait` seconds the write is refused with `files.Busy`, which says who holds it."""
 289     directory = Path(directory)
 290     arrays = check_tensors(request, tensors)
 291     name = key(request)
 292     directory.mkdir(parents=True, exist_ok=True)
 293     record_path, tensors_path = directory / f"{name}.json", directory / f"{name}.safetensors"
 294     if os.path.lexists(record_path):
 295         raise FileExistsError(f"the cache already has an entry {name}")
 296     staged = files.temporary(tensors_path)
 297     try:
 298         mx.save_safetensors(str(staged), arrays)
 299         files.sync_file(staged)
 300         record = {"format": FORMAT, "key": name, "request": request,
 301                   "tensors": {"file": tensors_path.name, "sha256": files.sha256_file(staged),
 302                               "arrays": _described(arrays)}}
 303         with files.exclusive(directory / LOCK, f"publishing cache entry {name[:12]}", wait=wait):
 304             if os.path.lexists(record_path):
 305                 raise FileExistsError(f"the cache already has an entry {name}")
 306             os.replace(staged, tensors_path)  # over nothing, or over tensors a dead writer left without a record
 307             try:
 308                 files.sync_directory(directory)  # the tensors' name on disk before a record can point at it
 309                 files.atomic_json(record_path, record)
 310             except files.NotDurable:
 311                 raise  # the record is in place: the entry is published and stays whole
 312             except BaseException:
 313                 tensors_path.unlink(missing_ok=True)
 314                 raise
 315     finally:
 316         staged.unlink(missing_ok=True)
 317     return name
```
📍 [`src/h3mlx/files.py:136-151`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/files.py#L136-L151)
```python
 136 def atomic_write(path, write) -> None:
 137     """`write(temporary_path)` makes the file; it is flushed and renamed onto `path`.
 138
 139     Two kinds of failure. Before the rename, anything that goes wrong removes the temporary file and leaves `path`
 140     as it was. After the rename the new file is in place, and the one thing left is flushing the directory: if that
 141     fails the error is `NotDurable`, and `path` is the new file."""
 142     path = Path(path)
 143     tmp = temporary(path)
 144     try:
 145         write(tmp)
 146         sync_file(tmp)
 147         os.replace(tmp, path)
 148     except BaseException:
 149         tmp.unlink(missing_ok=True)
 150         raise
 151     flush_renamed(path)
```
📍 [`src/h3mlx/files.py:52-87`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/files.py#L52-L87)
```python
  52 @contextlib.contextmanager
  53 def exclusive(path, what: str, wait: float = 0.0):
  54     """Hold the lock file at `path` for the body, and nobody else: an exclusive `flock`.
  55
  56     * It cannot be left stuck. The lock is the kernel's, attached to this process's open file: when the process
  57       ends, by returning, by an exception or by being killed, the lock is gone. The file stays, which means nothing.
  58     * It says who holds it: the holder writes its process id, `what` it is doing and since when into the file, and
  59       empties the file on release. A second taker is refused with `Busy`, which carries that.
  60     * `wait` is how many seconds to keep trying before refusing; zero refuses at once.
  61
  62     Advisory and local: it stops other takers of this lock, not a process that ignores it, and it is not to be relied
  63     on over a network filesystem."""
  64     path = Path(path)
  65     fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
  66     held = False
  67     try:
  68         deadline = time.monotonic() + wait
  69         while True:
  70             try:
  71                 fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  72                 break
  73             except BlockingIOError:
  74                 if time.monotonic() >= deadline:
  75                     raise Busy(path, lock_holder(path)) from None
  76                 time.sleep(0.01)
  77         held = True
  78         os.ftruncate(fd, 0)
  79         os.write(fd, json.dumps({"pid": os.getpid(), "what": what, "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  80                                  "command": sys.argv[:1]}).encode())
  81         os.fsync(fd)
  82         yield
  83     finally:
  84         if held:
  85             os.ftruncate(fd, 0)
  86             fcntl.flock(fd, fcntl.LOCK_UN)
  87         os.close(fd)
```
📍 [`src/h3mlx/cache.py:320-350`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L320-L350)
```python
 320 def load(record_path, request: dict) -> Entry:
 321     """The entry whose record is at `record_path`, if it is the one `request` asks for and whole; otherwise a
 322     `CacheMismatch` that says what differs."""
 323     record_path = Path(record_path)
 324     try:
 325         record = json.loads(record_path.read_text())
 326     except (OSError, ValueError) as e:
 327         raise CacheMismatch(f"{record_path.name}: the record cannot be read ({e})") from e
 328     if not isinstance(record, dict) or record.get("format") != FORMAT:
 329         raise CacheMismatch(f"{record_path.name}: not a {FORMAT} record")
 330     apart = differences(request, record.get("request"))
 331     if apart:
 332         raise CacheMismatch(f"{record_path.name} is not the entry asked for: " + "; ".join(apart[:6])
 333                             + (f"; and {len(apart) - 6} more" if len(apart) > 6 else ""))
 334     described = record.get("tensors") or {}
 335     if record.get("key") != key(request) or described.get("file") != f"{key(request)}.safetensors":
 336         raise CacheMismatch(f"{record_path.name}: the record's key is not its request's")
 337     tensors_path = record_path.with_name(described["file"])
 338     if not tensors_path.is_file():
 339         raise CacheMismatch(f"{record_path.name}: its tensors, {tensors_path.name}, are missing")
 340     if files.sha256_file(tensors_path) != described.get("sha256"):
 341         raise CacheMismatch(f"{tensors_path.name} is not the file its record describes (sha256 differs)")
 342     try:
 343         arrays = mx.load(str(tensors_path), format="safetensors")
 344     except Exception as e:  # MLX raises its own types for a file that is not safetensors
 345         raise CacheMismatch(f"{tensors_path.name} cannot be read as safetensors ({e})") from e
 346     if _described(arrays) != described.get("arrays") or sorted(arrays) != sorted(_names(request)):
 347         raise CacheMismatch(f"{tensors_path.name}: its arrays are not the ones its record lists")
 348     count = len(request["inputs"]["references"])
 349     return Entry(latents=arrays["latents"], condition_latents=[arrays[f"condition_latents.{i}"] for i in range(count)],
 350                  text=arrays["text"], text_token_tags=np.array(arrays["text_token_tags"]), record=record)
```
📍 [`src/h3mlx/cache.py:353-359`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L353-L359)
```python
 353 def read(directory, request: dict) -> Entry | None:
 354     """The entry for `request` in `directory`, or None when there is none. An entry that is there under the
 355     request's key and is not what was asked, or not whole, is an error (`CacheMismatch`), never a miss."""
 356     record_path = Path(directory) / f"{key(request)}.json"
 357     if not os.path.lexists(record_path):
 358         return None
 359     return load(record_path, request)
```

- **Format.** `<key>.safetensors` holds the arrays, `<key>.json` the record: the request in full, the arrays' names,
  dtypes and shapes, and the tensors file's sha256. Nothing is pickled, and the reader loads the tensors as
  safetensors and as nothing else.
- **Writing.** The tensors are written under a temporary name; then the entry is published, the tensors renamed into
  place and the record written last, so an entry exists only once it is whole. A write that fails at any of its steps
  leaves nothing behind (`test_a_write_that_fails_leaves_no_entry`); what a killed writer leaves is under a temporary
  name or without a record, and is not an entry. An entry is written once: a key that already has a record is
  refused.
- **One writer publishes at a time.** Publication happens while holding the cache directory's lock, `cache.lock`.
  Without it, two writers of one key could interleave: the second commits an entry between the first one's tensors
  and its record, the first then fails, and its cleanup deletes the tensors under the second one's record. With it,
  a writer that fails removes only what it alone put there, and an entry that has its record is never touched.
  A writer waits up to thirty seconds for the lock and is then refused, told who holds it.
- **Once the record is in place, the entry stands.** One failure can come after publication: the record has been
  renamed into place and flushing the directory afterwards fails. Cleaning up as for an earlier failure would delete
  the tensors under a published record. Rolling both back was the other choice, and the worse one: readers do not
  take the lock, so an entry that has been visible cannot be withdrawn without a reader seeing half of it, and
  what a cache risks by keeping an entry whose durability is in doubt is one entry, encoded again after a power
  loss. So nothing is removed, and the error is its own kind, `files.NotDurable`, which says the file is there.
  The tensors' own rename is flushed before the record is written, so a record on disk has its tensors on disk
  (`test_a_failure_after_the_records_rename_keeps_the_entry`,
  `test_atomic_write_says_when_the_failure_came_after_the_rename`).
- **The lock** is `files.exclusive`: the operating system's advisory lock on a file. It belongs to the process that
  took it, so it ends when that process does, however it ends, and cannot be left stuck. The holder writes its
  process id, what it is doing and since when into the lock file. It is advisory and local: it binds other users of
  the same lock, and is not to be relied on over a network filesystem.
- **What is refused before writing:** arrays that are not the request's (names, shapes, dtypes), a picture's latents
  that are not in the stated height-to-width ratio, tags that are not 0 or 1, and anything not finite.
- **Reading.** No record under the key is a miss. A record that is there is checked: its request against the request
  asked, field by field; its key; the tensors file's digest; the arrays against the record's list. Anything that
  does not hold is a `CacheMismatch` that says what differs. It is never a miss, which would let a trainer encode
  over the evidence.

Three tests of the lock around publication:

- `test_a_writer_that_fails_cannot_break_another_writers_entry` starts a second writer of the same key from inside
  the first one's publication, at the worst moment. The second is refused and leaves nothing; the first then fails
  and removes its own files; the key can be written afterwards.
- `test_an_entry_published_meanwhile_is_left_alone` is the other order: an entry is published between a writer's
  staging and its turn at the lock. The writer finds the record, stops, and leaves the committed entry whole.
- `test_two_processes_writing_one_key` does the first with two real processes. The other process is held in the
  middle of its publication until this one says go, so the order does not depend on timing: this process's write is
  refused, naming the other's process id; the other is released, fails and cleans up; this one then writes the key.

`test_a_changed_input_or_choice_is_not_served_the_old_entry` writes one entry and asks for it with one thing changed,
twenty-two times: the picture, a reference, their order, the prompt, the size, either seed, either rounding, the
sizing rule, either dtype, a pixel bound, each encoder's identity. Each is a miss, and the old record, handed to the
reader on purpose, is refused with the field named. `test_a_changed_code_or_library_version_is_another_key` does the
same for the source digest and for each library.

### 10.3 Sizing a reference picture: two named rules

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_encoder.py:60-67`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py#L60-L67)
```python
  60 def _size_in_area_range(width: int, height: int, min_pixels: int, max_pixels: int, multiple: int) -> tuple[int, int]:
  61     r"""
  62     The `(width, height)` of an image that keeps its size, or is scaled at its own aspect ratio to `min_pixels` or
  63     `max_pixels` when its area is outside that range, with both sides rounded to multiples of `multiple`.
  64     """
  65     area = width * height
  66     scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
  67     return tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (width, height))
```
📍 [`src/h3mlx/cache.py:98-104`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L98-L104)
```python
  98 def scale_down_to_area(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
  99     """(width, height) at the picture's own aspect ratio, scaled down to `pixels` when it has more and never scaled
 100     up, each side rounded to `multiple`. The recipe's rule for a still reference (`ostris/ai-toolkit` at d6a18ad7,
 101     `extensions_built_in/diffusion_models/minimax_h3/src/packing.py:112-124`)."""
 102     scale = min(1.0, math.sqrt(pixels / float(width * height)))
 103     return (max(multiple, round(width * scale / multiple) * multiple),
 104             max(multiple, round(height * scale / multiple) * multiple))
```
📍 [`src/h3mlx/cache.py:107-125`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L107-L125)
```python
 107 def reference_sizes(sizing: str, width: int, height: int, *, target_height: int, target_width: int,
 108                     min_pixels: int, max_pixels: int, vision_pixels: int, multiple: int) -> tuple[tuple, tuple]:
 109     """((width, height) the VAE encodes a reference at, (width, height) the text encoder sees it at).
 110
 111     * `"pipeline"`: what diffusers' edit pipeline does at sampling. For the VAE the picture keeps its own size,
 112       brought to the nearer bound when its area is outside [`min_pixels`, `max_pixels`]; the text encoder sees it at
 113       about `vision_pixels` (`image.size_in_area_range`, `image.size_at_area`). The target's size plays no part.
 114     * `"recipe"`: what the training recipe does. For the VAE the picture is scaled down to the *target's* pixel area
 115       when it has more, and never up; the text encoder sees it scaled down to `vision_pixels` (in whole thousands,
 116       as the recipe writes it) when it has more, and never up (`p_h3_image.py:304-311` of the same commit).
 117
 118     The two differ whenever a reference is larger than the target, or smaller than the pipeline's lower bound."""
 119     if sizing == "pipeline":
 120         return (I.size_in_area_range(width, height, min_pixels, max_pixels, multiple),
 121                 I.size_at_area(width, height, vision_pixels, multiple))
 122     if sizing == "recipe":
 123         return (scale_down_to_area(width, height, target_height * target_width, multiple),
 124                 scale_down_to_area(width, height, 1000 * (vision_pixels // 1000), multiple))
 125     raise ValueError(f"reference sizing is one of {SIZING}, got {sizing!r}")
```

diffusers' edit pipeline and the recipe size a reference differently, and a cache has to say which it used:

| | `"pipeline"` (diffusers, at sampling) | `"recipe"` (the trainer, at training) |
| --- | --- | --- |
| for the VAE | the picture's own size, brought to the nearer bound when its area is outside 1 to 8 MP | scaled down to the **target's** pixel area when larger, never up (`src/packing.py:112-124`, called at `minimax_h3.py:1292-1307`) |
| for the text encoder | about 1 MP, up or down (`_size_at_area`) | scaled down to 1 MP when larger, never up (`p_h3_image.py:304-311`) |
| depends on the target | no | yes |
| original to test against | diffusers' two functions | none that runs here |

- **`"pipeline"`** is checked against diffusers' `_size_in_area_range` and `_size_at_area` on 400 random sizes
  (`test_pipeline_sizing_is_diffusers`).
- **`"recipe"`** has no runnable original: the trainer was read, not run, and its code is not copied. The rule is
  three lines of arithmetic, and `test_recipe_sizing_scales_down_to_the_target_and_never_up` checks it against values
  worked by hand and against its stated properties (never larger than the picture, a multiple of 32 a side, the
  target's area when the picture has more).
- **For training, the recipe's rule is the one to state as the source,** because it is what a trainer for this model
  is known to do. Which rule suits the model better is not known here. The rule is an argument with no default.

The two differ whenever a reference is larger than the target or smaller than the pipeline's lower bound: a 4000×3000
reference beside a 1024×1024 target is encoded at 1184×896 by the recipe and at 3360×2496 by the pipeline.

**One thing the recipe does that the port does not.** The recipe resizes a reference with torch's antialiased
bicubic filter on values in [0, 1] and hands the VAE the floats (`minimax_h3.py:1309-1315`, `1327`). The port resizes
with Pillow's Lanczos filter on 8-bit pixels, as diffusers' pipeline does, for either sizing rule. The filter is
recorded in the request, and it has one value.

### 10.4 The encode

📍 [`src/h3mlx/cache.py:365-369`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L365-L369)
```python
 365 def oriented(picture):
 366     """The picture as RGB with its EXIF orientation applied. Every picture an entry is made from goes through this,
 367     whoever opened the file, so a photograph that says "rotate me" is encoded the way it is looked at, and once
 368     applied the orientation is no longer pending: doing it twice changes nothing."""
 369     return ImageOps.exif_transpose(picture).convert("RGB")
```
📍 [`src/h3mlx/cache.py:372-375`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L372-L375)
```python
 372 def load_picture(path):
 373     """A picture file as `encode` takes it: opened, oriented, RGB."""
 374     with Image.open(path) as opened:
 375         return oriented(opened)
```
📍 [`src/h3mlx/cache.py:413-422`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L413-L422)
```python
 413 def target_picture(picture, height: int, width: int, multiple: int):
 414     """The picture, oriented and RGB (`oriented`), at (width, height), resized with Lanczos when it is another size.
 415     Its aspect ratio must be the stated one to within one `multiple` a side: a picture of another shape is refused,
 416     not stretched or cropped."""
 417     picture = oriented(picture)
 418     own_width, own_height = picture.size
 419     if abs(own_width * height / own_height - width) > multiple or abs(own_height * width / own_width - height) > multiple:
 420         raise ValueError(f"a {own_width}x{own_height} picture does not have the shape of the stated {width}x{height} "
 421                          "(width x height): crop it first")
 422     return I.resize(picture, (width, height))
```
📍 [`src/h3mlx/cache.py:394-410`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L394-L410)
```python
 394 def load_encoders(weights, vae_dtype: str):
 395     """`image.Components` for `encode`, from a release folder: the VAE (loaded with `vae_dtype`'s policy), the text
 396     encoder as far as the connector's hidden states need, the connector, the tokenizer and the processor. This is
 397     the one place the encoders are loaded for the cache, so how they are loaded is part of the code digest. It needs
 398     the weights; the tokenizer and the processor are read from their local folders."""
 399     from transformers import AutoTokenizer, Qwen3VLProcessor
 400
 401     weights = Path(weights)
 402     transformer, _, settings = release_configs(weights)
 403     indices = tuple(json.loads((weights / "text_connector" / "config.json").read_text())["hidden_state_indices"])
 404     _, encode_states = loaders.load_text_encoder_states(weights / "text_encoder", indices)
 405     vae, mean, std = loaders.load_vae(weights / "vae", dtype=DTYPES[vae_dtype])
 406     return I.Components(
 407         tokenizer=AutoTokenizer.from_pretrained(str(weights / "tokenizer")),
 408         processor=Qwen3VLProcessor.from_pretrained(str(weights / "processor")), encode_states=encode_states,
 409         connector=loaders.load_safetensors(weights / "text_connector"), vae=vae, vae_latents_mean=mean,
 410         vae_latents_std=std, dit_params={}, dit_cfg=transformer, settings=settings)
```
📍 [`src/h3mlx/cache.py:425-466`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/cache.py#L425-L466)
```python
 425 def encode(components, request: dict, picture, references) -> dict:
 426     """The tensors of the entry `request` asks for, from the pictures themselves (PIL images, the ones whose digests
 427     the request names; each is oriented by its EXIF tag here, `oriented`, so it does not matter whether the caller
 428     did) and `components` (`image.Components`: the VAE, the text encoder's `encode_states`, the connector, the
 429     tokenizer and the processor, loaded as the request says: `load_encoders`).
 430
 431     * The picture: resized to the stated size (`target_picture`), encoded, one posterior sample from a generator
 432       seeded with the request's seed, rounded through float16 or not, normalised.
 433     * Each reference: sized by the request's rule (`reference_sizes`), encoded the same way with the references'
 434       seed and rounding.
 435     * The text: the prompt as written for text-to-image; for an edit, the pipeline's presentation (a vision block per
 436       reference, then the instruction with its "Edit <Picture 1>. " prefix), through the text encoder and the
 437       connector, cast to the request's text dtype."""
 438     references = [oriented(r) for r in references]
 439     if len(references) != len(request["inputs"]["references"]):
 440         raise ValueError(f"the request names {len(request['inputs']['references'])} references, got {len(references)}")
 441     if len(references) > I.MAX_EDIT_IMAGES:
 442         raise ValueError(f"an edit takes at most {I.MAX_EDIT_IMAGES} pictures, got {len(references)}")
 443     multiple, target = request["vae"]["canvas_multiple"], request["target"]
 444     if components.canvas_multiple != multiple:
 445         raise ValueError(f"the request was made for a canvas multiple of {multiple}; these components have "
 446                          f"{components.canvas_multiple}")
 447     resized = target_picture(picture, target["height"], target["width"], multiple)
 448     out = {"latents": I.encode_pictures(components, [resized], float16_round=target["float16_round"],
 449                                         seed=target["seed"])[0]}
 450     prompt = request["inputs"]["prompt"]
 451     if references:
 452         choice = request["references"]
 453         sizes = [reference_sizes(choice["sizing"], *r.size, target_height=target["height"], target_width=target["width"],
 454                                  min_pixels=choice["min_pixels"], max_pixels=choice["max_pixels"],
 455                                  vision_pixels=choice["vision_pixels"], multiple=multiple) for r in references]
 456         latents = I.encode_pictures(components, [I.resize(r, vae_size) for r, (vae_size, _) in zip(references, sizes)],
 457                                     float16_round=choice["float16_round"], seed=choice["seed"])
 458         out.update({f"condition_latents.{i}": x for i, x in enumerate(latents)})
 459         ids, tags, vision = I.edit_presentation(components, prompt, [I.resize(r, vision_size)
 460                                                                      for r, (_, vision_size) in zip(references, sizes)])
 461     else:
 462         ids, tags, vision = I.text2image_presentation(components, prompt)
 463     states = components.encode_states(ids, vision)
 464     out["text"] = I.connect(components.connector, states, DTYPES[request["text"]["dtype"]])
 465     out["text_token_tags"] = mx.array(np.asarray(tags, dtype=np.int64))
 466     return out
```

`load_encoders` and `encode` are the only functions here that need weights.

- **Every picture is oriented first.** A photograph can carry an EXIF tag that says how to turn it. `encode` applies
  it to the picture and to each reference itself, so the same file gives the same entry whether the caller opened it
  raw or through `load_picture`. Without that, one key could hold a picture the right way up or upside down
  depending on the caller. `test_a_picture_is_encoded_the_way_it_is_looked_at` and
  `test_a_reference_is_oriented_like_the_picture` check both ways in, with a tag that turns the picture upside down
  (the same size, so nothing else would notice) and one that swaps its sides.

- **The picture** is resized to the stated size and encoded as the pipeline encodes a reference (`vae.encode_condition`):
  one posterior sample, from a generator seeded with the request's seed, rounded through float16 or not. The recipe
  also takes a posterior sample for its training pictures, from a generator it does not seed (`minimax_h3.py:720`).
  A picture whose shape is not the stated one, to within one canvas multiple a side, is refused: nothing is cropped
  or stretched.
- **Each reference** is sized by the request's rule and encoded the same way, with the references' seed and rounding.
- **The text** is the pipeline's: the prompt as written for text-to-image; for an edit, a vision block per reference
  and the instruction with its `Edit <Picture 1>. ` prefix (the recipe prefixes the same way, `p_h3_image.py:313-322`).

What the tests show, and on what:

- **On a tiny VAE with a stub text side** (`test_a_text_to_image_entry_is_the_direct_encode`): the latents are
  `encode_condition`'s for the seed and rounding asked, bit for bit; the seed and the rounding each change them; a
  picture of the stated shape at another size is resized first. Written and read back, the entry builds the example
  that the direct encode builds.
- **On the tiny release with its text encoder and processor** (`test_an_edit_entry_is_the_pipelines_conditioning`,
  which needs the release fixtures): under the pipeline's choices an edit entry's reference latents, text rows and
  tags are exactly the ones the sampling pipeline makes for the same pictures and instruction, and the sampling
  pipeline is tested against diffusers. Under the recipe's sizing the references are encoded at the recipe's sizes.

**What cannot be tested without weights.** That the real VAE and text encoder, at real sizes, give latents and text
rows fit to train on; how long encoding takes and what it needs in memory; and anything about the recipe's own
encode, which was read in part and not run. The cache returns what was put in it. Whether what was put in is right
is the encoders' business, and they are tested where they live.

### 10.5 The tests

| test | pins |
| --- | --- |
| `test_an_entry_comes_back_as_it_went_in` (×4) | bits, dtypes and shapes through the two files; the entry builds the example |
| `test_a_changed_input_or_choice_is_not_served_the_old_entry` (×22) | one change, another key, a miss; the old record refused by name |
| `test_a_changed_code_or_library_version_is_another_key` | the source digest and each library's version |
| `test_an_example_without_references_ignores_the_reference_choices` | a text-to-image key does not move with them |
| `test_code_digest_follows_the_source`, `test_identity_reads_no_weights` | every source file by name and content; weights by name and size, unopened |
| `test_an_entry_put_under_another_requests_name_is_refused`, `test_differences_names_what_differs` | a stale entry made by hand, and the comparison it trips |
| `test_a_damaged_entry_is_an_error_not_a_miss` (×11) | other tensors, a truncated file, no tensors, a bad record, a pickle or an npz with the record's blessing |
| `test_a_write_that_fails_leaves_no_entry` (×5), `test_files_a_killed_writer_left_are_ignored`, `test_an_entry_is_written_once`, `test_atomic_write_keeps_the_old_file_when_the_new_one_fails` | whole or absent |
| `test_a_failure_after_the_records_rename_keeps_the_entry`, `test_atomic_write_says_when_the_failure_came_after_the_rename` | a failure after the rename: the entry, or the file, stands, and the error says so |
| `test_a_writer_that_fails_cannot_break_another_writers_entry`, `test_an_entry_published_meanwhile_is_left_alone`, `test_two_processes_writing_one_key` | one writer publishes at a time; a failed writer removes only its own; in one process and in two |
| `test_exclusive_is_released_when_its_holder_ends` | the lock: one holder, who it is, waiting, and a killed holder does not leave it held |
| `test_a_picture_is_encoded_the_way_it_is_looked_at`, `test_a_reference_is_oriented_like_the_picture` (release fixtures), `test_what_turns_a_request_into_an_entry_is_in_the_package` | EXIF orientation applied in the cache; nothing about the encode left to a script |
| `test_tensors_that_are_not_an_entry_are_not_written` (×12), `test_request_refusals` (×20), `test_a_request_has_no_defaults` | what is refused going in |
| `test_pipeline_sizing_is_diffusers`, `test_recipe_sizing_scales_down_to_the_target_and_never_up` | the two rules |
| `test_a_text_to_image_entry_is_the_direct_encode`, `test_an_edit_entry_is_the_pipelines_conditioning` (×2, release fixtures) | the encode on tiny encoders |

## 11. The training script

`scripts/train_lora.py` trains an adapter on a list of pictures, from the pieces of §1 to §10 in order. Its mechanics
are in `src/h3mlx/trainer.py`; the script reads the arguments, the configs and the dataset, fills the cache, and
loads the weights. There is no original for it. **It has run on a tiny random transformer and on nothing else.**

### 11.1 The arguments: nothing that decides the training has a default

Every option is required, except the ones listed below. Leave one out and the command stops and names it, before
anything is read (`test_a_missing_argument_is_an_error_that_names_it` drops each of 27 in turn):

| group | arguments |
| --- | --- |
| where | `--weights`, `--dataset`, `--cache-dir`, `--run-dir` |
| the adapter | `--rank`, `--alpha`, `--targets` (which attention and feed-forward matrices), `--refiner` or `--no-refiner` |
| hyper-parameters | `--learning-rate`, `--betas`, `--eps`, `--weight-decay`, `--max-grad-norm`, `--shift`, `--grid`, `--density`, `--weighting`, `--condition-t`, `--seed` |
| the encode (§10) | `--latent-seed`, `--latent-float16-round` or its `--no-`, `--vae-dtype`, `--text-dtype` |
| limits | `--max-steps`, `--max-seconds`, `--max-peak-memory-gb`, `--checkpoint-every` |

Not required, each for a reason (`test_only_the_backward_way_has_a_default` lists exactly these):

- `--logit-mean`, `--logit-std` and `--mode-scale` are required by the density that uses them and refused with one
  that does not;
- `--reference-sizing`, `--reference-seed` and `--reference-float16-round` are required when the dataset has an
  example with references;
- `--assistant`, a frozen adapter kept active while training (§5), is optional;
- `--backward` chooses how the gradient is taken with bounded memory. It defaults to `blockwise`, the one whose peak
  measured flat in the number of blocks (§8.4); `checkpointed` is the other. Both train the same weights
  (`test_the_two_bounded_gradients_train_the_same_weights`), so this is the one default, and it decides no number;
- `--plan-only` and `--resume` are the two modes.

The script downloads nothing. It sets `HF_HUB_OFFLINE` for its process before it imports anything, so a tokenizer or
processor folder that is missing is an error and not a fetch.

No learning-rate schedule: the rate is constant. The recipe's schedule and its optimizer settings were not read.

📍 [`src/h3mlx/trainer.py:126-183`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L126-L183)
```python
 126 def trajectory(*, cfg: dict, transformer: str, dataset, rank: int, alpha: float, targets, refiner: bool,
 127                assistant: str | None, learning_rate: float, betas, eps: float, weight_decay: float,
 128                max_grad_norm: float, shift: float, grid: int, density: str, logit_mean, logit_std, mode_scale,
 129                weighting: str, condition_t: float, seed: int, backward: str) -> dict:
 130     """Everything the trained weights depend on, checked, as one JSON value. No argument has a default.
 131
 132     `transformer` is the base weights' identity (`cache.identity`), `dataset` the examples' cache keys in order,
 133     `assistant` the sha256 of a frozen adapter's file or None. The density's own parameters are required for its
 134     scheme and must be None otherwise: `logit_mean` and `logit_std` for `"logit_normal"`, `mode_scale` for `"mode"`."""
 135     dataset = list(dataset)
 136     if not dataset or not all(isinstance(k, str) and len(k) == 64 for k in dataset):
 137         raise ValueError("the dataset is a non-empty list of cache keys")
 138     _count("rank", rank)
 139     lora.check_alpha("alpha", alpha)
 140     if not isinstance(refiner, bool):
 141         raise ValueError(f"refiner must be True or False, got {refiner!r}")
 142     modules = adapter_modules(cfg, targets, refiner)
 143     betas = list(betas)
 144     if len(betas) != 2 or not all(isinstance(b, float) and 0 <= b < 1 for b in betas):
 145         raise ValueError(f"betas must be two numbers in [0, 1), got {betas}")
 146     if not ops.usable_shift(shift):
 147         raise ValueError(f"shift must be positive and finite, got {shift!r}")
 148     train.training_sigmas(_count("grid", grid), shift)
 149     if density not in DENSITY:
 150         raise ValueError(f"density is one of {DENSITY}, got {density!r}")
 151     wanted = {"uniform": (), "logit_normal": ("logit_mean", "logit_std"), "mode": ("mode_scale",)}[density]
 152     for name, value in (("logit_mean", logit_mean), ("logit_std", logit_std), ("mode_scale", mode_scale)):
 153         if name in wanted:
 154             if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
 155                 raise ValueError(f"density {density!r} needs {name} as a finite number, got {value!r}")
 156         elif value is not None:
 157             raise ValueError(f"{name} is not used by density {density!r}: leave it out")
 158     if density == "logit_normal":
 159         _number("logit_std", logit_std)
 160     if weighting not in WEIGHTING:
 161         raise ValueError(f"weighting is one of {WEIGHTING}, got {weighting!r}")
 162     if not isinstance(condition_t, float) or not 0 <= condition_t <= 1:
 163         raise ValueError(f"condition_t must be a number in [0, 1], got {condition_t!r}")
 164     if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
 165         raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
 166     if backward not in BACKWARD:
 167         raise ValueError(f"backward is one of {sorted(BACKWARD)}, got {backward!r}")
 168     if assistant is not None and not (isinstance(assistant, str) and len(assistant) == 64):
 169         raise ValueError(f"assistant is the sha256 of the frozen adapter's file, or None; got {assistant!r}")
 170     return {
 171         "model": {"transformer": transformer, "config": json.loads(files.canonical_json(cfg))},
 172         "dataset": dataset,
 173         "adapter": {"name": TRAINED, "rank": rank, "alpha": alpha, "targets": list(targets), "refiner": refiner,
 174                     "modules": len(modules), "parameters": parameter_count(cfg, modules, rank)},
 175         "assistant": assistant,
 176         "hyper": {"learning_rate": _number("learning_rate", learning_rate), "betas": betas, "eps": _number("eps", eps),
 177                   "weight_decay": _number("weight_decay", weight_decay, strict=False),
 178                   "max_grad_norm": _number("max_grad_norm", max_grad_norm), "shift": shift, "grid": grid,
 179                   "density": {"scheme": density, "logit_mean": logit_mean, "logit_std": logit_std, "mode_scale": mode_scale},
 180                   "weighting": weighting, "condition_t": condition_t, "seed": seed, "backward": backward,
 181                   "audio_level": "the picture's"},
 182         "versions": cache.versions(),
 183     }
```
📍 [`src/h3mlx/trainer.py:186-192`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L186-L192)
```python
 186 def limits(*, max_steps: int, max_seconds: float, max_peak_memory_gb: float, checkpoint_every: int) -> dict:
 187     """When a session stops, and how often it checkpoints. Each is required, and each stops the run: reaching
 188     `max_steps` finishes it, and the wall clock or MLX's peak-memory counter going over its limit ends the session
 189     with the run resumable. No argument has a default."""
 190     return {"max_steps": _count("max_steps", max_steps), "max_seconds": _number("max_seconds", max_seconds),
 191             "max_peak_memory_gb": _number("max_peak_memory_gb", max_peak_memory_gb),
 192             "checkpoint_every": _count("checkpoint_every", checkpoint_every)}
```

The arguments fall into two groups. The **trajectory** is everything the trained weights depend on: the model's
identity and config, the examples' cache keys (and so every encode choice), the adapter, the hyper-parameters, the
seed, the versions. The **limits** say when a session stops and how often it checkpoints. A run is resumed only with
the same trajectory; its limits may change.

### 11.2 `--plan-only`

📍 [`src/h3mlx/trainer.py:228-265`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L228-L265)
```python
 228 def plan(traj: dict, lim: dict, cfg: dict, examples: list, *, run_dir, cache_dir, weights: dict | None = None,
 229          block_dtype: str | None = None) -> dict:
 230     """What a run would do, from the arguments and the configs alone.
 231
 232     `examples` holds one dict per example: its cache key, whether the cache has it, its sizes and its rows
 233     (`example_rows`). `weights` is `weights_summary` of the base weights' headers, and `block_dtype` the dtype the
 234     headers give a block's matrices; both may be None, and the estimates that need them are then left out."""
 235     adapter, hyper = traj["adapter"], traj["hyper"]
 236     modules = adapter_modules(cfg, adapter["targets"], adapter["refiner"])
 237     count = len(examples)
 238     largest = max(e["sequence_rows"] or e["sequence_rows_without_text"] for e in examples)
 239     steps = lim["max_steps"]
 240     checkpoints = sorted(set(range(lim["checkpoint_every"], steps + 1, lim["checkpoint_every"])) | {steps})
 241     sizes = {"adapter_bytes": 4 * adapter["parameters"], "optimizer_state_bytes": 8 * adapter["parameters"],
 242              "checkpoint_bytes": 12 * adapter["parameters"], "checkpoints_bytes": 12 * adapter["parameters"] * len(checkpoints),
 243              "largest_sequence_rows": largest,
 244              "largest_sequence_is_without_text": any(e["sequence_rows"] is None for e in examples)}
 245     if weights is not None:
 246         sizes["base_weights_bytes"] = weights["bytes"]
 247         sizes["base_weights_bytes_by_dtype"] = weights["bytes_by_dtype"]
 248     if block_dtype is not None and hyper["backward"] == "blockwise":
 249         sizes["saved_block_inputs_bytes"] = cfg["num_layers"] * largest * cfg["hidden_size"] * BYTES[block_dtype]
 250     run_dir = Path(run_dir)
 251     return {
 252         "examples": {"count": count, "to_encode": len({e["key"] for e in examples if not e["cached"]}), "list": examples},
 253         "steps": {"max_steps": steps, "passes_over_the_examples": round(steps / count, 3),
 254                   "checkpoint_every": lim["checkpoint_every"], "checkpoints_at": checkpoints},
 255         "adapter": {**adapter, "scaling": adapter["alpha"] / adapter["rank"], "first_modules": modules[:3],
 256                     "last_module": modules[-1]},
 257         "assistant": traj["assistant"],
 258         "sizes": sizes,
 259         "hyper": hyper,
 260         "limits": lim,
 261         "cache": {"directory": str(cache_dir), "keys": [e["key"] for e in examples]},
 262         "writes": {"run_directory": str(run_dir), "record": str(run_dir / "run.json"),
 263                    "checkpoints": str(run_dir / "checkpoints" / "step-NNNNNNNN"),
 264                    "final_adapter": str(run_dir / "adapter.safetensors")},
 265     }
```

`--plan-only` prints what a run would do and stops. It reads the configs, the dataset file, the pictures' bytes (for
their digests and sizes), the cache's records and the base weights' safetensors headers. It opens no weights and
writes nothing. What it prints:

- the examples: each one's cache key, whether the cache has it, its size, and its rows (video, condition, audio, and
  text when the cache knows it);
- the steps: how many, how many passes over the examples that is, and where the checkpoints fall;
- the adapter: which modules, how many, how many parameters;
- sizes it can compute without weights: the adapter's and the optimizer state's bytes, a checkpoint's, the base
  weights' bytes by dtype (from the headers), and for the block-by-block gradient the bytes of its saved block inputs
  (blocks × the longest sequence × hidden size × the blocks' dtype);
- the cache directory, and every path the run would write.

`test_plan_only_opens_no_weights_and_writes_nothing` runs it with every loader made to raise (`mx.load`, the
loaders of `h3mlx.load`, `lora.load`, the cache's reader and encoder): it prints the plan, creates no run directory
and leaves the cache as it was. The numbers it printed are then checked against a real run of the same command: the
parameter count against the trained adapter's elements, the sizes against the files' arrays, each example's rows
against the example the builder makes.

What the plan cannot say: peak memory and time. Those are measured, not computed.

The base weights' bytes come from their safetensors headers, by dtype. Every dtype a header can name has a size
there, and one that does not is reported as a bad argument is, with the tensor's name
(`test_every_safetensors_dtype_has_a_size`).

### 11.3 What a step draws

📍 [`src/h3mlx/trainer.py:65-69`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L65-L69)
```python
  65 def step_seed(seed: int, step: int, stream: str) -> int:
  66     """A seed for one of step `step`'s draws (`stream`: the example order, the noise level, the noise; `"init"` at
  67     step 0 is the adapter's initialisation), from the run's seed. numpy's `SeedSequence` mixes the three numbers, so
  68     neighbouring steps and streams share nothing. 63 bits, which is what a torch generator takes."""
  69     return int(np.random.SeedSequence([seed, step, STREAMS[stream]]).generate_state(1, dtype=np.uint64)[0] >> 1)
```
📍 [`src/h3mlx/trainer.py:72-76`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L72-L76)
```python
  72 def example_at(seed: int, step: int, count: int) -> int:
  73     """Which example step `step` trains on: the examples in a new random order for every pass over them."""
  74     epoch, position = divmod(step, count)
  75     order = np.random.default_rng(np.random.SeedSequence([seed, epoch, STREAMS["order"]])).permutation(count)
  76     return int(order[position])
```
📍 [`src/h3mlx/trainer.py:79-91`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L79-L91)
```python
  79 def noise_level(hyper: dict, grid: np.ndarray, step: int) -> tuple[np.float32, float | None]:
  80     """(sigma, loss weight or None) for step `step`: one draw of the density from the step's own generator, looked up
  81     in the grid, and the weighting of that level."""
  82     import torch
  83
  84     density = hyper["density"]
  85     generator = torch.Generator().manual_seed(step_seed(hyper["seed"], step, "level"))
  86     u = train.density_for_timestep_sampling(density["scheme"], 1, density["logit_mean"], density["logit_std"],
  87                                             density["mode_scale"], generator=generator)
  88     sigma = train.sample_sigmas(u, grid)[0][0]
  89     if hyper["weighting"] == "none":
  90         return sigma, None
  91     return sigma, float(train.loss_weighting_for_sd3(hyper["weighting"], np.float32([sigma]))[0])
```

No generator is carried from one step to the next. A step's example, its noise level and its noise each come from a
generator seeded with the run's seed, the step's number and the stream's, mixed by numpy's `SeedSequence`. The
adapter's initialisation is seeded the same way, and each picture's posterior sample from `--latent-seed` and the
picture's digest. So the random state of a run at step k is the pair (seed, k), and a run resumed at step k draws what
an uninterrupted run draws there. Nothing has to be restored, and nothing can be restored wrongly.

The examples are taken in a new random order on every pass, each once a pass.

### 11.4 One session in a run directory, and the checks that come first

📍 [`src/h3mlx/trainer.py:395-447`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L395-L447)
```python
 395 @contextlib.contextmanager
 396 def open_run(run_dir, traj: dict, lim: dict, cfg: dict, *, resume: bool):
 397     """Take a run directory for this process and check that the run can start or go on in it. Yields a `Session`.
 398
 399     Everything here is cheap and reads only the run directory: no weights are loaded, no example is encoded, and
 400     nothing in the directory is changed but its lock file. So a caller does it first, and a run that is refused
 401     costs nothing and leaves the directory as it was.
 402
 403     * **One session at a time.** The directory's lock file, `run.lock`, is held for as long as the session is open
 404       (`files.exclusive`: the lock is the process's, so a session that is killed does not leave it held). A second
 405       session on the same directory is refused, and told which process holds it.
 406     * **A new run** needs a directory that holds none of a run's outputs (`OUTPUTS`: `run.json`,
 407       `adapter.safetensors`, `checkpoints`), whether or not there is a record. Nothing is overwritten.
 408     * **A resume** needs the record, with exactly this trajectory; the latest checkpoint on disk, which must not be
 409       older than the latest one the record lists, and must be whole and this run's (`read_checkpoint`); and a
 410       `max_steps` that the run has not already passed. A run whose record lists no checkpoint and that has none on
 411       disk goes on from step 0, which is where its record says its last checkpoint is."""
 412     run_dir = Path(run_dir)
 413     if resume and not run_dir.is_dir():
 414         raise Refused(f"{run_dir / 'run.json'}: no readable run record to resume from (no such directory)")
 415     run_dir.mkdir(parents=True, exist_ok=True)
 416     with contextlib.ExitStack() as held:
 417         try:
 418             held.enter_context(files.exclusive(run_dir / LOCK, f"training in {run_dir.name}"))
 419         except files.Busy as busy:
 420             raise Refused(f"the run directory is in use: {busy}. One session trains in a run directory at a time") from None
 421         if not resume:
 422             there = [name for name in OUTPUTS if os.path.lexists(run_dir / name)]
 423             if there:
 424                 raise Refused(f"{run_dir} already holds {', '.join(there)}: a new run writes {', '.join(OUTPUTS)} "
 425                               "and overwrites none of them. Pass resume to go on with the run that is there, or "
 426                               "use another directory")
 427             record = {"format": RECORD, "trajectory": traj, "seed": traj["hyper"]["seed"],
 428                       "versions": traj["versions"], "sessions": [], "step": 0, "losses": [], "checkpoints": []}
 429             yield Session(run_dir, traj, lim, cfg, False, record, 0, None, None, [])
 430             return
 431         record = read_record(run_dir)
 432         apart = _differences(traj, record)
 433         if apart:
 434             raise Refused(f"{run_dir / 'run.json'} is another run: " + "; ".join(apart[:6])
 435                           + (f"; and {len(apart) - 6} more" if len(apart) > 6 else ""))
 436         listed = max(record.get("checkpoints") or [0])
 437         checkpoint = latest_checkpoint(run_dir)
 438         found = int(checkpoint.name[5:]) if checkpoint is not None else 0
 439         if found < listed:
 440             raise Refused(f"{run_dir / 'run.json'} lists a checkpoint at step {listed}, and the latest in "
 441                           f"{run_dir / 'checkpoints'} is " + (f"step {found}" if found else "none")
 442                           + ": the run is not gone back to an older checkpoint, or to its start, without being told to")
 443         step, matrices, saved, losses = (0, None, None, []) if checkpoint is None else read_checkpoint(checkpoint, traj, cfg)
 444         if step > lim["max_steps"]:
 445             raise Refused(f"the run is at step {step}, past a max_steps of {lim['max_steps']}: nothing to do, and "
 446                           "nothing is written")
 447         yield Session(run_dir, traj, lim, cfg, True, record, step, matrices, saved, losses)
```

Before anything heavy, the script takes the run directory for its process and checks that the run can start or go on
there. `open_run` reads only the run directory. It loads no weights, encodes nothing, and changes nothing in the
directory but its lock file, so **a run that is refused costs nothing and leaves the directory as it was**: the
record of a finished run included.

- **One session at a time.** The directory's lock file, `run.lock`, is held for the whole session (§10.2's
  `files.exclusive`). A second session on the same directory, a resume started twice for instance, is refused and
  told which process holds it. It does not get as far as clearing temporary checkpoint directories; without the lock
  that clearing would delete the first session's checkpoint in the middle of being written. A session that is killed
  does not leave the lock held.
- **A new run overwrites nothing.** It needs a directory that holds none of a run's three outputs: `run.json`,
  `adapter.safetensors`, `checkpoints`. A directory with someone's adapter in it and no record is refused like one
  with a record. Other files are not the run's and do not matter.
- **A resume** needs the record with this trajectory (§11.8), the checkpoint the record says is its latest, and room
  to go on:
  - It goes on from the latest checkpoint on disk, and that must not be older than the latest one the record lists.
    With the newest checkpoint folder gone it does not roll back to an older one; with the whole folder gone it does
    not start again from step 0. (A checkpoint newer than the record lists is the trace of a run stopped between the
    two writes, and is resumed from.)
  - A run already past the `--max-steps` given is refused, and nothing is written.
- **The order.** The run directory first, then the examples the cache lacks, then the transformer. A `--resume` into
  the wrong directory, or of another run, stops before an encoder is loaded.

Tests: `test_one_session_in_a_run_directory_at_a_time` starts the second session from inside the first one's
checkpoint rename; `test_a_second_process_is_refused_and_sigterm_stops_the_first_cleanly` does it with two real
processes; `test_a_new_run_overwrites_nothing`; `test_a_resume_goes_on_from_the_records_latest_checkpoint_or_not_at_all`;
`test_a_resume_past_its_max_steps_is_refused`; `test_a_refused_resume_encodes_nothing`. Where a run is refused
(over existing outputs, with a checkpoint missing or damaged, past its limit, as another run), the tests compare
every file of the run directory before and after, the lock file aside.

### 11.5 The loop is the pieces

`trainer.run` is, per step: `example_at`, `noise_level` (§4), `build_example` (§6), a bounded gradient (§8),
`optimizer_step` (§9). `test_the_loop_is_the_pieces_in_order` runs six steps of the script with every hyper-parameter
at a value of its own, and the same six steps written out in the test from those pieces, with the plain gradient
that §7 tests against torch. The trained matrices agree to 1e-7, and the script's recorded losses and gradient norms
are the hand-written loop's. A hyper-parameter that did not reach the loop, or reached the wrong place, shows there.

`test_the_loop_overfits_one_example` trains on one synthetic example for 150 steps: its loss at three fixed noise
levels with fixed noise falls from 2.74 to 1.94, and the record's own losses from 2.59 (the first 30 steps) to 1.96
(the last 30). That is a tiny random model memorising noise. It shows the loop turns the loss down and nothing about
what a real model learns.

### 11.6 The record

`run.json` in the run directory is written when the session starts and rewritten atomically (§10.2's
`atomic_write`) after every step. It holds the trajectory (every argument that decides the weights), the limits,
the versions, the seed, the plan, the step reached, the status and why it stopped, each session's command line and
span, and a line per step: the example, the noise level, the loss weight, the loss, the gradient's norm before
clipping, the seconds and MLX's peak memory (`test_the_record_holds_the_run`).

### 11.7 Limits stop the run, and so does SIGTERM

| limit | checked | what happens |
| --- | --- | --- |
| `--max-steps` | before each step | the run is finished: the final adapter is written, exit code 0 |
| `--max-seconds` | before each step, against the session's wall clock | the session ends at the last completed step: a checkpoint, status `stopped`, exit code 3 |
| `--max-peak-memory-gb` | against `mx.get_peak_memory()`, reset when the session starts: before the first step, after every step, and with the block-by-block gradient after every evaluated block | the step in progress is dropped; the session ends at the last completed step: a checkpoint, status `stopped`, exit code 3 |
| SIGTERM | at the same places as the memory limit | the same: the step in progress is dropped, a checkpoint at the last completed step, status `stopped`, reason `terminated`, exit code 3 |

None of them warns and goes on. SIGTERM is what a supervisor, a job timeout or a closing terminal sends: the session
takes it as a request to stop, and stops as it does at a limit, resumable. A signal that arrives in the middle of an
optimizer update is acted on after it. The handler is the session's only; the one before it is put back when the
session ends. Ctrl-C is not handled this way: it interrupts where it lands, and the run goes on from its last
periodic checkpoint.

- `test_sigterm_stops_at_the_next_boundary_with_a_checkpoint` sends the signal to the test's own process during the
  fourth step's update (the session ends at step 4) and again in the middle of a backward pass (that step is
  dropped), then resumes to the uninterrupted run's weights.
- `test_a_second_process_is_refused_and_sigterm_stops_the_first_cleanly` sends it to the script running as a process
  of its own: exit code 3, a checkpoint where it stopped, and a resume to the uninterrupted run's adapter, byte for
  byte. `--max-steps` is a limit and not part of the trajectory: a finished run resumed with
a higher one goes on, to the weights a run given the higher one from the start reaches.

- `test_the_wall_clock_stops_the_session_and_the_run_resumes` uses a clock that advances a second a reading.
- `test_the_memory_limit_stops_the_session` sets the limit below what loading takes (the run stops before its first
  step), and then makes the counter go over in the middle of the fourth step's backward pass: the session ends at
  step 3, and the checkpoint there holds the weights of an uninterrupted run after three steps.
- The memory limit is MLX's counter and nothing more. It does not see memory outside MLX, and it stops the run after
  the peak has happened, not before.

### 11.8 Checkpoints, and resuming

📍 [`src/h3mlx/trainer.py:293-320`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L293-L320)
```python
 293 def write_checkpoint(run_dir, step: int, traj: dict, trainable: dict, optimizer, losses: list) -> Path:
 294     """One checkpoint: the adapter as a diffusers LoRA file, the optimizer's state, and a record with the step, the
 295     losses so far and each file's digest. It is written as a directory under a temporary name and renamed into place,
 296     so a checkpoint directory is there only when it is whole. A failure before the rename removes the temporary
 297     directory. After the rename nothing is removed: if flushing the checkpoints directory then fails, the error is
 298     `files.NotDurable` and the checkpoint stands."""
 299     final = checkpoint_path(run_dir, step)
 300     final.parent.mkdir(parents=True, exist_ok=True)
 301     tmp = files.temporary(final)
 302     adapter = lora.Adapter(lora.file_tensors(trainable, TRAINED),
 303                            {m: traj["adapter"]["alpha"] for m in {lora.RUNTIME_RE.match(k)["module"] for k in trainable}})
 304     try:
 305         tmp.mkdir()
 306         lora.save(tmp / "adapter.safetensors", adapter)
 307         mx.save_safetensors(str(tmp / "optimizer.safetensors"), train.optimizer_state(optimizer, trainable))
 308         described = {"format": CHECKPOINT, "step": step, "trajectory": files.sha256_json(traj),
 309                      "rng": {"scheme": "seeded per step", "seed": traj["hyper"]["seed"], "next_step": step},
 310                      "files": {name: files.sha256_file(tmp / name) for name in ("adapter.safetensors", "optimizer.safetensors")},
 311                      "losses": losses}
 312         (tmp / "checkpoint.json").write_text(json.dumps(described, indent=1, sort_keys=True, allow_nan=False) + "\n")
 313         for name in ("adapter.safetensors", "optimizer.safetensors", "checkpoint.json"):
 314             files.sync_file(tmp / name)
 315         os.rename(tmp, final)
 316     except BaseException:
 317         shutil.rmtree(tmp, ignore_errors=True)
 318         raise
 319     files.flush_renamed(final)  # after the rename: a failure here is `files.NotDurable`, and the checkpoint stands
 320     return final
```
📍 [`src/h3mlx/trainer.py:331-361`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/trainer.py#L331-L361)
```python
 331 def read_checkpoint(path: Path, traj: dict, cfg: dict) -> tuple[int, dict, dict, list]:
 332     """(step, trainable matrices, optimizer state, losses) from a checkpoint directory, if it is whole and belongs to
 333     this trajectory; otherwise `Refused`. Nothing falls back to an older checkpoint: a damaged one is to be looked
 334     at, not skipped."""
 335     try:
 336         described = json.loads((path / "checkpoint.json").read_text())
 337     except (OSError, ValueError) as e:
 338         raise Refused(f"{path}: the checkpoint's record cannot be read ({e})") from e
 339     if not isinstance(described, dict) or described.get("format") != CHECKPOINT:
 340         raise Refused(f"{path} is not a {CHECKPOINT} checkpoint")
 341     step = described.get("step")
 342     if step != int(path.name[5:]) or described.get("trajectory") != files.sha256_json(traj):
 343         raise Refused(f"{path} belongs to another run or another step (its record says step {step})")
 344     for name, digest in described.get("files", {}).items():
 345         if not (path / name).is_file() or files.sha256_file(path / name) != digest:
 346             raise Refused(f"{path / name} is missing or is not the file the checkpoint describes")
 347     if set(described.get("files", {})) != {"adapter.safetensors", "optimizer.safetensors"}:
 348         raise Refused(f"{path}: the checkpoint does not list its two files")
 349     losses = described.get("losses")
 350     if not isinstance(losses, list) or len(losses) != step:
 351         raise Refused(f"{path}: {step} steps but {len(losses) if isinstance(losses, list) else 'no'} losses")
 352     adapter = lora.load(path / "adapter.safetensors")
 353     spec = traj["adapter"]
 354     modules = adapter_modules(cfg, spec["targets"], spec["refiner"])
 355     lora.check(adapter, cfg)
 356     if sorted(adapter.modules) != sorted(modules) or any(adapter.rank(m) != spec["rank"] or adapter.alpha[m] != spec["alpha"]
 357                                                           for m in modules):
 358         raise Refused(f"{path}: the saved adapter is not this run's (modules, rank or alpha differ)")
 359     trainable = {k: v.astype(mx.float32) for k, v in lora.runtime_tensors(adapter, TRAINED).items()}
 360     saved = mx.load(str(path / "optimizer.safetensors"), format="safetensors")
 361     return step, trainable, saved, losses
```

A checkpoint is a directory, `checkpoints/step-NNNNNNNN/`: the adapter as a diffusers LoRA file (§1.5), the
optimizer's state (§9.5), and a record with the step, the losses so far, each file's sha256, a digest of the
trajectory, and the random state, which is the seed and the next step (§11.3). It is written under a temporary name
and renamed, so a checkpoint directory is there only when it is whole. One is written every `--checkpoint-every`
steps and when a session ends.

`--resume` goes on from the latest checkpoint:

- **It refuses another run.** The record's trajectory must be this command's, field for field. A changed learning
  rate, seed, rank, clip, shift, density, target list, gradient method or encode choice is refused with the field
  named, and so is a changed version of this package or of a library
  (`test_a_resume_that_is_another_run_is_refused`, `test_a_resume_under_other_code_or_into_nothing_is_refused`).
- **It refuses a damaged checkpoint.** A file that is not the one the checkpoint describes, a missing record, a
  checkpoint of another trajectory or under another step's name: the resume stops and says so. It does not fall back
  to an older checkpoint, and it has loaded no weights and changed no file when it stops
  (`test_a_damaged_checkpoint_is_refused_not_skipped`). A checkpoint that is missing altogether is §11.4's case.
- **Steps after the last checkpoint are done again.** The record may be ahead of the checkpoint; the resumed session
  says how many steps it discards, and the losses come from the checkpoint.
- A new run refuses a directory that already holds any of a run's outputs (§11.4).

Two tests of the same claim, that a resumed run ends where an uninterrupted one does:

- `test_a_run_that_dies_resumes_to_the_uninterrupted_runs_weights`: a run made to fail in its eighth step, resumed.
  The final adapter is the uninterrupted run's byte for byte, and so are the optimizer's state at steps 10 and 12 and
  the loss at every step.
- `test_a_killed_process_resumes_to_the_same_weights`: the script as a process of its own, sent SIGKILL once its
  record shows 20 steps, wherever that lands. Resumed, it reaches step 400 with the uninterrupted run's adapter, byte
  for byte.

`test_checkpoints_are_whole_or_absent` covers the write itself: a rename that fails leaves no checkpoint directory; a
writer that dies between the checkpoint's two files with no chance to clean up leaves only a temporary directory,
which is not resumed from and is cleared by the next session.

A failure that comes after a rename, when the directory will not flush, removes nothing, here as in the cache
(§10.2): the checkpoint, the record or the adapter is in place and whole, the session ends with
`files.NotDurable`, a checkpoint that stands is in the record's list, and the run resumes to the uninterrupted
run's weights (`test_a_failure_after_a_rename_removes_nothing`, at each of the three writes). A run stops for this
rather than going on: storage that will not flush is to be looked at before hours of training are trusted to it.

Byte-for-byte equality here is on one machine, with adapters on attention and feed-forward matrices. On the default
device a gradient that passes through a bf16 modulation table can differ by a rounding step from run to run (§7.1),
so an adapter on those modules would resume to weights that close, not to the same bytes.

### 11.9 From pictures

When the cache lacks an example, the script loads the VAE and the text encoder once (`cache.load_encoders`), encodes what is missing, writes
the entries and frees both before the transformer is loaded. `test_a_run_from_pictures_encodes_what_the_cache_lacks`
runs that on the tiny release (it needs the release's tokenizer and processor files): a text-to-image example and an
edit example from an empty cache, two entries written, three steps trained; a second run finds both entries, loads no
encoder and trains the same weights.

On real weights this path has never run.

### 11.10 The tests

| test | pins |
| --- | --- |
| `test_a_missing_argument_is_an_error_that_names_it` (×27), `test_only_the_backward_way_has_a_default` | no defaults |
| `test_the_script_turns_the_hub_off_before_it_imports_anything` | a process running the script has the Hugging Face hub off: it cannot download |
| `test_arguments_that_do_not_make_a_run_are_refused` (×21), `test_a_dataset_that_is_not_one_is_refused` (×8) | what is refused before anything is read |
| `test_plan_only_opens_no_weights_and_writes_nothing` | the plan with every loader raising, and its numbers against a real run |
| `test_the_loop_is_the_pieces_in_order` | six steps against the same steps written out from the tested pieces |
| `test_the_loop_overfits_one_example` | the loss falls on one example; the adapter file |
| `test_the_record_holds_the_run` | every argument, the versions, the seed, the step, a line per step; each example once a pass |
| `test_weighting_and_density_reach_the_loop`, `test_the_two_bounded_gradients_train_the_same_weights`, `test_a_frozen_assistant_is_used_and_not_saved`, `test_a_bf16_base_trains` | the options |
| `test_a_bf16_base_with_sharp_attention_trains_on_the_right_gradient` | a bf16 base with attention scores in the tens of thousands: the logged loss and norm of the first step against the step by hand with a float32 attention written out, and what the script logs with the bf16 attention call put back |
| `test_the_wall_clock_stops_the_session_and_the_run_resumes`, `test_the_memory_limit_stops_the_session`, `test_max_steps_finishes_the_run_and_a_resume_can_extend_it` | each limit stops the run |
| `test_sigterm_stops_at_the_next_boundary_with_a_checkpoint`, `test_a_second_process_is_refused_and_sigterm_stops_the_first_cleanly` | SIGTERM: a clean stop with a checkpoint, in this process and in another; two processes on one run directory |
| `test_one_session_in_a_run_directory_at_a_time`, `test_a_new_run_overwrites_nothing` (×3) | the run directory's lock; a new run and the three outputs |
| `test_a_resume_goes_on_from_the_records_latest_checkpoint_or_not_at_all`, `test_a_resume_past_its_max_steps_is_refused`, `test_a_refused_resume_encodes_nothing` | no rollback, no restart from nothing, no resume past the limit, nothing heavy before a refusal |
| `test_every_safetensors_dtype_has_a_size` | the plan's size table |
| `test_a_run_that_dies_resumes_to_the_uninterrupted_runs_weights`, `test_a_killed_process_resumes_to_the_same_weights` | resume, after a failure and after a kill |
| `test_a_resume_that_is_another_run_is_refused` (×9), `test_a_resume_under_other_code_or_into_nothing_is_refused`, `test_a_damaged_checkpoint_is_refused_not_skipped` (×6) | what a resume refuses |
| `test_checkpoints_are_whole_or_absent` | atomic checkpoints |
| `test_a_failure_after_a_rename_removes_nothing` (×3) | a directory that will not flush after a checkpoint's, the record's or the adapter's rename: nothing removed, the run resumable |
| `test_every_draw_is_a_function_of_the_seed_and_the_step` | the seeds, pinned; the adapter's start; each picture's posterior seed |
| `test_a_run_from_pictures_encodes_what_the_cache_lacks` (release fixtures) | the encode path on tiny encoders |

## 12. Sampling with a trained adapter

A training run is judged by sampling with what it trained. The sampling script takes the adapter:

    uv run python scripts/run_p_h3_image.py --weights P-H3-Image --prompt-file prompt.txt --out out/img \
        --lora runs/first/adapter.safetensors --lora-scale 1.0

`--lora` is the `adapter.safetensors` the training script writes (a checkpoint's or the final one), or any LoRA file
with one component that `lora.load` reads (§1.5, §2). `--lora-scale` has no default: 1.0 is the adapter as trained,
and leaving it out is an error that names it. Without `--lora` the script does what it did. This section changes
neither the forward nor the sampler: `src/h3mlx/ops.py` and `src/h3mlx/image.py` are the files they were.

### 12.1 Attached, not merged

📍 [`src/h3mlx/lora.py:732-773`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L732-L773)
```python
 732 def for_sampling(params: dict, cfg: dict, adapter: Adapter, scale) -> dict:
 733     """The weights a sampler runs with `adapter` at `scale` times its trained strength; `params` is not modified.
 734
 735     The adapter is attached, not merged. It stays beside the base weights and `ops.plinear` adds its delta on every
 736     call: the computation diffusers runs after `load_lora_adapter`, and the one the port tests against PEFT. Its
 737     matrices are cast to their base weight's dtype, as that loader casts them. Merging is not the same picture on a
 738     bf16 base: the sum is rounded once into each weight, which has eight significant bits, so whatever part of the
 739     delta is under half a step of its weight is rounded away, and a small adapter can vanish whole. Attached, the
 740     adapter also keeps its precision when the base is quantized afterwards: `load.quantize_transformer` touches the
 741     base matrices only, so the result is the quantized base plus the dense delta, which is what PEFT computes over a
 742     quantized Linear.
 743
 744     A scale of exactly 0 returns `params` itself. The picture is then the one without the adapter bit for bit. An
 745     attached adapter scaled by zero (diffusers at weight 0) adds a zero delta instead: the same numbers, but a
 746     negative zero in a Linear's output becomes a positive one.
 747
 748     Whatever the scale, the adapter is checked against the transformer and its matrices are read and must be finite:
 749     one NaN in an adapter is a NaN in every picture, and here it is refused before anything else runs. They must
 750     stay finite once cast to their base weight's dtype too: a float32 factor of 1e5 is an infinity over a float16
 751     base (the release's bf16 has float32's range)."""
 752     check_scale(scale)
 753     check(adapter, cfg)
 754     finite = {key: mx.isfinite(tensor).all() for key, tensor in adapter.tensors.items()}
 755     mx.eval(finite)
 756     bad = sorted(key for key, ok in finite.items() if not ok.item())
 757     if bad:
 758         raise ValueError(f"{len(bad)} of the adapter's {len(finite)} matrices hold a NaN or an infinity, e.g. {bad[:3]}")
 759     carried = carried_lora_keys(params)
 760     if carried:  # on any Linear, not only the adapter's own: scale 0 hands these weights back, and they would run it
 761         raise ValueError(f"these weights already carry a LoRA ({len(carried)} keys, e.g. {carried[:3]}): the sampler "
 762                          "takes the base weights and one adapter")
 763     if scale == 0:
 764         return params
 765     out = attach(params, cfg, adapter, scale=scale)
 766     cast = {runtime_key(m, x): out[runtime_key(m, x)] for m in adapter.modules for x in ("A", "B")}
 767     finite = {key: mx.isfinite(tensor).all() for key, tensor in cast.items()}
 768     mx.eval(finite)
 769     bad = sorted(key for key, ok in finite.items() if not ok.item())
 770     if bad:
 771         raise ValueError(f"{len(bad)} of the adapter's {len(finite)} matrices are finite in the file and not in their "
 772                          f"base weight's dtype, e.g. {bad[:3]}")
 773     return out
```

The choice was between `attach` (§1.3) and `merge` (§1.4), and each of the sampler's options points the same way.

- **It is diffusers' computation.** `load_lora_adapter` wraps each `Linear`, and the forward adds
  `B(A x) * scaling` to the base result on every call; nothing is merged unless the caller asks for it
  (`fuse_lora`). `plinear` is that line (§1.1), so the sampler with an adapter is as close to diffusers as it is
  without one (§12.4).
- **A bf16 weight has eight significant bits.** Merging rounds `W + delta` once into each weight, and whatever part
  of the delta is under half a step of its weight is gone. Attached, the delta is computed on its own, at its own
  size, and only then added to the Linear's output. `test_merging_into_bf16_loses_a_small_adapter_and_attaching_keeps_it`
  builds the extreme case on one Linear: every weight between 2^-6 and 2^-5, and a rank-one delta of 3.05e-5 an
  entry. Merged, none of the 3,072 weights changes and the adapter has vanished. Attached, the output moves by
  1.5e-3, the size the delta predicts.
- **`--quantize` needs no second rule.** Merging would have to come before quantizing (`merge` refuses a quantized
  weight), and the delta would then be quantized with the base. Attached, the adapter goes on the weights as they
  are loaded, and `generate` quantizes the base matrices where it always does, after the text and the pictures are
  encoded. The quantizer's pattern matches base weights only, so the adapter's matrices stay dense beside them, in
  the base's dtype. A Linear is then the quantized base's product plus the dense delta, the shape PEFT's own layers
  over a quantized `Linear` have (`tuners/lora/bnb.py`, lines 251 to 269: the base layer's result, then
  `lora_B(lora_A(x)) * scaling` added).
- **The modulation cache needs nothing.** It is on by default, and its tables come from the same `plinear` call
  (§1.3). An adapter on the block matrices, which is what the training script trains, is not in those tables at
  all. One on the time embedder or the AdaLN projections is, and the cache carries it.
- **The cost** is two small products for each Linear on each call: the rank over the width of the base product's
  work, and the adapter's matrices in memory.

At a scale of exactly 0 nothing is attached, so the rows and the picture are those of a run without `--lora`, bit
for bit, by construction. diffusers at weight 0 keeps the adapter and adds a zero delta. That gave the same picture
on the tiny release, and it is not the same bits in general: a negative zero in a Linear's output becomes a positive
one (`test_an_adapter_attached_at_zero_is_not_bit_for_bit_nothing`, on the CPU stream, where a matmul can return
one). The adapter is still read and checked at 0. Its matrices must be finite at any scale: one NaN in an adapter is
a NaN in every picture, and it is refused when the file is read, before the text encoder runs.

### 12.2 The scale

📍 [`peft:src/peft/tuners/lora/layer.py:783-794`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L783-L794)
```python
 783     def set_scale(self, adapter: str, scale: float) -> None:
 784         """Set the scale of the given adapter to the initial scale multiplied by the provided factor
 785
 786         The initial scale is determined by the configured `r` (rank) and `lora_alpha`.
 787         """
 788         if adapter not in self.scaling:
 789             # Ignore the case where the adapter is not in the layer
 790             return
 791         if self.use_rslora.get(adapter, False):
 792             self.scaling[adapter] = scale * self.lora_alpha[adapter] / math.sqrt(self.r[adapter])
 793         else:
 794             self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]
```

📍 [`src/h3mlx/lora.py:205-209`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L205-L209)
```python
 205 def check_scale(scale) -> None:
 206     """A scale is a finite number. PEFT's `set_scale` takes anything; NaN or infinity would poison every Linear the
 207     adapter is on."""
 208     if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
 209         raise ValueError(f"an adapter's scale must be a finite number, got {scale!r}")
```

`attach` takes the scale (§1.3): a module's scaling becomes `scale * alpha / rank`, in that order, rounded to
float32 where the delta is multiplied (§1.1). That is `set_scale`, which diffusers reaches through
`set_adapters(name, weights=scale)` on the transformer. The pipeline has a second route, a scale for one call:

📍 [`diffusers:src/diffusers/utils/peft_utils.py:286-332`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/utils/peft_utils.py#L286-L332)
```python
 286 def apply_lora_scale(kwargs_name: str = "joint_attention_kwargs"):
 287     """
 288     Decorator to automatically handle LoRA layer scaling/unscaling in forward methods.
 289
 290     This decorator extracts the `lora_scale` from the specified kwargs parameter, applies scaling before the forward
 291     pass, and ensures unscaling happens after, even if an exception occurs.
 292
 293     Args:
 294         kwargs_name (`str`, defaults to `"joint_attention_kwargs"`):
 295             The name of the keyword argument that contains the LoRA scale. Common values include
 296             "joint_attention_kwargs", "attention_kwargs", "cross_attention_kwargs", etc.
 297     """
 298
 299     def decorator(forward_fn):
 300         @functools.wraps(forward_fn)
 301         def wrapper(self, *args, **kwargs):
 302             from . import USE_PEFT_BACKEND
 303
 304             lora_scale = 1.0
 305             attention_kwargs = kwargs.get(kwargs_name)
 306
 307             if attention_kwargs is not None:
 308                 attention_kwargs = attention_kwargs.copy()
 309                 kwargs[kwargs_name] = attention_kwargs
 310                 lora_scale = attention_kwargs.pop("scale", 1.0)
 311
 312                 if not USE_PEFT_BACKEND and lora_scale != 1.0:
 313                     logger.warning(
 314                         f"Passing `scale` via `{kwargs_name}` when not using the PEFT backend is ineffective."
 315                     )
 316
 317             # Apply LoRA scaling if using PEFT backend
 318             if USE_PEFT_BACKEND:
 319                 scale_lora_layers(self, lora_scale)
 320
 321             try:
 322                 # Execute the forward pass
 323                 result = forward_fn(self, *args, **kwargs)
 324                 return result
 325             finally:
 326                 # Always unscale, even if forward pass raises an exception
 327                 if USE_PEFT_BACKEND:
 328                     unscale_lora_layers(self, lora_scale)
 329
 330         return wrapper
 331
 332     return decorator
```

The transformer's forward carries that decorator, and the P-H3-Image pipeline passes its `attention_kwargs` through.
`scale_lora_layers` multiplies each layer's scaling in place and `unscale_lora_layers` divides it back afterwards, so
the number is `alpha / rank * scale`, the other order. The two routes gave identical pictures in every case tested,
and the port's is the first: it is a function of the arguments, with nothing to undo.

- **The scale is exactly a factor on the delta.** `test_scale_on_one_linear_is_the_delta_times_the_scale` holds one
  Linear to `base + B(A x) * float32(scale * alpha / rank)` bit for bit, in float32 and in bf16, and to PEFT's layer
  after `set_scale`. `test_a_power_of_two_scale_is_the_adapter_with_lora_b_scaled` holds the whole transformer, with
  an adapter on every Linear: at 0.5, 4 and -2 the output is, bit for bit, that of the adapter whose `lora_B` is
  that many times as large. `test_scale_is_diffusers_adapter_weight` compares the scalings and the forward with
  diffusers at 0.6, -1.3, 2 and 1/3, by both routes.
- **Any finite number.** A negative scale subtracts the adapter's delta, as it does in diffusers. NaN and infinity
  are refused. (An alpha is still positive, §1.3: the scale is the caller's, the alpha is the file's.)
- **Loading the file into diffusers to compare.** At the pinned commit, `load_lora_adapter` on a bare file path reads
  the tensors and not the header:

📍 [`diffusers:src/diffusers/loaders/lora_base.py:221-225`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/loaders/lora_base.py#L221-L225)
```python
 221     if not isinstance(pretrained_model_name_or_path_or_dict, dict):
 222         # Let's first try to load .safetensors weights
 223         if (use_safetensors and weight_name is None) or (
 224             weight_name is not None and weight_name.endswith(".safetensors")
 225         ):
```

  `use_safetensors` is `None` unless the caller sets it, so with no `weight_name` the branch that reads the
  metadata is skipped, the file is loaded by the fallback, and every module runs at alpha = rank. For a file of
  rank 4 and alpha 6 that is a scaling of 1.0 where the file says 1.5 (measured). With `weight_name=` or
  `use_safetensors=True` the metadata is read. The port reads the header always (§1.5), so a comparison has to load
  the file one of those two ways; the tests use `weight_name`.

### 12.3 The plan, the record, and what is refused

📍 [`src/h3mlx/lora.py:776-800`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L776-L800)
```python
 776 def from_header(path, prefix: str | None = None, head_dim: int = HEAD_DIM) -> Adapter:
 777     """The adapter `load` would give, from the file's safetensors header alone: the same modules, shapes and alphas,
 778     with every tensor a zero stand-in that is never evaluated. Nothing but the header is read from the file, so a
 779     plan can name an adapter (`summary`) and `check` can refuse one that does not fit before any tensor is opened.
 780
 781     The stand-ins go through `read`, the path a loaded file takes, so every layout it reads is covered and every
 782     refusal that needs no tensor value is raised here too. One thing only a load can see: a file in the original
 783     layout that carries `.alpha` tensors has them folded into its matrices when it is loaded, and a zero, negative
 784     or non-finite one is refused then."""
 785     from . import files
 786
 787     header = files.safetensors_header(path)
 788     if not isinstance(header, dict):
 789         raise ValueError(f"{path}: not a safetensors header")
 790     metadata = header.pop("__metadata__", None)
 791     metadata = {} if metadata is None else metadata  # absent or null; a falsy non-map (0, [], "") is still refused
 792     if not (isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())):
 793         raise ValueError(f"{path}: not a safetensors header (its __metadata__ is not a map of strings to strings)")
 794     dims_ok = lambda shape: all(type(d) is int and 0 <= d < 2**31 for d in shape)  # noqa: E731 (bool is not int here)
 795     if not all(isinstance(entry, dict) and isinstance(entry.get("shape"), list) and dims_ok(entry["shape"])
 796                for entry in header.values()):
 797         raise ValueError(f"{path}: not a safetensors header (a tensor without a shape of non-negative integers)")
 798     stand_ins = {key: mx.ones(()) if key.endswith(".alpha") else mx.zeros(tuple(entry["shape"]))
 799                  for key, entry in header.items()}
 800     return read(stand_ins, metadata, prefix, head_dim)
```
📍 [`src/h3mlx/lora.py:814-836`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/src/h3mlx/lora.py#L814-L836)
```python
 814 def summary(adapter: Adapter, scale=1.0) -> dict:
 815     """What an adapter is, for a plan or a record: how many Linears it is on and which (each kind of Linear with the
 816     blocks it covers, `*` standing for the block's index; a Linear outside the blocks maps to None), the ranks and
 817     alphas in use, the scalings the forward multiplies by at `scale` (`scale * alpha / rank`), and the number of
 818     parameters. From shapes alone, so it works on `from_header`'s stand-ins; a scale whose scalings overflow float32 is
 819     refused here as `attach` refuses it, so a plan does not pass what the run would refuse."""
 820     check_keys(adapter)
 821     check_scalings(adapter, scale)
 822     kinds = {}
 823     for module in adapter.modules:
 824         m = re.match(r"^(.*?)\.(\d+)\.(.*)$", module)
 825         if m is None:
 826             kinds[module] = None
 827         else:
 828             kinds.setdefault(f"{m[1]}.*.{m[3]}", []).append(int(m[2]))
 829     return {
 830         "modules": len(adapter.modules),
 831         "module_kinds": {kind: None if blocks is None else _ranges(blocks) for kind, blocks in sorted(kinds.items())},
 832         "ranks": sorted({adapter.rank(m) for m in adapter.modules}),
 833         "alphas": sorted({float(adapter.alpha[m]) for m in adapter.modules}),
 834         "scalings": sorted({float(scale * adapter.alpha[m] / adapter.rank(m)) for m in adapter.modules}),
 835         "parameters": sum(math.prod(t.shape) for t in adapter.tensors.values()),
 836     }
```
📍 [`scripts/run_p_h3_image.py:105-112`](https://github.com/sublimator/minimax-h3-mlx/blob/2bc9db813a91291e11fa0aeaa81ab0544715fcaf/scripts/run_p_h3_image.py#L105-L112)
```python
 105 def adapter_plan(path: Path, scale: float, cfg: dict) -> dict:
 106     """What `--lora` applies, from the adapter file's header alone (no tensor is read): the file, its hash, the scale,
 107     and the adapter's Linears, ranks, alphas and scalings (`lora.summary`). An adapter that is not a LoRA file the port
 108     reads, or that does not fit this transformer's Linears, is a ValueError here."""
 109     shapes = lora.from_header(path, head_dim=cfg["attention_head_dim"])
 110     lora.check(shapes, cfg)
 111     return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "scale": scale,
 112             "applied": scale != 0, **lora.summary(shapes, scale)}
```

- **`--plan-only` opens no weights.** The plan's `lora` entry gives the file, its hash and size, the scale, whether
  it is applied, the Linears by kind with the blocks each kind covers (`transformer_blocks.*.attn.to_q: 0-49`), the
  ranks, the alphas, the scalings the forward multiplies by, and the parameter count. All of it comes from the
  file's safetensors header. `from_header` puts zero stand-ins of the header's shapes through `read`, the path a
  loaded file takes, so each layout the loader reads is described the same way and no tensor is read
  (`test_from_header_is_the_loaded_adapter_and_reads_no_tensor`, eleven kinds of file with the tensor loader
  raising).
- **What does not fit is an argument error.** A missing scale, a scale without an adapter, a scale that is not
  finite, a file that is not there or is not a LoRA file, and an adapter that does not fit this transformer's
  Linears are refused when the arguments are checked, before a model loads and before `run.json` exists. Scale 0
  excuses none of them.
- **Finite, and still finite where it is used.** A scale can be a finite Python float and overflow float32, the
  dtype the forward multiplies by, once it is `scale * alpha / rank` (1e40 does): `check_scalings` refuses it, in
  the plan's summary and again in `attach`. The adapter's matrices are checked as read and again once cast to their
  base weight's dtype, since a finite float32 factor can be an infinity in float16 (bf16 has float32's range)
  (`test_a_scale_finite_in_python_and_not_in_float32_is_refused`,
  `test_factors_finite_in_the_file_and_not_in_the_base_dtype_are_refused`,
  `test_the_plan_refuses_a_scale_that_overflows_float32`).
- **Weights that already carry an adapter are refused at every scale.** At 0 the sampler hands back the weights it
  was given, so without the check those weights would still run the adapter already on them. The check is on the
  whole dict (`carried_lora_keys`), so an adapter on another Linear than the new one's is caught too, named or not.
- **A malformed header is a ValueError, so an argument error.** A header shorter than its stated length, a
  `__metadata__` that is not a map of strings to strings, a tensor whose shape is not a list of non-negative
  integers, and a `lora_adapter_metadata` that is not a JSON object are each refused as such
  (`test_from_header_refuses_a_malformed_header_as_a_value_error`). They used to escape as other exceptions, past
  the arguments' check.
- **An original-layout file is read at the transformer's head width.** A DiffSynth-Studio file fuses q, k and v per
  head (§2.3). The plan, the sampler's load and the training script's `--assistant` pass the transformer's
  `attention_head_dim`, where they used to take the default of 128: a tiny model's file was refused, and at 4 x 32
  one would have been split at the wrong stride with no error
  (`test_the_sampler_reads_an_original_file_at_the_transformers_head_width`).
- **The record.** `run.json` has the plan's entry, the two arguments, and a `lora` event when the adapter goes on,
  before the run starts. `final_rows.npz` carries the file's path, its hash and the scale in its `source` record
  beside the prompt and the seed, and `null` when there was no adapter; a re-decode (`--decode-from`) reports them as
  its source, and takes neither option itself.
- **The hash is the file that was read.** The plan hashes the file, and the script hashes it again once its tensors
  are in memory. A file that changed in between stops the run.

### 12.4 Against diffusers

The oracle is diffusers' P-H3-Image pipeline on the tiny release with the same file loaded into its transformer.
`test_end_to_end_with_an_adapter_matches_diffusers` runs both workflows with an adapter of rank 4 and alpha 6 on
every block matrix. The largest difference between the two float pictures, with the bound at 2e-5 as it is without an
adapter (the P-H3-Image guide, §10):

| | text-to-image | edit | the adapter moves diffusers' text-to-image picture by |
| --- | --- | --- | --- |
| no adapter | 1.8e-7 | 1.8e-7 | |
| as trained (1.0) | 1.8e-7 | 1.8e-7 | 3.6e-2 |
| 0.6 | 2.1e-7 | 1.8e-7 | 1.8e-2 |
| -1.3 | 1.8e-7 | 1.8e-7 | 3.2e-2 |

The last column is what makes the comparison a test: the adapter moves the picture by more than 500 times the
bound, and the test asserts that it does. Through the command line (`test_cli_lora_matches_diffusers`, a larger adapter), the PNG differs from
diffusers' by 0 levels of 255 at 1.0, 0.6 and -1.3, where the adapter moves diffusers' PNG by 76, 35 and 88 levels.

With the sampler's other options, on the tiny release:

| option | result | test |
| --- | --- | --- |
| modulation cache on and off | the same rows, bit for bit, for an adapter on the block matrices and for one on every Linear | `test_cli_lora_gives_the_same_rows_with_the_modulation_cache_on_and_off` |
| scale 0 | the rows and the PNG of the run without `--lora`, bit for bit | `test_cli_lora_scale_zero_is_the_run_without_one` |
| `--quantize 8` | 0.02 of 255 (mean) from the unquantized run with the adapter; 2.8 from the quantized run without it | `test_cli_lora_with_quantize_quantizes_the_base_and_keeps_the_adapter` |
| `--quantize 4` | 0.30 of 255 from the unquantized run with the adapter; 2.8 from the quantized run without it | the same |

The quantized rows are held to the bounds the quantized run has without an adapter (0.05 and 0.5).
`test_a_quantized_base_keeps_the_adapter_dense` checks the order in the package: after `for_sampling` the quantizer
takes the 18 base matrices and leaves every tensor of the adapter as it was, one Linear is the quantized product
plus the dense delta bit for bit, and the whole forward is the forward over the dequantized matrices with the same
adapter.

All of this is on tiny float32 models. The adapter over bf16 weights rests on §1.3's dtype test and on the one
Linear above; no real adapter has been sampled by these tests.

### 12.5 The tests

| test | pins |
| --- | --- |
| `test_scale_is_diffusers_adapter_weight` (×4) | each Linear's scaling and the forward against diffusers' `set_adapters(weights=)` and `attention_kwargs` |
| `test_scale_on_one_linear_is_the_delta_times_the_scale` (×8) | one Linear, bit for bit, float32 and bf16; PEFT's layer after `set_scale` |
| `test_a_power_of_two_scale_is_the_adapter_with_lora_b_scaled` (×3) | the whole transformer, bit for bit, an adapter on every Linear |
| `test_scale_one_changes_nothing_and_a_named_adapter_takes_a_scale` | the default scale leaves `attach` as it was; a named adapter's scaling; what a scale may be |
| `test_for_sampling_attaches_and_leaves_the_base_alone` | the base arrays are the same objects; the adapter cast and scaled as `attach` does |
| `test_merging_into_bf16_loses_a_small_adapter_and_attaching_keeps_it` | why not `merge` |
| `test_scale_zero_is_the_weights_themselves`, `test_an_adapter_attached_at_zero_is_not_bit_for_bit_nothing` | 0 attaches nothing, and why |
| `test_for_sampling_refuses` | a scale that is not a finite number; a NaN or an infinity in a matrix, at 1.0 and at 0; a misfit; weights that carry a LoRA |
| `test_a_quantized_base_keeps_the_adapter_dense` | the order with `--quantize` |
| `test_from_header_is_the_loaded_adapter_and_reads_no_tensor`, `test_from_header_refuses_what_is_not_a_lora_file` | the header-only adapter for every kind of file; what is not one |
| `test_summary_names_the_linears_the_ranks_and_the_scalings` | the plan's description, exactly |
| `test_end_to_end_with_an_adapter_matches_diffusers` (×2, release fixtures) | the pipeline against diffusers, three scales, both of its routes |
| `test_cli_lora_matches_diffusers` (×3, release fixtures) | the command line against diffusers |
| `test_cli_lora_scale_zero_is_the_run_without_one`, `test_cli_lora_gives_the_same_rows_with_the_modulation_cache_on_and_off` (×2) | bit-identical rows |
| `test_cli_lora_plan_only_names_the_adapter_and_opens_no_weights` | the plan's entry with every loader raising |
| `test_cli_lora_record_and_redecode` | the record, the event's place, the rows' source, a re-decode |
| `test_cli_lora_refuses_at_the_arguments` (ten cases), `test_cli_redecode_refuses_generation_options` (two more cases) | argument errors |
| `test_cli_lora_refuses_a_broken_adapter_before_the_text_encoder` | a NaN in the file; a file that changed while it was read |
| `test_cli_lora_with_quantize_quantizes_the_base_and_keeps_the_adapter` (×2) | `--quantize` with `--lora`, 8 and 4 bits |

## 13. What each test would catch

Each wrong implementation below was made on a copy of the sources and run against the tests; the named test failed.

| wrong implementation | caught by |
| --- | --- |
| `plinear` ignores the scaling; the scaling inverted | `test_plinear_scaling_matches_peft_linear`, `test_diffusers_file_loads_in_mlx` |
| the scaling applied with MLX's own half-precision product | `test_half_precision_scaling_rounds_as_torch` |
| Kaiming fan taken from the rank; gain for `a = 0` | `test_kaiming_uniform_is_torchs` |
| `lora_B` not zero; adapter created in bf16 | `test_init_is_a_float32_no_op` |
| `keep_dtype` ignored | `test_adapter_dtypes_follow_peft` |
| no metadata: one rank's alpha for every module | `test_no_metadata_is_scaling_one` |
| pattern keys matched as plain suffixes; alpha pattern ignored on read or dropped on write | `test_pattern_key_is_pefts`, `test_rank_and_alpha_patterns_from_diffusers`, `test_mixed_ranks_and_alphas_to_diffusers` |
| metadata keys written without the prefix | `test_mlx_file_loads_in_diffusers` |
| header `alpha` ignored | `test_header_alpha_is_one_network_alpha` |
| delta weight unscaled; merge with the wrong sign | `test_merged_equals_unmerged` |
| `merge` accepts weights that carry the adapter | `test_merge_takes_plain_weights_only` |
| `save` casts every tensor to float32 | `test_half_precision_adapter_keeps_its_dtype_in_the_file` |
| a zero, negative, NaN or infinite alpha accepted by `check`, `init`, `read` (metadata or header) or `save`; NaN let through alone; the writer's JSON not strict | `test_alpha_must_be_positive_and_finite`, `test_saved_metadata_is_strict_json` |
| DoRA keys checked across the whole file, or not checked in the component read | `test_prefix_reads_one_component_and_ignores_the_others` |
| `check` or `save` not comparing the tensors with the modules; extra tensors, or missing ones, not noticed | `test_an_adapter_must_match_its_own_modules` |
| an empty prefix taken by `save` or by `read`; a `bias` setting other than none accepted | `test_an_empty_prefix_is_refused`, `test_a_bias_setting_other_than_none_is_refused` |
| a target that is not a `Linear` accepted | `test_refusals` |
| feed-forward halves not swapped; q, k, v in another order; no de-interleave; a wrong rename | `test_converts_to_diffusers_dicts` |
| alpha not folded, or all of it put on `lora_A`; folded with MLX's half-precision product | `test_converts_to_diffusers_dicts` (the last at rank 6) |
| alpha applied again after the fold | `test_header_alpha_with_a_converted_file` |
| the fold's `.alpha` check weakened to refuse zero only (run on the NaN case; on a negative alpha it would loop, as theirs does); leftover keys ignored | `test_bad_alpha_is_refused`, `test_refusals_shared_with_diffusers` |
| an original-layout file filtered by the caller's prefix; half a pair unchecked; `dora_scale` left to the converter | `test_original_layout_file_is_read_whole_for_either_partition`, `test_half_a_pair_names_the_module`, `test_dora_scale_in_an_original_file_is_refused` |
| target `noise − x_0` | `test_one_euler_step_along_the_target_lands_on_x0` |
| sample and noise swapped; `1 − t` of a float taken in float32; a half-precision product done MLX's way | `test_scale_noise_with_a_float_is_the_schedulers` |
| a tensor timestep's `1 − t` taken from the uncast value | `test_scale_noise_with_an_array_is_the_schedulers` (the bf16 cases; in float32 the result is the same bits) |
| a numpy scalar taken as a tensor; a numpy array taken as a number or as a tensor; a timestep with more axes accepted | `test_scale_noise_with_a_numpy_scalar_is_the_schedulers`, `test_scale_noise_refuses_a_numpy_array_as_the_scheduler_does`, `test_scale_noise_timestep_axes` |
| loss summed, computed in bf16, weight ignored, weights on the wrong examples | `test_loss_is_torchs_mse`, `test_weighted_loss` |
| loss broadcasts a mismatched target; weight shape unchecked, or its axes of 1 refused | `test_loss_refuses_shapes_that_do_not_line_up` |
| logit-normal's mean and std swapped, or drawn as `randn * std + mean`; `mode` without `− 1 + u` or with `cos(pi·u)`; the generator ignored | `test_density_matches_diffusers` |
| a density parameter not required, or a non-finite one accepted; a NaN std made a ValueError | `test_density_parameters_are_required_and_finite` |
| `sigma_sqrt` as `1 / sigma` or by `np.power`; `cosmap` without the square | `test_loss_weighting_is_diffusers` |
| half-precision or integer levels upcast | `test_loss_weighting_takes_float32_or_float64_only` |
| grid one point short, with its terminal 0, or ascending | `test_training_sigmas_are_the_schedulers` |
| lookup rounded instead of floored, in float64, or unclipped | `test_same_seed_same_sigmas_uniform`, `test_sample_sigmas_lookup` |
| a NaN `u` let through the range test; the grid's shape or its levels unchecked | `test_sample_sigmas_refuses_nan_and_a_bad_grid` |
| `flow_sigmas` or `shift_grid` testing `shift <= 0` only; the shift not checked as a float32 | `test_a_shift_that_is_not_a_positive_finite_float32_is_refused` |
| a later adapter's input cast from the original input; the deltas summed before they meet the base result; adapters applied in sorted order | `test_stacked_rounding_is_pefts` |
| a named adapter's products in the base's dtype | `test_stacked_rounding_is_pefts`, `test_products_with_float32_matrices_are_float32_over_a_bf16_base` |
| a named adapter's scaling ignored; `set_active` leaving every adapter on, or ignoring the order given | `test_stacked_forward_matches_peft_in_every_state` |
| `attach` overwriting a `Linear`'s adapter list; `split` handing over every named adapter | `test_runtime_keys_are_pefts`, `test_file_keys_carry_no_adapter_name` |
| the frozen adapter's delta cut from the gradient | `test_gradients_reach_the_trainable_factors_only`, `test_loss_and_gradients_are_torchs` |
| an unnamed adapter attached beside a named one; `merge` onto weights with a named adapter; a name's use checked on the adapter's own modules only | `test_stacking_refusals` |
| `attach` looking at the matrices only, or missing a scaling or a list entry left behind; the forward on a frozen half left as a `KeyError`; `join` not checking | `test_the_frozen_half_of_a_split_is_not_a_model`, `test_check_trainable` |
| `check_trainable` accepting an inactive name, half a pair or an empty group | `test_check_trainable`, `test_the_loss_refuses_matrices_the_forward_would_not_read` |
| a block's AdaLN projection, the final one, `proj_in`, `audio_proj_out` or `context_embedder` not through `plinear` | `test_stacked_forward_matches_peft_in_every_state`, `test_every_converted_module_attaches_and_gives_the_same_forward` |
| the modulation cache building its tables without the LoRA, or leaving a dropped projection's LoRA behind | `test_modulation_cache_carries_a_lora_on_the_adaln_projections` |
| the example's target `noise − x_0`; the picture's noise drawn before the references'; the references or the picture mixed with a Python float; references left clean; the picture's rows ahead of the references'; an edit packed as text-to-image | `test_example_is_diffusers_pieces` |
| audio rows left as plain noise, or at the picture's level whatever `audio_sigma`; the references' timestep not raised to the picture's | `test_example_is_diffusers_pieces`, `test_example_guards` |
| the loss mask covering the condition rows; `generated` keeping them | `test_loss_mask_is_the_generated_video_rows` |
| float16 rounding always on, or inverted; the choice dropped on the way to the encode | `test_reference_encode_float16_rounding_is_a_choice`, `test_encode_references_needs_the_choice` |
| the example's sigma unchecked | `test_example_refusals` |
| a sigma whose timestep is 1 accepted; the bound put at the recipe's 1e-6; the comparison strict the wrong way | `test_a_sigma_whose_timestep_rounds_to_one_is_refused` |
| latents with no height or no width accepted, or only the height checked | `test_example_refusals` |
| prompt tags: any integers, the audio tag, an empty array, floats or two axes accepted | `test_text_token_tags_are_text_or_vision` |
| gradients taken for every weight; the loss over the condition rows too; every row at the first timestep; the audio rows left out of the forward | `test_loss_and_gradients_are_torchs` |
| the weight dropped from the example's loss | `test_weight_scales_the_loss_and_the_gradient` |
| the loss or its gradient not checking its inputs; a text accepted when it only broadcasts, or without its type checked | `test_the_loss_refuses_matrices_the_forward_would_not_read`, `test_the_loss_refuses_a_text_of_the_wrong_shape` |
| the training gather summing in the table's dtype, widening bf16 only, or returning float32 | `test_the_training_gathers_gradient_is_torchs_index_select`, `test_the_training_forward_is_the_sampling_forward` |
| the staged forward's blocks or its end using the sampling gather; `dit_block` or `dit_outputs` ignoring the gather for one of the tables | `test_a_tables_gradient_over_a_bf16_base_is_the_rows_sum` |
| the training forward, or its block stage, evaluating after every block | `test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept` |
| a checkpointed block capturing its trainable matrices, or the time embedding | `test_bounded_gradients_equal_the_plain_gradient` |
| no checkpoint at all | `test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept` |
| block by block: the blocks' time-embedding cotangent dropped, or not handed to the first stage; the last stage's cotangent not 1; the final norm's projection counted before the blocks | `test_bounded_gradients_equal_the_plain_gradient` |
| block by block: nothing evaluated in the backward walk | `test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept` |
| either bounded gradient dropping the loss weight, or not checking its inputs | `test_bounded_gradients_at_other_levels_and_layouts`, `test_bounded_gradients_refuse_what_the_loss_refuses` |
| AdamW without bias correction; the weight decay dropped; epsilon, a beta or the learning rate fixed | `test_steps_are_torchs` |
| no clipping; clipping per tensor; clipping to twice the bound; the norm returned after clipping | `test_steps_are_torchs`, `test_clipping_is_torchs` |
| half-precision matrices accepted; a non-finite norm applied, or checked after the optimizer's state changed; keys unchecked; a zero or boolean bound accepted | `test_optimizer_step_refusals` |
| a gradient's dtype, shape or type unchecked, the shape checked only up to broadcasting, or the check made after the optimizer was touched | `test_optimizer_step_refusals` |
| `adamw` taking a negative, NaN or infinite learning rate or weight decay, a schedule, betas outside [0, 1), or an epsilon of zero; refusing a learning rate of zero | `test_adamw_checks_its_arguments_as_torch_does_and_eps`, `test_mlx_adamw_needs_bias_correction_to_be_torchs` |
| a key in both groups accepted; an adapter held in part accepted; the rest of an adapter on the frozen side counted as missing | `test_the_two_groups_share_nothing_and_make_a_whole_model`, `test_the_frozen_half_of_a_split_is_not_a_model` |
| the weights' block count unchecked, or only checked for too many; a matrix on no Linear of the config, in another shape or of another rank accepted; the loss not checking the config; a stray block key filed before the blocks | `test_check_config`, `test_bounded_gradients_refuse_what_the_loss_refuses`, `test_a_config_of_fewer_blocks_than_the_weights_is_refused` |
| the final-norm op ignoring the gather it is given, for one table or both | `test_norm_out_matches` |
| the cache key leaving out the versions, the encoders, a seed, a rounding choice or the prompt, or sorting the references; reference choices kept in a text-to-image key | `test_a_changed_input_or_choice_is_not_served_the_old_entry`, `test_a_changed_code_or_library_version_is_another_key`, `test_an_example_without_references_ignores_the_reference_choices` |
| the code digest of one file, by name only, or by content only; an identity that reads the weights, or ignores their sizes | `test_code_digest_follows_the_source`, `test_identity_reads_no_weights` |
| the cache reader not comparing the request, the record's key, the tensors' digest, the arrays or the format; a damaged record or missing tensors taken for a miss; the tensors loaded by extension; the comparison blind to a 1 for a True | `test_a_changed_input_or_choice_is_not_served_the_old_entry`, `test_an_entry_put_under_another_requests_name_is_refused`, `test_a_damaged_entry_is_an_error_not_a_miss`, `test_no_pickle_in_the_module`, `test_differences_names_what_differs` |
| the cache writing non-finite arrays, over an existing entry, straight to the final name, or leaving files behind after a failure; tags or the latents' shape unchecked | `test_tensors_that_are_not_an_entry_are_not_written`, `test_an_entry_is_written_once`, `test_a_write_that_fails_leaves_no_entry` |
| the recipe's sizing scaling up, flooring, ignoring the target, or taking the pipeline's vision size; the pipeline's sizing without its lower bound | `test_recipe_sizing_scales_down_to_the_target_and_never_up`, `test_pipeline_sizing_is_diffusers` |
| the encode ignoring the latent seed or rounding, the references' seed or rounding, the text dtype, the picture's shape or the resize; references encoded at the vision size | `test_a_text_to_image_entry_is_the_direct_encode`, `test_an_edit_entry_is_the_pipelines_conditioning` |
| a step's seed ignoring the step or the stream; the same example order every pass; the adapter's start or a picture's posterior seed ignoring its seed | `test_the_record_holds_the_run`, `test_every_draw_is_a_function_of_the_seed_and_the_step`, `test_the_loop_is_the_pieces_in_order` |
| the weighting, the weight decay, the clip, the conditioning level, the shift or the references not reaching the loop; the assistant not attached | `test_the_loop_is_the_pieces_in_order`, `test_a_frozen_assistant_is_used_and_not_saved` |
| a resume without the optimizer's state, from a fresh adapter, or keeping the record's later losses; `--resume` ignored | `test_the_wall_clock_stops_the_session_and_the_run_resumes`, `test_a_run_that_dies_resumes_to_the_uninterrupted_runs_weights` |
| the optimizer's state restored without its step or its moments, or saved with the moments swapped; a state for other matrices or shapes accepted | `test_a_saved_optimizer_goes_on_where_it_stopped`, `test_optimizer_state_refusals` |
| no wall-clock limit; no memory limit, one looked at between steps only, or one in the wrong unit; the dropped step applied; a stopped session returning 0; one step too many | `test_the_wall_clock_stops_the_session_and_the_run_resumes`, `test_the_memory_limit_stops_the_session`, `test_plan_only_opens_no_weights_and_writes_nothing` |
| the block-by-block gradient not reporting after a stage, on the way up or back | `test_the_blockwise_gradient_reports_after_every_stage` |
| no checkpoint when a session ends; checkpoints a step off; a checkpoint written in place; its files, its trajectory or its step unverified | `test_plan_only_opens_no_weights_and_writes_nothing`, `test_the_record_holds_the_run`, `test_checkpoints_are_whole_or_absent`, `test_a_damaged_checkpoint_is_refused_not_skipped` |
| a resume that does not compare the trajectory; a trajectory without the versions; a new run over an old one; the latent seed left out of the request | `test_a_frozen_assistant_is_used_and_not_saved`, `test_a_resume_under_other_code_or_into_nothing_is_refused`, `test_a_resume_that_is_another_run_is_refused` |
| the record written only at the end | `test_a_killed_process_resumes_to_the_same_weights` |
| the plan's parameters counted on one side, its rows without the references, its saved inputs sized for float32 whatever the base; `--plan-only` reading the weights or making the run directory | `test_plan_only_opens_no_weights_and_writes_nothing`, `test_a_bf16_base_trains` |
| the shift or the rank given a default; unknown dataset keys ignored; the reference choices not required | `test_a_missing_argument_is_an_error_that_names_it`, `test_a_dataset_that_is_not_one_is_refused`, `test_arguments_that_do_not_make_a_run_are_refused` |
| a lock that takes nothing, never waits, says nothing about its holder, or leaves the holder's note after release | `test_exclusive_is_released_when_its_holder_ends`, `test_a_writer_that_fails_cannot_break_another_writers_entry` |
| a cache entry published without the lock, or without a second look for a record under it; a failed writer removing the final tensors whatever it found; a staged file left behind | `test_a_writer_that_fails_cannot_break_another_writers_entry`, `test_an_entry_published_meanwhile_is_left_alone`, `test_two_processes_writing_one_key`, `test_a_write_that_fails_leaves_no_entry` |
| an identity that counts files under a hidden directory; the version of mlx-vlm or of torchvision left out of the request | `test_identity_reads_no_weights`, `test_a_changed_code_or_library_version_is_another_key` |
| a picture, a reference or the target not oriented by its EXIF tag; the size reported as stored; the script opening pictures itself | `test_a_picture_is_encoded_the_way_it_is_looked_at`, `test_a_reference_is_oriented_like_the_picture`, `test_what_turns_a_request_into_an_entry_is_in_the_package` |
| a run directory taken without its lock; temporary checkpoint directories cleared before the lock is held | `test_one_session_in_a_run_directory_at_a_time`, `test_a_second_process_is_refused_and_sigterm_stops_the_first_cleanly` |
| a new run that looks for a record only | `test_a_new_run_overwrites_nothing` |
| a resume that takes whatever checkpoint is on disk, refuses the record's own latest one, goes ahead past `--max-steps`, or is refused at it; the record rewritten before the resume is checked; a checkpoint's tensors loaded before its losses are counted; examples encoded before the run directory is checked | `test_a_resume_goes_on_from_the_records_latest_checkpoint_or_not_at_all`, `test_a_resume_past_its_max_steps_is_refused`, `test_max_steps_finishes_the_run_and_a_resume_can_extend_it`, `test_a_damaged_checkpoint_is_refused_not_skipped`, `test_a_refused_resume_encodes_nothing` |
| SIGTERM never looked at, reported as the wall clock, or its handler not put back | `test_sigterm_stops_at_the_next_boundary_with_a_checkpoint` |
| an unsigned 64-bit tensor without a size; an unknown dtype as a `KeyError` | `test_every_safetensors_dtype_has_a_size` |
| a published entry's tensors removed, or the whole entry rolled back, when the directory will not flush after the record's rename; that failure raised as a plain I/O error; no flush between the two renames, or one that fails without cleanup | `test_a_failure_after_the_records_rename_keeps_the_entry`, `test_a_write_that_fails_leaves_no_entry` |
| a checkpoint removed when its directory will not flush, or left out of the record's list though it stands | `test_a_failure_after_a_rename_removes_nothing` |
| the training attention left as the bf16 call, for bf16 or for float16; its result not cast back; its mask dropped | `test_training_attention_backward_is_torchs_at_every_logit_scale`, `test_training_attention_takes_a_mask` |
| the attention argument not passed on by `attn_block`, by `dit_block`, by the token refiner or by `dit_inputs`; the training forward's blocks, or its start, left on the sampling attention | `test_gradients_through_sharp_attention_over_a_bf16_base` |
| the sampling forward given the training attention, looked up when it runs or captured as a default | `test_the_sampling_forward_keeps_the_half_precision_call` |

Two changes to the training attention are not caught, because on this MLX they change nothing: widening q and k
and not v, or v and not q and k. The call promotes the rest to float32 itself.

One change in this list is caught by a narrow margin. The block-by-block gradient's forward walk evaluates each block
before the next; without that evaluation the gradients are the same and only the memory differs, on the default
device only. The memory test sees it as a growth of 6% to 17% of the plain gradient's against a bound of 5%, with
numbers that vary from run to run: five runs of the wrong version failed five times, and under an earlier, looser
bound of a quarter it had passed.

Where a row names a dtype, the change is caught only there. A tensor timestep's `1 − t` taken in float64 from the
value before its cast is one such: for a float32 sample it rounds to the same float32 number, because that
subtraction is a single correctly rounded operation either way, and only the bf16 cases fail, where the timestep
must be cast to bf16 before the subtraction.

## 14. Runs on the released weights, and what is not done

### 14.1 What the runs showed

**The runs.** Four LoRA runs and a few short shape tests on the released P-H3-Image weights (revision `7f98ace`), on
an M3 Ultra (60-core GPU, 256 GB) with MLX 0.32.4: bf16 transformer, VAE in float32, text encoder in bf16. One
subject, a man generated for the purpose, in 11 and then 9 pictures at 768 x 768, each caption naming him "ohwx man"
and describing the scene. Rank 16, alpha 16, on the six attention and feed-forward matrices of every block; learning
rate 1e-4; AdamW with betas 0.9 and 0.999; a clip at 1.0; the shift-12 grid of 1,000 with uniform density and no
weighting; the block-by-block gradient (§8.2).

| run | what it changed | steps | a step | MLX peak | gradient norm before the clip |
| --- | --- | --- | --- | --- | --- |
| 1 | (the attention's backward in bf16, below) | 300 | 6.3 s | 69.7 GB | up to 3.6e8 |
| 2 | the training forward's attention in float32 (§7.1) | 300 | 6.4 s | 70.0 GB | 1.21 at most |
| 3 | the public training adapter as a frozen assistant (§5.4) | 770, stopped | 6.4 s | 70.2 GB | 5.14 at most |
| 4 | the token refiner's blocks trained too; two pictures with tiny faces dropped; eps 1e-6, weight decay 1e-4 | 758, stopped | 6.9 s | 70.2 GB | 1.88 at most |
| shape test | 20 pictures at 1536 x 1536 | 3 | 22.5 s | 73.3 GB | |

Runs 3 and 4 were stopped by SIGTERM once their checkpoints had answered what they were for; each wrote its
checkpoint and exited 3 (§11.7). The block-by-block gradient ran at both sizes without meeting the GPU watchdog.

**The defect the first run found.** A plausible loss with gradients wrong by orders of magnitude: MLX 0.32.4's fused
backward for `mx.fast.scaled_dot_product_attention` keeps the unscaled scores in the inputs' dtype before rebuilding
the probabilities, so in bf16 a large score loses enough bits to put a large error into the exponent (§7.1).
0.32.3 and earlier give bounded gradients on the same inputs, and MLX's main branch at `fc344c98` (2026-11-17) gives
0.32.4's numbers. The training forward's attention runs in float32; sampling is unchanged.

**The assistant changes the fit on this base.** At the first step, on the same picture and noise level, the loss was
1.62 without the training adapter and 0.39 with it. In these runs the early checkpoints without it sampled soft and
smeared, as ai-toolkit's notes on this distilled model would lead one to expect, and those with it sampled sharp; that
is an observation, not an isolated experiment (the first were sampled at 1024 x 1024, the second at 2048 x 2048 and
after more steps).

**Likeness.** Each checkpoint was sampled with `run_p_h3_image.py --lora` at scale 1.0, 2048 x 2048, seed 0, on a
portrait prompt like the captions and two new scenes, and the face scored with a face-recognition model (InsightFace
`buffalo_l`, cosine) against the subject's portrait, which his ten other training pictures score 0.48 to 0.95 against (median 0.69).

| adapter | portrait | new scene 1 | new scene 2 |
| --- | --- | --- | --- |
| none | 0.15 | (no face) | 0.01 |
| run 2, step 300 (1024 x 1024) | 0.40 | 0.13 | 0.05 |
| run 3, step 500 | 0.73 | 0.35 | 0.46 |
| run 3, step 750 | 0.75 | 0.07 (turned away) | 0.45 |
| run 4, step 500 | 0.74 | 0.58 | 0.50 |
| run 4, step 750 | 0.82 | 0.57 | 0.50 |

By eye the adapters learned his bone structure and styling and kept the base model's smoothing of age and skin; small
asymmetric marks (a notch in one eyebrow, a mole) were not learned. At 768 his face covers about 10 by 15 tokens of 32
pixels, coarser than such detail. A run on higher-resolution pictures was not made. For comparison, the edit workflow
with one portrait of him as Picture 1 and no adapter scored 0.74 to 0.95 on the same kinds of scene.

### 14.2 Not done

- **Pictures as they come.** The script takes each example's size from the dataset file and refuses a picture of
  another shape. There is no bucketing by aspect ratio, no cropping, no captioning, no augmentation and no batch
  larger than one.
- **Samples while it trains.** The sampling script takes a trained adapter (§12), and a checkpoint's
  `adapter.safetensors` is one, so a run can be looked at by hand (§14.1 was made that way). The training script does
  not sample, and it computes no validation loss.
- **The recipe's settings, from the recipe.** ai-toolkit's settings for this model were read on its main branch at
  `7a5fe1a8` after the first runs: the training adapter on by default, rank 16 and alpha 16, timestep type `shift`, no
  guidance loss, and adapters on every `Linear` whose name holds `blocks`, which includes the token refiner's; its
  general job defaults are 3,000 steps, learning rate 1e-4, `adamw8bit` with eps 1e-6 and weight decay 1e-4, caption
  dropout 0.05, and resolution buckets of 512, 768 and 1024. The script still takes every setting as an argument and
  suggests none.
- **The trained file in diffusers.** The adapter is written in diffusers' format, which §1.5 tests in both directions
  on tiny models. A file trained by this script has not been loaded into diffusers on real weights.
- **Longer runs on a distilled base.** ai-toolkit warns that even with the assistant the distillation breaks down over
  a long run. These runs stopped at 770 steps or fewer; longer ones are untested.
- **Housekeeping for long runs.** Every checkpoint is kept (the plan prints their total size); there is no pruning.
  The whole dataset's entries are held in memory. One process trains on one device.
- **A smaller base in memory.** The gradients are tested over a quantized base (§7.4, §8.5), but the script has no
  option to quantize the weights it loads. It also sets none of MLX's own bounds, the memory limit and the buffer
  cache limit, which the sampling script does set. Its `--max-peak-memory-gb` reads a counter and stops the run; it
  bounds nothing.
- **Adapters anywhere but the block matrices.** `--targets` chooses among the six attention and feed-forward
  matrices of a block. The package can put an adapter on any `Linear` (§1), and the gradient for the ones a
  modulation table feeds is tested (§7.1), but the script does not offer them.
- **Gradient accumulation.** One example, one optimizer step. Several examples' gradients are not summed before a
  step, which is how a trainer with a batch of one usually gets the effect of a larger one.
- **Prompts as the model will be given them.** The sampling script can expand a short request with the release's
  prompt upsampler before encoding it; the training script encodes each prompt as written, and has no caption
  dropout.
- **The checkpointed gradient at real size.** The runs used the block-by-block gradient. The checkpointed one (§8.1)
  evaluates the whole backward pass at once and was not tried at real sequence lengths, where the GPU watchdog could
  stop it.
- **Stopping from outside.** SIGTERM is handled (§11.7). Ctrl-C is not: it interrupts where it lands and leaves the
  last periodic checkpoint. Neither lock here is to be relied on over a network filesystem (§10.2).

### 14.3 Open

- **Sums on the GPU have no fixed order.** A gradient over a bf16 base is repeatable on the CPU stream and can
  differ by a bf16 rounding step from run to run on the default device (§7.1). A resumed run is byte-for-byte an
  uninterrupted one in the tests, with adapters on attention and feed-forward matrices; with adapters on the modules
  a modulation table feeds it would be close, not identical.
- **Other sums in half precision.** The modulation tables' gather was the one place found where MLX's backward adds
  many rows in bf16 on the way to a trainable matrix. A trainable matrix placed somewhere else than a `Linear`, or a
  frozen adapter turned trainable in bf16, would need the same look.
- **The reference resize.** The recipe resizes references with an antialiased bicubic filter on floats; the port uses
  Lanczos on 8-bit pixels for both sizing rules (§10.3).
- **An encoder's identity does not read its weights** (§10.1): a weight file swapped for another of the same name and
  size is not noticed by the cache or by a resume.
- **The recipe's unread parts.** Its settings were read after the first runs (§14.2); how its trainer mixes the picture
  with noise was not, beyond §6.1. No preset is defined.
- **The assistant adapter's values.** Read and attached in runs 3 and 4 (§14.1); its header says ai-toolkit 0.13.40,
  3,250 steps. Its values are not otherwise checked.
- **bf16.** Gradients over a bf16 base are tested on four tiny layers against torch's own error (§7.3), and a run of
  optimizer steps over one only loosely (§9.1). A bf16 product path for the trainable matrices is not built.
- **Real files.** One published adapter's tensors were read by this code, the training adapter's, in runs 3 and 4;
  the tests write their own. Layout details of
  real files that the converter's docstring does not mention would not be caught.
- **Routing between partitions.** `h3mlx.lora` returns an adapter and the caller attaches it to the weights it means.
  diffusers' routing of a file to `transformer` or `transformer_ref` is not ported. The sampling script's `--lora`
  takes a file with one component, and refuses one that holds both.
- **One adapter at sampling.** `--lora` is given once. The package stacks named adapters (§5) and `attach` scales
  each; the sampling script does not offer it.