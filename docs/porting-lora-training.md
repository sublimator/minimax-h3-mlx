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

In each section the original comes first, then the port, then what changed and which test ties them together. It
assumes the main guide (`docs/porting.md`) for the transformer and the scheduler, and the HyperFlow guide
(`docs/porting-hyperflow.md`) for how a LoRA reaches the forward (`ops.plinear`).

**Nothing here trains a real model yet.** There is no training loop, no script and no run on real weights; §11 lists
what is missing. The pieces have been run together for twelve optimizer steps on a tiny random model (§9), and no
further. Everything below is weight-free.

**Sources, pinned.**

| what | where | pin |
| --- | --- | --- |
| LoRA layer, scaling, init, merge (Apache-2.0) | `huggingface/peft`, `src/peft/tuners/lora/` | tag `v0.21.2` (`eaa35c7`), the version the tests import |
| LoRA files, the original-layout converter, the scheduler, `training_utils`, the P-H3-Image pipeline's layout and row steps (Apache-2.0) | `huggingface/diffusers` main | `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1`, the commit the tests import |
| a training recipe for this model (MIT; read, not run, not quoted) | `ostris/ai-toolkit`, `extensions_built_in/diffusion_models/minimax_h3/` | `d6a18ad72da9abca3ffb67ea8e42c25764eee61a` |

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
| named adapters on one `Linear` | `src/h3mlx/lora.py`, `ops.plinear` | PEFT `layer.py`; diffusers `load_lora_adapter`, `set_adapters` | `tests/test_lora_stack.py` (12, one of them on a release fixture) |
| one training example | `train.build_example` | diffusers `p_h3_image/before_denoise.py`, `scheduling_minimax_h3.py`; the recipe | `tests/test_train_example.py` (33) |
| the example's loss and gradient | `train.example_loss`, `train.loss_and_gradients` | torch autograd on diffusers' transformer with PEFT's layers | `tests/test_train_gradients.py` (31) |
| a bounded backward pass | `src/h3mlx/backward.py` | the plain gradient of the row above | `tests/test_backward.py` (37) |
| the optimizer step | `train.adamw`, `train.optimizer_step` | `torch.optim.AdamW`, `clip_grad_norm_` | `tests/test_train_optimizer.py` (13) |

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

📍 [`src/h3mlx/ops.py:100-105`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/ops.py#L100-L105)
```python
 100 def times_scalar(x: mx.array, s) -> mx.array:
 101     """torch's `tensor * python_float`. A half-precision tensor is multiplied in float32 by the float32 scalar and
 102     rounded once; MLX alone would first round the scalar to the tensor's dtype (0.7 becomes 0.69921875 in bf16)."""
 103     if x.dtype in (mx.float16, mx.bfloat16):
 104         return (x.astype(mx.float32) * s).astype(x.dtype)
 105     return x * s
```
📍 [`src/h3mlx/ops.py:108-146`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/ops.py#L108-L146)
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

📍 [`src/h3mlx/lora.py:138-148`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L138-L148)
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
📍 [`src/h3mlx/lora.py:151-173`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L151-L173)
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

📍 [`src/h3mlx/lora.py:38-63`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L38-L63)
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
📍 [`src/h3mlx/lora.py:69-82`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L69-L82)
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
📍 [`src/h3mlx/lora.py:114-132`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L114-L132)
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
📍 [`src/h3mlx/lora.py:198-240`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L198-L240)
```python
 198 def attach(params: dict, cfg: dict, adapter: Adapter, keep_dtype: bool = False, name: str | None = None) -> dict:
 199     """The transformer's weights with the adapter beside them, for `ops.plinear`; `params` is not modified.
 200
 201     By default each matrix is cast to its base weight's dtype, as PEFT's injection does (`load_lora_adapter`, and
 202     `hyperflow.attach`). `keep_dtype=True` leaves the adapter as it is: float32 factors over a bf16 base, what
 203     `cast_training_params` gives a trainer (and what diffusers' `low_cpu_mem_usage=True` loading gives a float32 file).
 204     A module whose scaling is not 1 also gets a `lora_scaling` entry.
 205
 206     Without a `name` the adapter is the only one its Linears can carry. With one it is stacked: its matrices go under
 207     PEFT's runtime names, and the name is appended to each Linear's `lora_adapters`, so adapters apply in the order
 208     they were attached (`set_active` changes that). A named and an unnamed adapter can't share a Linear, and a name is
 209     used once: diffusers' `load_lora_adapter` refuses a name already in use too. That includes a name these weights
 210     only list or carry a scaling for, which is what the frozen half of `split` looks like (`join` is the way back
 211     from it); attaching onto that half would name the adapter twice on each Linear and apply it twice."""
 212     check(adapter, cfg)
 213     if name is None:
 214         taken = [m for m in adapter.modules if _carries_lora(params, m)]
 215         if taken:
 216             raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: one unnamed adapter per "
 217                              "Linear; give each a name to stack them")
 218     else:
 219         if not name or "." in name:
 220             raise ValueError(f"an adapter name is a non-empty word without dots, got {name!r}")
 221         taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
 222         if taken:
 223             raise ValueError(f"{len(taken)} Linear(s) already carry an unnamed LoRA, e.g. {taken[:3]}")
 224         used = _named_parts(params, name)
 225         if used:
 226             raise ValueError(f"the adapter name {name!r} is already in use in these weights ({len(used)} keys, e.g. "
 227                              f"{used[:2]}). If they are the frozen half of `split`, `join` it with its trainable half; "
 228                              "a changed adapter goes in as that trainable half, not through a second `attach`")
 229     out = dict(params)
 230     for module in adapter.modules:
 231         dtype = params[module + ".weight"].dtype
 232         for matrix in ("A", "B"):
 233             tensor = adapter.tensors[f"{module}.lora_{matrix}.weight"]
 234             out[runtime_key(module, matrix, name)] = tensor if keep_dtype else tensor.astype(dtype)
 235         scaling = adapter.scaling(module)
 236         if scaling != 1.0:
 237             out[module + SCALING + ("" if name is None else "." + name)] = mx.array(scaling, dtype=mx.float32)
 238         if name is not None:
 239             out[module + ADAPTERS] = params.get(module + ADAPTERS, ()) + (name,)
 240     return out
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

📍 [`src/h3mlx/lora.py:91-100`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L91-L100)
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
📍 [`src/h3mlx/lora.py:103-111`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L103-L111)
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

📍 [`src/h3mlx/lora.py:330-338`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L330-L338)
```python
 330 def delta_weight(a: mx.array, b: mx.array, scaling: float) -> mx.array:
 331     """PEFT's `Linear.get_delta_weight` on CPU: `(B @ A) * scaling`, with half-precision factors multiplied in float32
 332     and the product cast back to their dtype."""
 333     dtype = b.dtype
 334     half = dtype in (mx.float16, mx.bfloat16)
 335     if half:
 336         a, b = a.astype(mx.float32), b.astype(mx.float32)
 337     out = (b @ a) * scaling
 338     return out.astype(dtype) if half else out
```
📍 [`src/h3mlx/lora.py:341-362`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L341-L362)
```python
 341 def merge(params: dict, cfg: dict, adapter: Adapter) -> dict:
 342     """The transformer's plain weights with the adapter added into them (PEFT's `Linear.merge`: `weight +=
 343     delta_weight`, in the weight's dtype); `params` is not modified, and the result has its keys and nothing of the
 344     adapter left to apply.
 345
 346     Attaching and merging are alternatives here. Weights that already carry a LoRA, named or not, on one of the
 347     adapter's Linears are refused: the attached factors would stay in the result, and `plinear` would apply them on top
 348     of the merged ones. (A PEFT layer holds both states itself, and its forward skips an adapter once it is merged.)
 349     Not for quantized weights."""
 350     check(adapter, cfg)
 351     taken = [m for m in adapter.modules if _carries_lora(params, m)]
 352     if taken:
 353         raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: merge into the plain "
 354                          "weights, not into attached ones (the adapter would be applied twice)")
 355     out = dict(params)
 356     for module in adapter.modules:
 357         w = params[module + ".weight"]
 358         if isinstance(w, ops.QuantizedWeight):
 359             raise ValueError(f"{module}: can't merge a LoRA into a quantized weight; merge first, then quantize")
 360         a, b = adapter.tensors[module + ".lora_A.weight"], adapter.tensors[module + ".lora_B.weight"]
 361         out[module + ".weight"] = (w + delta_weight(a, b, adapter.scaling(module))).astype(w.dtype)
 362     return out
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

📍 [`src/h3mlx/lora.py:533-539`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L533-L539)
```python
 533 def pattern_key(pattern_keys, key_to_match: str) -> str:
 534     """PEFT's `get_pattern_key`: the first `rank_pattern` / `alpha_pattern` key that matches the module name as a
 535     regular expression on its last dotted parts, else the name itself."""
 536     for key in pattern_keys:
 537         if re.match(rf"(.*\.)?({key})$", key_to_match):
 538             return key
 539     return key_to_match
```
📍 [`src/h3mlx/lora.py:542-557`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L542-L557)
```python
 542 def _component(tensors: dict, prefix: str | None) -> tuple[dict, str | None]:
 543     """The keys of one component with its prefix removed (`load_lora_adapter(prefix=...)`'s filter), and that prefix.
 544     With no `prefix` given the file must hold one component, prefixed or not."""
 545     if prefix == "":
 546         raise ValueError("an empty prefix names no component: pass None for a file without prefixes")
 547     found = [p for p in PREFIXES if any(k.startswith(p + ".") for k in tensors)]
 548     if prefix is None:
 549         if len(found) > 1:
 550             raise ValueError(f"this file holds LoRAs for {found}: say which with `prefix=`")
 551         prefix = found[0] if found else None
 552         other = () if prefix is None else [k for k in tensors if not k.startswith(prefix + ".")]
 553         if other:
 554             raise ValueError(f"keys with and without the `{prefix}.` prefix in one file, e.g. {sorted(other)[:3]}")
 555     if prefix is None:
 556         return dict(tensors), None
 557     return {k.removeprefix(prefix + "."): v for k, v in tensors.items() if k.startswith(prefix + ".")}, prefix
```
📍 [`src/h3mlx/lora.py:560-564`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L560-L564)
```python
 560 def _refuse_dora(keys) -> None:
 561     """DoRA is not implemented: refuse its keys (diffusers drops a kohya `dora_scale` with a warning, and loads PEFT's
 562     `lora_magnitude_vector` as DoRA)."""
 563     if any("dora_scale" in k or "lora_magnitude_vector" in k for k in keys):
 564         raise NotImplementedError("DoRA adapters are not supported")
```
📍 [`src/h3mlx/lora.py:567-640`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L567-L640)
```python
 567 def read(tensors: dict, header: dict | None = None, prefix: str | None = None, head_dim: int = HEAD_DIM) -> Adapter:
 568     """An adapter from a LoRA file's tensors and its safetensors header (`__metadata__`), as diffusers would load it.
 569
 570     Keys: `transformer.<module>.lora_A/B.weight` (what `save_lora_weights` writes and `load_lora_adapter` reads by
 571     default), the same without the prefix (what `save_lora_adapter` writes), PEFT's own state dict (a `.default.`
 572     infix), or the original checkpoint's names (`convert_original`, which folds any `.alpha` tensors into the matrices;
 573     `head_dim` is passed to it).
 574
 575     `prefix` picks one component's keys from a file that holds several, as `load_lora_adapter(prefix=...)` does; other
 576     components' keys are then not looked at. Without it the file must hold one component. An original-layout file
 577     names no component and is read whole whatever `prefix` says: the converter labels it `transformer.`, and the
 578     mixin loads those layers into either partition (`load_into_transformer_ref`). The adapter itself belongs to no
 579     partition; the caller attaches it to the weights it means.
 580
 581     Alpha, per module: from the `lora_adapter_metadata` diffusers writes (`r`, `lora_alpha`, `rank_pattern`,
 582     `alpha_pattern`, resolved as PEFT resolves them); else from a plain `alpha` entry in the header, taken as one
 583     network alpha for every module; else the module's own rank (scaling 1), which is `get_peft_kwargs`' rule. DoRA,
 584     rsLoRA, a LoRA bias, a `bias` setting other than none, `fan_in_fan_out` and an alpha that is not positive and
 585     finite are refused, not ignored."""
 586     header = dict(header or {})
 587     tensors = dict(tensors)
 588     has_alpha_tensors = any(k.endswith(".alpha") for k in tensors)
 589     if is_original_layout(tensors):
 590         _refuse_dora(tensors)  # before the converter, which would call a `dora_scale` key a leftover
 591         tensors = convert_original(tensors, head_dim)
 592         prefix = "transformer"  # the converter's label, not a choice of partition
 593     elif any(".default.weight" in k for k in tensors) and not any(k.startswith(tuple(p + "." for p in PREFIXES)) for k in tensors):
 594         tensors = {k.replace(".default.", "."): v for k, v in tensors.items()}  # a PEFT state dict, adapter "default"
 595     tensors, prefix = _component(tensors, prefix)
 596     _refuse_dora(tensors)  # of the component read, not of the others
 597
 598     by_module = {}
 599     for key in tensors:
 600         m = KEY_RE.match(key)
 601         if m is None:
 602             raise ValueError(f"unexpected key {key!r}: only <module>.lora_A/B.weight keys are supported")
 603         by_module.setdefault(m["module"], set()).add(m["matrix"])
 604     incomplete = sorted(k for k, v in by_module.items() if v != {"A", "B"})
 605     if incomplete:
 606         raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
 607     if not by_module:
 608         raise ValueError("no LoRA tensors" + (f" under the `{prefix}.` prefix" if prefix else "") + " in this file")
 609     ranks = {m: int(tensors[m + ".lora_B.weight"].shape[1]) for m in by_module}
 610
 611     config = json.loads(header[METADATA_KEY]) if header.get(METADATA_KEY) else None
 612     if config is not None and prefix is not None:
 613         config = {k.removeprefix(prefix + "."): v for k, v in config.items() if k.startswith(prefix + ".")}
 614     if config is not None:
 615         if "r" not in config or "lora_alpha" not in config:
 616             raise ValueError(f"the file's {METADATA_KEY} has no `r` / `lora_alpha`" + (f" under `{prefix}.`" if prefix else ""))
 617         on = [k for k in UNSUPPORTED if config.get(k)]
 618         if config.get("bias") not in (None, "none"):  # PEFT then trains and saves base-layer biases as well
 619             on.append(f"bias={config['bias']!r}")
 620         if on:
 621             raise NotImplementedError(f"this adapter was saved with {on}, which the port does not implement")
 622         rank_pattern, alpha_pattern = config.get("rank_pattern") or {}, config.get("alpha_pattern") or {}
 623         alpha = {}
 624         for module, rank in ranks.items():
 625             r = rank_pattern.get(pattern_key(rank_pattern, module), config["r"])
 626             if r != rank:
 627                 raise ValueError(f"{module}: the metadata says rank {r} but lora_B has rank {rank}")
 628             alpha[module] = alpha_pattern.get(pattern_key(alpha_pattern, module), config["lora_alpha"])
 629             check_alpha(f"{module} (the file's {METADATA_KEY})", alpha[module])
 630         return Adapter(tensors, alpha)
 631
 632     network_alpha = None
 633     if "alpha" in header and not has_alpha_tensors:
 634         try:
 635             network_alpha = float(header["alpha"])
 636         except ValueError:  # diffusers warns and loads with alpha == rank
 637             network_alpha = None
 638         else:
 639             check_alpha("the file header's `alpha`", network_alpha)
 640     return Adapter(tensors, {m: (rank if network_alpha is None else network_alpha) for m, rank in ranks.items()})
```
📍 [`src/h3mlx/lora.py:653-674`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L653-L674)
```python
 653 def config(adapter: Adapter) -> dict:
 654     """The `LoraConfig` arguments that describe the adapter, as `get_peft_kwargs` derives them: the most common rank and
 655     alpha, and a `^`-anchored pattern entry for each module that differs. The other entries state what the port
 656     implements (no dropout, DoRA, rsLoRA or LoRA bias). A subset of what `LoraConfig.to_dict()` holds, so it loads
 657     with `LoraConfig(**config)` on PEFT versions that add or drop other fields."""
 658     ranks = {m: adapter.rank(m) for m in adapter.modules}
 659     alphas = {m: _number(adapter.alpha[m]) for m in adapter.modules}
 660     r = Counter(ranks.values()).most_common()[0][0]
 661     lora_alpha = Counter(alphas.values()).most_common()[0][0]
 662     return {
 663         "r": r,
 664         "lora_alpha": lora_alpha,
 665         "rank_pattern": {f"^{m}": v for m, v in ranks.items() if v != r},
 666         "alpha_pattern": {f"^{m}": v for m, v in alphas.items() if v != lora_alpha},
 667         "target_modules": adapter.modules,
 668         "bias": "none",
 669         "lora_dropout": 0.0,
 670         "use_dora": False,
 671         "use_rslora": False,
 672         "lora_bias": False,
 673         "fan_in_fan_out": False,
 674     }
```
📍 [`src/h3mlx/lora.py:677-695`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L677-L695)
```python
 677 def save(path, adapter: Adapter, prefix: str | None = "transformer") -> None:
 678     """Write the adapter as diffusers does: `<prefix>.<module>.lora_A/B.weight` tensors and, in the safetensors header,
 679     `format: pt` plus the `LoraConfig` arguments as JSON under `lora_adapter_metadata` (each key prefixed too). That is
 680     `save_lora_weights`' file, which `load_lora_adapter` reads by default. `prefix=None` writes `save_lora_adapter`'s:
 681     no prefixes, read back with `load_lora_adapter(..., prefix=None)`. Tensors keep their dtype. Nothing is written
 682     for an adapter whose tensors do not match its modules, or with an alpha that is not positive and finite."""
 683     path = Path(path)
 684     if path.suffix != ".safetensors":
 685         raise ValueError(f"expected a .safetensors path, got {path}")
 686     if prefix == "":
 687         raise ValueError("an empty prefix would put a leading dot on every key: pass None for no prefix")
 688     check_keys(adapter)
 689     for module in adapter.modules:
 690         check_alpha(module, adapter.alpha[module])
 691     lead = "" if prefix is None else prefix + "."
 692     tensors = {lead + k: v for k, v in adapter.tensors.items()}
 693     meta = {lead + k: v for k, v in config(adapter).items()}
 694     described = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False)  # strict JSON: no NaN or Infinity tokens
 695     mx.save_safetensors(str(path), tensors, metadata={"format": "pt", METADATA_KEY: described})
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

📍 [`src/h3mlx/lora.py:367-378`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L367-L378)
```python
 367 _FLATTENED = [  # musubi-tuner's `lora_unet_` names, every `.` collapsed to `_`
 368     (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
 369     (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
 370     (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
 371     (r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj", r"token_refiner.blocks.\1.attn.\2_proj"),
 372     (r"token_refiner_blocks_(\d+)_mlp_fc([12])", r"token_refiner.blocks.\1.mlp.fc\2"),
 373     (r"(video|audio)_patch_proj", r"\1_patch_proj"),
 374     (r"condition_proj", "condition_proj"),
 375     (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
 376     (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
 377     (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
 378 ]
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

📍 [`src/h3mlx/lora.py:455-464`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L455-L464)
```python
 455         alpha = sd.pop(base + ".alpha", None)
 456         if alpha is not None:
 457             if not alpha.item() > 0 or math.isinf(alpha.item()):
 458                 raise ValueError(f"`{base}.alpha` is {alpha.item()}: a LoRA alpha must be positive and finite")
 459             # the forward scales by alpha / rank; split that factor between down and up
 460             scale_down, scale_up = alpha.item() / down.shape[0], 1.0
 461             while scale_down * 2 < scale_up:
 462                 scale_down *= 2
 463                 scale_up /= 2
 464             down, up = ops.times_scalar(down, scale_down), ops.times_scalar(up, scale_up)
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

📍 [`src/h3mlx/lora.py:485-503`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L485-L503)
```python
 485             qkv = pull(f"{source}.attn.qkv_proj")
 486             if qkv is not None:
 487                 down, up = qkv
 488                 if up.shape[0] % 3 != 0:
 489                     raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not divisible "
 490                                      "by 3. This is not a fused MiniMax-H3 QKV projection.")
 491                 if is_diffsynth:
 492                     # [head0: q k v, head1: q k v, ...] -> [q_all; k_all; v_all]
 493                     if up.shape[0] % (3 * head_dim) != 0:
 494                         raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not a "
 495                                          f"multiple of 3 * {head_dim}. This is not a per-head-interleaved MiniMax-H3 "
 496                                          "QKV projection.")
 497                     num_heads = up.shape[0] // (3 * head_dim)
 498                     grouped = up.reshape(num_heads, 3 * head_dim, up.shape[1])
 499                     up = mx.concatenate([s.reshape(num_heads * head_dim, up.shape[1])
 500                                          for s in mx.split(grouped, 3, axis=1)], axis=0)
 501                 for proj, up_proj in zip(("to_q", "to_k", "to_v"), mx.split(up, 3, axis=0)):
 502                     out[f"{target}.attn.{proj}.lora_A.weight"] = down
 503                     out[f"{target}.attn.{proj}.lora_B.weight"] = up_proj
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

📍 [`src/h3mlx/lora.py:507-515`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L507-L515)
```python
 507             fc1 = pull(f"{source}.mlp.fc1")
 508             if fc1 is not None:
 509                 down, up = fc1
 510                 if up.shape[0] % 2 != 0:
 511                     raise ValueError(f"`{source}.mlp.fc1` has {up.shape[0]} output rows, which is not even. This is "
 512                                      "not a fused MiniMax-H3 SwiGLU projection.")
 513                 up_gate, up_value = mx.split(up, 2, axis=0)
 514                 out[f"{target}.ff.net.0.proj.lora_A.weight"] = down
 515                 out[f"{target}.ff.net.0.proj.lora_B.weight"] = mx.concatenate([up_value, up_gate], axis=0)
```
📍 [`src/h3mlx/lora.py:379-388`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L379-L388)
```python
 379 _STANDALONE = {  # the projections outside the block stack
 380     "video_patch_proj": "proj_in",
 381     "audio_patch_proj": "audio_proj_in",
 382     "condition_proj": "context_embedder",
 383     "time_embedder.proj_in": "time_embedder.linear_1",
 384     "time_embedder.proj_out": "time_embedder.linear_2",
 385     "final_layer.adaln_proj.linear": "norm_out.linear",
 386     "final_layer.video_out": "proj_out",
 387     "final_layer.audio_out": "audio_proj_out",
 388 }
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

📍 [`src/h3mlx/train.py:30-54`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L30-L54)
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
📍 [`src/h3mlx/ops.py:385-393`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/ops.py#L385-L393)
```python
 385 def flow_step(v: mx.array, t: float, x: mx.array, sigma: float, sigma_next: float) -> mx.array:
 386     """x0 = x + (1 - t) * v (sigma from the timestep, in x's dtype); x_next = r*x + (1 - r)*x0, r = s_next / s, f32."""
 387     st = (mx.array(1.0, dtype=x.dtype) - mx.array(t, dtype=x.dtype)).astype(x.dtype)
 388     x0 = x + st * v
 389     cdt = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
 390     s = mx.array(sigma, dtype=cdt)
 391     sn = mx.array(sigma_next, dtype=cdt)
 392     r = sn / s
 393     return (r * x.astype(cdt) + (1.0 - r) * x0.astype(cdt)).astype(x.dtype)
```

With `x_t = t·x_0 + (1 − t)·noise` and `x_0 = x_t + (1 − t)·v`, the model's output must satisfy
`(1 − t)·v = (1 − t)·(x_0 − noise)`. So the target is `x_0 − noise` at every `t`.

📍 [`src/h3mlx/train.py:57-61`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L57-L61)
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

📍 [`src/h3mlx/train.py:64-87`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L64-L87)
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

📍 [`src/h3mlx/train.py:358-384`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L358-L384)
```python
 358 def density_for_timestep_sampling(weighting_scheme: str, batch_size: int, logit_mean: float | None = None,
 359                                   logit_std: float | None = None, mode_scale: float | None = None,
 360                                   generator=None) -> np.ndarray:
 361     """diffusers' `compute_density_for_timestep_sampling`: `batch_size` values `u` in [0, 1], float32.
 362
 363     * `"logit_normal"`: the sigmoid of a normal draw (`logit_mean`, `logit_std`).
 364     * `"mode"`: `1 - u - mode_scale * (cos(pi * u / 2) ** 2 - 1 + u)` of a uniform draw.
 365     * anything else: a uniform draw.
 366
 367     The draw is torch's, from `generator` (a `torch.Generator`; None is torch's global one), so it is the same stream
 368     as theirs. The arithmetic after it is numpy float32 in their order: equal to theirs bit for bit except for the
 369     sigmoid's `exp` and the `cos`, which numpy and torch round differently in the last place or two.
 370
 371     A scheme's own parameters are required, as in theirs (TypeError), and must be finite, which theirs does not check
 372     (ValueError here). A negative or NaN `logit_std` is torch's own RuntimeError on both sides."""
 373     import torch
 374
 375     if weighting_scheme == "logit_normal":
 376         _density_parameters(weighting_scheme, logit_mean=logit_mean, logit_std=logit_std)
 377         u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator).numpy()
 378         return (np.float32(1) / (np.float32(1) + np.exp(-u))).astype(np.float32)
 379     if weighting_scheme == "mode":
 380         _density_parameters(weighting_scheme, mode_scale=mode_scale)
 381         u = torch.rand(size=(batch_size,), generator=generator).numpy()
 382         c = np.cos(np.float32(math.pi) * u / np.float32(2))
 383         return (np.float32(1) - u - np.float32(mode_scale) * (c * c - np.float32(1) + u)).astype(np.float32)
 384     return torch.rand(size=(batch_size,), generator=generator).numpy()
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

📍 [`src/h3mlx/train.py:387-411`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L387-L411)
```python
 387 def loss_weighting_for_sd3(weighting_scheme: str, sigmas) -> np.ndarray:
 388     """diffusers' `compute_loss_weighting_for_sd3`, on float32 or float64 noise levels (numpy, or an `mx.array`).
 389
 390     * `"sigma_sqrt"`: `sigma ** -2`, which torch computes as `1 / (sigma * sigma)`; float32 (infinite at sigma 0).
 391     * `"cosmap"`: `2 / (pi * (1 - 2 * sigma + 2 * sigma ** 2))`, in the input's dtype.
 392     * anything else: ones.
 393
 394     `sigma` is the noise level in both conventions (1 is pure noise), so nothing changes for this model's `1 - t`.
 395
 396     Half-precision and integer levels are refused. Theirs computes a bf16 tensor's weighting in bf16, three digits of
 397     it; the numpy result here can't hold bf16, and upcasting without saying so would be neither theirs nor exact."""
 398     if isinstance(sigmas, mx.array):
 399         if sigmas.dtype not in (mx.float32, mx.float64):
 400             raise ValueError(f"sigmas must be float32 or float64, got {sigmas.dtype}: cast them first")
 401         sigmas = np.array(sigmas)
 402     s = np.asarray(sigmas)
 403     if s.dtype not in (np.float32, np.float64):
 404         raise ValueError(f"sigmas must be float32 or float64, got {s.dtype}: cast them first")
 405     one, two = s.dtype.type(1), s.dtype.type(2)
 406     if weighting_scheme == "sigma_sqrt":
 407         return (one / (s * s)).astype(np.float32)
 408     if weighting_scheme == "cosmap":
 409         bot = one - two * s + two * (s * s)
 410         return (two / (s.dtype.type(math.pi) * bot)).astype(s.dtype)
 411     return np.ones_like(s)
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
📍 [`src/h3mlx/ops.py:372-382`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/ops.py#L372-L382)
```python
 372 def flow_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
 373     """linspace(1, 0, n) in float32, exponential shift, consecutive duplicates collapsed. Terminal 0 included."""
 374     if num_inference_steps is None or num_inference_steps < 2:
 375         raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
 376     if not usable_shift(shift):
 377         raise ValueError(f"shift must be positive and finite, got {shift}")
 378     base = torch_linspace_f32(1.0, 0.0, num_inference_steps)
 379     shift32 = np.float32(shift)
 380     sig = (shift32 * base / (np.float32(1) + (shift32 - np.float32(1)) * base)).astype(np.float32)
 381     keep = np.concatenate([[True], sig[1:] != sig[:-1]])
 382     return sig[keep]
```

📍 [`src/h3mlx/train.py:414-423`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L414-L423)
```python
 414 def training_sigmas(num_points: int, shift: float) -> np.ndarray:
 415     """`num_points` noise levels to train at, from 1.0 (pure noise) down, float32: the scheduler's own shifted grid
 416     for `num_points + 1` points (`ops.flow_sigmas`: `linspace(1, 0, n)` through `s * sigma / (1 + (s - 1) * sigma)`)
 417     without its last point, sigma 0, where the sample is clean and there is nothing to predict. Both numbers are the
 418     caller's: this module has no default shift or grid size."""
 419     grid = ops.flow_sigmas(num_points + 1, shift)[:-1]
 420     if len(grid) != num_points:
 421         raise ValueError(f"shift {shift} merges neighbouring float32 levels of a {num_points}-point grid "
 422                          f"({len(grid)} distinct): ask for fewer points")
 423     return grid
```
📍 [`src/h3mlx/train.py:426-445`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L426-L445)
```python
 426 def sample_sigmas(u, grid) -> tuple[np.ndarray, np.ndarray]:
 427     """One noise level per `u`: `grid[floor(u * len(grid))]`, the multiplication in float32 (an SD3-style trainer's
 428     `(u * num_train_timesteps).long()` lookup). Returns (sigmas, indices). `u = 1.0` would index one past the end,
 429     which torch refuses; it is given the last level here (the logit-normal sigmoid rounds to 1.0 for a large draw, and
 430     `"mode"` maps a draw of 0 to 1.0). The model's timestep for a level is `1 - sigma` in float32; `scale_noise` takes
 431     those as `mx.array(1 - sigmas)`.
 432
 433     Every `u` must lie in [0, 1], which a NaN does not: cast to an index a NaN is 0 on this hardware, pure noise for
 434     every such draw, here and in the torch lookup. The grid must be one non-empty row of finite levels."""
 435     grid = np.asarray(grid)
 436     if grid.ndim != 1 or grid.size == 0:
 437         raise ValueError(f"the grid must be one non-empty row of noise levels, got shape {grid.shape}")
 438     if not np.isfinite(grid).all():
 439         raise ValueError("the grid holds levels that are not finite")
 440     u = np.asarray(u, dtype=np.float32)
 441     inside = (u >= 0) & (u <= 1)
 442     if not inside.all():
 443         raise ValueError(f"every u must lie in [0, 1] (a NaN does not): got {u[~inside][:4].tolist()}")
 444     indices = np.minimum((u * np.float32(len(grid))).astype(np.int64), len(grid) - 1)
 445     return grid[indices], indices
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

📍 [`src/h3mlx/lora.py:179-182`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L179-L182)
```python
 179 def runtime_key(module: str, matrix: str, name: str | None = None) -> str:
 180     """A matrix's key in the weights dict: `<module>.lora_A.weight`, or for a named adapter PEFT's runtime name,
 181     `<module>.lora_A.<name>.weight`. (In a file neither carries a name: `Adapter.tensors`.)"""
 182     return f"{module}.lora_{matrix}.weight" if name is None else f"{module}.lora_{matrix}.{name}.weight"
```
📍 [`src/h3mlx/lora.py:313-315`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L313-L315)
```python
 313 def runtime_tensors(adapter: Adapter, name: str) -> dict:
 314     """An adapter's tensors under their runtime names for `name` (what `split` returns as `trainable`)."""
 315     return {runtime_key(m, x, name): adapter.tensors[f"{m}.lora_{x}.weight"] for m in adapter.modules for x in "AB"}
```
📍 [`src/h3mlx/lora.py:318-327`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L318-L327)
```python
 318 def file_tensors(tensors: dict, name: str) -> dict:
 319     """The reverse: a named adapter's matrices back under the names they have in a file and in `Adapter.tensors`,
 320     `<module>.lora_A/B.weight`, as PEFT strips the adapter's name when it saves."""
 321     out = {}
 322     for key, value in tensors.items():
 323         m = RUNTIME_RE.match(key)
 324         if m is None or m["name"] != name:
 325             raise ValueError(f"{key!r} is not a matrix of the adapter {name!r}")
 326         out[f"{m['module']}.lora_{m['matrix']}.weight"] = value
 327     return out
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

📍 [`src/h3mlx/lora.py:198-240`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L198-L240)
```python
 198 def attach(params: dict, cfg: dict, adapter: Adapter, keep_dtype: bool = False, name: str | None = None) -> dict:
 199     """The transformer's weights with the adapter beside them, for `ops.plinear`; `params` is not modified.
 200
 201     By default each matrix is cast to its base weight's dtype, as PEFT's injection does (`load_lora_adapter`, and
 202     `hyperflow.attach`). `keep_dtype=True` leaves the adapter as it is: float32 factors over a bf16 base, what
 203     `cast_training_params` gives a trainer (and what diffusers' `low_cpu_mem_usage=True` loading gives a float32 file).
 204     A module whose scaling is not 1 also gets a `lora_scaling` entry.
 205
 206     Without a `name` the adapter is the only one its Linears can carry. With one it is stacked: its matrices go under
 207     PEFT's runtime names, and the name is appended to each Linear's `lora_adapters`, so adapters apply in the order
 208     they were attached (`set_active` changes that). A named and an unnamed adapter can't share a Linear, and a name is
 209     used once: diffusers' `load_lora_adapter` refuses a name already in use too. That includes a name these weights
 210     only list or carry a scaling for, which is what the frozen half of `split` looks like (`join` is the way back
 211     from it); attaching onto that half would name the adapter twice on each Linear and apply it twice."""
 212     check(adapter, cfg)
 213     if name is None:
 214         taken = [m for m in adapter.modules if _carries_lora(params, m)]
 215         if taken:
 216             raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: one unnamed adapter per "
 217                              "Linear; give each a name to stack them")
 218     else:
 219         if not name or "." in name:
 220             raise ValueError(f"an adapter name is a non-empty word without dots, got {name!r}")
 221         taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
 222         if taken:
 223             raise ValueError(f"{len(taken)} Linear(s) already carry an unnamed LoRA, e.g. {taken[:3]}")
 224         used = _named_parts(params, name)
 225         if used:
 226             raise ValueError(f"the adapter name {name!r} is already in use in these weights ({len(used)} keys, e.g. "
 227                              f"{used[:2]}). If they are the frozen half of `split`, `join` it with its trainable half; "
 228                              "a changed adapter goes in as that trainable half, not through a second `attach`")
 229     out = dict(params)
 230     for module in adapter.modules:
 231         dtype = params[module + ".weight"].dtype
 232         for matrix in ("A", "B"):
 233             tensor = adapter.tensors[f"{module}.lora_{matrix}.weight"]
 234             out[runtime_key(module, matrix, name)] = tensor if keep_dtype else tensor.astype(dtype)
 235         scaling = adapter.scaling(module)
 236         if scaling != 1.0:
 237             out[module + SCALING + ("" if name is None else "." + name)] = mx.array(scaling, dtype=mx.float32)
 238         if name is not None:
 239             out[module + ADAPTERS] = params.get(module + ADAPTERS, ()) + (name,)
 240     return out
```
📍 [`src/h3mlx/lora.py:248-261`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L248-L261)
```python
 248 def set_active(params: dict, names) -> dict:
 249     """The weights with the adapters `names` active, in that order, and every other named adapter switched off: PEFT's
 250     `set_adapters`. A Linear applies those of `names` it carries. The matrices of an adapter that is off stay in the
 251     dict, unused, so it can be switched back on; `params` is not modified."""
 252     names = tuple(names)
 253     unknown = sorted(set(names) - set(adapter_names(params)))
 254     if unknown or len(set(names)) != len(names):
 255         raise ValueError(f"adapters {unknown or list(names)} are not attached, or are named twice; attached: {adapter_names(params)}")
 256     out = dict(params)
 257     for key in params:
 258         if key.endswith(ADAPTERS):
 259             module = key[: -len(ADAPTERS)]
 260             out[key] = tuple(n for n in names if runtime_key(module, "A", n) in params)
 261     return out
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

📍 [`src/h3mlx/lora.py:264-274`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L264-L274)
```python
 264 def split(params: dict, name: str) -> tuple[dict, dict]:
 265     """(frozen, trainable): the weights without the two matrices of the adapter `name`, and those matrices alone under
 266     their runtime names. The two are separate groups: a trainer differentiates with respect to `trainable` only, and
 267     merges them (`{**frozen, **trainable}`) inside the function it differentiates, so the base weights and every other
 268     adapter are constants. The adapter's scalings and its place in each Linear's `lora_adapters` stay in `frozen`,
 269     which is therefore not a model on its own: the forward refuses it, and so does `attach` for the same name. `join`
 270     puts the two back together."""
 271     trainable = {k: v for k, v in params.items() if (m := RUNTIME_RE.match(k)) and m["name"] == name}
 272     if not trainable:
 273         raise ValueError(f"no adapter named {name!r} in these weights; attached: {adapter_names(params)}")
 274     return {k: v for k, v in params.items() if k not in trainable}, trainable
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

📍 [`src/h3mlx/lora.py:305-310`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L305-L310)
```python
 305 def join(frozen: dict, trainable: dict) -> dict:
 306     """The reverse of `split`: the two groups as one dict of weights again, for a forward outside the loss (a
 307     validation sample), for `set_active`, or for `file_tensors` and `save` after training. `trainable` is checked as
 308     a loss checks it (`check_trainable`)."""
 309     check_trainable(trainable, frozen)
 310     return {**frozen, **trainable}
```
📍 [`src/h3mlx/lora.py:277-302`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/lora.py#L277-L302)
```python
 277 def check_trainable(trainable: dict, frozen: dict) -> None:
 278     """Refuses a pair of groups in which `trainable` would not be what the forward trains.
 279
 280     `ops.plinear` reads a named adapter's matrices only where the Linear lists that name as active. Matrices under a
 281     file's names, for a name that was never attached, or for an adapter that `set_active` switched off are merged
 282     into the weights and then ignored: the loss comes out without them and their gradients are exactly zero, with no
 283     error. (A file's names on a Linear with no named adapter are read, but as an unnamed adapter of scaling 1.)
 284
 285     So: `trainable` is not empty; each of its keys is a runtime name, `<module>.lora_A/B.<name>.weight`; that name is
 286     in `frozen`'s list of active adapters for that module; and each module's two matrices come together."""
 287     if not trainable:
 288         raise ValueError("no trainable matrices: `trainable` is empty")
 289     seen: dict = {}
 290     for key in trainable:
 291         m = RUNTIME_RE.match(key)
 292         if m is None:
 293             raise ValueError(f"{key!r} is not an attached adapter's matrix (`<module>.lora_A/B.<name>.weight`): "
 294                              "`attach` the adapter under a name and `split` it; a file's keys are not read by the forward")
 295         active = frozen.get(m["module"] + ADAPTERS, ())
 296         if m["name"] not in active:
 297             raise ValueError(f"the adapter {m['name']!r} is not active on {m['module']!r} in the frozen weights (active "
 298                              f"there: {list(active)}): its matrices would not be used and their gradients would be zero")
 299         seen.setdefault((m["module"], m["name"]), set()).add(m["matrix"])
 300     half = sorted(module for (module, _), matrices in seen.items() if matrices != {"A", "B"})
 301     if half:
 302         raise ValueError(f"{len(half)} module(s) have one of lora_A / lora_B without the other, e.g. {half[:3]}")
```

`join` is the way back: the two halves as one dict again, for a forward outside the loss, for `set_active`, or for
saving. An updated adapter goes in as the trainable half, not through a second `attach`.

`check_trainable` is what `join` and every loss check first (§7). A named adapter's matrices are read only where the
`Linear` lists the name as active, so a trainable group that is not that (a file's keys, a name never attached, an
adapter switched off) is merged into the weights and then ignored: the forward runs, and the gradients come back
exactly zero with no error. PEFT has no such state to get into, because there the trainable parameters are the
module's own. `test_check_trainable` covers each case.

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

📍 [`src/h3mlx/train.py:98-118`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L98-L118)
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
📍 [`src/h3mlx/train.py:140-212`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L140-L212)
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

📍 [`src/h3mlx/train.py:121-127`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L121-L127)
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

📍 [`src/h3mlx/vae.py:253-264`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/vae.py#L253-L264)
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
📍 [`src/h3mlx/train.py:130-137`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L130-L137)
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

📍 [`src/h3mlx/train.py:218-227`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L218-L227)
```python
 218 def check_inputs(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array) -> None:
 219     """What every loss and gradient of an example checks first: `trainable` is what the forward will train
 220     (`lora.check_trainable`: not empty, runtime names, active in `frozen`, both matrices), and `text` is exactly
 221     (1, the layout's text rows, text dim). A text of one row would be broadcast over every text position by the
 222     forward's scatter, and matrices the forward does not read get a gradient of exactly zero, both without an error."""
 223     lora.check_trainable(trainable, frozen)
 224     want = (1, len(example.layout.text_indices), cfg["text_dim"])
 225     if not isinstance(text, mx.array) or tuple(text.shape) != want:
 226         raise ValueError(f"text must be an array of shape {want} (1, the layout's text rows, text dim), got "
 227                          f"{tuple(text.shape) if hasattr(text, 'shape') else type(text).__name__}")
```
📍 [`src/h3mlx/train.py:273-288`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L273-L288)
```python
 273 def example_loss(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None) -> mx.array:
 274     """One example's loss: the transformer's forward on the example's rows with the weights `{**frozen, **trainable}`
 275     (`ExampleForward`), then `flow_match_loss` of the generated video rows against the example's target. Nothing is
 276     taken from the audio output or from the condition rows.
 277
 278     The weights come in as two arguments, the trainable matrices first (`lora.split`), so the function can be
 279     differentiated with respect to them and to nothing else; `check_inputs` refuses a `trainable` the forward would
 280     not use and a `text` that is not (1, text rows, text dim), the prompt's embeddings. Products with float32 LoRA
 281     matrices are float32 whatever the base's dtype (`ops.plinear`), and the loss is float32."""
 282     check_inputs(trainable, frozen, cfg, example, text)
 283     p = {**frozen, **trainable}
 284     forward = ExampleForward(cfg, example, text, weight)
 285     x, temb = forward.start(p)
 286     for i in range(cfg["num_layers"]):
 287         x = forward.block(x, temb, ops._sub(p, f"transformer_blocks.{i}."))
 288     return forward.finish(x, temb, p)
```
📍 [`src/h3mlx/train.py:291-295`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L291-L295)
```python
 291 def loss_and_gradients(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None):
 292     """(loss, gradients) of `example_loss`. The gradients are a dict with `trainable`'s keys and no others: the base
 293     weights and any other adapter in `frozen` are constants of the differentiation and get none."""
 294     check_inputs(trainable, frozen, cfg, example, text)
 295     return mx.value_and_grad(example_loss)(trainable, frozen, cfg, example, text, weight)
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
- the text is exactly (1, the layout's text rows, text dim). The forward scatters the text rows into the packed
  sequence with an assignment, and an assignment broadcasts: one row of text would fill every text position
  (`test_the_loss_refuses_a_text_of_the_wrong_shape`).

### 7.1 The forward that is differentiated

📍 [`src/h3mlx/train.py:230-270`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L230-L270)
```python
 230 class ExampleForward:
 231     """The transformer's forward on one example, for differentiating, in three stages: `start` (the projections, the
 232     token refiner, the scatter and the time embedding), `block` (one transformer block) and `finish` (the final norm,
 233     the heads and the loss). Each stage takes its weights as a dict, so a caller chooses which of them are arguments
 234     of a transform.
 235
 236     These are `ops.dit_forward`'s operations in its order, and the loss is the same bit for bit. Two things differ,
 237     and neither changes a value. Nothing is evaluated between the stages: the sampling forward evaluates after every
 238     block, and an array evaluated while it is being traced keeps its graph. And the modulation tables' rows are
 239     gathered with `ops.take_rows_for_training`, whose gradient over a half-precision table is the sum torch's is."""
 240
 241     def __init__(self, cfg: dict, example: Example, text: mx.array, weight=None):
 242         lay = example.layout
 243         self.cfg, self.example, self.text, self.weight = cfg, example, text, weight
 244         self.video, self.audio = example.video_rows[None], example.audio_rows[None]
 245         self.timestep, self.t_rows = mx.array(example.timestep), mx.array(example.timestep_indices)
 246         self.video_indices, self.audio_indices = mx.array(lay.video_indices), mx.array(lay.audio_indices)
 247         self.text_indices = mx.array(lay.text_indices)
 248         self.seq = len(lay.token_tags)
 249         self.adaln_indices = self.t_rows * ops.MODALITY_NUM + mx.array(lay.token_tags)
 250         self.rotary = ops.rope_cos_sin(mx.array(lay.position_ids.astype(np.float32)),
 251                                        ops.rope_inv_freq(cfg["rope_freq_dim"], cfg["rope_theta"]))
 252
 253     def start(self, p: dict) -> tuple[mx.array, mx.array]:
 254         """(the packed rows before the first block, the time embedding) from the whole dict of weights."""
 255         x = ops.dit_inputs(p, self.cfg, self.video, self.audio, self.text, self.seq, self.video_indices,
 256                            self.audio_indices, self.text_indices)
 257         return x, ops.two_time_embedding(self.timestep, None, p, self.cfg)
 258
 259     def block(self, x: mx.array, temb: mx.array, block_p: dict) -> mx.array:
 260         """One block on its own weights (the keys under `transformer_blocks.<i>.`, without that prefix)."""
 261         cfg = self.cfg
 262         return ops.dit_block(x, temb, self.adaln_indices, self.rotary, block_p, cfg["num_attention_heads"],
 263                              cfg["hidden_size"], cfg["norm_eps"], cfg["qk_norm_eps"], take=ops.take_rows_for_training)
 264
 265     def finish(self, x: mx.array, temb: mx.array, p: dict) -> mx.array:
 266         """The loss from the last block's output, from the whole dict of weights."""
 267         shift, scale = ops.norm_out_tables(temb, p)
 268         video_pred, _ = ops.dit_outputs(x, p, self.cfg, shift, scale, self.t_rows, self.video_indices,
 269                                         self.audio_indices, take=ops.take_rows_for_training)
 270         return flow_match_loss(self.example.generated(video_pred), self.example.target[None], self.weight)
```

`example_loss` does not call `ops.dit_forward`. It runs that forward's operations in its order, from the functions
`dit_forward` is itself made of (`ops.dit_inputs`, `ops.dit_block`, `ops.dit_outputs`), in three stages. Two things
differ from the sampling forward, and neither changes a value: `test_the_training_forward_is_the_sampling_forward`
compares the loss with one taken through `ops.dit_forward`, bit for bit, on three examples, each in float32 and over
a bf16 base, with adapters on every `Linear`.

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
📍 [`src/h3mlx/ops.py:166-168`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/ops.py#L166-L168)
```python
 166 def take_rows(table: mx.array, rows: mx.array) -> mx.array:
 167     """table[rows] along the first axis: each sequence row's entry of a (t, modality) table."""
 168     return mx.take(table, rows, axis=0)
```
📍 [`src/h3mlx/ops.py:171-183`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/ops.py#L171-L183)
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
  On 2,316 rows the gradients with shared tables are within 0.44% of that reference (one rounding of each sum
  apart); taken through the sampling forward's gather they are 20% to 52% away.
- **Why sampling keeps the plain gather.** The training gather makes a float32 copy of the gathered rows, twice their
  size, and sampling has no backward. `dit_block` and `dit_outputs` take the gather as an argument, `take`, and
  `ops.dit_forward` leaves it at `take_rows`: its operations are the ones it had.
- **Order on the GPU.** On the default device MLX adds a gather's cotangents in no fixed order, so a float32 sum can
  differ in its last bit from one run to the next, and a bf16 result that sits on a rounding boundary can then move
  one step. The CPU stream is repeatable. Tests that need exact equality over a bf16 base run there.

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
| 0 | 0.0114 | 0.0117 | 0.0041 | 0.0053 |
| 1 | 0.0118 | 0.0116 | 0.0005 | 0.0005 |
| 2 | 0.0156 | 0.0151 | 0.0010 | 0.0015 |

The test asks that MLX's gradient error be at most 1.5 times torch's own plus 1e-3, and that both losses be within 1%
of the float32 loss.

One detail of the torch side matters here. Loading an adapter into a bf16 model rounds its matrices to bf16, and
`cast_training_params` widens their dtype without bringing the values back. A trainer's matrices leave that grid at
their first step, so the tests put the float32 values back into the PEFT model after the cast: both sides then hold
the same matrices.

**Float32 products, pinned.** The test above cannot tell float32 products from bf16 ones: with ordinary matrices
either is within bf16's noise of the float32 run. An adapter whose rows cancel can:

📍 [`tests/test_train_gradients.py:93-108`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/tests/test_train_gradients.py#L93-L108)
```python
  93 def cancelling_adapter(cfg, modules, half, delta, seed, std=0.1):
  94     """An adapter that only float32 products get right. lora_A's second half is minus its first half plus
  95     `delta` * N, lora_B's second half repeats its first, and the scaling is 1 / delta: the update is B1 (N x), the
  96     difference of two products `1 / delta` times its size. With delta = 2**-10, float32 keeps 14 bits of it and bf16
  97     none."""
  98     g = torch.Generator().manual_seed(seed)
  99     shapes = lora.wired_shapes(cfg)
 100     tensors, alpha = {}, {}
 101     for m in modules:
 102         out_f, in_f = shapes[m]
 103         a1, n = torch.randn(half, in_f, generator=g) * std, torch.randn(half, in_f, generator=g) * std
 104         b1 = torch.randn(out_f, half, generator=g) * std
 105         tensors[m + ".lora_A.weight"] = to_mx(torch.cat([a1, -a1 + delta * n]))
 106         tensors[m + ".lora_B.weight"] = to_mx(torch.cat([b1, b1], dim=1))
 107         alpha[m] = 2 * half / delta
 108     return lora.Adapter(tensors, alpha)
```

Its update is the small difference of two large products, B₁(N x) computed as (B₁ A₁ x − B₁ A₁ x + δ B₁ N x) / δ
with δ = 2⁻¹⁰. Float32 keeps about 14 bits of that difference; bf16, with 8, keeps none.
`test_products_with_float32_matrices_are_float32_over_a_bf16_base` runs it over the bf16 base against torch and
PEFT, and runs our side once more with the same matrices cast to bf16, which is what products in the base's dtype
would compute:

| seed | gradient error against float32, torch | MLX | MLX with the matrices in bf16 |
| --- | --- | --- | --- |
| 0 | 0.0098 | 0.0102 | 0.57 |
| 1 | 0.0118 | 0.0118 | 0.87 |
| 2 | 0.0109 | 0.0114 | 0.60 |

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
| `test_products_with_float32_matrices_are_float32_over_a_bf16_base` (×3) | the trainable products are float32: an adapter that bf16 products get 57% to 87% wrong |
| `test_the_training_forward_is_the_sampling_forward` (×6) | the staged forward's loss is `ops.dit_forward`'s, bit for bit |
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

📍 [`src/h3mlx/backward.py:37-53`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/backward.py#L37-L53)
```python
  37 def checkpointed_loss(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None) -> mx.array:
  38     """`train.example_loss` with each block run as an `mx.checkpoint` of (the block's input, the time embedding, the
  39     block's trainable matrices). The same operations, so the same loss bit for bit."""
  40     check_inputs(trainable, frozen, cfg, example, text)
  41     p = {**frozen, **trainable}
  42     forward = ExampleForward(cfg, example, text, weight)
  43     x, temb = forward.start(p)
  44     for i in range(cfg["num_layers"]):
  45         prefix = f"{BLOCK}{i}."
  46         names = [k[len(prefix):] for k in trainable if k.startswith(prefix)]
  47         constants = ops._sub(frozen, prefix)
  48
  49         def block(x, temb, *matrices, constants=constants, names=names):
  50             return forward.block(x, temb, {**constants, **dict(zip(names, matrices))})
  51
  52         x = mx.checkpoint(block)(x, temb, *[trainable[prefix + k] for k in names])
  53     return forward.finish(x, temb, p)
```
📍 [`src/h3mlx/backward.py:56-61`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/backward.py#L56-L61)
```python
  56 def loss_and_gradients_checkpointed(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array,
  57                                     weight=None):
  58     """(loss, gradients) as `train.loss_and_gradients` returns them, with each block checkpointed. One lazy graph:
  59     nothing is evaluated here, and the caller's evaluation must not come before the gradients are built."""
  60     check_inputs(trainable, frozen, cfg, example, text)
  61     return mx.value_and_grad(checkpointed_loss)(trainable, frozen, cfg, example, text, weight)
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

📍 [`src/h3mlx/backward.py:64-114`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/backward.py#L64-L114)
```python
  64 def loss_and_gradients_blockwise(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array,
  65                                  weight=None):
  66     """(loss, gradients) as `train.loss_and_gradients` returns them, one block at a time; evaluated on return.
  67
  68     Forward: the stages run in order, each evaluated, and each block's input is kept. Backward: the vector-Jacobian
  69     product of the last stage gives the cotangent of the last block's output; each block, last to first, turns the
  70     cotangent of its output into those of its input, of the time embedding and of its own trainable matrices, and is
  71     evaluated before the next; the stage before the blocks takes what is left. One block's graph is alive at a time,
  72     and what grows with depth is the kept inputs, one per block. Each block's forward runs twice (on the way up, and
  73     again inside its product)."""
  74     check_inputs(trainable, frozen, cfg, example, text)
  75     p = {**frozen, **trainable}
  76     forward = ExampleForward(cfg, example, text, weight)
  77     head_keys, block_keys, tail_keys = _stages(trainable, cfg)
  78
  79     def start(*matrices):
  80         return list(forward.start({**p, **dict(zip(head_keys, matrices))}))
  81
  82     def block(i):
  83         prefix = f"{BLOCK}{i}."
  84         constants = ops._sub(frozen, prefix)
  85         names = [k[len(prefix):] for k in block_keys[i]]
  86         return lambda x, temb, *matrices: forward.block(x, temb, {**constants, **dict(zip(names, matrices))})
  87
  88     def finish(x, temb, *matrices):
  89         return forward.finish(x, temb, {**p, **dict(zip(tail_keys, matrices))})
  90
  91     x, temb = start(*[trainable[k] for k in head_keys])
  92     mx.eval(x, temb)
  93     inputs = [x]
  94     for i in range(cfg["num_layers"]):
  95         x = block(i)(inputs[-1], temb, *[trainable[k] for k in block_keys[i]])
  96         mx.eval(x)
  97         inputs.append(x)
  98
  99     (loss,), (g_x, g_temb, *g_tail) = mx.vjp(finish, [inputs.pop(), temb, *[trainable[k] for k in tail_keys]],
 100                                              [mx.array(1.0, dtype=mx.float32)])
 101     mx.eval(loss, g_x, g_tail)
 102     grads = dict(zip(tail_keys, g_tail))
 103     for i in reversed(range(cfg["num_layers"])):
 104         _, (g_x, g_t, *g_block) = mx.vjp(block(i), [inputs.pop(), temb, *[trainable[k] for k in block_keys[i]]], [g_x])
 105         if head_keys:  # the time embedding's cotangent is for the stage before the blocks: left unevaluated without one
 106             g_temb = g_temb + g_t
 107             mx.eval(g_temb)
 108         mx.eval(g_x, g_block)
 109         grads.update(zip(block_keys[i], g_block))
 110     if head_keys:
 111         _, g_head = mx.vjp(start, [trainable[k] for k in head_keys], [g_x, g_temb])
 112         mx.eval(g_head)
 113         grads.update(zip(head_keys, g_head))
 114     return loss, {k: grads[k] for k in trainable}
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
| default device: plain | 68 | 106 | 153 | 7.1 |
| default device: checkpointed | 76 | 78 | 100 | 2.0 |
| default device: block by block | 30 | 31 | 30 | 0.0 |
| CPU stream: through the sampling forward | 80 | 159 | 316 | 19.7 |
| CPU stream: plain | 33 | 59 | 109 | 6.3 |
| CPU stream: checkpointed | 20 | 21 | 43 | 1.9 |
| CPU stream: block by block | 17 | 17 | 18 | 0.1 |

| 2,324 rows | 4 blocks | 8 blocks | 16 blocks | growth per block |
| --- | ---: | ---: | ---: | ---: |
| default device: through the sampling forward | 468 | 784 | 1564 | 91.4 |
| default device: plain | 509 | 672 | 1047 | 44.8 |
| default device: checkpointed | 431 | 442 | 536 | 8.7 |
| default device: block by block | 166 | 168 | 172 | 0.5 |
| CPU stream: through the sampling forward | 437 | 871 | 1739 | 108.5 |
| CPU stream: plain | 228 | 390 | 715 | 40.7 |
| CPU stream: checkpointed | 119 | 119 | 208 | 7.4 |
| CPU stream: block by block | 109 | 111 | 116 | 0.5 |

What the numbers show, and what they do not:

- **Differentiating the sampling forward as it is grows fastest,** on both devices, and at 16 blocks it is the
  largest number in every table. Taking the evals out of the forward (plain) cuts the growth to between a third and
  a half.
- **Block by block is nearly flat.** Its growth per block is the saved input: 596 rows × 96 × 2 bytes is 0.11 MB, and
  2,324 rows is 0.45 MB.
- **Checkpointing saves less, and less predictably.** On this model it grows more slowly than plain on both devices.
  On the sixteen-block chain of §8.1, which the script also prints, it saves a third on the CPU stream (86 MB plain,
  55 MB checkpointed) and nothing on the default device (195 MB plain, 209 MB checkpointed), where one evaluation
  holds its buffers until the device has run them. What it saves depends on the device and on the graph.
- **Time.** At 16 blocks the two bounded ways took 1.3 to 2 times the plain gradient's time on the default device
  and 1.5 to 1.6 times on the CPU stream (one run each; the script prints the seconds).
- **The default device's numbers move between runs,** by tens of MB in some cells: the checkpointed gradient at 4
  blocks and 596 rows measured 54 MB in one run and 76 MB in another. The CPU stream's repeat.
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
| `test_bounded_gradients_at_other_levels_and_layouts` (×2) | text-to-image and two references, a shared timestep, a loss weight |
| `test_bounded_gradients_over_a_quantized_base` (×2) | quantized weights captured by a checkpoint, and as constants of each product |
| `test_bounded_gradients_refuse_what_the_loss_refuses` (×3) | `check_inputs` before anything is computed |
| `test_stages_cover_every_trainable_key_once` | which stage of the forward each trainable matrix belongs to |
| `test_a_checkpointed_function_gives_no_gradient_to_what_it_captures` | the first fact about `mx.checkpoint` |
| `test_an_eval_in_the_loop_defeats_the_checkpoint` | the second |
| `test_peak_memory_grows_more_slowly_with_depth_the_less_is_kept` | the directions of the memory measurement |

## 9. The optimizer step

📍 [`src/h3mlx/train.py:301-306`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L301-L306)
```python
 301 def adamw(*, learning_rate, betas, eps: float, weight_decay: float) -> optim.AdamW:
 302     """MLX's AdamW set up to compute `torch.optim.AdamW`'s update: with `bias_correction=True`. MLX leaves the bias
 303     correction out by default, and its first step is then `(1 - beta1) / sqrt(1 - beta2)` times torch's, 3.2 times
 304     for betas of (0.9, 0.999). No hyper-parameter has a default here: every one is the caller's to give."""
 305     return optim.AdamW(learning_rate=learning_rate, betas=list(betas), eps=eps, weight_decay=weight_decay,
 306                        bias_correction=True)
```
📍 [`src/h3mlx/train.py:309-339`](https://github.com/sublimator/minimax-h3-mlx/blob/4d5a5efffa4b870b544079496a43848116ad8591/src/h3mlx/train.py#L309-L339)
```python
 309 def optimizer_step(optimizer, trainable: dict, grads: dict, *, max_grad_norm: float) -> tuple[dict, float]:
 310     """One step: (the updated matrices, the gradients' global norm before clipping), both evaluated.
 311
 312     The gradients are scaled down to a global norm of `max_grad_norm` when theirs is larger, as torch's
 313     `clip_grad_norm_` scales them (`math.inf` clips nothing), and the optimizer applies them. Refused before any
 314     matrix is touched:
 315
 316     * matrices that are not float32. At a learning rate of 1e-4 a bf16 element of magnitude 2**-5 or more does not
 317       move at all, because the update is under half its spacing, and a smaller one moves by whole steps of its
 318       spacing; in float16 the second moment of small gradients underflows and the update is NaN. So the trainable
 319       matrices are float32 over whatever base (`lora.attach(..., keep_dtype=True)`).
 320     * gradients whose keys are not the matrices' keys;
 321     * a `max_grad_norm` that is not a positive number;
 322     * a gradient norm that is not finite, which torch's clip refuses under `error_if_nonfinite` and otherwise lets
 323       through into the weights."""
 324     if not trainable or set(grads) != set(trainable):
 325         raise ValueError(f"the gradients' keys must be the trainable matrices' keys: {len(grads)} against {len(trainable)}")
 326     narrow = sorted(k for k, v in trainable.items() if v.dtype != mx.float32)
 327     if narrow:
 328         raise ValueError(f"the trainable matrices must be float32; {len(narrow)} are not, e.g. {narrow[0]} "
 329                          f"({trainable[narrow[0]].dtype}): half-precision matrices do not take updates of this size")
 330     if not isinstance(max_grad_norm, numbers.Real) or isinstance(max_grad_norm, bool) or not max_grad_norm > 0:
 331         raise ValueError(f"max_grad_norm must be a positive number (math.inf to clip nothing), got {max_grad_norm!r}")
 332     clipped, norm = optim.clip_grad_norm(grads, float(max_grad_norm))
 333     norm = float(norm)
 334     if not math.isfinite(norm):
 335         raise ValueError(f"the gradients' norm is {norm}: not applied")
 336     updated = optimizer.apply_gradients(clipped, trainable)
 337     updated = {k: updated[k] for k in trainable}
 338     mx.eval(updated, optimizer.state)
 339     return updated, norm
```

The original is torch's: `torch.optim.AdamW` after `torch.nn.utils.clip_grad_norm_`. The port uses MLX's own
`AdamW` and `clip_grad_norm`, whose formulas are the same ones (decoupled weight decay, then the bias-corrected
update; the gradients scaled by `min(1, max_norm / (norm + 1e-6))`), and checks the result against torch over a run
of steps. **No hyper-parameter has a default.** The learning rate, the betas, epsilon, the weight decay and the bound
on the gradient's norm are keyword arguments the caller must give; the recipe's values for them were not read.

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
- a bound on the norm that is not a positive number (`math.inf` clips nothing);
- a gradient norm that is not finite. torch's clip raises for one only under `error_if_nonfinite`, and otherwise
  the NaN goes into the weights. Here it is always an error, raised before the optimizer's state is touched.

### 9.5 The tests

| test | pins |
| --- | --- |
| `test_steps_are_torchs` (×4) | twelve steps against `torch.optim.AdamW` after `clip_grad_norm_`: losses, norms, and the matrices' movement |
| `test_steps_over_a_bf16_base_stay_near_torchs` | the same loop over a bf16 base, loosely |
| `test_clipping_is_torchs` (×4) | the clipped gradients and the returned norm against torch's |
| `test_mlx_adamw_needs_bias_correction_to_be_torchs` | MLX's default against torch's first step; `train.adamw` against five |
| `test_half_precision_matrices_do_not_take_updates_of_1e_4` | bf16 and float16 under AdamW at 1e-4 |
| `test_optimizer_step_refusals` | what the step refuses, with the optimizer's state untouched |
| `test_no_hyper_parameter_has_a_default` | every hyper-parameter keyword-only and required |

## 10. What each test would catch

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

One change in this list is caught by a narrow margin. The block-by-block gradient's forward walk evaluates each block
before the next; without that evaluation the gradients are the same and only the memory differs, on the default
device only. The memory test sees it as a growth of 6% to 17% of the plain gradient's against a bound of 5%, with
numbers that vary from run to run: five runs of the wrong version failed five times, and under an earlier, looser
bound of a quarter it had passed.

Where a row names a dtype, the change is caught only there. A tensor timestep's `1 − t` taken in float64 from the
value before its cast is one such: for a float32 sample it rounds to the same float32 number, because that
subtraction is a single correctly rounded operation either way, and only the bf16 cases fail, where the timestep
must be cast to bf16 before the subtraction.

## 11. Not done, and open

- **No training of a real model.** No loop, no latent or text caches, no script, and no run on real weights. On a
  tiny random model the pieces have been used together for twelve optimizer steps (§9), and no further.
- **Memory at real size.** §8's measurements are of a synthetic model with hidden size 96. Which of the three
  gradients fits 50 real blocks, how long a step takes, and whether the float32 copy the training gather makes
  (§7.1) matters there, are unmeasured.
- **The optimizer's state.** `optimizer_step` keeps it in MLX's optimizer object. Saving it and resuming from it are
  not built, and neither is a learning-rate schedule beyond what MLX's optimizer takes as its learning rate.
- **Sums on the GPU have no fixed order.** A gradient over a bf16 base is repeatable on the CPU stream and can
  differ by a bf16 rounding step from run to run on the default device (§7.1). Nothing here makes a training run
  bit-reproducible.
- **Other sums in half precision.** The modulation tables' gather was the one place found where MLX's backward adds
  many rows in bf16 on the way to a trainable matrix. A trainable matrix placed somewhere else than a `Linear`, or a
  frozen adapter turned trainable in bf16, would need the same look.
- **The recipe's unread parts.** How the trainer mixes the picture with noise, how it samples timesteps, its density,
  its loss weighting and its optimizer settings were not read (§6.1). No preset is defined.
- **A step-distilled base.** P-H3-Image samples in 8 steps without guidance. Whether a plain flow-matching loss on it
  keeps that behaviour, with or without the assistant adapter, is untested here.
- **The assistant adapter's values.** Only its header was read (§5.4).
- **Reference pictures' sizes.** The builder takes latents. Which sizing rule a trainer uses for references is not
  decided (§6.4).
- **Density and weighting defaults.** None are chosen. The tests cover the functions, not which settings suit this
  model.
- **bf16.** Gradients over a bf16 base are tested on four tiny layers against torch's own error (§7.3), and a run of
  optimizer steps over one only loosely (§9.1). A bf16 product path for the trainable matrices is not built.
- **Real files.** No published adapter's tensors were read by this code; the tests write their own. Layout details of
  real files that the converter's docstring does not mention would not be caught.
- **Routing between partitions.** `h3mlx.lora` returns an adapter and the caller attaches it to the weights it means.
  diffusers' routing of a file to `transformer` or `transformer_ref` is not ported.