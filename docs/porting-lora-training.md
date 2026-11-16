# Porting LoRA training to MLX, part 1: PEFT and diffusers → `h3mlx`

This guide covers the first pieces a LoRA trainer for the MLX transformer needs, each ported op by op and tested
against the PyTorch original on small random tensors:

1. **LoRA adapters:** PEFT's scaling and initialisation, attaching and merging, and the files diffusers reads and
   writes (§1).
2. **A converter** for adapters saved under the original checkpoint's key names (§2).
3. **The forward process, the training target and the loss** (§3).
4. **Sampling the noise level and weighting the loss** (§4).

In each section the original comes first, then the port, then what changed and which test ties them together. It
assumes the main guide (`docs/porting.md`) for the transformer and the scheduler, and the HyperFlow guide
(`docs/porting-hyperflow.md`) for how a LoRA reaches the forward (`ops.plinear`).

**Nothing here trains yet.** There is no training loop, no gradient test, no optimizer and no run on real weights;
§6 lists what is missing. Everything below is weight-free.

**Sources, pinned.**

| what | where | pin |
| --- | --- | --- |
| LoRA layer, scaling, init, merge (Apache-2.0) | `huggingface/peft`, `src/peft/tuners/lora/` | tag `v0.21.2` (`eaa35c7`), the version the tests import |
| LoRA files, the original-layout converter, the scheduler, `training_utils` (Apache-2.0) | `huggingface/diffusers` main | `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1`, the commit the tests import |

**The oracle, and what it is not.** The tests compare against PEFT, diffusers and torch run on tiny models and random
tensors. diffusers' example training scripts are not part of the checkout these guides read, and no trainer for this
model was run or read here. So this guide claims parity with the library functions it quotes and with nothing else:
where a trainer has to make a choice (the shift, the number of noise levels, the density, the weighting), the port
takes an argument and has no default.

| piece | ours | original | tests |
| --- | --- | --- | --- |
| scaling, init, attach, merge, files | `src/h3mlx/lora.py`, `ops.plinear` | PEFT `layer.py`; diffusers `loaders/peft.py`, `utils/peft_utils.py`, `loaders/lora_pipeline.py` | `tests/test_lora.py` (68) |
| original-layout converter | `lora.convert_original` | diffusers `loaders/lora_conversion_utils.py` | `tests/test_lora_convert.py` (53) |
| forward process, target, loss | `src/h3mlx/train.py` | diffusers `scheduling_minimax_h3.py`; torch `mse_loss` | `tests/test_train.py` (47) |
| noise-level sampling, loss weighting | `src/h3mlx/train.py` | diffusers `training_utils.py`, its two schedulers | `tests/test_train_sampling.py` (33) |

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

📍 [`src/h3mlx/ops.py:100-105`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/ops.py#L100-L105)
```python
 100 def times_scalar(x: mx.array, s) -> mx.array:
 101     """torch's `tensor * python_float`. A half-precision tensor is multiplied in float32 by the float32 scalar and
 102     rounded once; MLX alone would first round the scalar to the tensor's dtype (0.7 becomes 0.69921875 in bf16)."""
 103     if x.dtype in (mx.float16, mx.bfloat16):
 104         return (x.astype(mx.float32) * s).astype(x.dtype)
 105     return x * s
```
📍 [`src/h3mlx/ops.py:108-123`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/ops.py#L108-L123)
```python
 108 def plinear(x: mx.array, p: dict, name: str, bias: bool = False) -> mx.array:
 109     """The Linear `name` of state dict `p`, plus its LoRA when `name.lora_A.weight` / `name.lora_B.weight` are present.
 110
 111     PEFT's lora.Linear: the input is cast to the adapter's dtype, the two products round in that dtype, the delta is
 112     multiplied by the scaling (alpha / rank), and the sum is cast back to the base result's dtype. The scaling is the
 113     0-d float32 array `name.lora_scaling`, which `lora.attach` adds when it is not 1; without it the `* scaling` is
 114     exact and left out (HyperFlow's path: `hyperflow.attach` refuses any other scaling)."""
 115     y = linear(x, p[name + ".weight"], p[name + ".bias"] if bias else None)
 116     a = p.get(name + ".lora_A.weight")
 117     if a is None:
 118         return y
 119     delta = linear(linear(x.astype(a.dtype), a), p[name + ".lora_B.weight"])
 120     scaling = p.get(name + ".lora_scaling")
 121     if scaling is not None:
 122         delta = times_scalar(delta, scaling)
 123     return (y + delta).astype(y.dtype)
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

📍 [`src/h3mlx/lora.py:110-120`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L110-L120)
```python
 110 def kaiming_uniform(shape, generator) -> np.ndarray:
 111     """`nn.init.kaiming_uniform_(weight, a=math.sqrt(5))` for a (rank, in) `lora_A`, drawn from a torch generator so the
 112     values are torch's: uniform in +-bound, with bound = sqrt(3) * gain / sqrt(fan_in) and gain = sqrt(2 / (1 + a^2)),
 113     which is 1 / sqrt(in). The bound is computed in the same order as torch computes it, in Python floats."""
 114     import torch
 115
 116     fan_in = shape[1]
 117     gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
 118     std = gain / math.sqrt(fan_in)
 119     bound = math.sqrt(3.0) * std
 120     return torch.empty(tuple(shape), dtype=torch.float32).uniform_(-bound, bound, generator=generator).numpy()
```
📍 [`src/h3mlx/lora.py:123-145`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L123-L145)
```python
 123 def init(cfg: dict, rank: int, alpha: float | None = None, modules=None, seed: int = 0) -> Adapter:
 124     """A new adapter as PEFT initialises one (`init_lora_weights=True`): `lora_A` Kaiming-uniform, `lora_B` zero, so it
 125     starts as a no-op. Both are float32 whatever the base weights' dtype (a bf16 parameter is too coarse to train).
 126     `alpha` defaults to `rank` (scaling 1). `modules` defaults to `block_modules(cfg)`; they are drawn in sorted order
 127     from one torch generator seeded with `seed` (PEFT draws in the model's module order and from the global generator,
 128     so the same seed does not give PEFT's values)."""
 129     import torch
 130
 131     if rank <= 0:
 132         raise ValueError(f"`rank` should be a positive integer value but the value passed is {rank}")
 133     if alpha is not None:
 134         check_alpha("init", alpha)
 135     wired = wired_shapes(cfg)
 136     modules = sorted(block_modules(cfg) if modules is None else modules)
 137     generator = torch.Generator().manual_seed(seed)
 138     tensors = {}
 139     for module in modules:
 140         if module not in wired:
 141             raise ValueError(f"{module} is not a Linear the forward runs through ops.plinear")
 142         out_f, in_f = wired[module]
 143         tensors[module + ".lora_A.weight"] = mx.array(kaiming_uniform((rank, in_f), generator))
 144         tensors[module + ".lora_B.weight"] = mx.zeros((out_f, rank), dtype=mx.float32)
 145     return Adapter(tensors, {m: rank if alpha is None else alpha for m in modules})
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
  (`block_modules`). The time embedder's two linears can be named explicitly. Nothing else can (§1.3).

`test_init_is_a_float32_no_op` checks the shapes, the dtype over a bf16 base, the bound, the zero `lora_B`, the
order of the draws, and that the forward with a fresh adapter is the base model's bit for bit.

### 1.3 Attaching, and the dtype

📍 [`src/h3mlx/lora.py:35-55`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L35-L55)
```python
  35 @dataclass
  36 class Adapter:
  37     """LoRA factors for some of the transformer's Linears, in PEFT's parametrisation.
  38
  39     `tensors` maps `<module>.lora_A.weight` to (rank, in) and `<module>.lora_B.weight` to (out, rank), with diffusers
  40     module names and no component prefix, so the dict merges straight into the transformer's weights. `alpha` maps each
  41     module to its lora_alpha. A module's rank is its `lora_A`'s first axis; ranks and alphas may differ per module."""
  42
  43     tensors: dict
  44     alpha: dict
  45
  46     @property
  47     def modules(self) -> list[str]:
  48         return sorted(self.alpha)
  49
  50     def rank(self, module: str) -> int:
  51         return int(self.tensors[module + ".lora_A.weight"].shape[0])
  52
  53     def scaling(self, module: str) -> float:
  54         """`lora_alpha / r`, as PEFT's `update_layer` sets it (without rsLoRA)."""
  55         return self.alpha[module] / self.rank(module)
```
📍 [`src/h3mlx/lora.py:61-64`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L61-L64)
```python
  61 def wired_shapes(cfg: dict) -> dict:
  62     """{module: (out, in)} for every Linear the forward runs through `ops.plinear`: the six matrices of each block and
  63     refiner block, and the time embedder's two. (`hyperflow.wired_shapes` without HyperFlow's endpoint embedder.)"""
  64     return {m: s for m, s in hyperflow.wired_shapes(cfg).items() if not m.startswith(hyperflow.ENDPOINT)}
```
📍 [`src/h3mlx/lora.py:85-104`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L85-L104)
```python
  85 def check(adapter: Adapter, cfg: dict) -> None:
  86     """Every module is a Linear the forward applies a LoRA to (PEFT would wrap any `nn.Linear` a file names; a LoRA on
  87     `proj_in` or `adaln_proj` would be silently ignored here, so it is refused), each pair fits it, and each alpha is a
  88     positive, finite number."""
  89     if not adapter.alpha:
  90         raise ValueError("no LoRA tensors in this adapter")
  91     wired = wired_shapes(cfg)
  92     unwired = sorted(m for m in adapter.alpha if m not in wired)
  93     if unwired:
  94         raise ValueError(f"{len(unwired)} LoRA target(s) are not run through ops.plinear here (the port would ignore "
  95                          f"them), e.g. {unwired[:5]}")
  96     for module in adapter.modules:
  97         a = tuple(adapter.tensors[module + ".lora_A.weight"].shape)
  98         b = tuple(adapter.tensors[module + ".lora_B.weight"].shape)
  99         out_f, in_f = wired[module]
 100         if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
 101             raise ValueError(f"{module}: lora_A {a} / lora_B {b} do not fit the Linear ({out_f}, {in_f})")
 102         if a[0] <= 0:
 103             raise ValueError(f"{module}: LoRA rank must be positive, got {a[0]}")
 104         check_alpha(module, adapter.alpha[module])
```
📍 [`src/h3mlx/lora.py:151-171`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L151-L171)
```python
 151 def attach(params: dict, cfg: dict, adapter: Adapter, keep_dtype: bool = False) -> dict:
 152     """The transformer's weights with the adapter beside them, for `ops.plinear`; `params` is not modified.
 153
 154     By default each matrix is cast to its base weight's dtype, as PEFT's injection does (`load_lora_adapter`, and
 155     `hyperflow.attach`). `keep_dtype=True` leaves the adapter as it is: float32 factors over a bf16 base, what
 156     `cast_training_params` gives a trainer (and what diffusers' `low_cpu_mem_usage=True` loading gives a float32 file).
 157     A module whose scaling is not 1 also gets `<module>.lora_scaling`."""
 158     check(adapter, cfg)
 159     taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
 160     if taken:
 161         raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: one adapter per Linear")
 162     out = dict(params)
 163     for module in adapter.modules:
 164         dtype = params[module + ".weight"].dtype
 165         for matrix in ("A", "B"):
 166             key = f"{module}.lora_{matrix}.weight"
 167             out[key] = adapter.tensors[key] if keep_dtype else adapter.tensors[key].astype(dtype)
 168         scaling = adapter.scaling(module)
 169         if scaling != 1.0:
 170             out[module + SCALING] = mx.array(scaling, dtype=mx.float32)
 171     return out
```

- **Only wired linears.** PEFT wraps any `nn.Linear` a file names. Our forward applies a LoRA only where it calls
  `plinear`: the six matrices of each block and refiner block and the time embedder's two. A LoRA on `proj_in`, on an
  `adaln_proj` or on the output heads would be silently ignored, so it is refused, as in the HyperFlow loader.
- **One adapter per `Linear`.** Attaching to weights that already carry a LoRA, HyperFlow's included, is refused.
  Stacking two adapters is a later piece.
- **A positive, finite alpha. This differs from PEFT on purpose.** PEFT takes any number: zero leaves an adapter that
  does nothing and cannot train, a negative one flips the delta's sign, NaN and infinity poison the forward. `check`
  refuses them, and so do `init`, `read` and `save` (§1.5). `test_alpha_must_be_positive_and_finite` shows PEFT's
  layer taking each of the four and ours refusing it at every entry and exit.

📍 [`src/h3mlx/lora.py:73-82`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L73-L82)
```python
  73 def check_alpha(what: str, alpha) -> None:
  74     """An alpha is a positive, finite number. PEFT takes any: zero leaves an adapter that does nothing and cannot
  75     train, a negative one flips the delta's sign, and NaN or infinity poison the forward and are written to the file's
  76     header as `NaN` / `Infinity`, which is not JSON. Refused wherever an alpha comes in or goes out."""
  77     try:
  78         ok = math.isfinite(alpha) and alpha > 0
  79     except TypeError:
  80         ok = False
  81     if not ok:
  82         raise ValueError(f"{what}: lora_alpha is {alpha!r}; it must be a positive, finite number")
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

📍 [`src/h3mlx/lora.py:174-182`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L174-L182)
```python
 174 def delta_weight(a: mx.array, b: mx.array, scaling: float) -> mx.array:
 175     """PEFT's `Linear.get_delta_weight` on CPU: `(B @ A) * scaling`, with half-precision factors multiplied in float32
 176     and the product cast back to their dtype."""
 177     dtype = b.dtype
 178     half = dtype in (mx.float16, mx.bfloat16)
 179     if half:
 180         a, b = a.astype(mx.float32), b.astype(mx.float32)
 181     out = (b @ a) * scaling
 182     return out.astype(dtype) if half else out
```
📍 [`src/h3mlx/lora.py:185-206`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L185-L206)
```python
 185 def merge(params: dict, cfg: dict, adapter: Adapter) -> dict:
 186     """The transformer's plain weights with the adapter added into them (PEFT's `Linear.merge`: `weight +=
 187     delta_weight`, in the weight's dtype); `params` is not modified, and the result has its keys and nothing of the
 188     adapter left to apply.
 189
 190     Attaching and merging are alternatives here. Weights that already carry a LoRA on one of the adapter's Linears are
 191     refused: the attached factors would stay in the result, and `plinear` would apply them on top of the merged ones.
 192     (A PEFT layer holds both states itself, and its forward skips an adapter once it is merged.) Not for quantized
 193     weights."""
 194     check(adapter, cfg)
 195     taken = [m for m in adapter.modules if m + ".lora_A.weight" in params]
 196     if taken:
 197         raise ValueError(f"{len(taken)} Linear(s) already carry a LoRA, e.g. {taken[:3]}: merge into the plain "
 198                          "weights, not into attached ones (the adapter would be applied twice)")
 199     out = dict(params)
 200     for module in adapter.modules:
 201         w = params[module + ".weight"]
 202         if isinstance(w, ops.QuantizedWeight):
 203             raise ValueError(f"{module}: can't merge a LoRA into a quantized weight; merge first, then quantize")
 204         a, b = adapter.tensors[module + ".lora_A.weight"], adapter.tensors[module + ".lora_B.weight"]
 205         out[module + ".weight"] = (w + delta_weight(a, b, adapter.scaling(module))).astype(w.dtype)
 206     return out
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

📍 [`src/h3mlx/lora.py:377-383`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L377-L383)
```python
 377 def pattern_key(pattern_keys, key_to_match: str) -> str:
 378     """PEFT's `get_pattern_key`: the first `rank_pattern` / `alpha_pattern` key that matches the module name as a
 379     regular expression on its last dotted parts, else the name itself."""
 380     for key in pattern_keys:
 381         if re.match(rf"(.*\.)?({key})$", key_to_match):
 382             return key
 383     return key_to_match
```
📍 [`src/h3mlx/lora.py:386-399`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L386-L399)
```python
 386 def _component(tensors: dict, prefix: str | None) -> tuple[dict, str | None]:
 387     """The keys of one component with its prefix removed (`load_lora_adapter(prefix=...)`'s filter), and that prefix.
 388     With no `prefix` given the file must hold one component, prefixed or not."""
 389     found = [p for p in PREFIXES if any(k.startswith(p + ".") for k in tensors)]
 390     if prefix is None:
 391         if len(found) > 1:
 392             raise ValueError(f"this file holds LoRAs for {found}: say which with `prefix=`")
 393         prefix = found[0] if found else None
 394         other = () if prefix is None else [k for k in tensors if not k.startswith(prefix + ".")]
 395         if other:
 396             raise ValueError(f"keys with and without the `{prefix}.` prefix in one file, e.g. {sorted(other)[:3]}")
 397     if prefix is None:
 398         return dict(tensors), None
 399     return {k.removeprefix(prefix + "."): v for k, v in tensors.items() if k.startswith(prefix + ".")}, prefix
```
📍 [`src/h3mlx/lora.py:402-406`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L402-L406)
```python
 402 def _refuse_dora(keys) -> None:
 403     """DoRA is not implemented: refuse its keys (diffusers drops a kohya `dora_scale` with a warning, and loads PEFT's
 404     `lora_magnitude_vector` as DoRA)."""
 405     if any("dora_scale" in k or "lora_magnitude_vector" in k for k in keys):
 406         raise NotImplementedError("DoRA adapters are not supported")
```
📍 [`src/h3mlx/lora.py:409-479`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L409-L479)
```python
 409 def read(tensors: dict, header: dict | None = None, prefix: str | None = None, head_dim: int = HEAD_DIM) -> Adapter:
 410     """An adapter from a LoRA file's tensors and its safetensors header (`__metadata__`), as diffusers would load it.
 411
 412     Keys: `transformer.<module>.lora_A/B.weight` (what `save_lora_weights` writes and `load_lora_adapter` reads by
 413     default), the same without the prefix (what `save_lora_adapter` writes), PEFT's own state dict (a `.default.`
 414     infix), or the original checkpoint's names (`convert_original`, which folds any `.alpha` tensors into the matrices;
 415     `head_dim` is passed to it).
 416
 417     `prefix` picks one component's keys from a file that holds several, as `load_lora_adapter(prefix=...)` does; other
 418     components' keys are then not looked at. Without it the file must hold one component. An original-layout file
 419     names no component and is read whole whatever `prefix` says: the converter labels it `transformer.`, and the
 420     mixin loads those layers into either partition (`load_into_transformer_ref`). The adapter itself belongs to no
 421     partition; the caller attaches it to the weights it means.
 422
 423     Alpha, per module: from the `lora_adapter_metadata` diffusers writes (`r`, `lora_alpha`, `rank_pattern`,
 424     `alpha_pattern`, resolved as PEFT resolves them); else from a plain `alpha` entry in the header, taken as one
 425     network alpha for every module; else the module's own rank (scaling 1), which is `get_peft_kwargs`' rule. DoRA,
 426     rsLoRA, a LoRA bias, `fan_in_fan_out` and an alpha that is not positive and finite are refused, not ignored."""
 427     header = dict(header or {})
 428     tensors = dict(tensors)
 429     has_alpha_tensors = any(k.endswith(".alpha") for k in tensors)
 430     if is_original_layout(tensors):
 431         _refuse_dora(tensors)  # before the converter, which would call a `dora_scale` key a leftover
 432         tensors = convert_original(tensors, head_dim)
 433         prefix = "transformer"  # the converter's label, not a choice of partition
 434     elif any(".default.weight" in k for k in tensors) and not any(k.startswith(tuple(p + "." for p in PREFIXES)) for k in tensors):
 435         tensors = {k.replace(".default.", "."): v for k, v in tensors.items()}  # a PEFT state dict, adapter "default"
 436     tensors, prefix = _component(tensors, prefix)
 437     _refuse_dora(tensors)  # of the component read, not of the others
 438
 439     by_module = {}
 440     for key in tensors:
 441         m = KEY_RE.match(key)
 442         if m is None:
 443             raise ValueError(f"unexpected key {key!r}: only <module>.lora_A/B.weight keys are supported")
 444         by_module.setdefault(m["module"], set()).add(m["matrix"])
 445     incomplete = sorted(k for k, v in by_module.items() if v != {"A", "B"})
 446     if incomplete:
 447         raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
 448     if not by_module:
 449         raise ValueError("no LoRA tensors" + (f" under the `{prefix}.` prefix" if prefix else "") + " in this file")
 450     ranks = {m: int(tensors[m + ".lora_B.weight"].shape[1]) for m in by_module}
 451
 452     config = json.loads(header[METADATA_KEY]) if header.get(METADATA_KEY) else None
 453     if config is not None and prefix is not None:
 454         config = {k.removeprefix(prefix + "."): v for k, v in config.items() if k.startswith(prefix + ".")}
 455     if config is not None:
 456         if "r" not in config or "lora_alpha" not in config:
 457             raise ValueError(f"the file's {METADATA_KEY} has no `r` / `lora_alpha`" + (f" under `{prefix}.`" if prefix else ""))
 458         on = [k for k in UNSUPPORTED if config.get(k)]
 459         if on:
 460             raise NotImplementedError(f"this adapter was saved with {on}, which the port does not implement")
 461         rank_pattern, alpha_pattern = config.get("rank_pattern") or {}, config.get("alpha_pattern") or {}
 462         alpha = {}
 463         for module, rank in ranks.items():
 464             r = rank_pattern.get(pattern_key(rank_pattern, module), config["r"])
 465             if r != rank:
 466                 raise ValueError(f"{module}: the metadata says rank {r} but lora_B has rank {rank}")
 467             alpha[module] = alpha_pattern.get(pattern_key(alpha_pattern, module), config["lora_alpha"])
 468             check_alpha(f"{module} (the file's {METADATA_KEY})", alpha[module])
 469         return Adapter(tensors, alpha)
 470
 471     network_alpha = None
 472     if "alpha" in header and not has_alpha_tensors:
 473         try:
 474             network_alpha = float(header["alpha"])
 475         except ValueError:  # diffusers warns and loads with alpha == rank
 476             network_alpha = None
 477         else:
 478             check_alpha("the file header's `alpha`", network_alpha)
 479     return Adapter(tensors, {m: (rank if network_alpha is None else network_alpha) for m, rank in ranks.items()})
```
📍 [`src/h3mlx/lora.py:492-513`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L492-L513)
```python
 492 def config(adapter: Adapter) -> dict:
 493     """The `LoraConfig` arguments that describe the adapter, as `get_peft_kwargs` derives them: the most common rank and
 494     alpha, and a `^`-anchored pattern entry for each module that differs. The other entries state what the port
 495     implements (no dropout, DoRA, rsLoRA or LoRA bias). A subset of what `LoraConfig.to_dict()` holds, so it loads
 496     with `LoraConfig(**config)` on PEFT versions that add or drop other fields."""
 497     ranks = {m: adapter.rank(m) for m in adapter.modules}
 498     alphas = {m: _number(adapter.alpha[m]) for m in adapter.modules}
 499     r = Counter(ranks.values()).most_common()[0][0]
 500     lora_alpha = Counter(alphas.values()).most_common()[0][0]
 501     return {
 502         "r": r,
 503         "lora_alpha": lora_alpha,
 504         "rank_pattern": {f"^{m}": v for m, v in ranks.items() if v != r},
 505         "alpha_pattern": {f"^{m}": v for m, v in alphas.items() if v != lora_alpha},
 506         "target_modules": adapter.modules,
 507         "bias": "none",
 508         "lora_dropout": 0.0,
 509         "use_dora": False,
 510         "use_rslora": False,
 511         "lora_bias": False,
 512         "fan_in_fan_out": False,
 513     }
```
📍 [`src/h3mlx/lora.py:516-533`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L516-L533)
```python
 516 def save(path, adapter: Adapter, prefix: str | None = "transformer") -> None:
 517     """Write the adapter as diffusers does: `<prefix>.<module>.lora_A/B.weight` tensors and, in the safetensors header,
 518     `format: pt` plus the `LoraConfig` arguments as JSON under `lora_adapter_metadata` (each key prefixed too). That is
 519     `save_lora_weights`' file, which `load_lora_adapter` reads by default. `prefix=None` writes `save_lora_adapter`'s:
 520     no prefixes, read back with `load_lora_adapter(..., prefix=None)`. Tensors keep their dtype. Nothing is written
 521     for an adapter with an alpha that is not positive and finite."""
 522     path = Path(path)
 523     if path.suffix != ".safetensors":
 524         raise ValueError(f"expected a .safetensors path, got {path}")
 525     if not adapter.alpha:
 526         raise ValueError("no LoRA tensors in this adapter")
 527     for module in adapter.modules:
 528         check_alpha(module, adapter.alpha[module])
 529     lead = "" if prefix is None else prefix + "."
 530     tensors = {lead + k: v for k, v in adapter.tensors.items()}
 531     meta = {lead + k: v for k, v in config(adapter).items()}
 532     described = json.dumps(meta, indent=2, sort_keys=True, allow_nan=False)  # strict JSON: no NaN or Infinity tokens
 533     mx.save_safetensors(str(path), tensors, metadata={"format": "pt", METADATA_KEY: described})
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

📍 [`src/h3mlx/lora.py:211-222`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L211-L222)
```python
 211 _FLATTENED = [  # musubi-tuner's `lora_unet_` names, every `.` collapsed to `_`
 212     (r"blocks_(\d+)_attn_(qkv|out)_proj", r"blocks.\1.attn.\2_proj"),
 213     (r"blocks_(\d+)_mlp_fc([12])", r"blocks.\1.mlp.fc\2"),
 214     (r"blocks_(\d+)_adaln_proj_linear", r"blocks.\1.adaln_proj.linear"),
 215     (r"token_refiner_blocks_(\d+)_attn_(qkv|out)_proj", r"token_refiner.blocks.\1.attn.\2_proj"),
 216     (r"token_refiner_blocks_(\d+)_mlp_fc([12])", r"token_refiner.blocks.\1.mlp.fc\2"),
 217     (r"(video|audio)_patch_proj", r"\1_patch_proj"),
 218     (r"condition_proj", "condition_proj"),
 219     (r"time_embedder_proj_(in|out)", r"time_embedder.proj_\1"),
 220     (r"final_layer_adaln_proj_linear", "final_layer.adaln_proj.linear"),
 221     (r"final_layer_(video|audio)_out", r"final_layer.\1_out"),
 222 ]
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

📍 [`src/h3mlx/lora.py:299-308`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L299-L308)
```python
 299         alpha = sd.pop(base + ".alpha", None)
 300         if alpha is not None:
 301             if not alpha.item() > 0 or math.isinf(alpha.item()):
 302                 raise ValueError(f"`{base}.alpha` is {alpha.item()}: a LoRA alpha must be positive and finite")
 303             # the forward scales by alpha / rank; split that factor between down and up
 304             scale_down, scale_up = alpha.item() / down.shape[0], 1.0
 305             while scale_down * 2 < scale_up:
 306                 scale_down *= 2
 307                 scale_up /= 2
 308             down, up = ops.times_scalar(down, scale_down), ops.times_scalar(up, scale_up)
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

📍 [`src/h3mlx/lora.py:329-347`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L329-L347)
```python
 329             qkv = pull(f"{source}.attn.qkv_proj")
 330             if qkv is not None:
 331                 down, up = qkv
 332                 if up.shape[0] % 3 != 0:
 333                     raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not divisible "
 334                                      "by 3. This is not a fused MiniMax-H3 QKV projection.")
 335                 if is_diffsynth:
 336                     # [head0: q k v, head1: q k v, ...] -> [q_all; k_all; v_all]
 337                     if up.shape[0] % (3 * head_dim) != 0:
 338                         raise ValueError(f"`{source}.attn.qkv_proj` has {up.shape[0]} output rows, which is not a "
 339                                          f"multiple of 3 * {head_dim}. This is not a per-head-interleaved MiniMax-H3 "
 340                                          "QKV projection.")
 341                     num_heads = up.shape[0] // (3 * head_dim)
 342                     grouped = up.reshape(num_heads, 3 * head_dim, up.shape[1])
 343                     up = mx.concatenate([s.reshape(num_heads * head_dim, up.shape[1])
 344                                          for s in mx.split(grouped, 3, axis=1)], axis=0)
 345                 for proj, up_proj in zip(("to_q", "to_k", "to_v"), mx.split(up, 3, axis=0)):
 346                     out[f"{target}.attn.{proj}.lora_A.weight"] = down
 347                     out[f"{target}.attn.{proj}.lora_B.weight"] = up_proj
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

📍 [`src/h3mlx/lora.py:351-359`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L351-L359)
```python
 351             fc1 = pull(f"{source}.mlp.fc1")
 352             if fc1 is not None:
 353                 down, up = fc1
 354                 if up.shape[0] % 2 != 0:
 355                     raise ValueError(f"`{source}.mlp.fc1` has {up.shape[0]} output rows, which is not even. This is "
 356                                      "not a fused MiniMax-H3 SwiGLU projection.")
 357                 up_gate, up_value = mx.split(up, 2, axis=0)
 358                 out[f"{target}.ff.net.0.proj.lora_A.weight"] = down
 359                 out[f"{target}.ff.net.0.proj.lora_B.weight"] = mx.concatenate([up_value, up_gate], axis=0)
```
📍 [`src/h3mlx/lora.py:223-232`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/lora.py#L223-L232)
```python
 223 _STANDALONE = {  # the projections outside the block stack
 224     "video_patch_proj": "proj_in",
 225     "audio_patch_proj": "audio_proj_in",
 226     "condition_proj": "context_embedder",
 227     "time_embedder.proj_in": "time_embedder.linear_1",
 228     "time_embedder.proj_out": "time_embedder.linear_2",
 229     "final_layer.adaln_proj.linear": "norm_out.linear",
 230     "final_layer.video_out": "proj_out",
 231     "final_layer.audio_out": "audio_proj_out",
 232 }
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

- **Converted, then refused at attach.** The converter also maps LoRAs on the AdaLN projections and on the patch,
  condition and final projections, as theirs does. Our forward runs no LoRA there (§1.3), so `attach` refuses such an
  adapter instead of dropping part of it. The mixin's comment in §1.5 says published H3 LoRAs carry rank-16 LoRAs on
  the AdaLN projections: those files convert here but do not attach. Wiring `adaln_proj` through `plinear`, and through
  the modulation cache, is not done.

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
| `test_refusals_shared_with_diffusers`, `test_bad_alpha_is_refused` (×4), `test_half_a_pair_names_the_module` (×2), `test_dora_scale_in_an_original_file_is_refused`, `test_unwired_converted_modules_are_refused_at_attach`, `test_layout_detection` | the errors both sides raise; the three ours raises where theirs loops, raises a bare KeyError or drops keys; attach's refusal; which keys count as original |

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

📍 [`src/h3mlx/train.py:25-49`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L25-L49)
```python
  25 def scale_noise(sample: mx.array, timestep, noise: mx.array) -> mx.array:
  26     """`MiniMaxH3Scheduler.scale_noise`: `x_t = t * x_0 + (1 - t) * noise`, `t` in [0, 1] and 1 clean.
  27
  28     The timestep's kind decides the arithmetic, as in the scheduler:
  29
  30     * A number (a Python float or int, or a numpy scalar) takes `1 - t` in its own precision before it meets the
  31       noise: double for a Python float (in float32, `1 - 0.999` comes out 1.3e-5 relative too small), float32 for an
  32       `np.float32`. Each product is then tensor-times-number.
  33     * An `mx.array` is cast to the sample's dtype and lined up with its leading axes (a 0-d `t`, or one per example),
  34       and `1 - t` is taken in that dtype, which is how the pipelines noise their conditioning rows.
  35     * A numpy array is a TypeError, as it is for the scheduler: wrap it, `mx.array(timesteps)`, to have it treated as
  36       a tensor (the timesteps `1 - sigma` from `sample_sigmas` are numpy).
  37
  38     One difference, on purpose: a timestep with more axes than the sample is refused. The scheduler broadcasts it into
  39     a result larger than the sample."""
  40     if isinstance(timestep, np.ndarray):
  41         raise TypeError("a numpy array of timesteps is not accepted (nor by the scheduler): pass a number, or "
  42                         "mx.array(timesteps) to have them treated as a tensor")
  43     if not isinstance(timestep, mx.array):
  44         return ops.times_scalar(sample, float(timestep)) + ops.times_scalar(noise, float(1.0 - timestep))
  45     if timestep.ndim > sample.ndim:
  46         raise ValueError(f"a timestep of shape {timestep.shape} has more axes than the sample {sample.shape}")
  47     t = timestep.astype(sample.dtype)
  48     t = t.reshape(t.shape + (1,) * (sample.ndim - t.ndim))
  49     return t * sample + (1.0 - t) * noise
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
📍 [`src/h3mlx/ops.py:322-330`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/ops.py#L322-L330)
```python
 322 def flow_step(v: mx.array, t: float, x: mx.array, sigma: float, sigma_next: float) -> mx.array:
 323     """x0 = x + (1 - t) * v (sigma from the timestep, in x's dtype); x_next = r*x + (1 - r)*x0, r = s_next / s, f32."""
 324     st = (mx.array(1.0, dtype=x.dtype) - mx.array(t, dtype=x.dtype)).astype(x.dtype)
 325     x0 = x + st * v
 326     cdt = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
 327     s = mx.array(sigma, dtype=cdt)
 328     sn = mx.array(sigma_next, dtype=cdt)
 329     r = sn / s
 330     return (r * x.astype(cdt) + (1.0 - r) * x0.astype(cdt)).astype(x.dtype)
```

With `x_t = t·x_0 + (1 − t)·noise` and `x_0 = x_t + (1 − t)·v`, the model's output must satisfy
`(1 − t)·v = (1 − t)·(x_0 − noise)`. So the target is `x_0 − noise` at every `t`.

📍 [`src/h3mlx/train.py:52-56`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L52-L56)
```python
  52 def velocity_target(sample: mx.array, noise: mx.array) -> mx.array:
  53     """What the transformer is trained to predict for `scale_noise(sample, t, noise)`, at every `t`: `x_0 - noise`.
  54     From the scheduler's `x_0 = x_t + (1 - t) * v` with `x_t = t * x_0 + (1 - t) * noise`: `(1 - t) * v` must equal
  55     `(1 - t) * (x_0 - noise)`. Not `noise - x_0`."""
  56     return sample - noise
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

📍 [`src/h3mlx/train.py:59-82`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L59-L82)
```python
  59 def flow_match_loss(pred: mx.array, target: mx.array, weight=None) -> mx.array:
  60     """`mean(weight * (pred - target) ** 2)` over every element, in float32 whatever the inputs' dtypes. `weight` is a
  61     number, or an array lined up with the leading axes (one weight per example, from `loss_weighting_for_sd3`).
  62
  63     Pass the rows the model is trained on and nothing else: the generated video rows, picked by the layout as `denoise`
  64     picks them (`pred[:, layout.num_condition_video_rows:]`). Condition rows are inputs. Rows can't be told apart by
  65     their timestep index: at `t >= 0.999` a generated row and a condition row share one.
  66
  67     `pred` and `target` must have the same shape, and each axis of `weight` must be 1 or the size of the leading axis
  68     it lines up with. torch's `mse_loss` broadcasts a target of another shape with a warning; here it is an error,
  69     because a prediction that still holds its condition rows would be averaged against a one-row target without a
  70     word."""
  71     if tuple(pred.shape) != tuple(target.shape):
  72         raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ in shape: pass the same "
  73                          "rows of both")
  74     err = pred.astype(mx.float32) - target.astype(mx.float32)
  75     sq = err * err
  76     if weight is not None:
  77         w = weight.astype(mx.float32) if isinstance(weight, mx.array) else mx.array(weight, dtype=mx.float32)
  78         if w.ndim > sq.ndim or any(a not in (1, b) for a, b in zip(w.shape, sq.shape)):
  79             raise ValueError(f"a weight of shape {tuple(w.shape)} does not line up with the leading axes of "
  80                              f"{tuple(sq.shape)}")
  81         sq = w.reshape(w.shape + (1,) * (sq.ndim - w.ndim)) * sq
  82     return mx.mean(sq)
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

📍 [`src/h3mlx/train.py:101-127`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L101-L127)
```python
 101 def density_for_timestep_sampling(weighting_scheme: str, batch_size: int, logit_mean: float | None = None,
 102                                   logit_std: float | None = None, mode_scale: float | None = None,
 103                                   generator=None) -> np.ndarray:
 104     """diffusers' `compute_density_for_timestep_sampling`: `batch_size` values `u` in [0, 1], float32.
 105
 106     * `"logit_normal"`: the sigmoid of a normal draw (`logit_mean`, `logit_std`).
 107     * `"mode"`: `1 - u - mode_scale * (cos(pi * u / 2) ** 2 - 1 + u)` of a uniform draw.
 108     * anything else: a uniform draw.
 109
 110     The draw is torch's, from `generator` (a `torch.Generator`; None is torch's global one), so it is the same stream
 111     as theirs. The arithmetic after it is numpy float32 in their order: equal to theirs bit for bit except for the
 112     sigmoid's `exp` and the `cos`, which numpy and torch round differently in the last place or two.
 113
 114     A scheme's own parameters are required, as in theirs (TypeError), and must be finite, which theirs does not check
 115     (ValueError here). A negative or NaN `logit_std` is torch's own RuntimeError on both sides."""
 116     import torch
 117
 118     if weighting_scheme == "logit_normal":
 119         _density_parameters(weighting_scheme, logit_mean=logit_mean, logit_std=logit_std)
 120         u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator).numpy()
 121         return (np.float32(1) / (np.float32(1) + np.exp(-u))).astype(np.float32)
 122     if weighting_scheme == "mode":
 123         _density_parameters(weighting_scheme, mode_scale=mode_scale)
 124         u = torch.rand(size=(batch_size,), generator=generator).numpy()
 125         c = np.cos(np.float32(math.pi) * u / np.float32(2))
 126         return (np.float32(1) - u - np.float32(mode_scale) * (c * c - np.float32(1) + u)).astype(np.float32)
 127     return torch.rand(size=(batch_size,), generator=generator).numpy()
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

📍 [`src/h3mlx/train.py:130-154`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L130-L154)
```python
 130 def loss_weighting_for_sd3(weighting_scheme: str, sigmas) -> np.ndarray:
 131     """diffusers' `compute_loss_weighting_for_sd3`, on float32 or float64 noise levels (numpy, or an `mx.array`).
 132
 133     * `"sigma_sqrt"`: `sigma ** -2`, which torch computes as `1 / (sigma * sigma)`; float32 (infinite at sigma 0).
 134     * `"cosmap"`: `2 / (pi * (1 - 2 * sigma + 2 * sigma ** 2))`, in the input's dtype.
 135     * anything else: ones.
 136
 137     `sigma` is the noise level in both conventions (1 is pure noise), so nothing changes for this model's `1 - t`.
 138
 139     Half-precision and integer levels are refused. Theirs computes a bf16 tensor's weighting in bf16, three digits of
 140     it; the numpy result here can't hold bf16, and upcasting without saying so would be neither theirs nor exact."""
 141     if isinstance(sigmas, mx.array):
 142         if sigmas.dtype not in (mx.float32, mx.float64):
 143             raise ValueError(f"sigmas must be float32 or float64, got {sigmas.dtype}: cast them first")
 144         sigmas = np.array(sigmas)
 145     s = np.asarray(sigmas)
 146     if s.dtype not in (np.float32, np.float64):
 147         raise ValueError(f"sigmas must be float32 or float64, got {s.dtype}: cast them first")
 148     one, two = s.dtype.type(1), s.dtype.type(2)
 149     if weighting_scheme == "sigma_sqrt":
 150         return (one / (s * s)).astype(np.float32)
 151     if weighting_scheme == "cosmap":
 152         bot = one - two * s + two * (s * s)
 153         return (two / (s.dtype.type(math.pi) * bot)).astype(s.dtype)
 154     return np.ones_like(s)
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
📍 [`src/h3mlx/ops.py:309-319`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/ops.py#L309-L319)
```python
 309 def flow_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
 310     """linspace(1, 0, n) in float32, exponential shift, consecutive duplicates collapsed. Terminal 0 included."""
 311     if num_inference_steps is None or num_inference_steps < 2:
 312         raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
 313     if shift <= 0:
 314         raise ValueError(f"shift must be positive, got {shift}")
 315     base = torch_linspace_f32(1.0, 0.0, num_inference_steps)
 316     shift32 = np.float32(shift)
 317     sig = (shift32 * base / (np.float32(1) + (shift32 - np.float32(1)) * base)).astype(np.float32)
 318     keep = np.concatenate([[True], sig[1:] != sig[:-1]])
 319     return sig[keep]
```

📍 [`src/h3mlx/train.py:157-166`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L157-L166)
```python
 157 def training_sigmas(num_points: int, shift: float) -> np.ndarray:
 158     """`num_points` noise levels to train at, from 1.0 (pure noise) down, float32: the scheduler's own shifted grid
 159     for `num_points + 1` points (`ops.flow_sigmas`: `linspace(1, 0, n)` through `s * sigma / (1 + (s - 1) * sigma)`)
 160     without its last point, sigma 0, where the sample is clean and there is nothing to predict. Both numbers are the
 161     caller's: this module has no default shift or grid size."""
 162     grid = ops.flow_sigmas(num_points + 1, shift)[:-1]
 163     if len(grid) != num_points:
 164         raise ValueError(f"shift {shift} merges neighbouring float32 levels of a {num_points}-point grid "
 165                          f"({len(grid)} distinct): ask for fewer points")
 166     return grid
```
📍 [`src/h3mlx/train.py:169-186`](https://github.com/sublimator/minimax-h3-mlx/blob/0b3ec7d885761229a8e5c8a13942235587e0b497/src/h3mlx/train.py#L169-L186)
```python
 169 def sample_sigmas(u, grid) -> tuple[np.ndarray, np.ndarray]:
 170     """One noise level per `u`: `grid[floor(u * len(grid))]`, the multiplication in float32 (an SD3-style trainer's
 171     `(u * num_train_timesteps).long()` lookup). Returns (sigmas, indices). `u = 1.0` would index one past the end,
 172     which torch refuses; it is given the last level here (the logit-normal sigmoid rounds to 1.0 for a large draw, and
 173     `"mode"` maps a draw of 0 to 1.0). The model's timestep for a level is `1 - sigma` in float32; `scale_noise` takes
 174     those as `mx.array(1 - sigmas)`.
 175
 176     Every `u` must lie in [0, 1], which a NaN does not: cast to an index a NaN is 0 on this hardware, pure noise for
 177     every such draw, here and in the torch lookup. The grid must be one non-empty row of levels."""
 178     grid = np.asarray(grid)
 179     if grid.ndim != 1 or grid.size == 0:
 180         raise ValueError(f"the grid must be one non-empty row of noise levels, got shape {grid.shape}")
 181     u = np.asarray(u, dtype=np.float32)
 182     inside = (u >= 0) & (u <= 1)
 183     if not inside.all():
 184         raise ValueError(f"every u must lie in [0, 1] (a NaN does not): got {u[~inside][:4].tolist()}")
 185     indices = np.minimum((u * np.float32(len(grid))).astype(np.int64), len(grid) - 1)
 186     return grid[indices], indices
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

From a seed to sigmas, against the torch chain (diffusers' density, the scheduler's grid, the lookup):

| density | result |
| --- | --- |
| uniform | the same sigmas, bit for bit (`test_same_seed_same_sigmas_uniform`, 40 seeds × 3 grids) |
| `logit_normal`, `mode` | the same level except when `u` sits within a step or two of a boundary: at most 1 of 40,000 draws moved, to the neighbouring level, in each of five cases (`test_same_seed_nearly_same_sigmas_otherwise` allows 20) |

## 5. What each test would catch

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
| unwired targets accepted | `test_refusals` |
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
| a NaN `u` let through the range test; the grid's shape unchecked | `test_sample_sigmas_refuses_nan_and_a_bad_grid` |

Where a row names a dtype, the change is caught only there. A tensor timestep's `1 − t` taken in float64 from the
value before its cast is one such: for a float32 sample it rounds to the same float32 number, because that
subtraction is a single correctly rounded operation either way, and only the bf16 cases fail, where the timestep
must be cast to bf16 before the subtraction.

## 6. Not done, and open

- **No training.** No loop, no gradient test against torch, no memory-bounded backward pass, no optimizer step, no
  latent or text caches, no script, and no run on real weights. The pieces here have not been used together.
- **Two adapters on one `Linear`** (a frozen one beside a trainable one) are refused. PEFT's forward casts the input
  inside its loop over active adapters (§1.1's extract), which will matter then.
- **The example builder.** Which rows are noised, at which timesteps, what the audio rows hold during image training,
  and how condition rows are treated are not decided here. §3.3 says only what the loss needs from it.
- **A step-distilled base.** P-H3-Image samples in 8 steps without guidance. Whether a plain flow-matching loss on it
  keeps that behaviour is untested here.
- **LoRAs on the AdaLN projections** convert but do not attach (§2.4).
- **Density and weighting defaults.** None are chosen. The tests cover the functions, not which settings suit this
  model.
- **bf16.** Every forward test of an adapter here is float32, except `test_adapter_dtypes_follow_peft`, which is one
  forward on four tiny layers. Gradients in bf16 are untested.
- **Real files.** No published adapter was read by this code; the tests write their own. Layout details of real
  files that the converter's docstring does not mention would not be caught.