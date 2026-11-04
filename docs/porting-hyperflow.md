# Porting HyperFlow to MLX: Video Rebirth's loader → `h3mlx`

HyperFlow is Video Rebirth's LoRA for MiniMax H3. It brings the base model's 49 transformer calls per clip down to
**8**. Video Rebirth describe it as data-free flow self-distillation: the base model was the teacher. This guide maps
their loader onto our MLX port, side by side. In each section the original comes first, then the port, then what
changed and which test ties them together. It assumes the main guide (`docs/porting.md`), which covers the base
transformer, the scheduler and the denoise loop.

**Sources, pinned.**

| what | where | pin |
| --- | --- | --- |
| loader (Apache-2.0) | `github.com/Video-Rebirth/hyperflow` | `1dd2f342aba5ab51da02b62885939655e8e268da` |
| weights (MiniMax H3 Community License) | HF `VideoRebirthAI/hyperflow`, `minimax_h3_hyperflow_8step_v1.0.safetensors`, 2,795,328,008 bytes | revision `5bde94d7`, sha256 `9297f450…f447` (its `hyperflow.json`) |
| base model the LoRA was trained on | header `base_model_revision` | `MiniMaxAI/MiniMax-H3` `83db0c0e`; all 14 `transformer_ref` shards have the same LFS oids as our `42ed227` |

**Read, not run.** Their package was read, not executed here. The torch oracle in
the tests is therefore diffusers 0.42.1 and PEFT 0.21.2 doing the injection their `_load_into` does, plus two short
torch transcriptions of their code: `TwoTimeEmbedder.forward` and `build_row_time_pairs` (§8). A reviewer should
compare those transcriptions against the originals extracted below.

**What HyperFlow changes, and nothing else** (their README, "How it works", checked against the code):

| piece | official MiniMax H3 | with HyperFlow |
| --- | --- | --- |
| block and refiner linears (`to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`) | base weight | base + LoRA (rank 256, alpha 256) |
| time embedding | `time_embedder(time_proj(t))` | blend of `time_embedder` at `t` and a LoRA'd copy at the step's endpoint `r`, gate 0.25 |
| sigma grid | `linspace(1, 0, 50)`, shifted: 49 calls | a fixed 9-point raw grid from the file, shifted the same way: 8 calls |
| per-row timesteps | distinct `t` | distinct `(t, r)` pairs; conditioning rows keep `r = t` |
| `adaln_proj`, packing, text encoder, VAEs, Euler update, noise | — | unchanged |

`adaln_proj` carries the reference conditioning in `transformer_ref`, and the adapter never touches it. That is why
one file serves t2va, fl2va and ref2va (header `compatible_subfolders: ["transformer", "transformer_ref"]`).

## 1. The file: header and keys

The weights file is a plain safetensors file. Its keys are PEFT-style:
`transformer.<module>.lora_A.weight` / `lora_B.weight`. Its header carries the gate, the rank, the alpha, the grid and
the shifts. Their loader parses and checks the header, then maps each key onto a module. `endpoint_time_embedder.*` is
not a module of the official transformer: it names the copy of `time_embedder` that their wrapper creates.

📍 [`hyperflow:src/hyperflow_h3/lora.py:66-89`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/lora.py#L66-L89)
```python
  66     @classmethod
  67     def from_dict(cls, metadata: dict[str, str] | None) -> HyperFlowMetadata:
  68         metadata = dict(metadata or {})
  69         missing = [key for key in _REQUIRED_METADATA if key not in metadata]
  70         if missing or metadata.get("hyperflow", "").lower() != "true":
  71             raise ValueError(
  72                 "Not a HyperFlow weights file: the safetensors header must carry "
  73                 f'{list(_REQUIRED_METADATA)} with `hyperflow == "true"`; missing {missing or ["hyperflow=true"]}.'
  74             )
  75         return cls(
  76             version=metadata["hyperflow_version"],
  77             gate=float(metadata["hyperflow_gate"]),
  78             lora_alpha=float(metadata["lora_alpha"]),
  79             base_model=metadata["base_model"],
  80             base_model_revision=metadata.get("base_model_revision"),
  81             lora_rank=int(metadata["lora_rank"]) if "lora_rank" in metadata else None,
  82             sigmas=_json_tuple(metadata.get("hyperflow_sigmas"), float),
  83             video_shift=float(metadata["hyperflow_video_shift"]) if "hyperflow_video_shift" in metadata else None,
  84             audio_shift=float(metadata["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in metadata else None,
  85             trained_on_subfolder=metadata.get("trained_on_subfolder"),
  86             compatible_subfolders=_json_tuple(metadata.get("compatible_subfolders"), str) or (),
  87             tasks=_json_tuple(metadata.get("tasks"), str) or (),
  88             raw=metadata,
  89         )
```
📍 [`hyperflow:src/hyperflow_h3/lora.py:170-176`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/lora.py#L170-L176)
```python
 170 def _public_to_module_path(public_module: str) -> str:
 171     """Map a public key's module path onto the transformer with the [`TwoTimeEmbedder`] installed."""
 172     if public_module.startswith("endpoint_time_embedder."):
 173         return "time_embedder.endpoint." + public_module[len("endpoint_time_embedder.") :]
 174     if public_module.startswith("time_embedder."):
 175         return "time_embedder.base." + public_module[len("time_embedder.") :]
 176     return public_module
```
📍 [`hyperflow:src/hyperflow_h3/lora.py:179-194`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/lora.py#L179-L194)
```python
 179 def plan_from_keys(keys: Iterable[str]) -> dict[str, dict[str, str]]:
 180     """Group the file's keys by target module: ``{module_path: {"A": key, "B": key}}``."""
 181     plan: dict[str, dict[str, str]] = {}
 182     for key in keys:
 183         match = _KEY_RE.match(key)
 184         if match is None:
 185             raise ValueError(
 186                 f"Unexpected key {key!r} in a HyperFlow weights file. Only "
 187                 "`transformer.<module>.lora_A.weight` / `lora_B.weight` keys are allowed."
 188             )
 189         module_path = _public_to_module_path(match["module"])
 190         plan.setdefault(module_path, {})[match["matrix"]] = key
 191     incomplete = [module for module, matrices in plan.items() if set(matrices) != {"A", "B"}]
 192     if incomplete:
 193         raise ValueError(f"Modules with only one of lora_A / lora_B: {incomplete[:8]}")
 194     return plan
```

Ours parses the same header with the same required fields, and keeps the file's public module names. The endpoint
embedder is not a sub-module in MLX. It is a second name prefix in the flat weight dict (§2), and `_base_name` says
which base `Linear` each LoRA sits on. `check_plan` adds the shape checks their PEFT injection performs implicitly.
Every target must be a 2-D `Linear` weight that its LoRA fits, all LoRAs must share one rank, and the scaling must be
1 (§2).

📍 `src/h3mlx/hyperflow.py:53-77`
```python
  53     @classmethod
  54     def from_header(cls, md: dict | None) -> "HyperFlowMeta":
  55         md = dict(md or {})
  56         missing = [k for k in REQUIRED if k not in md]
  57         if missing or md.get("hyperflow", "").lower() != "true":
  58             raise ValueError(f"not a HyperFlow file: the header needs {list(REQUIRED)} with hyperflow == 'true'; "
  59                              f"missing {missing or ['hyperflow=true']}")
  60
  61         def lst(key, cast):
  62             if key not in md:
  63                 return None
  64             v = json.loads(md[key])
  65             if not isinstance(v, list):
  66                 raise ValueError(f"header {key}: expected a JSON list, got {md[key]!r}")
  67             return tuple(cast(x) for x in v)
  68
  69         return cls(
  70             version=md["hyperflow_version"], gate=float(md["hyperflow_gate"]), lora_alpha=float(md["lora_alpha"]),
  71             base_model=md["base_model"], base_model_revision=md.get("base_model_revision"),
  72             lora_rank=int(md["lora_rank"]) if "lora_rank" in md else None,
  73             sigmas=lst("hyperflow_sigmas", float) or DEFAULT_SIGMAS_8STEP,
  74             video_shift=float(md["hyperflow_video_shift"]) if "hyperflow_video_shift" in md else None,
  75             audio_shift=float(md["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in md else None,
  76             compatible_subfolders=lst("compatible_subfolders", str) or (), raw=md,
  77         )
```
📍 `src/h3mlx/hyperflow.py:92-103`
```python
  92 def plan_from_keys(keys) -> dict:
  93     """{module: {"A": key, "B": key}} in the file's own (public) module names; anything else is an error."""
  94     plan = {}
  95     for key in keys:
  96         m = KEY_RE.match(key)
  97         if m is None:
  98             raise ValueError(f"unexpected key {key!r}: only transformer.<module>.lora_A/B.weight keys are allowed")
  99         plan.setdefault(m["module"], {})[m["matrix"]] = key
 100     incomplete = [k for k, v in plan.items() if set(v) != {"A", "B"}]
 101     if incomplete:
 102         raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
 103     return plan
```
📍 `src/h3mlx/hyperflow.py:106-108`
```python
 106 def _base_name(module: str) -> str:
 107     """The transformer Linear a module's LoRA sits on: the endpoint embedder is a copy of `time_embedder`."""
 108     return BASE_TIME + module[len(ENDPOINT):] if module.startswith(ENDPOINT) else module
```
📍 `src/h3mlx/hyperflow.py:111-134`
```python
 111 def check_plan(plan: dict, shapes: dict, base_shapes: dict, meta: HyperFlowMeta) -> int:
 112     """Every LoRA lands on a Linear weight of the right shape, one rank throughout, scaling 1. `shapes` maps the file's
 113     keys to their shapes, `base_shapes` our parameter names to theirs. Returns the rank."""
 114     no_base = [m for m in plan if _base_name(m) + ".weight" not in base_shapes]
 115     if no_base:
 116         raise KeyError(f"{len(no_base)} LoRA target(s) have no Linear weight in this transformer, e.g. {no_base[:5]}")
 117     not_linear = [m for m in plan if len(base_shapes[_base_name(m) + ".weight"]) != 2]
 118     if not_linear:
 119         raise ValueError(f"LoRA target(s) whose weight is not a 2-D Linear weight: {not_linear[:5]}")
 120     ranks = set()
 121     for module, keys in plan.items():
 122         a, b = tuple(shapes[keys["A"]]), tuple(shapes[keys["B"]])
 123         out_f, in_f = tuple(base_shapes[_base_name(module) + ".weight"])
 124         if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
 125             raise ValueError(f"{module}: lora_A {a} / lora_B {b} do not fit the Linear ({out_f}, {in_f})")
 126         ranks.add(a[0])
 127     if len(ranks) != 1:
 128         raise ValueError(f"all LoRA targets must share one rank, found {sorted(ranks)}")
 129     rank = ranks.pop()
 130     if meta.lora_rank is not None and meta.lora_rank != rank:
 131         raise ValueError(f"header says lora_rank={meta.lora_rank} but the tensors have rank {rank}")
 132     if meta.lora_alpha / rank != 1.0:
 133         raise NotImplementedError(f"LoRA scaling alpha/rank = {meta.lora_alpha}/{rank} != 1 (ops.plinear assumes 1)")
 134     return rank
```

**The released v1.0 header**, fetched with an HTTP range request; the full header is in
`tests/data/hyperflow_v1.0.header.json`:

- **Tensors:** 632, covering 316 modules: 6 per block × 50 blocks, 6 per refiner block × 2 refiner blocks, and two
  linears each for `time_embedder` and `endpoint_time_embedder`.
- **LoRA:** rank 256, alpha 256, so scaling 1. Gate 0.25.
- **Grid:** `[1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]`.
- **Shifts:** video 12, audio 3.
- **Dtypes:** bf16 for the blocks, f32 for the time embedders.

`test_real_header_fits_transformer_ref` checks every LoRA against the real `transformer_ref` shapes, building the
model on torch's meta device. It also checks that every target is a released tensor and that none is an `adaln_proj`.

## 2. Injection: PEFT on their side, `plinear` on ours

Their loader installs the two-time wrapper, then lets PEFT wrap every target `nn.Linear`, then copies the tensors in.
Each tensor is cast to the dtype of the parameter PEFT created.

📍 [`hyperflow:src/hyperflow_h3/lora.py:288-363`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/lora.py#L288-L363)
```python
 288 def _load_into(
 289     transformer: nn.Module,
 290     handle: Any,
 291     plan: dict[str, dict[str, str]],
 292     metadata: HyperFlowMetadata,
 293     *,
 294     gate: float,
 295 ) -> None:
 296     try:
 297         from peft import LoraConfig
 298     except ImportError as error:  # pragma: no cover - import guard
 299         raise RuntimeError("Loading the HyperFlow LoRA requires `peft` (`pip install peft`).") from error
 300
 301     existing_adapters = sorted((getattr(transformer, "peft_config", None) or {}).keys())
 302     if existing_adapters:
 303         raise ValueError(
 304             f"This transformer already has PEFT adapter(s) {existing_adapters}. HyperFlow owns one fixed adapter and "
 305             "cannot be loaded alongside or on top of another adapter; use a fresh base model."
 306         )
 307
 308     embedder = TwoTimeEmbedder.wrap(transformer, gate)
 309     embedder.gate = gate
 310
 311     modules = dict(transformer.named_modules())
 312     missing_modules = [path for path in plan if not isinstance(modules.get(path), nn.Linear)]
 313     if missing_modules:
 314         raise KeyError(
 315             f"{len(missing_modules)} LoRA target(s) in the file have no `nn.Linear` on this transformer, e.g. "
 316             f"{missing_modules[:5]}. The file does not match this model architecture."
 317         )
 318
 319     ranks = {int(handle.get_slice(matrices["A"]).get_shape()[0]) for matrices in plan.values()}
 320     if len(ranks) != 1:
 321         raise ValueError(f"All LoRA targets must share one rank, found {sorted(ranks)}.")
 322     rank = ranks.pop()
 323     if metadata.lora_rank is not None and metadata.lora_rank != rank:
 324         raise ValueError(f"Header says lora_rank={metadata.lora_rank} but the tensors have rank {rank}.")
 325
 326     config = LoraConfig(
 327         r=rank,
 328         lora_alpha=metadata.lora_alpha,
 329         lora_dropout=0.0,
 330         bias="none",
 331         target_modules=sorted(plan),
 332         init_lora_weights=True,
 333     )
 334     transformer.add_adapter(config, adapter_name=_ADAPTER_NAME)
 335
 336     parameters = dict(transformer.named_parameters())
 337     expected = {
 338         name for name in parameters if f".lora_A.{_ADAPTER_NAME}." in name or f".lora_B.{_ADAPTER_NAME}." in name
 339     }
 340     state: dict[str, torch.Tensor] = {}
 341     for module_path, matrices in plan.items():
 342         for matrix, key in matrices.items():
 343             state[f"{module_path}.lora_{matrix}.{_ADAPTER_NAME}.weight"] = handle.get_tensor(key)
 344
 345     missing = sorted(expected - state.keys())
 346     unexpected = sorted(state.keys() - expected)
 347     if missing or unexpected:
 348         raise ValueError(
 349             "The injected adapter and the file disagree: "
 350             f"{len(missing)} parameter(s) without a tensor {missing[:4]}, "
 351             f"{len(unexpected)} tensor(s) without a parameter {unexpected[:4]}."
 352         )
 353     for name, tensor in state.items():
 354         if tuple(parameters[name].shape) != tuple(tensor.shape):
 355             raise ValueError(
 356                 f"Shape mismatch for {name}: model {tuple(parameters[name].shape)}, file {tuple(tensor.shape)}."
 357             )
 358
 359     with torch.no_grad():
 360         for name, tensor in state.items():
 361             parameter = parameters[name]
 362             parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
 363             parameter.requires_grad_(False)
```

What a PEFT-wrapped `Linear` computes, from the installed PEFT 0.21.2:

📍 PEFT 0.21.2, `peft/tuners/lora/layer.py:1074-1115`
```python
1074     def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
1075         self._check_forward_args(x, *args, **kwargs)
1076         adapter_names = kwargs.pop("adapter_names", None)
1077         variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer
1078
1079         if self.disable_adapters:
1080             if self.merged:
1081                 self.unmerge()
1082             result = self.base_layer(x, *args, **kwargs)
1083         elif adapter_names is not None:
1084             result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
1085         elif self.merged:
1086             result = self.base_layer(x, *args, **kwargs)
1087         else:
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
1114
1115         return result
```

Two facts set the port's arithmetic:

1. **The adapter is in the base layer's dtype.** This was probed with `add_adapter` on a tiny bf16 model with the
   release's f32 modules: block and refiner LoRAs come out bf16, time-embedder LoRAs f32.
2. **The input is cast to the adapter's dtype.** PEFT's `_cast_input_dtype` is `x.to(dtype)` unless the dtypes already
   match. The two products each round in that dtype, the scaled delta is added to the base result, and the sum is
   cast back to the base result's dtype.

`plinear` does the same. The `* scaling` is left out because `check_plan` refuses any file whose alpha/rank isn't
exactly 1, and multiplying by 1.0 is exact.

📍 `src/h3mlx/ops.py:67-77`
```python
  67 def plinear(x: mx.array, p: dict, name: str, bias: bool = False) -> mx.array:
  68     """The Linear `name` of state dict `p`, plus its LoRA when `name.lora_A.weight` / `name.lora_B.weight` are present.
  69
  70     PEFT's lora.Linear as HyperFlow loads it: the adapter is in the base layer's dtype, the input is cast to it, the
  71     two products round in that dtype, and the sum is cast back to the base result's dtype. Scaling (alpha / rank) must
  72     be 1, so the `* scaling` is exact and left out (`hyperflow.attach` checks it)."""
  73     y = linear(x, p[name + ".weight"], p[name + ".bias"] if bias else None)
  74     a = p.get(name + ".lora_A.weight")
  75     if a is None:
  76         return y
  77     return (y + linear(linear(x.astype(a.dtype), a), p[name + ".lora_B.weight"])).astype(y.dtype)
```

`attach` builds the new weight dict:

- the endpoint embedder shares the base embedder's arrays, so it costs no memory beyond its own LoRA;
- each LoRA is cast to its base weight's dtype;
- the gate goes into `cfg`;
- the input dict is left unmodified, and attaching twice is refused.

`load` reads the file lazily with `mx.load`, like the transformer.

📍 `src/h3mlx/hyperflow.py:137-153`
```python
 137 def attach(params: dict, cfg: dict, tensors: dict, meta: HyperFlowMeta, subfolder: str = "transformer_ref"):
 138     """Return (params, cfg) with HyperFlow attached; the inputs are not modified. Adapter tensors are cast to their
 139     base weight's dtype, as PEFT's injection does (bf16 on the blocks, f32 on the time embedder)."""
 140     if meta.compatible_subfolders and subfolder not in meta.compatible_subfolders:
 141         raise ValueError(f"this HyperFlow file lists {meta.compatible_subfolders}, not {subfolder}")
 142     if ENDPOINT + "linear_1.weight" in params:
 143         raise ValueError("HyperFlow is already attached to these parameters")
 144     plan = plan_from_keys(tensors)
 145     check_plan(plan, {k: v.shape for k, v in tensors.items()}, {k: v.shape for k, v in params.items()}, meta)
 146     out = dict(params)
 147     for name in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"):
 148         out[ENDPOINT + name] = params[BASE_TIME + name]  # a copy of the base embedder (shared arrays, no new memory)
 149     for module, keys in plan.items():
 150         dtype = params[_base_name(module) + ".weight"].dtype
 151         out[module + ".lora_A.weight"] = tensors[keys["A"]].astype(dtype)
 152         out[module + ".lora_B.weight"] = tensors[keys["B"]].astype(dtype)
 153     return out, {**cfg, "hyperflow_gate": float(meta.gate)}
```
📍 `src/h3mlx/hyperflow.py:170-176`
```python
 170 def load(path, params: dict, cfg: dict, subfolder: str = "transformer_ref"):
 171     """Load a HyperFlow .safetensors onto our transformer params: (params, cfg, meta)."""
 172     path = Path(path)
 173     tensors, md = mx.load(str(path), return_metadata=True)
 174     meta = HyperFlowMeta.from_header(md)
 175     params, cfg = attach(params, cfg, tensors, meta, subfolder)
 176     return params, cfg, meta
```

The attention, feed-forward and timestep ops now call `plinear` by name instead of `linear`. Without LoRA keys the
result is the same `linear` call as before, so the base model's tests still pin the base path (213 tests pass).

**Not ported.** These are left out:

- Merging the LoRA into the base weights. Unmerged costs about 2.3% more FLOPs per call at 52k rows and about 5.5% at
  6.8k, by the count in §7. Merging in bf16 would also change the rounding.
- `disable_hyperflow` and `enable_hyperflow`, which in their package give same-seed base/HyperFlow pairs from one
  loaded model. In ours, load a second param dict instead.

## 3. Two-time conditioning

The official forward embeds its distinct timesteps in two lines: `temb = self.time_proj(timestep)`, then
`temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))`. That is `transformer_minimax_h3.py`'s
`MiniMaxH3Transformer3DModel.forward`; the main guide's §2 covers it. HyperFlow swaps `time_embedder` for a wrapper:

📍 [`hyperflow:src/hyperflow_h3/embedder.py:52-62`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/embedder.py#L52-L62)
```python
  52     @classmethod
  53     def wrap(cls, transformer: nn.Module, gate: float) -> TwoTimeEmbedder:
  54         """Replace ``transformer.time_embedder`` in place and return the wrapper (idempotent)."""
  55         current = transformer.time_embedder
  56         if isinstance(current, cls):
  57             return current
  58         endpoint = copy.deepcopy(current)
  59         time_proj = copy.deepcopy(transformer.time_proj)
  60         wrapper = cls(current, endpoint, time_proj, gate)
  61         transformer.time_embedder = wrapper
  62         return wrapper
```
📍 [`hyperflow:src/hyperflow_h3/embedder.py:86-104`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/embedder.py#L86-L104)
```python
  86     def forward(self, sample: torch.Tensor) -> torch.Tensor:
  87         t_emb = self.base(sample)
  88         if self.passthrough:
  89             return t_emb
  90         if self._endpoint_timesteps is None:
  91             raise RuntimeError(
  92                 "HyperFlow needs the endpoint timesteps of this forward, but none were set. Run the transformer "
  93                 "through the HyperFlow denoise blocks (`hyperflow_blocks(...)`), or call `disable_hyperflow(...)` "
  94                 "to use it as the plain base model."
  95             )
  96         endpoint = self._endpoint_timesteps.to(device=sample.device)
  97         if endpoint.shape[0] != sample.shape[0]:
  98             raise ValueError(
  99                 f"Got {sample.shape[0]} distinct timesteps but {endpoint.shape[0]} endpoints; every timestep needs "
 100                 "exactly one endpoint. Build the plan with `build_row_time_pairs`."
 101             )
 102         r_proj = self.time_proj(endpoint).to(sample.dtype)
 103         r_emb = self.endpoint(r_proj)
 104         return t_emb + self.gate * (r_emb - t_emb)
```

Points that matter for the port:

- **Same base weights.** `endpoint` is a `deepcopy` of `time_embedder` made *before* any LoRA, so both embedders share
  base weights and differ only in their LoRAs.
- **`r` gets the same sinusoid.** `r` goes through a copy of the same parameter-free `time_proj`, in f32, cast to the
  dtype of `sample` (f32 here).
- **The blend.** It is `t_emb + gate * (r_emb - t_emb)`, with the gate a Python float: 0.25, exact in f32.
- **No silent fallback.** The wrapper refuses to run without endpoints, so the official loop can't silently drive a
  two-time adapter with single times.

Ours is one function. It refuses the mismatch in both directions: HyperFlow weights without endpoints, and endpoints
without HyperFlow weights. Like their wrapper, it also refuses an endpoint count that differs from the distinct
timestep count (MLX would otherwise broadcast a length-1 array silently).

📍 `src/h3mlx/ops.py:176-190`
```python
 176 def two_time_embedding(t: mx.array, r: mx.array | None, p: dict, cfg: dict) -> mx.array:
 177     """The DiT's timestep embedding; with HyperFlow loaded, HyperFlow's TwoTimeEmbedder: the step's start t and its
 178     endpoint r are embedded by `time_embedder` and `endpoint_time_embedder` (same base weights, separate LoRAs) and
 179     blended, emb(t) + gate * (emb_r(r) - emb(t)). Refuses to run a two-time model without endpoints, and the reverse."""
 180     two_time = "endpoint_time_embedder.linear_1.weight" in p
 181     if two_time != (r is not None):
 182         raise ValueError("HyperFlow weights need the step endpoints, and endpoints need HyperFlow weights"
 183                          f" (endpoint embedder loaded: {two_time}, endpoints given: {r is not None})")
 184     if r is not None and r.shape != t.shape:
 185         raise ValueError(f"{t.shape[0]} distinct timesteps but {r.shape[0]} endpoints: one endpoint per timestep")
 186     temb = timestep_embedding(t, _sub(p, "time_embedder."), cfg["freq_dim"])
 187     if r is None:
 188         return temb
 189     r_emb = timestep_embedding(r, _sub(p, "endpoint_time_embedder."), cfg["freq_dim"])
 190     return temb + cfg["hyperflow_gate"] * (r_emb - temb)
```

`dit_forward` takes `endpoint` aligned with its distinct `timestep` values. The AdaLN rows are unchanged
(`timestep_indices * 3 + modality_tag`); each distinct timestep is now a distinct `(t, r)` pair.

📍 `src/h3mlx/ops.py:369-382`
```python
 369     mx.eval(x)  # projections + refiner + scatter as their own graph, before the first block
 370     if cache is None:
 371         temb = two_time_embedding(timestep, endpoint, p, cfg)
 372         t_rows = timestep_indices
 373     else:
 374         if endpoint is not None or "endpoint_time_embedder.linear_1.weight" in p:
 375             raise NotImplementedError("ModulationCache is keyed by t alone; HyperFlow's tables depend on (t, r)")
 376         temb = None
 377         t_rows = cache.rows(timestep, timestep_indices)  # step-local timestep index -> row of the cached table
 378     adaln_indices = t_rows * MODALITY_NUM + token_tags
 379     for i in range(cfg["num_layers"]):
 380         x = dit_block(x, temb, adaln_indices, rotary, _sub(p, f"transformer_blocks.{i}."), heads, hidden, eps, qk_eps,
 381                       modulation=None if cache is None else cache.blocks[i])
 382         mx.eval(x)  # one block's graph at a time: bounds activation memory at 20-50k rows
```

The modulation cache (main guide, §2.6) is keyed by `t` alone. With HyperFlow the tables depend on `(t, r)`, so
`dit_forward` and `schedule_timesteps` raise rather than hand back wrong tables. Extending the cache to pairs is
possible but not done.

## 4. The sigma grid

The grid comes from the file, falling back to the same 9 values hard-coded in the package. It is checked, then shifted
per modality with the official formula. The endpoint of step `i` is `1 − sigma[i+1]`.

📍 [`hyperflow:src/hyperflow_h3/schedule.py:31`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/schedule.py#L31)
```python
  31 DEFAULT_SIGMAS_8STEP: tuple[float, ...] = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
```
📍 [`hyperflow:src/hyperflow_h3/schedule.py:39-52`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/schedule.py#L39-L52)
```python
  39 def validate_sigmas(sigmas: Sequence[float] | torch.Tensor) -> torch.Tensor:
  40     """Return ``sigmas`` as a float32 CPU tensor after checking it is a valid rectified-flow grid.
  41
  42     The grid must be strictly decreasing, start at or below ``1.0`` and end at exactly ``0.0`` — the same contract
  43     ``MiniMaxH3Scheduler.set_timesteps(sigmas=...)`` enforces.
  44     """
  45     grid = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
  46     if grid.numel() < 2:
  47         raise ValueError(f"A sigma grid needs at least two points, got {grid.numel()}.")
  48     if not bool((grid[1:] < grid[:-1]).all()):
  49         raise ValueError(f"The sigma grid must be strictly decreasing, got {grid.tolist()}.")
  50     if grid[0].item() > 1.0 or grid[-1].item() != 0.0:
  51         raise ValueError(f"The sigma grid must start at or below 1.0 and end at exactly 0.0, got {grid.tolist()}.")
  52     return grid
```
📍 [`hyperflow:src/hyperflow_h3/schedule.py:55-63`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/schedule.py#L55-L63)
```python
  55 def shift_sigmas(sigmas: Sequence[float] | torch.Tensor, shift: float) -> torch.Tensor:
  56     """Apply the exponential shift ``s * sigma / (1 + (s - 1) * sigma)``, the formula ``MiniMaxH3Scheduler`` uses.
  57
  58     The shift maps ``0`` to ``0`` and ``1`` to ``1``, so a valid grid stays valid.
  59     """
  60     if shift <= 0:
  61         raise ValueError(f"`shift` must be positive, got {shift}.")
  62     base = validate_sigmas(sigmas)
  63     return shift * base / (1.0 + (shift - 1.0) * base)
```
📍 [`hyperflow:src/hyperflow_h3/schedule.py:66-72`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/schedule.py#L66-L72)
```python
  66 def endpoints_from_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
  67     """The endpoint of every step: ``r_i = 1 - sigma_{i+1}`` on the ``t = 1 - sigma`` axis.
  68
  69     Pairs with ``MiniMaxH3Scheduler.timesteps == 1 - sigmas[:-1]``: step ``i`` moves the generated rows from
  70     ``t_i = 1 - sigma_i`` to ``r_i = 1 - sigma_{i+1}``.
  71     """
  72     return 1.0 - sigmas[1:]
```

The shifted grid is handed to diffusers' own scheduler, which installs an explicit grid verbatim and exposes
`timesteps = 1 − sigmas[:-1]`. The Euler update downstream is the official one, untouched.

📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:124-170`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/schedulers/scheduling_minimax_h3.py#L124-L170)
```python
 124     def set_timesteps(
 125         self,
 126         num_inference_steps: int | None = None,
 127         device: str | torch.device | None = None,
 128         sigmas: list[float] | torch.Tensor | None = None,
 129     ) -> None:
 130         r"""
 131         Build the sigma / timestep schedule.
 132
 133         The grid is `linspace(1, 0, num_inference_steps)` pushed through the exponential shift, with consecutive
 134         duplicates collapsed. The terminal `0` is already part of that grid — the shift maps `0` to exactly `0` — so
 135         the schedule holds `num_inference_steps` sigmas and drives `num_inference_steps - 1` model evaluations, exposed
 136         as `self.timesteps = 1 - sigmas[:-1]`.
 137
 138         Args:
 139             num_inference_steps (`int`, *optional*):
 140                 Number of sigma grid points, terminal `0` included. Ignored when `sigmas` is given.
 141             device (`str` or `torch.device`, *optional*):
 142                 Device the schedule tensors are moved to. The grid itself is always built on CPU in float32 so the
 143                 schedule does not depend on the accelerator.
 144             sigmas (`list[float]` or `torch.Tensor`, *optional*):
 145                 A fully-formed sigma schedule, used verbatim (no shifting, no deduplication). It must be strictly
 146                 decreasing and terminate at `0.0`.
 147         """
 148         if sigmas is None:
 149             if num_inference_steps is None or num_inference_steps < 2:
 150                 raise ValueError(
 151                     "`set_timesteps` requires either an explicit `sigmas` schedule or "
 152                     f"`num_inference_steps` >= 2, got {num_inference_steps}."
 153                 )
 154
 155             # The rectified-flow sigma range is fixed at [1.0, 0.0].
 156             base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
 157             sigmas = self._shift * base / (1 + (self._shift - 1) * base)
 158             # The shift compresses the grid near sigma = 1; collapse any float32 collisions it creates.
 159             sigmas = torch.unique_consecutive(sigmas)
 160         else:
 161             sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
 162             if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
 163                 raise ValueError("`sigmas` must hold at least two strictly decreasing values ending at 0.0.")
 164
 165         self.sigmas = sigmas.to(device=device)
 166         # t = 1 - sigma, and t = 1 is clean. The terminal sigma has no model evaluation.
 167         self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
 168         self.num_inference_steps = int(self.timesteps.numel())
 169         self._step_index = None
 170         self._begin_index = None
```

Ours does the same arithmetic in numpy float32. The scalars 12, 11, 3, 2 and 1 are exact in f32, and each operation is
a single IEEE float32 operation on both sides, so the result is bit-identical. `test_grid_matches_diffusers_scheduler`
checks it against torch's formula and against the scheduler's installed `sigmas` and `timesteps`, for both shifts.

📍 `src/h3mlx/denoise.py:20-28`
```python
  20 def shift_grid(raw, shift: float) -> np.ndarray:
  21     """A raw sigma grid through the exponential shift, `s * sigma / (1 + (s - 1) * sigma)` in float32: HyperFlow's
  22     `validate_sigmas` + `shift_sigmas` (the formula of `MiniMaxH3Scheduler.set_timesteps`)."""
  23     if shift <= 0:
  24         raise ValueError(f"`shift` must be positive, got {shift}")
  25     base = np.asarray(raw, dtype=np.float32).reshape(-1)
  26     if base.size < 2 or not (base[1:] < base[:-1]).all() or base[0] > 1.0 or base[-1] != 0.0:
  27         raise ValueError(f"a sigma grid must be strictly decreasing from at most 1.0 to exactly 0.0, got {base.tolist()}")
  28     return (np.float32(shift) * base / (np.float32(1.0) + np.float32(shift - 1.0) * base)).astype(np.float32)
```
📍 `src/h3mlx/denoise.py:31-60`
```python
  31 @dataclass
  32 class Schedules:
  33     video_sigmas: np.ndarray
  34     audio_sigmas: np.ndarray
  35     two_time: bool = False  # HyperFlow: every step is also conditioned on its endpoint r = 1 - sigma[i + 1]
  36
  37     @classmethod
  38     def make(cls, num_inference_steps: int, video_shift: float = 12.0, audio_shift: float = 3.0):
  39         return cls(ops.flow_sigmas(num_inference_steps, video_shift), ops.flow_sigmas(num_inference_steps, audio_shift))
  40
  41     @classmethod
  42     def from_grid(cls, raw, video_shift: float = 12.0, audio_shift: float = 3.0, two_time: bool = True):
  43         """A fixed raw grid (HyperFlow's 9 points), shifted per modality, used verbatim by the Euler steps."""
  44         return cls(shift_grid(raw, video_shift), shift_grid(raw, audio_shift), two_time)
  45
  46     @property
  47     def video_endpoints(self):
  48         return (np.float32(1) - self.video_sigmas[1:]).astype(np.float32)
  49
  50     @property
  51     def audio_endpoints(self):
  52         return (np.float32(1) - self.audio_sigmas[1:]).astype(np.float32)
  53
  54     @property
  55     def video_timesteps(self):
  56         return (np.float32(1) - self.video_sigmas[:-1]).astype(np.float32)
  57
  58     @property
  59     def audio_timesteps(self):
  60         return (np.float32(1) - self.audio_sigmas[:-1]).astype(np.float32)
```

The shifts come from the pipeline (the release's schedulers: 12 and 3), not from the file. Theirs warns when the file
was trained with different ones (`_check_shifts`), and `HyperFlowMeta.schedules` warns the same way.

📍 `src/h3mlx/hyperflow.py:83-89`
```python
  83     def schedules(self, video_shift: float = 12.0, audio_shift: float = 3.0) -> Schedules:
  84         """The file's grid through the pipeline's shifts (the release's 12 / 3); warns if the file was trained with
  85         others, as HyperFlow does."""
  86         trained = (self.video_shift, self.audio_shift)
  87         if None not in trained and trained != (video_shift, audio_shift):
  88             warnings.warn(f"HyperFlow was trained with shifts {trained}, the pipeline uses {(video_shift, audio_shift)}")
  89         return Schedules.from_grid(self.sigmas, video_shift, audio_shift, two_time=True)
```

## 5. Per-row `(t, r)` pairs

The official `build_row_timesteps` gives every row a `t` and reduces the rows to the distinct values (main guide,
§4). HyperFlow gives every row a pair:

- generated video rows and text rows: `(t_v, r_v)`;
- generated audio rows: `(t_a, r_a)`;
- visual conditioning rows: `(max(t_v, 0.999), same)`;
- audio reference rows: `(1, 1)`.

The pairs are reduced with `torch.unique(dim=0, sorted=True)`, which sorts them lexicographically.

📍 [`hyperflow:src/hyperflow_h3/schedule.py:75-126`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/schedule.py#L75-L126)
```python
  75 def build_row_time_pairs(
  76     *,
  77     video_indices: torch.Tensor,
  78     audio_indices: torch.Tensor,
  79     num_condition_video_rows: int,
  80     num_condition_audio_rows: int,
  81     num_text_tokens: int,
  82     video_timestep: float,
  83     video_endpoint: float,
  84     audio_timestep: float,
  85     audio_endpoint: float,
  86     condition_video_timestep: float,
  87     condition_audio_timestep: float = 1.0,
  88 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  89     """Assign a ``(t, r)`` pair to every row of the packed sequence and reduce it to the transformer's inputs.
  90
  91     This is the official ``MiniMaxH3SetTimestepsStep.build_row_timesteps`` extended with the endpoint. The pin rule
  92     is unchanged:
  93
  94     * generated video rows step ``(t_video, r_video)``; text rows never reach an output head and follow video;
  95     * generated audio rows step ``(t_audio, r_audio)`` on their own schedule;
  96     * visual conditioning rows — ``fl2va`` keyframes, ``ref2va`` image and video references — are held at the
  97       noise-augmentation level, and an anchor has nowhere to go, so ``r == t`` there;
  98     * ``ref2va`` audio reference rows are never noised: ``t == r == 1.0``.
  99
 100     Returns:
 101         ``(timestep, endpoint, indices)``: the distinct ``(t, r)`` pairs (sorted, as two ``(num_pairs,)`` float32
 102         tensors) and, for every row, the index of its pair. ``timestep`` and ``indices`` are what the official
 103         transformer forward takes; ``endpoint`` is handed to the [`TwoTimeEmbedder`].
 104     """
 105     video_indices = video_indices.cpu()
 106     audio_indices = audio_indices.cpu()
 107     sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
 108
 109     timestep = torch.full((sequence_length,), float(video_timestep), dtype=torch.float32)
 110     endpoint = torch.full((sequence_length,), float(video_endpoint), dtype=torch.float32)
 111
 112     condition_video = video_indices[:num_condition_video_rows]
 113     timestep[condition_video] = float(condition_video_timestep)
 114     endpoint[condition_video] = float(condition_video_timestep)
 115
 116     generated_audio = audio_indices[num_condition_audio_rows:]
 117     timestep[generated_audio] = float(audio_timestep)
 118     endpoint[generated_audio] = float(audio_endpoint)
 119
 120     condition_audio = audio_indices[:num_condition_audio_rows]
 121     timestep[condition_audio] = float(condition_audio_timestep)
 122     endpoint[condition_audio] = float(condition_audio_timestep)
 123
 124     pairs = torch.stack((timestep, endpoint), dim=-1)
 125     unique_pairs, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
 126     return unique_pairs[:, 0].contiguous(), unique_pairs[:, 1].contiguous(), indices
```

Ours writes the rows in the same order (later writes win, as theirs), and `np.unique(axis=0)` sorts the same
lexicographic way.

📍 `src/h3mlx/denoise.py:80-94`
```python
  80 def row_time_pairs(layout: Layout, video_t: float, video_r: float, audio_t: float, audio_r: float,
  81                    cond_video_t: float, cond_audio_t: float):
  82     """HyperFlow's `build_row_time_pairs`: every row gets (t, r); conditioning rows keep r = t. Reduced to the distinct
  83     pairs, sorted lexicographically like `torch.unique(dim=0)`, and each row's index into them."""
  84     seq = len(layout.video_indices) + len(layout.audio_indices) + len(layout.text_indices)
  85     t = np.full(seq, np.float32(video_t), dtype=np.float32)
  86     r = np.full(seq, np.float32(video_r), dtype=np.float32)
  87     cv = layout.video_indices[: layout.num_condition_video_rows]
  88     t[cv] = r[cv] = cond_video_t
  89     ga = layout.audio_indices[layout.num_condition_audio_rows :]
  90     t[ga], r[ga] = audio_t, audio_r
  91     ca = layout.audio_indices[: layout.num_condition_audio_rows]
  92     t[ca] = r[ca] = cond_audio_t
  93     pairs, inv = np.unique(np.stack([t, r], axis=-1), axis=0, return_inverse=True)
  94     return pairs[:, 0].copy(), pairs[:, 1].copy(), inv.reshape(-1)
```

`test_row_time_pairs_match` checks four cases bit for bit on a ref2va layout with an image and an audio reference:

- a general step;
- a step at the 0.999 pin, with `t_v ≥ 0.999`;
- the first step, `t = 0`;
- a step where the video and audio pairs coincide.

**The released grid**, after the shifts (`Schedules.from_grid`, printed to 6 places):

| step i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video t | 0 | 0.00609 | 0.015712 | 0.033936 | 0.076923 | 0.165058 | 0.303148 | 0.531247 |
| video r | 0.00609 | 0.015712 | 0.033936 | 0.076923 | 0.165058 | 0.303148 | 0.531247 | 1 |
| audio t | 0 | 0.023924 | 0.060021 | 0.123202 | 0.25 | 0.441575 | 0.63505 | 0.819275 |
| audio r | 0.023924 | 0.060021 | 0.123202 | 0.25 | 0.441575 | 0.63505 | 0.819275 | 1 |

Video `t` never reaches the 0.999 pin, so the conditioning pair `(0.999, 0.999)` is always a separate entry. On the
last step the generated rows aim at `r = 1`, which is clean. Each step has at most four distinct pairs: generated video
(which text shares), generated audio, conditioning video, and audio references.

## 6. The loop

Their `set_timesteps` block builds the whole plan up front: one `(timestep, endpoint, indices)` triple per step. Their
loop denoiser hands each step's endpoints to the wrapper for that one forward.

📍 [`hyperflow:src/hyperflow_h3/blocks.py:156-213`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/blocks.py#L156-L213)
```python
 156     @torch.no_grad()
 157     def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
 158         block_state = self.get_block_state(state)
 159         if _hyperflow_is_disabled(components):
 160             # This block made the input optional so an enabled HyperFlow pipeline does not inherit Diffusers' generic
 161             # 50-point default. Restore that default only for the base-model path, then delegate without changing the
 162             # official schedule or row-timestep plan.
 163             if block_state.num_inference_steps is None:
 164                 state.set("num_inference_steps", _DIFFUSERS_DEFAULT_SIGMA_POINTS)
 165             return super().__call__(components, state)
 166
 167         device = components._execution_device
 168
 169         requested = block_state.num_inference_steps
 170         if requested is not None and int(requested) != self.num_steps:
 171             raise ValueError(
 172                 f"HyperFlow runs a fixed {self.num_steps}-step grid; `num_inference_steps={requested}` cannot be "
 173                 "honoured. Omit it, or build the blocks with `hyperflow_blocks(..., sigmas=<your grid>)`."
 174             )
 175
 176         video_shift = float(components.scheduler.shift)
 177         audio_shift = float(components.audio_scheduler.shift)
 178         self._check_shifts(video_shift, audio_shift)
 179         video_sigmas = shift_sigmas(self.sigmas, video_shift)
 180         audio_sigmas = shift_sigmas(self.sigmas, audio_shift)
 181
 182         components.scheduler.set_timesteps(sigmas=video_sigmas, device=device)
 183         components.audio_scheduler.set_timesteps(sigmas=audio_sigmas, device=device)
 184         block_state.timesteps = components.scheduler.timesteps
 185         block_state.audio_timesteps = components.audio_scheduler.timesteps
 186
 187         # The loop's `t` comes from `scheduler.timesteps`; read the same values back so the plan matches bit for bit.
 188         video_timesteps = components.scheduler.timesteps.cpu()
 189         audio_timesteps = components.audio_scheduler.timesteps.cpu()
 190         video_endpoints = endpoints_from_sigmas(video_sigmas)
 191         audio_endpoints = endpoints_from_sigmas(audio_sigmas)
 192
 193         plan = []
 194         for step in range(self.num_steps):
 195             video_timestep = float(video_timesteps[step])
 196             triple = build_row_time_pairs(
 197                 video_indices=block_state.video_indices,
 198                 audio_indices=block_state.audio_indices,
 199                 num_condition_video_rows=block_state.num_condition_video_rows,
 200                 num_condition_audio_rows=block_state.num_condition_audio_rows,
 201                 num_text_tokens=block_state.text_indices.numel(),
 202                 video_timestep=video_timestep,
 203                 video_endpoint=float(video_endpoints[step]),
 204                 audio_timestep=float(audio_timesteps[step]),
 205                 audio_endpoint=float(audio_endpoints[step]),
 206                 condition_video_timestep=max(video_timestep, components.keyframe_noise_aug),
 207                 condition_audio_timestep=1.0,
 208             )
 209             plan.append(tuple(tensor.to(device) for tensor in triple))
 210         block_state.row_timestep_plan = plan
 211
 212         self.set_block_state(state, block_state)
 213         return components, state
```
📍 [`hyperflow:src/hyperflow_h3/blocks.py:242-282`](https://github.com/Video-Rebirth/hyperflow/blob/1dd2f342aba5ab51da02b62885939655e8e268da/src/hyperflow_h3/blocks.py#L242-L282)
```python
 242     @torch.no_grad()
 243     def __call__(self, components: Any, block_state: Any, i: int, t: torch.Tensor):
 244         transformer = getattr(components, self.transformer_name)
 245         embedder = transformer.time_embedder
 246         if not isinstance(embedder, TwoTimeEmbedder):
 247             raise RuntimeError(
 248                 f"`{self.transformer_name}` has no HyperFlow LoRA loaded. Call `load_hyperflow_lora(pipe, ...)` after "
 249                 "`pipe.load_components(...)`."
 250             )
 251         if embedder.passthrough:
 252             return super().__call__(components, block_state, i, t)
 253
 254         entry = block_state.row_timestep_plan[i]
 255         if len(entry) != 3:
 256             raise RuntimeError(
 257                 "`row_timestep_plan` holds `(timestep, timestep_indices)` pairs, i.e. it was built by the official "
 258                 "`set_timesteps` block. Build the blocks with `hyperflow_blocks(...)` so both blocks are swapped."
 259             )
 260         timestep, endpoint, timestep_indices = entry
 261
 262         layout_kwargs = {
 263             name: value
 264             for name, value in block_state.denoiser_input_fields.items()
 265             if name in inspect.signature(transformer.forward).parameters
 266         }
 267         token = _current_step.set(i)
 268         try:
 269             with embedder.endpoint_context(endpoint):
 270                 block_state.noise_pred, block_state.audio_noise_pred = transformer(
 271                     hidden_states=block_state.latents[None],
 272                     audio_hidden_states=block_state.audio_latents[None],
 273                     encoder_hidden_states=block_state.prompt_embeds,
 274                     timestep=timestep,
 275                     timestep_indices=timestep_indices,
 276                     attention_kwargs=block_state.attention_kwargs,
 277                     return_dict=False,
 278                     **layout_kwargs,
 279                 )
 280         finally:
 281             _current_step.reset(token)
 282         return components, block_state
```

Ours builds the pairs per step inside `denoise` when `schedules.two_time` is set, and passes `endpoint` to
`dit_forward`. Everything else in the loop, the Euler update included, is the official one from the main guide.

📍 `src/h3mlx/denoise.py:97-146`
```python
  97 def denoise(
  98     params: dict,
  99     cfg: dict,
 100     layout: Layout,
 101     prompt_embeds: mx.array,
 102     video_rows: mx.array,
 103     audio_rows: mx.array,
 104     schedules: Schedules,
 105     on_step=None,
 106     cache: "ops.ModulationCache | None" = None,
 107     check_finite: bool = False,
 108 ):
 109     """video_rows (n_video, patch_dim) with conditioning rows first; audio_rows (n_audio, audio_ch) with reference rows
 110     first; prompt_embeds (1, n_text, text_dim). Returns the final (video_rows, audio_rows). With `cache`
 111     (`ops.ModulationCache` built from `schedule_timesteps`), the AdaLN projections are not evaluated per step."""
 112     lay = {
 113         "token_tags": mx.array(layout.token_tags),
 114         "position_ids": mx.array(layout.position_ids.astype(np.float32)),
 115         "video_indices": mx.array(layout.video_indices),
 116         "audio_indices": mx.array(layout.audio_indices),
 117         "text_indices": mx.array(layout.text_indices),
 118     }
 119     ncv, nca = layout.num_condition_video_rows, layout.num_condition_audio_rows
 120     vts, ats = schedules.video_timesteps, schedules.audio_timesteps
 121     for i, (t, ta) in enumerate(zip(vts, ats)):
 122         cond_t = max(float(t), KEYFRAME_NOISE_AUG)
 123         if schedules.two_time:
 124             uniq, ends, inv = row_time_pairs(layout, float(t), float(schedules.video_endpoints[i]), float(ta),
 125                                              float(schedules.audio_endpoints[i]), cond_t, 1.0)
 126             endpoint = mx.array(ends)
 127         else:
 128             uniq, inv = row_timesteps(layout, float(t), float(ta), cond_t, 1.0)
 129             endpoint = None
 130         v_pred, a_pred = ops.dit_forward(
 131             params, cfg, video_rows[None], audio_rows[None], prompt_embeds, mx.array(uniq), mx.array(inv), **lay,
 132             cache=cache, endpoint=endpoint,
 133         )
 134         new_v = ops.flow_step(v_pred[0, ncv:].astype(mx.float32), float(t), video_rows[ncv:],
 135                               float(schedules.video_sigmas[i]), float(schedules.video_sigmas[i + 1]))
 136         new_a = ops.flow_step(a_pred[0, nca:].astype(mx.float32), float(ta), audio_rows[nca:],
 137                               float(schedules.audio_sigmas[i]), float(schedules.audio_sigmas[i + 1]))
 138         video_rows = mx.concatenate([video_rows[:ncv], new_v]) if ncv else new_v
 139         audio_rows = mx.concatenate([audio_rows[:nca], new_a]) if nca else new_a
 140         mx.eval(video_rows, audio_rows)
 141         if check_finite:
 142             assert_finite(f"step {i} video rows", video_rows)
 143             assert_finite(f"step {i} audio rows", audio_rows)
 144         if on_step:
 145             on_step(i, t)
 146     return video_rows, audio_rows
```

`test_denoise_two_time_matches` runs 8 calls on a tiny model over a ref2va layout. The torch side is:

- the PEFT-injected diffusers transformer, with the transcribed wrapper;
- diffusers' `MiniMaxH3Scheduler` with the shifted grids installed;
- the transcribed pair builder.

Results: video and audio agree within 5e-4, the conditioning rows are untouched, and `on_step` fires exactly 8 times.

## 7. The run script

`--hyperflow <file>` selects the file's grid and refuses `--steps`; their pipeline also raises on a different step
count. The plan records the header's `lora_rank`, or the rank read from the tensor shapes when a header omits it. The file is checked twice:

- **Header:** read before anything loads, so a wrong file fails in seconds.
- **sha256:** hashed before the transformer loads and logged (`hyperflow_file` event). A file named like the released
  v1.0 whose hash isn't the published one is refused.

`plan` records the file's version, gate, rank, grid and base revision, and counts `grid_points` and `model_calls`
from the schedule actually used. The TFLOP/s in each step event adds the LoRA's matmuls:

📍 `scripts/run_ref2va.py:73-77`
```python
  73 def lora_tflop_per_call(rows: int, cfg: dict, rank: int) -> float:
  74     """HyperFlow's unmerged LoRA on q, k, v, out, ff in, ff out: 2 * S * rank * (in + out) each, per block."""
  75     h, inner, ffn = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["ffn_dim"]
  76     per_block = 2 * rows * rank * (4 * (h + inner) + (h + 2 * ffn) + (ffn + h))
  77     return per_block * cfg["num_layers"] / 1e12
```
📍 `scripts/run_ref2va.py:178-185`
```python
 178 def load_weights(c: Components, weights: Path, hyperflow_path: Path | None = None):
 179     """Attach the weights lazily (mx.load reads data on first use). The audio VAE is small and loaded eagerly. With
 180     `hyperflow_path`, the LoRA is attached to transformer_ref's params (2.8 GB more, read on first use too)."""
 181     c.vae, _, _ = load.load_vae(weights / "vae")
 182     c.audio_vae, _, _ = load.load_audio_vae(weights / "audio_vae")
 183     c.dit_params, _ = load.load_transformer(weights / "transformer_ref")
 184     if hyperflow_path is not None:
 185         c.dit_params, c.dit_cfg, _ = hyperflow.load(hyperflow_path, c.dit_params, c.dit_cfg, "transformer_ref")
```

**What to expect on an M3 Ultra (a projection, not a measurement).** run-002, at 1344×768 with a 2048 px image reference and
52,487 rows, measured 368 s per call at 16.2 TFLOP/s. By the FLOP count above, the LoRA adds about 2.3% per call at that
size. So 8 calls ≈ 8 × 368 × 1.023 ≈ 50 min of denoise, against about 5 h for 49 calls. Text, references and decode
are unchanged (33 s, 17 s, and a decode not yet measured at 768p). Memory rises by the LoRA's 2.8 GB on top of
run-002's peak.

## 8. The tests and the oracle

`tests/test_hyperflow.py` builds HyperFlow-format files for tiny models in the public key format, written by the test
itself rather than by our loader. It loads each file both ways:

- **Torch:** PEFT's `add_adapter` / `inject_adapter_in_model`, as their `_load_into` does, plus the transcribed wrapper.
- **MLX:** `hyperflow.attach`.

The transcriptions to compare against §3 and §5:

📍 `tests/test_hyperflow.py:37-53`
```python
  37 class TwoTimeEmbedder(torch.nn.Module):
  38     """embedder.py `TwoTimeEmbedder.forward`, endpoint set per forward."""
  39
  40     def __init__(self, base, endpoint, time_proj, gate):
  41         super().__init__()
  42         self.base, self.endpoint, self.time_proj, self.gate = base, endpoint, time_proj, float(gate)
  43         self.r = None
  44
  45     @property
  46     def linear_1(self):
  47         return self.base.linear_1
  48
  49     def forward(self, sample):
  50         t_emb = self.base(sample)
  51         r_proj = self.time_proj(self.r.to(device=sample.device)).to(sample.dtype)
  52         r_emb = self.endpoint(r_proj)
  53         return t_emb + self.gate * (r_emb - t_emb)
```
📍 `tests/test_hyperflow.py:56-68`
```python
  56 def torch_row_time_pairs(video_indices, audio_indices, ncv, nca, n_text, vt, vr, at, ar, cvt, cat=1.0):
  57     """schedule.py `build_row_time_pairs`."""
  58     n = int(video_indices.numel() + audio_indices.numel() + n_text)
  59     t = torch.full((n,), float(vt), dtype=torch.float32)
  60     r = torch.full((n,), float(vr), dtype=torch.float32)
  61     t[video_indices[:ncv]] = float(cvt)
  62     r[video_indices[:ncv]] = float(cvt)
  63     t[audio_indices[nca:]] = float(at)
  64     r[audio_indices[nca:]] = float(ar)
  65     t[audio_indices[:nca]] = float(cat)
  66     r[audio_indices[:nca]] = float(cat)
  67     pairs, inv = torch.unique(torch.stack((t, r), dim=-1), dim=0, sorted=True, return_inverse=True)
  68     return pairs[:, 0].contiguous(), pairs[:, 1].contiguous(), inv
```
📍 `tests/test_hyperflow.py:98-118`
```python
  98 def torch_hyperflow(model, tensors, meta):
  99     """PEFT injection as HyperFlow's `_load_into` does it, on a copy: endpoint embedder copied before any LoRA."""
 100     m = copy.deepcopy(model)
 101     endpoint = copy.deepcopy(m.time_embedder)
 102     cfg = dict(r=RANK, lora_alpha=float(meta["lora_alpha"]), lora_dropout=0.0, bias="none", init_lora_weights=True)
 103     main = sorted(t for t in public_targets(m.config) if not t.startswith("endpoint_"))
 104     m.add_adapter(LoraConfig(target_modules=main, **cfg), adapter_name="hyperflow")
 105     inject_adapter_in_model(LoraConfig(target_modules=["linear_1", "linear_2"], **cfg), endpoint, "hyperflow")
 106     params = dict(m.named_parameters())
 107     ep = dict(endpoint.named_parameters())
 108     with torch.no_grad():
 109         for key, value in tensors.items():
 110             module, ab = key[len("transformer."):].rsplit(".lora_", 1)
 111             ab = ab.split(".")[0]
 112             if module.startswith("endpoint_time_embedder."):
 113                 dst = ep[f"{module[len('endpoint_time_embedder.'):]}.lora_{ab}.hyperflow.weight"]
 114             else:
 115                 dst = params[f"{module}.lora_{ab}.hyperflow.weight"]
 116             dst.copy_(value.to(dst.dtype))
 117     m.time_embedder = TwoTimeEmbedder(m.time_embedder, endpoint, copy.deepcopy(m.time_proj), float(meta["hyperflow_gate"]))
 118     return m
```

| test | pins |
| --- | --- |
| `test_grid_matches_diffusers_scheduler` (×2 shifts) | grid, timesteps and endpoints bit-exact against diffusers' scheduler |
| `test_grid_rejects_invalid` (×4) | not decreasing, start above 1, end not 0, a single point |
| `test_row_time_pairs_match` (×4) | pairs and indices bit-exact against the transcription |
| `test_forward_matches_peft` | f32 forward within 2e-4. Guards: the LoRA moves the output by more than 5%, and moving `r` changes it |
| `test_mixed_precision_tracks_peft` | the release's dtypes: MLX error against f32 truth ≤ 1.5× torch's bf16 error + 1e-3 (video 0.0132 vs 0.0097, audio 0.0070 vs 0.0076); adapter dtypes asserted on both sides |
| `test_denoise_two_time_matches` | 8 calls, ref2va layout, within 5e-4 |
| `test_real_header_fits_transformer_ref` | the released header against `transformer_ref`'s real shapes |
| `test_read_header_from_a_file`, `test_refusals` | header-only read, `load`, and each refusal (incl. an endpoint count that doesn't match the distinct timesteps, as their wrapper checks, and a LoRA on a 1-D weight) |
| `test_run_script_with_hyperflow`, `..._refuses_a_released_name_with_other_bytes` (`tests/test_run_script.py`) | 8 step events, 9 grid points, the plan block, the hash event, a 124-frame mp4; a hash mismatch fails the run |

**The bf16 bound is close to the noise.** Over 8 seeds (a scratch probe, not a test), MLX's bf16 error divided by
torch's ranged:

- **HyperFlow:** 0.58–1.52 (video) and 0.93–1.54 (audio); the median of the 16 is about 1.04.
- **Base model:** 0.94–1.16 (video) and 0.64–1.22 (audio).

Two of the 16 HyperFlow ratios (1.52 and 1.54) were above 1.5. The test's bound also has an absolute slack of 1e-3,
which at these error sizes (torch about 0.008–0.01) lifts it to about 1.6×, above every observed ratio. The margin is
thin. The probe shows a wider spread than the base model's, not a bias: the median is about 1.

## 9. Not done, and open

- **No real-weight run yet.** No real-weight HyperFlow run had been made when this was written. Quality,
  memory and speed with HyperFlow are unmeasured.
- **Not ported:** Sol-Attn (CUDA only), Ulysses context parallel, merged LoRA, a two-time modulation cache, and the
  disable/enable toggle.
- **Licence:** the weights carry the same MiniMax H3 Community License as the base model, with the same territory
  exclusions (EU, UK, KR, US). Check its terms before use.
- **Our own claim, not theirs:** the README's speed table is H200; the M3 Ultra figure in §7 is a projection.