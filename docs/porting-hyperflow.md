# Porting HyperFlow to MLX: Video Rebirth's loader → `h3mlx`

<sub>Revision 3: after a review, memory and speed in §7 were measured and qualified, the seed probe committed, the
provenance files saved, and read_header and the run script's file check linked. Revision 2: a review found that the plan
step skipped the key and shape checks, LoRAs on unwired linears were accepted, and an explicit empty grid fell back to
the default. All fixed; §1, §3, §7 and §8 updated.</sub>

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
| base model the LoRA was trained on | header `base_model_revision` | `MiniMaxAI/MiniMax-H3` `83db0c0e`; all 16 `transformer_ref` files (14 shards, config, index) have the same oids as our `42ed227` |

The manifest and both revisions' `transformer_ref` file listings are release fixtures, fetched by
`scripts/fetch_test_fixtures.py` (`hyperflow.manifest-5bde94d7.json`, `transformer_ref.lfs-42ed227-vs-83db0c0.json`), and
`test_provenance_files_back_the_constants` ties the run script's sha256 and the base revision to them.

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
which base `Linear` each LoRA sits on.

`check_plan` makes the checks their `_load_into` makes explicitly (every target an `nn.Linear`, one rank matching the
header, each tensor's shape against its parameter), and one theirs doesn't need:

- **Shapes:** every LoRA must fit its base 2-D `Linear` weight.
- **Rank and scaling:** all LoRAs share one positive rank, and the scaling must be 1 (§2).
- **Wiring:** every target must be a linear that our forward actually runs through `plinear` (`wired_shapes`). PEFT
  would wrap *any* `nn.Linear` a file names, `proj_in` or `adaln_proj` included. Our forward calls plain `linear` for
  those, so a LoRA there would be silently ignored; it is refused instead. The released file targets exactly the wired
  set.

`check_file` runs all of this from the header alone, against shapes derived from the transformer's config, and also
checks the grid. The run script calls it in the plan step (§7).

Two header details:

- **An empty grid.** A header that holds `"hyperflow_sigmas": "[]"` is refused, as theirs refuses it in
  `validate_sigmas`. Only a missing key falls back to the default grid.
- **`compatible_subfolders`.** Stricter than theirs, on purpose: ours refuses a file that lists subfolders not
  including `transformer_ref`. Their loader parses the field but never reads it.

📍 [`src/h3mlx/hyperflow.py:53-78`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L53-L78)
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
  73             # absent -> the package default; present but empty or invalid -> refused later by shift_grid, as theirs
  74             sigmas=lst("hyperflow_sigmas", float) if "hyperflow_sigmas" in md else DEFAULT_SIGMAS_8STEP,
  75             video_shift=float(md["hyperflow_video_shift"]) if "hyperflow_video_shift" in md else None,
  76             audio_shift=float(md["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in md else None,
  77             compatible_subfolders=lst("compatible_subfolders", str) or (), raw=md,
  78         )
```
📍 [`src/h3mlx/hyperflow.py:93-104`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L93-L104)
```python
  93 def plan_from_keys(keys) -> dict:
  94     """{module: {"A": key, "B": key}} in the file's own (public) module names; anything else is an error."""
  95     plan = {}
  96     for key in keys:
  97         m = KEY_RE.match(key)
  98         if m is None:
  99             raise ValueError(f"unexpected key {key!r}: only transformer.<module>.lora_A/B.weight keys are allowed")
 100         plan.setdefault(m["module"], {})[m["matrix"]] = key
 101     incomplete = [k for k, v in plan.items() if set(v) != {"A", "B"}]
 102     if incomplete:
 103         raise ValueError(f"modules with only one of lora_A / lora_B: {incomplete[:8]}")
 104     return plan
```
📍 [`src/h3mlx/hyperflow.py:127-129`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L127-L129)
```python
 127 def _base_name(module: str) -> str:
 128     """The transformer Linear a module's LoRA sits on: the endpoint embedder is a copy of `time_embedder`."""
 129     return BASE_TIME + module[len(ENDPOINT):] if module.startswith(ENDPOINT) else module
```
📍 [`src/h3mlx/hyperflow.py:110-124`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L110-L124)
```python
 110 def wired_shapes(cfg: dict) -> dict:
 111     """{module: (out, in)} for every Linear our forward runs through `ops.plinear`, so the only places a LoRA can act.
 112     A LoRA anywhere else (proj_in, adaln_proj, norm_out, the heads...) would be ignored by the port, so it is refused."""
 113     h, inner, ffn = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["ffn_dim"]
 114     per_block = {"attn.to_q": (inner, h), "attn.to_k": (inner, h), "attn.to_v": (inner, h),
 115                  "attn.to_out.0": (h, inner), "ff.net.0.proj": (2 * ffn, h), "ff.net.2": (h, ffn)}
 116     out = {}
 117     for prefix, n in (("transformer_blocks", cfg["num_layers"]), ("token_refiner.refiner_blocks", cfg["num_refiner_layers"])):
 118         for i in range(n):
 119             out.update({f"{prefix}.{i}.{k}": v for k, v in per_block.items()})
 120     te = {"linear_1": (cfg["time_embed_hidden_dim"], cfg["freq_dim"]),
 121           "linear_2": (cfg["time_embed_dim"], cfg["time_embed_hidden_dim"])}
 122     for prefix in (BASE_TIME, ENDPOINT):
 123         out.update({prefix + k: v for k, v in te.items()})
 124     return out
```
📍 [`src/h3mlx/hyperflow.py:132-165`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L132-L165)
```python
 132 def check_plan(plan: dict, shapes: dict, base_shapes: dict, meta: HyperFlowMeta, wired=None) -> int:
 133     """Every LoRA lands on a Linear weight of the right shape, one rank throughout, scaling 1. `shapes` maps the file's
 134     keys to their shapes, `base_shapes` our parameter names to theirs, `wired` the modules `ops.plinear` serves (any
 135     other target is refused: the port would silently skip it). Returns the rank."""
 136     if not plan:
 137         raise ValueError("no LoRA tensors in this file")
 138     if wired is not None:
 139         unwired = sorted(m for m in plan if m not in wired)
 140         if unwired:
 141             raise ValueError(f"{len(unwired)} LoRA target(s) are not run through ops.plinear here (the port would ignore "
 142                              f"them), e.g. {unwired[:5]}")
 143     no_base = [m for m in plan if _base_name(m) + ".weight" not in base_shapes]
 144     if no_base:
 145         raise KeyError(f"{len(no_base)} LoRA target(s) have no Linear weight in this transformer, e.g. {no_base[:5]}")
 146     not_linear = [m for m in plan if len(base_shapes[_base_name(m) + ".weight"]) != 2]
 147     if not_linear:
 148         raise ValueError(f"LoRA target(s) whose weight is not a 2-D Linear weight: {not_linear[:5]}")
 149     ranks = set()
 150     for module, keys in plan.items():
 151         a, b = tuple(shapes[keys["A"]]), tuple(shapes[keys["B"]])
 152         out_f, in_f = tuple(base_shapes[_base_name(module) + ".weight"])
 153         if len(a) != 2 or len(b) != 2 or a[1] != in_f or b[0] != out_f or a[0] != b[1]:
 154             raise ValueError(f"{module}: lora_A {a} / lora_B {b} do not fit the Linear ({out_f}, {in_f})")
 155         ranks.add(a[0])
 156     if len(ranks) != 1:
 157         raise ValueError(f"all LoRA targets must share one rank, found {sorted(ranks)}")
 158     rank = ranks.pop()
 159     if rank <= 0:
 160         raise ValueError(f"LoRA rank must be positive, got {rank}")
 161     if meta.lora_rank is not None and meta.lora_rank != rank:
 162         raise ValueError(f"header says lora_rank={meta.lora_rank} but the tensors have rank {rank}")
 163     if meta.lora_alpha / rank != 1.0:
 164         raise NotImplementedError(f"LoRA scaling alpha/rank = {meta.lora_alpha}/{rank} != 1 (ops.plinear assumes 1)")
 165     return rank
```
📍 [`src/h3mlx/hyperflow.py:193-200`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L193-L200)
```python
 193 def check_file(meta: HyperFlowMeta, shapes: dict, cfg: dict, subfolder: str = "transformer_ref") -> int:
 194     """Everything `attach` checks, from the header alone against the transformer's config (no tensor data, no base
 195     weights): subfolder, keys, wiring, shapes, rank, scaling, and the grid. Returns the rank."""
 196     _check_subfolder(meta, subfolder)
 197     base = {m + ".weight": s for m, s in wired_shapes(cfg).items()}
 198     rank = check_plan(plan_from_keys(shapes), shapes, base, meta, wired=wired_shapes(cfg))
 199     meta.schedules()  # raises on an invalid grid
 200     return rank
```

**The released v1.0 header**, fetched with an HTTP range request; `scripts/fetch_test_fixtures.py` writes the full
header to `hyperflow_v1.0.header.json`:

- **Tensors:** 632, covering 316 modules: 6 per block × 50 blocks, 6 per refiner block × 2 refiner blocks, and two
  linears each for `time_embedder` and `endpoint_time_embedder`.
- **LoRA:** rank 256, alpha 256, so scaling 1. Gate 0.25.
- **Grid:** `[1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]`.
- **Shifts:** video 12, audio 3.
- **Dtypes:** bf16 for the blocks, f32 for the time embedders.

`test_real_header_fits_transformer_ref` checks every LoRA against the real `transformer_ref` shapes, building the
model on torch's meta device. It also checks:

- every target is a released tensor, and none is an `adaln_proj`;
- the file's targets are exactly `wired_shapes`;
- the config-derived shapes equal the real model's;
- `check_file` passes.

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

What a PEFT-wrapped `Linear` computes, from PEFT 0.21.2 (Apache-2.0, see `THIRD_PARTY_NOTICES.md`):

📍 [`peft:src/peft/tuners/lora/layer.py:1074-1115`](https://github.com/huggingface/peft/blob/eaa35c777c7ba554a2a3639ac9c79bb690f39a57/src/peft/tuners/lora/layer.py#L1074-L1115)
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

📍 [`src/h3mlx/ops.py:67-77`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/ops.py#L67-L77)
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

📍 [`src/h3mlx/hyperflow.py:168-184`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L168-L184)
```python
 168 def attach(params: dict, cfg: dict, tensors: dict, meta: HyperFlowMeta, subfolder: str = "transformer_ref"):
 169     """Return (params, cfg) with HyperFlow attached; the inputs are not modified. Adapter tensors are cast to their
 170     base weight's dtype, as PEFT's injection does (bf16 on the blocks, f32 on the time embedder)."""
 171     _check_subfolder(meta, subfolder)
 172     if ENDPOINT + "linear_1.weight" in params:
 173         raise ValueError("HyperFlow is already attached to these parameters")
 174     plan = plan_from_keys(tensors)
 175     check_plan(plan, {k: v.shape for k, v in tensors.items()}, {k: v.shape for k, v in params.items()}, meta,
 176                wired=wired_shapes(cfg))
 177     out = dict(params)
 178     for name in ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"):
 179         out[ENDPOINT + name] = params[BASE_TIME + name]  # a copy of the base embedder (shared arrays, no new memory)
 180     for module, keys in plan.items():
 181         dtype = params[_base_name(module) + ".weight"].dtype
 182         out[module + ".lora_A.weight"] = tensors[keys["A"]].astype(dtype)
 183         out[module + ".lora_B.weight"] = tensors[keys["B"]].astype(dtype)
 184     return out, {**cfg, "hyperflow_gate": float(meta.gate)}
```
📍 [`src/h3mlx/hyperflow.py:203-209`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L203-L209)
```python
 203 def read_header(path) -> tuple[HyperFlowMeta, dict]:
 204     """(meta, {key: shape}) from the safetensors header alone, no tensor data."""
 205     with open(path, "rb") as f:
 206         n = int.from_bytes(f.read(8), "little")
 207         header = json.loads(f.read(n))
 208     md = header.pop("__metadata__", None)
 209     return HyperFlowMeta.from_header(md), {k: tuple(v["shape"]) for k, v in header.items()}
```
📍 [`src/h3mlx/hyperflow.py:212-218`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L212-L218)
```python
 212 def load(path, params: dict, cfg: dict, subfolder: str = "transformer_ref"):
 213     """Load a HyperFlow .safetensors onto our transformer params: (params, cfg, meta)."""
 214     path = Path(path)
 215     tensors, md = mx.load(str(path), return_metadata=True)
 216     meta = HyperFlowMeta.from_header(md)
 217     params, cfg = attach(params, cfg, tensors, meta, subfolder)
 218     return params, cfg, meta
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

- **Same base weights.** `endpoint` is a `deepcopy` of `time_embedder` made *before* any LoRA: new storage with the
  same values, so the two embedders have equal base weights and differ only in their LoRAs. Ours aliases the same
  arrays instead (§2), which is the same at inference: nothing writes the weights.
- **`r` gets the same sinusoid.** `r` goes through a copy of the same parameter-free `time_proj`, in f32, cast to the
  dtype of `sample` (f32 here).
- **The blend.** It is `t_emb + gate * (r_emb - t_emb)`, with the gate a Python float: 0.25, exact in f32.
- **No silent fallback.** The wrapper refuses to run without endpoints, so the official loop can't silently drive a
  two-time adapter with single times.

Ours is one function. It refuses the mismatch in both directions: HyperFlow weights without endpoints, and endpoints
without HyperFlow weights. Like their wrapper, it also refuses an endpoint count that differs from the distinct
timestep count (MLX would otherwise broadcast a length-1 array silently).

📍 [`src/h3mlx/ops.py:176-190`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/ops.py#L176-L190)
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

📍 [`src/h3mlx/ops.py:369-382`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/ops.py#L369-L382)
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
`ModulationCache` itself, `dit_forward` and `schedule_timesteps` all raise rather than build or use wrong tables. (`ref2va` never builds a cache.) Extending the cache to pairs is
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

📍 [`src/h3mlx/denoise.py:20-28`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/denoise.py#L20-L28)
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
📍 [`src/h3mlx/denoise.py:31-60`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/denoise.py#L31-L60)
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

📍 [`src/h3mlx/hyperflow.py:84-90`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/hyperflow.py#L84-L90)
```python
  84     def schedules(self, video_shift: float = 12.0, audio_shift: float = 3.0) -> Schedules:
  85         """The file's grid through the pipeline's shifts (the release's 12 / 3); warns if the file was trained with
  86         others, as HyperFlow does."""
  87         trained = (self.video_shift, self.audio_shift)
  88         if None not in trained and trained != (video_shift, audio_shift):
  89             warnings.warn(f"HyperFlow was trained with shifts {trained}, the pipeline uses {(video_shift, audio_shift)}")
  90         return Schedules.from_grid(self.sigmas, video_shift, audio_shift, two_time=True)
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

📍 [`src/h3mlx/denoise.py:80-94`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/denoise.py#L80-L94)
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

📍 [`src/h3mlx/denoise.py:97-146`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/src/h3mlx/denoise.py#L97-L146)
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
count. The file is checked three times:

- **Header metadata:** parsed right after the arguments, before anything loads.
- **The whole header (`check_file`):** keys, wiring, shapes against `transformer_ref`'s config, rank, scaling, subfolder
  and grid, in the plan step. That step runs with `--plan-only` too, and before the hash. The plan records the rank
  `check_file` found.
- **sha256:** hashed after the plan and before the transformer loads, and logged (`hyperflow_file` event). A file named
  like the released v1.0 whose hash isn't the published one is refused.

`plan` records the file's version, gate, rank, grid and base revision, and counts `grid_points` and `model_calls`
from the schedule actually used. The TFLOP/s in each step event adds the LoRA's matmuls:

📍 [`scripts/run_ref2va.py:73-77`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/scripts/run_ref2va.py#L73-L77)
```python
  73 def lora_tflop_per_call(rows: int, cfg: dict, rank: int) -> float:
  74     """HyperFlow's unmerged LoRA on q, k, v, out, ff in, ff out: 2 * S * rank * (in + out) each, per block."""
  75     h, inner, ffn = cfg["hidden_size"], cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["ffn_dim"]
  76     per_block = 2 * rows * rank * (4 * (h + inner) + (h + 2 * ffn) + (ffn + h))
  77     return per_block * cfg["num_layers"] / 1e12
```
📍 [`scripts/run_ref2va.py:183-190`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/scripts/run_ref2va.py#L183-L190)
```python
 183 def load_weights(c: Components, weights: Path, hyperflow_path: Path | None = None):
 184     """Attach the weights lazily (mx.load reads data on first use). The audio VAE is small and loaded eagerly. With
 185     `hyperflow_path`, the LoRA is attached to transformer_ref's params (2.8 GB more, read on first use too)."""
 186     c.vae, _, _ = load.load_vae(weights / "vae")
 187     c.audio_vae, _, _ = load.load_audio_vae(weights / "audio_vae")
 188     c.dit_params, _ = load.load_transformer(weights / "transformer_ref")
 189     if hyperflow_path is not None:
 190         c.dit_params, c.dit_cfg, _ = hyperflow.load(hyperflow_path, c.dit_params, c.dit_cfg, "transformer_ref")
```
📍 [`scripts/run_ref2va.py:193-219`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/scripts/run_ref2va.py#L193-L219)
```python
 193 def plan(c: Components, prompt: str, image: Image.Image, args) -> dict:
 194     """Everything up to the layout, without the models: sizes the run would use."""
 195     h, w, n, refs = setup(c, [ImageRef(image)], args.height, args.width, args.frames)
 196     vision, ic, vc, vts = vision_features(c, refs)
 197     ids, tags = build_presentation(c.tokenizer, prompt, [Ref(r.kind, r.has_audio) for r in refs], ic, vc, vts)
 198     ratio = c.vae.cfg.spatial_ratio
 199     ref_shapes = [(1, r.image.size[1] // ratio, r.image.size[0] // ratio) for r in refs]
 200     lf, lh, lw = R.video_latent_num_frames(n), h // ratio, w // ratio
 201     na = R.audio_latent_num_frames(n)
 202     layout = build_ref2va_packed_sequence(np.asarray(tags), [Ref(r.kind, r.has_audio) for r in refs], ref_shapes, [],
 203                                           lf, lh, lw, na, tuple(c.dit_cfg["patch_size"]), 2)
 204     if args.hyperflow:
 205         meta, shapes = hyperflow.read_header(args.hyperflow)
 206         sch = meta.schedules()
 207     else:
 208         sch = Schedules.make(args.steps)
 209     extra = {}
 210     if args.hyperflow:
 211         rank = hyperflow.check_file(meta, shapes, c.dit_cfg, "transformer_ref")  # before the hash and any weights
 212         extra["hyperflow"] = {"file": str(args.hyperflow), "version": meta.version, "gate": meta.gate,
 213                               "rank": rank, "sigmas": list(meta.sigmas),
 214                               "base_model_revision": meta.base_model_revision}
 215     return {**extra, "height": h, "width": w, "frames": n, "seconds": round(n / R.FPS, 3), "latent_frames": lf,
 216             "reference_size": list(refs[0].image.size), "text_rows": len(ids), "vision_tokens": int(sum(ic)),
 217             "reference_video_rows": layout.num_condition_video_rows, "target_audio_rows": na * 2,
 218             "sequence_rows": int(layout.position_ids.shape[0]), "grid_points": int(len(sch.video_sigmas)),
 219             "model_calls": int(len(sch.video_timesteps))}
```
📍 [`scripts/run_ref2va.py:312-318`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/scripts/run_ref2va.py#L312-L318)
```python
 312         if args.hyperflow and not args.decode_from:  # a re-decode never reads the adapter
 313             hf = args.hyperflow
 314             got = sha256(hf)
 315             want = hyperflow.KNOWN_SHA256.get(hf.name)
 316             event("hyperflow_file", file=hf.name, bytes=hf.stat().st_size, sha256=got, known_sha256=want)
 317             if want is not None and got != want:
 318                 raise ValueError(f"{hf.name}: sha256 {got} is not the released {want}")
```

`ref2va` frees the weight dict before decoding (`free_transformer`, main guide §8). The dict holds the LoRAs and the
endpoint aliases too, so nothing of HyperFlow outlives the denoise. A `--decode-from` re-decode with `--hyperflow`
still runs the plan's header check, but doesn't hash or load the adapter.

**What to expect on an M3 Ultra (a projection, not a measurement).** run-002, at 1344×768 with a 2048 px image reference and
52,487 rows, measured 368 s per call at 16.2 TFLOP/s.

- **Speed.** The LoRA adds 2.34% more FLOPs per call at that size (5.51% at run-001's 6,823 rows). Its small rank-256
  matmuls run a little less efficiently than the big ones, so FLOPs undercount the time a little.
  `bench/lora_speed.py` times the largest projection (`ff.net.0.proj`, 52,487 rows, bf16, M2 Max), warm, over 3 repeats:

  | mode | time per call |
  | --- | --- |
  | base | 1.664 s |
  | with the LoRA | 1.793 s (+7.8%, against +5.7% FLOPs for that op) |
  | merged into the weight | 1.669 s |

  Scaling the whole call's 2.34% by that ratio gives about 3%. So 8 calls ≈ 8 × 368 × 1.03 ≈ 50–51 min of denoise,
  against about 5 h for 49 calls, assuming the M3 Ultra keeps run-002's throughput otherwise.
- **Cold calls.** A reviewer's single cold call of the same op took 4.17 s against 2.31 s, under a 20 GB
  limit with no allocator cache. Expect the first call to be slower.
- **Unchanged stages.** Text, references and decode are unchanged: 33 s, 17 s, and a decode not yet measured at 768p.
- **Memory.** It adds 2.8 GB of adapter weights plus transient LoRA outputs. The largest transient is the
  `ff.net.0.proj` delta, rows × 28,672 in bf16, about 3.0 GB at 52,487 rows. The same probe measured that op's peak
  over its inputs going from 3.01 to 6.05 GB. Blocks are evaluated one at a time, so these transients don't pile up
  across blocks. HyperFlow's whole-run peak is unmeasured.
- **Merging.** Merging the LoRA into the weights at load (`W + B·A`) would remove both the extra time and the
  transients. It is not done: it rounds differently from PEFT's unmerged path, and it would need its own error budget,
  like the modulation cache's.

## 8. The tests and the oracle

`tests/test_hyperflow.py` builds HyperFlow-format files for tiny models in the public key format, written by the test
itself rather than by our loader. It loads each file both ways:

- **Torch:** PEFT's `add_adapter` / `inject_adapter_in_model`, as their `_load_into` does, plus the transcribed wrapper.
- **MLX:** `hyperflow.attach`.

The transcriptions to compare against §3 and §5:

📍 [`tests/test_hyperflow.py:35-51`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/tests/test_hyperflow.py#L35-L51)
```python
  35 class TwoTimeEmbedder(torch.nn.Module):
  36     """embedder.py `TwoTimeEmbedder.forward`, endpoint set per forward."""
  37
  38     def __init__(self, base, endpoint, time_proj, gate):
  39         super().__init__()
  40         self.base, self.endpoint, self.time_proj, self.gate = base, endpoint, time_proj, float(gate)
  41         self.r = None
  42
  43     @property
  44     def linear_1(self):
  45         return self.base.linear_1
  46
  47     def forward(self, sample):
  48         t_emb = self.base(sample)
  49         r_proj = self.time_proj(self.r.to(device=sample.device)).to(sample.dtype)
  50         r_emb = self.endpoint(r_proj)
  51         return t_emb + self.gate * (r_emb - t_emb)
```
📍 [`tests/test_hyperflow.py:54-66`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/tests/test_hyperflow.py#L54-L66)
```python
  54 def torch_row_time_pairs(video_indices, audio_indices, ncv, nca, n_text, vt, vr, at, ar, cvt, cat=1.0):
  55     """schedule.py `build_row_time_pairs`."""
  56     n = int(video_indices.numel() + audio_indices.numel() + n_text)
  57     t = torch.full((n,), float(vt), dtype=torch.float32)
  58     r = torch.full((n,), float(vr), dtype=torch.float32)
  59     t[video_indices[:ncv]] = float(cvt)
  60     r[video_indices[:ncv]] = float(cvt)
  61     t[audio_indices[nca:]] = float(at)
  62     r[audio_indices[nca:]] = float(ar)
  63     t[audio_indices[:nca]] = float(cat)
  64     r[audio_indices[:nca]] = float(cat)
  65     pairs, inv = torch.unique(torch.stack((t, r), dim=-1), dim=0, sorted=True, return_inverse=True)
  66     return pairs[:, 0].contiguous(), pairs[:, 1].contiguous(), inv
```
📍 [`tests/test_hyperflow.py:96-116`](https://github.com/sublimator/minimax-h3-mlx/blob/5eaa03bceeaeb1d1d8884ba2047225c416526f08/tests/test_hyperflow.py#L96-L116)
```python
  96 def torch_hyperflow(model, tensors, meta):
  97     """PEFT injection as HyperFlow's `_load_into` does it, on a copy: endpoint embedder copied before any LoRA."""
  98     m = copy.deepcopy(model)
  99     endpoint = copy.deepcopy(m.time_embedder)
 100     cfg = dict(r=RANK, lora_alpha=float(meta["lora_alpha"]), lora_dropout=0.0, bias="none", init_lora_weights=True)
 101     main = sorted(t for t in public_targets(m.config) if not t.startswith("endpoint_"))
 102     m.add_adapter(LoraConfig(target_modules=main, **cfg), adapter_name="hyperflow")
 103     inject_adapter_in_model(LoraConfig(target_modules=["linear_1", "linear_2"], **cfg), endpoint, "hyperflow")
 104     params = dict(m.named_parameters())
 105     ep = dict(endpoint.named_parameters())
 106     with torch.no_grad():
 107         for key, value in tensors.items():
 108             module, ab = key[len("transformer."):].rsplit(".lora_", 1)
 109             ab = ab.split(".")[0]
 110             if module.startswith("endpoint_time_embedder."):
 111                 dst = ep[f"{module[len('endpoint_time_embedder.'):]}.lora_{ab}.hyperflow.weight"]
 112             else:
 113                 dst = params[f"{module}.lora_{ab}.hyperflow.weight"]
 114             dst.copy_(value.to(dst.dtype))
 115     m.time_embedder = TwoTimeEmbedder(m.time_embedder, endpoint, copy.deepcopy(m.time_proj), float(meta["hyperflow_gate"]))
 116     return m
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
| `test_read_header_from_a_file`, `test_refusals` | header-only read, `load`, and each refusal: an endpoint count that doesn't match the distinct timesteps (as their wrapper checks); a LoRA on an unwired linear (`proj_in`, `adaln_proj`) or a 1-D weight; an explicit empty grid; rank 0; no LoRA tensors; a modulation cache on HyperFlow params; a header without `lora_rank` still checks |
| `test_run_script_with_hyperflow`, `..._refuses_a_released_name_with_other_bytes`, `test_plan_only_checks_the_hyperflow_file` (`tests/test_run_script.py`) | 8 step events, 9 grid points, the plan block, the hash event, a 124-frame mp4; a hash mismatch fails the run; a file with an unwired LoRA fails `--plan-only`, before any hash |

**The bf16 bound is close to the noise.** Over 8 seeds (`bench/hyperflow_bf16_seeds.py`, not a test; model seed `s`,
file seed `s + 50`, input seed `100 + s`, 4 tiny layers), MLX's bf16 error divided by torch's ranged:

- **HyperFlow:** 0.58–1.52 (video) and 0.93–1.54 (audio); the median of the 16 is about 1.04.
- **Base model:** 0.94–1.16 (video) and 0.64–1.22 (audio).

Two of the 16 HyperFlow ratios (1.52 and 1.54) were above 1.5. The test's bound also has an absolute slack of 1e-3,
which at these error sizes (torch about 0.008–0.01) lifts it to about 1.6×, above every observed ratio. The margin is
thin. These are ratios of error norms, so they say how large MLX's error is next to torch's, not in which direction.
Eight tiny seeds show a wider spread than the base model's; they say nothing about the real rank-256, 50-layer,
8-call trajectory.

## 9. Not done, and open

- **No real-weight run yet.** No real-weight HyperFlow run had been made when this was written. Quality,
  memory and speed with HyperFlow are unmeasured. The tests pin a tiny model: one f32 8-call loop at rank 4, and one
  mixed-precision forward on 4 layers. Nothing here compares a decoded clip against their diffusers pipeline.
- **Shifts.** The run script uses the release's shifts (12 and 3). It does not read a modified scheduler config; a
  caller who changes those passes them to `HyperFlowMeta.schedules`.
- **Not ported:** Sol-Attn (CUDA only), Ulysses context parallel, merged LoRA, a two-time modulation cache, and the
  disable/enable toggle.
- **Licence:** the weights carry the same MiniMax H3 Community License as the base model, with the same territory
  exclusions (EU, UK, KR, US). Check its terms before use.
- **Our own claim, not theirs:** the README's speed table is H200; the M3 Ultra figure in §7 is a projection.