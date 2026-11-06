# Porting P-H3-Image to MLX: diffusers' `PH3Image*` → `h3mlx`

P-H3-Image is Pruna AI's image model built from MiniMax H3. It does text-to-image and editing with up to 8 pictures,
in 8 steps with no guidance, best at 8 MP. This guide maps its diffusers implementation onto our MLX package, side by
side: in each section the original comes first, then the port, then what changed and which test ties them together.
It assumes the main guide (`docs/porting.md`), which covers the transformer, the packed sequence, the scheduler, the
denoise loop, the Qwen3-VL presentation and the video VAE. Most of P-H3-Image is those, unchanged.

**Sources, pinned.**

| what | where | pin |
| --- | --- | --- |
| pipeline, image VAE, text connector (Apache-2.0, Pruna AI and Hugging Face) | `huggingface/diffusers` main | `d3d8af89e5b08a352cf47bba3a451c5fed9a49b1` |
| weights, configs, tokenizer, processors | Hugging Face `PrunaAI/P-H3-Image` | revision `7f98acef` |
| text encoder and upsampler runtime | `Blaizzy/mlx-vlm` | `v0.7.11` (`527c513`) |

**What P-H3-Image changes, against MiniMax H3:**

| piece | MiniMax H3 | P-H3-Image |
| --- | --- | --- |
| transformer | `MiniMaxH3Transformer3DModel`, 33.1B | the same class, config and 638 weight names, fine-tuned |
| text encoder | Qwen3-VL, 64 layers of width 5120; hidden state 50, pre-norm | Qwen3-VL-4B, 36 layers of width 2560; states 13, 19, 25, 31 and the final normed one, through a connector |
| VAE | video: 17-frame chunks, a ViT decoder with a temporal patch of 4 | H3's encoder and latent space; a single-frame ViT decoder |
| audio | an audio VAE; the soundtrack is generated | no audio VAE; two audio latents ride along and are thrown away |
| sampling | `linspace` grid shifted per modality | one fixed 9-point grid, used verbatim for both modalities: 8 calls |
| prompt | written by hand | an upsampler (a Qwen3.5-9B fine-tune) writes it, on by default |

The packing, the transformer forward, the per-row timesteps and the Euler update are H3's: diffusers marks those
P-H3-Image steps "Copied from" the MiniMax H3 ones, and our port calls the same `packing`, `ops.dit_forward` and
`denoise` code the video pipeline does. The new code is `src/h3mlx/image.py` (the pipeline), `src/h3mlx/upsampler.py`,
a single-frame mode in `vae.py` and `vae_ops.py`, and a multi-state hook in `load.py`.

## 1. The text connector

H3 read one hidden state of its text encoder. P-H3-Image reads five of a much smaller one and maps them to the
transformer's 5120-wide text input with one linear layer between two standardisations. Token 0 is an attention sink
that carries the same state for every prompt, so it is replaced by a learned embedding.

📍 [`diffusers:src/diffusers/models/condition_embedders/condition_embedder_p_h3_image.py:61-77`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/condition_embedders/condition_embedder_p_h3_image.py#L61-L77)
```python
  61     def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
  62         r"""
  63         Args:
  64             hidden_states (`tuple[torch.Tensor, ...]`):
  65                 The hidden states of the text encoder, as `output_hidden_states=True` returns them: one tensor of shape
  66                 `(batch_size, num_tokens, text_hidden_size)` per entry.
  67
  68         Returns:
  69             `torch.Tensor`: the conditioning, of shape `(batch_size, num_tokens, out_channels)`, in the dtype of the
  70             connector's parameters.
  71         """
  72         dtype = self.dtype
  73         hidden_states = torch.cat([hidden_states[index] for index in self.config.hidden_state_indices], dim=-1)
  74         hidden_states = (hidden_states.to(dtype) - self.input_mean) / self.input_std
  75         hidden_states = self.proj(hidden_states) * self.output_std + self.output_mean
  76         sink = self.sink_embedding.expand(hidden_states.shape[0], 1, -1)
  77         return torch.cat([sink, hidden_states[:, 1:]], dim=1)
```

The released connector keeps every parameter in float32 (`_keep_in_fp32_modules` lists all six), and the pipeline casts
the result to the text encoder's dtype. Ours does the same arithmetic in float32 and casts after:

📍 [`src/h3mlx/image.py:140-148`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L140-L148)
```python
 140 def connect(p: dict, states, out_dtype=mx.bfloat16) -> mx.array:
 141     """`PH3ImageTextConnector`: concatenate the hidden states, standardise, one linear map, destandardise, and put the
 142     learned sink in place of token 0. In float32 (the connector is kept in float32), cast to `out_dtype` after."""
 143     f32 = mx.float32
 144     x = mx.concatenate([s.astype(f32) for s in states], axis=-1)
 145     x = (x - p["input_mean"].astype(f32)) / p["input_std"].astype(f32)
 146     x = (x @ p["proj.weight"].astype(f32).T) * p["output_std"].astype(f32) + p["output_mean"].astype(f32)
 147     sink = mx.broadcast_to(p["sink_embedding"].astype(f32), (x.shape[0], 1, x.shape[-1]))
 148     return mx.concatenate([sink, x[:, 1:]], axis=1).astype(out_dtype)
```

The cast target is the text encoder's dtype in both: the pipeline passes `dtype=components.text_encoder.dtype`, and
`generate` uses the dtype the hidden states come out in unless `Components.text_dtype` sets one. For the release
that is bfloat16. `test_connector_matches_diffusers` compares the two on random parameters, including that token 0 is
the sink exactly.

## 2. Five hidden states of the text encoder

The pipeline runs the whole Qwen3-VL with `output_hidden_states=True` and hands the tuple to the connector:

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/encoders.py:68-111`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/encoders.py#L68-L111)
```python
  68 def get_p_h3_image_prompt_embeds(
  69     text_encoder,
  70     text_connector,
  71     processor,
  72     token_ids: list[int],
  73     vision_inputs: dict | None = None,
  74     device: torch.device | None = None,
  75     dtype: torch.dtype | None = None,
  76 ) -> torch.Tensor:
  77     r"""
  78     Encode one tokenized P-H3-Image presentation: the hidden states of the text encoder, mapped to the transformer's
  79     text conditioning by the text connector.
  80
  81     Args:
  82         text_encoder (`Qwen3VLForConditionalGeneration`): The text encoder.
  83         text_connector (`PH3ImageTextConnector`): Maps the text encoder's hidden states to the conditioning.
  84         processor (`Qwen3VLProcessor`): The text encoder's processor, used for the token type ids.
  85         token_ids (`list[int]`): The tokenized presentation.
  86         vision_inputs (`dict`, *optional*):
  87             The `pixel_values` and `image_grid_thw` of the presentation's image.
  88         device (`torch.device`, *optional*): The device to run the text encoder on.
  89         dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.
  90
  91     Returns:
  92         `torch.Tensor`: the `(1, num_text_tokens, 5120)` conditioning.
  93     """
  94     input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
  95     mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
  96
  97     vision_kwargs = {}
  98     for name, value in (vision_inputs or {}).items():
  99         vision_kwargs[name] = value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)
 100
 101     # Only the hidden states are used; `logits_to_keep=1` keeps the language-model head to a single token.
 102     outputs = text_encoder(
 103         input_ids=input_ids,
 104         attention_mask=torch.ones_like(input_ids),
 105         mm_token_type_ids=mm_token_type_ids,
 106         use_cache=False,
 107         output_hidden_states=True,
 108         logits_to_keep=1,
 109         **vision_kwargs,
 110     )
 111     return text_connector(outputs.hidden_states).to(device=device, dtype=dtype)
```

In transformers' numbering, entry 0 is the embedding output, entry `i` the output of decoder layer `i`, and the last
entry (36 here) the final normed state. Two details decide what "the output of layer `i`" means:

- **Deepstack.** Qwen3-VL adds visual features to the hidden states of its first three decoder layers, after each
  layer returns. transformers records a layer's output as the layer returns it, so the recorded state is before that
  add. The add itself reaches every later layer, so it does shape an edit's conditioning; what does not matter for
  P-H3-Image is the recording point, because it reads layers 13 and up, past the three that have an add. The test
  pins the recording point anyway.
- **The last entry is normed.** transformers replaces the last recorded state with the model's final output, after
  the final RMSNorm.

mlx-vlm's Qwen3-VL returns one state at a time:

📍 [`mlx-vlm:mlx_vlm/models/qwen3_vl/qwen3_vl.py:164-185`](https://github.com/Blaizzy/mlx-vlm/blob/527c513a0768245644a70266a22126d880767b18/mlx_vlm/models/qwen3_vl/qwen3_vl.py#L164-L185)
```python
 164     def hidden_state_at_layer(
 165         self,
 166         input_ids: mx.array,
 167         layer: int,
 168         pixel_values: Optional[mx.array] = None,
 169         mask: Optional[mx.array] = None,
 170         **kwargs,
 171     ) -> mx.array:
 172         """Return the pre-final-norm hidden state after ``layer`` decoder layers."""
 173         features = self.get_input_embeddings(
 174             input_ids, pixel_values, mask=mask, **kwargs
 175         )
 176         return self.language_model.model(
 177             input_ids,
 178             inputs_embeds=features.inputs_embeds,
 179             mask=mask,
 180             position_ids=features.position_ids,
 181             visual_pos_masks=features.visual_pos_masks,
 182             deepstack_visual_embeds=features.deepstack_visual_embeds,
 183             stop_after_layer=layer,
 184             apply_final_norm=False,
 185         )
```

Rather than run the model five times, ours records layer outputs as they happen. The H3 port already wraps every
decoder layer with `eval_each_layer`, so each layer evaluates its own output (one long graph over a 7,000-token prompt
tripped macOS's GPU watchdog). That wrapper now also appends the layer's return to a list when one is registered for
it:

📍 [`src/h3mlx/load.py:113-130`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/load.py#L113-L130)
```python
 113 def eval_each_layer(layers) -> None:
 114     """Make every decoder layer evaluate its output, so the 50-layer pass is 50 small GPU graphs instead of one. A single
 115     long graph at 7,000+ tokens tripped macOS's GPU watchdog (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
 116     in a pre-flight probe. The layer keeps its class's attributes and parameters: only `__call__` changes."""
 117     for layer in layers:
 118         base = type(layer)
 119         if base in _EVAL_EACH.values():
 120             continue
 121         if base not in _EVAL_EACH:
 122             def call(self, *a, _base=base, **k):
 123                 out = _base.__call__(self, *a, **k)
 124                 mx.eval(out)
 125                 sink = _CAPTURE.get(id(self))
 126                 if sink is not None:
 127                     sink.append(out)
 128                 return out
 129             _EVAL_EACH[base] = type(base.__name__ + "EvalEach", (base,), {"__call__": call})
 130         layer.__class__ = _EVAL_EACH[base]
```
📍 [`src/h3mlx/load.py:175-209`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/load.py#L175-L209)
```python
 175 def hidden_states_at(model, token_ids, vision_inputs, indices, num_layers: int) -> list:
 176     """transformers' `output_hidden_states=True` entries `indices` of an mlx-vlm Qwen3-VL whose decoder layers went
 177     through `eval_each_layer`: 0 is the embedding output, `i` the output of decoder layer `i` (as the layer returns
 178     it, before a deepstack add), and `num_layers` the final normed state. `num_layers` is the release's layer count,
 179     which the model must hold in full when it is asked for."""
 180     kw = {k: mx.array(np.asarray(v)) for k, v in (vision_inputs or {}).items()}
 181     ids = mx.array([token_ids])
 182     feats = model.get_input_embeddings(ids, kw.pop("pixel_values", None), mask=None, **kw)
 183     lm = model.language_model.model
 184     built = len(lm.layers)
 185     for i in indices:
 186         if not 0 <= i <= num_layers or (i == num_layers and built != num_layers) or (0 < i < num_layers and i > built):
 187             raise ValueError(f"hidden state {i} is not available from {built} of {num_layers} layers")
 188     sinks = {i: [] for i in indices if 0 < i < num_layers}
 189     for i, sink in sinks.items():
 190         _CAPTURE[id(lm.layers[i - 1])] = sink
 191     try:
 192         last = max(i for i in indices)
 193         final = lm(ids, inputs_embeds=feats.inputs_embeds, mask=None, position_ids=feats.position_ids,
 194                    visual_pos_masks=feats.visual_pos_masks, deepstack_visual_embeds=feats.deepstack_visual_embeds,
 195                    stop_after_layer=None if last == num_layers else last, apply_final_norm=last == num_layers)
 196     finally:
 197         for i in sinks:
 198             _CAPTURE.pop(id(lm.layers[i - 1]), None)
 199     out = []
 200     for i in indices:
 201         if i == 0:
 202             out.append(feats.inputs_embeds)
 203         elif i == num_layers:
 204             out.append(final)
 205         else:
 206             if len(sinks[i]) != 1:
 207                 raise RuntimeError(f"decoder layer {i} ran {len(sinks[i])} times; were its layers wrapped by eval_each_layer?")
 208             out.append(sinks[i][0])
 209     return out
```

The capture lists live in a module-level dict keyed by the layer's id: an mlx `Module` stores a list attribute as a
child, which would put the capture into its parameter tree. The registrations are removed in `finally`, so the H3
video pipeline (which reads one state through `hidden_state_at_layer`) never sees them.
`test_hidden_states_and_connector_match_pipeline` runs `get_p_h3_image_prompt_embeds` against
`hidden_states_at` + `connect` on a tiny 4-layer Qwen3-VL with deepstack, two pictures, and indices 1 to 4: two of
them are deepstack layers and the last is the final norm.

## 3. What the text encoder reads

**Text-to-image:** the prompt as written. No chat template, no special tokens, every row tagged text.

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/encoders.py:150-174`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/encoders.py#L150-L174)
```python
 150     @torch.no_grad()
 151     def __call__(
 152         self, components: PH3ImageModularPipeline, state: PipelineState
 153     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 154         block_state = self.get_block_state(state)
 155         if not isinstance(block_state.prompt, str):
 156             raise ValueError(
 157                 "P-H3-Image packs one request into one sequence, so `prompt` must be a single string, got "
 158                 f"{type(block_state.prompt)}."
 159             )
 160
 161         token_ids = components.tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"]
 162
 163         block_state.prompt_embeds = get_p_h3_image_prompt_embeds(
 164             components.text_encoder,
 165             components.text_connector,
 166             components.processor,
 167             token_ids,
 168             device=components._execution_device,
 169             dtype=components.text_encoder.dtype,
 170         )
 171         block_state.text_token_tags = torch.full((len(token_ids),), components.text_tag, dtype=torch.long)
 172
 173         self.set_block_state(state, block_state)
 174         return components, state
```
📍 [`src/h3mlx/image.py:166-168`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L166-L168)
```python
 166 def text2image_presentation(c: Components, prompt: str):
 167     ids = c.tokenizer(prompt, add_special_tokens=False)["input_ids"]
 168     return ids, [1] * len(ids), {}
```

**Edit:** H3's ref2va presentation with pictures only (`"<Picture i>: "` and a vision block per picture, the block's
rows tagged video), then the instruction with `"Edit <Picture 1>. "` in front, unless it already starts with that or
with `subject_definitions:`, the first line of an upsampler brief (§9).

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/encoders.py:225-269`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/encoders.py#L225-L269)
```python
 225     @torch.no_grad()
 226     def __call__(
 227         self, components: PH3ImageModularPipeline, state: PipelineState
 228     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 229         block_state = self.get_block_state(state)
 230         if not isinstance(block_state.prompt, str):
 231             raise ValueError(
 232                 "P-H3-Image packs one request into one sequence, so `prompt` must be a single string, got "
 233                 f"{type(block_state.prompt)}."
 234             )
 235         tokenizer, processor = components.tokenizer, components.processor
 236         text_tag, video_tag = components.text_tag, components.video_tag
 237
 238         vision = processor.image_processor(images=block_state.vision_images, return_tensors="pt")
 239         merge_size = processor.image_processor.merge_size**2
 240         token_ids, token_tags = [], []
 241         for index, grid in enumerate(vision["image_grid_thw"]):
 242             label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
 243             vision_ids = (
 244                 [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
 245                 + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge_size)
 246                 + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
 247             )
 248             token_ids += label_ids + vision_ids
 249             token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)
 250         prompt = block_state.prompt
 251         if not prompt.lstrip().startswith(("Edit <Picture 1>.", "subject_definitions:")):
 252             prompt = f"Edit <Picture 1>. {prompt}"
 253         prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
 254         token_ids += prompt_ids
 255         token_tags += [text_tag] * len(prompt_ids)
 256
 257         block_state.prompt_embeds = get_p_h3_image_prompt_embeds(
 258             components.text_encoder,
 259             components.text_connector,
 260             processor,
 261             token_ids,
 262             {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]},
 263             device=components._execution_device,
 264             dtype=components.text_encoder.dtype,
 265         )
 266         block_state.text_token_tags = torch.tensor(token_tags, dtype=torch.long)
 267
 268         self.set_block_state(state, block_state)
 269         return components, state
```

Ours reuses the H3 presentation builder (main guide §5) with image references only:

📍 [`src/h3mlx/image.py:151-154`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L151-L154)
```python
 151 def edit_prompt(prompt: str) -> str:
 152     """The instruction as the edit presentation encodes it: `"Edit <Picture 1>. "` in front, unless it already starts
 153     with that or is an upsampler brief."""
 154     return prompt if prompt.lstrip().startswith(EDIT_PREFIXES) else f"Edit <Picture 1>. {prompt}"
```
📍 [`src/h3mlx/image.py:157-163`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L157-L163)
```python
 157 def edit_presentation(c: Components, prompt: str, vision_images):
 158     """`PH3ImageEditTextEncoderStep`'s tokens: (ids, tags, vision inputs)."""
 159     f = c.processor.image_processor(images=list(vision_images), return_tensors="np")
 160     merge = c.processor.image_processor.merge_size**2
 161     counts = [int(np.prod(g)) // merge for g in f["image_grid_thw"]]
 162     ids, tags = build_presentation(c.tokenizer, edit_prompt(prompt), [Ref("image")] * len(counts), counts, [], [])
 163     return ids, tags, {"pixel_values": f["pixel_values"], "image_grid_thw": f["image_grid_thw"]}
```

The pictures the encoder sees are the ~1 MP copies from §4, not the ones the VAE encodes. The end-to-end tests (§10)
compare the token tags exactly and the prompt embeddings to 1e-4. A tiny random model barely reads its prompt (a wholly
different prompt moves its image by about 2e-3), so the image alone could not catch a presentation error.

## 4. Edit sizes

An edit resizes every picture twice: for the VAE, each keeps its own size unless it is under 1 MP or over 8 MP; for the
text encoder, each goes to about 1 MP. Without `height` and `width`, the canvas is Picture 1's VAE size, scaled up to at
least 4 MP. Sides are multiples of 32 throughout (the VAE's 16 times the patch's 2). Passing `height` and `width` sets
the canvas only; the pictures keep their sizes.

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_encoder.py:31-43`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py#L31-L43)
```python
  31 def _size_at_area(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
  32     r"""
  33     The `(width, height)` of an image resized to about `pixels` at its own aspect ratio, with both sides multiples of
  34     `multiple`. An image with sides that are already multiples of `multiple` and an area within 20% of `pixels` keeps
  35     its size.
  36     """
  37     if width % multiple == 0 and height % multiple == 0 and 0.8 * pixels <= width * height <= 1.2 * pixels:
  38         return width, height
  39     ratio = width / height
  40     return (
  41         max(multiple, round(math.sqrt(pixels * ratio) / multiple) * multiple),
  42         max(multiple, round(math.sqrt(pixels / ratio) / multiple) * multiple),
  43     )
```
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
📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_encoder.py:150-182`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_encoder.py#L150-L182)
```python
 150     @torch.no_grad()
 151     def __call__(
 152         self, components: PH3ImageModularPipeline, state: PipelineState
 153     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 154         block_state = self.get_block_state(state)
 155         images = block_state.image if isinstance(block_state.image, list) else [block_state.image]
 156         images = [image.convert("RGB") for image in images]
 157         if len(images) > MAX_EDIT_IMAGES:
 158             raise ValueError(f"An edit takes at most {MAX_EDIT_IMAGES} images, got {len(images)}.")
 159
 160         if (block_state.height is None) != (block_state.width is None):
 161             raise ValueError("`height` and `width` have to be passed together, or neither of them.")
 162         multiple = components.canvas_multiple
 163         min_pixels, max_pixels = components.config.reference_min_pixels, components.config.reference_max_pixels
 164         reference_sizes = [_size_in_area_range(*image.size, min_pixels, max_pixels, multiple) for image in images]
 165         if block_state.height is not None:
 166             block_state.height, block_state.width = round_canvas(block_state.height, block_state.width, multiple)
 167         else:
 168             canvas_min_pixels = max(components.config.canvas_min_pixels, min_pixels)
 169             reference_sizes[0] = _size_in_area_range(*images[0].size, canvas_min_pixels, max_pixels, multiple)
 170             block_state.width, block_state.height = reference_sizes[0]
 171         vision_sizes = [_size_at_area(*image.size, components.config.vision_pixels, multiple) for image in images]
 172
 173         def resize(image, size):
 174             if image.size == size:
 175                 return image
 176             return components.image_processor.resize(image, height=size[1], width=size[0])
 177
 178         block_state.reference_images = [resize(image, size) for image, size in zip(images, reference_sizes)]
 179         block_state.vision_images = [resize(image, size) for image, size in zip(images, vision_sizes)]
 180
 181         self.set_block_state(state, block_state)
 182         return components, state
```

Ours is the same arithmetic. Python's `round` (half to even) is the same function on both sides, and the resize is
PIL's Lanczos, which `VaeImageProcessor.resize` uses by default:

📍 [`src/h3mlx/image.py:80-87`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L80-L87)
```python
  80 def size_at_area(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
  81     """`_size_at_area`: (width, height) at about `pixels`, own aspect ratio, sides multiples of `multiple`; a size
  82     already on the multiple and within 20% of `pixels` is kept."""
  83     if width % multiple == 0 and height % multiple == 0 and 0.8 * pixels <= width * height <= 1.2 * pixels:
  84         return width, height
  85     ratio = width / height
  86     return (max(multiple, round(math.sqrt(pixels * ratio) / multiple) * multiple),
  87             max(multiple, round(math.sqrt(pixels / ratio) / multiple) * multiple))
```
📍 [`src/h3mlx/image.py:90-94`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L90-L94)
```python
  90 def size_in_area_range(width: int, height: int, min_pixels: int, max_pixels: int, multiple: int) -> tuple[int, int]:
  91     """`_size_in_area_range`: own size, or scaled to the nearer bound when outside it; sides rounded to `multiple`."""
  92     area = width * height
  93     scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
  94     return tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (width, height))
```
📍 [`src/h3mlx/image.py:115-134`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L115-L134)
```python
 115 def prepare_edit(c: Components, images, height=None, width=None):
 116     """`PH3ImageEditResizeStep`: -> (height, width, reference_images for the VAE, vision_images for the encoder)."""
 117     images = [im.convert("RGB") for im in (images if isinstance(images, (list, tuple)) else [images])]
 118     if not images:
 119         raise ValueError("an edit needs at least one image")
 120     if len(images) > MAX_EDIT_IMAGES:
 121         raise ValueError(f"an edit takes at most {MAX_EDIT_IMAGES} images, got {len(images)}")
 122     if (height is None) != (width is None):
 123         raise ValueError("height and width have to be passed together, or neither of them")
 124     s, m = c.settings, c.canvas_multiple
 125     ref_sizes = [size_in_area_range(*im.size, s.reference_min_pixels, s.reference_max_pixels, m) for im in images]
 126     if height is not None:
 127         height, width = round_canvas(height, width, m)
 128     else:
 129         canvas_min = max(s.canvas_min_pixels, s.reference_min_pixels)
 130         ref_sizes[0] = size_in_area_range(*images[0].size, canvas_min, s.reference_max_pixels, m)
 131         width, height = ref_sizes[0]
 132     vision_sizes = [size_at_area(*im.size, s.vision_pixels, m) for im in images]
 133     return (height, width, [resize(im, sz) for im, sz in zip(images, ref_sizes)],
 134             [resize(im, sz) for im, sz in zip(images, vision_sizes)])
```

`test_size_rules_match` runs both size functions and `round_canvas` on 400 random sizes;
`test_prepare_edit_matches_resize_step` runs diffusers' own `PH3ImageEditResizeStep` as a pipeline and compares the
canvas and every resized picture pixel for pixel, with and without a canvas.

## 5. Encoding the pictures

The pictures go through H3's conditioning recipe unchanged: ImageNet normalisation, the posterior *sampled* under a
fresh generator seeded 42 for each picture, a float16 round trip, then the latent normalisation. A single frame takes the
VAE's tiled spatial encoder alone, as an H3 image reference does.

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/encoders.py:31-65`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/encoders.py#L31-L65)
```python
  31 def encode_vae_condition(
  32     vae, pixels: torch.Tensor, pixel_mean: tuple, pixel_std: tuple, encode_seed: int = 42
  33 ) -> torch.Tensor:
  34     r"""
  35     Encode one visual condition — a keyframe or a `ref2va` reference — into normalized conditioning latents.
  36
  37     The recipe is the released model's, and every part of it is needed to reproduce its conditioning: the pixels are
  38     ImageNet-normalized, the posterior is *sampled* under a fresh generator seeded independently of the request, and
  39     the sampled latent is rounded to float16 (~11 bits of every conditioning latent) before being normalized. A single
  40     frame is encoded by the (tiled) spatial encoder alone; a frame stack goes through the temporal chunking, which is
  41     what turns `17 * n + 5` frames into `5 * n + 2` latent frames.
  42
  43     Args:
  44         vae (`AutoencoderKLPH3Image`): The video VAE.
  45         pixels (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`):
  46             The condition's `uint8` pixels, on the VAE's device. `num_frames` is 1 for a keyframe or an image
  47             reference.
  48         pixel_mean (`tuple[float, float, float]`), pixel_std (`tuple[float, float, float]`):
  49             The video VAE's pixel convention, i.e. `components.pixel_mean` / `components.pixel_std`.
  50         encode_seed (`int`, *optional*, defaults to 42):
  51             Seed the posterior is sampled under, i.e. `components.keyframe_encode_seed`.
  52
  53     Returns:
  54         `torch.Tensor`: one `(1, latent_channels, num_latent_frames, latent_height, latent_width)` float32 CPU tensor.
  55     """
  56     latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
  57     latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
  58     pixel_mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
  59     pixel_std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)
  60
  61     pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
  62     posterior = vae.encode(pixels, return_dict=False)[0]
  63     latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
  64     latents = latents.to(torch.float16).float().cpu()
  65     return (latents - latents_mean) / latents_std
```
📍 [`src/h3mlx/image.py:182-194`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L182-L194)
```python
 182 def encode_pictures(c: Components, reference_images) -> list:
 183     """`PH3ImageEditVaeEncoderStep`: H3's conditioning recipe per picture, a fresh seed-42 generator each."""
 184     out = []
 185     for im in reference_images:
 186         fresh = Noise(ENCODE_SEED)
 187
 188         def noise_cl(shape_cl, fresh=fresh):  # torch draws (B, C, F, H, W); hand back channels-last
 189             b, f, h, w, ch = shape_cl
 190             return mx.array(fresh((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 191
 192         px = mx.array(np.asarray(im)[None, None])  # (1, 1, H, W, 3) uint8
 193         out.append(encode_condition(c.vae, px, PIXEL_MEAN, PIXEL_STD, c.vae_latents_mean, c.vae_latents_std, noise_cl))
 194     return out
```

`encode_condition` is the H3 port's (main guide §6). `test_image_encode_and_condition_recipe` compares the moments and
the final latents against diffusers' function on the image VAE. The float16 round trip means an encoder difference of
1e-6 can land one float16 step apart, so the end-to-end test bounds those rows by how many elements differ, not only by
the largest difference.

## 6. The packed layouts

Both layouts are H3's. Text-to-image is the t2va layout for one latent frame and two audio latents per channel. Its
`height` and `width` default to 2912 each on their own, then are rounded to multiples of 32:

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:336-380`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py#L336-L380)
```python
 336     @torch.no_grad()
 337     def __call__(
 338         self, components: PH3ImageModularPipeline, state: PipelineState
 339     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 340         block_state = self.get_block_state(state)
 341         device = components._execution_device
 342
 343         block_state.height, block_state.width = round_canvas(
 344             block_state.height, block_state.width, components.canvas_multiple
 345         )
 346         ratio = components.vae_spatial_compression_ratio
 347         block_state.num_latent_frames = 1
 348         block_state.latent_height = block_state.height // ratio
 349         block_state.latent_width = block_state.width // ratio
 350         block_state.num_audio_latents = components.num_audio_latents
 351
 352         (
 353             position_ids,
 354             token_tags,
 355             video_indices,
 356             audio_indices,
 357             text_indices,
 358             block_state.num_condition_video_rows,
 359             block_state.num_condition_audio_rows,
 360         ) = self.build_packed_sequence(
 361             block_state.text_token_tags,
 362             block_state.num_latent_frames,
 363             block_state.latent_height,
 364             block_state.latent_width,
 365             block_state.num_audio_latents,
 366             components.patch_size,
 367             components.audio_channels,
 368             components.audio_tag,
 369             components.video_tag,
 370         )
 371         # MPS, NPU and Neuron have no float64; the rope casts the grid to float32 anyway.
 372         position_ids_dtype = maybe_adjust_dtype_for_device(position_ids.dtype, device)
 373         block_state.position_ids = position_ids.to(device, position_ids_dtype)
 374         block_state.token_tags = token_tags.to(device)
 375         block_state.video_indices = video_indices.to(device)
 376         block_state.audio_indices = audio_indices.to(device)
 377         block_state.text_indices = text_indices.to(device)
 378
 379         self.set_block_state(state, block_state)
 380         return components, state
```
📍 [`src/h3mlx/image.py:105-107`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L105-L107)
```python
 105 def text2image_size(height, width, multiple: int) -> tuple[int, int]:
 106     """`PH3ImagePrepareLayoutStep`: each side defaults to 2912 on its own, then both are rounded to the multiple."""
 107     return round_canvas(height or DEFAULT_SIZE, width or DEFAULT_SIZE, multiple)
```

An edit is the ref2va layout with image references only: each picture takes one integer slot on the rotary clock, and
the generated rows share the slot after the last picture.

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:470-556`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py#L470-L556)
```python
 470     @staticmethod
 471     def build_edit_packed_sequence(
 472         text_token_tags: torch.Tensor,
 473         condition_latents: list[torch.Tensor],
 474         latent_height: int,
 475         latent_width: int,
 476         num_audio_latents: int,
 477         patch_size: tuple[int, int, int],
 478         audio_channels: int,
 479         audio_tag: int,
 480         video_tag: int,
 481     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
 482         r"""
 483         Build the `[text | images | target audio | target image]` layout of an edit: MiniMax-H3's `ref2va` layout with
 484         image references only.
 485
 486         Args:
 487             text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
 488                 The modality tag of every text row. Text is tagged `1`, except for the rows of an image's vision block,
 489                 which are tagged `0` (video).
 490             condition_latents (`list[torch.Tensor]`):
 491                 One `(1, channels, 1, latent_height, latent_width)` tensor per image of the edit, in packed order.
 492             latent_height (`int`): Target latent height.
 493             latent_width (`int`): Target latent width.
 494             num_audio_latents (`int`): Number of target audio latents per channel.
 495             patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
 496             audio_channels (`int`): Channels the audio rows are packed channel-major over, i.e.
 497                 `components.audio_channels`.
 498             audio_tag (`int`), video_tag (`int`):
 499                 The modality tags an audio and an image row carry, i.e. `components.audio_tag` and
 500                 `components.video_tag`.
 501
 502         Returns:
 503             `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
 504             leading image and audio rows that are conditioning rather than generated.
 505         """
 506         _, patch_h, patch_w = patch_size
 507         num_text_tokens = text_token_tags.shape[0]
 508         num_condition_rows = sum(
 509             (latents.shape[3] // patch_h) * (latents.shape[4] // patch_w) for latents in condition_latents
 510         )
 511         num_audio_rows = num_audio_latents * audio_channels
 512         num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
 513         sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_target_rows
 514
 515         position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
 516         position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
 517
 518         # The images, in request order. Each takes a single integer slot on the rotary clock, which starts where the
 519         # text rows end.
 520         cursor = num_text_tokens
 521         rotary_time = float(num_text_tokens)
 522         for latents in condition_latents:
 523             frame_grid, _ = _frame_position_grid(latents.shape[3], latents.shape[4], patch_h, patch_w)
 524             rows = slice(cursor, cursor + frame_grid.shape[0])
 525             position_ids[rows, 0] = rotary_time
 526             position_ids[rows, 1:] = frame_grid
 527             cursor = rows.stop
 528             rotary_time += 1.0
 529
 530         # The generated rows. Target audio and target image share the origin the images left behind.
 531         audio_start = cursor
 532         video_start = audio_start + num_audio_rows
 533         target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
 534         _fill_audio_positions(
 535             position_ids,
 536             slice(audio_start, video_start),
 537             num_audio_latents,
 538             rotary_time,
 539             target_width_grid,
 540             audio_channels,
 541         )
 542         position_ids[video_start:, 0] = rotary_time
 543         position_ids[video_start:, 1:] = target_frame_grid
 544
 545         video_indices = torch.cat(
 546             [torch.arange(num_text_tokens, audio_start), torch.arange(video_start, sequence_length)]
 547         )
 548         audio_indices = torch.arange(audio_start, video_start)
 549         text_indices = torch.arange(num_text_tokens)
 550
 551         token_tags = torch.empty(sequence_length, dtype=torch.long)
 552         token_tags[text_indices] = text_token_tags.to(torch.long)
 553         token_tags[audio_indices] = audio_tag
 554         token_tags[video_indices] = video_tag
 555
 556         return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0
```

Ours calls the two packers the video pipeline uses (main guide §1), so no layout code is new:

📍 [`src/h3mlx/image.py:274-279`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L274-L279)
```python
 274     if edit:
 275         layout = build_ref2va_packed_sequence(np.asarray(tags), [Ref("image")] * len(cond),
 276                                               [tuple(x.shape[1:4]) for x in cond], [], 1, lh, lw, NUM_AUDIO_LATENTS,
 277                                               patch, AUDIO_CHANNELS)
 278     else:
 279         layout = build_packed_sequence(np.asarray(tags), 1, lh, lw, NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)
```

`test_text2image_layout_matches` and `test_edit_layout_matches` compare positions (float64, bit for bit), tags, indices
and the conditioning-row counts for three canvases and for one, three and eight pictures.

## 7. Noise and the schedule

The draw order is H3's: one draw per picture from the request's generator (each noised to `t = 0.999` and held there),
then the target image, then the audio rows.

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
📍 [`src/h3mlx/image.py:283-293`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L283-L293)
```python
 283     gen = Noise(seed)
 284     rows = []
 285     for x in cond:  # one draw per picture, packed order, then held at t = 0.999 (float32, as the pipeline mixes it)
 286         b, f, h, w, ch = x.shape
 287         nz = mx.array(gen((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 288         t32 = np.float32(NOISE_AUG)
 289         noised = mx.array(t32) * x + mx.array(np.float32(1) - t32) * nz
 290         rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
 291     rows.append(ops.patchify(mx.array(gen((1, c.dit_cfg["in_channels"], 1, lh, lw))), patch))
 292     video_rows = mx.concatenate(rows)
 293     audio_rows = mx.array(gen((NUM_AUDIO_LATENTS * AUDIO_CHANNELS, c.dit_cfg["audio_in_channels"])))
```

Both schedulers get the checkpoint's own 9-point grid. The released scheduler config sets
`shift_provided_sigmas: false`, so the grid is installed verbatim (no shift), and the 12 / 3 shifts of H3 play no part:

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/before_denoise.py:946-978`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/before_denoise.py#L946-L978)
```python
 946     @torch.no_grad()
 947     def __call__(
 948         self, components: PH3ImageModularPipeline, state: PipelineState
 949     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 950         block_state = self.get_block_state(state)
 951         device = components._execution_device
 952
 953         sigmas = block_state.sigmas if block_state.sigmas is not None else components.config.sample_sigmas
 954         components.scheduler.set_timesteps(sigmas=sigmas, device=device)
 955         components.audio_scheduler.set_timesteps(sigmas=sigmas, device=device)
 956         block_state.timesteps = components.scheduler.timesteps
 957         block_state.audio_timesteps = components.audio_scheduler.timesteps
 958
 959         block_state.row_timestep_plan = [
 960             tuple(
 961                 tensor.to(device)
 962                 for tensor in self.build_row_timesteps(
 963                     block_state.video_indices,
 964                     block_state.audio_indices,
 965                     block_state.num_condition_video_rows,
 966                     block_state.num_condition_audio_rows,
 967                     block_state.text_indices.numel(),
 968                     float(timestep),
 969                     float(audio_timestep),
 970                     max(float(timestep), components.keyframe_noise_aug),
 971                     1.0,
 972                 )
 973             )
 974             for timestep, audio_timestep in zip(block_state.timesteps, block_state.audio_timesteps)
 975         ]
 976
 977         self.set_block_state(state, block_state)
 978         return components, state
```
📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:142-203`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/schedulers/scheduling_minimax_h3.py#L142-L203)
```python
 142     def set_timesteps(
 143         self,
 144         num_inference_steps: int | None = None,
 145         device: str | torch.device | None = None,
 146         sigmas: list[float] | torch.Tensor | None = None,
 147     ) -> None:
 148         r"""
 149         Build the sigma / timestep schedule.
 150
 151         The grid is `linspace(1, 0, num_inference_steps)` pushed through the exponential shift, with consecutive
 152         duplicates collapsed, or the explicit `sigmas`, shifted as well when `shift_provided_sigmas` is set. The
 153         terminal `0` is already part of the grid — the shift maps `0` to exactly `0` — so the schedule holds
 154         `num_inference_steps` sigmas and drives `num_inference_steps - 1` model evaluations, exposed as `self.timesteps
 155         = 1 - sigmas[:-1]`.
 156
 157         Args:
 158             num_inference_steps (`int`, *optional*):
 159                 Number of sigma grid points, terminal `0` included. Ignored when `sigmas` is given.
 160             device (`str` or `torch.device`, *optional*):
 161                 Device the schedule tensors are moved to. The grid itself is always built on CPU in float32 so the
 162                 schedule does not depend on the accelerator.
 163             sigmas (`list[float]` or `torch.Tensor`, *optional*):
 164                 An explicit grid, strictly decreasing and terminating at `0.0`. With `shift_provided_sigmas=True` it is
 165                 a base grid in `[0, 1]` that goes through the shift like the `linspace` grid; otherwise it is used
 166                 verbatim (deprecated default).
 167         """
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
 180         else:
 181             sigmas = torch.as_tensor(sigmas, dtype=torch.float64).flatten().cpu()
 182             if self.config.shift_provided_sigmas is None:
 183                 deprecate(
 184                     "sigmas",
 185                     "0.44.0",
 186                     "`MiniMaxH3Scheduler.set_timesteps` takes explicit `sigmas` verbatim, but from 0.44.0 it will "
 187                     "push them through the scheduler's shift. Set `shift_provided_sigmas=True` in the scheduler "
 188                     "config to opt in now, or `shift_provided_sigmas=False` to keep passing a finished schedule.",
 189                     standard_warn=False,
 190                 )
 191             if self.config.shift_provided_sigmas:
 192                 # Shifted in float64 before the float32 cast, which is how the few-step reference builds its schedule.
 193                 sigmas = self._shift * sigmas / (1 + (self._shift - 1) * sigmas)
 194             sigmas = sigmas.to(torch.float32)
 195             if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
 196                 raise ValueError("`sigmas` must hold at least two strictly decreasing values ending at 0.0.")
 197
 198         self.sigmas = sigmas.to(device=device)
 199         # t = 1 - sigma, and t = 1 is clean. The terminal sigma has no model evaluation.
 200         self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
 201         self.num_inference_steps = int(self.timesteps.numel())
 202         self._step_index = None
 203         self._begin_index = None
```

Ours builds H3's `Schedules` from the grid for both modalities, so the denoise loop is the H3 one, single-time:

📍 [`src/h3mlx/image.py:174-179`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L174-L179)
```python
 174 def schedules(sigmas=None) -> Schedules:
 175     """Both modalities on the same explicit grid, used verbatim (the scheduler's `shift_provided_sigmas: false`)."""
 176     s = np.asarray(SAMPLE_SIGMAS if sigmas is None else sigmas, dtype=np.float64).reshape(-1).astype(np.float32)
 177     if s.size < 2 or not (s[1:] < s[:-1]).all() or s[-1] != 0.0:
 178         raise ValueError(f"sigmas must be at least two strictly decreasing values ending at 0.0, got {s.tolist()}")
 179     return Schedules(s, s.copy(), two_time=False)
```

The grid goes through float64 to float32, as the scheduler's `torch.as_tensor(..., float64)` then `.to(float32)` does.
`test_schedule_is_the_grid_verbatim` compares sigmas and timesteps exactly with the scheduler's.

## 8. The image decoder

The image VAE keeps H3's encoder and latent space and swaps the decoder. H3's ViT decoder turns each latent voxel into a
4 × 16 × 16 pixel block, with the time axis spread over the clip:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:449-502`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L449-L502)
```python
 449     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
 450         batch_size, num_channels, num_frames, height, width = hidden_states.shape
 451
 452         hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
 453             batch_size, num_frames * height * width, num_channels
 454         )
 455         hidden_states = self.proj_in(hidden_states)
 456         num_patches = hidden_states.shape[1]
 457
 458         # `proj_in` is pinned to float32 and `register_tokens` is not, so align rather than rely on promotion.
 459         register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
 460         cls_token = torch.zeros_like(hidden_states[:, :1, :])
 461         hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)
 462
 463         grids = [
 464             2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
 465             for size in (num_frames, height, width)
 466         ]
 467         position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
 468         position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
 469         suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
 470         position_ids = torch.cat([position_ids, suffix_ids], dim=1)
 471         rotary_emb = self.rope(position_ids)
 472
 473         for block in self.transformer_blocks:
 474             if torch.is_grad_enabled() and self.gradient_checkpointing:
 475                 hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
 476             else:
 477                 hidden_states = block(hidden_states, rotary_emb)
 478
 479         # `norm_out` is pinned to float32 and `proj_out` is not.
 480         hidden_states = self.norm_out(hidden_states).to(get_parameter_dtype(self.proj_out))
 481         hidden_states = self.proj_out(hidden_states)
 482         hidden_states = hidden_states[:, :num_patches, :]
 483
 484         patch_size, patch_size_t = self.patch_size, self.patch_size_t
 485         hidden_states = hidden_states.view(
 486             batch_size,
 487             num_frames,
 488             height,
 489             width,
 490             self.out_channels,
 491             patch_size_t,
 492             patch_size,
 493             patch_size,
 494         )
 495         hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
 496         return hidden_states.reshape(
 497             batch_size,
 498             self.out_channels,
 499             num_frames * patch_size_t,
 500             height * patch_size,
 501             width * patch_size,
 502         )
```

P-H3-Image's decodes one frame. Each image-latent token becomes a 16 × 16 block, and every one of them sits at one
fixed time coordinate: the one H3's decoder gives the first frame of a 7-frame window. The register tokens and the
zero token appended after them sit at position 0 in both decoders, and are dropped before the pixels.

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py:407`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L407)
```python
 407 FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
```
📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py:456-490`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L456-L490)
```python
 456     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
 457         batch_size, num_channels, _, height, width = hidden_states.shape
 458
 459         hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, height * width, num_channels)
 460         hidden_states = self.proj_in(hidden_states)
 461         num_patches = hidden_states.shape[1]
 462
 463         register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
 464         cls_token = torch.zeros_like(hidden_states[:, :1, :])
 465         hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)
 466
 467         grids = [torch.full((1,), FRAME_POSITION, device=hidden_states.device)] + [
 468             2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
 469             for size in (height, width)
 470         ]
 471         position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
 472         position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
 473         suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
 474         position_ids = torch.cat([position_ids, suffix_ids], dim=1)
 475         rotary_emb = self.rope(position_ids)
 476
 477         for block in self.transformer_blocks:
 478             if torch.is_grad_enabled() and self.gradient_checkpointing:
 479                 hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
 480             else:
 481                 hidden_states = block(hidden_states, rotary_emb)
 482
 483         hidden_states = self.norm_out(hidden_states).to(get_parameter_dtype(self.proj_out))
 484         hidden_states = self.proj_out(hidden_states)
 485         hidden_states = hidden_states[:, :num_patches, :]
 486
 487         patch_size = self.patch_size
 488         hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
 489         hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
 490         return hidden_states.reshape(batch_size, self.out_channels, 1, height * patch_size, width * patch_size)
```

Ours adds both differences to H3's decoder as parameters, so the blocks, attention and tiling are shared. The fixed
coordinate replaces the time axis of the position grid; `_decoder` passes a temporal patch of 1 and the coordinate when
the VAE is single-frame:

📍 [`src/h3mlx/vae_ops.py:121-132`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/vae_ops.py#L121-L132)
```python
 121 def vit_positions(frames: int, height: int, width: int, extra: int, frame_position: float | None = None) -> mx.array:
 122     """Voxel centres normalised to [-1, 1) per axis, (t, h, w) in raster order, then `extra` zero rows. With
 123     `frame_position` (one frame only) the time axis is that fixed coordinate instead (P-H3-Image's decoder)."""
 124     import numpy as np
 125
 126     axes = [2.0 * (np.arange(0.5, n, dtype=np.float32) / np.float32(n)) - 1.0 for n in (frames, height, width)]
 127     if frame_position is not None:
 128         if frames != 1:
 129             raise ValueError(f"a fixed frame position decodes one latent frame, got {frames}")
 130         axes[0] = np.array([frame_position], dtype=np.float32)
 131     grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
 132     return mx.array(np.concatenate([grid, np.zeros((extra, 3), dtype=np.float32)]))
```
📍 [`src/h3mlx/vae.py:16`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/vae.py#L16)
```python
  16 IMAGE_FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
```
📍 [`src/h3mlx/vae.py:191-197`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/vae.py#L191-L197)
```python
 191     def _decoder(self, z):
 192         c = self.cfg
 193         z = V.causal_conv3d(z, self.p["post_quant_conv.weight"], self.p["post_quant_conv.bias"])
 194         return V.vit_decoder(z, self.dec, c.decoder_num_layers, c.decoder_num_attention_heads, c.spatial_ratio,
 195                              1 if c.single_frame else c.temporal_ratio, 3,
 196                              int(c.decoder_attention_head_dim * c.decoder_rope_dim_ratio), c.decoder_rope_theta,
 197                              c.decoder_norm_eps, IMAGE_FRAME_POSITION if c.single_frame else None)
```
📍 [`src/h3mlx/vae.py:210-244`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/vae.py#L210-L244)
```python
 210     def decode(self, z):
 211         """Latents (B, F, H, W, latent) -> pixels (B, frames, H*r, W*r, 3), chunked and cross-faded like `_decode`.
 212         A single-frame VAE decodes its one latent frame into one image frame (spatially tiled)."""
 213         c = self.cfg
 214         if c.single_frame:
 215             if z.shape[1] != 1:
 216                 raise ValueError(f"this VAE decodes single latent frames, got {z.shape[1]}")
 217             return self.decode_clip(z.astype(self.p["post_quant_conv.weight"].dtype))
 218         tcs, drop, tr = c.tokens_chunk_size, c.token_drop, c.temporal_ratio
 219         chunk_frames = tcs * tr
 220         num_tokens = z.shape[1] + drop
 221         pad = (-num_tokens) % tcs
 222         n_chunks = (num_tokens + pad) // tcs - int(drop > 0)
 223         if pad:
 224             z = mx.concatenate([z, mx.repeat(z[:, -1:], pad, axis=1)], axis=1)
 225         out, overlap = [], None
 226         for i in range(n_chunks):
 227             clip = _evaluated(self.decode_clip(z[:, i * tcs : i * tcs + tcs + c.token_overlap]))
 228             for j in range(int(drop > 0) + 1):
 229                 ch = clip[:, j * chunk_frames : j * chunk_frames + chunk_frames][:, c.frame_pre_padding :]
 230                 if j == 0:
 231                     if overlap is not None:
 232                         ch = blend(overlap, ch, c.frame_overlap, axis=1)
 233                     out.append(ch)
 234                 else:
 235                     overlap = ch
 236         if overlap is not None:
 237             out.append(overlap)
 238         dec = mx.concatenate(out, axis=1)
 239         if pad:
 240             tail = c.clip_length % tr
 241             before = z.shape[1] - pad
 242             pad_frames = sum(tail if tail and (before + k) % tcs == 0 else tr for k in range(pad))
 243             dec = dec[:, :-pad_frames]
 244         return dec
```

`VaeConfig.from_diffusers` turns the mode on from the config's `_class_name`. The image VAE's `decode` and `encode`
refuse more than one frame, and ours does too.

**Precision.** With a `torch_dtype`, diffusers loads every parameter in that dtype except the class's
`_keep_in_fp32_modules`, matched as a component of the dotted name. The two VAE classes keep different lists:
`["encoder", "quant_conv", "proj_out"]` for the image VAE, and `encoder`, `quant_conv`, the three norms and the two
layer scales for H3's. The release stores the image decoder in bf16 (its `proj_out` included) and the encoder in
float32, so loading in bf16 upcasts `proj_out` to float32. `decode` then casts the latents to `post_quant_conv`'s
dtype:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py:757-782`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py#L757-L782)
```python
 757     @apply_forward_hook
 758     def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
 759         r"""
 760         Decode a batch of latent images.
 761
 762         Args:
 763             z (`torch.Tensor`):
 764                 Input batch of latents, shape `(batch_size, latent_channels, 1, height, width)`.
 765             return_dict (`bool`, *optional*, defaults to `True`):
 766                 Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.
 767
 768         Returns:
 769             [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
 770                 The decoded images, shape `(batch_size, out_channels, 1, height, width)`.
 771         """
 772         if z.shape[2] != 1:
 773             raise ValueError(f"`AutoencoderKLPH3Image` decodes single latent frames, got {z.shape[2]} frames.")
 774         # The decoder is mixed precision, so align with the first module that consumes the latents.
 775         z = z.to(get_parameter_dtype(self.post_quant_conv))
 776         if self.use_slicing and z.shape[0] > 1:
 777             decoded = torch.cat([self._decode_clip(z_slice) for z_slice in z.split(1)])
 778         else:
 779             decoded = self._decode_clip(z)
 780         if not return_dict:
 781             return (decoded,)
 782         return DecoderOutput(sample=decoded)
```

Ours applies the same rule per class:

📍 [`src/h3mlx/load.py:58-61`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/load.py#L58-L61)
```python
  58 VAE_KEEP_F32 = {
  59     "AutoencoderKLMiniMaxH3": ("encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"),
  60     "AutoencoderKLPH3Image": ("encoder", "quant_conv", "proj_out"),
  61 }
```
📍 [`src/h3mlx/load.py:64-67`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/load.py#L64-L67)
```python
  64 def vae_dtypes(names, class_name: str, dtype) -> dict:
  65     """The dtype each parameter loads in under `dtype`, by diffusers' rule for `class_name`."""
  66     keep = VAE_KEEP_F32.get(class_name, ())
  67     return {k: mx.float32 if any(m in k.split(".") for m in keep) else dtype for k in names}
```
📍 [`src/h3mlx/load.py:70-80`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/load.py#L70-L80)
```python
  70 def load_vae(d, dtype=None, **tiling) -> tuple[VideoVAE, tuple, tuple]:
  71     """`dtype=None` keeps the stored dtypes (the H3 release is float32). With a dtype, floating weights load the way
  72     diffusers' `from_pretrained(torch_dtype=dtype)` loads them: the class's float32 modules stay float32, the rest is
  73     cast (P-H3-Image stores its decoder in bf16 with `proj_out` among the float32 modules)."""
  74     cfg = _config(d)
  75     # contiguous once: a transposed view would be copied again inside every tile's graph (121-209 tiles per reference)
  76     p = {k: (mx.contiguous(V.conv3d_weight(v)) if v.ndim == 5 else v) for k, v in load_safetensors(d).items()}
  77     if dtype is not None:
  78         want = vae_dtypes(p, cfg.get("_class_name", ""), dtype)
  79         p = {k: v.astype(want[k]) if mx.issubdtype(v.dtype, mx.floating) else v for k, v in p.items()}
  80     return VideoVAE(p, VaeConfig.from_diffusers(cfg, **tiling)), tuple(cfg["latents_mean"]), tuple(cfg["latents_std"])
```

The run script loads the image VAE in float32 by default, as h3mlx decodes H3; `--vae-dtype bfloat16` loads it the
pipeline's way. `test_vae_bf16_loading_matches_from_pretrained` compares every parameter's dtype with
`from_pretrained(torch_dtype=bfloat16)` for both classes.

The pixels are then un-normalised and rounded the way `VaeImageProcessor` rounds:

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/decoders.py:169-187`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/decoders.py#L169-L187)
```python
 169     @torch.no_grad()
 170     def __call__(
 171         self, components: PH3ImageModularPipeline, state: PipelineState
 172     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 173         block_state = self.get_block_state(state)
 174         device = components._execution_device
 175
 176         latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
 177         latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
 178         latents = block_state.latents * latents_std + latents_mean
 179
 180         image = components.vae.decode(latents, return_dict=False)[0][:, :, 0]
 181         pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1)
 182         pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1)
 183         image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)
 184         block_state.images = components.image_processor.postprocess(image, output_type=block_state.output_type)
 185
 186         self.set_block_state(state, block_state)
 187         return components, state
```
📍 [`src/h3mlx/image.py:197-205`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L197-L205)
```python
 197 def decode(c: Components, video_rows: mx.array, ncv: int, lh: int, lw: int) -> np.ndarray:
 198     """Final packed rows -> float32 (H, W, 3) in [0, 1]: conditioning rows dropped, latents de-normalised, decoded,
 199     ImageNet's normalisation undone, clamped."""
 200     lat = ops.unpatchify(video_rows[ncv:], c.dit_cfg["in_channels"], 1, lh, lw, tuple(c.dit_cfg["patch_size"]))
 201     lat = lat.transpose(0, 2, 3, 4, 1) * mx.array(c.vae_latents_std) + mx.array(c.vae_latents_mean)
 202     px = c.vae.decode(lat)[0, 0].astype(mx.float32)
 203     img = mx.clip(px * mx.array(PIXEL_STD) + mx.array(PIXEL_MEAN), 0.0, 1.0)
 204     mx.eval(img)
 205     return np.array(img)
```

`test_image_decode_matches` (three latent sizes, tiled), `test_image_decode_untiled_and_frame_position` and
`test_image_decode_bf16_policy_matches_pipeline` compare the decoder with diffusers'.

## 9. The prompt upsampler

The upsampler is a Qwen3.5-9B fine-tune. It gets the request in a chat template, with the aspect ratio for
text-to-image or the pictures for an edit, and one greedy generation writes its reasoning, `</think>`, then the prompt.
An edit's answer must be a brief in six sections or a one-paragraph instruction that says what it keeps.

📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/prompt_upsampler.py:83-135`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/prompt_upsampler.py#L83-L135)
```python
  83 def upsample_prompt(
  84     prompt_upsampler: Qwen3_5ForConditionalGeneration,
  85     processor: AutoProcessor,
  86     request: str,
  87     images: list[PIL.Image.Image] | None = None,
  88     aspect_ratio: str = "1:1",
  89     max_new_tokens: int = T2I_MAX_NEW_TOKENS,
  90 ) -> str:
  91     """
  92     The prompt P-H3-Image renders for `request`: a text-to-image prompt for `aspect_ratio`, or an edit prompt written
  93     from the prepared `images` (the image to edit first). One greedy generation of at most `max_new_tokens`; the
  94     upsampler reasons first and the prompt is what follows `</think>`.
  95     """
  96     if not isinstance(request, str):
  97         raise ValueError(
  98             f"The prompt upsampler takes one request at a time, so `prompt` must be a single string, got "
  99             f"{type(request).__name__}."
 100         )
 101     tokenizer = processor.tokenizer
 102     if images:
 103         content = []
 104         for index in range(len(images)):
 105             content += [{"type": "text", "text": f"<Picture {index + 1}>: "}, {"type": "image"}]
 106         content.append({"type": "text", "text": f"Edit <Picture 1>. {request.strip()}"})
 107     else:
 108         content = f"Aspect ratio: {aspect_ratio}\nRequest: {request.strip()}"
 109     text = tokenizer.apply_chat_template(
 110         [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
 111     )
 112     inputs = processor(text=[text], images=images or None, return_tensors="pt").to(prompt_upsampler.device)
 113
 114     end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
 115     generated = prompt_upsampler.generate(
 116         **inputs,
 117         max_new_tokens=max_new_tokens,
 118         do_sample=False,
 119         eos_token_id=end_id,
 120         pad_token_id=tokenizer.pad_token_id,
 121     )
 122     new_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
 123     ended = end_id in new_ids
 124     answer = tokenizer.decode(new_ids[: new_ids.index(end_id)] if ended else new_ids, skip_special_tokens=False)
 125     prompt = answer.partition("</think>")[2].strip()
 126     valid = ended and answer.count("</think>") == 1 and len(prompt.split()) >= 5
 127     if images:
 128         valid = valid and is_edit_prompt(prompt)
 129     if not valid:
 130         raise RuntimeError(
 131             "The prompt upsampler did not return a usable prompt "
 132             f"({'it was cut off at max_new_tokens' if not ended else 'it does not have the expected form'}):\n"
 133             f"{answer}\nRaise `max_new_tokens`, or remove the `prompt_upsampler` step to render the prompt as given."
 134         )
 135     return prompt
```
📍 [`diffusers:src/diffusers/modular_pipelines/p_h3_image/prompt_upsampler.py:175-188`](https://github.com/huggingface/diffusers/blob/d3d8af89e5b08a352cf47bba3a451c5fed9a49b1/src/diffusers/modular_pipelines/p_h3_image/prompt_upsampler.py#L175-L188)
```python
 175     @torch.no_grad()
 176     def __call__(
 177         self, components: PH3ImageModularPipeline, state: PipelineState
 178     ) -> tuple[PH3ImageModularPipeline, PipelineState]:
 179         block_state = self.get_block_state(state)
 180         block_state.prompt = upsample_prompt(
 181             components.prompt_upsampler,
 182             components.prompt_upsampler_processor,
 183             request=block_state.prompt,
 184             aspect_ratio=aspect_ratio_label(block_state.width, block_state.height),
 185             max_new_tokens=block_state.max_new_tokens,
 186         )
 187         self.set_block_state(state, block_state)
 188         return components, state
```

Ours builds the same request with the same processor and runs the model in mlx-vlm, whose `generate_step` decodes
greedily at temperature 0:

📍 [`src/h3mlx/upsampler.py:58-67`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/upsampler.py#L58-L67)
```python
  58 def chat_text(tokenizer, request: str, n_images: int = 0, aspect_ratio: str = "1:1") -> str:
  59     if n_images:
  60         content = []
  61         for i in range(n_images):
  62             content += [{"type": "text", "text": f"<Picture {i + 1}>: "}, {"type": "image"}]
  63         content.append({"type": "text", "text": f"Edit <Picture 1>. {request.strip()}"})
  64     else:
  65         content = f"Aspect ratio: {aspect_ratio}\nRequest: {request.strip()}"
  66     return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
  67                                          add_generation_prompt=True)
```
📍 [`src/h3mlx/upsampler.py:78-93`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/upsampler.py#L78-L93)
```python
  78 def generate_ids(model, inputs: dict, end_id: int, max_new_tokens: int) -> list[int]:
  79     """Greedy tokens after the prompt, up to and including `end_id` or `max_new_tokens`."""
  80     from mlx_vlm.generate.ar import generate_step
  81
  82     kw = {k: mx.array(np.asarray(v)) for k, v in inputs.items() if k not in ("input_ids", "attention_mask",
  83                                                                                "pixel_values", "mm_token_type_ids")}
  84     pv = inputs.get("pixel_values")
  85     out = []
  86     for tok, _ in generate_step(mx.array(np.asarray(inputs["input_ids"])), model,
  87                                 None if pv is None else mx.array(np.asarray(pv)),
  88                                 mx.array(np.asarray(inputs["attention_mask"])), max_tokens=max_new_tokens,
  89                                 temperature=0.0, **kw):
  90         out.append(int(tok))
  91         if out[-1] == end_id:
  92             break
  93     return out
```
📍 [`src/h3mlx/upsampler.py:96-123`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/upsampler.py#L96-L123)
```python
  96 def upsample(model, processor, request: str, images=None, aspect_ratio: str = "1:1", max_new_tokens: int | None = None,
  97              on_answer=None) -> str:
  98     """`upsample_prompt`: a text-to-image prompt for `aspect_ratio`, or an edit prompt written from `images` (the image
  99     to edit first; prepared here). Raises if the answer was cut off or is not of the expected form."""
 100     if not isinstance(request, str):
 101         raise ValueError(f"the upsampler takes one request at a time, got {type(request).__name__}")
 102     images = [prepare_image(im) for im in (images or [])]
 103     if len(images) > MAX_EDIT_IMAGES:
 104         raise ValueError(f"the upsampler takes at most {MAX_EDIT_IMAGES} images, got {len(images)}")
 105     if max_new_tokens is None:
 106         max_new_tokens = EDIT_MAX_NEW_TOKENS if images else T2I_MAX_NEW_TOKENS
 107     tok = processor.tokenizer
 108     text = chat_text(tok, request, len(images), aspect_ratio)
 109     inputs = processor(text=[text], images=images or None, return_tensors="np")
 110     end_id = tok.convert_tokens_to_ids("<|im_end|>")
 111     new_ids = generate_ids(model, dict(inputs), end_id, max_new_tokens)
 112     ended = end_id in new_ids
 113     answer = tok.decode(new_ids[: new_ids.index(end_id)] if ended else new_ids, skip_special_tokens=False)
 114     if on_answer:
 115         on_answer(answer, len(new_ids))
 116     prompt = answer.partition("</think>")[2].strip()
 117     valid = ended and answer.count("</think>") == 1 and len(prompt.split()) >= 5
 118     if images:
 119         valid = valid and is_edit_prompt(prompt)
 120     if not valid:
 121         raise RuntimeError("the prompt upsampler did not return a usable prompt "
 122                            f"({'cut off at max_new_tokens' if not ended else 'not of the expected form'}):\n{answer}")
 123     return prompt
```

The text-to-image label is taken from the requested size before the canvas is rounded, as the pipeline does
(`run_p_h3_image.upsampler_aspect`). Greedy decoding in MLX and in torch can part ways once two logits are within
rounding of each other, so on the 9B model the prompt it writes can differ from torch's in places; the same validity
checks decide whether it is usable. The tests: `test_requests_and_answers_match_diffusers` runs diffusers'
`upsample_prompt` with a stand-in model that records its inputs and answers with fixed text, and compares input ids,
pixels, grids and the returned prompt; `test_answers_are_judged_as_diffusers_judges_them` and
`test_edit_answers_must_be_edit_prompts` feed both sides the same answers (cut off, no `</think>`, two of them, too
short, a text-to-image answer to an edit) and check they accept and refuse the same ones;
`test_tiny_prefill_logits_match_transformers` and
`test_tiny_greedy_tokens_match_transformers` compare a tiny random Qwen3.5 in transformers with the same weights in
mlx-vlm (with and without pictures). On the release, for "give her a wide straw hat", the upsampler generated 312
tokens (its reasoning, the brief and the end token) in 14 s, and the brief it wrote is valid and 224 tokens long.

## 10. End to end

📍 [`src/h3mlx/image.py:226-312`](https://github.com/sublimator/minimax-h3-mlx/blob/aad9a114c0230632f274d5c51c93c8e7c38b1739/src/h3mlx/image.py#L226-L312)
```python
 226 def generate(c: Components, prompt: str, images=None, height=None, width=None, seed=0, sigmas=None, on_step=None,
 227              debug: dict | None = None, check_finite: bool = False, on_denoised=None, free_transformer: bool = False,
 228              on_stage=None) -> np.ndarray:
 229     """One P-H3-Image request: text-to-image without `images`, an edit with them. Returns float32 (H, W, 3) in
 230     [0, 1]. The prompt is rendered as given (run the prompt upsampler first to expand a short request).
 231
 232     `debug` collects intermediates under diffusers' state names; `on_denoised(video_rows, meta)` runs before decoding;
 233     `free_transformer` makes the components single-use: the connector is dropped once the prompt is encoded, the VAE
 234     encoder once the pictures are, and `c.dit_params` before decoding. `on_stage(name)`: text, pictures, layout,
 235     denoise, decode."""
 236     dbg = debug if debug is not None else {}
 237     stage = on_stage or (lambda name: None)
 238     m = c.canvas_multiple
 239     edit = bool(images)
 240     if edit:
 241         height, width, ref_images, vision_images = prepare_edit(c, images, height, width)
 242         ids, tags, vision = edit_presentation(c, prompt, vision_images)
 243     else:
 244         height, width = text2image_size(height, width, m)
 245         ids, tags, vision = text2image_presentation(c, prompt)
 246     stage("text")
 247     states = c.encode_states(ids, vision)
 248     text = connect(c.connector, states, c.text_dtype if c.text_dtype is not None else states[0].dtype)
 249     if free_transformer:
 250         mx.eval(text)
 251         c.connector = {}
 252     dbg.update(prompt_embeds=text, text_token_tags=np.asarray(tags), token_ids=ids, height=height, width=width)
 253     if check_finite:
 254         assert_finite("prompt embeddings", text)
 255
 256     cond = []
 257     if edit:
 258         stage("pictures")
 259         cond = encode_pictures(c, ref_images)
 260         if free_transformer:
 261             mx.eval(cond)
 262             c.vae.drop_encoder()
 263             mx.clear_cache()
 264         dbg.update(condition_latents=cond)
 265         if check_finite:
 266             for i, x in enumerate(cond):
 267                 assert_finite(f"picture latents {i}", x)
 268
 269     stage("layout")
 270     ratio = c.vae.cfg.spatial_ratio
 271     lh, lw = height // ratio, width // ratio
 272     patch = tuple(c.dit_cfg["patch_size"])
 273     #@@start layout
 274     if edit:
 275         layout = build_ref2va_packed_sequence(np.asarray(tags), [Ref("image")] * len(cond),
 276                                               [tuple(x.shape[1:4]) for x in cond], [], 1, lh, lw, NUM_AUDIO_LATENTS,
 277                                               patch, AUDIO_CHANNELS)
 278     else:
 279         layout = build_packed_sequence(np.asarray(tags), 1, lh, lw, NUM_AUDIO_LATENTS, patch, AUDIO_CHANNELS)
 280     #@@end layout
 281
 282     #@@start noise-draws
 283     gen = Noise(seed)
 284     rows = []
 285     for x in cond:  # one draw per picture, packed order, then held at t = 0.999 (float32, as the pipeline mixes it)
 286         b, f, h, w, ch = x.shape
 287         nz = mx.array(gen((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 288         t32 = np.float32(NOISE_AUG)
 289         noised = mx.array(t32) * x + mx.array(np.float32(1) - t32) * nz
 290         rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
 291     rows.append(ops.patchify(mx.array(gen((1, c.dit_cfg["in_channels"], 1, lh, lw))), patch))
 292     video_rows = mx.concatenate(rows)
 293     audio_rows = mx.array(gen((NUM_AUDIO_LATENTS * AUDIO_CHANNELS, c.dit_cfg["audio_in_channels"])))
 294     #@@end noise-draws
 295     dbg.update(layout=layout, initial_video_rows=video_rows, initial_audio_rows=audio_rows)
 296
 297     stage("denoise")
 298     sched = schedules(sigmas if sigmas is not None else c.settings.sample_sigmas)
 299     v, _ = denoise(c.dit_params, c.dit_cfg, layout, text, video_rows, audio_rows, sched, on_step=on_step,
 300                    check_finite=check_finite)
 301     dbg.update(final_video_rows=v)
 302     meta = dict(ncv=layout.num_condition_video_rows, lh=lh, lw=lw)
 303     if on_denoised:
 304         on_denoised(v, meta)
 305     if free_transformer:
 306         c.dit_params.clear()
 307         mx.clear_cache()
 308     stage("decode")
 309     image = decode(c, v, **meta)
 310     if check_finite:
 311         assert_finite("decoded image", mx.array(image))
 312     return image
```

With `free_transformer`, `generate` also drops the connector once the prompt is encoded and, for an edit, the VAE
encoder once the pictures are (`VideoVAE.drop_encoder`). A text-to-image request never evaluates the encoder's lazily
loaded weights. `test_single_use_components_free_their_weights_and_change_nothing` runs an edit both ways and checks
the image is the same.

`test_end_to_end_matches_diffusers` runs diffusers' `PH3ImageBlocks` (upsampler removed) and `generate` on the same
tiny random components: text-to-image, an edit with two pictures at Picture 1's size, and one on a set canvas. It
compares the token tags exactly, the prompt embeddings to 1e-4, the noised picture rows, and the decoded images to
2e-5 (measured: under 1e-6). `tests/test_run_p_h3_image.py` saves those components in the release's own layout and
runs the command line against `PH3ImageBlocks` for both workflows, plus the re-decode path and its refusals.

**Measured on an M3 Ultra (60-core GPU, 256 GB)** with the release, 8 calls, bf16 transformer, float32 decode:

| request | size | rows | per call | total | peak |
| --- | --- | --- | --- | --- | --- |
| text-to-image, prompt as written | 2912 × 2912 | 8,395 | 26.0 s | 4.2 min | 69 GB |
| edit, one 8.5 MP picture, upsampled | 2912 × 2912 | 17,749 | 69.4 s | 10.5 min | 72 GB |

## 11. Not done, and open

- **Real-weight equivalence with torch is not measured.** Every comparison above uses tiny random models; the release
  has only run on our side. A bf16 run of both pipelines on the same seed would show how far they drift.
- **Not ported:** diffusers' component offloading and tensor parallelism, and Pruna's ComfyUI files (int8, w4a8, and a
  20B transformer whose AdaLN projections are replaced by a table).
- **The modulation cache** (main guide §2.6) would apply unchanged: P-H3-Image's steps are single-time. It is not wired
  into `generate`.