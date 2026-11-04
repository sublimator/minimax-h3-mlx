# Porting MiniMax H3 to MLX: diffusers 0.42.1 → `h3mlx`

This is a side-by-side tour of how MiniMax H3's reference implementation, **diffusers 0.42.1** (`huggingface/diffusers`
tag `v0.42.1`), maps onto our MLX package `h3mlx`. Each section shows the original first, then the port, then what
changed and which test pins the two together. The aim is the **ref2va** workflow (reference images, videos and
audio → video with sound), which uses the same transformer class as text-to-video, with different weights
(`transformer_ref`) and a different packed layout.

The third-party MLX port (`yshenaw/ComfyUI-MiniMax-H3-MLX-SolAttn`) is **not** the
source of this code; everything here was written from diffusers.

**Conventions of the port**

- **Functional, diffusers names.** Ops take a flat `dict` of weights keyed exactly as the diffusers state dict, so the
  released `transformer_ref/` safetensors load with no renaming (`tests/test_weight_names.py`).
- **Channels-last for convolutions.** MLX convolutions want `(B, F, H, W, C)`; diffusers uses `(B, C, F, H, W)`.
  Conv weights are transposed once at load (`(O, I, k…)` → `(O, k…, I)`).
- **Precision as released.** The block stack is bf16; `proj_in`/`proj_out`, the audio heads, the time embedder and
  rope stay f32. Activations follow torch's rounding (one rounding per op, e.g. `silu` in f32 opmath).
- **Randomness from torch's CPU generator.** Noise is drawn with `torch.Generator` in diffusers' channels-first
  shapes and order, so a seed reproduces diffusers' draws; everything else is MLX.

---

## 1. The idea everything hangs on: one packed sequence

MiniMax H3 has no cross-attention. Text, reference blocks, target audio and target video are rows of **one** 1-D
sequence with full self-attention. Each row carries a modality tag, a timestep index and a `(t, h, w)` rotary
coordinate. For ref2va the order is `[text | reference blocks | target audio | target video]`, and the references
advance a shared audio/video clock, so their order is part of the request.

The original:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_denoise.py:566-729`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L566-L729)
```python
 566     @staticmethod
 567     def build_ref2va_packed_sequence(
 568         text_token_tags: torch.Tensor,
 569         references: list[MiniMaxH3Reference],
 570         condition_latents: list[torch.Tensor],
 571         audio_condition_latents: list[torch.Tensor],
 572         num_latent_frames: int,
 573         latent_height: int,
 574         latent_width: int,
 575         num_audio_latents: int,
 576         patch_size: tuple[int, int, int],
 577         audio_channels: int,
 578         audio_tag: int,
 579         video_tag: int,
 580     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
 581         r"""
 582         Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task.
 583
 584         Args:
 585             text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
 586                 The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision
 587                 block, which MiniMax-H3 tags `0` (video).
 588             references (`list[MiniMaxH3Reference]`):
 589                 The references, in packed order. Only their modality is read here; the geometry comes from the latents.
 590             condition_latents (`list[torch.Tensor]`):
 591                 One `(1, channels, num_latent_frames, latent_height, latent_width)` tensor per image and video
 592                 reference, in packed order, as [`MiniMaxH3Ref2VAReferenceEncoderStep`] produced them.
 593             audio_condition_latents (`list[torch.Tensor]`):
 594                 One `(num_audio_latents * 2, audio_latent_channels)` tensor per audio-bearing reference, in packed
 595                 order.
 596             num_latent_frames (`int`): Number of target latent frames.
 597             latent_height (`int`): Target latent height.
 598             latent_width (`int`): Target latent width.
 599             num_audio_latents (`int`): Number of target audio latents per channel.
 600             patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
 601             audio_channels (`int`): Channels the soundtrack is packed channel-major over, i.e.
 602                 `components.audio_channels`.
 603             audio_tag (`int`), video_tag (`int`):
 604                 The modality tags an audio and a video row carry, i.e. `components.audio_tag` and
 605                 `components.video_tag`.
 606
 607         Returns:
 608             `tuple`: `position_ids`, `token_tags`, `video_indices`, `audio_indices`, `text_indices`, and the number of
 609             leading video and audio rows that are references rather than generated.
 610         """
 611         _, patch_h, patch_w = patch_size
 612         num_text_tokens = text_token_tags.shape[0]
 613         num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
 614         num_target_audio_rows = num_audio_latents * audio_channels
 615
 616         # The geometry of every reference block is the shape of what the encoder produced for it, so the two can never
 617         # disagree. Both lists are in packed order but skip the references they do not apply to, so they are consumed as
 618         # iterators alongside the reference list rather than indexed by it.
 619         visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
 620         audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
 621         num_reference_video_rows = sum(
 622             frames * (height // patch_h) * (width // patch_w)
 623             for frames, height, width in (tuple(latents.shape[2:5]) for latents in condition_latents)
 624         )
 625         num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
 626         sequence_length = (
 627             num_text_tokens
 628             + num_reference_video_rows
 629             + num_reference_audio_rows
 630             + num_target_audio_rows
 631             + num_target_video_rows
 632         )
 633
 634         position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
 635         position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
 636         target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
 637
 638         # Reference blocks, in request order. `rotary_time` is the shared audio/video clock: it starts where the text
 639         # rows end and every block pushes it forward by the time that block occupies.
 640         video_indices, audio_indices = [], []
 641         cursor = num_text_tokens
 642         rotary_time = float(num_text_tokens)
 643         for reference in references:
 644             if reference.kind == "image":
 645                 num_latent_frames_, reference_height, reference_width = next(visual_geometry)
 646                 num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
 647                 rows = slice(cursor, cursor + num_video_rows)
 648                 cursor = rows.stop
 649                 video_indices.append(torch.arange(rows.start, rows.stop))
 650                 frame_grid, _ = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
 651                 position_ids[rows, 0] = rotary_time
 652                 position_ids[rows, 1:] = frame_grid
 653                 # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
 654                 rotary_time += 1.0
 655             elif reference.kind == "audio":
 656                 num_audio_rows = next(audio_row_counts)
 657                 reference_audio_latents = num_audio_rows // audio_channels
 658                 rows = slice(cursor, cursor + num_audio_rows)
 659                 cursor = rows.stop
 660                 audio_indices.append(torch.arange(rows.start, rows.stop))
 661                 _fill_audio_positions(
 662                     position_ids, rows, reference_audio_latents, rotary_time, target_width_grid, audio_channels
 663                 )
 664                 rotary_time += float(reference_audio_latents)
 665             elif reference.kind == "video":
 666                 # A video reference's soundtrack rows are packed immediately before its video rows and share their
 667                 # origin, so the two are rotary-aligned exactly as the generated audio and video are.
 668                 num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
 669                 reference_audio_latents = num_audio_rows // audio_channels
 670                 num_latent_frames_, reference_height, reference_width = next(visual_geometry)
 671                 num_video_rows = num_latent_frames_ * (reference_height // patch_h) * (reference_width // patch_w)
 672                 audio_rows = slice(cursor, cursor + num_audio_rows)
 673                 video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
 674                 cursor = video_rows.stop
 675                 audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
 676                 video_indices.append(torch.arange(video_rows.start, video_rows.stop))
 677
 678                 frame_grid, width_grid = _frame_position_grid(reference_height, reference_width, patch_h, patch_w)
 679                 _fill_audio_positions(
 680                     position_ids, audio_rows, reference_audio_latents, rotary_time, width_grid, audio_channels
 681                 )
 682                 frame_time = _temporal_position_grid(num_latent_frames_, rotary_time)
 683                 position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
 684                 position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
 685                 # The rotary time this reference advances the clock by, summed sequentially in float64. That is *not*
 686                 # how `_temporal_position_span_pairwise` sums the same series — it reproduces a numpy pairwise sum,
 687                 # and the two orders differ in the last ulp from 16 latent frames onwards. The reference
 688                 # implementation keeps both, one per call site, so the port has to as well.
 689                 video_span = sum(
 690                     _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
 691                     for index in range(num_latent_frames_)
 692                 )
 693                 rotary_time += max(float(reference_audio_latents), video_span)
 694             else:
 695                 raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}.")
 696
 697         # The generated rows. Target audio and target video share the origin the reference blocks left behind.
 698         audio_start = cursor
 699         video_start = audio_start + num_target_audio_rows
 700         _fill_audio_positions(
 701             position_ids,
 702             slice(audio_start, video_start),
 703             num_audio_latents,
 704             rotary_time,
 705             target_width_grid,
 706             audio_channels,
 707         )
 708         frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
 709         position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
 710         position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)
 711
 712         video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
 713         audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
 714         text_indices = torch.arange(num_text_tokens)
 715
 716         token_tags = torch.empty(sequence_length, dtype=torch.long)
 717         token_tags[text_indices] = text_token_tags.to(torch.long)
 718         token_tags[audio_indices] = audio_tag
 719         token_tags[video_indices] = video_tag
 720
 721         return (
 722             position_ids,
 723             token_tags,
 724             video_indices,
 725             audio_indices,
 726             text_indices,
 727             num_reference_video_rows,
 728             num_reference_audio_rows,
 729         )
```

The port is the same algorithm in numpy (float64 positions, int64 indices). It reads reference geometry from shapes
rather than tensors, so it needs no torch:

📍 `src/h3mlx/packing.py:112-192`
```python
 112 def build_ref2va_packed_sequence(
 113     text_token_tags,
 114     references: list[Ref],
 115     reference_video_shapes: list[tuple[int, int, int]],
 116     reference_audio_rows: list[int],
 117     num_latent_frames: int,
 118     latent_height: int,
 119     latent_width: int,
 120     num_audio_latents: int,
 121     patch_size=(1, 2, 2),
 122     audio_channels: int = 2,
 123     audio_tag: int = 2,
 124     video_tag: int = 0,
 125 ) -> Layout:
 126     """`reference_video_shapes`: (latent frames, latent h, latent w) per image/video reference in packed order;
 127     `reference_audio_rows`: row count per audio-bearing reference (audio refs and videos with audio), in packed order.
 128     The order of references advances the shared audio/video rotary clock, so it is part of the request."""
 129     _, ph, pw = patch_size
 130     n_text = len(text_token_tags)
 131     n_target_video = num_latent_frames * (latent_height // ph) * (latent_width // pw)
 132     n_target_audio = num_audio_latents * audio_channels
 133     vis = iter(reference_video_shapes)
 134     aud = iter(reference_audio_rows)
 135     n_ref_video = sum(f * (h // ph) * (w // pw) for f, h, w in reference_video_shapes)
 136     n_ref_audio = sum(reference_audio_rows)
 137     seq = n_text + n_ref_video + n_ref_audio + n_target_audio + n_target_video
 138
 139     pos = np.zeros((seq, 3), dtype=np.float64)
 140     pos[:n_text, 0] = np.arange(n_text, dtype=np.float64)
 141     target_frame_grid, target_width_grid = frame_position_grid(latent_height, latent_width, ph, pw)
 142
 143     video_idx, audio_idx = [], []
 144     cursor = n_text
 145     rotary_time = float(n_text)
 146     for ref in references:
 147         if ref.kind == "image":
 148             f, h, w = next(vis)
 149             n = f * (h // ph) * (w // pw)
 150             video_idx.append(np.arange(cursor, cursor + n))
 151             grid, _ = frame_position_grid(h, w, ph, pw)
 152             pos[cursor : cursor + n, 0] = rotary_time
 153             pos[cursor : cursor + n, 1:] = grid
 154             cursor += n
 155             rotary_time += 1.0  # an image takes one integer rotary slot
 156         elif ref.kind == "audio":
 157             n = next(aud)
 158             lat = n // audio_channels
 159             audio_idx.append(np.arange(cursor, cursor + n))
 160             _fill_audio_positions(pos, cursor, cursor + n, lat, rotary_time, target_width_grid, audio_channels)
 161             cursor += n
 162             rotary_time += float(lat)
 163         elif ref.kind == "video":
 164             na = next(aud) if ref.has_audio else 0
 165             lat = na // audio_channels
 166             f, h, w = next(vis)
 167             nv = f * (h // ph) * (w // pw)
 168             a0, v0 = cursor, cursor + na
 169             cursor = v0 + nv
 170             audio_idx.append(np.arange(a0, v0))
 171             video_idx.append(np.arange(v0, cursor))
 172             grid, wgrid = frame_position_grid(h, w, ph, pw)
 173             _fill_audio_positions(pos, a0, v0, lat, rotary_time, wgrid, audio_channels)
 174             pos[v0:cursor, 0] = np.repeat(temporal_position_grid(f, rotary_time), grid.shape[0])
 175             pos[v0:cursor, 1:] = np.tile(grid, (f, 1))
 176             # sequential float64 sum, NOT numpy's pairwise sum (differs in the last ulp from 16 latent frames on)
 177             span = sum(ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(f))
 178             rotary_time += max(float(lat), span)
 179         else:
 180             raise ValueError(f"reference kind must be 'image', 'video' or 'audio', got {ref.kind!r}")
 181
 182     audio_start = cursor
 183     video_start = audio_start + n_target_audio
 184     _fill_audio_positions(pos, audio_start, video_start, num_audio_latents, rotary_time, target_width_grid, audio_channels)
 185     pos[video_start:, 0] = np.repeat(temporal_position_grid(num_latent_frames, rotary_time), target_frame_grid.shape[0])
 186     pos[video_start:, 1:] = np.tile(target_frame_grid, (num_latent_frames, 1))
 187
 188     video_indices = np.concatenate(video_idx + [np.arange(video_start, seq)])
 189     audio_indices = np.concatenate(audio_idx + [np.arange(audio_start, video_start)])
 190     text_indices = np.arange(n_text)
 191     tags = _tags(seq, text_token_tags, video_indices, audio_indices, text_indices, audio_tag, video_tag)
 192     return Layout(pos, tags, video_indices, audio_indices, text_indices, n_ref_video, n_ref_audio)
```

Two details matter for bit-exactness. The spatial grid uses `np.linspace(..., endpoint=False)`, which the original
also calls (not `torch.linspace`). And a video reference advances the clock by a **sequential** float64 sum, while the
fl2va "last" anchor uses numpy's **pairwise** sum; the two differ in the last ulp at 15, 21, 26… latent frames, so the
port keeps both, like the original. `tests/test_packing.py` compares positions, tags, indices and counts bit for bit
on 4×4 t2va/fl2va cases and 7 ref2va mixes, including a 26-latent-frame video that only passes with the right order.

The rotary grids themselves:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_denoise.py:89-98`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L89-L98)
```python
  89 def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
  90     r"""The rotary time of every latent frame, starting at `origin`. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
  91     spans = torch.tensor(
  92         [
  93             _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
  94             for index in range(num_latent_frames)
  95         ],
  96         dtype=torch.float64,
  97     )
  98     return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
```
📍 `src/h3mlx/ops.py:197-202`
```python
 197 def temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
 198     spans = np.array(
 199         [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
 200         dtype=np.float64,
 201     )
 202     return origin + np.concatenate([np.zeros(1), np.cumsum(spans[:-1])])
```

---

## 2. The transformer

### 2.1 Rotary embedding

One `inv_freq` vector serves all three axes; the `(t, h, w)` angle blocks are concatenated and doubled, and only the
leading `2 · 3 · 16 = 96` of each head's 128 channels rotate.

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:93-100`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/transformers/transformer_minimax_h3.py#L93-L100)
```python
  93     def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  94         # position_ids: (seq_len, 3) -> cos/sin: (seq_len, 2 * 3 * rope_freq_dim)
  95         position_ids = position_ids.to(torch.float32)
  96         freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # (seq_len, 3, rope_freq_dim)
  97         freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
  98         freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
  99         freqs = torch.cat((freqs, freqs), dim=-1)
 100         return freqs.cos(), freqs.sin()
```
📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:59-73`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/transformers/transformer_minimax_h3.py#L59-L73)
```python
  59 def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  60     r"""
  61     Rotate the leading `rotary_dim` channels of every head and pass the remaining channels through unchanged.
  62     `hidden_states` is `(batch_size, seq_len, num_heads, head_dim)` and `cos`/`sin` are `(seq_len, rotary_dim)`.
  63     """
  64     rotary_dim = cos.shape[-1]
  65     hidden_states_rotary = hidden_states[..., :rotary_dim]
  66     hidden_states_pass = hidden_states[..., rotary_dim:]
  67
  68     cos = cos.to(hidden_states.dtype)[None, :, None, :]
  69     sin = sin.to(hidden_states.dtype)[None, :, None, :]
  70     x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
  71     hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
  72     hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
  73     return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()
```

📍 `src/h3mlx/ops.py:18-21`
```python
  18 def rope_inv_freq(rope_freq_dim: int = 16, rope_theta: float = 10000.0) -> mx.array:
  19     """Built in numpy float32: bit-identical to torch's `theta ** (arange / 2f)`; MLX's pow is 1 ulp off at f=16."""
  20     exps = np.arange(0, 2 * rope_freq_dim, 2, dtype=np.float32) / np.float32(2 * rope_freq_dim)
  21     return mx.array((np.float32(1) / np.power(np.float32(rope_theta), exps)).astype(np.float32))
```
📍 `src/h3mlx/ops.py:24-30`
```python
  24 def rope_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
  25     """(seq, 3) positions -> cos, sin of shape (seq, 2 * 3 * rope_freq_dim), float32."""
  26     pos = position_ids.astype(mx.float32)
  27     freqs = pos[:, :, None] * inv_freq[None, None, :]  # (seq, 3, f)
  28     freqs = freqs.reshape(pos.shape[0], -1)  # t block, h block, w block
  29     freqs = mx.concatenate([freqs, freqs], axis=-1)
  30     return mx.cos(freqs), mx.sin(freqs)
```
📍 `src/h3mlx/ops.py:33-41`
```python
  33 def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
  34     """x: (batch, seq, heads, head_dim); rotates the leading cos.shape[-1] channels (rotate_half), passes the rest."""
  35     rot = cos.shape[-1]
  36     xr, xp = x[..., :rot], x[..., rot:]
  37     c = cos.astype(x.dtype)[None, :, None, :]
  38     s = sin.astype(x.dtype)[None, :, None, :]
  39     x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
  40     rotated = mx.concatenate([-x2, x1], axis=-1)
  41     return mx.concatenate([xr * c + rotated * s, xp], axis=-1)
```

`inv_freq` is built in numpy float32: MLX's `pow` is one ulp off torch's at `rope_freq_dim = 16`, and numpy's float32
`power` matches torch bit for bit.

### 2.2 The AdaLN table

Each block projects the shared timestep embedding into six modulations **per (timestep, modality) pair**; a row picks
its pair with `timestep_index · 3 + modality_tag`.

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:124-131`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/transformers/transformer_minimax_h3.py#L124-L131)
```python
 124     def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
 125         # The activation runs at `temb`'s own precision — float32, since `time_embedder` is a float32 module in this
 126         # mixed-precision checkpoint — and only its result is cast down to the bfloat16 projection. Every block reads
 127         # the same `temb`, so a rounding applied before the activation biases every block's modulation parameters
 128         # identically at every sampling step, which accumulates coherently over the denoising trajectory.
 129         temb = self.linear(nn.functional.silu(temb).to(get_parameter_dtype(self.linear)))
 130         temb = temb.view(-1, 6 * self.hidden_size)
 131         return temb.chunk(6, dim=-1)
```
📍 `src/h3mlx/ops.py:67-75`
```python
  67 def adaln_modulation(temb: mx.array, w: mx.array, b: mx.array, hidden_size: int) -> list[mx.array]:
  68     """MiniMaxH3AdaLayerNormModulation: (num_t, temb_dim) -> 6 x (num_t * 3, hidden), rows [t0m0, t0m1, t0m2, t1m0..].
  69
  70     Order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp. silu at temb's precision, then cast to
  71     the projection's dtype.
  72     """
  73     out = linear(silu(temb).astype(w.dtype), w, b)
  74     out = out.reshape(-1, 6 * hidden_size)
  75     return mx.split(out, 6, axis=-1)
```
📍 `src/h3mlx/ops.py:78-80`
```python
  78 def modulate(x: mx.array, shift: mx.array, scale: mx.array, rows: mx.array) -> mx.array:
  79     """x * (1 + scale[rows]) + shift[rows]; x is (batch, seq, hidden), rows (seq,) index the AdaLN table."""
  80     return x * (1.0 + mx.take(scale, rows, axis=0)) + mx.take(shift, rows, axis=0)
```

The `silu` runs at `temb`'s precision (f32) and only its result is cast to the bf16 projection, as the original's
comment explains. Our `silu` rounds once, like torch's:

📍 `src/h3mlx/ops.py:54-58`
```python
  54 def silu(x: mx.array) -> mx.array:
  55     """torch's silu rounds once: compute in float32, cast back (a bf16 x * sigmoid(x) is off by one ulp at ~30% of
  56     inputs)."""
  57     xf = x.astype(mx.float32)
  58     return (xf * mx.sigmoid(xf)).astype(x.dtype)
```

### 2.3 Attention

Separate `to_q`/`to_k`/`to_v` with no bias, a per-head RMSNorm on q and k, rope, scaled-dot-product attention, and
`to_out`. Note that `heads · head_dim = 7168` is wider than the residual stream (5376).

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:168-209`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/transformers/transformer_minimax_h3.py#L168-L209)
```python
 168     def __call__(
 169         self,
 170         attn: "MiniMaxH3Attention",
 171         hidden_states: torch.Tensor,
 172         rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
 173         attention_mask: torch.Tensor | None = None,
 174     ) -> torch.Tensor:
 175         if attn.fused_projections:
 176             query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
 177         else:
 178             query = attn.to_q(hidden_states)
 179             key = attn.to_k(hidden_states)
 180             value = attn.to_v(hidden_states)
 181
 182         query = query.unflatten(-1, (attn.heads, -1))
 183         key = key.unflatten(-1, (attn.heads, -1))
 184         value = value.unflatten(-1, (attn.heads, -1))
 185
 186         query = attn.norm_q(query)
 187         key = attn.norm_k(key)
 188
 189         if rotary_emb is not None:
 190             query = _apply_rotary_emb(query, *rotary_emb)
 191             key = _apply_rotary_emb(key, *rotary_emb)
 192
 193         # MiniMax-H3 packs one request into a single attention document, so the model passes no mask and every
 194         # attention backend stays available; `attention_mask` is here because it is the processor signature every
 195         # other one in diffusers has, and a custom processor may need it.
 196         hidden_states = dispatch_attention_fn(
 197             query,
 198             key,
 199             value,
 200             attn_mask=attention_mask,
 201             dropout_p=0.0,
 202             is_causal=False,
 203             backend=self._attention_backend,
 204             parallel_config=self._parallel_config,
 205         )
 206         hidden_states = hidden_states.flatten(2, 3).type_as(query)
 207         hidden_states = attn.to_out[0](hidden_states)
 208         hidden_states = attn.to_out[1](hidden_states)
 209         return hidden_states
```
📍 `src/h3mlx/ops.py:113-132`
```python
 113 def attn_block(
 114     x: mx.array,
 115     p: dict,
 116     heads: int,
 117     qk_eps: float,
 118     rotary: tuple[mx.array, mx.array] | None,
 119     mask: mx.array | None = None,
 120 ) -> mx.array:
 121     """MiniMaxH3AttnProcessor with separate to_q/to_k/to_v (no bias), per-head RMSNorm on q/k, rope, to_out."""
 122     b, s, _ = x.shape
 123     q = linear(x, p["to_q.weight"]).reshape(b, s, heads, -1)
 124     k = linear(x, p["to_k.weight"]).reshape(b, s, heads, -1)
 125     v = linear(x, p["to_v.weight"]).reshape(b, s, heads, -1)
 126     q = rms_norm(q, p["norm_q.weight"], qk_eps)
 127     k = rms_norm(k, p["norm_k.weight"], qk_eps)
 128     if rotary is not None:
 129         q = apply_rotary(q, *rotary)
 130         k = apply_rotary(k, *rotary)
 131     o = attention(q, k, v, mask).reshape(b, s, -1)
 132     return linear(o, p["to_out.0.weight"])
```
📍 `src/h3mlx/ops.py:95-103`
```python
  95 def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
  96     """Full self-attention. q, k, v: (batch, seq, heads, head_dim) as diffusers lays them out; returns the same layout.
  97
  98     `mask` is an additive or boolean mask broadcastable to (batch, heads, seq_q, seq_k); the model itself passes none.
  99     """
 100     scale = 1.0 / math.sqrt(q.shape[-1])
 101     qt, kt, vt = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
 102     o = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale, mask=mask)
 103     return o.transpose(0, 2, 1, 3)
```

The original packs one request into one unpadded document, so no mask is passed. MiniMax's own reference pads to a
multiple of 64 and masks the tail; `tests/test_ops.py::test_attention_padded_to_64_with_mask_equals_unpadded` shows the
two agree.

### 2.4 One block

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:347-373`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/transformers/transformer_minimax_h3.py#L347-L373)
```python
 347     def forward(
 348         self,
 349         hidden_states: torch.Tensor,
 350         temb: torch.Tensor,
 351         adaln_indices: torch.Tensor,
 352         rotary_emb: tuple[torch.Tensor, torch.Tensor],
 353         attention_mask: torch.Tensor | None = None,
 354     ) -> torch.Tensor:
 355         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
 356
 357         residual = hidden_states
 358         norm_hidden_states = self.norm1(hidden_states)
 359         norm_hidden_states = norm_hidden_states * (
 360             1.0 + scale_msa.index_select(0, adaln_indices)
 361         ) + shift_msa.index_select(0, adaln_indices)
 362         attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
 363         hidden_states = residual + gate_msa.index_select(0, adaln_indices) * attn_output
 364
 365         residual = hidden_states
 366         norm_hidden_states = self.norm2(hidden_states)
 367         norm_hidden_states = norm_hidden_states * (
 368             1.0 + scale_mlp.index_select(0, adaln_indices)
 369         ) + shift_mlp.index_select(0, adaln_indices)
 370         ff_output = self.ff(norm_hidden_states)
 371         hidden_states = residual + gate_mlp.index_select(0, adaln_indices) * ff_output
 372
 373         return hidden_states
```
📍 `src/h3mlx/ops.py:260-281`
```python
 260 def dit_block(
 261     x: mx.array,
 262     temb: mx.array,
 263     adaln_indices: mx.array,
 264     rotary: tuple[mx.array, mx.array],
 265     p: dict,
 266     heads: int,
 267     hidden_size: int,
 268     norm_eps: float = 1e-5,
 269     qk_eps: float = 1e-5,
 270     modulation: list[mx.array] | None = None,
 271 ) -> mx.array:
 272     """Pre-norm attention and SwiGLU FF, each shift/scale-modulated and gated per row from the (t, modality) table.
 273     `modulation`: the six precomputed tables (from `ModulationCache`) instead of projecting `temb` here."""
 274     if modulation is None:
 275         modulation = adaln_modulation(temb, p["adaln_proj.linear.weight"], p["adaln_proj.linear.bias"], hidden_size)
 276     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
 277     sub = lambda prefix: {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}  # noqa: E731
 278     h = modulate(rms_norm(x, p["norm1.weight"], norm_eps), shift_msa, scale_msa, adaln_indices)
 279     x = x + mx.take(gate_msa, adaln_indices, axis=0) * attn_block(h, sub("attn."), heads, qk_eps, rotary)
 280     h = modulate(rms_norm(x, p["norm2.weight"], norm_eps), shift_mlp, scale_mlp, adaln_indices)
 281     return x + mx.take(gate_mlp, adaln_indices, axis=0) * swiglu_ff(h, sub("ff."))
```

`modulation=` lets a precomputed table replace the per-step projection (§2.6).

### 2.5 The whole forward

📍 [`diffusers:src/diffusers/models/transformers/transformer_minimax_h3.py:563-666`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/transformers/transformer_minimax_h3.py#L563-L666)
```python
 563     @apply_lora_scale("attention_kwargs")
 564     def forward(
 565         self,
 566         hidden_states: torch.Tensor,
 567         audio_hidden_states: torch.Tensor,
 568         encoder_hidden_states: torch.Tensor,
 569         timestep: torch.Tensor,
 570         timestep_indices: torch.Tensor,
 571         token_tags: torch.Tensor,
 572         position_ids: torch.Tensor,
 573         video_indices: torch.Tensor,
 574         audio_indices: torch.Tensor,
 575         text_indices: torch.Tensor,
 576         attention_kwargs: dict[str, Any] | None = None,
 577         return_dict: bool = True,
 578     ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
 579         r"""
 580         Args:
 581             hidden_states (`torch.Tensor` of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))`):
 582                 Patchified video latent rows — conditioning rows and target rows — ordered as they appear in the packed
 583                 sequence, i.e. matching `video_indices`.
 584             audio_hidden_states (`torch.Tensor` of shape `(batch_size, num_audio_tokens, audio_in_channels)`):
 585                 Audio latent rows, ordered to match `audio_indices`.
 586             encoder_hidden_states (`torch.Tensor` of shape `(batch_size, num_text_tokens, text_dim)`):
 587                 Text conditioning, ordered to match `text_indices`.
 588             timestep (`torch.Tensor` of shape `(num_timesteps,)`):
 589                 The *distinct* timestep values present in the packed sequence, in `[0, 1]` and unscaled. One forward
 590                 serves rows at different noise levels (target video, target audio, conditioning rows).
 591             timestep_indices (`torch.Tensor` of shape `(seq_len,)`):
 592                 For every row of the packed sequence, the index of its timestep in `timestep`.
 593             token_tags (`torch.Tensor` of shape `(seq_len,)`):
 594                 For every row of the packed sequence, its modality: `0` video, `1` text, `2` audio.
 595             position_ids (`torch.Tensor` of shape `(seq_len, 3)`):
 596                 The `(t, h, w)` rotary coordinates of every row of the packed sequence.
 597             video_indices (`torch.Tensor` of shape `(num_video_tokens,)`):
 598                 Positions of the video rows in the packed sequence.
 599             audio_indices (`torch.Tensor` of shape `(num_audio_tokens,)`):
 600                 Positions of the audio rows in the packed sequence.
 601             text_indices (`torch.Tensor` of shape `(num_text_tokens,)`):
 602                 Positions of the text rows in the packed sequence.
 603             attention_kwargs (`dict`, *optional*):
 604                 A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
 605             return_dict (`bool`, defaults to `True`):
 606                 Whether to return a [`MiniMaxH3TransformerOutput`] instead of a plain tuple.
 607
 608         Returns:
 609             [`MiniMaxH3TransformerOutput`] or `tuple`:
 610                 The video velocity of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))` and the
 611                 audio velocity of shape `(batch_size, num_audio_tokens, audio_in_channels)`, in the row order of
 612                 `video_indices` and `audio_indices`.
 613         """
 614         # `attention_kwargs` is consumed by the `@apply_lora_scale` decorator on this method.
 615         if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
 616             raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
 617         sequence_length = position_ids.shape[0]
 618         if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
 619             raise ValueError(
 620                 "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
 621                 f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
 622             )
 623
 624         rotary_emb = self.rope(position_ids)
 625
 626         # 1. Project each modality and scatter the rows into the packed sequence buffer. The checkpoint is
 627         # mixed-precision (the two patch projections are float32 while `context_embedder` and the block stack are
 628         # bfloat16 — see `_keep_in_fp32_modules`), so every input is aligned with its projection's parameter dtype,
 629         # mirroring the reference's explicit casts. The text stream sets the dtype of the packed sequence.
 630         video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
 631         audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
 632         text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
 633         text_embeds = self.token_refiner(text_embeds)
 634
 635         hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
 636         hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
 637         hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
 638         hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
 639
 640         # 2. One timestep embedding per distinct noise level. `temb` is shared by all AdaLN projections, which are
 641         # bfloat16 in the checkpoint while `time_embedder` is float32, so it stays at the time embedder's precision:
 642         # each AdaLN module applies its own activation to it and casts to its projection's dtype afterwards.
 643         temb = self.time_proj(timestep)
 644         temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
 645
 646         # 3. Row -> AdaLN table row.
 647         adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
 648
 649         for block in self.transformer_blocks:
 650             if torch.is_grad_enabled() and self.gradient_checkpointing:
 651                 hidden_states = self._gradient_checkpointing_func(
 652                     block, hidden_states, temb, adaln_indices, rotary_emb
 653                 )
 654             else:
 655                 hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
 656
 657         # 5. Both heads run over every row, then the rows of each modality are selected. The heads are listed in
 658         # `_keep_in_fp32_modules`, so they stay float32 while the block stack runs in the requested `torch_dtype`;
 659         # align the activation with their parameter dtype.
 660         hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
 661         video_output = self.proj_out(hidden_states).index_select(1, video_indices)
 662         audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
 663
 664         if not return_dict:
 665             return (video_output, audio_output)
 666         return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
```

The port, stage by stage. Each modality is projected at its own parameter dtype, the text goes through the 2-layer
token refiner, and the rows are scattered into the packed buffer:

📍 `src/h3mlx/ops.py:324-332`
```python
 324     v = linear(video.astype(p["proj_in.weight"].dtype), p["proj_in.weight"], p["proj_in.bias"])
 325     a = linear(audio.astype(p["audio_proj_in.weight"].dtype), p["audio_proj_in.weight"], p["audio_proj_in.bias"])
 326     t = linear(text.astype(p["context_embedder.weight"].dtype), p["context_embedder.weight"], p["context_embedder.bias"])
 327     t = token_refiner(t, _sub(p, "token_refiner."), cfg["num_refiner_layers"], heads, eps, qk_eps, final_eps)
 328
 329     x = mx.zeros((t.shape[0], seq, t.shape[-1]), dtype=t.dtype)
 330     x[:, text_indices] = t
 331     x[:, video_indices] = v.astype(t.dtype)
 332     x[:, audio_indices] = a.astype(t.dtype)
```

The AdaLN rows, then the 50 blocks, each evaluated before the next to bound memory at 20–50k rows:

📍 `src/h3mlx/ops.py:336-346`
```python
 336     if cache is None:
 337         temb = timestep_embedding(timestep, _sub(p, "time_embedder."), cfg["freq_dim"])
 338         t_rows = timestep_indices
 339     else:
 340         temb = None
 341         t_rows = cache.rows(timestep, timestep_indices)  # step-local timestep index -> row of the cached table
 342     adaln_indices = t_rows * MODALITY_NUM + token_tags
 343     for i in range(cfg["num_layers"]):
 344         x = dit_block(x, temb, adaln_indices, rotary, _sub(p, f"transformer_blocks.{i}."), heads, hidden, eps, qk_eps,
 345                       modulation=None if cache is None else cache.blocks[i])
 346         mx.eval(x)  # one block's graph at a time: bounds activation memory at 20-50k rows
```

The final shift/scale norm, the cast to the f32 heads, and gathering the video and audio rows:

📍 `src/h3mlx/ops.py:350-359`
```python
 350     if cache is None:
 351         x = norm_out(x, temb, p["norm_out.norm.weight"], p["norm_out.linear.weight"], p["norm_out.linear.bias"],
 352                      timestep_indices, final_eps)
 353     else:
 354         shift, scale = cache.out
 355         h = rms_norm(x, p["norm_out.norm.weight"], final_eps)
 356         x = h * (1.0 + mx.take(scale, t_rows, axis=0)) + mx.take(shift, t_rows, axis=0)
 357     x = x.astype(p["proj_out.weight"].dtype)
 358     vid = linear(x, p["proj_out.weight"], p["proj_out.bias"])[:, video_indices]
 359     aud = linear(x, p["audio_proj_out.weight"], p["audio_proj_out.bias"])[:, audio_indices]
```

`tests/test_dit_tiny.py` runs both forwards on a tiny random config with shuffled modalities and three noise levels.
`tests/test_dit_mixed_precision.py` scores both bf16 runs against one f32 run: MLX's relative error is 0.59% where
torch's own is 0.55%.

### 2.6 Ours only: the modulation cache

The modulations depend only on the timestep, never on the latents. One run can therefore compute every block's
tables once for every timestep it will use, and drop the `adaln_proj` weights: about 26 GB of `transformer_ref`'s
66 GB in bf16. The idea comes from the third-party port; this implementation is ours and is bit-identical to the
uncached path (`tests/test_modulation_cache.py`).

📍 `src/h3mlx/ops.py:367-405`
```python
 367 class ModulationCache:
 368     """Every block's six AdaLN tables (and norm_out's shift/scale) for a fixed set of timestep values, computed once.
 369
 370     The modulation depends only on t and the weights, never on the latents, so a sampling run can precompute it for
 371     the union of its timesteps and drop the `adaln_proj` weights (~26 GB of transformer_ref's ~66 GB in bf16). The
 372     rows of each table are [t0m0, t0m1, t0m2, t1m0, ...] over `self.t`, exactly what `adaln_modulation` returns.
 373     """
 374
 375     def __init__(self, p: dict, cfg: dict, timesteps):
 376         t = np.unique(np.asarray(timesteps, dtype=np.float32))
 377         self.t = t
 378         self._index = {float(v): i for i, v in enumerate(t)}
 379         temb = timestep_embedding(mx.array(t), _sub(p, "time_embedder."), cfg["freq_dim"])
 380         hidden = cfg["hidden_size"]
 381         self.blocks = []
 382         for i in range(cfg["num_layers"]):
 383             b = f"transformer_blocks.{i}.adaln_proj.linear."
 384             tables = adaln_modulation(temb, p[b + "weight"], p[b + "bias"], hidden)
 385             mx.eval(tables)
 386             self.blocks.append(tables)
 387         w, bias = p["norm_out.linear.weight"], p["norm_out.linear.bias"]
 388         self.out = mx.split(linear(silu(temb).astype(w.dtype), w, bias), 2, axis=-1)
 389         mx.eval(self.out)
 390
 391     def rows(self, timestep: mx.array, timestep_indices: mx.array) -> mx.array:
 392         """Map a step's (distinct timesteps, per-row index into them) onto rows of the cached tables."""
 393         local = np.array(timestep.astype(mx.float32))
 394         try:
 395             glob = np.array([self._index[float(v)] for v in local], dtype=np.int64)
 396         except KeyError as e:
 397             raise KeyError(f"timestep {e} not in the cache (built for {len(self.t)} values)") from None
 398         return mx.take(mx.array(glob), timestep_indices)
 399
 400     @staticmethod
 401     def drop_weights(p: dict, cfg: dict) -> dict:
 402         """The params without the projections the cache replaces."""
 403         gone = {f"transformer_blocks.{i}.adaln_proj.linear.{n}" for i in range(cfg["num_layers"]) for n in ("weight", "bias")}
 404         gone |= {"norm_out.linear.weight", "norm_out.linear.bias"}
 405         return {k: v for k, v in p.items() if k not in gone}
```

---

## 3. The scheduler

Rectified flow with an exponential shift: 12 for video, 3 for audio. The model predicts a **data-ward** velocity
(`x0 = x + (1 − t)·v`, note the `+`), and the Euler update is a blend computed in f32.

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
📍 `src/h3mlx/ops.py:233-243`
```python
 233 def flow_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
 234     """linspace(1, 0, n) in float32, exponential shift, consecutive duplicates collapsed. Terminal 0 included."""
 235     if num_inference_steps is None or num_inference_steps < 2:
 236         raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
 237     if shift <= 0:
 238         raise ValueError(f"shift must be positive, got {shift}")
 239     base = torch_linspace_f32(1.0, 0.0, num_inference_steps)
 240     shift32 = np.float32(shift)
 241     sig = (shift32 * base / (np.float32(1) + (shift32 - np.float32(1)) * base)).astype(np.float32)
 242     keep = np.concatenate([[True], sig[1:] != sig[:-1]])
 243     return sig[keep]
```

`torch.linspace` on arm64 fills each half with a fused multiply-add from its own end, so `np.linspace` differs by
one ulp (for example at 50 steps). The port reproduces torch's fill, and it is bit-exact for every `n` from 2 to 1999:

📍 `src/h3mlx/ops.py:216-230`
```python
 216 def torch_linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
 217     """torch.linspace(start, end, steps, dtype=float32) bit for bit (CPU, arm64): float32 step, the first half filled
 218     from `start` and the second from `end`, each as one fused multiply-add (exact product, single rounding).
 219     np.linspace differs by 1 ulp at e.g. steps=50."""
 220     if steps < 2:
 221         raise ValueError(f"steps must be >= 2, got {steps}")
 222     start32, end32 = np.float32(start), np.float32(end)
 223     step = np.float32((end32 - start32) / np.float32(steps - 1))
 224     i = np.arange(steps, dtype=np.float64)
 225     out = np.where(
 226         i < steps // 2,
 227         np.float64(start32) + np.float64(step) * i,
 228         np.float64(end32) - np.float64(step) * (steps - 1 - i),
 229     )
 230     return out.astype(np.float32)
```

📍 [`diffusers:src/diffusers/schedulers/scheduling_minimax_h3.py:223-283`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/schedulers/scheduling_minimax_h3.py#L223-L283)
```python
 223     def step(
 224         self,
 225         model_output: torch.FloatTensor,
 226         timestep: float | torch.FloatTensor,
 227         sample: torch.FloatTensor,
 228         return_dict: bool = True,
 229     ) -> MiniMaxH3SchedulerOutput | tuple:
 230         r"""
 231         Take one Euler (`eta = 0`) step.
 232
 233         The model output is a data-ward velocity, so the denoised estimate is `x0 = x_t + (1 - t) * v` — note the `+`,
 234         the opposite of the usual flow-match convention. The update is then the blend `x_next = r*x_t + (1 - r)*x0`
 235         with `r = sigma_next / sigma`, evaluated in float32 for half-precision samples.
 236
 237         Args:
 238             model_output (`torch.FloatTensor`):
 239                 The transformer's velocity prediction at `timestep`.
 240             timestep (`float` or `torch.FloatTensor`):
 241                 The current timestep, one of `self.timesteps` (so `timestep == 1 - sigma`).
 242             sample (`torch.FloatTensor`):
 243                 The current sample `x_t`.
 244             return_dict (`bool`, defaults to `True`):
 245                 Whether to return a [`MiniMaxH3SchedulerOutput`] instead of a plain tuple.
 246
 247         Returns:
 248             [`MiniMaxH3SchedulerOutput`] or `tuple`: the sample for the next step.
 249         """
 250         if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
 251             raise ValueError(
 252                 "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
 253                 " `MiniMaxH3Scheduler.step()` is not supported. Make sure to pass one of the"
 254                 " `scheduler.timesteps` values."
 255             )
 256
 257         if self._step_index is None:
 258             self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index
 259
 260         # x0 from the data-ward velocity. The sigma used here is recovered from the *timestep* the
 261         # transformer was conditioned on, whereas the Euler ratio below uses the sigma grid: for
 262         # sigma < 0.5 the float32 round trip `1 - (1 - sigma)` is not exact, and the reference keeps
 263         # the two sources apart.
 264         if not isinstance(timestep, torch.Tensor):
 265             timestep = torch.tensor(timestep, dtype=sample.dtype)
 266         sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
 267         while sigma_from_timestep.ndim < sample.ndim:
 268             sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
 269         denoised = sample + sigma_from_timestep * model_output
 270
 271         # Euler with eta = 0, written as an x_t / x0 blend and evaluated in float32.
 272         compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
 273         sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
 274         sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
 275         ratio = sigma_next / sigma
 276         prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
 277         prev_sample = prev_sample.to(dtype=sample.dtype)
 278
 279         self._step_index += 1
 280
 281         if not return_dict:
 282             return (prev_sample,)
 283         return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)
```
📍 `src/h3mlx/ops.py:246-254`
```python
 246 def flow_step(v: mx.array, t: float, x: mx.array, sigma: float, sigma_next: float) -> mx.array:
 247     """x0 = x + (1 - t) * v (sigma from the timestep, in x's dtype); x_next = r*x + (1 - r)*x0, r = s_next / s, f32."""
 248     st = (mx.array(1.0, dtype=x.dtype) - mx.array(t, dtype=x.dtype)).astype(x.dtype)
 249     x0 = x + st * v
 250     cdt = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
 251     s = mx.array(sigma, dtype=cdt)
 252     sn = mx.array(sigma_next, dtype=cdt)
 253     r = sn / s
 254     return (r * x.astype(cdt) + (1.0 - r) * x0.astype(cdt)).astype(x.dtype)
```

---

## 4. The denoise loop

One forward per step serves every row. Generated video and audio rows step down their own schedules; visual
conditioning rows stay at `max(t, 0.999)`, audio references at 1.0, and text inherits the video timestep. The
checkpoint is guidance-distilled, so there is no unconditional pass.

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_denoise.py:1197-1236`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L1197-L1236)
```python
1197     @staticmethod
1198     def build_row_timesteps(
1199         video_indices: torch.Tensor,
1200         audio_indices: torch.Tensor,
1201         num_condition_video_rows: int,
1202         num_condition_audio_rows: int,
1203         num_text_tokens: int,
1204         video_timestep: float,
1205         audio_timestep: float,
1206         condition_video_timestep: float,
1207         condition_audio_timestep: float,
1208     ) -> tuple[torch.Tensor, torch.Tensor]:
1209         r"""
1210         Assign a timestep to every row of the packed sequence and reduce it to the transformer's `(timestep,
1211         timestep_indices)` pair.
1212
1213         One forward serves rows at different noise levels: the generated video and audio rows step down their own
1214         schedules while the conditioning rows stay pinned at their noise-augmentation level. Text rows never reach an
1215         output head and inherit the video timestep.
1216
1217         Args:
1218             video_indices (`torch.Tensor`): Sequence positions of the video rows, conditioning rows first.
1219             audio_indices (`torch.Tensor`): Sequence positions of the audio rows, reference rows first.
1220             num_condition_video_rows (`int`): How many leading video rows are conditioning rows.
1221             num_condition_audio_rows (`int`): How many leading audio rows are reference rows.
1222             num_text_tokens (`int`): Number of text rows, which never reach an output head.
1223             video_timestep (`float`): Timestep of the generated video rows.
1224             audio_timestep (`float`): Timestep of the generated audio rows.
1225             condition_video_timestep (`float`): Timestep of the video conditioning rows.
1226             condition_audio_timestep (`float`): Timestep of the audio reference rows.
1227
1228         Returns:
1229             `tuple[torch.Tensor, torch.Tensor]`: the distinct timesteps, sorted, and the index of every row into them.
1230         """
1231         sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
1232         row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
1233         row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
1234         row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
1235         row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
1236         return torch.unique(row_timesteps, sorted=True, return_inverse=True)
```
📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/denoise.py:222-241`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/denoise.py#L222-L241)
```python
 222     @torch.no_grad()
 223     def __call__(
 224         self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
 225     ) -> tuple[MiniMaxH3ModularPipeline, BlockState]:
 226         num_condition_video_rows = block_state.num_condition_video_rows
 227         num_condition_audio_rows = block_state.num_condition_audio_rows
 228
 229         block_state.latents[num_condition_video_rows:] = components.scheduler.step(
 230             block_state.noise_pred[0, num_condition_video_rows:].float(),
 231             t,
 232             block_state.latents[num_condition_video_rows:],
 233             return_dict=False,
 234         )[0]
 235         block_state.audio_latents[num_condition_audio_rows:] = components.audio_scheduler.step(
 236             block_state.audio_noise_pred[0, num_condition_audio_rows:].float(),
 237             block_state.audio_timesteps[i],
 238             block_state.audio_latents[num_condition_audio_rows:],
 239             return_dict=False,
 240         )[0]
 241         return components, block_state
```

📍 `src/h3mlx/denoise.py:38-46`
```python
  38 def row_timesteps(layout: Layout, video_t: float, audio_t: float, cond_video_t: float, cond_audio_t: float):
  39     """Per-row timesteps reduced to (sorted distinct values, inverse index), like `build_row_timesteps`."""
  40     seq = len(layout.video_indices) + len(layout.audio_indices) + len(layout.text_indices)
  41     rt = np.full(seq, np.float32(video_t), dtype=np.float32)
  42     rt[layout.video_indices[: layout.num_condition_video_rows]] = cond_video_t
  43     rt[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_t
  44     rt[layout.audio_indices[: layout.num_condition_audio_rows]] = cond_audio_t
  45     uniq, inv = np.unique(rt, return_inverse=True)
  46     return uniq, inv
```
📍 `src/h3mlx/denoise.py:49-87`
```python
  49 def denoise(
  50     params: dict,
  51     cfg: dict,
  52     layout: Layout,
  53     prompt_embeds: mx.array,
  54     video_rows: mx.array,
  55     audio_rows: mx.array,
  56     schedules: Schedules,
  57     on_step=None,
  58     cache: "ops.ModulationCache | None" = None,
  59 ):
  60     """video_rows (n_video, patch_dim) with conditioning rows first; audio_rows (n_audio, audio_ch) with reference rows
  61     first; prompt_embeds (1, n_text, text_dim). Returns the final (video_rows, audio_rows). With `cache`
  62     (`ops.ModulationCache` built from `schedule_timesteps`), the AdaLN projections are not evaluated per step."""
  63     lay = {
  64         "token_tags": mx.array(layout.token_tags),
  65         "position_ids": mx.array(layout.position_ids.astype(np.float32)),
  66         "video_indices": mx.array(layout.video_indices),
  67         "audio_indices": mx.array(layout.audio_indices),
  68         "text_indices": mx.array(layout.text_indices),
  69     }
  70     ncv, nca = layout.num_condition_video_rows, layout.num_condition_audio_rows
  71     vts, ats = schedules.video_timesteps, schedules.audio_timesteps
  72     for i, (t, ta) in enumerate(zip(vts, ats)):
  73         uniq, inv = row_timesteps(layout, float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0)
  74         v_pred, a_pred = ops.dit_forward(
  75             params, cfg, video_rows[None], audio_rows[None], prompt_embeds, mx.array(uniq), mx.array(inv), **lay,
  76             cache=cache,
  77         )
  78         new_v = ops.flow_step(v_pred[0, ncv:].astype(mx.float32), float(t), video_rows[ncv:],
  79                               float(schedules.video_sigmas[i]), float(schedules.video_sigmas[i + 1]))
  80         new_a = ops.flow_step(a_pred[0, nca:].astype(mx.float32), float(ta), audio_rows[nca:],
  81                               float(schedules.audio_sigmas[i]), float(schedules.audio_sigmas[i + 1]))
  82         video_rows = mx.concatenate([video_rows[:ncv], new_v]) if ncv else new_v
  83         audio_rows = mx.concatenate([audio_rows[:nca], new_a]) if nca else new_a
  84         mx.eval(video_rows, audio_rows)
  85         if on_step:
  86             on_step(i, t)
  87     return video_rows, audio_rows
```

`tests/test_denoise.py` runs the port against a torch loop assembled from diffusers' own scheduler, row plan and
transformer, for 3 and 5 steps; the conditioning rows come back untouched.

---

## 5. The text side: the Qwen3-VL presentation

MiniMax conditions on hidden state **50** of a 64-layer Qwen3-VL. That state is pre-norm, and the last layer's would
be the wrong one. Every reference gets a label in the prompt, numbered per modality: `<Picture i>: ` plus an image
block, `<Audio j>: ` alone, `<Video k>: ` plus one timestamped block per merged frame pair. Then comes the prompt
verbatim, with no chat template.

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/encoders.py:543-612`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L543-L612)
```python
 543     @staticmethod
 544     def _build_presentation(
 545         tokenizer,
 546         prompt: str,
 547         references: list[MiniMaxH3Reference],
 548         image_token_counts: list[int],
 549         video_block_token_counts: list[int],
 550         video_block_timestamps: list[list[float]],
 551         text_tag: int = 1,
 552         video_tag: int = 0,
 553     ) -> tuple[list[int], list[int]]:
 554         r"""
 555         Tokenize MiniMax-H3's presentation of a `ref2va` request.
 556
 557         Every reference prepends a label, in packed order and numbered per modality: `"<Picture i>: "` plus a vision
 558         block for an image, `"<Audio j>: "` alone for audio — a waveform never reaches the conditioner — and `"<Video
 559         k>: "` plus one timestamped vision block per merged frame pair for a video. A video that carries sound is
 560         labelled `"<Audio j>: "` *before* `"<Video k>: "`, mirroring the order its rows are packed in. The prompt
 561         follows verbatim, with no chat template and no special tokens.
 562
 563         Args:
 564             tokenizer (`Qwen2TokenizerFast`): Tokenizer of the conditioner.
 565             prompt (`str`): The prompt, appended verbatim.
 566             references (`list[MiniMaxH3Reference]`): The normalized references, in packed order.
 567             image_token_counts (`list[int]`): Number of vision tokens of every image reference's block.
 568             video_block_token_counts (`list[int]`): Number of vision tokens per block of every video reference.
 569             video_block_timestamps (`list[list[float]]`): The timestamp of every vision block, per video reference.
 570             text_tag (`int`, *optional*, defaults to 1): MiniMax-H3's modality tag for a text row.
 571             video_tag (`int`, *optional*, defaults to 0): MiniMax-H3's modality tag for a vision block's rows.
 572
 573         Returns:
 574             `tuple[list[int], list[int]]`: the token ids and their modality tags.
 575         """
 576
 577         def text(value: str) -> tuple[list[int], list[int]]:
 578             token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
 579             return token_ids, [text_tag] * len(token_ids)
 580
 581         def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
 582             token_ids = (
 583                 [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
 584                 + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
 585                 + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
 586             )
 587             return token_ids, [video_tag] * len(token_ids)
 588
 589         token_ids, token_tags = [], []
 590
 591         def emit(segment: tuple[list[int], list[int]]) -> None:
 592             token_ids.extend(segment[0])
 593             token_tags.extend(segment[1])
 594
 595         counts = {"image": 0, "video": 0, "audio": 0}
 596         for reference in references:
 597             if reference.has_audio:
 598                 counts["audio"] += 1
 599                 emit(text(f"<Audio {counts['audio']}>: "))
 600             if reference.kind == "image":
 601                 counts["image"] += 1
 602                 emit(text(f"<Picture {counts['image']}>: "))
 603                 emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
 604             elif reference.kind == "video":
 605                 counts["video"] += 1
 606                 emit(text(f"<Video {counts['video']}>: "))
 607                 for timestamp in video_block_timestamps[counts["video"] - 1]:
 608                     # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
 609                     emit(text(f"<{timestamp:.1f} seconds>"))
 610                     emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
 611         emit(text(prompt))
 612         return token_ids, token_tags
```
📍 `src/h3mlx/presentation.py:30-74`
```python
  30 def build_presentation(
  31     tokenizer,
  32     prompt: str,
  33     references: list[Ref],
  34     image_token_counts: list[int],
  35     video_block_token_counts: list[int],
  36     video_block_timestamps: list[list[float]],
  37     text_tag: int = 1,
  38     video_tag: int = 0,
  39 ) -> tuple[list[int], list[int]]:
  40     def text(s):
  41         ids = tokenizer(s, add_special_tokens=False)["input_ids"]
  42         return ids, [text_tag] * len(ids)
  43
  44     def vision(pad, n):
  45         ids = (
  46             [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
  47             + [tokenizer.convert_tokens_to_ids(pad)] * n
  48             + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
  49         )
  50         return ids, [video_tag] * len(ids)
  51
  52     ids, tags = [], []
  53
  54     def emit(seg):
  55         ids.extend(seg[0])
  56         tags.extend(seg[1])
  57
  58     n = {"image": 0, "video": 0, "audio": 0}
  59     for r in references:
  60         if r.has_audio:
  61             n["audio"] += 1
  62             emit(text(f"<Audio {n['audio']}>: "))
  63         if r.kind == "image":
  64             n["image"] += 1
  65             emit(text(f"<Picture {n['image']}>: "))
  66             emit(vision("<|image_pad|>", image_token_counts[n["image"] - 1]))
  67         elif r.kind == "video":
  68             n["video"] += 1
  69             emit(text(f"<Video {n['video']}>: "))
  70             for t in video_block_timestamps[n["video"] - 1]:
  71                 emit(text(f"<{t:.1f} seconds>"))  # round half to even: a 2 fps pair renders "<0.2 seconds>"
  72                 emit(vision("<|video_pad|>", video_block_token_counts[n["video"] - 1]))
  73     emit(text(prompt))
  74     return ids, tags
```

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/encoders.py:447-489`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L447-L489)
```python
 447     @staticmethod
 448     def _sample_video_condition_frames(
 449         frames: np.ndarray, fps: float, sample_fps: float, temporal_patch: int
 450     ) -> tuple[list[np.ndarray], list[float]]:
 451         r"""
 452         Sample the frames the conditioner sees from a normalized reference video, and label their vision blocks.
 453
 454         The conditioner reads a reference at `sample_fps`: every `fps / sample_fps`-th frame, deduplicated. Qwen3-VL
 455         then merges the sampled frames in groups of `temporal_patch` — repeating the last one when the count does not
 456         divide — and a merged group is labelled with the mean of its timestamps, which `"<{timestamp:.1f} seconds>"`
 457         renders with Python's round-half-to-even, so the first block of a 2 fps pair is `"<0.2 seconds>"` rather than
 458         `"<0.3 seconds>"`.
 459
 460         Args:
 461             frames (`np.ndarray` of shape `(num_frames, height, width, 3)`): The normalized reference video.
 462             fps (`float`): The rate `frames` is at, i.e. `components.fps`.
 463             sample_fps (`float`): The rate the conditioner reads at, i.e. the block's `video_sample_fps`.
 464             temporal_patch (`int`): Qwen3-VL's temporal patch, read off the processor.
 465
 466         Returns:
 467             `tuple[list[np.ndarray], list[float]]`: the sampled frames and one timestamp per vision block.
 468         """
 469         stride = fps / sample_fps
 470         indices, cursor = [], 0.0
 471         while round(cursor) < frames.shape[0]:
 472             if not indices or round(cursor) > indices[-1]:
 473                 indices.append(round(cursor))
 474             cursor += stride
 475         if len(indices) < temporal_patch:
 476             minimum = round((temporal_patch - 1) * stride) + 1
 477             raise ValueError(
 478                 f"A reference video is read at {sample_fps:g} fps and its sampled frames are merged in groups of "
 479                 f"{temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
 480                 f"({minimum / fps:.2g} seconds), got {frames.shape[0]}."
 481             )
 482
 483         timestamps = [index / sample_fps for index in range(len(indices))]
 484         timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
 485         block_timestamps = [
 486             (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
 487             for index in range(0, len(timestamps), temporal_patch)
 488         ]
 489         return [frames[index] for index in indices], block_timestamps
```
📍 `src/h3mlx/presentation.py:12-27`
```python
  12 def sample_video_condition_frames(num_frames: int, fps: float, sample_fps: float, temporal_patch: int):
  13     """Frame indices the conditioner reads (every fps/sample_fps-th, deduplicated) and one timestamp per merged block
  14     (mean of each group of `temporal_patch`, last timestamp repeated to fill the group)."""
  15     stride = fps / sample_fps
  16     indices, cursor = [], 0.0
  17     while round(cursor) < num_frames:
  18         if not indices or round(cursor) > indices[-1]:
  19             indices.append(round(cursor))
  20         cursor += stride
  21     if len(indices) < temporal_patch:
  22         minimum = round((temporal_patch - 1) * stride) + 1
  23         raise ValueError(f"a reference video needs at least {minimum} frames at {fps:g} fps, got {num_frames}")
  24     ts = [i / sample_fps for i in range(len(indices))]
  25     ts += [ts[-1]] * (-len(ts) % temporal_patch)
  26     blocks = [(ts[i] + ts[i + temporal_patch - 1]) / 2 for i in range(0, len(ts), temporal_patch)]
  27     return indices, blocks
```

The encoder itself is mlx-vlm's Qwen3-VL. `hidden_state_at_layer` stops after layer 50 without the final norm. The
loader builds only those 50 layers, drops the LM head, and never reads the other layers from disk:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/encoders.py:30-99`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L30-L99)
```python
  30 def get_qwen3vl_prompt_embeds(
  31     text_encoder,
  32     processor,
  33     token_ids: list[int],
  34     vision_inputs: dict | None = None,
  35     text_encoder_layer: int = 50,
  36     device: torch.device | None = None,
  37     dtype: torch.dtype | None = None,
  38 ) -> torch.Tensor:
  39     r"""
  40     Encode one tokenized MiniMax-H3 presentation with the Qwen3-VL conditioner.
  41
  42     Unlike its `get_<encoder>_prompt_embeds` siblings this takes the *tokenized* presentation rather than a prompt
  43     string: MiniMax-H3's tokenization is the presentation-building phase that precedes it, where every reference or
  44     keyframe prepends a label and a vision block. The `mm_token_type_ids` built here are Qwen-internal (`0` text, `1`
  45     image, `2` video — they drive Qwen3-VL's per-modality-run rotary layout and never leave this function) and are not
  46     MiniMax-H3's own per-row modality tags, which the presentation phase produces alongside `token_ids`.
  47
  48     Args:
  49         text_encoder (`Qwen3VLForConditionalGeneration`): The conditioner.
  50         processor (`Qwen3VLProcessor`): Its processor, which derives the token type ids from the vision pad ids.
  51         token_ids (`list[int]`): The tokenized presentation.
  52         vision_inputs (`dict`, *optional*):
  53             The vision tensors of the presentation's blocks, by the conditioner's own parameter names — `pixel_values`
  54             / `image_grid_thw` for images, `pixel_values_videos` / `video_grid_thw` for videos.
  55         text_encoder_layer (`int`, *optional*, defaults to 50):
  56             Which hidden state conditions the transformer, i.e. `components.text_encoder_layer`.
  57         device (`torch.device`, *optional*): The device to run the conditioner on.
  58         dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.
  59
  60     Returns:
  61         `torch.Tensor`: the `(1, num_text_tokens, 5120)` hidden state after decoder layer `text_encoder_layer`.
  62     """
  63     num_layers = text_encoder.config.text_config.num_hidden_layers
  64     if num_layers <= text_encoder_layer:
  65         raise ValueError(
  66             f"MiniMax-H3 conditions on `hidden_states[{text_encoder_layer}]` of its Qwen3-VL "
  67             f"conditioner, which needs more than {text_encoder_layer} decoder layers, but "
  68             f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
  69             f"{text_encoder_layer} layers is post-norm and is not the conditioning MiniMax-H3 expects."
  70         )
  71
  72     input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
  73     mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
  74
  75     vision_kwargs = {}
  76     for name, value in (vision_inputs or {}).items():
  77         vision_kwargs[name] = value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)
  78
  79     # `text_encoder.model` is a submodule, and a CPU-offload hook — accelerate's or the one the
  80     # `ComponentsManager` attaches — wraps the *top-level* module's `forward` alone, so calling the submodule
  81     # directly would leave the conditioner on the CPU. Fire the hook by hand instead of routing through
  82     # `text_encoder(...)`: MiniMax-H3 reads `hidden_states[50]` and never uses the language-model head, whose
  83     # vocabulary-wide projection over every token is all the top-level forward would add.
  84     # TODO: firing another module's offload hook by hand is not something a block should do. It is here
  85     # because MiniMax-H3 reads `hidden_states[50]` off `text_encoder.model` while the hook wraps only the
  86     # top-level `forward`. Needs a real answer — an opt-in on the conditioner, or a hook that follows
  87     # submodule calls.
  88     hook = getattr(text_encoder, "_hf_hook", None)
  89     if hook is not None and hasattr(hook, "pre_forward"):
  90         hook.pre_forward(text_encoder)
  91     outputs = text_encoder.model(
  92         input_ids=input_ids,
  93         attention_mask=torch.ones_like(input_ids),
  94         mm_token_type_ids=mm_token_type_ids,
  95         use_cache=False,
  96         output_hidden_states=True,
  97         **vision_kwargs,
  98     )
  99     return outputs.hidden_states[text_encoder_layer].to(device=device, dtype=dtype)
```
📍 `src/h3mlx/load.py:85-123`
```python
  85 def load_text_encoder(d, keep_layers: int = 50, dtype=None):
  86     """mlx-vlm Qwen3-VL from the release's `text_encoder/`, with only decoder layers [0, keep_layers) and no lm_head.
  87     Returns (model, encode(token_ids, vision_inputs) -> (1, n, hidden) after `keep_layers` layers, pre-norm)."""
  88     from mlx.utils import tree_flatten
  89     from mlx_vlm.models.qwen3_vl import Model, ModelConfig
  90     from mlx_vlm.models.qwen3_vl.vision import VisionModel
  91     from mlx_vlm.utils import sanitize_weights
  92
  93     config = ModelConfig.from_dict(qwen_params(_config(d)))
  94     n_layers = config.text_config.num_hidden_layers
  95     if not 0 < keep_layers <= n_layers:
  96         raise ValueError(f"keep_layers must be in (0, {n_layers}], got {keep_layers}")
  97
  98     config.text_config.num_hidden_layers = keep_layers  # never build the layers past the cut
  99     model = Model(config)
 100     model.language_model.pop("lm_head", None)  # unused; would be a random 5120 x 151936 matrix
 101     layer_re = re.compile(r"language_model\.layers\.(\d+)\.")
 102
 103     def keep(k):
 104         if "lm_head" in k:
 105             return False
 106         m = layer_re.search(k)
 107         return m is None or int(m.group(1)) < keep_layers
 108
 109     w = load_safetensors(d, keep)
 110     w = sanitize_weights(VisionModel, sanitize_weights(model, w), config.vision_config)
 111     if dtype is not None:
 112         w = {k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v for k, v in w.items()}
 113     model.load_weights(list(w.items()), strict=False)
 114     want = {k for k, _ in tree_flatten(model.parameters()) if "lm_head" not in k}
 115     missing = want - set(w)
 116     if missing:
 117         raise ValueError(f"text encoder weights missing: {sorted(missing)[:6]} (+{max(0, len(missing) - 6)})")
 118
 119     def encode(token_ids, vision_inputs):
 120         kw = {k: mx.array(np.asarray(v)) for k, v in (vision_inputs or {}).items()}
 121         return model.hidden_state_at_layer(mx.array([token_ids]), keep_layers, **kw)
 122
 123     return model, encode
```
📍 `src/h3mlx/load.py:126-138`
```python
 126 def staged_text_encoder(d, keep_layers: int = 50, dtype=None):
 127     """An `encode(token_ids, vision_inputs)` that loads the text encoder, runs it, and frees it before returning, so
 128     it is never resident together with the transformer (~51 GB bf16 for 50 layers of the release)."""
 129
 130     def encode(token_ids, vision_inputs):
 131         model, run = load_text_encoder(d, keep_layers, dtype)
 132         out = run(token_ids, vision_inputs)
 133         mx.eval(out)
 134         del model, run
 135         mx.clear_cache()
 136         return out
 137
 138     return encode
```

`tests/test_presentation.py` checks token ids and tags bit-exact with the released tokenizer.
`tests/test_text_encoder_tiny.py` compares mlx-vlm with transformers on a tiny Qwen3-VL that keeps interleaved mrope
and deepstack, with a picture and a video in one presentation. Without `video_grid_thw`, mlx-vlm 0.7.11 raises rather
than misplace the video tokens.

---

## 6. The video VAE

A causal 3-D CNN encoder (reflect padding in space, zero frames prepended in time, group norm per frame) and a
non-causal ViT decoder, both tiled spatially at 256 px with 64 px overlap.

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:57-65`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L57-L65)
```python
  57     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
  58         if self.spatial_padding > 0:
  59             padding = self.spatial_padding
  60             hidden_states = F.pad(
  61                 hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
  62             )
  63         if self.temporal_padding > 0:
  64             hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
  65         return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)
```
📍 `src/h3mlx/vae_ops.py:24-40`
```python
  24 def causal_conv3d(
  25     x: mx.array,
  26     w: mx.array,
  27     b: mx.array | None,
  28     stride: tuple[int, int, int] = (1, 1, 1),
  29     spatial_padding: int = 0,
  30     temporal_padding: int = 0,
  31 ) -> mx.array:
  32     """MiniMaxH3VideoCausalConv3d: symmetric reflect pad on H, W; `temporal_padding` zero frames prepended only.
  33     x: (B, F, H, W, C); w: MLX layout (O, kt, kh, kw, I)."""
  34     if spatial_padding > 0:
  35         x = reflect_pad(x, 2, spatial_padding, spatial_padding)
  36         x = reflect_pad(x, 3, spatial_padding, spatial_padding)
  37     if temporal_padding > 0:
  38         x = mx.pad(x, [(0, 0), (temporal_padding, 0), (0, 0), (0, 0), (0, 0)])
  39     y = mx.conv3d(x, w, stride=stride, padding=0)
  40     return y + b if b is not None else y
```

MLX's `pad` has no reflect mode, so the port gathers the mirrored indices:

📍 `src/h3mlx/vae_ops.py:15-21`
```python
  15 def reflect_pad(x: mx.array, axis: int, before: int, after: int) -> mx.array:
  16     """torch F.pad(mode="reflect") along one axis (edge sample not repeated)."""
  17     n = x.shape[axis]
  18     if before >= n or after >= n:
  19         raise ValueError(f"reflect pad ({before}, {after}) needs size > pad, got {n}")
  20     idx = list(range(before, 0, -1)) + list(range(n)) + list(range(n - 2, n - 2 - after, -1))
  21     return mx.take(x, mx.array(idx), axis=axis)
```

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:74-80`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L74-L80)
```python
  74     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
  75         batch_size, num_channels, num_frames, height, width = hidden_states.shape
  76         hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
  77         hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
  78         hidden_states = super().forward(hidden_states)
  79         hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
  80         return hidden_states.permute(0, 2, 1, 3, 4).contiguous()
```
📍 `src/h3mlx/vae_ops.py:43-51`
```python
  43 def group_norm_per_frame(x: mx.array, groups: int, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
  44     """MiniMaxH3VideoGroupNorm: GroupNorm with statistics per (batch, frame), never across frames. Biased variance,
  45     computed in float32."""
  46     b, f, h, w, c = x.shape
  47     xf = x.astype(mx.float32).reshape(b, f, h * w, groups, c // groups)
  48     mean = xf.mean(axis=(2, 4), keepdims=True)
  49     var = ((xf - mean) ** 2).mean(axis=(2, 4), keepdims=True)
  50     y = ((xf - mean) * mx.rsqrt(var + eps)).reshape(b, f, h, w, c).astype(x.dtype)
  51     return y * weight + bias
```

**A still goes through the spatial encoder alone.** Padding a single image to a 17-frame clip would run the temporal
path, which is not the conditioning MiniMax trained with. The third-party port's public `encode` gets this wrong.

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:776-802`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L776-L802)
```python
 776     def _encode(self, x: torch.Tensor) -> torch.Tensor:
 777         r"""
 778         Encode a video in `clip_length`-frame chunks and drop the `token_drop` trailing latent frames.
 779
 780         A single frame has no temporal extent to chunk, so it goes through the spatial encoder alone. Padding it up to
 781         `clip_length` by repetition instead would run the temporal path over `clip_length` copies of the same image and
 782         return `clip_length // temporal_compression_ratio - token_drop` latent frames rather than one — which is not
 783         the conditioning MiniMax-H3 was trained with.
 784         """
 785         clip_length = self.config.clip_length
 786         num_frames = x.shape[2]
 787         if num_frames == 1:
 788             return self._encode_clip(x)
 789         if num_frames % clip_length != 0:
 790             pad_frames = x[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
 791             x = torch.cat([x, pad_frames], dim=2)
 792
 793         moments = torch.cat(
 794             [
 795                 self._encode_clip(x[:, :, i * clip_length : (i + 1) * clip_length])
 796                 for i in range(x.shape[2] // clip_length)
 797             ],
 798             dim=2,
 799         )
 800         if self.config.token_drop > 0:
 801             moments = moments[:, :, : -self.config.token_drop]
 802         return moments
```
📍 `src/h3mlx/vae.py:159-172`
```python
 159     def encode_moments(self, x):
 160         """Pixels (B, F, H, W, 3) -> moments (B, F', H', W', 2 * latent). One frame: spatial encoder alone."""
 161         c = self.cfg
 162         n = x.shape[1]
 163         if n == 1:
 164             return self.encode_clip(x)
 165         if n % c.clip_length:
 166             x = mx.concatenate([x, mx.repeat(x[:, -1:], (-n) % c.clip_length, axis=1)], axis=1)
 167         m = mx.concatenate(
 168             [_evaluated(self.encode_clip(x[:, i * c.clip_length : (i + 1) * c.clip_length]))
 169              for i in range(x.shape[1] // c.clip_length)],
 170             axis=1,
 171         )
 172         return m[:, : -c.token_drop] if c.token_drop > 0 else m
```

Tiling and the chunked, cross-faded decode:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:656-677`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L656-L677)
```python
 656     def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
 657         r"""
 658         Lay `tile_size`-wide tiles over `length` pixels. The number of tiles is the smallest one whose union can cover
 659         `length` while keeping every overlap at least `min_overlap`; the slack is then distributed round-robin over the
 660         overlaps in whole `spatial_compression_ratio` steps so that every tile boundary stays latent-aligned.
 661         """
 662         if tile_size >= length:
 663             return [0], [length], []
 664
 665         num_tiles = math.ceil(length / tile_size)
 666         while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
 667             num_tiles += 1
 668
 669         overlaps = [min_overlap] * (num_tiles - 1)
 670         remaining = tile_size * num_tiles - sum(overlaps) - length
 671         for i in range(remaining // self.spatial_compression_ratio):
 672             overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio
 673
 674         tile_start_indices = [0]
 675         for i in range(num_tiles - 1):
 676             tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
 677         return tile_start_indices, [tile_size] * num_tiles, overlaps
```
📍 `src/h3mlx/vae.py:69-82`
```python
  69 def split_tiles(length: int, tile: int, min_overlap: int, ratio: int):
  70     if tile >= length:
  71         return [0], [length], []
  72     n = math.ceil(length / tile)
  73     while tile * n - min_overlap * (n - 1) - length < 0:
  74         n += 1
  75     overlaps = [min_overlap] * (n - 1)
  76     remaining = tile * n - sum(overlaps) - length
  77     for i in range(remaining // ratio):
  78         overlaps[i % (n - 1)] += ratio
  79     starts = [0]
  80     for i in range(n - 1):
  81         starts.append(starts[-1] + tile - overlaps[i])
  82     return starts, [tile] * n, overlaps
```
📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:804-853`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L804-L853)
```python
 804     def _decode(self, z: torch.Tensor) -> torch.Tensor:
 805         r"""
 806         Decode a latent video, mirroring the chunking that `_encode` applied.
 807
 808         `token_drop` removed the tail of every encoded chunk, so consecutive decoded chunks overlap by `frame_overlap`
 809         pixel frames and are linearly cross-faded. Latent frames are repeated at the end when the length is not a whole
 810         number of chunks; the extra pixel frames are cut off again at the end.
 811         """
 812         tokens_chunk_size = self.tokens_chunk_size
 813         token_drop = self.config.token_drop
 814         temporal_ratio = self.temporal_compression_ratio
 815         chunk_num_frames = tokens_chunk_size * temporal_ratio
 816
 817         num_tokens = z.shape[2] + token_drop
 818         pad_tokens = (-num_tokens) % tokens_chunk_size
 819         num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
 820         if pad_tokens > 0:
 821             z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
 822
 823         decoded_chunks = []
 824         overlap = None
 825         for i in range(num_chunks):
 826             start = i * tokens_chunk_size
 827             clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
 828             for j in range(int(token_drop > 0) + 1):
 829                 frame_start = j * chunk_num_frames
 830                 chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
 831                 chunk = chunk[:, :, self.frame_pre_padding :]
 832                 if j == 0:
 833                     if overlap is not None:
 834                         chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
 835                     decoded_chunks.append(chunk)
 836                 else:
 837                     overlap = chunk
 838         if overlap is not None:
 839             decoded_chunks.append(overlap)
 840
 841         dec = torch.cat(decoded_chunks, dim=2)
 842
 843         # `pad_tokens` repeated latent frames produced trailing pixel frames that were never requested. A chunk's
 844         # last latent frame only covers `clip_length % temporal_ratio` pixel frames, the others cover `temporal_ratio`.
 845         if pad_tokens > 0:
 846             intra_tail = self.config.clip_length % temporal_ratio
 847             num_tokens_before_pad = z.shape[2] - pad_tokens
 848             pad_frames = sum(
 849                 intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
 850                 for k in range(pad_tokens)
 851             )
 852             dec = dec[:, :, :-pad_frames]
 853         return dec
```
📍 `src/h3mlx/vae.py:194-223`
```python
 194     def decode(self, z):
 195         """Latents (B, F, H, W, latent) -> pixels (B, frames, H*r, W*r, 3), chunked and cross-faded like `_decode`."""
 196         c = self.cfg
 197         tcs, drop, tr = c.tokens_chunk_size, c.token_drop, c.temporal_ratio
 198         chunk_frames = tcs * tr
 199         num_tokens = z.shape[1] + drop
 200         pad = (-num_tokens) % tcs
 201         n_chunks = (num_tokens + pad) // tcs - int(drop > 0)
 202         if pad:
 203             z = mx.concatenate([z, mx.repeat(z[:, -1:], pad, axis=1)], axis=1)
 204         out, overlap = [], None
 205         for i in range(n_chunks):
 206             clip = _evaluated(self.decode_clip(z[:, i * tcs : i * tcs + tcs + c.token_overlap]))
 207             for j in range(int(drop > 0) + 1):
 208                 ch = clip[:, j * chunk_frames : j * chunk_frames + chunk_frames][:, c.frame_pre_padding :]
 209                 if j == 0:
 210                     if overlap is not None:
 211                         ch = blend(overlap, ch, c.frame_overlap, axis=1)
 212                     out.append(ch)
 213                 else:
 214                     overlap = ch
 215         if overlap is not None:
 216             out.append(overlap)
 217         dec = mx.concatenate(out, axis=1)
 218         if pad:
 219             tail = c.clip_length % tr
 220             before = z.shape[1] - pad
 221             pad_frames = sum(tail if tail and (before + k) % tcs == 0 else tr for k in range(pad))
 222             dec = dec[:, :-pad_frames]
 223         return dec
```

The ViT decoder: every latent voxel is one token, register tokens plus one zero token sit at position 0, norms run in
f32, and there are layer scales:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:449-502`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L449-L502)
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
📍 `src/h3mlx/vae_ops.py:182-200`
```python
 182 def vit_decoder(
 183     z: mx.array, p: dict, num_layers: int, heads: int, patch_size: int, patch_size_t: int, out_channels: int,
 184     rope_dim: int, rope_theta: float = 100.0, eps: float = 1e-5,
 185 ) -> mx.array:
 186     """MiniMaxH3VideoViTDecoder3d. z: channels-last latents (B, F, H, W, C) -> pixels (B, F*pt, H*p, W*p, out)."""
 187     b, f, h, w, c = z.shape
 188     x = _lin(z.reshape(b, f * h * w, c), p, "proj_in")
 189     n = x.shape[1]
 190     regs = mx.broadcast_to(p["register_tokens"].astype(x.dtype), (b, *p["register_tokens"].shape[1:]))
 191     x = mx.concatenate([x, regs, mx.zeros((b, 1, x.shape[-1]), dtype=x.dtype)], axis=1)
 192     rotary = vit_rope(vit_positions(f, h, w, regs.shape[1] + 1), vit_rope_inv_freq(rope_dim, rope_theta))
 193     for i in range(num_layers):
 194         x = vit_block(x, _sub(p, f"transformer_blocks.{i}."), heads, eps, rotary)
 195     x = mx.fast.layer_norm(x.astype(mx.float32), p["norm_out.weight"].astype(mx.float32),
 196                            p["norm_out.bias"].astype(mx.float32), eps)
 197     x = _lin(x.astype(p["proj_out.weight"].dtype), p, "proj_out")[:, :n]
 198     x = x.reshape(b, f, h, w, out_channels, patch_size_t, patch_size, patch_size)
 199     x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (b, f, pt, h, p, w, p, c)
 200     return x.reshape(b, f * patch_size_t, h * patch_size, w * patch_size, out_channels)
```

The conditioning recipe: ImageNet-normalised pixels, the posterior **sampled** from a fresh seed-42 generator, a
float16 round trip, then normalisation by the latent statistics:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/encoders.py:102-136`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L102-L136)
```python
 102 def encode_vae_condition(
 103     vae, pixels: torch.Tensor, pixel_mean: tuple, pixel_std: tuple, encode_seed: int = 42
 104 ) -> torch.Tensor:
 105     r"""
 106     Encode one visual condition — a keyframe or a `ref2va` reference — into normalized conditioning latents.
 107
 108     The recipe is the released model's, and every part of it is needed to reproduce its conditioning: the pixels are
 109     ImageNet-normalized, the posterior is *sampled* under a fresh generator seeded independently of the request, and
 110     the sampled latent is rounded to float16 (~11 bits of every conditioning latent) before being normalized. A single
 111     frame is encoded by the (tiled) spatial encoder alone; a frame stack goes through the temporal chunking, which is
 112     what turns `17 * n + 5` frames into `5 * n + 2` latent frames.
 113
 114     Args:
 115         vae (`AutoencoderKLMiniMaxH3`): The video VAE.
 116         pixels (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`):
 117             The condition's `uint8` pixels, on the VAE's device. `num_frames` is 1 for a keyframe or an image
 118             reference.
 119         pixel_mean (`tuple[float, float, float]`), pixel_std (`tuple[float, float, float]`):
 120             The video VAE's pixel convention, i.e. `components.pixel_mean` / `components.pixel_std`.
 121         encode_seed (`int`, *optional*, defaults to 42):
 122             Seed the posterior is sampled under, i.e. `components.keyframe_encode_seed`.
 123
 124     Returns:
 125         `torch.Tensor`: one `(1, latent_channels, num_latent_frames, latent_height, latent_width)` float32 CPU tensor.
 126     """
 127     latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
 128     latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
 129     pixel_mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
 130     pixel_std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)
 131
 132     pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
 133     posterior = vae.encode(pixels, return_dict=False)[0]
 134     latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
 135     latents = latents.to(torch.float16).float().cpu()
 136     return (latents - latents_mean) / latents_std
```
📍 `src/h3mlx/vae.py:232-240`
```python
 232 def encode_condition(vae: VideoVAE, pixels_u8: mx.array, pixel_mean, pixel_std, latents_mean, latents_std, noise_fn):
 233     """`encode_vae_condition`: uint8 pixels (1, F, H, W, 3) -> ImageNet-normalised -> moments -> posterior sample
 234     (noise from `noise_fn(shape)`, which must reproduce torch.Generator().manual_seed(42) to match the release) ->
 235     float16 round trip -> (latent - mean) / std."""
 236     x = (pixels_u8.astype(mx.float32) / 255.0 - mx.array(pixel_mean)) / mx.array(pixel_std)
 237     m = vae.encode_moments(x)
 238     lat = posterior_sample(m, noise_fn(m.shape[:-1] + (m.shape[-1] // 2,)))
 239     lat = lat.astype(mx.float16).astype(mx.float32)
 240     return (lat - mx.array(latents_mean)) / mx.array(latents_std)
```

---

## 7. The audio VAE

A DAC encoder and a BigVGAN decoder on mono 32 kHz audio (stereo is a batch of two), kept in float32: diffusers notes
that bf16 decodes come out about 20 dB quieter. The alias-free SnakeBeta activation upsamples ×2, activates, then
low-passes back down:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:185-191`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py#L185-L191)
```python
 185     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
 186         num_channels = hidden_states.shape[1]
 187         hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
 188         hidden_states = self.ratio * F.conv_transpose1d(
 189             hidden_states, self.filter.expand(num_channels, -1, -1), stride=self.stride, groups=num_channels
 190         )
 191         return hidden_states[..., self.pad_left : -self.pad_right]
```
📍 `src/h3mlx/audio_vae.py:73-86`
```python
  73 def upsample(x, filt: mx.array, ratio: int):
  74     """ratio * conv_transpose1d(replicate-pad(x, pad), filter, stride=ratio, groups=C), then crop."""
  75     k = filt.size
  76     pad = k // ratio - 1
  77     pad_left = pad * ratio + (k - ratio) // 2
  78     pad_right = pad * ratio + (k - ratio + 1) // 2
  79     x = mx.pad(x, [(0, 0), (pad, pad), (0, 0)], mode="edge")
  80     b, n, c = x.shape
  81     # transposed depthwise conv as zero-insertion + conv with the flipped kernel
  82     up = mx.zeros((b, (n - 1) * ratio + 1, c), dtype=x.dtype)
  83     up[:, ::ratio] = x
  84     up = mx.pad(up, [(0, 0), (k - 1, k - 1), (0, 0)])
  85     y = ratio * mx.conv1d(up, _depthwise(filt.reshape(-1)[::-1], c), groups=c)
  86     return y[:, pad_left : y.shape[1] - pad_right]
```

MLX's `conv_transpose1d` has no grouped form here, so the port inserts zeros and convolves with the flipped kernel.
The Kaiser filters are read from the checkpoint's own buffers instead of being recomputed:

📍 `src/h3mlx/audio_vae.py:89-94`
```python
  89 def activation1d(x, p, ratio: int = 2):
  90     """Alias-free SnakeBeta: upsample x2, activate, low-pass downsample x2. Filters are the checkpoint's buffers
  91     (`upsample.filter`, `downsample.lowpass.filter`), Kaiser-sinc, kernel 12."""
  92     x = upsample(x, p["upsample.filter"], ratio)
  93     x = snake_beta(x, p["act.alpha"], p["act.beta"])
  94     return lowpass(x, p["downsample.lowpass.filter"], ratio)
```

The causal attention in `pre_block` mean-pools its heads away, then adaptively average-pools the head dimension:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:315-338`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py#L315-L338)
```python
 315     def __call__(self, attn: "MiniMaxH3AudioCausalAttention", hidden_states: torch.Tensor) -> torch.Tensor:
 316         batch_size, seq_len, _ = hidden_states.shape
 317         qkv = F.linear(
 318             input=hidden_states,
 319             weight=attn.qkv.weight,
 320             bias=torch.cat((attn.q_bias, attn.zero_k_bias, attn.v_bias)),
 321         )
 322         query, key, value = (
 323             qkv.reshape(batch_size, seq_len, 3, attn.num_heads, attn.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
 324         )
 325         hidden_states = dispatch_attention_fn(
 326             query,
 327             key,
 328             value,
 329             attn_mask=None,
 330             is_causal=True,
 331             backend=self._attention_backend,
 332             parallel_config=self._parallel_config,
 333         )
 334         # The heads are mean-pooled away instead of being concatenated, and the head dimension that
 335         # remains is adaptively average-pooled down to `out_dim`.
 336         hidden_states = torch.mean(hidden_states, dim=2)
 337         hidden_states = F.adaptive_avg_pool1d(hidden_states, attn.out_dim)
 338         return attn.proj(hidden_states)
```
📍 `src/h3mlx/audio_vae.py:155-162`
```python
 155 def causal_attention(x, p, heads, out_dim):
 156     b, n, c = x.shape
 157     bias = mx.concatenate([p["q_bias"], mx.zeros_like(p["q_bias"]), p["v_bias"]])
 158     qkv = (x @ p["qkv.weight"].T + bias).reshape(b, n, 3, heads, c // heads)
 159     q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
 160     o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(c // heads), mask="causal")
 161     o = o.transpose(0, 2, 1, 3).mean(axis=2)  # heads mean-pooled away
 162     return lin(adaptive_avg_pool_last(o, out_dim), p, "proj")
```

Weight-norm pairs (`weight_g`, `weight_v`) are folded once at load:

📍 `src/h3mlx/audio_vae.py:17-31`
```python
  17 def fold_weight_norm(sd: dict) -> dict:
  18     """torch.nn.utils.weight_norm (dim=0): weight = g * v / ||v|| with the norm over every axis but 0."""
  19     out = {}
  20     for k, v in sd.items():
  21         if k.endswith(".weight_g"):
  22             continue
  23         if k.endswith(".weight_v"):
  24             base = k[: -len("_v")]
  25             g = np.asarray(sd[base + "_g"], dtype=np.float32)
  26             vv = np.asarray(v, dtype=np.float32)
  27             norm = np.sqrt((vv.astype(np.float64) ** 2).sum(axis=tuple(range(1, vv.ndim)), keepdims=True))
  28             out[base] = (g * vv / norm).astype(np.float32)
  29         else:
  30             out[k] = np.asarray(v)
  31     return out
```

Reference soundtracks are resampled to 32 kHz with torchaudio's sinc resampler, as diffusers does. torchaudio's
current wheels pin an older torch than ours, so `src/h3mlx/resample.py` carries torchaudio 2.9.1's two functions
verbatim under its BSD-2 licence. `tests/test_resample.py` checks it against real torchaudio outputs.

---

## 8. ref2va end to end

The setup step validates the request (canvas multiples, at most 9/3/3/12 references, not audio alone), resolves the
canvas (16:9 with a 768 short edge by default; image references get their **own** 2048 px short edge) and normalises
media:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:387-522`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py#L387-L522)
```python
 387     @torch.no_grad()
 388     def __call__(
 389         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 390     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 391         block_state = self.get_block_state(state)
 392
 393         # 1. Validate the request.
 394         if (block_state.height is None) != (block_state.width is None):
 395             raise ValueError("`height` and `width` have to be passed together, or neither of them.")
 396         multiple = components.canvas_multiple
 397         if block_state.height is not None and (block_state.height % multiple or block_state.width % multiple):
 398             raise ValueError(
 399                 f"`height` and `width` must be multiples of {multiple}, got {block_state.height}x{block_state.width}."
 400             )
 401         if not block_state.references:
 402             raise ValueError("`ref2va` needs at least one reference; use the `t2va` workflow for text-only requests.")
 403         for index, entry in enumerate(block_state.references):
 404             if not isinstance(entry, MiniMaxH3Reference):
 405                 raise ValueError(
 406                     f"`references[{index}]` must be a [`MiniMaxH3ImageReference`], [`MiniMaxH3VideoReference`] or "
 407                     f"[`MiniMaxH3AudioReference`], got {type(entry)}. MiniMax-H3 blocks never open media files, so a "
 408                     "request that holds paths decodes them first, with each class's `from_file` classmethod."
 409                 )
 410         kinds = [entry.kind for entry in block_state.references]
 411         for kind, limit in (("image", self.max_images), ("video", self.max_videos), ("audio", self.max_audios)):
 412             if kinds.count(kind) > limit:
 413                 raise ValueError(f"MiniMax-H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}.")
 414         if len(kinds) > self.max_references:
 415             raise ValueError(
 416                 f"MiniMax-H3 accepts at most {self.max_references} references in total, got {len(kinds)}."
 417             )
 418         if set(kinds) == {"audio"}:
 419             raise ValueError(
 420                 "An audio reference has to be paired with at least one image or video reference and cannot be used "
 421                 "on its own."
 422             )
 423
 424         # 2. Resolve the canvas and the frame count. The duration the request generates is the one of the *aligned*
 425         # frame count, so that is what the ceiling holds for: 346 frames would otherwise pass the check and then be
 426         # rounded up to 362, i.e. 15.083 seconds.
 427         if block_state.height is None:
 428             block_state.height, block_state.width = resolve_canvas_size(
 429                 16, 9, multiple, components.config.canvas_short_edge, components.config.canvas_max_pixels
 430             )
 431         aligned_num_frames = align_num_frames(
 432             block_state.num_frames, components.vae_frames_per_chunk, components.vae_latents_per_chunk
 433         )
 434         duration = aligned_num_frames / components.fps
 435         if not components.min_duration <= duration <= components.max_duration:
 436             raise ValueError(
 437                 f"MiniMax-H3 generates between {components.min_duration} and {components.max_duration} seconds at "
 438                 f"{components.fps} fps, so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can "
 439                 f"encode, must be between {int(components.min_duration * components.fps)} and "
 440                 f"{int(components.max_duration * components.fps)}, got {block_state.num_frames} (rounded up to "
 441                 f"{aligned_num_frames})."
 442             )
 443         if aligned_num_frames != block_state.num_frames:
 444             logger.warning(
 445                 f"`num_frames` has to be of the form 17 * n + 5 for the video VAE; rounding {block_state.num_frames} "
 446                 f"up to {aligned_num_frames}."
 447             )
 448         block_state.num_frames = aligned_num_frames
 449
 450         # 3. Normalize the media, in packed order.
 451         normalized = []
 452         for entry in block_state.references:
 453             waveform = None
 454             if entry.has_audio:
 455                 sample_rate = entry.sample_rate
 456                 if sample_rate is None:
 457                     sample_rate = components.audio_sampling_rate
 458                 waveform = self._normalize_audio_condition(
 459                     entry.audio,
 460                     sample_rate,
 461                     components.audio_sampling_rate,
 462                     max_duration=block_state.num_frames / components.fps,
 463                 )
 464
 465             if entry.kind == "image":
 466                 # Images are encoded at high detail — a short edge of their own, 2048 for the released checkpoint,
 467                 # upscaling included and with *no* area cap — unlike video references and the target itself, which
 468                 # share the one canvas rule.
 469                 #
 470                 # Any accepted layout onto a PIL image first, through the processor's own conversions: a
 471                 # `torch.Tensor` is channels-first, as everywhere else in diffusers, and a `np.ndarray`
 472                 # channels-last. Both carry floating point over `[0, 1]`, which is what `numpy_to_pil` scales
 473                 # back up, so `uint8` is normalized before it gets there. Everything below is the geometry of a
 474                 # decoded image, and `image_processor.resize` interpolates an array with `F.interpolate` rather
 475                 # than the LANCZOS the released model was conditioned on, so nothing may reach it as one.
 476                 image = entry.image
 477                 if isinstance(image, torch.Tensor):
 478                     if image.dtype == torch.uint8:
 479                         image = image.float() / 255.0
 480                     image = components.image_processor.pt_to_numpy(image[None])[0]
 481                 if isinstance(image, np.ndarray):
 482                     if image.ndim != 3 or image.shape[2] != 3:
 483                         raise ValueError(
 484                             f"A reference image must be `(height, width, 3)` RGB pixels, got {tuple(image.shape)}."
 485                         )
 486                     if image.dtype == np.uint8:
 487                         image = image.astype(np.float32) / 255.0
 488                     image = components.image_processor.numpy_to_pil(image)[0]
 489                 if image.size[0] <= 0 or image.size[1] <= 0:
 490                     raise ValueError(f"A reference image must have a positive size, got {image.size}.")
 491                 width, height = image.size
 492                 if width > 4 * height or height > 4 * width:
 493                     raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
 494                 scale = components.config.reference_image_short_edge / min(width, height)
 495                 target_height = max(multiple, round(height * scale / multiple) * multiple)
 496                 target_width = max(multiple, round(width * scale / multiple) * multiple)
 497                 if image.size != (target_width, target_height):
 498                     image = components.image_processor.resize(image, height=target_height, width=target_width)
 499                 normalized.append(MiniMaxH3ImageReference(image=image))
 500             elif entry.kind == "video":
 501                 normalized.append(
 502                     MiniMaxH3VideoReference(
 503                         frames=self._normalize_video_condition(
 504                             entry.frames,
 505                             float(entry.fps),
 506                             block_state.num_frames,
 507                             multiple,
 508                             components.config.canvas_short_edge,
 509                             components.config.canvas_max_pixels,
 510                             float(components.fps),
 511                         ),
 512                         fps=float(components.fps),
 513                         audio=waveform,
 514                         sample_rate=None if waveform is None else components.audio_sampling_rate,
 515                     )
 516                 )
 517             else:
 518                 normalized.append(MiniMaxH3AudioReference(audio=waveform, sample_rate=components.audio_sampling_rate))
 519         block_state.normalized_references = normalized
 520
 521         self.set_block_state(state, block_state)
 522         return components, state
```
📍 `src/h3mlx/pipeline.py:115-152`
```python
 115 def setup(c: Components, references, height, width, num_frames):
 116     """`MiniMaxH3Ref2VASetupStep`: validate, resolve the canvas and frame count, normalise references in order."""
 117     m = c.canvas_multiple
 118     if (height is None) != (width is None):
 119         raise ValueError("height and width have to be passed together, or neither of them")
 120     if height is not None and (height % m or width % m):
 121         raise ValueError(f"height and width must be multiples of {m}, got {height}x{width}")
 122     if not references:
 123         raise ValueError("ref2va needs at least one reference")
 124     kinds = [getattr(r, "kind", None) for r in references]
 125     if any(k not in ("image", "video", "audio") for k in kinds):
 126         raise ValueError(f"references must be ImageRef, VideoRef or AudioRef, got kinds {kinds}")
 127     for kind, limit in (("image", MAX_IMAGES), ("video", MAX_VIDEOS), ("audio", MAX_AUDIOS)):
 128         if kinds.count(kind) > limit:
 129             raise ValueError(f"at most {limit} {kind} references, got {kinds.count(kind)}")
 130     if len(kinds) > MAX_REFERENCES:
 131         raise ValueError(f"at most {MAX_REFERENCES} references in total, got {len(kinds)}")
 132     if set(kinds) == {"audio"}:
 133         raise ValueError("an audio reference has to be paired with at least one image or video reference")
 134     if height is None:
 135         height, width = R.resolve_canvas_size(16, 9, m, c.canvas_short_edge, c.canvas_max_pixels)
 136     num_frames = R.align_num_frames(num_frames)
 137     if not 5.0 <= num_frames / R.FPS <= 15.0:
 138         raise ValueError(f"duration must be 5-15 s at 24 fps, got {num_frames} frames")
 139     out = []
 140     for r in references:
 141         wave = None
 142         if r.has_audio:
 143             wave = _normalize_audio(r.audio, r.sample_rate or c.audio_sampling_rate, c.audio_sampling_rate, num_frames / R.FPS)
 144         if r.kind == "image":
 145             out.append(ImageRef(R.normalize_image(r.image, c.reference_image_short_edge, m)))
 146         elif r.kind == "video":
 147             frames = R.normalize_video_frames(r.frames, float(r.fps), num_frames, m, c.canvas_short_edge,
 148                                               c.canvas_max_pixels)
 149             out.append(VideoRef(frames, float(R.FPS), wave, None if wave is None else c.audio_sampling_rate))
 150         else:
 151             out.append(AudioRef(wave, c.audio_sampling_rate))
 152     return height, width, num_frames, out
```

The noise draws happen in one fixed order from **one** generator: each condition, then the target video tensor, then
the target audio rows. That order is part of what a seed reproduces:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_denoise.py:954-983`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L954-L983)
```python
 954     @torch.no_grad()
 955     def __call__(
 956         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 957     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 958         block_state = self.get_block_state(state)
 959         device = components._execution_device
 960         patch_size = components.patch_size
 961
 962         # One draw per condition, in packed order. Each is packed on its own because `ref2va` references are encoded
 963         # at their own resolutions, so their latents do not share a shape.
 964         packed = []
 965         for condition in block_state.condition_latents:
 966             noise = randn_tensor(condition.shape, generator=block_state.generator, device=device, dtype=torch.float32)
 967             # The anchors are not fully clean: the released model noises them to `t = 0.999` and holds them there
 968             # for every step. Mixing before the patchify is the same arithmetic, since patchify only permutes.
 969             noised = components.scheduler.scale_noise(condition.to(device), components.keyframe_noise_aug, noise)
 970             packed.append(patchify_video_latents(noised, patch_size))
 971         block_state.condition_rows = torch.cat(packed)
 972         # In a hand-assembled chain the canvas reaching the layout is user input, so it can disagree with the
 973         # conditioning that was actually encoded. Left alone the mismatch first surfaces as an `index_copy` shape
 974         # error inside the transformer, 50 layers deep.
 975         if block_state.condition_rows.shape[0] != block_state.num_condition_video_rows:
 976             raise ValueError(
 977                 f"The layout reserved {block_state.num_condition_video_rows} conditioning rows but the encoded "
 978                 f"conditioning latents pack into {block_state.condition_rows.shape[0]}. The canvas the layout was "
 979                 "built from and the one the conditioning was encoded at do not agree."
 980             )
 981
 982         self.set_block_state(state, block_state)
 983         return components, state
```
📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_denoise.py:853-898`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L853-L898)
```python
 853     @torch.no_grad()
 854     def __call__(
 855         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 856     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 857         block_state = self.get_block_state(state)
 858         device = components._execution_device
 859         patch_size = components.patch_size
 860
 861         # A request draws every stream from the one generator it is given, and the order is part of what that
 862         # generator reproduces: any conditioning noise first (drawn before this block), then the video noise as a
 863         # latent tensor, then the audio noise directly in row layout. Passing `latents` or `audio_latents` skips its
 864         # draw and shifts the ones after it.
 865         latents = block_state.latents
 866         if latents is None:
 867             latents = randn_tensor(
 868                 (
 869                     1,
 870                     components.vae_latent_channels,
 871                     block_state.num_latent_frames,
 872                     block_state.latent_height,
 873                     block_state.latent_width,
 874                 ),
 875                 generator=block_state.generator,
 876                 device=device,
 877                 dtype=torch.float32,
 878             )
 879         video_rows = patchify_video_latents(latents.to(device, torch.float32), patch_size)
 880
 881         if block_state.audio_latents is None:
 882             audio_rows = randn_tensor(
 883                 (block_state.num_audio_latents * components.audio_channels, components.audio_latent_channels),
 884                 generator=block_state.generator,
 885                 device=device,
 886                 dtype=torch.float32,
 887             )
 888         else:
 889             audio_rows = (
 890                 block_state.audio_latents.to(device, torch.float32)
 891                 .permute(0, 2, 1)
 892                 .reshape(-1, components.audio_latent_channels)
 893             )
 894
 895         block_state.latents, block_state.audio_latents = video_rows, audio_rows
 896
 897         self.set_block_state(state, block_state)
 898         return components, state
```
📍 `src/h3mlx/pipeline.py:234-248`
```python
 234     gen = Noise(seed)
 235     patch = tuple(c.dit_cfg["patch_size"])
 236     rows = []
 237     for x in cond:  # one draw per condition, packed order, then held at t = 0.999
 238         b, f, h, w, ch = x.shape
 239         nz = mx.array(gen((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 240         t32 = np.float32(NOISE_AUG)  # scale_noise: t * x + (1 - t) * noise, both factors in float32
 241         noised = mx.array(t32) * x + mx.array(np.float32(1) - t32) * nz
 242         rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
 243     lat_ch = c.dit_cfg["in_channels"]
 244     target = mx.array(gen((1, lat_ch, lf, lh, lw)))
 245     rows.append(ops.patchify(target, patch))
 246     video_rows = mx.concatenate(rows)
 247     audio_ch = c.dit_cfg["audio_in_channels"]
 248     audio_rows = mx.concatenate(ref_audio + [mx.array(gen((na * AUDIO_CHANNELS, audio_ch)))])
```

And the decode, which reverses the latent normalisation, then ImageNet's:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/decoders.py:173-197`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/decoders.py#L173-L197)
```python
 173     @torch.no_grad()
 174     def __call__(
 175         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 176     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 177         block_state = self.get_block_state(state)
 178         device = components._execution_device
 179
 180         if block_state.output_type not in ("pil", "np", "pt"):
 181             raise ValueError(
 182                 f"`output_type` must be one of 'pil', 'np' or 'pt', got {block_state.output_type!r}. To keep the "
 183                 "latents instead of decoding them, run a pipeline that does not include the decode blocks."
 184             )
 185
 186         latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
 187         latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
 188         latents = block_state.latents * latents_std + latents_mean
 189
 190         video = components.vae.decode(latents, return_dict=False)[0]
 191         pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
 192         pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
 193         video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
 194         block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)
 195
 196         self.set_block_state(state, block_state)
 197         return components, state
```
📍 `src/h3mlx/pipeline.py:257-264`
```python
 257     v_lat = ops.unpatchify(v[layout.num_condition_video_rows :], lat_ch, lf, lh, lw, patch)  # (1, C, F, H, W)
 258     v_lat = v_lat.transpose(0, 2, 3, 4, 1) * mx.array(c.vae_latents_std) + mx.array(c.vae_latents_mean)
 259     px = c.vae.decode(v_lat)[0]  # (frames, H, W, 3)
 260     video = mx.clip(px * mx.array(PIXEL_STD) + mx.array(PIXEL_MEAN), 0.0, 1.0)
 261
 262     a_lat = a[layout.num_condition_audio_rows :].reshape(AUDIO_CHANNELS, na, audio_ch)
 263     a_lat = a_lat * mx.array(c.audio_latents_std) + mx.array(c.audio_latents_mean)
 264     audio = c.audio_vae.decode(a_lat)[:, :, 0]  # (2, samples)
```

`tests/test_pipeline_e2e.py` runs a whole ref2va request through diffusers' own `MiniMaxH3Blocks` and through
`h3mlx.pipeline.ref2va`. The components are tiny random models plus the released tokenizer and processor; the
references are an image and a video with a soundtrack; 3 steps. Decoded video agrees to 3.6e-7 and audio to 1.5e-7.

---

## 9. What is not ported, or not yet

- **Sparse attention, Turbo LoRAs, quantisation.** These are optional speedups. A dense bf16 baseline comes first.
- **t2va and fl2va pipelines.** The packers and ops are here and tested (`build_packed_sequence`); only ref2va is
  assembled end to end.
- **Real weights.** Everything above was checked with tiny random weights and the release's configs, index and
  tokenizer. The first run with the released 66 GB `transformer_ref` and 67 GB `text_encoder` is still ahead.