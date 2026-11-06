# Porting MiniMax H3 to MLX: diffusers 0.42.1 → `h3mlx`

This is a side-by-side tour of how MiniMax H3's reference implementation, **diffusers 0.42.1** (`huggingface/diffusers`
tag `v0.42.1`), maps onto our MLX package `h3mlx`. Each section shows the original first, then the port, then what
changed and which test ties the two together. The aim is the **ref2va** workflow: reference images, videos and audio
in, video with sound out. It uses the same transformer class as text-to-video, with different weights
(`transformer_ref`) and a different packed layout.

**Where the code comes from.** The MiniMax H3 maths follows diffusers. The audio resampler is copied verbatim from
torchaudio 2.9.1 (§7). The text encoder is mlx-vlm 0.7.11's Qwen3-VL (§5). The third-party MLX port
`yshenaw/ComfyUI-MiniMax-H3-MLX-SolAttn` is not a source; one idea from it, the
modulation cache (§2.6), was re-implemented here.

**What runs where.** The models run in MLX. CPU code around them:

- **torch (CPU):** the seeded noise, so a seed reproduces diffusers' draws, and the audio resampler.
- **transformers + torchvision:** the Qwen3-VL tokenizer and image/video processor.
- **numpy:** packed layouts, sigma schedules, weight-norm folding.
- **PIL:** image references.

All of these are runtime dependencies in `pyproject.toml`.

**Weights.** Ops take a flat `dict` of weights:

- **`transformer_ref`:** the released safetensors load by their diffusers names with no renaming
  (`tests/test_weight_names.py`).
- **Convolutions:** MLX wants channels-last, `(B, F, H, W, C)` instead of diffusers' `(B, C, F, H, W)`. Conv weights go
  `(O, I, k…)` → `(O, k…, I)`.
- **Audio's transposed convolutions** are stored `(I, O, K)` and become `(O, K, I)` (`to_mlx_layout`, §7).
- **Audio weight-norm pairs** are folded into plain `.weight` tensors.
- **Qwen keys** are renamed by mlx-vlm's own sanitiser.

**Precision.** As released:

- The transformer's block stack is bf16.
- Its `proj_in`/`proj_out`, audio heads and time embedder are f32.
- Rotary angles and their cos/sin are computed in f32; cos/sin are then cast to the query/key dtype before they are
  applied.
- The audio VAE is f32.

Explicit casts in diffusers are kept, and pointwise activations compute in f32 and round once, as torch's do. The
tests bound the differences; they do not promise bit-identity everywhere.

**Randomness.** Noise comes from torch's CPU generator, drawn in diffusers' channels-first shapes and order.

**Words used below.**

- **ulp:** the gap between two neighbouring floating-point numbers, i.e. the smallest possible difference.
- **opmath:** the precision an op computes in before casting its output.
- **t and σ:** MiniMax's time `t = 1 − σ` runs from 0 (pure noise) to 1 (clean).
- **mrope:** Qwen3-VL's rotary positions with separate time, height and width sections, interleaved.
- **deepstack:** Qwen3-VL adds features from vision layers 8, 16 and 24 into its first three language layers.

**Shapes at the release configuration**

| what | value |
| --- | --- |
| video VAE | spatial ratio 16, temporal ratio 4; `17n + 5` frames → `5n + 2` latent frames; 24 latent channels |
| transformer patch | `(1, 2, 2)`: one row per 2×2 latent pixels per latent frame |
| audio | 32 kHz, stereo as two mono batch items, hop 800 → 40 latents/s, 32 latent channels |
| default canvas | 1344×768 (16:9 preferred, area-capped at 768·1344, rounded to multiples of 32) |
| image reference | its own 2048 px short edge, no area cap |
| 5 s target (124 frames) | 37 latent frames, 207 audio latents per channel |

Each visual reference adds **two** blocks of rows: its Qwen vision tokens in the text, and its VAE latent rows in the
packed sequence. A review counted one 2048×2048 image with the prompt "A person smiles." at the default
canvas and 124 frames: **45,914 rows**. That is 37,296 target video, 414 target audio, 4,096 image VAE rows and 4,108
text/vision rows. A 480p (864×480) target has 14,985 video rows instead of 37,296.

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

The port is the same algorithm in numpy (float64 positions, int64 indices). It reads each reference's geometry from a
shape rather than a tensor, so it needs no torch:

📍 [`src/h3mlx/packing.py:112-192`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/packing.py#L112-L192)
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

**Summation order.** Two clock advances add up the series `5/3 · (1, 4, 4, 4, 4)`:

- **A video reference's span:** both sides call Python's built-in `sum()`.
- **The fl2va "last" anchor:** both sides use numpy's pairwise `.sum()`.

Since Python 3.12 the built-in `sum()` of floats is *compensated* (Neumaier), not plain left-to-right addition. Of the
term counts tried (15, 16, 21, 26, 31, 37), on Python 3.13 it differs from a plain loop at all but 26; on 3.11 it equals
the loop at all six.
The port calls the same `sum()`, so it matches diffusers on the same interpreter.

These are float64 positions, and the rotary embedding casts them to float32 first. In a probe of 47,200 target clocks
(2 to 119 reference frames × 0 to 399 text rows), none differed after that cast.

`tests/test_packing.py` compares positions, tags, indices and counts bit for bit:

- 16 t2va/fl2va layouts: 4 anchor choices × 4 latent geometries;
- 9 ref2va mixes;
- two 26-frame videos whose own span decides the clock, one silent and one with a short soundtrack. Both fail if the
  span is summed pairwise instead.

A 26-frame video with a *long* soundtrack would not catch that: `max(audio, video)` picks the audio.

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
📍 [`src/h3mlx/ops.py:227-232`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L227-L232)
```python
 227 def temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
 228     spans = np.array(
 229         [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
 230         dtype=np.float64,
 231     )
 232     return origin + np.concatenate([np.zeros(1), np.cumsum(spans[:-1])])
```

---

## 2. The transformer

### 2.1 Rotary embedding

One `inv_freq` vector serves all three axes. The `(t, h, w)` angle blocks are concatenated and doubled, so the leading
`2 · 3 · 16 = 96` of each head's 128 channels rotate and the other 32 pass through unchanged.

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

📍 [`src/h3mlx/ops.py:18-21`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L18-L21)
```python
  18 def rope_inv_freq(rope_freq_dim: int = 16, rope_theta: float = 10000.0) -> mx.array:
  19     """Built in numpy float32: bit-identical to torch's `theta ** (arange / 2f)`; MLX's pow is 1 ulp off at f=16."""
  20     exps = np.arange(0, 2 * rope_freq_dim, 2, dtype=np.float32) / np.float32(2 * rope_freq_dim)
  21     return mx.array((np.float32(1) / np.power(np.float32(rope_theta), exps)).astype(np.float32))
```
📍 [`src/h3mlx/ops.py:24-30`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L24-L30)
```python
  24 def rope_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
  25     """(seq, 3) positions -> cos, sin of shape (seq, 2 * 3 * rope_freq_dim), float32."""
  26     pos = position_ids.astype(mx.float32)
  27     freqs = pos[:, :, None] * inv_freq[None, None, :]  # (seq, 3, f)
  28     freqs = freqs.reshape(pos.shape[0], -1)  # t block, h block, w block
  29     freqs = mx.concatenate([freqs, freqs], axis=-1)
  30     return mx.cos(freqs), mx.sin(freqs)
```
📍 [`src/h3mlx/ops.py:33-41`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L33-L41)
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

`inv_freq` is built in numpy float32, which matches torch bit for bit. MLX's `pow` is one ulp off at two of the 16
frequencies.

### 2.2 The AdaLN table

Each block projects the shared timestep embedding into six modulations **per (timestep, modality) pair**. A row picks
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
📍 [`src/h3mlx/ops.py:80-88`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L80-L88)
```python
  80 def adaln_modulation(temb: mx.array, w: mx.array, b: mx.array, hidden_size: int) -> list[mx.array]:
  81     """MiniMaxH3AdaLayerNormModulation: (num_t, temb_dim) -> 6 x (num_t * 3, hidden), rows [t0m0, t0m1, t0m2, t1m0..].
  82
  83     Order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp. silu at temb's precision, then cast to
  84     the projection's dtype.
  85     """
  86     out = linear(silu(temb).astype(w.dtype), w, b)
  87     out = out.reshape(-1, 6 * hidden_size)
  88     return mx.split(out, 6, axis=-1)
```
📍 [`src/h3mlx/ops.py:91-93`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L91-L93)
```python
  91 def modulate(x: mx.array, shift: mx.array, scale: mx.array, rows: mx.array) -> mx.array:
  92     """x * (1 + scale[rows]) + shift[rows]; x is (batch, seq, hidden), rows (seq,) index the AdaLN table."""
  93     return x * (1.0 + mx.take(scale, rows, axis=0)) + mx.take(shift, rows, axis=0)
```

The `silu` here runs on the f32 timestep embedding, and only its result is cast to the bf16 projection, as the
original's comment explains. The place where a bf16 `silu` could round differently is the feed-forward gate. There our
`silu` computes in f32 and rounds once, as torch's does. Before this, a review found a one-ulp difference at 304 of 1,001
bf16 values in [−5, 5]. (`plinear(x, p, "net.0.proj")` below is exactly `linear(x, p["net.0.proj.weight"])` unless the
weight dict also carries a LoRA for that name; only HyperFlow does, see §9.)

📍 [`src/h3mlx/ops.py:148-152`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L148-L152)
```python
 148 def swiglu_ff(x: mx.array, p: dict) -> mx.array:
 149     """diffusers FeedForward(activation_fn="swiglu", bias=False): proj -> (h, gate) -> h * silu(gate) -> out."""
 150     h = plinear(x, p, "net.0.proj")
 151     h, gate = mx.split(h, 2, axis=-1)
 152     return plinear(h * silu(gate), p, "net.2")
```
📍 [`src/h3mlx/ops.py:54-58`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L54-L58)
```python
  54 def silu(x: mx.array) -> mx.array:
  55     """torch's silu rounds once: compute in float32, cast back (a bf16 x * sigmoid(x) is off by one ulp at ~30% of
  56     inputs)."""
  57     xf = x.astype(mx.float32)
  58     return (xf * mx.sigmoid(xf)).astype(x.dtype)
```

### 2.3 Attention

Separate `to_q`/`to_k`/`to_v` with no bias, a per-head RMSNorm on q and k, rope, scaled-dot-product attention, and
`to_out`. The attention width, `heads · head_dim = 7168`, is wider than the residual stream (5376). The original also
supports a fused QKV projection; the release uses separate ones, and so does the port. `plinear` is §2.2's
LoRA-aware `linear`.

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
📍 [`src/h3mlx/ops.py:126-145`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L126-L145)
```python
 126 def attn_block(
 127     x: mx.array,
 128     p: dict,
 129     heads: int,
 130     qk_eps: float,
 131     rotary: tuple[mx.array, mx.array] | None,
 132     mask: mx.array | None = None,
 133 ) -> mx.array:
 134     """MiniMaxH3AttnProcessor with separate to_q/to_k/to_v (no bias), per-head RMSNorm on q/k, rope, to_out."""
 135     b, s, _ = x.shape
 136     q = plinear(x, p, "to_q").reshape(b, s, heads, -1)
 137     k = plinear(x, p, "to_k").reshape(b, s, heads, -1)
 138     v = plinear(x, p, "to_v").reshape(b, s, heads, -1)
 139     q = rms_norm(q, p["norm_q.weight"], qk_eps)
 140     k = rms_norm(k, p["norm_k.weight"], qk_eps)
 141     if rotary is not None:
 142         q = apply_rotary(q, *rotary)
 143         k = apply_rotary(k, *rotary)
 144     o = attention(q, k, v, mask).reshape(b, s, -1)
 145     return plinear(o, p, "to_out.0")
```
📍 [`src/h3mlx/ops.py:108-116`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L108-L116)
```python
 108 def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
 109     """Full self-attention. q, k, v: (batch, seq, heads, head_dim) as diffusers lays them out; returns the same layout.
 110
 111     `mask` is an additive or boolean mask broadcastable to (batch, heads, seq_q, seq_k); the model itself passes none.
 112     """
 113     scale = 1.0 / math.sqrt(q.shape[-1])
 114     qt, kt, vt = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
 115     o = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale, mask=mask)
 116     return o.transpose(0, 2, 1, 3)
```

The original packs one request into one unpadded document, so no mask is passed. diffusers' comment says MiniMax's own
reference pads to a multiple of 64 and masks the tail. `tests/test_ops.py::test_attention_padded_to_64_with_mask_equals_unpadded`
shows the two give the same rows in MLX, with a junk tail that does change the output when unmasked.

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
📍 [`src/h3mlx/ops.py:290-311`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L290-L311)
```python
 290 def dit_block(
 291     x: mx.array,
 292     temb: mx.array,
 293     adaln_indices: mx.array,
 294     rotary: tuple[mx.array, mx.array],
 295     p: dict,
 296     heads: int,
 297     hidden_size: int,
 298     norm_eps: float = 1e-5,
 299     qk_eps: float = 1e-5,
 300     modulation: list[mx.array] | None = None,
 301 ) -> mx.array:
 302     """Pre-norm attention and SwiGLU FF, each shift/scale-modulated and gated per row from the (t, modality) table.
 303     `modulation`: the six precomputed tables (from `ModulationCache`) instead of projecting `temb` here."""
 304     if modulation is None:
 305         modulation = adaln_modulation(temb, p["adaln_proj.linear.weight"], p["adaln_proj.linear.bias"], hidden_size)
 306     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
 307     sub = lambda prefix: {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}  # noqa: E731
 308     h = modulate(rms_norm(x, p["norm1.weight"], norm_eps), shift_msa, scale_msa, adaln_indices)
 309     x = x + mx.take(gate_msa, adaln_indices, axis=0) * attn_block(h, sub("attn."), heads, qk_eps, rotary)
 310     h = modulate(rms_norm(x, p["norm2.weight"], norm_eps), shift_mlp, scale_mlp, adaln_indices)
 311     return x + mx.take(gate_mlp, adaln_indices, axis=0) * swiglu_ff(h, sub("ff."))
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

The port, stage by stage. Each modality is projected at its own weights' dtype, the text goes through the 2-layer token
refiner, and the rows are scattered into the packed buffer:

📍 [`src/h3mlx/ops.py:357-365`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L357-L365)
```python
 357     v = linear(video.astype(p["proj_in.weight"].dtype), p["proj_in.weight"], p["proj_in.bias"])
 358     a = linear(audio.astype(p["audio_proj_in.weight"].dtype), p["audio_proj_in.weight"], p["audio_proj_in.bias"])
 359     t = linear(text.astype(p["context_embedder.weight"].dtype), p["context_embedder.weight"], p["context_embedder.bias"])
 360     t = token_refiner(t, _sub(p, "token_refiner."), cfg["num_refiner_layers"], heads, eps, qk_eps, final_eps)
 361
 362     x = mx.zeros((t.shape[0], seq, t.shape[-1]), dtype=t.dtype)
 363     x[:, text_indices] = t
 364     x[:, video_indices] = v.astype(t.dtype)
 365     x[:, audio_indices] = a.astype(t.dtype)
```

The AdaLN rows, then the 50 blocks. Each block is evaluated before the next is built, so MLX's lazy graph never spans
more than one block. This does not cap the sequence length or the total memory. Without HyperFlow, `endpoint` is
`None` and `two_time_embedding` is exactly `timestep_embedding` (time_proj, then time_embedder); with it, it is
HyperFlow's two-time embedder (§9 and `docs/porting-hyperflow.md` §3):

📍 [`src/h3mlx/ops.py:369-382`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L369-L382)
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

The final shift/scale norm, the cast to the f32 heads, and gathering the video and audio rows:

📍 [`src/h3mlx/ops.py:386-395`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L386-L395)
```python
 386     if cache is None:
 387         x = norm_out(x, temb, p["norm_out.norm.weight"], p["norm_out.linear.weight"], p["norm_out.linear.bias"],
 388                      timestep_indices, final_eps)
 389     else:
 390         shift, scale = cache.out
 391         h = rms_norm(x, p["norm_out.norm.weight"], final_eps)
 392         x = h * (1.0 + mx.take(scale, t_rows, axis=0)) + mx.take(shift, t_rows, axis=0)
 393     x = x.astype(p["proj_out.weight"].dtype)
 394     vid = linear(x, p["proj_out.weight"], p["proj_out.bias"])[:, video_indices]
 395     aud = linear(x, p["audio_proj_out.weight"], p["audio_proj_out.bias"])[:, audio_indices]
```

The port leaves out the original's input shape checks, the LoRA scale hook and the output class. It is a forward
function, not a drop-in replacement for the module.

**Tests.**

- `tests/test_dit_tiny.py` runs both forwards on a tiny random config with shuffled modalities and three noise levels.
- `tests/test_dit_mixed_precision.py` uses the release's precision split on a 4-layer, width-48 random model. It scores
  each bf16 run against one f32 torch run and requires MLX's relative error to be at most 1.5 × torch's + 0.001.
  Observed: video 0.61% (torch 0.55%), audio 0.61% (torch 0.68%).

### 2.6 Ours only: the modulation cache (opt-in, not used by `ref2va`)

The modulations depend only on the timestep, never on the latents. A run can therefore compute every block's tables
once for every timestep it will use, and drop the `adaln_proj` weights.

**What that saves.** The dropped projections hold 26.08 GB of `transformer_ref`'s 66.28 GB of bf16 parameters. The
tables cost about 1 GB at the real size for a 50-point schedule: 98 distinct timesteps with image references only, 99
with an audio reference. These are parameter bytes, not a measured peak.

**How to use it.**

1. Build the cache for every timestep the run will see: `ModulationCache(params, cfg, schedule_timesteps(layout,
   schedules))`.
2. Pass `cache=` to `dit_forward` or `denoise`.
3. Pass `ModulationCache.drop_weights(params, cfg)` as the params. This returns a **new** dict; the memory is freed only
   when nothing else, such as the original dict, still holds those arrays.

`pipeline.ref2va` does not do this. The runs on the M3 Ultra (256 GB) were made without it; the README gives their
times and the denoise peak (70-88 GB).

**Accuracy.** The maths is the same, but the floating-point result can differ. The cache multiplies all timesteps in
one batch; the uncached path multiplies only that step's. A review found a larger mixed-precision model
where they differ by 0.0028 at one step of 49. `tests/test_modulation_cache.py` checks two things:

- **On small fixtures:** cached and uncached are exactly equal.
- **On that larger model at each of the schedule's 49 timestep plans:** one forward per plan on fixed inputs (not an
  accumulated trajectory). Cached and uncached must both be within 1.25× torch-bf16's own error against an f32 run,
  plus 0.0001. The worst observed is 1.06×.

📍 [`src/h3mlx/ops.py:403-444`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L403-L444)
```python
 403 class ModulationCache:
 404     """Every block's six AdaLN tables (and norm_out's shift/scale) for a fixed set of timestep values, computed once.
 405
 406     The modulation depends only on t and the weights, never on the latents, so a sampling run can precompute it for
 407     the union of its timesteps and drop the `adaln_proj` weights (~26 GB of transformer_ref's ~66 GB in bf16). The
 408     rows of each table are [t0m0, t0m1, t0m2, t1m0, ...] over `self.t`, exactly what `adaln_modulation` returns.
 409     """
 410
 411     def __init__(self, p: dict, cfg: dict, timesteps):
 412         if "endpoint_time_embedder.linear_1.weight" in p:
 413             raise NotImplementedError("ModulationCache is keyed by t alone; HyperFlow's tables depend on (t, r)")
 414         t = np.unique(np.asarray(timesteps, dtype=np.float32))
 415         self.t = t
 416         self._index = {float(v): i for i, v in enumerate(t)}
 417         temb = timestep_embedding(mx.array(t), _sub(p, "time_embedder."), cfg["freq_dim"])
 418         hidden = cfg["hidden_size"]
 419         self.blocks = []
 420         for i in range(cfg["num_layers"]):
 421             b = f"transformer_blocks.{i}.adaln_proj.linear."
 422             tables = adaln_modulation(temb, p[b + "weight"], p[b + "bias"], hidden)
 423             mx.eval(tables)
 424             self.blocks.append(tables)
 425         w, bias = p["norm_out.linear.weight"], p["norm_out.linear.bias"]
 426         self.out = mx.split(linear(silu(temb).astype(w.dtype), w, bias), 2, axis=-1)
 427         mx.eval(self.out)
 428
 429     def rows(self, timestep: mx.array, timestep_indices: mx.array) -> mx.array:
 430         """Map a step's (distinct timesteps, per-row index into them) onto rows of the cached tables."""
 431         local = np.array(timestep.astype(mx.float32))
 432         try:
 433             glob = np.array([self._index[float(v)] for v in local], dtype=np.int64)
 434         except KeyError as e:
 435             raise KeyError(f"timestep {e} not in the cache (built for {len(self.t)} values)") from None
 436         return mx.take(mx.array(glob), timestep_indices)
 437
 438     @staticmethod
 439     def drop_weights(p: dict, cfg: dict) -> dict:
 440         """A new dict without the projections the cache replaces. The memory is freed only when nothing else (e.g. the
 441         caller's original dict) still references those arrays."""
 442         gone = {f"transformer_blocks.{i}.adaln_proj.linear.{n}" for i in range(cfg["num_layers"]) for n in ("weight", "bias")}
 443         gone |= {"norm_out.linear.weight", "norm_out.linear.bias"}
 444         return {k: v for k, v in p.items() if k not in gone}
```
📍 [`src/h3mlx/denoise.py:149-157`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/denoise.py#L149-L157)
```python
 149 def schedule_timesteps(layout: Layout, schedules: Schedules) -> np.ndarray:
 150     """Every distinct timestep any row takes over the whole run: what a ModulationCache needs."""
 151     if schedules.two_time:
 152         raise NotImplementedError("ModulationCache is keyed by t alone; HyperFlow's tables depend on (t, r)")
 153     vals = []
 154     for t, ta in zip(schedules.video_timesteps, schedules.audio_timesteps):
 155         uniq, _ = row_timesteps(layout, float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0)
 156         vals.append(uniq)
 157     return np.unique(np.concatenate(vals))
```

---

## 3. The scheduler

Rectified flow with an exponential shift: 12 for video, 3 for audio. The model predicts a **data-ward** velocity
(`x0 = x + (1 − t)·v`, note the `+`), and the Euler update is a blend computed in f32.

**Counting steps.** `num_inference_steps = N` means N sigma grid points including the final 0, so **N − 1 model
calls**. diffusers' default of 50 runs 49 forwards. The tests' "3" and "5" run 2 and 4.

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
📍 [`src/h3mlx/ops.py:263-273`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L263-L273)
```python
 263 def flow_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
 264     """linspace(1, 0, n) in float32, exponential shift, consecutive duplicates collapsed. Terminal 0 included."""
 265     if num_inference_steps is None or num_inference_steps < 2:
 266         raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
 267     if shift <= 0:
 268         raise ValueError(f"shift must be positive, got {shift}")
 269     base = torch_linspace_f32(1.0, 0.0, num_inference_steps)
 270     shift32 = np.float32(shift)
 271     sig = (shift32 * base / (np.float32(1) + (shift32 - np.float32(1)) * base)).astype(np.float32)
 272     keep = np.concatenate([[True], sig[1:] != sig[:-1]])
 273     return sig[keep]
```

On arm64, `torch.linspace` fills each half of the grid from its own end, with a fused multiply-add. So `np.linspace`
differs in the last bit at nine positions for N = 50. The port reproduces torch's fill; `tests/test_ops.py` checks it
bit-exact for every N from 2 to 1999.

📍 [`src/h3mlx/ops.py:246-260`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L246-L260)
```python
 246 def torch_linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
 247     """torch.linspace(start, end, steps, dtype=float32) bit for bit (CPU, arm64): float32 step, the first half filled
 248     from `start` and the second from `end`, each as one fused multiply-add (exact product, single rounding).
 249     np.linspace differs by 1 ulp at e.g. steps=50."""
 250     if steps < 2:
 251         raise ValueError(f"steps must be >= 2, got {steps}")
 252     start32, end32 = np.float32(start), np.float32(end)
 253     step = np.float32((end32 - start32) / np.float32(steps - 1))
 254     i = np.arange(steps, dtype=np.float64)
 255     out = np.where(
 256         i < steps // 2,
 257         np.float64(start32) + np.float64(step) * i,
 258         np.float64(end32) - np.float64(step) * (steps - 1 - i),
 259     )
 260     return out.astype(np.float32)
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
📍 [`src/h3mlx/ops.py:276-284`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/ops.py#L276-L284)
```python
 276 def flow_step(v: mx.array, t: float, x: mx.array, sigma: float, sigma_next: float) -> mx.array:
 277     """x0 = x + (1 - t) * v (sigma from the timestep, in x's dtype); x_next = r*x + (1 - r)*x0, r = s_next / s, f32."""
 278     st = (mx.array(1.0, dtype=x.dtype) - mx.array(t, dtype=x.dtype)).astype(x.dtype)
 279     x0 = x + st * v
 280     cdt = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
 281     s = mx.array(sigma, dtype=cdt)
 282     sn = mx.array(sigma_next, dtype=cdt)
 283     r = sn / s
 284     return (r * x.astype(cdt) + (1.0 - r) * x0.astype(cdt)).astype(x.dtype)
```

The original's scheduler also accepts custom sigma arrays and keeps a step index; the port's `flow_step` takes the two
sigmas explicitly.

---

## 4. The denoise loop

One forward per step serves every row:

- **generated video and audio rows** step down their own schedules;
- **visual reference rows** stay at `max(t, 0.999)`;
- **audio reference rows** stay at 1.0;
- **text rows** take the video's timestep.

The checkpoint is guidance-distilled, so there is no unconditional pass.

diffusers builds the per-row plan once, then each loop step runs the transformer and the two scheduler updates:

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
📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/before_denoise.py:1238-1269`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L1238-L1269)
```python
1238     @torch.no_grad()
1239     def __call__(
1240         self, components: MiniMaxH3ModularPipeline, state: PipelineState
1241     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
1242         block_state = self.get_block_state(state)
1243         device = components._execution_device
1244
1245         components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
1246         components.audio_scheduler.set_timesteps(block_state.num_inference_steps, device=device)
1247         block_state.timesteps = components.scheduler.timesteps
1248         block_state.audio_timesteps = components.audio_scheduler.timesteps
1249
1250         block_state.row_timestep_plan = [
1251             tuple(
1252                 tensor.to(device)
1253                 for tensor in self.build_row_timesteps(
1254                     block_state.video_indices,
1255                     block_state.audio_indices,
1256                     block_state.num_condition_video_rows,
1257                     block_state.num_condition_audio_rows,
1258                     block_state.text_indices.numel(),
1259                     float(timestep),
1260                     float(audio_timestep),
1261                     max(float(timestep), components.keyframe_noise_aug),
1262                     1.0,
1263                 )
1264             )
1265             for timestep, audio_timestep in zip(block_state.timesteps, block_state.audio_timesteps)
1266         ]
1267
1268         self.set_block_state(state, block_state)
1269         return components, state
```
📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/denoise.py:110-133`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/denoise.py#L110-L133)
```python
 110     @torch.no_grad()
 111     def __call__(
 112         self, components: MiniMaxH3ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
 113     ) -> tuple[MiniMaxH3ModularPipeline, BlockState]:
 114         transformer = getattr(components, self.transformer_name)
 115         unique_timesteps, timestep_indices = block_state.row_timestep_plan[i]
 116         # The layout tags its outputs `denoiser_input_fields`, and their names are the transformer's own parameter
 117         # names, so the rows of the packed sequence are described to it without this block enumerating them.
 118         layout_kwargs = {
 119             name: value
 120             for name, value in block_state.denoiser_input_fields.items()
 121             if name in inspect.signature(transformer.forward).parameters
 122         }
 123         block_state.noise_pred, block_state.audio_noise_pred = transformer(
 124             hidden_states=block_state.latents[None],
 125             audio_hidden_states=block_state.audio_latents[None],
 126             encoder_hidden_states=block_state.prompt_embeds,
 127             timestep=unique_timesteps,
 128             timestep_indices=timestep_indices,
 129             attention_kwargs=block_state.attention_kwargs,
 130             return_dict=False,
 131             **layout_kwargs,
 132         )
 133         return components, block_state
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

The port does all three in one loop:

📍 [`src/h3mlx/denoise.py:69-77`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/denoise.py#L69-L77)
```python
  69 def row_timesteps(layout: Layout, video_t: float, audio_t: float, cond_video_t: float, cond_audio_t: float):
  70     """Per-row timesteps reduced to (sorted distinct values, inverse index), like `build_row_timesteps`."""
  71     seq = len(layout.video_indices) + len(layout.audio_indices) + len(layout.text_indices)
  72     rt = np.full(seq, np.float32(video_t), dtype=np.float32)
  73     rt[layout.video_indices[: layout.num_condition_video_rows]] = cond_video_t
  74     rt[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_t
  75     rt[layout.audio_indices[: layout.num_condition_audio_rows]] = cond_audio_t
  76     uniq, inv = np.unique(rt, return_inverse=True)
  77     return uniq, inv
```
📍 [`src/h3mlx/denoise.py:97-146`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/denoise.py#L97-L146)
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

`tests/test_denoise.py` runs the port against a torch loop assembled from diffusers' own scheduler, row plan and
transformer, for 3 and 5 grid points. The reference rows come back untouched.

---

## 5. The text side: the Qwen3-VL presentation

MiniMax conditions on hidden state **50** of a 64-layer Qwen3-VL. That state comes before the final norm; the last
layer's output would be the wrong one.

**The prompt.** Every reference gets a label, numbered per modality:

- an image: `<Picture i>: ` plus an image block;
- audio: `<Audio j>: ` alone;
- a video: `<Video k>: ` plus one timestamped block per merged frame pair.

The prompt follows verbatim, with no chat template.

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
📍 [`src/h3mlx/presentation.py:30-74`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/presentation.py#L30-L74)
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
📍 [`src/h3mlx/presentation.py:12-27`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/presentation.py#L12-L27)
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

The image and video pixels go through the released Qwen3-VL processor, batched per modality, which also gives the
token count per block:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/encoders.py:491-541`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L491-L541)
```python
 491     def _gather_vision_features(
 492         self, processor, references: list[MiniMaxH3Reference], fps: float
 493     ) -> tuple[dict, list[int], list[int], list[list[float]]]:
 494         r"""
 495         Run the references' pixels through the conditioner's processors, batched per modality.
 496
 497         The vision tensors are batched per modality while the presentation is tokenized in request order; the two agree
 498         because the filtering here preserves relative order within each modality and Qwen3-VL fills the n-th pad *run*
 499         of a modality with the n-th entry of that modality's batch. Audio contributes nothing — a waveform never
 500         reaches the conditioner.
 501
 502         Returns:
 503             `tuple`: the vision tensors by the conditioner's parameter names, the vision token count per image
 504             reference, the per-block token count per video reference, and the block timestamps per video reference.
 505         """
 506         merge_size = processor.image_processor.merge_size**2
 507         vision_inputs = {}
 508
 509         image_token_counts = []
 510         images = [reference.image for reference in references if reference.kind == "image"]
 511         if images:
 512             image_features = processor.image_processor(images=images, return_tensors="pt")
 513             vision_inputs["pixel_values"] = image_features["pixel_values"]
 514             vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
 515             image_token_counts = [int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]]
 516
 517         video_block_token_counts, video_block_timestamps = [], []
 518         videos = [reference for reference in references if reference.kind == "video"]
 519         if videos:
 520             temporal_patch = processor.video_processor.temporal_patch_size
 521             sampled = [
 522                 self._sample_video_condition_frames(reference.frames, fps, self.video_sample_fps, temporal_patch)
 523                 for reference in videos
 524             ]
 525             video_block_timestamps = [timestamps for _, timestamps in sampled]
 526             video_features = processor.video_processor(
 527                 videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
 528             )
 529             vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
 530             vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
 531             video_block_token_counts = [
 532                 int(grid[1]) * int(grid[2]) // merge_size for grid in video_features["video_grid_thw"]
 533             ]
 534             for timestamps, grid in zip(video_block_timestamps, video_features["video_grid_thw"]):
 535                 if int(grid[0]) != len(timestamps):
 536                     raise ValueError(
 537                         f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
 538                         f"labels {len(timestamps)} of them."
 539                     )
 540
 541         return vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps
```
📍 [`src/h3mlx/pipeline.py:155-178`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/pipeline.py#L155-L178)
```python
 155 def vision_features(c: Components, refs):
 156     proc = c.processor
 157     merge = proc.image_processor.merge_size**2
 158     vision, img_counts, vid_counts, vid_ts = {}, [], [], []
 159     images = [r.image for r in refs if r.kind == "image"]
 160     if images:
 161         f = proc.image_processor(images=images, return_tensors="np")
 162         vision["pixel_values"], vision["image_grid_thw"] = f["pixel_values"], f["image_grid_thw"]
 163         img_counts = [int(np.prod(g)) // merge for g in f["image_grid_thw"]]
 164     videos = [r for r in refs if r.kind == "video"]
 165     if videos:
 166         tp = proc.video_processor.temporal_patch_size
 167         sampled = []
 168         for r in videos:
 169             idx, ts = sample_video_condition_frames(r.frames.shape[0], float(R.FPS), c.video_sample_fps, tp)
 170             sampled.append(np.stack([r.frames[i] for i in idx]))
 171             vid_ts.append(ts)
 172         f = proc.video_processor(videos=sampled, do_sample_frames=False, return_tensors="np")
 173         vision["pixel_values_videos"], vision["video_grid_thw"] = f["pixel_values_videos"], f["video_grid_thw"]
 174         vid_counts = [int(g[1]) * int(g[2]) // merge for g in f["video_grid_thw"]]
 175         for ts, g in zip(vid_ts, f["video_grid_thw"]):
 176             if int(g[0]) != len(ts):
 177                 raise ValueError(f"processor made {int(g[0])} video blocks, presentation labels {len(ts)}")
 178     return vision, img_counts, vid_counts, vid_ts
```

**Running the encoder.** diffusers runs transformers' Qwen3-VL to the end and reads `hidden_states[50]`. The port asks
mlx-vlm's model to stop after layer 50 without the final norm, through the `encode` function the loader returns:

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
📍 [`src/h3mlx/load.py:133-172`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/load.py#L133-L172)
```python
 133 def load_text_encoder(d, keep_layers: int = 50, dtype=None):
 134     """mlx-vlm Qwen3-VL from the release's `text_encoder/`, with only decoder layers [0, keep_layers) and no lm_head.
 135     Returns (model, encode(token_ids, vision_inputs) -> (1, n, hidden) after `keep_layers` layers, pre-norm)."""
 136     from mlx.utils import tree_flatten
 137     from mlx_vlm.models.qwen3_vl import Model, ModelConfig
 138     from mlx_vlm.models.qwen3_vl.vision import VisionModel
 139     from mlx_vlm.utils import sanitize_weights
 140
 141     config = ModelConfig.from_dict(qwen_params(_config(d)))
 142     n_layers = config.text_config.num_hidden_layers
 143     if not 0 < keep_layers <= n_layers:
 144         raise ValueError(f"keep_layers must be in (0, {n_layers}], got {keep_layers}")
 145
 146     config.text_config.num_hidden_layers = keep_layers  # never build the layers past the cut
 147     model = Model(config)
 148     model.language_model.pop("lm_head", None)  # unused; would be a random 5120 x 151936 matrix
 149     layer_re = re.compile(r"language_model\.layers\.(\d+)\.")
 150
 151     def keep(k):
 152         if "lm_head" in k:
 153             return False
 154         m = layer_re.search(k)
 155         return m is None or int(m.group(1)) < keep_layers
 156
 157     w = load_safetensors(d, keep)
 158     w = sanitize_weights(VisionModel, sanitize_weights(model, w), config.vision_config)
 159     if dtype is not None:
 160         w = {k: v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v for k, v in w.items()}
 161     model.load_weights(list(w.items()), strict=False)
 162     want = {k for k, _ in tree_flatten(model.parameters()) if "lm_head" not in k}
 163     missing = want - set(w)
 164     if missing:
 165         raise ValueError(f"text encoder weights missing: {sorted(missing)[:6]} (+{max(0, len(missing) - 6)})")
 166     eval_each_layer(model.language_model.model.layers)
 167
 168     def encode(token_ids, vision_inputs):
 169         kw = {k: mx.array(np.asarray(v)) for k, v in (vision_inputs or {}).items()}
 170         return model.hidden_state_at_layer(mx.array([token_ids]), keep_layers, **kw)
 171
 172     return model, encode
```

The loader builds only 50 decoder layers and no LM head. It opens every shard with `mx.load`, which is lazy: it reads
each file's header and creates arrays without reading their data. Keys past the cut are then dropped before anything
is evaluated, so their data is never loaded. A 1 GB test file loaded in 0.1 ms; that is a probe, not an I/O guarantee
for every shard.

`staged_text_encoder` loads, encodes and drops its own references. It does not unload anything else. To keep the text
encoder and the transformer from being resident together, leave the transformer unevaluated until the text is done:

📍 [`src/h3mlx/load.py:241-255`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/load.py#L241-L255)
```python
 241 def staged_text_encoder(d, keep_layers: int = 50, dtype=None):
 242     """An `encode(token_ids, vision_inputs)` that loads the text encoder, runs it, and drops its own references before
 243     returning (~51 GB bf16 for 50 layers of the release). It does not unload anything else: keep the transformer
 244     unevaluated (lazy `mx.load`) until text encoding is done if the two must not be resident together."""
 245
 246     def encode(token_ids, vision_inputs):
 247         model, run = load_text_encoder(d, keep_layers, dtype)
 248         out = run(token_ids, vision_inputs)
 249         mx.eval(out)
 250         del model, run
 251         gc.collect()  # a reference cycle would otherwise keep ~51 GB alive into the denoise
 252         mx.clear_cache()
 253         return out
 254
 255     return encode
```

**Tests.**

- `tests/test_presentation.py`: token ids and tags bit-exact with the released tokenizer.
- `tests/test_text_encoder_tiny.py`: mlx-vlm against transformers on a tiny Qwen3-VL that keeps interleaved mrope and
  deepstack, with a picture and a video in one presentation. Without `video_grid_thw`, mlx-vlm 0.7.11 raises rather
  than misplace the video tokens.

---

## 6. The video VAE

**Encoder:** a causal 3-D CNN. Reflect padding in space, zero frames prepended in time, group norm per frame.
**Decoder:** a non-causal ViT. Both are tiled spatially (below).

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
📍 [`src/h3mlx/vae_ops.py:25-41`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae_ops.py#L25-L41)
```python
  25 def causal_conv3d(
  26     x: mx.array,
  27     w: mx.array,
  28     b: mx.array | None,
  29     stride: tuple[int, int, int] = (1, 1, 1),
  30     spatial_padding: int = 0,
  31     temporal_padding: int = 0,
  32 ) -> mx.array:
  33     """MiniMaxH3VideoCausalConv3d: symmetric reflect pad on H, W; `temporal_padding` zero frames prepended only.
  34     x: (B, F, H, W, C); w: MLX layout (O, kt, kh, kw, I)."""
  35     if spatial_padding > 0:
  36         x = reflect_pad(x, 2, spatial_padding, spatial_padding)
  37         x = reflect_pad(x, 3, spatial_padding, spatial_padding)
  38     if temporal_padding > 0:
  39         x = mx.pad(x, [(0, 0), (temporal_padding, 0), (0, 0), (0, 0), (0, 0)])
  40     y = mx.conv3d(x, w, stride=stride, padding=0)
  41     return y + b if b is not None else y
```
📍 [`src/h3mlx/vae_ops.py:15-22`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae_ops.py#L15-L22)
```python
  15 def reflect_pad(x: mx.array, axis: int, before: int, after: int) -> mx.array:
  16     """torch F.pad(mode="reflect") along one axis (edge sample not repeated)."""
  17     n = x.shape[axis]
  18     if before >= n or after >= n:
  19         raise ValueError(f"reflect pad ({before}, {after}) needs size > pad, got {n}")
  20     widths = [(0, 0)] * x.ndim
  21     widths[axis] = (before, after)
  22     return mx.pad(x, widths, mode="reflect")
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
📍 [`src/h3mlx/vae_ops.py:44-52`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae_ops.py#L44-L52)
```python
  44 def group_norm_per_frame(x: mx.array, groups: int, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
  45     """MiniMaxH3VideoGroupNorm: GroupNorm with statistics per (batch, frame), never across frames. Biased variance,
  46     computed in float32."""
  47     b, f, h, w, c = x.shape
  48     xf = x.astype(mx.float32).reshape(b, f, h * w, groups, c // groups)
  49     mean = xf.mean(axis=(2, 4), keepdims=True)
  50     var = ((xf - mean) ** 2).mean(axis=(2, 4), keepdims=True)
  51     y = ((xf - mean) * mx.rsqrt(var + eps)).reshape(b, f, h, w, c).astype(x.dtype)
  52     return y * weight + bias
```

**A still goes through the spatial encoder alone.** diffusers' comment explains why: padding one image to a 17-frame
clip would run the temporal path and is "not the conditioning MiniMax-H3 was trained with". The third-party port's
public `encode` pads it.

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
📍 [`src/h3mlx/vae.py:167-182`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L167-L182)
```python
 167     def encode_moments(self, x):
 168         """Pixels (B, F, H, W, 3) -> moments (B, F', H', W', 2 * latent). One frame: spatial encoder alone."""
 169         c = self.cfg
 170         n = x.shape[1]
 171         if n == 1:
 172             return self.encode_clip(x)
 173         if c.single_frame:
 174             raise ValueError(f"this VAE encodes single frames, got {n}")
 175         if n % c.clip_length:
 176             x = mx.concatenate([x, mx.repeat(x[:, -1:], (-n) % c.clip_length, axis=1)], axis=1)
 177         m = mx.concatenate(
 178             [_evaluated(self.encode_clip(x[:, i * c.clip_length : (i + 1) * c.clip_length]))
 179              for i in range(x.shape[1] // c.clip_length)],
 180             axis=1,
 181         )
 182         return m[:, : -c.token_drop] if c.token_drop > 0 else m
```

**Tiling.** Tiles are 256 px with **at least** 64 px overlap. The overlaps are then widened in 16 px steps so the tiles
cover the frame exactly: a 2048 px edge gives 11 tiles with overlaps of 80 and 64, and a 512 px edge gives overlaps of
128. Each tile is evaluated as soon as it is built, so a 2048 px reference (121 tiles at 2048×2048, 209 at 2048×3648) never
holds all of their graphs at once. Tiles are blended linearly across each overlap and stitched:

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
📍 [`src/h3mlx/vae.py:77-90`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L77-L90)
```python
  77 def split_tiles(length: int, tile: int, min_overlap: int, ratio: int):
  78     if tile >= length:
  79         return [0], [length], []
  80     n = math.ceil(length / tile)
  81     while tile * n - min_overlap * (n - 1) - length < 0:
  82         n += 1
  83     overlaps = [min_overlap] * (n - 1)
  84     remaining = tile * n - sum(overlaps) - length
  85     for i in range(remaining // ratio):
  86         overlaps[i % (n - 1)] += ratio
  87     starts = [0]
  88     for i in range(n - 1):
  89         starts.append(starts[-1] + tile - overlaps[i])
  90     return starts, [tile] * n, overlaps
```
📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:721-744`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L721-L744)
```python
 721     def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
 722         r"""Encode one temporal clip, spatially tiled when tiling is enabled."""
 723         if not self.use_tiling:
 724             return self.quant_conv(self.encoder(x))
 725
 726         height, width = x.shape[-2], x.shape[-1]
 727         y_indices, y_lengths, y_overlaps = self._split_tiles(
 728             height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
 729         )
 730         x_indices, x_lengths, x_overlaps = self._split_tiles(
 731             width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
 732         )
 733
 734         rows = []
 735         for i_pos, i_len in zip(y_indices, y_lengths):
 736             row = []
 737             for j_pos, j_len in zip(x_indices, x_lengths):
 738                 tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
 739                 row.append(self.quant_conv(self.encoder(tile)))
 740             rows.append(row)
 741
 742         latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
 743         latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
 744         return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)
```
📍 [`src/h3mlx/vae.py:156-165`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L156-L165)
```python
 156     def encode_clip(self, x):
 157         c = self.cfg
 158         if not c.use_tiling:
 159             return self._encoder(x)
 160         ys, yl, yo = split_tiles(x.shape[2], c.tile_h, c.tile_overlap_h, c.spatial_ratio)
 161         xs, xl, xo = split_tiles(x.shape[3], c.tile_w, c.tile_overlap_w, c.spatial_ratio)
 162         rows = [[_evaluated(self._encoder(x[:, :, i : i + il, j : j + jl])) for j, jl in zip(xs, xl)]
 163                 for i, il in zip(ys, yl)]  # one tile at a time: a 2048 px reference is ~200 tiles
 164         r = c.spatial_ratio
 165         return stitch_tiles(rows, [o // r for o in yo], [o // r for o in xo])
```
📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:699-719`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L699-L719)
```python
 699     def _stitch_tiles(
 700         self,
 701         tiles: list[list[torch.Tensor]],
 702         height_overlaps: list[int],
 703         width_overlaps: list[int],
 704     ) -> torch.Tensor:
 705         result_rows = []
 706         for i, row in enumerate(tiles):
 707             result_row = []
 708             for j, tile in enumerate(row):
 709                 if i > 0:
 710                     tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
 711                 if j > 0:
 712                     tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
 713                 if i < len(tiles) - 1:
 714                     tile = tile[..., : -height_overlaps[i], :]
 715                 if j < len(row) - 1:
 716                     tile = tile[..., :, : -width_overlaps[j]]
 717                 result_row.append(tile)
 718             result_rows.append(torch.cat(result_row, dim=-1))
 719         return torch.cat(result_rows, dim=-2)
```
📍 [`src/h3mlx/vae.py:113-128`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L113-L128)
```python
 113 def stitch_tiles(tiles, h_overlaps, w_overlaps, h_axis=2, w_axis=3):
 114     rows = []
 115     for i, row in enumerate(tiles):
 116         out_row = []
 117         for j, t in enumerate(row):
 118             if i > 0:
 119                 t = blend(tiles[i - 1][j], t, h_overlaps[i - 1], h_axis)
 120             if j > 0:
 121                 t = blend(row[j - 1], t, w_overlaps[j - 1], w_axis)
 122             if i < len(tiles) - 1:
 123                 t = _slice(t, h_axis, 0, -h_overlaps[i])
 124             if j < len(row) - 1:
 125                 t = _slice(t, w_axis, 0, -w_overlaps[j])
 126             out_row.append(t)
 127         rows.append(mx.concatenate(out_row, axis=w_axis))
 128     return mx.concatenate(rows, axis=h_axis)
```
📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:679-697`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L679-L697)
```python
 679     def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
 680         blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
 681         positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
 682         shape = [1] * a.ndim
 683         shape[dim] = blend_extent
 684         weight_a = (1 - positions / blend_extent).view(shape)
 685         weight_b = (positions / blend_extent).view(shape)
 686
 687         slice_a = [slice(None)] * a.ndim
 688         slice_a[dim] = slice(-blend_extent, None)
 689         slice_b = [slice(None)] * b.ndim
 690         slice_b[dim] = slice(0, blend_extent)
 691         blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
 692
 693         if blend_extent == b.shape[dim]:
 694             return blended
 695         slice_rest = [slice(None)] * b.ndim
 696         slice_rest[dim] = slice(blend_extent, None)
 697         return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)
```
📍 [`src/h3mlx/vae.py:99-110`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L99-L110)
```python
  99 def blend(a: mx.array, b: mx.array, extent: int, axis: int) -> mx.array:
 100     """Linear cross-fade of a's tail into b's head over `extent` along `axis`; ramp built in b's dtype."""
 101     extent = min(a.shape[axis], b.shape[axis], extent)
 102     pos = mx.arange(extent).astype(b.dtype)
 103     shape = [1] * a.ndim
 104     shape[axis] = extent
 105     wa = (1 - pos / extent).reshape(shape)
 106     wb = (pos / extent).reshape(shape)
 107     out = _slice(a, axis, -extent, None) * wa + _slice(b, axis, 0, extent) * wb
 108     if extent == b.shape[axis]:
 109         return out
 110     return mx.concatenate([out, _slice(b, axis, extent, None)], axis=axis)
```

**Decoding time.** The decode is chunked in time and cross-faded where chunks overlap:

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
📍 [`src/h3mlx/vae.py:210-244`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L210-L244)
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

**The ViT decoder.** Every latent voxel is one token. Register tokens and one zero-valued token are **appended after**
the voxel tokens, with rotary coordinates `(0, 0, 0)`. Each block's norms run in f32 and its two residuals are scaled
by learned layer scales:

📍 [`diffusers:src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py:387-397`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py#L387-L397)
```python
 387     def forward(
 388         self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
 389     ) -> torch.Tensor:
 390         # The reference normalizes in float32 regardless of the compute dtype. The residual stream is float32 too
 391         # because the anchors are, so the normed activation follows the projections instead.
 392         compute_dtype = get_parameter_dtype(self.attn.to_q)
 393         norm_hidden_states = self.norm1(hidden_states.float()).to(compute_dtype)
 394         hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
 395         norm_hidden_states = self.norm2(hidden_states.float()).to(compute_dtype)
 396         hidden_states = hidden_states + self.ff(norm_hidden_states) * self.scale2
 397         return hidden_states
```
📍 [`src/h3mlx/vae_ops.py:177-185`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae_ops.py#L177-L185)
```python
 177 def vit_block(x: mx.array, p: dict, heads: int, eps: float, rotary) -> mx.array:
 178     """Norms in float32, residual stream float32, attention/FF at the projections' dtype, layer scales."""
 179     cdt = p["attn.to_q.weight"].dtype
 180     h = mx.fast.rms_norm(x.astype(mx.float32), p["norm1.weight"].astype(mx.float32), eps).astype(cdt)
 181     x = x + vit_attention(h, _sub(p, "attn."), heads, eps, rotary) * p["scale1"]
 182     h = mx.fast.rms_norm(x.astype(mx.float32), p["norm2.weight"].astype(mx.float32), eps).astype(cdt)
 183     f = _lin(h, p, "ff.net.0.proj")
 184     f, gate = mx.split(f, 2, axis=-1)
 185     return x + _lin(f * silu(gate), p, "ff.net.2") * p["scale2"]
```
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
📍 [`src/h3mlx/vae_ops.py:188-208`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae_ops.py#L188-L208)
```python
 188 def vit_decoder(
 189     z: mx.array, p: dict, num_layers: int, heads: int, patch_size: int, patch_size_t: int, out_channels: int,
 190     rope_dim: int, rope_theta: float = 100.0, eps: float = 1e-5, frame_position: float | None = None,
 191 ) -> mx.array:
 192     """MiniMaxH3VideoViTDecoder3d. z: channels-last latents (B, F, H, W, C) -> pixels (B, F*pt, H*p, W*p, out).
 193     P-H3-Image's `PH3ImageViTDecoder` is the same with `patch_size_t=1` and one frame at `frame_position`."""
 194     b, f, h, w, c = z.shape
 195     x = _lin(z.reshape(b, f * h * w, c), p, "proj_in")
 196     n = x.shape[1]
 197     regs = mx.broadcast_to(p["register_tokens"].astype(x.dtype), (b, *p["register_tokens"].shape[1:]))
 198     x = mx.concatenate([x, regs, mx.zeros((b, 1, x.shape[-1]), dtype=x.dtype)], axis=1)
 199     rotary = vit_rope(vit_positions(f, h, w, regs.shape[1] + 1, frame_position), vit_rope_inv_freq(rope_dim, rope_theta))
 200     for i in range(num_layers):
 201         x = vit_block(x, _sub(p, f"transformer_blocks.{i}."), heads, eps, rotary)
 202         mx.eval(x)  # 36 f32 blocks per tile: keep each one its own GPU graph
 203     x = mx.fast.layer_norm(x.astype(mx.float32), p["norm_out.weight"].astype(mx.float32),
 204                            p["norm_out.bias"].astype(mx.float32), eps)
 205     x = _lin(x.astype(p["proj_out.weight"].dtype), p, "proj_out")[:, :n]
 206     x = x.reshape(b, f, h, w, out_channels, patch_size_t, patch_size, patch_size)
 207     x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (b, f, pt, h, p, w, p, c)
 208     return x.reshape(b, f * patch_size_t, h * patch_size, w * patch_size, out_channels)
```

**The conditioning recipe for visual references:**

1. normalise the pixels with ImageNet's mean and standard deviation;
2. **sample** the posterior with a fresh seed-42 generator;
3. round to float16 and back;
4. normalise by the latent statistics.

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
📍 [`src/h3mlx/vae.py:253-261`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/vae.py#L253-L261)
```python
 253 def encode_condition(vae: VideoVAE, pixels_u8: mx.array, pixel_mean, pixel_std, latents_mean, latents_std, noise_fn):
 254     """`encode_vae_condition`: uint8 pixels (1, F, H, W, 3) -> ImageNet-normalised -> moments -> posterior sample
 255     (noise from `noise_fn(shape)`, which must reproduce torch.Generator().manual_seed(42) to match the release) ->
 256     float16 round trip -> (latent - mean) / std."""
 257     x = (pixels_u8.astype(mx.float32) / 255.0 - mx.array(pixel_mean)) / mx.array(pixel_std)
 258     m = vae.encode_moments(x)
 259     lat = posterior_sample(m, noise_fn(m.shape[:-1] + (m.shape[-1] // 2,)))
 260     lat = lat.astype(mx.float16).astype(mx.float32)
 261     return (lat - mx.array(latents_mean)) / mx.array(latents_std)
```

**Tests.** `tests/test_vae_ops.py` and `tests/test_vae.py` use reduced configs: spatial ratio 4, 3 levels and 32 px
tiles with 8 px overlap, chosen so tiling, the single-frame branch, chunking and decode padding all run. Two one-off
checks also ran ratio 16 with the real 256/64 tiles and a 124-frame encode at the real ratios; they are not in this
repo's suite.

---

## 7. The audio VAE

A DAC encoder and a BigVGAN decoder on mono 32 kHz audio; stereo is a batch of two. It is kept in float32: diffusers
notes that bf16 decodes come out about 20 dB quieter.

**Visual and audio references differ:** visual references use a **sampled** posterior (§6), but reference audio uses
the posterior **mean**. diffusers' reference encoder does both. It also cuts a video reference to its first
`max(1, (n − 5) // 17) · 17 + 5` frames: the longest `17k + 5` that fits for clips of 22 frames or more (31 frames
become 22, not the nearer 39). A clip under 22 frames is kept whole, since the slice can't add frames; the VAE's
17-frame chunking then pads it:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/encoders.py:715-770`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L715-L770)
```python
 715     @torch.no_grad()
 716     def __call__(
 717         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 718     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 719         block_state = self.get_block_state(state)
 720         device = components._execution_device
 721
 722         audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean).view(1, 1, -1)
 723         audio_latents_std = torch.tensor(components.audio_vae.config.latents_std).view(1, 1, -1)
 724
 725         condition_latents, audio_condition_latents = [], []
 726         for reference in block_state.normalized_references:
 727             if reference.kind == "image":
 728                 pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
 729                 condition_latents.append(
 730                     encode_vae_condition(
 731                         components.vae,
 732                         pixels,
 733                         components.pixel_mean,
 734                         components.pixel_std,
 735                         components.keyframe_encode_seed,
 736                     )
 737                 )
 738             elif reference.kind == "video":
 739                 # Snap *down* to `17 * n + 5` so the VAE encodes without padding; this only bites when the reference
 740                 # is shorter than the target, whose own frame count already has that form.
 741                 frames_per_chunk = components.vae_frames_per_chunk
 742                 latents_per_chunk = components.vae_latents_per_chunk
 743                 num_frames = reference.frames.shape[0]
 744                 num_frames = (
 745                     max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk + latents_per_chunk
 746                 )
 747                 pixels = torch.from_numpy(reference.frames[:num_frames].copy()).to(device).permute(3, 0, 1, 2)[None]
 748                 condition_latents.append(
 749                     encode_vae_condition(
 750                         components.vae,
 751                         pixels,
 752                         components.pixel_mean,
 753                         components.pixel_std,
 754                         components.keyframe_encode_seed,
 755                     )
 756                 )
 757
 758             if reference.has_audio:
 759                 posterior = components.audio_vae.encode(reference.audio.to(device)[:, None], return_dict=False)[0]
 760                 # Channel-major rows: the two stereo channels are two batch items of the mono audio VAE. Soundtracks
 761                 # take the posterior *mean*, and are never sampled.
 762                 latents = posterior.mode().float().cpu().transpose(1, 2)
 763                 normalized = (latents - audio_latents_mean) / audio_latents_std
 764                 audio_condition_latents.append(normalized.reshape(-1, components.audio_latent_channels))
 765
 766         block_state.condition_latents = condition_latents
 767         block_state.audio_condition_latents = audio_condition_latents
 768
 769         self.set_block_state(state, block_state)
 770         return components, state
```
📍 [`src/h3mlx/pipeline.py:181-205`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/pipeline.py#L181-L205)
```python
 181 def encode_references(c: Components, refs):
 182     cond, audio_rows = [], []
 183     a_mean = mx.array(c.audio_latents_mean)
 184     a_std = mx.array(c.audio_latents_std)
 185     for r in refs:
 186         if r.kind in ("image", "video"):
 187             if r.kind == "image":
 188                 px = np.asarray(r.image)[None, None]  # (1, 1, H, W, 3)
 189             else:
 190                 n = r.frames.shape[0]
 191                 n = max(1, (n - R.LATENTS_PER_CHUNK) // R.FRAMES_PER_CHUNK) * R.FRAMES_PER_CHUNK + R.LATENTS_PER_CHUNK
 192                 px = r.frames[:n][None]
 193             fresh = Noise(ENCODE_SEED)  # a fresh seed-42 generator per condition, as the release does
 194
 195             def noise_cl(shape_cl, fresh=fresh):  # torch draws (B, C, F, H, W); hand back channels-last
 196                 b, f, h, w, ch = shape_cl
 197                 return mx.array(fresh((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 198
 199             cond.append(encode_condition(c.vae, mx.array(px), PIXEL_MEAN, PIXEL_STD, c.vae_latents_mean,
 200                                          c.vae_latents_std, noise_cl))
 201         if r.has_audio:
 202             wave = mx.array(r.audio[:, :, None])  # (2, samples, 1): stereo channels as two batch items
 203             lat = c.audio_vae.encode_mean(wave)  # (2, frames, 32)
 204             audio_rows.append(((lat - a_mean) / a_std).reshape(-1, lat.shape[-1]))
 205     return cond, audio_rows
```
📍 [`src/h3mlx/audio_vae.py:202-209`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/audio_vae.py#L202-L209)
```python
 202     def encode_mean(self, wave: mx.array) -> mx.array:
 203         """(B, samples, 1) mono -> posterior mean (B, samples / hop, latent_channels); right-pads to the hop."""
 204         pad = math.ceil(wave.shape[1] / self.hop) * self.hop - wave.shape[1]
 205         if pad:
 206             wave = mx.pad(wave, [(0, 0), (0, pad), (0, 0)])
 207         h = encoder(wave, _sub(self.p, "encoder."), self.cfg["encoder_rates"])
 208         h = pre_block(h, _sub(self.p, "pre_block."), self.cfg["num_attention_heads"], self.cfg["latent_channels"])
 209         return conv1d(h, self.p, "mean_proj")
```

`AudioVAE.encode_mean` takes channels-last `(B, samples, 1)`, where diffusers takes `(B, 1, samples)`.

**The alias-free SnakeBeta activation** upsamples ×2, applies the activation, then low-passes back down. The port uses
MLX's grouped `conv_transpose1d` for the upsampling and reads the Kaiser filters from the checkpoint's own buffers:

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
📍 [`src/h3mlx/audio_vae.py:73-82`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/audio_vae.py#L73-L82)
```python
  73 def upsample(x, filt: mx.array, ratio: int):
  74     """ratio * conv_transpose1d(replicate-pad(x, pad), filter, stride=ratio, groups=C), then crop."""
  75     k = filt.size
  76     pad = k // ratio - 1
  77     pad_left = pad * ratio + (k - ratio) // 2
  78     pad_right = pad * ratio + (k - ratio + 1) // 2
  79     x = mx.pad(x, [(0, 0), (pad, pad), (0, 0)], mode="edge")
  80     c = x.shape[-1]
  81     y = ratio * mx.conv_transpose1d(x, _depthwise(filt, c), stride=ratio, groups=c)
  82     return y[:, pad_left : y.shape[1] - pad_right]
```
📍 [`src/h3mlx/audio_vae.py:85-90`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/audio_vae.py#L85-L90)
```python
  85 def activation1d(x, p, ratio: int = 2):
  86     """Alias-free SnakeBeta: upsample x2, activate, low-pass downsample x2. Filters are the checkpoint's buffers
  87     (`upsample.filter`, `downsample.lowpass.filter`), Kaiser-sinc, kernel 12."""
  88     x = upsample(x, p["upsample.filter"], ratio)
  89     x = snake_beta(x, p["act.alpha"], p["act.beta"])
  90     return lowpass(x, p["downsample.lowpass.filter"], ratio)
```

**Attention in `pre_block`.** The causal attention mean-pools its heads away, then adaptively average-pools the head
dimension:

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
📍 [`src/h3mlx/audio_vae.py:151-158`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/audio_vae.py#L151-L158)
```python
 151 def causal_attention(x, p, heads, out_dim):
 152     b, n, c = x.shape
 153     bias = mx.concatenate([p["q_bias"], mx.zeros_like(p["q_bias"]), p["v_bias"]])
 154     qkv = (x @ p["qkv.weight"].T + bias).reshape(b, n, 3, heads, c // heads)
 155     q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
 156     o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(c // heads), mask="causal")
 157     o = o.transpose(0, 2, 1, 3).mean(axis=2)  # heads mean-pooled away
 158     return lin(adaptive_avg_pool_last(o, out_dim), p, "proj")
```

**Weights.** Weight-norm pairs are folded once. The transposed-convolution weights get their own layout rule:

📍 [`src/h3mlx/audio_vae.py:17-31`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/audio_vae.py#L17-L31)
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
📍 [`src/h3mlx/audio_vae.py:34-42`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/audio_vae.py#L34-L42)
```python
  34 def to_mlx_layout(sd: dict, transpose_conv_prefixes=("decoder.ups.",)) -> dict:
  35     """numpy torch-layout weights -> MLX arrays in MLX conv layout. 3-D `weight`s are convs."""
  36     out = {}
  37     for k, v in sd.items():
  38         a = np.asarray(v, dtype=np.float32) if v.dtype != np.float32 else np.asarray(v)
  39         if k.endswith("weight") and a.ndim == 3 and not k.endswith("filter"):
  40             a = a.transpose(1, 2, 0) if k.startswith(transpose_conv_prefixes) else a.transpose(0, 2, 1)
  41         out[k] = mx.array(a)
  42     return out
```

**Resampling.** Reference soundtracks are resampled to 32 kHz with torchaudio's sinc resampler, as diffusers does.
torchaudio 2.9.1 requires torch 2.9.1 and we run torch 2.15, so `src/h3mlx/resample.py` carries its two resampling
functions verbatim under torchaudio's BSD-2 licence. `tests/test_resample.py` checks the copy against outputs from
real torchaudio 2.9.1 at 48, 44.1, 22.05 and 16 kHz.

**Tests.** `tests/test_audio_vae.py` covers a tiny config. The release's hop of 800 was also checked once, outside
the suite.

This section is a partial tour. The full DAC encoder and BigVGAN decoder are `encoder`, `pre_block`, `amp_block` and
`bigvgan` in `src/h3mlx/audio_vae.py`.

---

## 8. ref2va end to end

**Setup** validates the request:

- the canvas must be a multiple of 32;
- at most 9 images, 3 videos and 3 audio clips, 12 in total;
- audio cannot be the only reference.

It then resolves the canvas (by default 1344×768), aligns the frame count to `17n + 5`, and normalises the media:
images to their own 2048 px short edge, videos onto 24 fps and their own canvas, audio onto 32 kHz.

**Accepted inputs differ.** The port accepts PIL images, uint8 numpy video `(T, H, W, 3)` and numpy audio
`(channels, samples)`. diffusers also accepts tensors and other array layouts, and lets the caller supply their own
latents; the port does not.

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
📍 [`src/h3mlx/pipeline.py:115-152`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/pipeline.py#L115-L152)
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

**Noise draws** come from one generator in a fixed order: each condition, then the target video tensor, then the
target audio rows. That order is part of what a seed reproduces:

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
📍 [`src/h3mlx/pipeline.py:268-282`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/pipeline.py#L268-L282)
```python
 268     gen = Noise(seed)
 269     patch = tuple(c.dit_cfg["patch_size"])
 270     rows = []
 271     for x in cond:  # one draw per condition, packed order, then held at t = 0.999
 272         b, f, h, w, ch = x.shape
 273         nz = mx.array(gen((b, ch, f, h, w)).transpose(0, 2, 3, 4, 1))
 274         t32 = np.float32(NOISE_AUG)  # scale_noise: t * x + (1 - t) * noise, both factors in float32
 275         noised = mx.array(t32) * x + mx.array(np.float32(1) - t32) * nz
 276         rows.append(ops.patchify(noised.transpose(0, 4, 1, 2, 3), patch))
 277     lat_ch = c.dit_cfg["in_channels"]
 278     target = mx.array(gen((1, lat_ch, lf, lh, lw)))
 279     rows.append(ops.patchify(target, patch))
 280     video_rows = mx.concatenate(rows)
 281     audio_ch = c.dit_cfg["audio_in_channels"]
 282     audio_rows = mx.concatenate(ref_audio + [mx.array(gen((na * AUDIO_CHANNELS, audio_ch)))])
```

**After denoising**, diffusers splits it into three steps:

1. drop the reference rows and unpack the video rows;
2. undo the video latent normalisation, decode, then undo ImageNet's normalisation on the pixels;
3. undo the audio latent normalisation, then decode.

The port does all three in one block:

📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/decoders.py:98-129`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/decoders.py#L98-L129)
```python
  98     @torch.no_grad()
  99     def __call__(
 100         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 101     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 102         block_state = self.get_block_state(state)
 103         patch_t, patch_h, patch_w = components.patch_size
 104         channels = components.vae_latent_channels
 105
 106         # The inverse of the patchify in the prepare-latents step: rows are frame-major then row-major.
 107         rows = block_state.latents[block_state.num_condition_video_rows :]
 108         rows = rows.reshape(
 109             -1,
 110             block_state.num_latent_frames // patch_t,
 111             block_state.latent_height // patch_h,
 112             block_state.latent_width // patch_w,
 113             channels,
 114             patch_t,
 115             patch_h,
 116             patch_w,
 117         )
 118         rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
 119         block_state.latents = rows.reshape(
 120             -1, channels, block_state.num_latent_frames, block_state.latent_height, block_state.latent_width
 121         ).contiguous()
 122
 123         # Audio rows are channel-major, and the mono audio VAE takes the two stereo channels as two batch items.
 124         audio_rows = block_state.audio_latents[block_state.num_condition_audio_rows :]
 125         audio_rows = audio_rows.reshape(components.audio_channels, block_state.num_audio_latents, audio_rows.shape[-1])
 126         block_state.audio_latents = audio_rows.permute(0, 2, 1).contiguous()
 127
 128         self.set_block_state(state, block_state)
 129         return components, state
```
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
📍 [`diffusers:src/diffusers/modular_pipelines/minimax_h3/decoders.py:240-256`](https://github.com/huggingface/diffusers/blob/2a1433913c7044f34fa4767dd84b28bba0bb0ecf/src/diffusers/modular_pipelines/minimax_h3/decoders.py#L240-L256)
```python
 240     @torch.no_grad()
 241     def __call__(
 242         self, components: MiniMaxH3ModularPipeline, state: PipelineState
 243     ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
 244         block_state = self.get_block_state(state)
 245         device = components._execution_device
 246
 247         audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
 248         audio_latents_std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
 249         audio_latents = block_state.audio_latents * audio_latents_std + audio_latents_mean
 250
 251         audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
 252         block_state.audio = audio.float().permute(1, 0, 2)
 253         block_state.sampling_rate = components.audio_sampling_rate
 254
 255         self.set_block_state(state, block_state)
 256         return components, state
```
📍 [`src/h3mlx/pipeline.py:216-223`](https://github.com/sublimator/minimax-h3-mlx/blob/c7919dfe05859c5b93394b11044de9f12df6c901/src/h3mlx/pipeline.py#L216-L223)
```python
 216     v_lat = ops.unpatchify(video_rows[ncv:], lat_ch, lf, lh, lw, patch)  # (1, C, F, H, W)
 217     v_lat = v_lat.transpose(0, 2, 3, 4, 1) * mx.array(c.vae_latents_std) + mx.array(c.vae_latents_mean)
 218     px = c.vae.decode(v_lat)[0]  # (frames, H, W, 3)
 219     video = mx.clip(px * mx.array(PIXEL_STD) + mx.array(PIXEL_MEAN), 0.0, 1.0)
 220
 221     a_lat = audio_rows[nca:].reshape(AUDIO_CHANNELS, na, audio_ch)
 222     a_lat = a_lat * mx.array(c.audio_latents_std) + mx.array(c.audio_latents_mean)
 223     audio = c.audio_vae.decode(a_lat)[:, :, 0]  # (2, samples)
```

**End-to-end test.** `tests/test_pipeline_e2e.py` runs one whole ref2va request through diffusers' own
`MiniMaxH3Blocks` and through `h3mlx.pipeline.ref2va`:

- tiny random f32 models, plus the released tokenizer and processor;
- an image reference and a video reference with a soundtrack;
- 3 grid points, so 2 model calls.

The test requires the decoded video and audio to agree within 2e-3. Observed: 3.6e-7 for video and 1.4e-7
for audio. This is one tiny f32 configuration, not the bf16 release.

---

## 9. What is not ported, or not yet

- **LoRA.** diffusers' generic MiniMax H3 LoRA loader (which routes each LoRA to `transformer` or `transformer_ref`)
  is not ported. One specific LoRA is: HyperFlow, Video Rebirth's 8-step LoRA, with its two-time embedder and fixed
  grid; see `docs/porting-hyperflow.md`. Its hook is `ops.plinear`: every targeted linear adds a LoRA when the weight
  dict carries `<name>.lora_A.weight` / `.lora_B.weight`, and without them it is exactly `linear(x, p[name + ".weight"], p[name + ".bias"] if bias else
  None)`, the call §2's attention, feed-forward and time-embedder code made before HyperFlow. Only the linears that go
  through `plinear` can carry a LoRA (`hyperflow.wired_shapes`); a file targeting any other is refused.
- **Speedups.** Sparse attention and quantisation: a dense bf16 baseline comes first.
- **Caller-supplied inputs.** diffusers accepts caller-supplied latents and tensor inputs; the port takes the input
  types listed in §8 only.
- **t2va and fl2va pipelines.** Their packers and ops are here and tested (`build_packed_sequence`); only ref2va is
  assembled end to end.
- **Real weights.** Everything above was checked with tiny random weights and the release's configs, index, tokenizer
  and processor. The released weights have run once end to end on an M3 Ultra (256 GB): run-001, 512×288, 124 frames, 49 calls,
  1,014 s, with the reference image at a 512 px short edge (off-spec; the release normalises to 2048). It made a clip
  judged coherent by eye. Quality against diffusers' own output on the same
  inputs is not measured.