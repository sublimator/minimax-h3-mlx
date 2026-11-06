# h3mlx

Two models in [MLX](https://github.com/ml-explore/mlx), for Apple silicon:

- **MiniMax H3:** text and reference images to video with sound.
- **[P-H3-Image](#p-h3-image):** Pruna AI's image model built on H3's transformer, for text-to-image and editing with up
  to 8 pictures.

Both were built op by op and tested against their implementations in
[diffusers](https://github.com/huggingface/diffusers) (`transformer_minimax_h3.py`, `scheduling_minimax_h3.py`,
`modular_pipelines/minimax_h3/`, `modular_pipelines/p_h3_image/` and the autoencoders) on small random tensors.

## What's here

- **ref2va end to end:** reference images (and video/audio references) to video and audio, staged so the text encoder
  and the transformer are never resident together.
- **The parts:** the DiT and its token refiner, the scheduler and denoise loop, the packed-sequence layouts (t2va,
  fl2va, ref2va), the Qwen3-VL presentation and text encoder (via mlx-vlm), the video VAE (encoder, ViT decoder,
  tiling) and the audio VAE.
- **HyperFlow:** Video Rebirth's [8-step LoRA](https://github.com/Video-Rebirth/hyperflow) (`--hyperflow`), 8
  transformer calls instead of 49.
- **P-H3-Image:** both workflows end to end, its text connector, the single-frame image decoder, and its prompt
  upsampler (Qwen3.5-9B, via mlx-vlm). It reuses H3's transformer, packing and denoise loop unchanged.
- **Porting guides:** [`docs/porting.md`](docs/porting.md) (diffusers -> h3mlx) and
  [`docs/porting-hyperflow.md`](docs/porting-hyperflow.md), side by side, with permalinks to the originals. P-H3-Image
  has no guide yet.

## Tests

    uv sync
    uv run pytest

No weights are needed. A plain clone runs 212 tests and skips 33. Those 33 need the releases' tokenizers, processors,
configs and weight indexes, plus HyperFlow's manifest and adapter header. These files aren't shipped here because they
belong to MiniMax, Video Rebirth and Pruna. Fetch them from Hugging Face at pinned revisions (about 50 MB, no weights)
and all 245 run:

    uv run python scripts/fetch_test_fixtures.py   # writes tests/data/release/ (git-ignored)
    uv run pytest

Or point `H3MLX_FIXTURES` at a copy. Seven of the 33 also need `ffmpeg` and `ffprobe` on the `PATH`.

The oracle is diffusers main at `d3d8af89` (P-H3-Image isn't in a diffusers release yet); its H3 code is 0.42.1's plus a
tensor-parallel plan, and the H3 tests pass against both.

## Running H3

You need:
- **The MiniMax H3 weights:** `MiniMaxAI/MiniMax-H3` on Hugging Face, tested at revision `42ed227` (about 144 GB).
  `scripts/check_weights.sh` checks a local copy against its sizes and sha256s.
- **For `--hyperflow`, the HyperFlow adapter:** `minimax_h3_hyperflow_8step_v1.0.safetensors` from
  `VideoRebirthAI/hyperflow`, revision `5bde94d7`. The run script checks its sha256 (`9297f450...f447`).
- **`ffmpeg` and `ffprobe`** with H.264 and AAC encoders. The script checks the writer before loading any model.
- **A reference image and a prompt file.**

    bash scripts/check_weights.sh MiniMax-H3
    uv run python scripts/run_ref2va.py --weights MiniMax-H3 --image ref.png --prompt-file prompt.txt \
        --height 576 --width 1024 --frames 124 --seed 0 --ref-short-edge 768 --out out/run \
        --hyperflow minimax_h3_hyperflow_8step_v1.0.safetensors

`--plan-only` prints the sizes a run would use without reading any weights. The script writes `video.mp4` (with sound),
`audio.wav`, the final latents and a `run.json` with per-stage times and memory.

`--ref-short-edge 768` sizes the reference image at 768 px on its short edge. The release's rule (the default) is 2048,
which adds many rows to the sequence and makes runs much slower (see the table).

**Measured on an M3 Ultra (60-core GPU, 256 GB),** one image reference, 124 frames (5.2 s), bf16, dense attention:

| output | reference short edge | calls | total |
| --- | --- | --- | --- |
| 512x288 | 512 | 49 (base) | 16.9 min |
| 512x288 | 512 | 8 (HyperFlow) | 3.5 min |
| 672x384 | 768 | 8 (HyperFlow) | 6.7 min |
| 1024x576 | 768 | 8 (HyperFlow) | 16.5 min |
| 1344x768 | 2048 | 8 (HyperFlow) | 54 min |

The denoise peak was 70-88 GB. At 384p faces and clothing broke down; 576p held up better. HyperFlow adds some
frame-to-frame shimmer against the base model.

## P-H3-Image

[P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image) (Pruna AI) is an image model built from MiniMax H3: H3's
transformer (the same architecture and weight names, fine-tuned), a Qwen3-VL-4B text encoder whose hidden states 13,
19, 25, 31 and 36 a small connector maps to the transformer's text input, an image VAE (H3's encoder and latent space,
a single-frame ViT decoder), and a Qwen3.5-9B prompt upsampler. It samples 8 steps on its own sigma grid with no
guidance and works best at 8 MP. `h3mlx.image` runs both workflows, text-to-image and editing (Picture 1 is edited,
up to 7 more are references), and `h3mlx.upsampler` runs the upsampler with mlx-vlm.

You need the weights: `PrunaAI/P-H3-Image`, tested at revision `7f98ace` (about 100 GB with the upsampler, 81 GB
without).

    uv run python scripts/run_p_h3_image.py --weights P-H3-Image --prompt-file prompt.txt --out out/img
    uv run python scripts/run_p_h3_image.py --weights P-H3-Image --prompt-file edit.txt --image photo.png \
        --image hat.png --upsample --out out/edit

Without `--upsample` the prompt is rendered as written, so write it the way the model card says: say what the image
is, describe one scene, quote every word of text, leave out hype words. `--upsample` runs the upsampler first, as the
diffusers pipeline does by default (its greedy decoding can differ from torch's in places; the prompt it wrote is saved
as `prompt.txt`). `--plan-only` prints the sizes without reading weights.

Sizes follow the pipeline. Text-to-image defaults to 2912x2912 (8 MP); the card lists the 8 MP size for each aspect
ratio, and other sizes are rounded to multiples of 32. An edit is generated at Picture 1's size, scaled up to at least
4 MP, unless `--height` and `--width` set the canvas; the pictures keep their own sizes between 1 and 8 MP, so each
one adds its rows to the sequence. `--sigmas` replaces the 9-point grid, `--vae-dtype bfloat16` decodes at the stored
precision instead of float32, and `--decode-from` decodes saved latents. The script writes `image.png`, `prompt.txt`
(the prompt rendered), the final latents and a `run.json` with per-stage times and memory.

**Measured on the M3 Ultra (60-core GPU, 256 GB),** 8 calls, bf16 transformer, float32 decode:

| request | size | rows | per call | total | peak |
| --- | --- | --- | --- | --- | --- |
| text-to-image, prompt as written | 2912x2912 | 8,395 | 26 s | 4.2 min | 69 GB |
| edit, one 8.5 MP picture, `--upsample` | 2912x2912 | 17,749 | 69 s | 10.5 min | 72 GB |

The upsampler took 14 s, the text encoder 2 s, the decode 34 s. The edit's picture doubled the sequence; an edit at
4 MP should take about 40% of the time, around 4.5 min (an estimate from these runs, not measured).

## Licence

The code is under the Apache License 2.0 ([`LICENSE`](LICENSE)). Parts are translated from, or quote, diffusers
(its P-H3-Image code is Pruna AI's), HyperFlow, PEFT, mlx-vlm and torchaudio. Their notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`NOTICE`](NOTICE) says what came from where. The MiniMax H3
and P-H3-Image weights and release files are not included. They come under MiniMax's own licence, which has territory
exclusions, so read it before you use them.
