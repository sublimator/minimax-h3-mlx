# h3mlx

MiniMax H3 (text and reference images to video with sound) in [MLX](https://github.com/ml-explore/mlx), for Apple
silicon. It was built op by op and tested against the MiniMax H3 implementation in
[diffusers](https://github.com/huggingface/diffusers) 0.42.1 (`transformer_minimax_h3.py`, `scheduling_minimax_h3.py`,
`modular_pipelines/minimax_h3/`) on small random tensors.

## What's here

- **ref2va end to end:** reference images (and video/audio references) to video and audio, staged so the text encoder
  and the transformer are never resident together.
- **The parts:** the DiT and its token refiner, the scheduler and denoise loop, the packed-sequence layouts (t2va,
  fl2va, ref2va), the Qwen3-VL presentation and text encoder (via mlx-vlm), the video VAE (encoder, ViT decoder,
  tiling) and the audio VAE.
- **HyperFlow:** Video Rebirth's [8-step LoRA](https://github.com/Video-Rebirth/hyperflow) (`--hyperflow`), 8
  transformer calls instead of 49.
- **Porting guides:** [`docs/porting.md`](docs/porting.md) (diffusers -> h3mlx) and
  [`docs/porting-hyperflow.md`](docs/porting-hyperflow.md), side by side, with permalinks to the originals.

## Tests

    uv sync
    uv run pytest

No weights are needed. A plain clone runs 194 tests and skips 21. Those 21 need the release's tokenizer, processor,
configs and weight index, plus HyperFlow's manifest and adapter header. These files aren't shipped here because they
belong to MiniMax and Video Rebirth. Fetch them from Hugging Face at pinned revisions (about 20 MB, no weights) and all
215 run:

    uv run python scripts/fetch_test_fixtures.py   # writes tests/data/release/ (git-ignored)
    uv run pytest

Or point `H3MLX_FIXTURES` at a copy. Seven of the 21 also need `ffmpeg` and `ffprobe` on the `PATH`.

## Running it

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

## Licence

The code is under the Apache License 2.0 ([`LICENSE`](LICENSE)). Parts are translated from, or quote, diffusers,
HyperFlow, PEFT, mlx-vlm and torchaudio. Their notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and
[`NOTICE`](NOTICE) says what came from where. The MiniMax H3 weights and release files are not included. They come
under MiniMax's own licence, which has territory exclusions, so read it before you use them.
