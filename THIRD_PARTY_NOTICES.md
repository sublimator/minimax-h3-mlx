# Third-party notices

h3mlx is licensed under the Apache License, Version 2.0 (`LICENSE`). Parts of it are translated from, or quote, the
projects below. Their notices are reproduced here as their licences require. Translated files are modified: the code
was rewritten from PyTorch to MLX and NumPy.

## diffusers (Hugging Face)

https://github.com/huggingface/diffusers, tag v0.42.1. Apache License 2.0.

The MiniMax H3 transformer, scheduler, modular pipeline and autoencoders
(`src/diffusers/models/transformers/transformer_minimax_h3.py`, `src/diffusers/schedulers/scheduling_minimax_h3.py`,
`src/diffusers/modular_pipelines/minimax_h3/`, `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py`,
`autoencoder_kl_minimax_h3_audio.py`) are the source of the translated modules in `src/h3mlx/`, and are quoted in
`docs/porting.md` and `docs/porting-hyperflow.md`. Its P-H3-Image implementation, at main commit d3d8af89
(`src/diffusers/modular_pipelines/p_h3_image/`, `src/diffusers/models/autoencoders/autoencoder_kl_p_h3_image.py`,
`src/diffusers/models/condition_embedders/condition_embedder_p_h3_image.py`), is the source of `src/h3mlx/image.py`,
`src/h3mlx/upsampler.py` and the single-frame decoder in `src/h3mlx/vae.py` and `vae_ops.py`, and is quoted in
`docs/porting-p-h3-image.md`. Its LoRA loading and saving (`src/diffusers/loaders/peft.py`, `loaders/lora_pipeline.py`,
`loaders/lora_base.py`, `utils/peft_utils.py`), its MiniMax H3 LoRA converter (`loaders/lora_conversion_utils.py`) and
its training utilities (`training_utils.py`), at the same commit, are the source of `src/h3mlx/lora.py` and
`src/h3mlx/train.py`, and are quoted in `docs/porting-lora-training.md`, which also quotes seven lines of
`schedulers/scheduling_flow_match_euler_discrete.py`. Those files carry these notices:

```
Copyright 2025 The HuggingFace Inc. team.
Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
Copyright 2026 Pruna AI and The HuggingFace Team. All rights reserved.
Copyright 2026 Pruna AI, The MiniMax and HuggingFace Teams. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## HyperFlow (Video Rebirth)

https://github.com/Video-Rebirth/hyperflow, commit 1dd2f342. Apache License 2.0.

Its loader (`src/hyperflow_h3/lora.py`), two-time embedder (`embedder.py`) and schedule (`schedule.py`) are translated
in `src/h3mlx/hyperflow.py`, `ops.py` and `denoise.py`, transcribed in torch in `tests/test_hyperflow.py`, and quoted
in `docs/porting-hyperflow.md`. Its own third-party notices (parts derived from diffusers, above) apply to those parts.

```
Copyright 2026 The HyperFlow authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## PEFT (Hugging Face)

https://github.com/huggingface/peft, v0.21.2. Apache License 2.0. `docs/porting-hyperflow.md` quotes LoRA
`Linear.forward` (`src/peft/tuners/lora/layer.py`). The scaling, the initialisation, `Linear.get_delta_weight`,
`Linear.merge` and `Linear.forward` of that file, the rank and alpha pattern lookup of `src/peft/tuners/lora/model.py`
and `get_pattern_key` (`src/peft/utils/other.py`) are translated in `src/h3mlx/lora.py` and `ops.py`, and quoted in
`docs/porting-lora-training.md`. `layer.py` notes that its `Linear` is based on Microsoft's loralib
(https://github.com/microsoft/LoRA, MIT License, Copyright (c) Microsoft Corporation).

```
Copyright 2023-present the HuggingFace Inc. team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## ai-toolkit (Ostris)

https://github.com/ostris/ai-toolkit, commit d6a18ad72da9abca3ffb67ea8e42c25764eee61a. MIT License, Copyright (c) 2024
Ostris, LLC.

No code of it is copied, translated or quoted here. Its MiniMax H3 and P-H3-Image model files
(`extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py`, `p_h3_image.py` and `src/packing.py`) were read as a
training recipe, and `src/h3mlx/train.py`'s example builder follows that recipe where diffusers has no code to follow:
silence for a picture's audio rows, no loss on them, references held at the conditioning timestep, and the reference
encode without float16 rounding as an option. `docs/porting-lora-training.md` cites the lines. It is a third-party
trainer's recipe, not the model maker's.

## mlx-vlm

https://github.com/Blaizzy/mlx-vlm, tag v0.7.11. MIT License. `docs/porting.md` and `docs/porting-p-h3-image.md`
quote the Qwen3-VL `hidden_state_at_layer` method.

```
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## torchaudio (PyTorch)

https://github.com/pytorch/audio, v2.9.1. BSD 2-Clause License. `src/h3mlx/resample.py` is its sinc resampler, copied
verbatim; the file carries torchaudio's copyright notice, conditions and disclaimer in full.
