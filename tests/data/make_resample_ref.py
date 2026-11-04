"""Makes tests/data/resample_ref.npz in its own env (torchaudio pins an older torch than ours):
    uv run --no-project --python 3.12 --with torch==2.9.1 --with torchaudio==2.9.1 --with numpy \\
        python -I tests/data/make_resample_ref.py tests/data/resample_ref.npz
"""

import sys

import numpy as np
import torch
import torchaudio

out = {}
rng = np.random.default_rng(0)
for sr in (48000, 44100, 22050, 16000):
    for ch in (1, 2):
        w = (rng.standard_normal((ch, 4801)) * 0.3).astype(np.float32)
        out[f"in_{sr}_{ch}"] = w
        x = torch.from_numpy(w)
        if ch == 1:
            x = x.expand(2, -1).contiguous()
        out[f"out_{sr}_{ch}"] = torchaudio.transforms.Resample(sr, 32000)(x).numpy()
np.savez(sys.argv[1], **out, versions=np.array([torch.__version__, torchaudio.__version__]))
print("torch", torch.__version__, "torchaudio", torchaudio.__version__, len(out))
