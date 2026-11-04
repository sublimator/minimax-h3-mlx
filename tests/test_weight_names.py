"""The released transformer_ref (HF MiniMaxAI/MiniMax-H3 @ 42ed227, root diffusers layout) loads into dit_forward by
name, with no renaming. Uses the shipped config and safetensors index only (tests/data), never the weights."""

import json
import re
from pathlib import Path

import mlx.core as mx
import torch
from conftest import state_mx, to_mx
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from test_dit_tiny import TINY, packed_layout

from h3mlx import ops

DATA = Path(__file__).parent / "data"


def released():
    cfg = json.loads((DATA / "transformer_ref.config.json").read_text())
    keys = set(json.loads((DATA / "transformer_ref.index.json").read_text())["weight_map"])
    return {k: v for k, v in cfg.items() if not k.startswith("_")}, keys


def test_index_matches_diffusers_module_names_at_full_size():
    cfg, keys = released()
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**cfg)
    assert set(model.state_dict()) == keys
    assert len(keys) == 638


class Recorder(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.read = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def items(self):  # _sub() walks items(); count those reads too
        for k, v in super().items():
            self.read.add(k)
            yield k, v


def layer_pattern(keys, n_layers, n_ref):
    out = set()
    for k in keys:
        k = re.sub(r"^transformer_blocks\.\d+\.", "transformer_blocks.N.", k)
        k = re.sub(r"^token_refiner\.refiner_blocks\.\d+\.", "token_refiner.refiner_blocks.N.", k)
        out.add(k)
    return out


def test_dit_forward_reads_exactly_the_released_names():
    """Every name dit_forward reads exists in the release, and it reads all of them (per-layer names up to the index)."""
    _, keys = released()
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    lay = packed_layout()
    p = Recorder(state_mx(model))
    ops.dit_forward(p, dict(model.config), mx.zeros((1, 12, 16)), mx.zeros((1, 7, 6)), mx.zeros((1, 5, 20)),
                    **{k: to_mx(v) for k, v in lay.items()})
    assert layer_pattern(p.read, 2, 1) == layer_pattern(keys, 50, 2)



def test_precision_rule_matches_the_released_headers():
    """tests/data/transformer_ref.dtypes.json: every tensor's dtype from the 14 real shard headers (HTTP range reads,
    no weights). load.KEEP_F32 must reproduce it exactly, or check_headers would block a good run."""
    from h3mlx.load import KEEP_F32

    dtypes = json.loads((DATA / "transformer_ref.dtypes.json").read_text())
    assert len(dtypes) == 638
    for k, v in dtypes.items():
        assert v == ("F32" if k.split(".")[0] in KEEP_F32 else "BF16"), k
