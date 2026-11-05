"""Fetch the release metadata some tests need. No weights are downloaded.

    uv run python scripts/fetch_test_fixtures.py            # writes tests/data/release/ (git-ignored)
    uv run python scripts/fetch_test_fixtures.py --out DIR   # then: H3MLX_FIXTURES=DIR uv run pytest

What it writes, all from Hugging Face at pinned revisions:
  MiniMax-H3/                             MiniMaxAI/MiniMax-H3 @ 42ed227: model_index.json, the transformer_ref and
                                          text_encoder configs, the transformer_ref weight index, the tokenizer and
                                          the processor
  transformer_ref.dtypes.json             every transformer_ref tensor's dtype, read from the 14 shard headers
  transformer_ref.lfs-42ed227-vs-83db0c0.json
                                          transformer_ref's file ids at 42ed227 and at 83db0c0 (HyperFlow's base)
  hyperflow_v1.0.header.json              the HyperFlow v1.0 adapter's safetensors header (VideoRebirthAI/hyperflow)
  hyperflow.manifest-5bde94d7.json        that repo's hyperflow.json

These files belong to MiniMax and Video Rebirth and come under their licences; they are not part of this repository.
Without them the tests that need them are skipped.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

H3 = "MiniMaxAI/MiniMax-H3"
H3_REV = "42ed227ee7df40d41602854ae760620d6eb651fe"
H3_BASE = "83db0c0efe6ef9824e0e194be110346c0a9542ed"  # the revision HyperFlow v1.0 was trained on
HYPERFLOW = "VideoRebirthAI/hyperflow"
HYPERFLOW_REV = "5bde94d7de3e78dafa2abc43ee2f280a1cb244e8"
ADAPTER = "minimax_h3_hyperflow_8step_v1.0.safetensors"


def tree(api: HfApi, revision: str) -> dict:
    out = {}
    for f in api.list_repo_tree(H3, path_in_repo="transformer_ref", revision=revision, expand=True):
        if hasattr(f, "size"):
            out[f.path.split("/", 1)[1]] = {
                "lfs_sha256": f.lfs.sha256 if f.lfs else None, "oid": f.blob_id, "size": f.size}
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "data" / "release")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    snapshot_download(H3, revision=H3_REV, local_dir=out / "MiniMax-H3", allow_patterns=[
        "model_index.json", "transformer_ref/config.json", "transformer_ref/*.index.json", "text_encoder/config.json",
        "tokenizer/*", "processor/*"])
    shutil.rmtree(out / "MiniMax-H3" / ".cache", ignore_errors=True)

    index = json.loads((out / "MiniMax-H3" / "transformer_ref" / "diffusion_pytorch_model.safetensors.index.json")
                       .read_text())
    dtypes = {}
    for shard in sorted(set(index["weight_map"].values())):
        meta = api.parse_safetensors_file_metadata(H3, f"transformer_ref/{shard}", revision=H3_REV)
        dtypes.update({name: t.dtype for name, t in meta.tensors.items()})
    (out / "transformer_ref.dtypes.json").write_text(json.dumps(dict(sorted(dtypes.items()))))

    revs = {rev: tree(api, rev) for rev in (H3_REV, H3_BASE)}
    (out / "transformer_ref.lfs-42ed227-vs-83db0c0.json").write_text(json.dumps({
        "identical": revs[H3_REV] == revs[H3_BASE], "revisions": revs,
        "source": f"https://huggingface.co/api/models/{H3}/tree/<revision>/transformer_ref"}, indent=1))

    meta = api.parse_safetensors_file_metadata(HYPERFLOW, ADAPTER, revision=HYPERFLOW_REV)
    header = {"__metadata__": meta.metadata}
    header.update({name: {"data_offsets": list(t.data_offsets), "dtype": t.dtype, "shape": list(t.shape)}
                   for name, t in sorted(meta.tensors.items())})
    (out / "hyperflow_v1.0.header.json").write_text(json.dumps(header, indent=1))

    shutil.copyfile(hf_hub_download(HYPERFLOW, "hyperflow.json", revision=HYPERFLOW_REV),
                    out / "hyperflow.manifest-5bde94d7.json")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
