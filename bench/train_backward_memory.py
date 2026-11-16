"""Measurement behind docs/porting-lora-training.md's section on the bounded backward pass: the peak of MLX's active
memory during one gradient of a synthetic transformer, as its number of blocks grows, four ways:

  sampling forward   mx.value_and_grad through ops.dit_forward, which evaluates after every block
  plain              train.loss_and_gradients (nothing evaluated inside the forward)
  checkpointed       backward.loss_and_gradients_checkpointed (each block an mx.checkpoint)
  blockwise          backward.loss_and_gradients_blockwise (a vector-Jacobian product per block)

on the default device and on the CPU stream. Every cell is its own process. The model is the one
tests/test_backward.py builds (`synthetic`: hidden 96, random weights in the release's mixed precision, a rank-4
float32 adapter on the block linears, one text-to-image example of side x side latents, 596 rows at the default 48).
The number is `mx.get_peak_memory()` over the active memory before the gradient: buffers MLX holds in its cache are
not in it. It shows how the four grow with depth on this model and this machine; it does not predict 50 real blocks.

Last, the sixteen-block chain of tests/test_backward.py::test_an_eval_in_the_loop_defeats_the_checkpoint (`chain_peak`:
1,024 rows, width 256, no transformer) on both devices: plain and checkpointed, without and with an evaluation after
every block.

Run from the repo root: python bench/train_backward_memory.py [--side 48] [--layers 4 8 16]"""
import argparse
import json
import subprocess
import sys

WAYS = ("sampling forward", "plain", "checkpointed", "blockwise")


def cell(way: str, layers: int, device: str, side: int) -> None:
    sys.path.insert(0, "tests")
    import mlx.core as mx
    from test_backward import gradient_peak

    peak, seconds, loss = gradient_peak(way, layers, mx.cpu if device == "cpu" else None, side)
    print(json.dumps({"peak": peak, "seconds": seconds, "loss": loss}))


def chain_cell(checkpoint: str, evaluate: str, device: str) -> None:
    sys.path.insert(0, "tests")
    import mlx.core as mx
    from test_backward import chain_peak

    peak, value = chain_peak(checkpoint == "checkpointed", evaluate == "evals", device=mx.cpu if device == "cpu" else None)
    print(json.dumps({"peak": peak, "value": value}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", type=int, default=48, help="the example's latent height and width")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--cell", nargs=3, metavar=("WAY", "LAYERS", "DEVICE"), help=argparse.SUPPRESS)
    parser.add_argument("--chain-cell", nargs=3, metavar=("WAY", "EVALS", "DEVICE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cell:
        return cell(args.cell[0], int(args.cell[1]), args.cell[2], args.side)
    if args.chain_cell:
        return chain_cell(*args.chain_cell)
    for device in ("default", "cpu"):
        print(f"\n{'default device' if device == 'default' else 'cpu stream'}, side {args.side}: peak MB over the start (seconds)")
        print("| way | " + " | ".join(f"{n} blocks" for n in args.layers) + " | growth per block, MB |")
        print("| --- |" + " ---: |" * (len(args.layers) + 1))
        losses = {}
        for way in WAYS:
            row = []
            for n in args.layers:
                out = subprocess.run([sys.executable, __file__, "--side", str(args.side), "--cell", way, str(n), device],
                                     capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
                row.append(json.loads(out))
                losses.setdefault(n, []).append(row[-1]["loss"])
            growth = (row[-1]["peak"] - row[0]["peak"]) / (args.layers[-1] - args.layers[0]) if len(row) > 1 else float("nan")
            print(f"| {way} | " + " | ".join(f"{r['peak']:.0f} ({r['seconds']:.2f} s)" for r in row) + f" | {growth:.1f} |")
        spread = max(max(v) - min(v) for v in losses.values())
        print(f"the four losses agree to {spread:.1e} at every depth")

    print("\nthe sixteen-block chain: peak MB over the start")
    print("| device | plain | checkpointed | plain, an eval per block | checkpointed, an eval per block |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for device in ("default", "cpu"):
        row = []
        for evaluate in ("no evals", "evals"):
            for way in ("plain", "checkpointed"):
                out = subprocess.run([sys.executable, __file__, "--chain-cell", way, evaluate, device],
                                     capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
                row.append(json.loads(out)["peak"])
        print(f"| {'default device' if device == 'default' else 'cpu stream'} | " + " | ".join(f"{v:.0f}" for v in row) + " |")


if __name__ == "__main__":
    main()
