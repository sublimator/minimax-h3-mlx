"""Measurement behind docs/porting-lora-training.md's section on the modulation tables' gradient: the gradient of a
six-row half-precision table gathered into many sequence rows, as the rows grow, against the exact sum (float64) of
what each row sends back. Three gathers: `ops.take_rows` (the sampling forward's, summed in the table's dtype),
`ops.take_rows_for_training` (through float32) and torch's `index_select`, which the reference module uses. The rows'
factors are standard normal draws, with a mean of zero and with a mean of 0.5 (a common sign makes a running bf16 sum
stall sooner). Relative error of the whole gradient, on the CPU stream. Not a test: the test at one size is
tests/test_train_gradients.py::test_the_training_gathers_gradient_is_torchs_index_select.

Run from the repo root: python bench/train_gather_gradient.py"""
import mlx.core as mx
import numpy as np
import torch

from h3mlx import ops

HIDDEN, TABLE_ROWS = 64, 6


def error(got, want):
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


print("| rows | per table row | mean | take_rows | take_rows_for_training | torch index_select |")
print("| ---: | ---: | ---: | ---: | ---: | ---: |")
for rows in (600, 6000, 24000, 48000):
    for mean in (0.0, 0.5):
        rng = np.random.default_rng(rows)
        index = rng.integers(0, TABLE_ROWS, rows)
        factors = mx.array((rng.standard_normal((rows, HIDDEN)) + mean).astype(np.float32)).astype(mx.bfloat16)
        factors_np = np.array(factors.astype(mx.float32))
        exact = np.zeros((TABLE_ROWS, HIDDEN))
        np.add.at(exact, index, factors_np.astype(np.float64))
        table = mx.array(rng.standard_normal((TABLE_ROWS, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        ours = {}
        with mx.stream(mx.cpu):
            for take in (ops.take_rows, ops.take_rows_for_training):
                g = mx.grad(lambda t, take=take: (take(t, mx.array(index)) * factors).astype(mx.float32).sum())(table)
                ours[take.__name__] = np.array(g.astype(mx.float32))
        theirs = torch.from_numpy(np.array(table.astype(mx.float32))).to(torch.bfloat16).requires_grad_(True)
        (theirs.index_select(0, torch.from_numpy(index)) * torch.from_numpy(factors_np).to(torch.bfloat16)).float().sum().backward()
        print(f"| {rows:,} | {rows // TABLE_ROWS:,} | {mean} | {error(ours['take_rows'], exact):.4f} | "
              f"{error(ours['take_rows_for_training'], exact):.4f} | {error(theirs.grad.float().numpy(), exact):.4f} |")
