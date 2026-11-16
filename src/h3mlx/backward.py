"""An example's loss and gradients without the whole backward pass held in memory at once: two ways.

`train.loss_and_gradients` differentiates the forward as one graph, and every block's intermediates are alive until
the backward pass has used them. Both ways here run the same forward (`train.ExampleForward`) and return the same
loss and the same gradients:

* `loss_and_gradients_checkpointed`: each block is an `mx.checkpoint`, so the backward pass recomputes its
  intermediates instead of keeping them. A checkpointed function gets gradients for its arguments only: an array it
  captures from outside gets zero, with no error. So the block's input, the time embedding and the block's trainable
  matrices are arguments, and its frozen weights, for which zero is the gradient wanted, are captured.
* `loss_and_gradients_blockwise`: by hand. The blocks are walked forward keeping each block's input, then backward
  one at a time with a vector-Jacobian product, each evaluated before the next.

What either saves depends on the device and the sizes (`bench/train_backward_memory.py` measures a synthetic model),
and a small model does not predict the real one.
"""

import mlx.core as mx

from . import ops
from .train import Example, ExampleForward, check_inputs

BLOCK = "transformer_blocks."
TAIL = ("norm_out.", "proj_out.", "audio_proj_out.")


def _stages(trainable: dict, cfg: dict) -> tuple[list, list, list]:
    """The trainable keys by the stage of the forward that reads them: (before the blocks, one list per block, after
    the blocks). Before: the input projections, the token refiner and the time embedder. After: the final norm's
    projection and the two heads."""
    blocks = [[k for k in trainable if k.startswith(f"{BLOCK}{i}.")] for i in range(cfg["num_layers"])]
    tail = [k for k in trainable if k.startswith(TAIL)]
    taken = set(tail).union(*blocks)
    head = [k for k in trainable if k not in taken]
    stray = [k for k in head if k.startswith(BLOCK)]
    if stray:  # `check_inputs` refuses these first; the stage before the blocks is not a place for what is left over
        raise ValueError(f"{len(stray)} trainable key(s) are on blocks this config does not have, e.g. {stray[0]!r}")
    return head, blocks, tail


def checkpointed_loss(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array, weight=None) -> mx.array:
    """`train.example_loss` with each block run as an `mx.checkpoint` of (the block's input, the time embedding, the
    block's trainable matrices). The same operations, so the same loss bit for bit."""
    check_inputs(trainable, frozen, cfg, example, text)
    p = {**frozen, **trainable}
    forward = ExampleForward(cfg, example, text, weight)
    x, temb = forward.start(p)
    for i in range(cfg["num_layers"]):
        prefix = f"{BLOCK}{i}."
        names = [k[len(prefix):] for k in trainable if k.startswith(prefix)]
        constants = ops._sub(frozen, prefix)

        def block(x, temb, *matrices, constants=constants, names=names):
            return forward.block(x, temb, {**constants, **dict(zip(names, matrices))})

        x = mx.checkpoint(block)(x, temb, *[trainable[prefix + k] for k in names])
    return forward.finish(x, temb, p)


def loss_and_gradients_checkpointed(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array,
                                    weight=None):
    """(loss, gradients) as `train.loss_and_gradients` returns them, with each block checkpointed. One lazy graph:
    nothing is evaluated here, and the caller's evaluation must not come before the gradients are built."""
    check_inputs(trainable, frozen, cfg, example, text)
    return mx.value_and_grad(checkpointed_loss)(trainable, frozen, cfg, example, text, weight)


def loss_and_gradients_blockwise(trainable: dict, frozen: dict, cfg: dict, example: Example, text: mx.array,
                                 weight=None, after_stage=None):
    """(loss, gradients) as `train.loss_and_gradients` returns them, one block at a time; evaluated on return.
    `after_stage()`, when given, is called after every evaluation (each block on the way up and on the way back):
    the place for a caller to look at memory and stop by raising.

    Forward: the stages run in order, each evaluated, and each block's input is kept. Backward: the vector-Jacobian
    product of the last stage gives the cotangent of the last block's output; each block, last to first, turns the
    cotangent of its output into those of its input, of the time embedding and of its own trainable matrices, and is
    evaluated before the next; the stage before the blocks takes what is left. One block's graph is alive at a time,
    and what grows with depth is the kept inputs, one per block. Each block's forward runs twice (on the way up, and
    again inside its product)."""
    check_inputs(trainable, frozen, cfg, example, text)
    p = {**frozen, **trainable}
    forward = ExampleForward(cfg, example, text, weight)
    head_keys, block_keys, tail_keys = _stages(trainable, cfg)

    def start(*matrices):
        return list(forward.start({**p, **dict(zip(head_keys, matrices))}))

    def block(i):
        prefix = f"{BLOCK}{i}."
        constants = ops._sub(frozen, prefix)
        names = [k[len(prefix):] for k in block_keys[i]]
        return lambda x, temb, *matrices: forward.block(x, temb, {**constants, **dict(zip(names, matrices))})

    def finish(x, temb, *matrices):
        return forward.finish(x, temb, {**p, **dict(zip(tail_keys, matrices))})

    after_stage = after_stage or (lambda: None)
    x, temb = start(*[trainable[k] for k in head_keys])
    mx.eval(x, temb)
    after_stage()
    inputs = [x]
    for i in range(cfg["num_layers"]):
        x = block(i)(inputs[-1], temb, *[trainable[k] for k in block_keys[i]])
        mx.eval(x)
        after_stage()
        inputs.append(x)

    (loss,), (g_x, g_temb, *g_tail) = mx.vjp(finish, [inputs.pop(), temb, *[trainable[k] for k in tail_keys]],
                                             [mx.array(1.0, dtype=mx.float32)])
    mx.eval(loss, g_x, g_tail)
    after_stage()
    grads = dict(zip(tail_keys, g_tail))
    for i in reversed(range(cfg["num_layers"])):
        _, (g_x, g_t, *g_block) = mx.vjp(block(i), [inputs.pop(), temb, *[trainable[k] for k in block_keys[i]]], [g_x])
        if head_keys:  # the time embedding's cotangent is for the stage before the blocks: left unevaluated without one
            g_temb = g_temb + g_t
            mx.eval(g_temb)
        mx.eval(g_x, g_block)
        after_stage()
        grads.update(zip(block_keys[i], g_block))
    if head_keys:
        _, g_head = mx.vjp(start, [trainable[k] for k in head_keys], [g_x, g_temb])
        mx.eval(g_head)
        grads.update(zip(head_keys, g_head))
    return loss, {k: grads[k] for k in trainable}
