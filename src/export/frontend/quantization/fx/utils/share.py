"""What the FX stages share: the ATen ops a scheme can quantize, node helpers, and the data they pass.

export_graph captures each entry point undecomposed, so every op a scheme
quantizes is one aten.linear, aten.conv2d or aten.embedding node of the
unlifted module (``ep.module()``), its weight a get_attr. ``OPS`` says, per op
name (schemes.py's ``ops``), which node that is and how to read its
arguments; ``select``, ``parse`` and ``layer_key`` find the nodes, read them
and name their layers. ``Scales`` (Calibrator -> Quantizer) and ``QuantizedOp``
(Quantizer -> Rewriter) are what the stages pass on, one per layer key.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
from torch import fx

aten = torch.ops.aten


def _linear(node: fx.Node) -> dict:
    # aten.linear(x, weight, bias=None)
    x, w, b = list(node.args) + [None] * (3 - len(node.args))
    return {'x': x, 'weight': w, 'bias': b}


def _conv2d(node: fx.Node) -> dict:
    # aten.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    defaults = [None, None, None, [1, 1], [0, 0], [1, 1], 1]
    x, w, b, stride, padding, dilation, groups = list(node.args) + defaults[len(node.args):]
    return {'x': x, 'weight': w, 'bias': b, 'stride': tuple(stride), 'padding': tuple(padding),
            'dilation': tuple(dilation), 'groups': groups}


def _embedding(node: fx.Node) -> dict:
    # aten.embedding(weight, ids, ...)
    w, ids = node.args[:2]
    return {'weight': w, 'ids': ids}


class AtenOp(NamedTuple):
    """One op a scheme can quantize, at the ATen level."""
    target: object                      # the aten node
    parse: Callable[[fx.Node], dict]    # the node -> its arguments by name; 'weight' is the get_attr
                                        # node, 'x' the input of an op whose input is quantized


OPS = {
    'linear': AtenOp(aten.linear.default, _linear),
    'conv2d': AtenOp(aten.conv2d.default, _conv2d),
    'embedding': AtenOp(aten.embedding.default, _embedding),
}
# A node's target -> its op name in OPS.
OP_NAMES = {op.target: name for name, op in OPS.items()}


class Scales(NamedTuple):
    """One layer's scales, as the Calibrator computes them and a QuantizedLayer uses them."""
    weight_scale: torch.Tensor                     # [N] float32, one per output channel (per row for a table)
    activation_scale: torch.Tensor | None = None   # [1] float32; None when the input is not quantized


class QuantizedOp(NamedTuple):
    """One layer's constants, as a QuantizedLayer makes them and the Rewriter puts them in (a plan entry)."""
    weight: torch.Tensor                           # INT8, in the layout the Rewriter's integer form reads
    weight_scale: torch.Tensor                     # [N] float32
    activation_scale: torch.Tensor | None = None   # [1] float32; None when the input is not quantized

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The tensors that become buffers {weight}_{op}_{i}, in this order."""
        return tuple(t for t in self if t is not None)


def select(gm: fx.GraphModule, ops) -> list[fx.Node]:
    """The nodes of ``gm`` that the op names ``ops`` cover, in graph order."""
    targets = {OPS[op].target for op in ops}
    return [n for n in gm.graph.nodes if n.op == 'call_function' and n.target in targets]


def parse(node: fx.Node) -> tuple[str, dict]:
    """``node``'s op name and its arguments by name."""
    kind = OP_NAMES[node.target]
    return kind, OPS[kind].parse(node)


def layer_key(gm: fx.GraphModule, kind: str, args: dict) -> tuple:
    """The key of a parsed node's layer: its op name and where its float weight's data lives.

    Taken before any transpose or permute. The same parameter reaches every
    program as the same storage, so this names a layer's weight across
    programs, unlike its module path; a weight two kinds of op use (a tied
    lm_head and embedding) has one key per kind."""
    weight = args['weight']
    if weight.op != 'get_attr':
        raise ValueError(f'{kind} weight {weight.name} is not a constant of the model')
    t = attr(gm, weight.target)
    return (kind, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))


def attr(gm: fx.GraphModule, target: str) -> torch.Tensor:
    """The tensor a get_attr ``target`` names."""
    obj = gm
    for part in target.split('.'):
        obj = getattr(obj, part)
    return obj
