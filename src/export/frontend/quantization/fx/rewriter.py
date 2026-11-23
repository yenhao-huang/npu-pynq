"""Scheme-agnostic FX graph surgery: the ATen ops a scheme can quantize, and the Rewriter.

export_graph captures each entry point undecomposed, so every op a scheme
quantizes is one aten.linear, aten.conv2d or aten.embedding node of the
unlifted module (``ep.module()``), its weight a get_attr. ``OPS`` says, per op
name (schemes.py's ``ops``), which node that is and how to read its
arguments; ``select`` and ``layer_key`` find the nodes and name their layers
for the calibrator and the quantizer.

The Rewriter replaces each selected node with the call the quantizer builds
for it, from the tensors the quantizer prepared, which become buffers of the
module; it decides nothing about numerics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


@dataclass(frozen=True)
class AtenOp:
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


def select(gm: fx.GraphModule, ops) -> list[fx.Node]:
    """The nodes of ``gm`` that the op names ``ops`` cover, in graph order."""
    targets = {OPS[op].target for op in ops}
    return [n for n in gm.graph.nodes if n.op == 'call_function' and n.target in targets]


def parse(node: fx.Node) -> tuple[str, dict]:
    """``node``'s op name and its arguments by name."""
    kind = OP_NAMES[node.target]
    return kind, OPS[kind].parse(node)


def layer_key(gm: fx.GraphModule, node: fx.Node) -> tuple:
    """The key of ``node``'s layer: its op name and where its float weight's data lives.

    Taken before any transpose or permute. The same parameter reaches every
    program as the same storage, so this names a layer's weight across
    programs, unlike its module path; a weight two kinds of op use (a tied
    lm_head and embedding) has one key per kind."""
    kind, args = parse(node)
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


class Rewriter:
    """Replaces every selected node of each program with the quantizer's call; re-captures.

    The quantizer gives the op names (``ops``), the tensors of each layer
    (``tensors(key)``) and the call that replaces a node (``replacement``)."""

    def __init__(self, quantizer) -> None:
        self.quantizer = quantizer
        self.gm: fx.GraphModule | None = None   # the module being rewritten

    def run(self, programs: dict[str, torch.export.ExportedProgram]
            ) -> dict[str, torch.export.ExportedProgram]:
        out = {}
        for name, ep in programs.items():
            self.gm = ep.module()
            for node in select(self.gm, self.quantizer.ops):
                self.rewrite(node)
            self.finish()
            args, kwargs = ep.example_inputs
            with torch.no_grad():
                out[name] = torch.export.export(self.gm, args, kwargs, strict=False)
        return out

    def rewrite(self, node: fx.Node) -> None:
        """Replace ``node`` with the quantizer's call, its tensors read as buffers."""
        kind, args = parse(node)
        weight = args['weight']
        tensors = self.quantizer.tensors(layer_key(self.gm, node))
        with self.gm.graph.inserting_before(node):
            buffers = tuple(self._buffer(f'{weight.target}_{kind}_{i}', t) for i, t in enumerate(tensors))
            fn, call_args = self.quantizer.replacement(kind, args, buffers, attr(self.gm, weight.target))
            new = self.gm.graph.call_function(fn, call_args)
        node.replace_all_uses_with(new)
        self.gm.graph.erase_node(node)

    def finish(self) -> None:
        """After every rewrite: drop dead nodes and the float weights no node reads, recompile.

        Without this, the re-export would keep the float weights."""
        gm = self.gm
        gm.graph.eliminate_dead_code()
        used = {n.target for n in gm.graph.nodes if n.op == 'get_attr'}
        # remove_duplicate=False lists every name of a tied weight (lm_head and the embedding).
        for name, _ in [*gm.named_parameters(remove_duplicate=False), *gm.named_buffers(remove_duplicate=False)]:
            if name not in used:
                owner, _, leaf = name.rpartition('.')
                delattr(gm.get_submodule(owner) if owner else gm, leaf)
        gm.recompile()

    def _buffer(self, hint: str, t: torch.Tensor) -> fx.Node:
        """A get_attr node for ``t``, registered as a buffer the first time it is used."""
        name = next((n for n, b in self.gm.named_buffers() if b is t), None)
        if name is None:
            name = hint.replace('.', '_')
            self.gm.register_buffer(name, t)
        return self.gm.graph.get_attr(name)
