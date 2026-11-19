"""The quantization pass: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8')

export_graph captures each entry point before any decomposition, so every
linear, convolution and embedding is still one aten.linear, aten.conv2d or
aten.embedding node whose weight is a get_attr of the unlifted module
(``ep.module()``). For each such node the pass quantizes the weight, stores the
INT8 weight and its scales as buffers, and replaces the node with a call to the
scheme's function (``w8a8.linear`` ...). torch.export then re-captures the
module, tracing that function into plain ATen ops.

Weights are quantized once per tensor across all programs: decode and prefill
share every weight, and giving both the same quantized tensors keeps to_mlir
freezing one constant per weight.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import fx

from src.export.frontend.quantization import w8a8

aten = torch.ops.aten


def _attr(gm: fx.GraphModule, target: str) -> torch.Tensor:
    obj = gm
    for part in target.split('.'):
        obj = getattr(obj, part)
    return obj


class _Pass:
    """One scheme applied to one GraphModule, with a cache shared by every program."""

    def __init__(self, scheme, cache: dict) -> None:
        self.scheme, self.cache = scheme, cache

    def run(self, gm: fx.GraphModule) -> fx.GraphModule:
        self.gm, self.names = gm, {}
        rewrites: dict[object, Callable[[fx.Node], fx.Node]] = {
            aten.linear.default: self.linear,
            aten.conv2d.default: self.conv2d,
            aten.embedding.default: self.embedding,
        }
        for node in list(gm.graph.nodes):
            rewrite = rewrites.get(node.target) if node.op == 'call_function' else None
            if rewrite is None:
                continue
            with gm.graph.inserting_before(node):
                new = rewrite(node)
            node.replace_all_uses_with(new)
            gm.graph.erase_node(node)
        gm.graph.eliminate_dead_code()
        self.drop_unused_weights()
        gm.recompile()
        return gm

    # --- one rewrite per op -------------------------------------------------

    def linear(self, node: fx.Node) -> fx.Node:
        x, w, b = list(node.args) + [None] * (3 - len(node.args))   # aten.linear(x, weight, bias=None)
        qw, s = self.quantized('linear', w, self.scheme.prepare_linear)
        return self.gm.graph.call_function(self.scheme.linear, (x, qw, s, b))

    def conv2d(self, node: fx.Node) -> fx.Node:
        # aten.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        defaults = [None, None, None, [1, 1], [0, 0], [1, 1], 1]
        x, w, b, stride, padding, dilation, groups = list(node.args) + defaults[len(node.args):]
        if groups != 1 or tuple(dilation) != (1, 1):
            raise ValueError('w8a8 supports ungrouped, undilated convolutions only')
        kernel = tuple(_attr(self.gm, w.target).shape[2:])
        qw, s = self.quantized('conv2d', w, self.scheme.prepare_conv2d)
        return self.gm.graph.call_function(self.scheme.conv2d,
                                           (x, qw, s, b, kernel, tuple(stride), tuple(padding)))

    def embedding(self, node: fx.Node) -> fx.Node:
        w, ids = node.args[:2]
        qw, s = self.quantized('embedding', w, self.scheme.prepare_embedding)
        return self.gm.graph.call_function(self.scheme.embedding, (ids, qw, s))

    # --- weights ------------------------------------------------------------

    def quantized(self, kind: str, weight: fx.Node, prepare) -> tuple[fx.Node, fx.Node]:
        """get_attr nodes for the quantized ``weight``, quantizing it only the first time."""
        if weight.op != 'get_attr':
            raise ValueError(f'{kind} weight {weight.name} is not a constant of the model')
        t = _attr(self.gm, weight.target)
        # The same parameter reaches every program as the same storage, so the
        # cache key is where its data lives, not the module path it has here.
        key = (kind, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))
        if key not in self.cache:
            self.cache[key] = prepare(t)
        return tuple(self.buffer(f'{weight.target}_{kind}_{i}', q) for i, q in enumerate(self.cache[key]))

    def buffer(self, hint: str, t: torch.Tensor) -> fx.Node:
        if id(t) not in self.names:
            name = hint.replace('.', '_')
            self.gm.register_buffer(name, t)
            self.names[id(t)] = name
        return self.gm.graph.get_attr(self.names[id(t)])

    def drop_unused_weights(self) -> None:
        """Remove the float weights no node reads any more, so re-capture does not keep them."""
        used = {n.target for n in self.gm.graph.nodes if n.op == 'get_attr'}
        # remove_duplicate=False lists every name of a tied weight (lm_head and the embedding).
        for name, _ in [*self.gm.named_parameters(remove_duplicate=False), *self.gm.named_buffers(remove_duplicate=False)]:
            if name not in used:
                owner, _, leaf = name.rpartition('.')
                module = self.gm.get_submodule(owner) if owner else self.gm
                delattr(module, leaf)


SCHEMES = {'w8a8': w8a8}


def quantize(programs: dict[str, torch.export.ExportedProgram], scheme: str
             ) -> dict[str, torch.export.ExportedProgram]:
    """Quantize every program with ``scheme``; return them re-captured, undecomposed."""
    try:
        ops = SCHEMES[scheme]
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; known: {sorted(SCHEMES)}') from None
    cache: dict = {}
    out = {}
    for name, ep in programs.items():
        gm = _Pass(ops, cache).run(ep.module())
        args, kwargs = ep.example_inputs
        with torch.no_grad():
            out[name] = torch.export.export(gm, args, kwargs, strict=False)
    return out
