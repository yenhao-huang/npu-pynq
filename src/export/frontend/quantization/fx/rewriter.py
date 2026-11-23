"""The Rewriter: scheme-agnostic graph surgery from a plan (layer key -> QuantizedOp).

It applies the plan to every selected node of each program and re-captures
the programs; it decides nothing about numerics and knows no scheme.
"""

from __future__ import annotations

import torch
from torch import fx

from src.export.frontend.quantization.fx.utils.share import QuantizedOp, layer_key, parse, select


class Rewriter:
    """Replaces every node of ``ops`` in each program with its plan entry; re-captures."""

    def __init__(self, ops, plan: dict[tuple, QuantizedOp]) -> None:
        self.ops = tuple(ops)          # the op names whose nodes are replaced
        self.plan = plan               # layer key -> QuantizedOp
        self.gm: fx.GraphModule | None = None   # the module being rewritten

    def run(self, programs: dict[str, torch.export.ExportedProgram]
            ) -> dict[str, torch.export.ExportedProgram]:
        out = {}
        for name, ep in programs.items():
            self.gm = ep.module()
            for node in select(self.gm, self.ops):
                self.rewrite(node)
            self.finish()
            args, kwargs = ep.example_inputs
            with torch.no_grad():
                out[name] = torch.export.export(self.gm, args, kwargs, strict=False)
        return out

    def rewrite(self, node: fx.Node) -> None:
        """Replace ``node`` with its plan entry's function, its tensors read as buffers."""
        kind, args = parse(node)
        key = layer_key(self.gm, kind, args)
        if key not in self.plan:
            raise KeyError(f'no plan entry for {kind} weight {args["weight"].target}; '
                           'the plan must cover every node of the ops being replaced')
        replacement = self.plan[key]
        weight = args['weight'].target
        with self.gm.graph.inserting_before(node):
            buffers = tuple(self._buffer(f'{weight}_{kind}_{i}', t) for i, t in enumerate(replacement.tensors))
            new = self.gm.graph.call_function(replacement.function, replacement.arguments(args, buffers))
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
