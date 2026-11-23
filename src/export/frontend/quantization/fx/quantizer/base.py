"""QuantizedLayer, what a scheme implements, and Quantizer, the loop every scheme shares.

Quantization only: the scales come from the Calibrator. A scheme subclasses
QuantizedLayer in fx/quantizer/<scheme>.py and overrides one method per op it
supports, named after the op: (float weight, its Scales) -> the layer's
QuantizedOp, the weight laid out for the op and rounded with the given scale.
``check`` is an optional hook that refuses a node. Quantizer runs a
QuantizedLayer over every program and returns the plan.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.utils.share import QuantizedOp, Scales, attr, layer_key, parse, select


class QuantizedLayer:
    """How one scheme quantizes each op. Each method: float weight and its Scales -> QuantizedOp."""

    def linear(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        raise NotImplementedError('linear')

    def conv2d(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        raise NotImplementedError('conv2d')

    def embedding(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        raise NotImplementedError('embedding')

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node the scheme cannot compute. Accepts everything unless overridden."""


class Quantizer:
    """Runs a scheme's QuantizedLayer over every program: checks each node, quantizes each layer once."""

    def __init__(self, layer: QuantizedLayer, ops, scales: dict[tuple, Scales]) -> None:
        missing = [op for op in ops
                   if getattr(type(layer), op, None) is getattr(QuantizedLayer, op, object())]
        if missing:
            raise ValueError(f'{type(layer).__name__} has no method for op(s) {missing}')
        self.layer = layer
        self.ops = tuple(ops)          # the scheme's op names (schemes.py's ``ops``)
        self.scales = scales           # layer key -> Scales, from the Calibrator

    def run(self, programs: dict[str, torch.export.ExportedProgram]) -> dict[tuple, QuantizedOp]:
        """Check every node of ``ops`` in every program, quantize each layer once; return the plan.

        A weight shared by several programs, or by several nodes, has one
        layer key and is quantized once."""
        plan: dict[tuple, QuantizedOp] = {}
        for ep in programs.values():
            gm = ep.module()
            for node in select(gm, self.ops):
                kind, args = parse(node)
                self.layer.check(kind, args)
                key = layer_key(gm, kind, args)
                if key in plan:
                    continue
                name = args['weight'].target
                if key not in self.scales:
                    raise KeyError(f'no Scales for {kind} weight {name}; '
                                   'the Calibrator must cover every layer the Quantizer quantizes')
                plan[key] = getattr(self.layer, kind)(attr(gm, name), self.scales[key])
        return plan
