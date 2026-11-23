"""QuantizedLayer, what a scheme implements, and Quantizer, the loop every scheme shares.

A scheme subclasses QuantizedLayer in fx/quantizer/<scheme>.py and overrides
one method per op it supports, named after the op: the float weight (and,
for an op whose input is calibrated, its largest |input|) -> the layer's
Replacement. ``check`` is an optional hook that refuses a node. Quantizer
runs a QuantizedLayer over every program and returns the plan.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.utils.share import Replacement, attr, layer_key, parse, select


class QuantizedLayer:
    """How one scheme quantizes each op. Each method: float weight (+ activation amax) -> Replacement."""

    name = ''   # the scheme's name, for error messages

    def linear(self, weight: torch.Tensor, amax: float) -> Replacement:
        raise NotImplementedError('linear')

    def conv2d(self, weight: torch.Tensor, amax: float) -> Replacement:
        raise NotImplementedError('conv2d')

    def embedding(self, weight: torch.Tensor) -> Replacement:
        raise NotImplementedError('embedding')

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node the scheme cannot compute. Accepts everything unless overridden."""


class Quantizer:
    """Runs a scheme's QuantizedLayer over every program: checks each node, quantizes each layer once."""

    def __init__(self, layer: QuantizedLayer, ops, calibrated_ops, stats: dict) -> None:
        missing = [op for op in ops
                   if getattr(type(layer), op, None) is getattr(QuantizedLayer, op, object())]
        if missing:
            raise ValueError(f'{type(layer).__name__} has no method for op(s) {missing}')
        self.layer = layer
        self.ops = tuple(ops)                        # the scheme's op names (schemes.py's ``ops``)
        self.calibrated_ops = tuple(calibrated_ops)  # those whose input has a calibrated scale
        self.stats = stats                           # layer key -> largest |input|, from the Calibrator

    def run(self, programs: dict[str, torch.export.ExportedProgram]) -> dict[tuple, Replacement]:
        """Check every node of ``ops`` in every program, quantize each layer once; return the plan.

        A weight shared by several programs, or by several nodes, has one
        layer key and is quantized once."""
        plan: dict[tuple, Replacement] = {}
        for ep in programs.values():
            gm = ep.module()
            for node in select(gm, self.ops):
                kind, args = parse(node)
                self.layer.check(kind, args)
                key = layer_key(gm, kind, args)
                if key in plan:
                    continue
                name = args['weight'].target
                weight = attr(gm, name)
                method = getattr(self.layer, kind)
                if kind not in self.calibrated_ops:
                    plan[key] = method(weight)
                    continue
                if key not in self.stats:
                    raise ValueError(f'no calibration statistics for {kind} weight {name}; '
                                     f'{self.layer.name} needs calibration inputs that reach every layer')
                plan[key] = method(weight, self.stats[key])
        return plan
