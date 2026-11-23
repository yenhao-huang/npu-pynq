"""The Quantizer base class: the loop every scheme shares, and the method a scheme implements.

A scheme subclasses Quantizer in fx/quantizer/<scheme>.py and implements
``quantize_layer``, which decides every number of one layer and returns its
Replacement; ``check`` is an optional hook that refuses a node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.export.frontend.quantization.fx.utils.share import Replacement, attr, layer_key, parse, select


class Quantizer(ABC):
    """One quantization scheme: decides every number, does no graph surgery."""

    def __init__(self, ops, calibrated_ops, stats: dict) -> None:
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
                self.check(kind, args)
                key = layer_key(gm, kind, args)
                if key not in plan:
                    name = args['weight'].target
                    plan[key] = self.quantize_layer(key, kind, attr(gm, name), name)
        return plan

    @abstractmethod
    def quantize_layer(self, key: tuple, kind: str, weight: torch.Tensor, name: str) -> Replacement:
        """The Replacement of a ``kind`` layer: its float ``weight`` (get_attr ``name``)
        quantized, with its activation scale from ``self.stats[key]`` if it has one."""

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node the scheme cannot compute. Accepts everything unless overridden."""
