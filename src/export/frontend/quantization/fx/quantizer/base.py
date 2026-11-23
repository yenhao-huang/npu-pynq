"""The Quantizer base class: what every quantization scheme implements, and what it shares.

A scheme decides every number and does no graph surgery. Which ops it
quantizes, and which of them have a calibrated input, are data in
schemes.py (``ops``, ``calibrated_ops``). ``run`` quantizes each selected
layer once and returns the plan: layer key -> Replacement (utils/share.py), which the
Rewriter applies. A quantizer imports neither the Calibrator nor the
Rewriter; it receives statistics and returns the plan as plain data.

A scheme subclasses Quantizer in fx/quantizer/<scheme>.py and implements
``quantize_layer`` and ``replacement``; ``check`` is an optional hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch

from src.export.frontend.quantization.fx.utils.share import Replacement, attr, layer_key, parse, select


class QuantizedLayer(NamedTuple):
    """One weight's quantized tensors, as ``quantize_layer`` makes them."""
    weight: torch.Tensor                        # INT8
    weight_scale: torch.Tensor                  # float32, per output channel (per row for a table)
    activation_scale: torch.Tensor | None = None   # [1] float32; None when the input is not quantized

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The tensors that become buffers, in this order."""
        return tuple(t for t in self if t is not None)


class Quantizer(ABC):
    """One quantization scheme: decides every number, does no graph surgery."""

    def __init__(self, ops, calibrated_ops, stats: dict) -> None:
        self.ops = tuple(ops)                        # the scheme's op names (schemes.py's ``ops``)
        self.calibrated_ops = tuple(calibrated_ops)  # those whose input has a calibrated scale
        self.stats = stats                           # layer key -> largest |input|, from the Calibrator

    # --- shared ---------------------------------------------------------------

    def run(self, programs: dict[str, torch.export.ExportedProgram]) -> dict[tuple, Replacement]:
        """Quantize every node of ``ops`` in every program, once per layer key; return the plan.

        The key (``layer_key``) is the op name and where the float weight's data
        lives, before any transpose: a weight shared by several programs, or by
        several nodes, is quantized once, and a tied weight two kinds of op use
        has one entry per kind."""
        plan: dict[tuple, Replacement] = {}
        for ep in programs.values():
            gm = ep.module()
            for node in select(gm, self.ops):
                kind, args = parse(node)
                self.check(kind, args)
                key = layer_key(gm, node)
                if key not in plan:
                    name = args['weight'].target
                    weight = attr(gm, name)
                    plan[key] = self.replacement(kind, self.quantize_layer(key, kind, weight, name), weight)
        return plan

    # --- what a scheme implements ---------------------------------------------

    @abstractmethod
    def quantize_layer(self, key: tuple, kind: str, weight: torch.Tensor, name: str) -> QuantizedLayer:
        """The float ``weight`` (get_attr ``name``) of a ``kind`` layer, quantized; ``key`` finds its stats."""

    @abstractmethod
    def replacement(self, kind: str, layer: QuantizedLayer, weight: torch.Tensor) -> Replacement:
        """What replaces every ``kind`` node of the layer: its function, ``layer``'s tensors, and
        how to build the function's arguments. ``weight`` is the float weight."""

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node the scheme cannot compute. Accepts everything unless overridden."""
