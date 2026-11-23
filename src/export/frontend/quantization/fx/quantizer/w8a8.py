"""W8A8: per-output-channel INT8 weights, static per-tensor INT8 activations.

The scheme's parameters are schemes.W8A8. The Calibrator computes the scales
(per output channel for weights, one [1] scale per layer input for
activations); W8A8QuantizedLayer only lays each weight out for its op and
rounds it with its scale, symmetric with zero point 0:

    q = clamp(round(weight / weight_scale), -127, 127)    half to even (torch.round)

and returns the layer's QuantizedOp: the INT8 weight in the layout the
Rewriter's integer form of the op reads ([K, N] for linear and conv2d, [V, H]
for an embedding table), its scale and the layer's activation scale.

Overflow: the Rewriter's aten._int_mm accumulates |acc| <= K * 127 * 127, which
stays below 2**31 while K is under about 133,000, so an INT32 accumulator
cannot overflow or saturate.

This module holds only the class: every helper is one of its members.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.quantizer.base import QuantizedLayer
from src.export.frontend.quantization.fx.utils.share import QuantizedOp, Scales
from src.export.frontend.quantization.schemes import W8A8


class W8A8QuantizedLayer(QuantizedLayer):
    """W8A8's weight layouts and rounding: per-channel INT8 weights."""

    WEIGHT_QMAX = W8A8['weight_qmax']

    # --- the ops ----------------------------------------------------------------

    def linear(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        """aten.linear: [out, in] float weight -> [K, N] = [in, out] INT8, so the
        contraction's right operand is the weight itself; one scale per output channel N."""
        w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
        qmax = self.WEIGHT_QMAX
        q = torch.clamp(torch.round(w / scales.weight_scale), -qmax, qmax).to(torch.int8)
        return QuantizedOp(q, scales.weight_scale, scales.activation_scale)

    def conv2d(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        """aten.conv2d: [Cout, Cin, kh, kw] float weight -> [kh * kw * Cin, Cout] INT8.

        K follows the (ky, kx, c) order of the Rewriter's im2col (rewriter._im2col)."""
        w = weight.detach().permute(0, 2, 3, 1).reshape(weight.shape[0], -1)   # [Cout, (ky, kx, c)]
        q = self._round(w.to(torch.float32).t().contiguous(), scales.weight_scale)   # [K, Cout]
        return QuantizedOp(q, scales.weight_scale, scales.activation_scale)

    def embedding(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        """aten.embedding: [V, H] float table -> [V, H] INT8 with one scale per row."""
        q = self._round(weight.detach().to(torch.float32), scales.weight_scale[:, None])
        return QuantizedOp(q, scales.weight_scale)

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node W8A8 cannot compute."""
        if kind == 'conv2d' and (args['groups'] != 1 or args['dilation'] != (1, 1)):
            raise ValueError('w8a8 supports ungrouped, undilated convolutions only')

    @classmethod
    def _round(cls, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Float ``w`` -> INT8 with ``scale`` broadcast over it; ``linear`` writes this out in full."""
        return torch.clamp(torch.round(w / scale), -cls.WEIGHT_QMAX, cls.WEIGHT_QMAX).to(torch.int8)
