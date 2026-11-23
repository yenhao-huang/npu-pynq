"""W8A8: per-output-channel INT8 weights, static per-tensor INT8 activations.

The scheme's parameters are schemes.W8A8. Weights, per output channel, and
activations, one scale per layer input, are both symmetric with zero point 0:

    scale = max(amax, EPS) / 127    EPS avoids dividing by zero on an all-zero channel or input
    q     = round(value / scale)    half to even (torch.round), clamped to [-127, 127]

A weight's amax is its channel's; an activation's is the largest |input| the
Calibrator recorded for the layer.

W8A8QuantizedLayer quantizes each layer's weight, fixes its activation scale
and returns its Replacement: a call to one of its ``_*_graph`` functions,
which the re-export traces. A contraction becomes

    xq  = quant.quantize_per_tensor(x, x_scale)    activations to INT8, constant [1] scale
    acc = aten._int_mm(xq, w)                      INT8 x INT8 -> INT32, constant [K, N] weight
    y   = quant.dequantize(acc, x_scale, w_scale)
    y   = y + bias

with the custom ops of fx/custom_ops.py. The embedding is a lookup in the INT8
table, in plain ATen ops, and has no activation scale.

Overflow: |acc| <= K * 127 * 127, which stays below 2**31 while K is under about
133,000, so an INT32 accumulator cannot overflow or saturate.

This module holds only the class: every helper is one of its members.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx import custom_ops
from src.export.frontend.quantization.fx.quantizer.base import QuantizedLayer
from src.export.frontend.quantization.fx.utils.share import Replacement
from src.export.frontend.quantization.schemes import W8A8


class W8A8QuantizedLayer(QuantizedLayer):
    """W8A8's numbers: per-channel INT8 weights and a calibrated [1] scale per layer input."""

    name = W8A8['name']
    EPS = W8A8['eps']
    QMAX = W8A8['activation_qmax']
    WEIGHT_QMAX = W8A8['weight_qmax']

    # --- the ops ----------------------------------------------------------------

    def linear(self, weight: torch.Tensor, amax: float) -> Replacement:
        """aten.linear: [out, in] float weight -> [K, N] = [in, out] INT8, so the
        contraction's right operand is the weight itself; per-output-channel scale."""
        w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
        qmax = self.WEIGHT_QMAX
        scale = torch.clamp(w.abs().amax(0), min=self.EPS) / qmax
        q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
        tensors = (q, scale.to(torch.float32), self._activation_scale(amax))
        return Replacement(self._linear_graph, tensors,
                           lambda args, buffers: (args['x'], *buffers, args['bias']))

    def conv2d(self, weight: torch.Tensor, amax: float) -> Replacement:
        """aten.conv2d: [Cout, Cin, kh, kw] float weight -> [kh * kw * Cin, Cout] INT8.

        K follows the (ky, kx, c) order that ``_conv2d_graph``'s im2col uses."""
        w = weight.detach().permute(0, 2, 3, 1).reshape(weight.shape[0], -1)   # [Cout, (ky, kx, c)]
        q, scale = self._per_channel(w)
        kernel = tuple(weight.shape[2:])
        return Replacement(self._conv2d_graph, (q, scale, self._activation_scale(amax)),
                           lambda args, buffers: (args['x'], *buffers, args['bias'], kernel,
                                                  args['stride'], args['padding']))

    def embedding(self, weight: torch.Tensor) -> Replacement:
        """aten.embedding: [V, H] float table -> [V, H] INT8 with one scale per row."""
        q, scale = self._per_channel(weight)      # per output channel of [V, H].T = per row
        return Replacement(self._embedding_graph, (q.t().contiguous(), scale),
                           lambda args, buffers: (args['ids'], *buffers))

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node W8A8 cannot compute."""
        if kind == 'conv2d' and (args['groups'] != 1 or args['dilation'] != (1, 1)):
            raise ValueError('w8a8 supports ungrouped, undilated convolutions only')

    # --- quantization shared by the ops ---------------------------------------------

    @classmethod
    def _per_channel(cls, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[out, in] float -> ([in, out] int8, [out] float32 scale), symmetric per output channel.

        ``linear`` writes the same arithmetic out in full."""
        w = weight.detach().to(torch.float32).t().contiguous()
        scale = torch.clamp(w.abs().amax(0), min=cls.EPS) / cls.WEIGHT_QMAX
        q = torch.clamp(torch.round(w / scale), -cls.WEIGHT_QMAX, cls.WEIGHT_QMAX).to(torch.int8)
        return q, scale.to(torch.float32)

    @classmethod
    def _activation_scale(cls, amax: float) -> torch.Tensor:
        """The calibrated max |input| of a layer -> its [1] float32 activation scale."""
        return torch.clamp(torch.tensor([amax], dtype=torch.float32), min=cls.EPS) / cls.QMAX

    # --- the replacements, traced into the graph ---------------------------------

    @staticmethod
    def _contract(rows: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
                  bias: torch.Tensor | None) -> torch.Tensor:
        """[M, K] float rows -> [M, N]: (int8(rows) @ W_q) * x_scale * w_scale + bias."""
        xq = custom_ops.quantize_per_tensor(rows, x_scale, W8A8QuantizedLayer.QMAX)
        acc = torch._int_mm(xq, qweight)
        y = custom_ops.dequantize(acc, x_scale, scale)
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def _linear_graph(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
                      bias: torch.Tensor | None) -> torch.Tensor:
        """aten.linear in W8A8; ``qweight`` is ``linear``'s [K, N]."""
        shape = x.shape
        k, n = qweight.shape
        # The contraction is rank 2.
        y = W8A8QuantizedLayer._contract(x.reshape(-1, k).to(torch.float32), qweight, scale, x_scale, bias)
        return y.reshape(*shape[:-1], n)

    @staticmethod
    def _conv2d_graph(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
                      bias: torch.Tensor | None, kernel: tuple[int, int], stride: tuple[int, int],
                      padding: tuple[int, int]) -> torch.Tensor:
        """A convolution as im2col and one W8A8 contraction.

        Rows are output pixels and K is kh * kw * Cin in (ky, kx, c) order.
        Batch 1, no groups or dilation.
        """
        (kh, kw), (sh, sw), (ph, pw) = kernel, stride, padding
        cout = qweight.shape[1]
        x = torch.nn.functional.pad(x.to(torch.float32), (pw, pw, ph, ph))
        n, c, h, w = x.shape
        oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
        # K must follow the (ky, kx, c) order ``conv2d`` lays the weight out in.
        # Otherwise the im2col columns no longer line up with the weight's rows.
        patches = [x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw]
                   for ky in range(kh) for kx in range(kw)]
        cols = torch.stack(patches, dim=1).permute(0, 3, 4, 1, 2).reshape(n * oh * ow, kh * kw * c)
        y = W8A8QuantizedLayer._contract(cols, qweight, scale, x_scale, bias)
        return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2)

    @staticmethod
    def _embedding_graph(ids: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Rows of the INT8 table, rescaled. A table lookup, not a contraction."""
        rows = qweight.index_select(0, ids.reshape(-1)).to(torch.float32)
        out = rows * scale.index_select(0, ids.reshape(-1))[:, None]
        return out.reshape(*ids.shape, qweight.shape[1])
