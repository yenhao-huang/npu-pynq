"""W8A8: per-output-channel INT8 weights, static per-tensor INT8 activations.

The scheme's parameters are schemes.W8A8. The Calibrator computes the scales
(per output channel for weights, one [1] scale per layer input for
activations); W8A8QuantizedLayer only lays each weight out for its op and
rounds it with its scale, symmetric with zero point 0:

    q = clamp(round(weight / weight_scale), -127, 127)    half to even (torch.round)

and returns the layer's QuantizedOp: a call to one of its ``_*_graph``
functions, which the re-export traces. A contraction becomes

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
from src.export.frontend.quantization.fx.utils.share import QuantizedOp, Scales
from src.export.frontend.quantization.schemes import W8A8


class W8A8QuantizedLayer(QuantizedLayer):
    """W8A8's layouts and rounding: per-channel INT8 weights; inputs quantized with a [1] scale in the graph."""

    QMAX = W8A8['activation_qmax']
    WEIGHT_QMAX = W8A8['weight_qmax']

    # --- the ops ----------------------------------------------------------------

    def linear(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        """aten.linear: [out, in] float weight -> [K, N] = [in, out] INT8, so the
        contraction's right operand is the weight itself; one scale per output channel N."""
        w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
        qmax = self.WEIGHT_QMAX
        q = torch.clamp(torch.round(w / scales.weight_scale), -qmax, qmax).to(torch.int8)
        return QuantizedOp(self._linear_graph, (q, scales.weight_scale, scales.activation_scale),
                           lambda args, buffers: (args['x'], *buffers, args['bias']))

    def conv2d(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        """aten.conv2d: [Cout, Cin, kh, kw] float weight -> [kh * kw * Cin, Cout] INT8.

        K follows the (ky, kx, c) order that ``_conv2d_graph``'s im2col uses."""
        w = weight.detach().permute(0, 2, 3, 1).reshape(weight.shape[0], -1)   # [Cout, (ky, kx, c)]
        q = self._round(w.to(torch.float32).t().contiguous(), scales.weight_scale)   # [K, Cout]
        kernel = tuple(weight.shape[2:])
        return QuantizedOp(self._conv2d_graph, (q, scales.weight_scale, scales.activation_scale),
                           lambda args, buffers: (args['x'], *buffers, args['bias'], kernel,
                                                  args['stride'], args['padding']))

    def embedding(self, weight: torch.Tensor, scales: Scales) -> QuantizedOp:
        """aten.embedding: [V, H] float table -> [V, H] INT8 with one scale per row."""
        q = self._round(weight.detach().to(torch.float32), scales.weight_scale[:, None])
        return QuantizedOp(self._embedding_graph, (q, scales.weight_scale),
                           lambda args, buffers: (args['ids'], *buffers))

    def check(self, kind: str, args: dict) -> None:
        """Refuse a node W8A8 cannot compute."""
        if kind == 'conv2d' and (args['groups'] != 1 or args['dilation'] != (1, 1)):
            raise ValueError('w8a8 supports ungrouped, undilated convolutions only')

    @classmethod
    def _round(cls, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Float ``w`` -> INT8 with ``scale`` broadcast over it; ``linear`` writes this out in full."""
        return torch.clamp(torch.round(w / scale), -cls.WEIGHT_QMAX, cls.WEIGHT_QMAX).to(torch.int8)

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
