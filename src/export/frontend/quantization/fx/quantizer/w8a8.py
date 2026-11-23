"""W8A8: per-output-channel INT8 weights, static per-tensor INT8 activations.

The scheme's parameters are schemes.W8A8. Weights, per output channel, and
activations, one scale per layer input, are both symmetric with zero point 0:

    scale = max(amax, EPS) / 127    EPS avoids dividing by zero on an all-zero channel or input
    q     = round(value / scale)    half to even (torch.round), clamped to [-127, 127]

A weight's amax is its channel's; an activation's is the largest |input| the
Calibrator recorded for the layer.

W8A8Quantizer quantizes every selected layer's weight once and fixes its
activation scale (a QuantizedLayer); the Rewriter then replaces each node with
the call ``replacement`` builds, to ``linear``, ``conv2d`` or ``embedding``
below. The re-export traces those functions; a contraction becomes

    xq  = quant.quantize_per_tensor(x, x_scale)    activations to INT8, constant [1] scale
    acc = aten._int_mm(xq, w)                      INT8 x INT8 -> INT32, constant [K, N] weight
    y   = quant.dequantize(acc, x_scale, w_scale)
    y   = y + bias

with the custom ops of fx/custom_ops.py. The embedding is a lookup in the INT8
table, in plain ATen ops, and has no activation scale.

Overflow: |acc| <= K * 127 * 127, which stays below 2**31 while K is under about
133,000, so an INT32 accumulator cannot overflow or saturate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch

from src.export.frontend.quantization.fx import custom_ops
from src.export.frontend.quantization.fx.rewriter import attr, layer_key, parse, select
from src.export.frontend.quantization.schemes import W8A8

EPS = W8A8['eps']
QMAX = W8A8['activation_qmax']


class QuantizedLayer(NamedTuple):
    """One weight's quantized tensors, as W8A8Quantizer makes them and the Rewriter reads them."""
    weight: torch.Tensor                        # INT8
    weight_scale: torch.Tensor                  # float32, per output channel (per row for a table)
    activation_scale: torch.Tensor | None = None   # [1] float32; None when the input is not quantized

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The tensors that become buffers, in this order."""
        return tuple(t for t in self if t is not None)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] float -> ([in, out] int8, [out] float32 scale), symmetric per output channel.

    Stored [K, N] so the contraction's right operand is the weight itself."""
    w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
    qmax = W8A8['weight_qmax']
    scale = torch.clamp(w.abs().amax(0), min=EPS) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def activation_scale(amax: float) -> torch.Tensor:
    """The calibrated max |input| of a layer -> its [1] float32 activation scale."""
    return torch.clamp(torch.tensor([amax], dtype=torch.float32), min=EPS) / QMAX


# --- the replacements, traced into the graph, each with its weight layout ---

def _contract(rows: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
              bias: torch.Tensor | None) -> torch.Tensor:
    """[M, K] float rows -> [M, N]: (int8(rows) @ W_q) * x_scale * w_scale + bias."""
    xq = custom_ops.quantize_per_tensor(rows, x_scale, QMAX)
    acc = torch._int_mm(xq, qweight)
    y = custom_ops.dequantize(acc, x_scale, scale)
    if bias is not None:
        y = y + bias
    return y


def linear(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
           bias: torch.Tensor | None) -> torch.Tensor:
    """aten.linear in W8A8; ``qweight`` is ``quantize_weight``'s [K, N]."""
    shape = x.shape
    k, n = qweight.shape
    # The contraction is rank 2.
    y = _contract(x.reshape(-1, k).to(torch.float32), qweight, scale, x_scale, bias)
    return y.reshape(*shape[:-1], n)


def quantize_conv2d_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[Cout, Cin, kh, kw] float -> ([kh * kw * Cin, Cout] int8, [Cout] scale).

    K follows the (ky, kx, c) order that ``conv2d``'s im2col uses."""
    w = weight.detach().permute(0, 2, 3, 1).reshape(weight.shape[0], -1)   # [Cout, (ky, kx, c)]
    return quantize_weight(w)


def conv2d(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, x_scale: torch.Tensor,
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
    # K must follow the (ky, kx, c) order used in quantize_conv2d_weight.
    # Otherwise the im2col columns no longer line up with the weight's rows.
    patches = [x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw]
               for ky in range(kh) for kx in range(kw)]
    cols = torch.stack(patches, dim=1).permute(0, 3, 4, 1, 2).reshape(n * oh * ow, kh * kw * c)
    y = _contract(cols, qweight, scale, x_scale, bias)
    return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2)


def quantize_embedding_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[V, H] table -> ([V, H] int8, [V] scale): one scale per row."""
    q, s = quantize_weight(weight)      # per output channel of [V, H].T = per row
    return q.t().contiguous(), s


def embedding(ids: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Rows of the INT8 table, rescaled. A table lookup, not a contraction."""
    rows = qweight.index_select(0, ids.reshape(-1)).to(torch.float32)
    out = rows * scale.index_select(0, ids.reshape(-1))[:, None]
    return out.reshape(*ids.shape, qweight.shape[1])


# --- per op: how W8A8 quantizes it and what replaces it ---------------------

@dataclass(frozen=True)
class Op:
    quantize_weight: Callable      # float weight -> (INT8 weight, weight scale)
    calibrated: bool               # its input is quantized, so it needs an activation scale
    call: Callable                 # (args, buffers, float weight) -> (function, its arguments)


OPS = {
    'linear': Op(quantize_weight, True,
                 lambda args, buffers, weight: (linear, (args['x'], *buffers, args['bias']))),
    'conv2d': Op(quantize_conv2d_weight, True,
                 lambda args, buffers, weight: (conv2d, (args['x'], *buffers, args['bias'], tuple(weight.shape[2:]),
                                                         args['stride'], args['padding']))),
    'embedding': Op(quantize_embedding_weight, False,
                    lambda args, buffers, weight: (embedding, (args['ids'], *buffers))),
}


class W8A8Quantizer:
    """Quantizes every selected layer once and holds the results, by layer key."""

    def __init__(self, ops, stats: dict) -> None:
        self.ops = tuple(ops)          # the scheme's op names (schemes.W8A8['ops'])
        self.stats = stats             # layer key -> largest |input|, from the Calibrator
        self.layers: dict[tuple, QuantizedLayer] = {}

    @staticmethod
    def calibrated_ops(ops) -> tuple[str, ...]:
        """The op names among ``ops`` whose input is quantized, and so calibrated."""
        return tuple(op for op in ops if OPS[op].calibrated)

    def run(self, programs: dict[str, torch.export.ExportedProgram]) -> None:
        """Quantize the weight of every node of ``ops`` in every program, once per layer key.

        A weight shared by several programs, or by several nodes, is quantized
        once; a tied weight two kinds of op use has one entry per kind."""
        for ep in programs.values():
            gm = ep.module()
            for node in select(gm, self.ops):
                kind, args = parse(node)
                self.check(kind, args)
                key = layer_key(gm, node)
                if key not in self.layers:
                    self.layers[key] = self.quantize_layer(key, kind, attr(gm, args['weight'].target),
                                                           args['weight'].target)

    @staticmethod
    def check(kind: str, args: dict) -> None:
        """Refuse a node W8A8 cannot compute."""
        if kind == 'conv2d' and (args['groups'] != 1 or args['dilation'] != (1, 1)):
            raise ValueError('w8a8 supports ungrouped, undilated convolutions only')

    def quantize_layer(self, key: tuple, kind: str, weight: torch.Tensor, name: str) -> QuantizedLayer:
        """``weight`` quantized for ``kind``, plus its activation scale when the op is calibrated."""
        q, scale = OPS[kind].quantize_weight(weight)
        if not OPS[kind].calibrated:
            return QuantizedLayer(q, scale)
        if key not in self.stats:
            raise ValueError(f'no calibration statistics for {kind} weight {name}; '
                             'w8a8 needs calibration inputs that reach every layer')
        return QuantizedLayer(q, scale, activation_scale(self.stats[key]))

    def tensors(self, key: tuple) -> tuple[torch.Tensor, ...]:
        """The tensors of the layer ``key``, in the order they become buffers."""
        if key not in self.layers:
            raise KeyError(f'layer {key[0]} was not quantized; run must cover every node the Rewriter replaces')
        return self.layers[key].tensors()

    @staticmethod
    def replacement(kind: str, args: dict, buffers: tuple, weight: torch.Tensor) -> tuple[Callable, tuple]:
        """The function that replaces a ``kind`` node and its arguments, ``buffers`` in tensors' order."""
        return OPS[kind].call(args, buffers, weight)
