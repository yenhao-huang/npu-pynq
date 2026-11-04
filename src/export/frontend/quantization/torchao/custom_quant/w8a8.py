"""W8A8: per-output-channel INT8 weights, per-token (per-row) INT8 activations.

The scheme is defined in src/vendors/npu_pynq/spec/quant.py:

    scale = max(amax, EPS) / 127    EPS avoids dividing by zero on an all-zero row or channel
    q     = round(value / scale)    half to even (torch.round), clamped to [-127, 127]

When each part is quantized:

    weights        once, in the quantization stage
    activations    on every call, in the graph, on the CPU
                   (so there is no calibration step and no stored activation scale)

The forward uses only plain ATen ops. ``torch._int_mm`` is the INT8 x INT8 -> INT32
contraction that the back end claims for the NPU.

Overflow: src/test/model defines the array's MACs, which accumulate in saturating INT32.
Here |acc| <= K * 127 * 127, which stays below 2**31 while K is under about 133,000.
Saturation is therefore unreachable, and ``torch._int_mm`` gives the same result.
"""

from __future__ import annotations

import torch
from torch import nn

from src.vendors.npu_pynq.spec.quant import W8A8

EPS = W8A8['eps']
QMAX = W8A8['activation_qmax']


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] float -> ([in, out] int8, [out] float32 scale), symmetric per output channel."""
    w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
    qmax = W8A8['weight_qmax']
    scale = torch.clamp(w.abs().amax(0), min=EPS) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] float -> ([M, K] int8, [M] float32 scale), symmetric per row."""
    scale = torch.clamp(x.abs().amax(-1), min=EPS) * (1.0 / QMAX)
    q = torch.clamp(torch.round(x * (1.0 / scale)[:, None]), -QMAX, QMAX).to(torch.int8)
    return q, scale


class W8A8Linear(nn.Module):
    """y = (int8(x) @ W_q) * x_scale * w_scale + bias, the contraction on the NPU."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.in_features, self.out_features = qweight.shape
        # Stored [K, N] so the contraction's right operand is the weight itself.
        # It must stay a buffer. to_mlir freezes buffers into constants, and --npu-claim
        # only takes contractions on a constant weight.
        self.register_buffer('qweight', qweight)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', None if bias is None else bias.detach().to(torch.float32))

    @classmethod
    def from_float(cls, linear: nn.Linear) -> 'W8A8Linear':
        q, s = quantize_weight(linear.weight)
        return cls(q, s, linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        # --npu-claim only takes rank-2 operands of static shape.
        # Flattening to rows also gives every token its own activation scale.
        x2 = x.reshape(-1, self.in_features).to(torch.float32)
        xq, xs = quantize_rows(x2)
        acc = torch._int_mm(xq, self.qweight)
        y = acc.to(torch.float32) * xs[:, None] * self.scale[None, :]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, scheme=w8a8'


class W8A8Embedding(nn.Module):
    """Embedding rows stored as INT8 with one scale per row (per vocabulary entry).

    This is a table lookup, not a contraction, so it runs on the CPU.
    """

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = qweight.shape
        self.register_buffer('qweight', qweight)
        self.register_buffer('scale', scale)

    @classmethod
    def from_float(cls, emb: nn.Embedding) -> 'W8A8Embedding':
        q, s = quantize_weight(emb.weight)      # per output channel of [V, H].T = per row
        return cls(q.t().contiguous(), s)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.qweight.index_select(0, ids.reshape(-1)).to(torch.float32)
        out = rows * self.scale.index_select(0, ids.reshape(-1))[:, None]
        return out.reshape(*ids.shape, self.embedding_dim)


class W8A8Conv2d(nn.Module):
    """A convolution as im2col on the CPU and one W8A8 contraction on the NPU.

    Rows are output pixels, K is kh * kw * Cin in (ky, kx, c) order, so the
    activation scale is per output pixel, as in src/test/model. Batch 1,
    no groups or dilation.
    """

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
                 kernel: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]) -> None:
        super().__init__()
        self.in_channels = qweight.shape[0] // (kernel[0] * kernel[1])
        self.out_channels = qweight.shape[1]
        self.kernel, self.stride, self.padding = kernel, stride, padding
        self.register_buffer('qweight', qweight)      # [kh * kw * Cin, Cout]
        self.register_buffer('scale', scale)
        self.register_buffer('bias', None if bias is None else bias.detach().to(torch.float32))

    @classmethod
    def from_float(cls, conv: nn.Conv2d) -> 'W8A8Conv2d':
        if conv.groups != 1 or tuple(conv.dilation) != (1, 1) or not isinstance(conv.padding, tuple):
            raise ValueError('W8A8Conv2d supports ungrouped, undilated convolutions with explicit padding')
        w = conv.weight.detach().permute(0, 2, 3, 1).reshape(conv.out_channels, -1)   # [Cout, (ky, kx, c)]
        q, s = quantize_weight(w)
        return cls(q, s, conv.bias, tuple(conv.kernel_size), tuple(conv.stride), tuple(conv.padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (kh, kw), (sh, sw), (ph, pw) = self.kernel, self.stride, self.padding
        x = torch.nn.functional.pad(x.to(torch.float32), (pw, pw, ph, ph))
        n, c, h, w = x.shape
        oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
        # K must follow the (ky, kx, c) order used in from_float.
        # Otherwise the im2col columns no longer line up with the weight's rows.
        patches = [x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw]
                   for ky in range(kh) for kx in range(kw)]
        cols = torch.stack(patches, dim=1).permute(0, 3, 4, 1, 2).reshape(n * oh * ow, kh * kw * c)
        xq, xs = quantize_rows(cols)
        acc = torch._int_mm(xq, self.qweight)
        y = acc.to(torch.float32) * xs[:, None] * self.scale[None, :]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(n, oh, ow, self.out_channels).permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return (f'{self.in_channels}, {self.out_channels}, kernel={self.kernel}, stride={self.stride}, '
                f'padding={self.padding}, scheme=w8a8')
