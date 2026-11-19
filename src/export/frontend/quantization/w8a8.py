"""W8A8: per-output-channel INT8 weights, per-token INT8 activations.

The scheme is defined in quantization/schemes.py:

    scale = max(amax, EPS) / 127    EPS avoids dividing by zero on an all-zero row or channel
    q     = round(value / scale)    half to even (torch.round), clamped to [-127, 127]

When each part is quantized:

    weights        once, by the graph pass (``prepare_*``)
    activations    on every call, in the graph, on the CPU
                   (so there is no calibration step and no stored activation scale)

``linear``, ``conv2d`` and ``embedding`` replace aten.linear, aten.conv2d and
aten.embedding in a captured graph. They use only plain ATen ops, so the
re-export traces them into the graph. ``torch._int_mm`` is the INT8 x INT8 ->
INT32 contraction, on a constant weight.

Overflow: |acc| <= K * 127 * 127, which stays below 2**31 while K is under about
133,000, so an INT32 accumulator cannot overflow or saturate.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.schemes import W8A8

EPS = W8A8['eps']
QMAX = W8A8['activation_qmax']


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] float -> ([in, out] int8, [out] float32 scale), symmetric per output channel."""
    w = weight.detach().to(torch.float32).t().contiguous()           # [K, N]
    qmax = W8A8['weight_qmax']
    scale = torch.clamp(w.abs().amax(0), min=EPS) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
    return q, scale.to(torch.float32)


def quantize_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] float -> ([M, K] int8, [M] float32 scale), symmetric per token."""
    scale = torch.clamp(x.abs().amax(-1), min=EPS) * (1.0 / QMAX)
    q = torch.clamp(torch.round(x * (1.0 / scale)[:, None]), -QMAX, QMAX).to(torch.int8)
    return q, scale


# --- weights, quantized once by the pass ------------------------------------

def prepare_linear(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """aten.linear's [out, in] weight -> ([in, out] int8, [out] scale).

    Stored [K, N] so the contraction's right operand is the weight itself."""
    return quantize_weight(weight)


def prepare_conv2d(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """aten.conv2d's [Cout, Cin, kh, kw] weight -> ([kh * kw * Cin, Cout] int8, [Cout] scale).

    K follows the (ky, kx, c) order that ``conv2d``'s im2col uses."""
    w = weight.detach().permute(0, 2, 3, 1).reshape(weight.shape[0], -1)   # [Cout, (ky, kx, c)]
    return quantize_weight(w)


def prepare_embedding(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """aten.embedding's [V, H] table -> ([V, H] int8, [V] scale): one scale per row."""
    q, s = quantize_weight(weight)      # per output channel of [V, H].T = per row
    return q.t().contiguous(), s


# --- the replacements, traced into the graph --------------------------------

def linear(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
           bias: torch.Tensor | None) -> torch.Tensor:
    """y = (int8(x) @ W_q) * x_scale * w_scale + bias."""
    shape = x.shape
    k, n = qweight.shape
    # The contraction is rank 2, and flattening gives every token its own
    # activation scale.
    x2 = x.reshape(-1, k).to(torch.float32)
    xq, xs = quantize_tokens(x2)
    acc = torch._int_mm(xq, qweight)
    y = acc.to(torch.float32) * xs[:, None] * scale[None, :]
    if bias is not None:
        y = y + bias
    return y.reshape(*shape[:-1], n)


def conv2d(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
           kernel: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]) -> torch.Tensor:
    """A convolution as im2col and one W8A8 contraction.

    Rows are output pixels, K is kh * kw * Cin in (ky, kx, c) order, so the
    activation scale is per output pixel. Batch 1, no groups or dilation.
    """
    (kh, kw), (sh, sw), (ph, pw) = kernel, stride, padding
    cout = qweight.shape[1]
    x = torch.nn.functional.pad(x.to(torch.float32), (pw, pw, ph, ph))
    n, c, h, w = x.shape
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    # K must follow the (ky, kx, c) order used in prepare_conv2d.
    # Otherwise the im2col columns no longer line up with the weight's rows.
    patches = [x[:, :, ky:ky + sh * (oh - 1) + 1:sh, kx:kx + sw * (ow - 1) + 1:sw]
               for ky in range(kh) for kx in range(kw)]
    cols = torch.stack(patches, dim=1).permute(0, 3, 4, 1, 2).reshape(n * oh * ow, kh * kw * c)
    xq, xs = quantize_tokens(cols)
    acc = torch._int_mm(xq, qweight)
    y = acc.to(torch.float32) * xs[:, None] * scale[None, :]
    if bias is not None:
        y = y + bias
    return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2)


def embedding(ids: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Rows of the INT8 table, rescaled. A table lookup, not a contraction."""
    rows = qweight.index_select(0, ids.reshape(-1)).to(torch.float32)
    out = rows * scale.index_select(0, ids.reshape(-1))[:, None]
    return out.reshape(*ids.shape, qweight.shape[1])
