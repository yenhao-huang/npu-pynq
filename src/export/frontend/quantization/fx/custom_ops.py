"""Custom torch.library ops for quantization, in the ``quant`` namespace.

    quant::quantize_per_tensor(x, scale, qmax) -> q
        float32 x, [1] float32 scale -> int8 q of x's shape, symmetric, zero point 0:
        q = clamp(round(x * (1 / scale)), -qmax, qmax)        round half to even

    quant::dequantize(acc, x_scale, w_scale) -> y
        [M, N] int32, [M] or [1] x_scale, [N] w_scale -> [M, N] float32:
        y = float(acc) * x_scale[:, None] * w_scale[None, :]

torch.export traces each one as a single node (``register_fake`` gives its
output shapes) and run_decompositions keeps it, because a custom op has no
decomposition. torch-mlir imports it as ``torch.operator "torch.quant.<name>"``;
the middle end's DecomposeQuantOps expands it into the torch-dialect ops of
the arithmetic above before lowering. The eager implementations here and that
expansion must stay the same arithmetic, op for op.

scale and qmax are arguments, not constants of the op, so a scheme's
parameters stay in schemes.py and its calibrated scales in the graph.
"""

from __future__ import annotations

import torch

NAMESPACE = 'quant'


@torch.library.custom_op(f'{NAMESPACE}::quantize_per_tensor', mutates_args=())
def quantize_per_tensor(x: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    return torch.clamp(torch.round(x * (1.0 / scale)), -qmax, qmax).to(torch.int8)


@quantize_per_tensor.register_fake
def _(x: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.int8)


@torch.library.custom_op(f'{NAMESPACE}::dequantize', mutates_args=())
def dequantize(acc: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return acc.to(torch.float32) * x_scale[:, None] * w_scale[None, :]


@dequantize.register_fake
def _(acc: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(acc, dtype=torch.float32)
