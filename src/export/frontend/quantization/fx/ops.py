"""The custom ops the schemes' functions put in the graph, in the ``quant`` namespace.

    quant::quantize_per_token(x, eps, qmax) -> (q, scale)
        [M, K] float32 -> ([M, K] int8, [M] float32), symmetric per token:
        scale = max(amax(|x|), eps) * (1 / qmax)
        q     = clamp(round(x * (1 / scale)), -qmax, qmax)    round half to even

    quant::dequantize(acc, x_scale, w_scale) -> y
        [M, N] int32 -> [M, N] float32:
        y = float(acc) * x_scale[:, None] * w_scale[None, :]

torch.export traces each one as a single node (``register_fake`` gives its
output shapes) and run_decompositions keeps it, because a custom op has no
decomposition. torch-mlir imports it as ``torch.operator "quant.<name>"``; the
middle end's DecomposeQuantOps expands it into the torch-dialect ops of the
arithmetic above before lowering. The eager implementations here and that
expansion must stay the same arithmetic, op for op.

eps and qmax are arguments, not constants, so a scheme's parameters stay in
schemes.py.
"""

from __future__ import annotations

import torch

NAMESPACE = 'quant'


@torch.library.custom_op(f'{NAMESPACE}::quantize_per_token', mutates_args=())
def quantize_per_token(x: torch.Tensor, eps: float, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.clamp(x.abs().amax(-1), min=eps) * (1.0 / qmax)
    q = torch.clamp(torch.round(x * (1.0 / scale)[:, None]), -qmax, qmax).to(torch.int8)
    return q, scale


@quantize_per_token.register_fake
def _(x: torch.Tensor, eps: float, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=torch.int8), x.new_empty(x.shape[:1])


@torch.library.custom_op(f'{NAMESPACE}::dequantize', mutates_args=())
def dequantize(acc: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return acc.to(torch.float32) * x_scale[:, None] * w_scale[None, :]


@dequantize.register_fake
def _(acc: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(acc, dtype=torch.float32)
