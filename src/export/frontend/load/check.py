"""Reject a model the NPU cannot run: every contraction must be in an accepted scheme."""

from __future__ import annotations

import torch
from torch import nn

from src.export.frontend.quantization.torchao.custom_quant.w8a8 import W8A8Embedding, W8A8Linear


class QuantizationError(ValueError):
    pass


SCHEME_MODULES = {'w8a8': (W8A8Linear,)}


def check(model: nn.Module, require: str) -> None:
    """Raise QuantizationError unless every linear layer is quantized to ``require``."""
    accepted = SCHEME_MODULES.get(require)
    if accepted is None:
        raise QuantizationError(f'unknown scheme {require!r}')
    bad = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            bad.append(f'{name}: float {type(module).__name__}')
        elif isinstance(module, W8A8Linear) and require == 'w8a8':
            q = module.qweight
            if q.dtype != torch.int8 or int(q.min()) < -127:
                bad.append(f'{name}: weights outside the symmetric INT8 range')
        elif isinstance(module, (W8A8Linear, W8A8Embedding)) or not hasattr(module, 'weight'):
            continue
        elif type(getattr(module, 'weight', None)).__name__ not in ('Parameter', 'Tensor'):
            bad.append(f'{name}: weight quantized by another tool ({type(module.weight).__name__})')
    if bad:
        raise QuantizationError(f'model is not {require}: ' + '; '.join(bad[:8]))
