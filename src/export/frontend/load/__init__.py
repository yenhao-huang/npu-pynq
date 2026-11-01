"""``load``: a PyTorch model from a framework, quantized for the NPU at load.

    load('huggingface', 'path/to/checkpoint', quantization='w8a8', require='w8a8')

The framework builds the model; ``quantization`` runs our torchao config on
it; ``require`` refuses the model unless every layer ended in that scheme.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from torch import nn

from .check import QuantizationError, check

FRAMEWORKS = {'huggingface': 'huggingface', 'torchvision': 'torchvision_models'}


def framework(name: str):
    try:
        return import_module(f'{__name__}.frameworks.{FRAMEWORKS[name]}')
    except KeyError:
        raise ValueError(f'unknown framework {name!r}; known: {sorted(FRAMEWORKS)}') from None


def load(name: str, source: str | Path, quantization: str | None = None, require: str | None = None,
         **kwargs) -> nn.Module:
    model = framework(name).load(source, **kwargs)
    if quantization:
        from src.export.frontend.quantization.torchao.config import quantize

        model = quantize(model, quantization)
    if require:
        check(model, require)
    return model


__all__ = ['QuantizationError', 'check', 'framework', 'load']
