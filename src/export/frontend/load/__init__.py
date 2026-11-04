"""``load``: a float PyTorch model from a framework.

    load('huggingface', 'path/to/checkpoint')

The next stage, ``quantization``, quantizes it for the NPU.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from torch import nn

FRAMEWORKS = {'huggingface': 'huggingface', 'torchvision': 'torchvision_models'}


def framework(name: str):
    try:
        return import_module(f'{__name__}.frameworks.{FRAMEWORKS[name]}')
    except KeyError:
        raise ValueError(f'unknown framework {name!r}; known: {sorted(FRAMEWORKS)}') from None


def load(name: str, source: str | Path, **kwargs) -> nn.Module:
    return framework(name).load(source, **kwargs)


__all__ = ['framework', 'load']
