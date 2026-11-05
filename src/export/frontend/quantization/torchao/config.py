"""Our quantization schemes as torchao configs.

torchao's ``quantize_(model, config)`` walks the model and calls the handler
registered for the config's type on every module the filter selects. Our
handlers swap in the modules of ``custom_quant/``; the schemes themselves are in
``schemes.py``.

BatchNorm stays in the model here; a later compilation stage folds it.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .custom_quant.w8a8 import W8A8Conv2d, W8A8Embedding, W8A8Linear


@dataclass
class W8A8Config(AOBaseConfig):
    """W8A8 for nn.Linear, nn.Conv2d and nn.Embedding (with ``filter_fn=quantizable_layers``)."""


@register_quantize_module_handler(W8A8Config)
def _w8a8(module: nn.Module, config: W8A8Config) -> nn.Module:
    if isinstance(module, nn.Linear):
        return W8A8Linear.from_float(module)
    if isinstance(module, nn.Conv2d):
        return W8A8Conv2d.from_float(module)
    if isinstance(module, nn.Embedding):
        return W8A8Embedding.from_float(module)
    raise TypeError(f'W8A8Config does not quantize {type(module).__name__}')


def quantizable_layers(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding))


CONFIGS = {'w8a8': W8A8Config}


def config_for(scheme: str) -> AOBaseConfig:
    try:
        return CONFIGS[scheme]()
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; known: {sorted(CONFIGS)}') from None


def quantize(model: nn.Module, scheme: str) -> nn.Module:
    from torchao.quantization import quantize_

    # quantize_ works in place: each layer that quantizable_layers selects is replaced
    # by the module that the config's handler returns.
    quantize_(model, config_for(scheme), filter_fn=quantizable_layers)
    return model
