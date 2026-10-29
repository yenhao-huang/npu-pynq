"""Our quantization schemes as torchao configs.

torchao's ``quantize_(model, config)`` walks the model and calls the handler
registered for the config's type on every module the filter selects. Our
handlers swap in the modules of ``custom_quant/``; the schemes themselves are
the vendor's (src/vendors/npu_pynq/spec/quant.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .custom_quant.w8a8 import W8A8Embedding, W8A8Linear


@dataclass
class NpuW8A8Config(AOBaseConfig):
    """W8A8 for nn.Linear (and nn.Embedding, with ``filter_fn=embedding_and_linear``)."""


@register_quantize_module_handler(NpuW8A8Config)
def _w8a8(module: nn.Module, config: NpuW8A8Config) -> nn.Module:
    if isinstance(module, nn.Linear):
        return W8A8Linear.from_float(module)
    if isinstance(module, nn.Embedding):
        return W8A8Embedding.from_float(module)
    raise TypeError(f'NpuW8A8Config does not quantize {type(module).__name__}')


def embedding_and_linear(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, (nn.Linear, nn.Embedding))


CONFIGS = {'w8a8': NpuW8A8Config}


def config_for(scheme: str) -> AOBaseConfig:
    try:
        return CONFIGS[scheme]()
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; known: {sorted(CONFIGS)}') from None
