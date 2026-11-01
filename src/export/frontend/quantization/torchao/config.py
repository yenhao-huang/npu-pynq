"""Our quantization schemes as torchao configs.

torchao's ``quantize_(model, config)`` walks the model and calls the handler
registered for the config's type on every module the filter selects. Our
handlers swap in the modules of ``custom_quant/``; the schemes themselves are
the vendor's (src/vendors/npu_pynq/spec/quant.py).

``quantize`` is what ``load`` runs: BatchNorm is folded into the
convolution before it (an inference-only rewrite PyTorch provides), so the
folded weight is what gets quantized.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .custom_quant.w8a8 import W8A8Conv2d, W8A8Embedding, W8A8Linear


@dataclass
class NpuW8A8Config(AOBaseConfig):
    """W8A8 for nn.Linear, nn.Conv2d and nn.Embedding (with ``filter_fn=npu_layers``)."""


@register_quantize_module_handler(NpuW8A8Config)
def _w8a8(module: nn.Module, config: NpuW8A8Config) -> nn.Module:
    if isinstance(module, nn.Linear):
        return W8A8Linear.from_float(module)
    if isinstance(module, nn.Conv2d):
        return W8A8Conv2d.from_float(module)
    if isinstance(module, nn.Embedding):
        return W8A8Embedding.from_float(module)
    raise TypeError(f'NpuW8A8Config does not quantize {type(module).__name__}')


def npu_layers(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding))


CONFIGS = {'w8a8': NpuW8A8Config}


def config_for(scheme: str) -> AOBaseConfig:
    try:
        return CONFIGS[scheme]()
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; known: {sorted(CONFIGS)}') from None


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Conv2d + BatchNorm2d pairs folded into one Conv2d (torch.fx's inference fusion)."""
    if not any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
        return model
    from torch.fx.experimental.optimization import fuse

    folded = fuse(model)
    if any(isinstance(m, nn.BatchNorm2d) for m in folded.modules()):
        raise ValueError('a BatchNorm2d does not follow a convolution and cannot be folded')
    return folded


def quantize(model: nn.Module, scheme: str) -> nn.Module:
    from torchao.quantization import quantize_

    model = fold_batchnorm(model)
    quantize_(model, config_for(scheme), filter_fn=npu_layers)
    return model
