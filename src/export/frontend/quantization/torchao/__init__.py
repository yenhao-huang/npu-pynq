"""Our quantization schemes as torchao configs: ``quantize_(model, W8A8Config())``."""

from .config import W8A8Config, config_for

__all__ = ['W8A8Config', 'config_for']
