"""NPU quantization as torchao configs: ``quantize_(model, NpuW8A8Config())``."""

from .config import NpuW8A8Config, config_for

__all__ = ['NpuW8A8Config', 'config_for']
