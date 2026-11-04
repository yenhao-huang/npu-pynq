"""``quantization``: a float PyTorch model -> a model in a scheme the NPU accepts.

    quantize(model, 'w8a8')

``scheme`` runs our torchao config on the model.
"""

from .torchao.config import quantize

__all__ = ['quantize']
