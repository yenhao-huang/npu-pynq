"""``quantization``: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8')

A graph pass (``graph.py``) rewrites each aten.linear, aten.conv2d and
aten.embedding into the scheme's plain-ATen computation (``w8a8.py``); the
schemes themselves are data in ``schemes.py``.
"""

from .graph import quantize

__all__ = ['quantize']
