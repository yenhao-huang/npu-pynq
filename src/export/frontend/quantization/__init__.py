"""``quantization``: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8')

The schemes are data in ``schemes.py``, independent of how they are applied.
Each subpackage applies them with one framework; ``quantize`` is the one in use:

    fx/    an FX graph pass (``fx/transforms/quantize.py``) rewrites each
           aten.linear, aten.conv2d and aten.embedding into the scheme's
           computation (``fx/<scheme>.py``, with the custom ops of ``fx/ops.py``)
"""

from .fx import quantize

__all__ = ['quantize']
