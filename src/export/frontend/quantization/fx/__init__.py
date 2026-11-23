"""Quantization as an FX graph pass on the captured, undecomposed graphs.

    transforms/quantize.py    the pass: finds each layer's node and replaces it
    <scheme>.py               the scheme's weight preparation and the plain-ATen
                              functions the pass puts in place of each node
"""

from .transforms.quantize import quantize

__all__ = ['quantize']
