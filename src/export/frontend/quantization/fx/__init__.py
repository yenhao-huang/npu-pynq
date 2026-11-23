"""Quantization as an FX graph pass on the captured, undecomposed graphs.

    transforms/quantize.py    the pass: finds each layer's node and replaces it
    <scheme>.py               the scheme's weight preparation and the functions
                              the pass puts in place of each node
    ops.py                    the quant.* custom ops those functions call, each
                              one node in the re-captured graph
"""

from .transforms.quantize import quantize

__all__ = ['quantize']
