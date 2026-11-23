"""The quantization stage: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8', calibration={'prefill': [args, ...]})

export_graph captures each entry point before any decomposition, so every op
a scheme quantizes is still one node (aten.linear ...) whose weight is a
get_attr of the unlifted module (``ep.module()``). The stage runs in three steps:

1. calibrate (fx/calibrator.py): runs each program named in ``calibration``
   on its argument tuples, in float, and records the largest |input| of every
   layer whose input the scheme quantizes;
2. quantize (fx/quantizer/<scheme>.py): the scheme quantizes the weight of
   every node of its ops (``schemes.py``), with its activation scale, once
   per weight;
3. replace (fx/rewriter.py): in every program, each of those nodes becomes the
   scheme's computation, reading step 2's tensors; the module is re-captured
   with torch.export.

Statistics and quantized weights are kept per weight tensor, not per program:
decode and prefill share every weight, so calibrating prefill alone gives
decode the same activation scales, and giving both the same quantized tensors
keeps to_mlir freezing one constant per weight.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.calibrator import Calibrator
from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8Quantizer
from src.export.frontend.quantization.fx.rewriter import Rewriter
from src.export.frontend.quantization.schemes import SCHEMES

# Scheme name -> the quantizer that applies it. Its parameters are schemes.SCHEMES.
QUANTIZERS = {'w8a8': W8A8Quantizer}


def quantize(programs: dict[str, torch.export.ExportedProgram], scheme: str,
             calibration: dict[str, list[tuple]] | None = None
             ) -> dict[str, torch.export.ExportedProgram]:
    """Quantize every program with ``scheme``; return them re-captured, undecomposed.

    ``calibration`` maps a program's name to the argument tuples to calibrate
    it on; each one is passed as ``ep.module()(*args)``."""
    try:
        quantizer_class = QUANTIZERS[scheme]
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; '
                         f'known: {sorted(QUANTIZERS)}') from None
    ops = SCHEMES[scheme]['ops']
    stats = Calibrator(quantizer_class.calibrated_ops(ops)).run(programs, calibration or {})   # 1. calibrate
    quantizer = quantizer_class(ops, stats)
    quantizer.run(programs)                                                                    # 2. quantize weights
    return Rewriter(quantizer).run(programs)                                                   # 3. replace nodes
