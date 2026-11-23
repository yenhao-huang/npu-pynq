"""The quantization stage: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8', calibration={'prefill': [args, ...]})

export_graph captures each entry point before any decomposition, so every op
a scheme quantizes is still one node (aten.linear ...) whose weight is a
get_attr of the unlifted module (``ep.module()``). The stage runs in three steps:

1. calibrate (fx/calibrator.py): from the scheme's parameters, computes every
   layer's Scales: a weight scale per output channel, and, for the scheme's
   calibrated_ops, an activation scale from the largest |input| seen while
   each program named in ``calibration`` ran on its argument tuples, in float;
2. quantize (fx/quantizer/): Quantizer runs the scheme's QuantizedLayer
   (fx/quantizer/<scheme>.py) over every node of its ops (``schemes.py``): it
   lays each weight out and rounds it with its scale, once per weight, and
   returns the plan, layer key -> QuantizedOp;
3. replace (fx/rewriter.py): in every program, quantize/dequantize go around
   each of those nodes with its plan entry's tensors as buffers, then each op
   becomes its integer form (aten._int_mm); the module is re-captured with
   torch.export.

The three stages never import each other: they exchange plain data (the
scales, the plan) through this module, and share only fx/utils/share.py.

Scales and quantized weights are kept per weight tensor, not per program:
decode and prefill share every weight, so calibrating prefill alone gives
decode the same activation scales, and giving both the same quantized tensors
keeps to_mlir freezing one constant per weight.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.calibrator import Calibrator
from src.export.frontend.quantization.fx.quantizer.base import QuantizedLayer, Quantizer
from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8QuantizedLayer
from src.export.frontend.quantization.fx.rewriter import Rewriter
from src.export.frontend.quantization.schemes import SCHEMES

# Scheme name -> how it quantizes each op. Its parameters are schemes.SCHEMES.
LAYERS: dict[str, type[QuantizedLayer]] = {'w8a8': W8A8QuantizedLayer}


def quantize(programs: dict[str, torch.export.ExportedProgram], scheme: str,
             calibration: dict[str, list[tuple]] | None = None
             ) -> dict[str, torch.export.ExportedProgram]:
    """Quantize every program with ``scheme``; return them re-captured, undecomposed.

    ``calibration`` maps a program's name to the argument tuples to calibrate
    it on; each one is passed as ``ep.module()(*args)``."""
    try:
        layer_class = LAYERS[scheme]
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; '
                         f'known: {sorted(LAYERS)}') from None
    spec = SCHEMES[scheme]
    if not set(spec['calibrated_ops']) <= set(spec['ops']):
        raise ValueError(f"{scheme}: calibrated_ops {spec['calibrated_ops']} must be among its ops {spec['ops']}")
    scales = Calibrator(spec).run(programs, calibration or {})                     # 1. statistics -> scales
    plan = Quantizer(layer_class(), spec['ops'], scales).run(programs)             # 2. quantize weights with them
    return Rewriter(spec['ops'], plan, spec['activation_qmax']).run(programs)      # 3. replace nodes
