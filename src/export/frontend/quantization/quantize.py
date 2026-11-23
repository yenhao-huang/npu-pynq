"""The quantization stage: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8', calibration={'prefill': [args, ...]})

export_graph captures each entry point before any decomposition, so every op
a scheme quantizes is still one node (aten.linear ...) whose weight is a
get_attr of the unlifted module (``ep.module()``). The stage

1. calibrates: runs each program named in ``calibration`` on its argument
   tuples, in float, and lets the scheme's transform record what its static
   activation scales need;
2. rewrites: selects the nodes of the scheme's ops (``schemes.py``) in every
   program and has the transform replace each one;
3. re-captures each module with torch.export.

Statistics and quantized weights are kept per weight tensor, not per program:
decode and prefill share every weight, so calibrating prefill alone gives
decode the same activation scales, and giving both the same quantized tensors
keeps to_mlir freezing one constant per weight.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.transforms import w8a8
from src.export.frontend.quantization.schemes import SCHEMES

# Scheme name -> the transform that applies it. Its parameters are schemes.SCHEMES.
TRANSFORMS = {'w8a8': w8a8}


def quantize(programs: dict[str, torch.export.ExportedProgram], scheme: str,
             calibration: dict[str, list[tuple]] | None = None
             ) -> dict[str, torch.export.ExportedProgram]:
    """Quantize every program with ``scheme``; return them re-captured, undecomposed.

    ``calibration`` maps a program's name to the argument tuples to calibrate
    it on; each one is passed as ``ep.module()(*args)``."""
    try:
        transform = TRANSFORMS[scheme]
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; '
                         f'known: {sorted(TRANSFORMS)}') from None
    ops = SCHEMES[scheme]['ops']
    stats: dict = {}   # float weight -> what calibration recorded for its layer
    for name, inputs in (calibration or {}).items():
        transform.observe(programs[name].module(), ops, inputs, stats)
    cache: dict = {}   # float weight -> its quantized tensors, shared by every program
    out = {}
    for name, ep in programs.items():
        gm = ep.module()
        for node in transform.select(gm, ops):
            transform.rewrite(gm, node, cache, stats)
        transform.finish(gm)
        args, kwargs = ep.example_inputs
        with torch.no_grad():
            out[name] = torch.export.export(gm, args, kwargs, strict=False)
    return out
