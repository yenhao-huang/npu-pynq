"""The quantization stage: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8')

export_graph captures each entry point before any decomposition, so every op
a scheme quantizes is still one node (aten.linear ...) whose weight is a
get_attr of the unlifted module (``ep.module()``). For each program this
selects the nodes of the scheme's ops (``schemes.py``), has the scheme's
transform rewrite each one, and re-captures the module with torch.export.

Weights are quantized once per tensor across all programs: decode and prefill
share every weight, and giving both the same quantized tensors keeps to_mlir
freezing one constant per weight.
"""

from __future__ import annotations

import torch

from src.export.frontend.quantization.fx.transforms import w8a8
from src.export.frontend.quantization.schemes import SCHEMES

# Scheme name -> the transform that applies it. Its parameters are schemes.SCHEMES.
TRANSFORMS = {'w8a8': w8a8}


def quantize(programs: dict[str, torch.export.ExportedProgram], scheme: str
             ) -> dict[str, torch.export.ExportedProgram]:
    """Quantize every program with ``scheme``; return them re-captured, undecomposed."""
    try:
        transform = TRANSFORMS[scheme]
    except KeyError:
        raise ValueError(f'unknown quantization scheme {scheme!r}; '
                         f'known: {sorted(TRANSFORMS)}') from None
    ops = SCHEMES[scheme]['ops']
    cache: dict = {}   # float weight -> its quantized tensors, shared by every program
    out = {}
    for name, ep in programs.items():
        gm = ep.module()
        for node in transform.select(gm, ops):
            transform.rewrite(gm, node, cache)
        transform.finish(gm)
        args, kwargs = ep.example_inputs
        with torch.no_grad():
            out[name] = torch.export.export(gm, args, kwargs, strict=False)
    return out
