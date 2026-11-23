"""The quantization stage: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8', calibration={'prefill': [args, ...]})

export_graph captures each entry point before any decomposition, so every op
a scheme quantizes is still one node (aten.linear ...) whose weight is a
get_attr of the unlifted module (``ep.module()``). The stage does two things:

1. calibrate: runs each program named in ``calibration`` on its argument
   tuples, in float, and records the largest |input| of every layer the
   scheme's transform calibrates;
2. replace: in every program, has the transform replace each node of the
   scheme's ops (``schemes.py``), then re-captures the module with torch.export.

Statistics and quantized weights are kept per weight tensor, not per program:
decode and prefill share every weight, so calibrating prefill alone gives
decode the same activation scales, and giving both the same quantized tensors
keeps to_mlir freezing one constant per weight.
"""

from __future__ import annotations

import torch
from torch import fx

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
    stats = calibrate(programs, transform, ops, calibration or {})       # 1. calibrate
    cache: dict = {}   # float weight -> its quantized tensors, shared by every program
    return {name: replace(ep, transform, ops, cache, stats)              # 2. replace
            for name, ep in programs.items()}


def calibrate(programs: dict[str, torch.export.ExportedProgram], transform, ops,
              calibration: dict[str, list[tuple]]) -> dict:
    """Run each program named in ``calibration`` in float; return {layer key: largest |input|}.

    The transform names the layers to observe and the key of each, which is
    its float weight's, so every program that shares the weight shares the
    statistic. A layer's input is its node's first argument."""
    stats: dict = {}
    for name, inputs in calibration.items():
        gm = programs[name].module()
        layers = transform.calibrated_layers(gm, ops)

        class Observer(fx.Interpreter):
            def run_node(self, n):
                if n in layers:
                    amax = float(self.env[n.args[0]].detach().abs().amax())
                    stats[layers[n]] = max(stats.get(layers[n], 0.0), amax)
                return super().run_node(n)

        observer = Observer(gm)
        with torch.no_grad():
            for args in inputs:
                observer.run(*args)
    return stats


def replace(ep: torch.export.ExportedProgram, transform, ops, cache: dict, stats: dict
            ) -> torch.export.ExportedProgram:
    """Replace every node of ``ops`` in ``ep`` with the transform's computation; re-capture."""
    gm = ep.module()
    for node in transform.select(gm, ops):
        transform.rewrite(gm, node, cache, stats)
    transform.finish(gm)
    args, kwargs = ep.example_inputs
    with torch.no_grad():
        return torch.export.export(gm, args, kwargs, strict=False)
