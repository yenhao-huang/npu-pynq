"""The quantization stage: captured ExportedPrograms -> quantized ExportedPrograms.

    quantize({'decode': ep, 'prefill': ep}, 'w8a8', calibration={'prefill': [args, ...]})

export_graph captures each entry point before any decomposition, so every op
a scheme quantizes is still one node (aten.linear ...) whose weight is a
get_attr of the unlifted module (``ep.module()``). The stage runs in three steps:

1. calibrate: runs each program named in ``calibration`` on its argument
   tuples, in float, and records the largest |input| of every layer the
   scheme's transform calibrates;
2. prepare: has the transform quantize the weight of every node of the
   scheme's ops (``schemes.py``), with its activation scale, once per weight;
3. replace: in every program, has the transform replace each of those nodes
   with its quantized computation, reading step 2's tensors, then re-captures
   the module with torch.export.

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
    weights = prepare(programs, transform, ops, stats)                   # 2. quantize the weights
    return replace(programs, transform, ops, weights)                    # 3. replace the nodes


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


def prepare(programs: dict[str, torch.export.ExportedProgram], transform, ops, stats: dict) -> dict:
    """Quantize the weight of every node of ``ops`` in every program; return {layer key: tensors}.

    A weight shared by several programs, or by several nodes, has one key and
    is quantized once. A tied weight used by two kinds of op (lm_head and the
    embedding) has one key per kind."""
    weights: dict = {}
    for ep in programs.values():
        gm = ep.module()
        for node in transform.select(gm, ops):
            key = transform.layer_key(gm, node)
            if key not in weights:
                weights[key] = transform.quantize_layer(gm, node, stats)
    return weights


def replace(programs: dict[str, torch.export.ExportedProgram], transform, ops, weights: dict
            ) -> dict[str, torch.export.ExportedProgram]:
    """Replace every node of ``ops`` in each program with the transform's computation; re-capture."""
    out = {}
    for name, ep in programs.items():
        gm = ep.module()
        for node in transform.select(gm, ops):
            transform.rewrite(gm, node, weights)
        transform.finish(gm)
        args, kwargs = ep.example_inputs
        with torch.no_grad():
            out[name] = torch.export.export(gm, args, kwargs, strict=False)
    return out
