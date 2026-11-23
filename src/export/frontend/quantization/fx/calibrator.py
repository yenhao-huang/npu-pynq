"""The Calibrator: records each calibrated layer's largest |input| on calibration data.

It runs the float programs with ``torch.fx.Interpreter`` and observes the
input (``'x'``) of every node of the given ops. Each statistic is keyed like
the layer's weight (``layer_key``), so every program that shares the weight
shares it: calibrating an LLM's prefill gives decode its scales too.
"""

from __future__ import annotations

import torch
from torch import fx

from src.export.frontend.quantization.fx.utils.share import layer_key, parse, select


class Calibrator:
    """Holds {layer key: largest |input| seen} over every calibration input."""

    def __init__(self, ops) -> None:
        self.ops = tuple(ops)          # the op names whose input is calibrated
        self.stats: dict[tuple, float] = {}

    def run(self, programs: dict[str, torch.export.ExportedProgram],
            calibration: dict[str, list[tuple]]) -> dict[tuple, float]:
        """Run each program named in ``calibration`` on its argument tuples, in float; return the stats.

        A convolution's input is taken before its padding, which adds zeros only."""
        for name, inputs in calibration.items():
            gm = programs[name].module()
            layers = {}
            for n in select(gm, self.ops):
                kind, args = parse(n)
                layers[n] = (layer_key(gm, kind, args), args['x'])
            stats = self.stats

            class Observer(fx.Interpreter):
                def run_node(self, n):
                    if n in layers:
                        key, x = layers[n]
                        amax = float(self.env[x].detach().abs().amax())
                        stats[key] = max(stats.get(key, 0.0), amax)
                    return super().run_node(n)

            observer = Observer(gm)
            with torch.no_grad():
                for args in inputs:
                    observer.run(*args)
        return self.stats
