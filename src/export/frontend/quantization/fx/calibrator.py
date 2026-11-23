"""The Calibrator: statistics in, every layer's scales out.

From a scheme's parameters (schemes.py: eps, weight_qmax, activation_qmax and
the two granularities) it computes, per layer key, a Scales:

    weight_scale       per_channel: max(amax, eps) / weight_qmax, amax over every
                       dim of the float32 weight but dim 0 (the output channel of
                       linear [out, in] and conv2d [Cout, Cin, kh, kw], the row of
                       an embedding table [V, H]); no calibration data needed
    activation_scale   per_tensor, for the scheme's calibrated_ops only:
                       max(amax, eps) / activation_qmax as a [1] tensor, amax the
                       largest |input| seen while the float programs ran on the
                       calibration inputs (torch.fx.Interpreter)

Statistics are keyed like the layer's weight (``layer_key``), so every program
that shares the weight shares them: calibrating an LLM's prefill gives decode
its scales too.
"""

from __future__ import annotations

import torch
from torch import fx

from src.export.frontend.quantization.fx.utils.share import Scales, attr, layer_key, parse, select


class Calibrator:
    """Holds the scheme's parameters and the largest |input| seen per layer key."""

    def __init__(self, scheme: dict) -> None:
        if scheme['weight_granularity'] != 'per_channel':
            raise ValueError(f"{scheme['name']}: weight granularity {scheme['weight_granularity']!r} "
                             "is not implemented; the Calibrator computes per_channel")
        if scheme['calibrated_ops'] and scheme['activation_granularity'] != 'per_tensor':
            raise ValueError(f"{scheme['name']}: activation granularity {scheme['activation_granularity']!r} "
                             "is not implemented; the Calibrator computes per_tensor")
        self.scheme = scheme
        self.amax: dict[tuple, float] = {}   # layer key -> largest |input| seen

    def run(self, programs: dict[str, torch.export.ExportedProgram],
            calibration: dict[str, list[tuple]]) -> dict[tuple, Scales]:
        """Observe the calibration inputs, then return the Scales of every layer of the scheme's ops."""
        self.observe(programs, calibration)
        scheme = self.scheme
        scales: dict[tuple, Scales] = {}
        for ep in programs.values():
            gm = ep.module()
            for node in select(gm, scheme['ops']):
                kind, args = parse(node)
                key = layer_key(gm, kind, args)
                if key in scales:
                    continue
                name = args['weight'].target
                activation = None
                if kind in scheme['calibrated_ops']:
                    if key not in self.amax:
                        raise ValueError(f'no calibration statistics for {kind} weight {name}; '
                                         f"{scheme['name']} needs calibration inputs that reach every layer")
                    activation = self.activation_scale(self.amax[key])
                scales[key] = Scales(self.weight_scale(attr(gm, name)), activation)
        return scales

    def observe(self, programs: dict[str, torch.export.ExportedProgram],
                calibration: dict[str, list[tuple]]) -> None:
        """Run each program named in ``calibration`` on its argument tuples, in float, recording
        each calibrated layer's largest |input| (``'x'``; a convolution's before its padding,
        which adds zeros only)."""
        for name, inputs in calibration.items():
            gm = programs[name].module()
            layers = {}
            for n in select(gm, self.scheme['calibrated_ops']):
                kind, args = parse(n)
                layers[n] = (layer_key(gm, kind, args), args['x'])
            amax = self.amax

            class Observer(fx.Interpreter):
                def run_node(self, n):
                    if n in layers:
                        key, x = layers[n]
                        amax[key] = max(amax.get(key, 0.0), float(self.env[x].detach().abs().amax()))
                    return super().run_node(n)

            observer = Observer(gm)
            with torch.no_grad():
                for args in inputs:
                    observer.run(*args)

    def weight_scale(self, weight: torch.Tensor) -> torch.Tensor:
        """Per output channel: [N] float32, N = the weight's dim 0."""
        w = weight.detach().to(torch.float32)
        amax = w.abs().amax(dim=tuple(range(1, w.dim())))
        return (torch.clamp(amax, min=self.scheme['eps']) / self.scheme['weight_qmax']).to(torch.float32)

    def activation_scale(self, amax: float) -> torch.Tensor:
        """Per tensor: the largest |input| -> [1] float32."""
        return torch.clamp(torch.tensor([amax], dtype=torch.float32), min=self.scheme['eps']) / \
            self.scheme['activation_qmax']
