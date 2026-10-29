"""Turn a model's source tensors into what the operators consume.

An architecture describes its weights as a list of ``WeightSpec``: which
source tensors make one parameter (concatenated in torch [out, in] layout,
or combined by ``transform``) and how that parameter is stored. This module
is the only place that applies the description: per layer it gathers the
tensors, applies the AWQ hook when given, then quantizes every matrix the
same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from src.export.ir.options import QuantRecipe
from src.export.quantize.int8 import quantize_weight

AwqHook = Callable[[int, dict, dict], tuple[dict, dict]]

KINDS = ('linear', 'embed', 'fp32', 'float', 'raw')


@dataclass
class WeightSpec:
    name: str                    # parameter name, e.g. 'l3.qkv'
    sources: tuple[str, ...]     # source tensor names, in order
    kind: str                    # linear: recipe.bits | embed: recipe.embed_bits | fp32: unquantized
                                 # matrix | float: float32 vector | raw: unchanged
    axis: int = 0                # concatenation axis when there are several sources
    transform: Callable[..., np.ndarray] | None = None   # combines the sources instead of concatenating
    layer: int | None = None     # decoder layer, for the AWQ hook
    role: str | None = None      # key in the AWQ hook's mats/norms: qkv, o, gate_up, down, ln1, ln2

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f'{self.name}: unknown weight kind {self.kind}')


def gather(tensors, spec: WeightSpec) -> np.ndarray:
    """``tensors`` is anything indexable by source name."""
    arrays = [tensors[name] for name in spec.sources]
    if spec.transform is not None:
        return spec.transform(*arrays)
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=spec.axis)


def store(array: np.ndarray, kind: str, recipe: QuantRecipe):
    if kind == 'linear':
        return quantize_weight(array, recipe.bits, recipe.group)
    if kind == 'embed':
        return quantize_weight(array, recipe.embed_bits, recipe.group)
    if kind == 'fp32':
        return quantize_weight(array, 32)
    if kind == 'float':
        return array.astype(np.float32)
    return array


class _Reads(dict):
    def __init__(self, tensors) -> None:
        super().__init__()
        self.tensors = tensors

    def __missing__(self, name: str) -> np.ndarray:
        self[name] = array = self.tensors[name]
        return array


def _by_layer(specs: Iterable[WeightSpec]) -> list[list[WeightSpec]]:
    """Consecutive runs of specs that share a layer, in plan order."""
    runs: list[list[WeightSpec]] = []
    for spec in specs:
        if runs and runs[-1][0].layer == spec.layer:
            runs[-1].append(spec)
        else:
            runs.append([spec])
    return runs


def run_plan(tensors, specs: Iterable[WeightSpec], recipe: QuantRecipe, awq: AwqHook | None = None) -> dict:
    """Return ``{spec.name: stored parameter}``, one layer at a time."""
    params = {}
    for run in _by_layer(specs):
        # Read each source tensor once per layer (stacked expert tensors serve every expert).
        cache = _Reads(tensors)
        arrays = {spec.name: gather(cache, spec) for spec in run}
        layer = run[0].layer
        if awq is not None and layer is not None:
            mats = {s.role: arrays[s.name] for s in run if s.role and s.kind == 'linear'}
            norms = {s.role: arrays[s.name] for s in run if s.role and s.kind == 'float'}
            mats, norms = awq(layer, mats, norms)
            for s in run:
                if s.role and s.kind in ('linear', 'float'):
                    arrays[s.name] = (mats if s.kind == 'linear' else norms)[s.role]
        for spec in run:
            params[spec.name] = store(arrays[spec.name], spec.kind, recipe)
    return params

