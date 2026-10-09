"""Shared state of one function under construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.compiler.mlir_builder import FuncBuilder, Value


@dataclass
class Ctx:
    b: FuncBuilder
    graph: "object"  # src.compiler.export.ModelGraph
    arena: Value
    cache: dict = field(default_factory=dict)

    def weight(self, name: str, array: np.ndarray | None = None, shape: Sequence[int] | None = None,
               dtype: str | None = None) -> Value:
        """Register ``array`` (once per graph) and reference it in this function."""
        if array is not None and name not in self.graph.weights:
            self.graph.add_weight(name, array)
        stored = self.graph.weights[name]
        shape = list(stored.shape) if shape is None else list(shape)
        dtype = dtype or {np.dtype(np.float32): "f32", np.dtype(np.int8): "i8",
                          np.dtype(np.int32): "i32"}[stored.dtype]
        key = ("weight", name, tuple(shape))
        if key not in self.cache:
            self.cache[key] = self.b.weight(self.arena, name, shape, dtype)
        return self.cache[key]

    def if_(self, cond: Value, result_types, then_fn, else_fn):
        """scf.if whose branches may reference weights (cached per branch)."""
        saved = dict(self.cache)

        def scoped(fn):
            def run():
                self.cache = dict(saved)
                return fn()
            return run

        try:
            return self.b.if_(cond, result_types, scoped(then_fn), scoped(else_fn))
        finally:
            self.cache = saved
