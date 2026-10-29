"""The graph a model frontend hands to the compiler, and the entry-point
conventions shared by every LLM package.

Each LLM exports two functions with the same argument order:

    decode (weights, *state, token: index,            pos: index, logits: memref<1xVxf32>)
    prefill(weights, *state, tokens: memref<Txi32>,   pos: index, last: index, logits)

``state`` is the model's KV cache and any recurrent state, in the order the
manifest lists them. ``prefill`` consumes a fixed chunk of T tokens starting
at ``pos`` and returns the logits of token ``last`` of the chunk; padding past
``last`` writes cache positions that later tokens overwrite before reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np
from src.export.compilers.mlir.builder import ModuleBuilder, Value, memref_type
from src.export.compilers.mlir.ops.context import Ctx


@dataclass
class ModelGraph:
    """What a model frontend hands to the compiler."""

    name: str
    module: ModuleBuilder
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    entry_points: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)  # copied verbatim (tokenizer, ...)

    def add_weight(self, name: str, array: np.ndarray) -> str:
        if name in self.weights:
            raise ValueError(f"duplicate weight {name}")
        self.weights[name] = np.ascontiguousarray(array)
        return name


@dataclass
class StateSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str  # "f32" | "i8"


def build_prefill_decode_graphs(graph: ModelGraph, state: list[StateSpec], vocab: int, chunk: int,
                       body: Callable[[Ctx, Value, dict[str, Value], Value, Value | None], Value]) -> None:
    """Emit decode and prefill; ``body(ctx, tokens_i32[T], state, pos, last) -> logits [1, V]``."""
    for name, tokens in (("decode", 1), ("prefill", chunk)):
        args = [("ws", "memref<?xi8>")] + [(s.name, memref_type(s.shape, s.dtype)) for s in state]
        if name == "decode":
            args += [("token", "index"), ("pos", "index")]
        else:
            args += [("tokens", memref_type([tokens], "i32")), ("pos", "index"), ("last", "index")]
        args += [("logits", memref_type([1, vocab], "f32"))]
        f = graph.module.func(name, args)
        ctx = Ctx(f, graph, f.arg("ws"))
        if name == "decode":
            t32 = f.op(f"arith.index_cast %token : index to i32", "i32")
            toks = f.op(f"tensor.from_elements {t32} : tensor<1xi32>", "tensor<1xi32>")
            last = None
        else:
            toks = f.to_tensor(f.arg("tokens"))
            last = f.arg("last")
        logits = body(ctx, toks, {s.name: f.arg(s.name) for s in state}, f.arg("pos"), last)
        f.store(logits, f.arg("logits"))
        f.emit("return")
    graph.meta["state"] = [{"name": s.name, "shape": list(s.shape), "dtype": s.dtype} for s in state]
    graph.meta["vocab"] = vocab
    graph.meta["chunk"] = chunk
    graph.entry_points = {
        "decode": ["weights", *(s.name for s in state), "token", "pos", "logits"],
        "prefill": ["weights", *(s.name for s in state), "tokens", "pos", "last", "logits"],
    }


def new_graph(name: str) -> ModelGraph:
    return ModelGraph(name, ModuleBuilder())
