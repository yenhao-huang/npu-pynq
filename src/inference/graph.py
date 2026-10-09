"""Entry-point conventions shared by every LLM package.

Each LLM exports two functions with the same argument order:

    decode (weights, *state, token: index,            pos: index, logits: memref<1xVxf32>)
    prefill(weights, *state, tokens: memref<Txi32>,   pos: index, last: index, logits)

``state`` is the model's KV cache and any recurrent state, in the order the
manifest lists them. ``prefill`` consumes a fixed chunk of T tokens starting
at ``pos`` and returns the logits of token ``last`` of the chunk; padding past
``last`` writes cache positions that later tokens overwrite before reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from src.compiler.export import ModelGraph
from src.compiler.mlir_builder import ModuleBuilder, Value, memref_type
from src.compiler.ops.context import Ctx
from src.isa import isa


@dataclass
class LLMOptions:
    chunk: int = 16              # prefill tokens per call
    max_seq: int = 512           # KV cache positions
    bits: int = 8                # linear weights: 8 (per channel) or 4 (group-wise)
    group: int = 128             # INT4 group size along K
    embed_bits: int = 8          # tied embedding / lm_head
    kv: str = "f32"              # "f32" or "int8" KV cache
    attention: str = "online"    # "online" or "naive"
    awq: bool = False            # activation-aware scaling before quantization
    awq_samples: list[str] = field(default_factory=list)
    layers: int | None = None    # debug: keep only the first N decoder layers
    limits: isa.Limits = field(default_factory=isa.Limits)

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("awq_samples")
        return d


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
        "decode": ["weights", "state", "token", "pos", "logits"],
        "prefill": ["weights", "state", "tokens", "pos", "last", "logits"],
    }


def choose_group(requested: int, dims: list[int]) -> int:
    """The INT4 group size: ``requested`` if it divides every K, else the
    largest of 128/64/32 that does (SmolLM's hidden size is 576)."""
    for g in (requested, 128, 64, 32, 16):
        if g and all(d % g == 0 for d in dims):
            return g
    raise ValueError(f"no INT4 group size divides {dims}")


def new_graph(name: str) -> ModelGraph:
    return ModelGraph(name, ModuleBuilder())


def np_dtype(dtype: str):
    return {"f32": np.float32, "i8": np.int8}[dtype]
