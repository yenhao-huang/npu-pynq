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
import json
from pathlib import Path


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
        "decode": ["weights", *(s.name for s in state), "token", "pos", "logits"],
        "prefill": ["weights", *(s.name for s in state), "tokens", "pos", "last", "logits"],
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


class Dequantized:
    def __init__(self, q, scale, zero, axis=1):
        self.q, self.scale, self.zero, self.axis = q, scale, zero, axis

    def materialize(self):
        shape = [1] * self.q.ndim
        if self.scale.ndim:
            shape[self.axis] = self.scale.size
        return (self.q.astype(np.float32) - self.zero.reshape(shape)) * self.scale.reshape(shape)


class ONNXGraph:
    def __init__(self, package, integer_gemm=None, float_ops=None):
        package = Path(package)
        self.manifest = json.loads((package / "graph.json").read_text())
        # One mapping avoids exhausting ARM process file descriptors when the
        # notebook holds both the physical engine and its CPU reference.
        self._tensor_bytes = np.memmap(package / "tensors.bin", mode="r", dtype=np.uint8)
        self.weights = {name: np.ndarray(tuple(spec["shape"]), dtype=spec["dtype"],
                                        buffer=self._tensor_bytes, offset=spec["offset"])
                        for name, spec in self.manifest["tensors"].items()}
        self.integer_gemm = integer_gemm
        self.float_ops = float_ops
        self.calls = 0
        self.last_uses = {}
        for index, node in enumerate(self.manifest["nodes"]):
            for name in node["inputs"]:
                self.last_uses[name] = index

    @staticmethod
    def norm(x, gamma, epsilon):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(epsilon)) * gamma

    def run(self, inputs, capture=None):
        env = dict(self.weights)
        env.update(inputs)
        outputs = set(self.manifest["outputs"])
        for index, node in enumerate(self.manifest["nodes"]):
            args = [env[n] if n else None for n in node["inputs"]]
            result = self.op(node["op"], args, node["attrs"])
            if not isinstance(result, tuple):
                result = (result,)
            for name, value in zip(node["outputs"], result):
                if name:
                    env[name] = value
                    if capture is not None and name in capture:
                        capture[name] = np.asarray(value).copy()
            for name in node["inputs"]:
                if name and self.last_uses[name] == index and name not in outputs and name not in self.weights:
                    env.pop(name, None)
        return {name: env[name] for name in self.manifest["outputs"]}

    def op(self, op, x, a):
        if self.float_ops is not None:
            from src.runtime.compiled import ORT_OPS
            if op in ORT_OPS:
                return self.float_ops(op, x)
        if op == "Shape": return np.asarray(x[0].shape, dtype=np.int64)
        if op == "Gather": return np.take(x[0], x[1].astype(np.intp), axis=a.get("axis", 0))
        if op == "Transpose": return x[0].transpose(a.get("perm"))
        if op == "ReduceSum": return x[0].sum(axis=tuple(x[1]), keepdims=bool(a.get("keepdims", 1)))
        if op == "Sub": return x[0] - x[1]
        if op == "Mul": return x[0] * x[1]
        if op == "Sigmoid": return np.float32(1) / (np.float32(1) + np.exp(-x[0]))
        if op == "Cast": return x[0].astype({1: np.float32, 6: np.int32, 7: np.int64}[a["to"]])
        if op == "DequantizeLinear":
            q = Dequantized(x[0], x[1], x[2], a.get("axis", 1))
            return q if x[0].size > 1000000 else q.materialize()
        if op == "SimplifiedLayerNormalization": return self.norm(x[0], x[1], a["epsilon"])
        if op == "SkipSimplifiedLayerNormalization":
            residual = x[0] + x[1]
            return self.norm(residual, x[2], a["epsilon"]), None, None, residual
        if op == "DynamicQuantizeLinear":
            lo = min(np.float32(0), x[0].min())
            hi = max(np.float32(0), x[0].max())
            scale = np.float32((hi - lo) / np.float32(255)) if hi != lo else np.float32(1)
            zero = np.asarray(np.clip(np.rint(-lo / scale), 0, 255), dtype=np.uint8)
            q = np.clip(np.rint(x[0] / scale) + zero, 0, 255).astype(np.uint8)
            return q, np.asarray(scale), zero
        if op == "MatMulInteger":
            az = x[2] if len(x) > 2 and x[2] is not None else 0
            bz = x[3] if len(x) > 3 and x[3] is not None else 0
            self.calls += 1
            if self.integer_gemm is None:
                return (x[0].astype(np.int32) - az) @ (x[1].astype(np.int32) - bz)
            signed = (x[0].astype(np.int16) - 128).astype(np.int8)
            shape = signed.shape
            raw = self.integer_gemm(signed.reshape(-1, shape[-1]), x[1]).reshape(shape[:-1] + (x[1].shape[-1],))
            correction = (128 - np.asarray(az, dtype=np.int32)) * x[1].sum(axis=0, dtype=np.int32)
            correction = correction - np.asarray(bz, dtype=np.int32) * ((x[0].astype(np.int32) - az).sum(axis=-1, keepdims=True))
            return raw + correction
        if op == "RotaryEmbedding":
            value, positions, cos, sin = x
            dim = cos.shape[-1] * 2
            shaped = value.reshape(value.shape[:2] + (-1, dim))
            first, second = np.split(shaped, 2, axis=-1)
            c, s = cos[positions][:, :, None, :], sin[positions][:, :, None, :]
            return np.concatenate((first * c - second * s, second * c + first * s), axis=-1).reshape(value.shape)
        if op == "GroupQueryAttention":
            q, k, v, past_k, past_v = x[:5]
            batch, seq = q.shape[:2]
            heads, kv_heads = a["num_heads"], a["kv_num_heads"]
            q = q.reshape(batch, seq, heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq, kv_heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq, kv_heads, -1).transpose(0, 2, 1, 3)
            k, v = np.concatenate((past_k, k), axis=2), np.concatenate((past_v, v), axis=2)
            score = (q @ np.repeat(k, heads // kv_heads, axis=1).swapaxes(-1, -2)) * np.float32(a["scale"])
            causal = np.arange(k.shape[2])[None, :] > (past_k.shape[2] + np.arange(seq))[:, None]
            score = np.where(causal, -np.inf, score)
            probability = np.exp(score - score.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            y = probability @ np.repeat(v, heads // kv_heads, axis=1)
            return y.transpose(0, 2, 1, 3).reshape(batch, seq, -1), k, v
        if op == "MatMul":
            if isinstance(x[1], Dequantized):
                w = x[1]
                assert w.scale.size == 1 and w.zero.size == 1
                out = np.empty(x[0].shape[:-1] + (w.q.shape[-1],), dtype=np.float32)
                for start in range(0, w.q.shape[-1], 1024):
                    panel = (w.q[:, start:start+1024].astype(np.float32) - w.zero) * w.scale
                    out[..., start:start+1024] = x[0] @ panel
                return out
            return x[0] @ x[1]
        raise NotImplementedError(op)


def __getattr__(name):
    if name == 'OfficialGraph':
        return ONNXGraph
    raise AttributeError(name)
