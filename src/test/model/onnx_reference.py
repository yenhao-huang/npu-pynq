"""Golden reference for the official ONNX INT8 SmolLM: interpret the graph op by op.

The compiled path (src/runtime/core/engine.py) runs the same model through the
MLIR compiler. This module skips the compiler entirely: it reads the exported
ONNX graph (graph.json + tensors.bin) and evaluates every node with NumPy, so a
compiler bug cannot hide in both sides of a comparison. ``integer_gemm`` routes
MatMulInteger through the NPU and ``float_ops`` routes the ORT-contract
operators through onnxruntime, to isolate the hardware or ORT from the
compiler; ``capture`` records intermediate tensors for layer-by-layer checks.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from src.runtime.core.package import onnx_asset_dir
from src.runtime.core.tokenizer import Tokenizer
from src.runtime.cpu.ort import ORT_OPS


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


class ONNXReferenceEngine:
    def __init__(self, model_dir, integer_gemm=None, float_ops=None):
        model_dir = Path(model_dir)
        # Accept the previous experiment bundle layout for existing notebooks.
        if not (onnx_asset_dir(model_dir, "onnx-graph")).is_dir():
            model_dir = model_dir / "packages"
        self.graph = ONNXGraph(onnx_asset_dir(model_dir, "onnx-graph"), integer_gemm, float_ops)
        self.tokenizer = Tokenizer(onnx_asset_dir(model_dir, "onnx"))
        self.max_seq = 512
        self.reset()

    def reset(self):
        self.past = {f"past_key_values.{i}.{kind}": np.empty((1,3,0,64), np.float32)
                     for i in range(30) for kind in ("key", "value")}
        self.logits = None

    def forward(self, ids):
        length = next(iter(self.past.values())).shape[2]
        if length + len(ids) > self.max_seq:
            raise ValueError("Context exceeds configured board memory limit")
        result = self.graph.run({"input_ids": np.asarray([ids], np.int64),
                                 "attention_mask": np.ones((1,length+len(ids)), np.int64),
                                 "position_ids": np.arange(length,length+len(ids),dtype=np.int64)[None], **self.past})
        self.past = {k.replace("present.","past_key_values."): v for k,v in result.items() if k != "logits"}
        self.logits = result["logits"][:, -1]
        return self.logits

    def prefill(self, ids):
        return self.forward(ids)

    def decode(self, token):
        return self.forward([token])

    def generate(self, prompt, max_new_tokens=24, chat=True, on_token=None):
        self.reset()
        ids = self.tokenizer.encode(self.tokenizer.chat([{"role":"user","content":prompt}]) if chat else prompt)
        start = time.monotonic()
        self.prefill(ids)
        prefill = time.monotonic()-start
        generated, times = [], []
        for _ in range(min(max_new_tokens, self.max_seq-len(ids))):
            token = int(self.logits[0].argmax())
            generated.append(token)
            if on_token:
                on_token(token, self.tokenizer.decode([token]))
            if token in self.tokenizer.special_ids:
                break
            start = time.monotonic()
            self.decode(token)
            times.append(time.monotonic()-start)
        return generated, self.tokenizer.decode(generated), {"prefill_seconds":prefill,"per_token_seconds":times}


def __getattr__(name):
    # Saved notebooks refer to these old names; all new callers use the names above.
    aliases = {'OfficialGraph': ONNXGraph, 'OfficialEngine': ONNXReferenceEngine}
    if name in aliases:
        return aliases[name]
    raise AttributeError(name)
