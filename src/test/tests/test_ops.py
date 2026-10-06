"""Operator library: each op compiled through MLIR/LLVM on the host and
compared with its NumPy reference. Skipped without LLVM 22."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.compiler import toolchain

try:
    toolchain.llvm_bin()
    HAVE_LLVM = True
except toolchain.ToolchainError:
    HAVE_LLVM = False


def compile_host(build, args, root):
    """``build(ctx, values)`` emits a function over memref ``args`` [(name, type)]."""
    from src.compiler.export import ModelGraph, export
    from src.compiler.mlir_builder import ModuleBuilder
    from src.compiler.ops.context import Ctx
    from src.runtime.compiled import CompiledModel

    mod = ModuleBuilder()
    f = mod.func("forward", [("ws", "memref<?xi8>"), *args])
    graph = ModelGraph("op", mod)
    ctx = Ctx(f, graph, f.arg("ws"))
    build(ctx, {name: f.arg(name) for name, _ in args})
    f.emit("return")
    pkg = export(graph, Path(root) / "pkg", targets=("host",))
    return CompiledModel(pkg, "sim")


@unittest.skipUnless(HAVE_LLVM, "LLVM 22 with MLIR is not installed")
class AttentionTest(unittest.TestCase):
    def run_case(self, cfg, steps):
        from src.compiler.mlir_builder import memref_type
        from src.compiler.ops import attention as A

        T, H, HK, D = cfg.tokens, cfg.heads, cfg.kv_heads, cfg.head_dim
        shapes = cfg.cache_shapes()
        args = [("q", memref_type([T, H * D], "f32")), ("k", memref_type([T, HK * D], "f32")),
                ("v", memref_type([T, HK * D], "f32"))]
        args += [(f"c{n}", memref_type(s, d)) for n, (s, d) in shapes.items()]
        args += [("out", memref_type([T, H * D], "f32")), ("layer", "index"), ("pos", "index")]

        def build(ctx, v):
            b = ctx.b
            caches = {n: v[f"c{n}"] for n in shapes}
            o = A.attention(ctx, cfg, b.to_tensor(v["q"]), b.to_tensor(v["k"]), b.to_tensor(v["v"]),
                            caches, v["layer"], v["pos"])
            b.store(o, v["out"])

        with tempfile.TemporaryDirectory() as tmp:
            model = compile_host(build, args, tmp)
            np_dtype = {"f32": np.float32, "i8": np.int8}
            cache = {n: np.zeros(s, np_dtype[d]) for n, (s, d) in shapes.items()}
            ref_cache = {("ks" if n == "ks" else "vs" if n == "vs" else n): np.zeros(s, np_dtype[d])
                         for n, (s, d) in shapes.items()}
            rng = np.random.default_rng(7)
            for layer, pos in steps:
                q = rng.normal(size=(T, H * D)).astype(np.float32)
                k = rng.normal(size=(T, HK * D)).astype(np.float32)
                v = rng.normal(size=(T, HK * D)).astype(np.float32)
                out = np.zeros((T, H * D), np.float32)
                model.call("forward", "weights", q, k, v, *cache.values(), out, layer, pos)
                ref = A.ref_attention(cfg, q, k, v, ref_cache, layer, pos)
                np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)

    def test_variants(self):
        from src.compiler.ops.attention import AttnConfig

        for mode in ("naive", "online"):
            for kv in ("f32", "int8"):
                for tokens, heads, kv_heads in ((1, 6, 2), (4, 4, 1)):
                    with self.subTest(mode=mode, kv=kv, tokens=tokens, heads=heads, kv_heads=kv_heads):
                        cfg = AttnConfig(tokens, heads, kv_heads, 16, 2, 32, mode, kv)
                        steps = [(0, 0), (1, 0), (0, tokens), (0, 2 * tokens), (1, tokens)]
                        self.run_case(cfg, steps)



@unittest.skipUnless(HAVE_LLVM, "LLVM 22 with MLIR is not installed")
class ElementwiseOpsTest(unittest.TestCase):
    def test_rmsnorm_softmax_silu_rope_and_tied_lookup(self):
        from src.compiler.mlir_builder import memref_type
        from src.compiler.ops import core, embedding as E, linear as L
        from src.isa import isa

        rng = np.random.default_rng(3)
        T, heads, D, S, hidden, vocab = 3, 4, 16, 32, 300, 70
        x = rng.normal(size=(T, heads * D)).astype(np.float32)
        w = rng.normal(size=(heads * D,)).astype(np.float32)
        cos, sin = E.rope_tables(S, D, 10000.0)
        cos_p, sin_p = E.rope_tables(S, D // 4, 10000.0)  # partial rotary: frequencies over R dims
        emb = L.quantize_weight(rng.normal(size=(vocab, hidden)).astype(np.float32), 8)
        tokens = np.array([0, 69, 33], np.int32)
        limits = isa.Limits()  # the exporter's; hidden=300 spans two K chunks
        args = [("x", memref_type([T, heads * D], "f32")), ("tok", memref_type([T], "i32")),
                ("norm", memref_type([T, heads * D], "f32")), ("soft", memref_type([T, heads * D], "f32")),
                ("silu", memref_type([T, heads * D], "f32")), ("rope", memref_type([T, heads * D], "f32")),
                ("prope", memref_type([T, heads * D], "f32")), ("emb", memref_type([T, hidden], "f32")),
                ("logits", memref_type([T, vocab], "f32")), ("pos", "index")]

        def build(ctx, v):
            b = ctx.b
            xt = b.to_tensor(v["x"])
            b.store(core.rmsnorm(b, xt, ctx.weight("w", w), 1e-5), v["norm"])
            b.store(core.softmax_last(b, xt), v["soft"])
            b.store(core.silu(b, xt), v["silu"])
            c, s = ctx.weight("cos", cos), ctx.weight("sin", sin)
            b.store(E.rope(ctx, xt, heads, D, v["pos"], c, s), v["rope"])
            cp, sp = ctx.weight("cos_p", cos_p), ctx.weight("sin_p", sin_p)
            b.store(E.rope(ctx, xt, heads, D, v["pos"], cp, sp, rotary_dim=D // 4), v["prope"])
            rows = E.tied_lookup(ctx, b.to_tensor(v["tok"]), "emb", emb, limits)
            b.store(rows, v["emb"])
            b.store(L.linear(ctx, rows, "emb", emb), v["logits"])

        with tempfile.TemporaryDirectory() as tmp:
            model = compile_host(build, args, tmp)
            outs = {n: np.zeros(s, np.float32) for n, s in [
                ("norm", (T, heads * D)), ("soft", (T, heads * D)), ("silu", (T, heads * D)),
                ("rope", (T, heads * D)), ("prope", (T, heads * D)), ("emb", (T, hidden)), ("logits", (T, vocab))]}
            pos = 5
            model.call("forward", "weights", x, tokens, *outs.values(), pos)
        tol = dict(rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(outs["norm"], core.ref_rmsnorm(x, w, 1e-5), **tol)
        np.testing.assert_allclose(outs["soft"], core.ref_softmax(x), **tol)
        np.testing.assert_allclose(outs["silu"], core.ref_silu(x), **tol)
        np.testing.assert_allclose(outs["rope"], E.ref_rope(x, heads, D, pos, cos, sin), **tol)
        np.testing.assert_allclose(outs["prope"], E.ref_rope(x, heads, D, pos, cos_p, sin_p, D // 4), **tol)
        np.testing.assert_allclose(outs["emb"], emb.dequantize().T[tokens], **tol)
        np.testing.assert_allclose(outs["logits"], L.ref_linear(outs["emb"], emb), rtol=1e-4, atol=1e-4)


class FrameworkTest(unittest.TestCase):
    def test_sampling(self):
        from src.inference.sampling import Sampler, SamplingParams

        logits = np.array([0.0, 3.0, 1.0, 2.9])
        self.assertEqual(Sampler(SamplingParams())(logits, []), 1)
        self.assertEqual(Sampler(SamplingParams(repetition_penalty=2.0))(logits, [1]), 3)
        picks = {Sampler(SamplingParams(temperature=1.0, top_k=2, seed=s))(logits, []) for s in range(40)}
        self.assertEqual(picks, {1, 3})
        picks = {Sampler(SamplingParams(temperature=1.0, top_p=0.05, seed=s))(logits, []) for s in range(10)}
        self.assertEqual(picks, {1})

    def test_safetensors_reader(self):
        import json
        import struct
        from src.inference.weights import SafeTensors

        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        bf16 = (a.view(np.uint32) >> 16).astype(np.uint16)
        header = {"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
                  "b": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [24, 36]}}
        blob = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            path.write_bytes(struct.pack("<Q", len(blob)) + blob + a.tobytes() + bf16.tobytes())
            st = SafeTensors(path)
            np.testing.assert_array_equal(st["a"], a)
            np.testing.assert_array_equal(st["b"], a)

    def test_tokenizer_round_trip_without_regex_package(self):
        import json
        import sys
        from src.inference.tokenizer import Tokenizer, bytes_to_unicode

        enc = bytes_to_unicode()
        vocab = {enc[b]: b for b in range(256)}
        vocab["ĠĠ"] = 256
        vocab["<|im_end|>"] = 257
        spec = {"model": {"type": "BPE", "vocab": vocab, "merges": ["Ġ Ġ"]},
                "added_tokens": [{"id": 257, "content": "<|im_end|>", "special": True}],
                "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}
        saved = sys.modules.get("regex")
        sys.modules["regex"] = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "tokenizer.json").write_text(json.dumps(spec))
                tok = Tokenizer(tmp)
                text = "naïve  test<|im_end|>"
                ids = tok.encode(text)
                self.assertIn(257, ids)
                self.assertEqual(tok.decode(ids, skip_special=False), text)
        finally:
            if saved is None:
                del sys.modules["regex"]
            else:
                sys.modules["regex"] = saved


if __name__ == "__main__":
    unittest.main()
