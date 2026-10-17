"""MLIR/LLVM compiler: partition, ISA encoding parity, and host execution.

Skipped when no LLVM 22 with MLIR is installed (the hosted CI image).
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.compiler import toolchain
from src.isa import isa, layout
import os
import shutil
import subprocess
import sys
from src.inference.graph import ONNXGraph
from src.inference.tokenizer import Tokenizer
from src.inference.tokenizer import bytes_to_unicode
from src.compiler.export import ModelGraph, export
from src.compiler.mlir_builder import ModuleBuilder
from src.compiler.ops.context import Ctx
from src.compiler.ops.linear import projection, native, multiply
from src.runtime.compiled import CompiledModel


try:
    toolchain.llvm_bin()
    HAVE_LLVM = True
except toolchain.ToolchainError:
    HAVE_LLVM = False


def build_mlp(m, k, h, n, bits2, group2, rng):
    from src.compiler.export import ModelGraph
    from src.compiler.mlir_builder import ModuleBuilder, memref_type
    from src.compiler.ops import core, linear as L
    from src.compiler.ops.context import Ctx

    w1 = L.quantize_weight(rng.normal(size=(h, k)).astype(np.float32) * 0.05, 8)
    w2 = L.quantize_weight(rng.normal(size=(n, h)).astype(np.float32) * 0.1, bits2, group=group2)
    bias = np.linspace(-1, 1, h).astype(np.float32)
    mod = ModuleBuilder()
    f = mod.func("forward", [("ws", "memref<?xi8>"), ("x", memref_type([m, k], "f32")),
                             ("out", memref_type([m, n], "f32"))])
    graph = ModelGraph("mlp", mod)
    ctx = Ctx(f, graph, f.arg("ws"))
    x = f.to_tensor(f.arg("x"))
    y = L.linear(ctx, core.relu(f, L.linear(ctx, x, "fc1", w1, bias)), "fc2", w2)
    f.store(y, f.arg("out"))
    f.emit("return")

    def reference(x_np):
        return L.ref_linear(np.maximum(L.ref_linear(x_np, w1, bias), 0), w2)

    return graph, reference


@unittest.skipUnless(HAVE_LLVM, "LLVM 22 with MLIR is not installed")
class CompilerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def compile_and_run(self, bits2, group2, targets=("host",)):
        from src.compiler.export import export
        from src.runtime.compiled import CompiledModel

        rng = np.random.default_rng(bits2)
        m, k, h, n = 5, 300, 48, 40
        graph, reference = build_mlp(m, k, h, n, bits2, group2, rng)
        pkg = export(graph, self.root / f"mlp{bits2}", targets=targets)
        model = CompiledModel(pkg, "sim")
        x = rng.normal(size=(m, k)).astype(np.float32)
        out = np.zeros((m, n), np.float32)
        model.call("forward", "weights", x, out)
        return pkg, out, reference(x), model.stats()

    def test_w8_and_w4_mlp_match_numpy_on_the_simulated_npu(self):
        for bits, group in ((8, 0), (4, 16)):
            with self.subTest(bits=bits):
                pkg, out, ref, stats = self.compile_and_run(bits, group)
                np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
                self.assertEqual(stats["calls"], 2)
                report = json.loads((pkg / "partition.json").read_text())
                self.assertEqual([t["weight"] for t in report["npu_tasks"]], ["fc1.q", "fc2.q"])
                self.assertEqual(report["npu_tasks"][1]["source"], "grouped" if bits == 4 else "linalg.matmul")

    def test_streamed_segments_and_packed_int4(self):
        from src.compiler.export import export
        from src.runtime.compiled import CompiledModel

        rng = np.random.default_rng(11)
        m, k, h, n = 3, 300, 64, 200
        graph, reference = build_mlp(m, k, h, n, 4, 32, rng)
        pkg = export(graph, self.root / "streamed", targets=("host",), segment_bytes=2048)
        table = json.loads((pkg / "programs.json").read_text())
        self.assertGreater(sum(len(t["segments"]) for t in table["tasks"]), 4)
        manifest = json.loads((pkg / "manifest.json").read_text())
        self.assertTrue(manifest["weight_layouts"]["fc2.q"].startswith("npu_b4"))
        x = rng.normal(size=(m, k)).astype(np.float32)
        model = CompiledModel(pkg, "sim")
        self.assertTrue(model.stream_weights, "packed INT4 weights must stream")
        out = np.zeros((m, n), np.float32)
        model.call("forward", "weights", x, out)
        np.testing.assert_allclose(out, reference(x), rtol=1e-5, atol=1e-5)
        # INT8 weights run resident or streamed, with the same result.
        graph, reference = build_mlp(m, k, h, n, 8, 0, rng)
        pkg = export(graph, self.root / "w8", targets=("host",), segment_bytes=2048)
        for stream in (False, True):
            model = CompiledModel(pkg, "sim", stream_weights=stream)
            out = np.zeros((m, n), np.float32)
            model.call("forward", "weights", x, out)
            np.testing.assert_allclose(out, reference(x), rtol=1e-5, atol=1e-5)

    def test_cpp_encoder_matches_python_layout(self):
        pkg, *_ = self.compile_and_run(8, 0)
        table = json.loads((pkg / "programs.json").read_text())
        words = isa.unpack((pkg / "programs.bin").read_bytes())
        for t in table["tasks"]:
            plan = layout.plan_gemm(t["m"], t["n"], t["k"], isa.Limits(), t["tk"])
            self.assertEqual(t["c_base"] + t["c_bytes"], isa.align(plan.a_bytes, 64) + plan.c_bytes)
            for g in t["segments"]:
                expected = layout.encode(plan, 0, 0, isa.align(plan.a_bytes, 64), g["ni_lo"], g["ni_hi"]) + [isa.end()]
                self.assertEqual(words[g["word_offset"]:g["word_offset"] + g["length"]], expected)
                self.assertEqual(g["weight_offset"] - t["weight_offset"], g["ni_lo"] * plan.b_panel)

    def test_board_library_is_armv7_hard_float(self):
        pkg, *_ = self.compile_and_run(8, 0, targets=("host", "pynq"))
        header = (pkg / "pynq/model.so").read_bytes()[:52]
        self.assertEqual(header[:4], b"\x7fELF")
        self.assertEqual(header[4], 1)  # 32-bit
        self.assertEqual(int.from_bytes(header[18:20], "little"), 40)  # EM_ARM
        flags = int.from_bytes(header[36:40], "little")
        self.assertTrue(flags & 0x400, "EF_ARM_ABI_FLOAT_HARD")


class ONNXGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.package = Path(self.tmp.name)
        tensors = {}
        with (self.package / 'tensors.bin').open('wb') as f:
            for i in range(800):
                tensors[str(i)] = {'offset': f.tell(), 'dtype': '<i8', 'shape': []}
                f.write(np.asarray(i, dtype='<i8').tobytes())
                f.write(bytes(56))
        (self.package / 'graph.json').write_text(json.dumps(
            {'tensors': tensors, 'nodes': [], 'inputs': [], 'outputs': []}), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_models_preserve_all_readonly_scalars(self):
        for graph in (ONNXGraph(self.package), ONNXGraph(self.package)):
            for i in range(800):
                value = graph.weights[str(i)]
                self.assertEqual(value.shape, ())
                self.assertEqual(int(value), i)
                with self.assertRaises(ValueError):
                    value[...] = -1

    @unittest.skipUnless(os.name == 'posix', 'POSIX descriptor limit')
    def test_two_models_under_low_descriptor_limit(self):
        import resource
        previous = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, previous[0]), previous[1]))
            a, b = ONNXGraph(self.package), ONNXGraph(self.package)
            self.assertEqual(int(a.weights['799']), int(b.weights['799']))
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, previous)

    def test_unsigned_activation_and_weight_zero_points(self):
        graph = ONNXGraph(self.package)
        calls = []
        def signed_gemm(a, b):
            self.assertEqual(a.dtype, np.int8)
            self.assertEqual(b.dtype, np.int8)
            calls.append(a.shape)
            return (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
        graph.integer_gemm = signed_gemm
        rng = np.random.default_rng(126)
        a = rng.integers(0, 256, (2, 3, 37), dtype=np.uint8)
        a.flat[0], a.flat[1] = 0, 255
        b = rng.integers(-128, 128, (37, 19), dtype=np.int8)
        for az in (0, 128, 255):
            for bz in (np.int8(-9), np.int8(7), np.arange(-9, 10, dtype=np.int8)):
                with self.subTest(activation_zero=az, weight_zero=str(bz)):
                    actual = graph.op('MatMulInteger', (a, b, np.uint8(az), bz), {})
                    expected = (a.astype(np.int64) - az) @ (b.astype(np.int64) - bz.astype(np.int64))
                    np.testing.assert_array_equal(actual, expected.astype(np.int32))
        self.assertEqual(calls, [(6, 37)] * 9)


class TokenizerTest(unittest.TestCase):
    def test_utf8_bpe_and_onnx_system_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            vocab = {c: i for i, c in bytes_to_unicode().items()}
            spec = {'model': {'type': 'BPE', 'vocab': vocab, 'merges': []},
                    'pre_tokenizer': {'type': 'Split', 'pattern': {'Regex': '.+'}}}
            (package / 'tokenizer.json').write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8')
            system = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face'
            (package / 'tokenizer_config.json').write_text(json.dumps({'chat_template':
                "{{ '<|im_start|>system\n" + system + "<|im_end|>\n' }}"}), encoding='utf-8')
            tokenizer = Tokenizer(package)
            text = 'caf\u00e9 \u4f60\u597d'
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.chat([{'role': 'user', 'content': 'Hello'}]),
                '<|im_start|>system\n' + system + '<|im_end|>\n'
                '<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n')
            explicit = tokenizer.chat([{'role': 'system', 'content': 'Keep it short.'},
                                       {'role': 'user', 'content': 'Hello'}])
            self.assertEqual(explicit.count('<|im_start|>system\n'), 1)
            self.assertIn('Keep it short.', explicit)
            self.assertNotIn(system, explicit)


class ONNXSourceIndependenceTest(unittest.TestCase):
    def test_entry_points_with_only_src_present(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            isolated = Path(tmp)
            for source in (root / 'src').rglob('*'):
                if source.is_file() and source.suffix in ('.py', '.json'):
                    target = isolated / source.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            env = dict(os.environ, PYTHONPATH=str(isolated))
            for command in ((), ("run",), ("reference",), ("prepare-onnx",), ("export",)):
                result = subprocess.run([sys.executable, '-m', 'src.inference.cli', *command, '--help'],
                    cwd=isolated, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, '-c',
                'from src.runtime.isa_runtime import NpuGemm; '
                'from src.runtime.compiled import NativeOrtOperators; '
                'from src.inference.engine import LLMEngine'],
                cwd=isolated, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((isolated / 'exp').exists())


class ONNXCompiledQuantizationTest(unittest.TestCase):
    def test_native_read_boundary_preserves_residual(self):
        try:
            toolchain.llvm_bin()
        except toolchain.ToolchainError:
            self.skipTest('LLVM 22 with MLIR is unavailable')
        from src.compiler.mlir_builder import identity_map
        graph=ModelGraph('onnx-native-residual',ModuleBuilder())
        f=graph.module.func('run',[('ws','memref<?xi8>'),('x','memref<3x7xf32>'),
                                 ('out','memref<3x7xf32>')])
        ctx=Ctx(f,graph,f.arg('ws'))
        source=f.to_tensor(f.arg('x'))
        x=f.generic([], [f.empty([3,7],'f32')], [identity_map(2)],
            ['parallel','parallel'],lambda r:[r.extract(source,[r.index(0),r.index(1)])])[0]
        y=native(ctx,'onnx_test_copy',[x],[([3,7],'f32')])[0]
        squared=multiply(f,y,y)
        out=native(ctx,'onnx_test_add',[x,squared],[([3,7],'f32')])[0]
        f.store(out,f.arg('out')); f.emit('return')
        graph.entry_points['run']=['weights','x','out']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=root/'native.c'; obj=root/'native.o'
            c.write_text('''#include <stdint.h>
typedef struct { void *allocated,*aligned; intptr_t offset,sizes[2],strides[2]; } M;
void _mlir_ciface_onnx_test_copy(M *x,M *y) {
 for(int i=0;i<21;i++) ((float*)y->aligned)[y->offset+i]=((float*)x->aligned)[x->offset+i];
}
void _mlir_ciface_onnx_test_add(M *x,M *y,M *z) {
 for(int i=0;i<21;i++) ((float*)z->aligned)[z->offset+i]=((float*)x->aligned)[x->offset+i]+((float*)y->aligned)[y->offset+i];
}
''')
            toolchain.run([toolchain.tool('clang'),'-O2','-fPIC','-c',str(c),'-o',str(obj)],
                          'compiling test native kernels')
            package=export(graph,root/'package',targets=('host',),extra_objects={'host':(obj,)})
            model=CompiledModel(package)
            x=np.arange(21,dtype=np.float32).reshape(3,7)/4
            actual=np.empty_like(x)
            model.call('run','weights',x,actual)
            np.testing.assert_array_equal(actual,x+x*x)

    def test_fused_columns_keep_original_scales_and_unsigned_zero(self):
        try:
            toolchain.llvm_bin()
        except toolchain.ToolchainError:
            self.skipTest('LLVM 22 with MLIR is unavailable')
        graph=ModelGraph('onnx-projection-test',ModuleBuilder())
        f=graph.module.func('run',[('ws','memref<?xi8>'),('q','memref<3x7xi8>'),
            ('scale','memref<1xf32>'),('zero','memref<1xi8>'),('out','memref<3x9xf32>')])
        ctx=Ctx(f,graph,f.arg('ws'))
        rng=np.random.default_rng(119)
        weights=rng.integers(-128,128,(7,9),dtype=np.int8)
        scales=np.asarray([.01]*3+[.05]*6,np.float32)
        activation=tuple(f.to_tensor(f.arg(name)) for name in ('q','scale','zero'))
        out=projection(ctx,activation,'fused',weights,scales)
        f.store(out,f.arg('out')); f.emit('return')
        graph.entry_points['run']=['weights','q','scale','zero','out']
        q=rng.integers(0,256,(3,7),dtype=np.uint8)
        q.flat[0],q.flat[1]=0,255
        xs=np.asarray([.03125],np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            package=export(graph,Path(tmp)/'package',targets=('host',))
            model=CompiledModel(package,'sim')
            for zero in (0,127,128,130,255):
                with self.subTest(zero=zero):
                    actual=np.empty((3,9),np.float32)
                    model.call('run','weights',q,xs,np.asarray([zero],np.uint8),actual)
                    integer=(q.astype(np.int64)-zero)@weights.astype(np.int64)
                    expected=integer.astype(np.float32)*(xs[0]*scales)
                    np.testing.assert_array_equal(actual,expected)
            self.assertGreater(model.stats()['jobs'],0)
            self.assertEqual(model.stats()['error'],0)


class SharedExportEngineTest(unittest.TestCase):
    def test_safetensors_source_exports_and_runs_original_scalar_abi(self):
        try:
            toolchain.llvm_bin()
        except toolchain.ToolchainError:
            self.skipTest('LLVM 22 with MLIR is unavailable')
        import struct
        from src.inference.engine import LLMEngine
        from src.inference.graph import LLMOptions
        rng = np.random.default_rng(119)
        cfg = dict(hidden_size=8, intermediate_size=16, num_attention_heads=2,
                   num_key_value_heads=1, num_hidden_layers=1, rms_norm_eps=1e-5,
                   vocab_size=8, tie_word_embeddings=True)
        arrays = {'model.embed_tokens.weight':rng.normal(size=(8,8)).astype(np.float32),
                  'model.norm.weight':np.ones(8,np.float32)}
        prefix = 'model.layers.0.'
        for name, shape in {'self_attn.q_proj.weight':(8,8), 'self_attn.k_proj.weight':(4,8),
                            'self_attn.v_proj.weight':(4,8), 'self_attn.o_proj.weight':(8,8),
                            'mlp.gate_proj.weight':(16,8), 'mlp.up_proj.weight':(16,8),
                            'mlp.down_proj.weight':(8,16)}.items():
            arrays[prefix+name] = (rng.normal(size=shape)*.1).astype(np.float32)
        for name in ('input_layernorm.weight','post_attention_layernorm.weight'):
            arrays[prefix+name] = np.ones(8,np.float32)
        header, blocks, offset = {}, [], 0
        for name, array in arrays.items():
            block = array.tobytes()
            header[name] = {'dtype':'F32','shape':list(array.shape),
                            'data_offsets':[offset,offset+len(block)]}
            blocks.append(block); offset += len(block)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'model.safetensors'
            encoded = json.dumps(header).encode()
            checkpoint.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(blocks))
            (root/'config.json').write_text(json.dumps(cfg))
            (root/'tokenizer.json').write_text(json.dumps({'model':{'type':'BPE',
                'vocab':{c:i for i,c in enumerate('abcdefgh')},'merges':[]},'added_tokens':[]}))
            package = export(checkpoint,root/'package',model='smollm',targets=('host',),
                opts=LLMOptions(chunk=3,max_seq=16,group=8,attention='naive'))
            engine = LLMEngine(package,'sim')
            ids, _, stats = engine.generate([1,2,3,4],max_new_tokens=3)
            self.assertEqual(len(ids),3)
            self.assertTrue(np.isfinite(engine.logits).all())
            self.assertGreater(stats.npu['jobs'],0)
            # Read an original commit's placeholder state ABI as well.
            manifest = engine.model.manifest
            manifest['entry_points']['decode'] = ['weights','state','token','pos','logits']
            manifest['entry_points']['prefill'] = ['weights','state','tokens','pos','last','logits']
            repeated, _, _ = engine.generate([1,2,3,4],max_new_tokens=3)
            self.assertEqual(repeated,ids)
            with self.assertRaises(ValueError):
                export(checkpoint,root,model='smollm',targets=('host',),
                       opts=LLMOptions(chunk=3,max_seq=16,group=8,attention='naive'))
            self.assertTrue(checkpoint.is_file())

    def test_auto_onnx_source_uses_common_model_builder(self):
        from unittest.mock import patch
        from src.models.smollm import model
        from src.inference.graph import LLMOptions
        from src.inference.engine import LLMEngine
        with patch.object(model,'_build_quantized_onnx',return_value=('graph','source_model')) as build, \
             patch('src.compiler.export.attach_float_runtime'), \
             patch.object(model,'Config',return_value='config'), \
             patch.object(Path,'read_text',return_value='{}'):
            graph, cfg = model.build('assets/onnx/model_int8.onnx',
                                     opts=LLMOptions(),prefill_lengths=(42,))
            self.assertEqual((graph,cfg),('graph','config'))
            self.assertEqual(build.call_args.args[1],(42,))


class SmolLMCDTest(unittest.TestCase):
    def helper(self):
        import importlib.util
        path = Path(__file__).resolve().parents[3] / '.github/cd/smollm.py'
        spec = importlib.util.spec_from_file_location('smollm_cd_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_frozen_output_rejects_one_token_or_text_difference(self):
        cd = self.helper()
        cd.check_output(list(cd.EXPECTED_IDS), cd.EXPECTED_TEXT)
        for ids, text in ((cd.EXPECTED_IDS[:-1], cd.EXPECTED_TEXT),
                          ([0] + cd.EXPECTED_IDS[1:], cd.EXPECTED_TEXT),
                          (cd.EXPECTED_IDS, cd.EXPECTED_TEXT + ' '), ([62], 'N')):
            with self.subTest(ids=ids, text=text), self.assertRaises(ValueError):
                cd.check_output(ids, text)

    def test_sealed_package_rejects_tampering_and_wrong_commit(self):
        from unittest.mock import patch
        cd = self.helper()
        commit = 'a' * 40
        with tempfile.TemporaryDirectory() as tmp, patch.object(cd, 'identity'):
            root = Path(tmp)
            asset = root / 'weights.bin'
            asset.write_bytes(b'original')
            cd.seal(root, commit, 'v0.1.0')
            cd.verify(root, commit, 'v0.1.0')
            with self.assertRaises(ValueError):
                cd.verify(root, 'b' * 40, 'v0.1.0')
            asset.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                cd.verify(root, commit, 'v0.1.0')


if __name__ == "__main__":
    unittest.main()
