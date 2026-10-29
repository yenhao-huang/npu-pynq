"""Imported ONNX semantics cross the MLIR version boundary without ORT."""

import unittest
import ctypes
import tempfile
from pathlib import Path

import numpy as np

from src.export.compilers.mlir.builder import ModuleBuilder
from src.export.compilers.onnx_mlir.lowering import elementwise
from src.export.compilers.onnx_mlir import lowering
from src.export.compilers.mlir import toolchain
from src.export.compilers.mlir.builder import memref_type
from src.export.compilers.mlir.builder import float_literal

try:
    toolchain.llvm_bin()
    import onnxruntime
    HAVE_COMPILER_REFERENCE = True
except (toolchain.ToolchainError, ImportError):
    HAVE_COMPILER_REFERENCE = False


class ImportedLoweringTests(unittest.TestCase):
    def test_constant_formatting_preserves_values_near_fp32_midpoints(self):
        value = 6.08574864600143e-5
        self.assertEqual(float(float_literal(value)), value)
        self.assertEqual(np.float32(float_literal(value)), np.float32(value))

    def test_sigmoid_emits_computation_without_external_calls_or_fast_math(self):
        module = ModuleBuilder()
        function = module.func('forward', [('input', 'memref<2x4xf32>'),
                                           ('output', 'memref<2x4xf32>')])
        value = function.to_tensor(function.arg('input'))
        result = elementwise(function, {'op': 'onnx.Sigmoid'}, [value])
        function.store(result, function.arg('output'))
        function.emit('return')
        text = module.render()
        self.assertIn('arith.divf', text)
        self.assertNotIn('fastmath', text)
        self.assertNotIn('func.call', text)
        self.assertNotIn('onnx_', text)

    def test_unknown_operation_fails_instead_of_falling_back_to_a_runtime(self):
        with self.assertRaisesRegex(NotImplementedError, 'onnx.Unknown'):
            elementwise(None, {'op': 'onnx.Unknown'}, [])

    def test_custom_conversion_rejects_semantics_it_does_not_implement(self):
        for attribute in ({'interleaved': 1}, {'unexpected': 1}):
            node = {'op': 'onnx.Custom', 'attributes': {
                'function_name': 'RotaryEmbedding', 'domain_name': 'com.microsoft', **attribute}}
            with self.assertRaises(NotImplementedError):
                lowering.custom(None, node, [])


@unittest.skipUnless(HAVE_COMPILER_REFERENCE, 'LLVM 22 and ONNX Runtime reference are required')
class CompiledOperatorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.export.ir.graph import ModelGraph
        from src.export.compilers.mlir.ops.context import Ctx
        from src.export.compilers.mlir.ops.attention import AttnConfig
        from src.export.targets import target

        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        module = ModuleBuilder()
        graph = ModelGraph('onnx-operators', module)

        def function(name, shapes):
            return module.func(name, [(label, memref_type(shape, dtype))
                                     for label, shape, dtype in shapes])

        shape = (2, 64)
        for name in ('norm', 'skip', 'sigmoid', 'quant'):
            arguments = [('x', shape, 'f32')]
            if name == 'skip':
                arguments += [('y', shape, 'f32')]
            if name in ('norm', 'skip'):
                arguments += [('gamma', (64,), 'f32')]
            arguments += [('out', shape, 'i8' if name == 'quant' else 'f32')]
            if name == 'skip':
                arguments += [('residual', shape, 'f32')]
            if name == 'quant':
                arguments += [('scale', (1,), 'f32'), ('zero', (1,), 'i8')]
            f = function(name, arguments)
            x = f.to_tensor(f.arg('x'))
            if name == 'norm':
                outputs = [lowering.rms_norm(f, x, f.to_tensor(f.arg('gamma')))]
            elif name == 'skip':
                outputs = lowering.skip_norm(f, x, f.to_tensor(f.arg('y')), f.to_tensor(f.arg('gamma')))
            elif name == 'sigmoid':
                outputs = [lowering.sigmoid(f, x)]
            else:
                outputs = lowering.dynamic_quantize(f, x)
            labels = {'skip': ('out', 'residual'), 'quant': ('out', 'scale', 'zero')}.get(name, ('out',))
            for value, label in zip(outputs, labels):
                f.store(value, f.arg(label))
            f.emit('return')

        f = function('rope', [('x', (3, 24), 'f32'), ('cos', (16, 4), 'f32'),
            ('sin', (16, 4), 'f32'), ('out', (3, 24), 'f32')])
        # Position is a runtime scalar, just as it is in a compiled model.
        from src.export.compilers.mlir.builder import Value
        f.args.append(Value('%pos', 'index'))
        result = lowering.rotary(f, f.to_tensor(f.arg('x')), f.to_tensor(f.arg('cos')),
                                 f.to_tensor(f.arg('sin')), f.arg('pos'), head_dim=8)
        f.store(result, f.arg('out'))
        f.emit('return')

        for tokens, heads, kv_heads, dim, context in ((3, 6, 2, 8, 16), (1, 6, 2, 8, 16),
                                                     (40, 9, 3, 64, 64), (1, 9, 3, 64, 64)):
            cache_shape = (2, kv_heads, context, dim)
            name = 'attn_decode_real' if tokens == 1 and dim == 64 else f'attn{tokens}'
            f = function(name, [('q', (tokens, heads * dim), 'f32'),
                ('k', (tokens, kv_heads * dim), 'f32'), ('v', (tokens, kv_heads * dim), 'f32'),
                ('key', cache_shape, 'f32'), ('value', cache_shape, 'f32'),
                ('out', (tokens, heads * dim), 'f32')])
            f.args += [Value('%layer', 'index'), Value('%pos', 'index')]
            ctx = Ctx(f, graph, None)
            config = AttnConfig(tokens, heads, kv_heads, dim, 2, context, mode='naive', scale=0.125)
            output = lowering.grouped_attention(ctx, config, *[f.to_tensor(f.arg(n)) for n in ('q', 'k', 'v')],
                {'k': f.arg('key'), 'v': f.arg('value')}, f.arg('layer'), f.arg('pos'))
            f.store(output, f.arg('out'))
            f.emit('return')

        source = cls.root / 'operators.mlir'
        source.write_text(module.render())
        if 'fastmath' in source.read_text():
            raise AssertionError('Imported operator lowering must preserve FP32 evaluation')
        cls.library = ctypes.CDLL(str(toolchain.codegen(source, cls.root, target('host'))))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def call(self, name, *args):
        from src.vendors.npu_pynq.driver.abi import descriptor
        descriptors = [descriptor(arg) if isinstance(arg, np.ndarray) else None for arg in args]
        values = [ctypes.byref(desc) if desc is not None else ctypes.c_ssize_t(arg)
                  for desc, arg in zip(descriptors, args)]
        getattr(self.library, '_mlir_ciface_' + name)(*values)

    def reference(self, operation, inputs, output_names=('y0',), domain='', **attributes):
        import onnx
        import onnxruntime as ort
        from onnx import helper, numpy_helper
        names = ['x' + str(i) for i in range(len(inputs))]
        values = [helper.make_tensor_value_info(name, numpy_helper.from_array(array).data_type, array.shape)
                  for name, array in zip(names, inputs)]
        outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None) for name in output_names]
        if operation == 'DynamicQuantizeLinear':
            outputs[0].type.tensor_type.elem_type = onnx.TensorProto.UINT8
            outputs[2].type.tensor_type.elem_type = onnx.TensorProto.UINT8
        node_outputs = list(output_names)
        if operation == 'SkipSimplifiedLayerNormalization':
            node_outputs = ['y0', '', '', 'y3']
        model = helper.make_model(helper.make_graph([helper.make_node(operation, names, node_outputs,
            domain=domain, **attributes)], operation, values, outputs),
            opset_imports=[helper.make_opsetid('', 14)] + ([helper.make_opsetid(domain, 1)] if domain else []))
        model.ir_version = 10
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
        return session.run(None, dict(zip(names, inputs)))

    def test_normalization_and_residual(self):
        rng = np.random.default_rng(27)
        gamma = rng.normal(size=64).astype('f4')
        for scale in (0.0, 1e-8, 1.0, 1e8):
            x = (rng.normal(size=(2, 64)) * scale).astype('f4')
            y = (rng.normal(size=(2, 64)) * scale).astype('f4')
            out, residual = np.zeros_like(x), np.zeros_like(x)
            self.call('norm', x, gamma, out)
            expected = self.reference('SimplifiedLayerNormalization', [x, gamma],
                                      axis=-1, epsilon=1e-5)[0]
            np.testing.assert_array_equal(out, expected)
            self.call('skip', x, y, gamma, out, residual)
            expected, expected_residual = self.reference('SkipSimplifiedLayerNormalization',
                [x[None], y[None], gamma], ('y0', 'y3'), domain='com.microsoft', epsilon=1e-5)
            np.testing.assert_array_equal(residual, expected_residual[0])
            np.testing.assert_array_equal(out, expected[0])

    def test_quantization_exact_for_edge_cases_and_random_ranges(self):
        rng = np.random.default_rng(39)
        cases = [np.zeros((2, 64), 'f4'), np.ones((2, 64), 'f4'), -np.ones((2, 64), 'f4'),
                 np.linspace(-128, 127, 128, dtype='f4').reshape(2, 64)]
        ties = np.arange(128, dtype='f4') + 0.5
        ties[0], ties[-1] = 0, 255
        cases.append(ties.reshape(2, 64))
        cases += [(rng.normal(size=(2, 64)) * 10 ** exponent).astype('f4') for exponent in range(-6, 7)]
        for x in cases:
            q, scale, zero = np.zeros(x.shape, 'u1'), np.zeros(1, 'f4'), np.zeros(1, 'u1')
            self.call('quant', x, q, scale, zero)
            expected = self.reference('DynamicQuantizeLinear', [x], ('y0', 'y1', 'y2'))
            np.testing.assert_array_equal(q, expected[0])
            np.testing.assert_array_equal(scale[0], expected[1])
            np.testing.assert_array_equal(zero[0], expected[2])

    def test_sigmoid_extreme_and_finite_inputs(self):
        x = np.linspace(-100, 100, 128, dtype='f4').reshape(2, 64)
        out = np.zeros_like(x)
        self.call('sigmoid', x, out)
        expected = self.reference('Sigmoid', [x])[0]
        np.testing.assert_array_equal(out, expected)
        mathematical = 1.0 / (1.0 + np.exp(-x.astype('f8')))
        np.testing.assert_allclose(out, mathematical, rtol=0, atol=1.5e-7)

    def test_rotary_runtime_positions_and_head_pairing(self):
        rng = np.random.default_rng(45)
        x = rng.normal(size=(3, 24)).astype('f4')
        angles = rng.normal(size=(16, 4)).astype('f4')
        cos, sin = np.cos(angles), np.sin(angles)
        for position in (0, 5, 13):
            output = np.zeros_like(x)
            self.call('rope', x, cos, sin, output, position)
            expected = self.reference('RotaryEmbedding', [x[None],
                np.arange(position, position + 3, dtype='i8')[None], cos, sin],
                domain='com.microsoft', interleaved=0, num_heads=0, rotary_embedding_dim=0)[0]
            np.testing.assert_allclose(output, expected[0], rtol=2e-6, atol=1e-6)

    def test_attention_prefill_decode_masks_and_cache_updates(self):
        rng = np.random.default_rng(58)
        key, value = [np.zeros((2, 2, 16, 8), 'f4') for _ in range(2)]
        for tokens, position in ((3, 0), (1, 3), (1, 4), (3, 5)):
            q = rng.normal(size=(tokens, 48)).astype('f4')
            k, v = [rng.normal(size=(tokens, 16)).astype('f4') for _ in range(2)]
            past_k, past_v = [np.ascontiguousarray(cache[1:2, :, :position]) for cache in (key, value)]
            expected, expected_k, expected_v = self.reference('GroupQueryAttention',
                [q[None], k[None], v[None], past_k, past_v,
                 np.array([position + tokens - 1], 'i4'), np.array(position + tokens, 'i4')],
                ('y0', 'y1', 'y2'), domain='com.microsoft', num_heads=6, kv_num_heads=2,
                scale=0.125, do_rotary=0, local_window_size=-1, softcap=0.0)
            out = np.zeros_like(q)
            self.call(f'attn{tokens}', q, k, v, key, value, out, 1, position)
            np.testing.assert_allclose(out, expected[0], rtol=3e-6, atol=1e-6)
            np.testing.assert_array_equal(key[1:2, :, :position + tokens], expected_k)
            np.testing.assert_array_equal(value[1:2, :, :position + tokens], expected_v)
            np.testing.assert_array_equal(key[0], 0)
            np.testing.assert_array_equal(value[0], 0)

    def test_attention_real_head_width_and_partial_softmax_lanes(self):
        rng = np.random.default_rng(92)
        q = rng.normal(size=(40, 576)).astype('f4')
        k, v = [rng.normal(size=(40, 192)).astype('f4') for _ in range(2)]
        key, value = [np.zeros((2, 3, 64, 64), 'f4') for _ in range(2)]
        past_k, past_v = [rng.normal(size=(1, 3, 2, 64)).astype('f4') for _ in range(2)]
        key[1:2, :, :2], value[1:2, :, :2] = past_k, past_v
        expected, expected_k, expected_v = self.reference('GroupQueryAttention',
            [q[None], k[None], v[None], past_k, past_v, np.array([41], 'i4'), np.array(42, 'i4')],
            ('y0', 'y1', 'y2'), domain='com.microsoft', num_heads=9, kv_num_heads=3,
            scale=0.125, do_rotary=0, local_window_size=-1, softcap=0.0)
        out = np.zeros_like(q)
        self.call('attn40', q, k, v, key, value, out, 1, 2)
        np.testing.assert_array_equal(out, expected[0])
        np.testing.assert_array_equal(key[1:2, :, :42], expected_k)
        np.testing.assert_array_equal(value[1:2, :, :42], expected_v)

    def test_attention_decode_real_head_width_all_accumulation_tails(self):
        rng = np.random.default_rng(95)
        for position in (3, 4, 5, 6, 36, 37, 38, 39):
            q = rng.normal(size=(1, 576)).astype('f4')
            k, v = [rng.normal(size=(1, 192)).astype('f4') for _ in range(2)]
            key, value = [np.zeros((2, 3, 64, 64), 'f4') for _ in range(2)]
            past_k, past_v = [rng.normal(size=(1, 3, position, 64)).astype('f4') for _ in range(2)]
            key[1:2, :, :position], value[1:2, :, :position] = past_k, past_v
            expected, expected_k, expected_v = self.reference('GroupQueryAttention',
                [q[None], k[None], v[None], past_k, past_v,
                 np.array([position], 'i4'), np.array(position + 1, 'i4')],
                ('y0', 'y1', 'y2'), domain='com.microsoft', num_heads=9, kv_num_heads=3,
                scale=0.125, do_rotary=0, local_window_size=-1, softcap=0.0)
            out = np.zeros_like(q)
            self.call('attn_decode_real', q, k, v, key, value, out, 1, position)
            np.testing.assert_array_equal(out, expected[0])
            np.testing.assert_array_equal(key[1:2, :, :position + 1], expected_k)
            np.testing.assert_array_equal(value[1:2, :, :position + 1], expected_v)


if __name__ == '__main__':
    unittest.main()
