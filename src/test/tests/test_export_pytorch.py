"""The PyTorch front end: W8A8 numerics, the quantization pass, and the
functional-cache export against transformers' own forward. Skipped where
torch or transformers is not installed."""

import importlib.util
import unittest

import numpy as np

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ('torch', 'transformers'))


EPS = 1e-8   # src/export/frontend/quantization/schemes.py


def reference_weight(w_out_in):
    """The contract's W8: per output channel, scale = max(amax, EPS) / 127, round half even."""
    w = np.asarray(w_out_in, np.float32).T
    scale = (np.maximum(np.abs(w).max(0), EPS) / 127.0).astype(np.float32)
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale


def reference_activation_scale(amax):
    """The contract's A8 scale: static per tensor, max(calibrated amax, EPS) / 127, as [1] float32."""
    return (np.maximum(np.float32(amax), np.float32(EPS)) / np.float32(127)).astype(np.float32).reshape(1)


def reference_activations(x, scale):
    """The contract's A8: q = clamp(round(x * (1 / scale)), -127, 127), round half even, zero point 0."""
    inverse = (np.float32(1) / scale).astype(np.float32)
    return np.clip(np.round(x * inverse), -127, 127).astype(np.int8)


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class W8A8NumericsTest(unittest.TestCase):
    def test_weights_match_the_numeric_contract_bit_for_bit(self):
        # linear writes the per-channel arithmetic out; embedding and conv2d
        # share _per_channel. Both must be the contract's.
        import torch
        from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8QuantizedLayer

        layer = W8A8QuantizedLayer()
        w = np.random.default_rng(0).normal(size=(48, 64)).astype(np.float32)
        rq, rs = reference_weight(w)
        q, s, _ = layer.linear(torch.from_numpy(w), 1.0).tensors
        np.testing.assert_array_equal(q.numpy(), rq)
        np.testing.assert_array_equal(s.numpy(), rs)
        q, s = layer.embedding(torch.from_numpy(w)).tensors     # per row of the [V, H] table
        np.testing.assert_array_equal(q.numpy(), rq.T)
        np.testing.assert_array_equal(s.numpy(), rs)

    def test_activations_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.frontend.quantization.fx import custom_ops
        from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8QuantizedLayer

        def activation_scale(amax):
            return W8A8QuantizedLayer().linear(torch.ones(2, 3), amax).tensors[2]

        x = (np.random.default_rng(1).normal(size=(5, 64)) * 3).astype(np.float32)
        # A calibrated amax below the input's own: the largest values saturate.
        amax = float(np.abs(x).max()) * 0.75
        s = activation_scale(amax)
        np.testing.assert_array_equal(s.numpy(), reference_activation_scale(amax))
        q = custom_ops.quantize_per_tensor(torch.from_numpy(x), s, 127)
        np.testing.assert_array_equal(q.numpy(), reference_activations(x, s.numpy()))
        self.assertEqual(int(q.abs().max()), 127)
        # An all-zero calibration takes the EPS scale instead of dividing by zero.
        np.testing.assert_array_equal(activation_scale(0.0).numpy(), reference_activation_scale(0.0))

    def test_custom_ops_are_their_aten_arithmetic_bit_for_bit(self):
        # quant.quantize_per_tensor and quant.dequantize against the ATen
        # expressions DecomposeQuantOps expands them into.
        import torch
        from src.export.frontend.quantization.fx import custom_ops as ops

        rng = np.random.default_rng(6)
        x = torch.from_numpy((rng.normal(size=(7, 40)) * 5).astype(np.float32))
        scale = torch.tensor([0.03], dtype=torch.float32)
        q = ops.quantize_per_tensor(x, scale, 127)
        expected = torch.clamp(torch.round(x * (1.0 / scale)), -127, 127).to(torch.int8)
        self.assertEqual(q.dtype, torch.int8)
        self.assertTrue(torch.equal(q, expected))

        acc = torch.from_numpy(rng.integers(-2**20, 2**20, size=(7, 9)).astype(np.int32))
        w_scale = torch.from_numpy(rng.uniform(1e-4, 1e-1, size=9).astype(np.float32))
        for x_scale in (scale, torch.from_numpy(rng.uniform(1e-3, 1e-1, size=7).astype(np.float32))):
            with self.subTest(x_scale=tuple(x_scale.shape)):
                y = ops.dequantize(acc, x_scale, w_scale)
                self.assertTrue(torch.equal(y, acc.to(torch.float32) * x_scale[:, None] * w_scale[None, :]))

    def test_linear_is_the_int8_contraction_rescaled(self):
        import torch
        from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8QuantizedLayer

        rng = np.random.default_rng(2)
        w = rng.normal(size=(32, 64)).astype(np.float32)
        x = rng.normal(size=(3, 64)).astype(np.float32)
        r = W8A8QuantizedLayer().linear(torch.from_numpy(w), float(np.abs(x).max()))
        y = r.function(torch.from_numpy(x), *r.tensors, None).numpy()
        xs = r.tensors[2]
        wq, ws = reference_weight(w)
        xq = reference_activations(x, xs.numpy())
        acc = xq.astype(np.int32) @ wq.astype(np.int32)
        np.testing.assert_allclose(y, acc.astype(np.float32) * xs.numpy() * ws[None, :], rtol=1e-6, atol=1e-6)

    def test_convolution_is_the_contraction_of_its_patches(self):
        import torch
        from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8QuantizedLayer

        torch.manual_seed(4)
        conv = torch.nn.Conv2d(5, 7, 3, stride=2, padding=1)
        x = torch.randn(1, 5, 9, 11)
        with torch.no_grad():
            r = W8A8QuantizedLayer().conv2d(conv.weight, float(x.abs().max()))
            y = r.function(x, *r.tensors, conv.bias, (3, 3), (2, 2), (1, 1))
            expected = conv(x)
        self.assertEqual(tuple(y.shape), tuple(expected.shape))
        # The W8A8 result stays within quantization error of the float convolution.
        torch.testing.assert_close(y, expected, rtol=0.05, atol=0.05)

    def test_batchnorm_stays_after_quantization(self):
        import torch
        from src.export.frontend.quantization import quantize

        torch.manual_seed(5)
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
                                  torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, 10)).eval()
        net[1].running_mean.uniform_(-1, 1)
        net[1].running_var.uniform_(0.5, 2)
        x = torch.randn(1, 3, 6, 6)
        with torch.no_grad():
            expected = net(x)
            ep = quantize({'forward': torch.export.export(net, (x,), strict=False)}, 'w8a8',
                          {'forward': [(x,)]})['forward']
            got = ep.module()(x)
        self.assertEqual(targets(ep)['aten.batch_norm.default'], 1)
        torch.testing.assert_close(got, expected, rtol=0.1, atol=0.1)


def call_targets(ep):
    """The targets of ``ep``'s call_function nodes in order, without metadata asserts."""
    return [str(n.target) for n in ep.graph.nodes
            if n.op == 'call_function' and str(n.target) != 'aten._assert_tensor_metadata.default']


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class QuantizerContractTest(unittest.TestCase):
    def test_a_scheme_needs_a_method_for_each_of_its_ops(self):
        import torch
        from src.export.frontend.quantization.fx.quantizer.base import QuantizedLayer, Quantizer
        from src.export.frontend.quantization.fx.quantizer.w8a8 import W8A8QuantizedLayer

        base = QuantizedLayer()
        for call in (lambda: base.linear(torch.ones(2, 2), 1.0), lambda: base.conv2d(torch.ones(1, 1, 1, 1), 1.0),
                     lambda: base.embedding(torch.ones(2, 2))):
            with self.assertRaises(NotImplementedError):
                call()

        class OnlyLinear(QuantizedLayer):
            def linear(self, weight, amax):
                raise AssertionError('not called')

        Quantizer(OnlyLinear(), ('linear',), ('linear',), {})
        with self.assertRaisesRegex(ValueError, "op\\(s\\) \\['embedding'\\]"):
            Quantizer(OnlyLinear(), ('linear', 'embedding'), ('linear',), {})
        Quantizer(W8A8QuantizedLayer(), ('linear', 'conv2d', 'embedding'), ('linear', 'conv2d'), {})

    def test_calibrated_ops_must_be_among_the_ops(self):
        from unittest import mock

        import torch
        from src.export.frontend.quantization import quantize
        from src.export.frontend.quantization.schemes import SCHEMES

        net = torch.nn.Linear(4, 2).eval()
        with torch.no_grad():
            programs = {'forward': torch.export.export(net, (torch.zeros(1, 4),), strict=False)}
        with mock.patch.dict(SCHEMES['w8a8'], ops=('linear',), calibrated_ops=('linear', 'conv2d')), \
                self.assertRaisesRegex(ValueError, 'calibrated_ops'):
            quantize(programs, 'w8a8', {'forward': [(torch.ones(1, 4),)]})


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class QuantizedGraphTest(unittest.TestCase):
    def quantized_linear(self):
        import torch
        from src.export.frontend.quantization import quantize

        torch.manual_seed(7)
        net = torch.nn.Linear(16, 8).eval()
        x = torch.randn(2, 3, 16)
        calibration = [(torch.randn(2, 3, 16) * 2,), (torch.randn(2, 3, 16),)]
        with torch.no_grad():
            ep = quantize({'forward': torch.export.export(net, (x,), strict=False)}, 'w8a8',
                          {'forward': calibration})['forward']
        return net, x, calibration, ep

    def test_a_quantized_linear_is_four_nodes(self):
        # quantize_per_tensor -> _int_mm -> dequantize -> + bias, between the
        # reshapes that make the contraction rank 2; still so after
        # run_decompositions, which to_mlir applies.
        import torch

        net, x, calibration, ep = self.quantized_linear()
        self.assertEqual(call_targets(ep.run_decompositions()),
                         ['aten.view.default', 'quant.quantize_per_tensor.default', 'aten._int_mm.default',
                          'quant.dequantize.default', 'aten.add.Tensor', 'aten.view.default'])
        self.assertFalse([n for n, t in ep.state_dict.items() if t.dtype == torch.float32 and t.dim() == 2])

    def test_the_graph_is_the_reference_with_the_calibrated_scale(self):
        # The activation scale is the largest |input| over every calibration
        # input, and the graph computes the contract's arithmetic bit for bit.
        import torch

        net, x, calibration, ep = self.quantized_linear()
        amax = max(float(args[0].abs().max()) for args in calibration)
        xs = reference_activation_scale(amax)
        scales = [t for t in ep.state_dict.values() if t.dtype == torch.float32 and tuple(t.shape) == (1,)]
        self.assertEqual(len(scales), 1)
        np.testing.assert_array_equal(scales[0].numpy(), xs)
        wq, ws = reference_weight(net.weight.detach().numpy())
        acc = reference_activations(x.reshape(-1, 16).numpy(), xs).astype(np.int32) @ wq.astype(np.int32)
        expected = (torch.from_numpy(acc).to(torch.float32) * torch.from_numpy(xs)[:, None] *
                    torch.from_numpy(ws)[None, :] + net.bias).reshape(2, 3, 8)
        with torch.no_grad():
            self.assertTrue(torch.equal(ep.module()(x), expected))

    def test_a_layer_without_calibration_is_refused(self):
        import torch
        from src.export.frontend.quantization import quantize

        net = torch.nn.Linear(16, 8).eval()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'calibration'):
            quantize({'forward': torch.export.export(net, (torch.zeros(1, 16),), strict=False)}, 'w8a8')


def targets(ep):
    """op name -> how many call_function nodes of ``ep`` call it."""
    import collections

    return collections.Counter(str(n.target) for n in ep.graph.nodes if n.op == 'call_function')


def tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True)
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class QuantizationPassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from src.export.frontend.export_graph import export_llm
        from src.export.frontend.quantization import quantize

        cls.model = tiny_llama()
        cls.float_programs, spec = export_llm(cls.model, 32, 4)
        # Prefill alone is calibrated; decode reuses each layer's scale.
        tokens = torch.randint(0, 128, (3, 1, 4), generator=torch.Generator().manual_seed(8))
        calibration = [(t, torch.tensor([4 * i]), torch.tensor([3]), *spec.zeros()) for i, t in enumerate(tokens)]
        cls.programs = quantize(cls.float_programs, 'w8a8', {'prefill': calibration})

    def test_every_contraction_is_rewritten(self):
        # 2 layers x 7 projections + lm_head; the embedding becomes an INT8 lookup.
        for name, ep in self.programs.items():
            with self.subTest(program=name):
                ops = targets(ep)
                self.assertNotIn('aten.linear.default', ops)
                self.assertNotIn('aten.embedding.default', ops)
                self.assertEqual(ops['aten._int_mm.default'], 15)
                self.assertEqual(ops['quant.quantize_per_tensor.default'], 15)
                self.assertEqual(ops['quant.dequantize.default'], 15)
                # The activation quantization is inside the custom ops, not spelled out in ATen.
                self.assertNotIn('aten.round.default', ops)

    def test_no_float_weight_is_left(self):
        import torch

        for name, ep in self.programs.items():
            with self.subTest(program=name):
                self.assertFalse([n for n, t in ep.state_dict.items() if t.dtype == torch.float32 and t.dim() == 2])

    def test_decode_reads_the_scales_prefill_calibrated(self):
        # Every activation scale is one [1] tensor that both programs read.
        import torch

        scales = {k: [t for t in self.programs[k].state_dict.values()
                      if t.dtype == torch.float32 and tuple(t.shape) == (1,)] for k in ('decode', 'prefill')}
        self.assertEqual(len(scales['prefill']), 15)
        self.assertEqual(sorted(id(t) for t in scales['decode']), sorted(id(t) for t in scales['prefill']))

    def test_programs_share_each_quantized_weight(self):
        decode, prefill = (self.programs[k].state_dict for k in ('decode', 'prefill'))
        self.assertTrue(decode)
        for name, t in decode.items():
            with self.subTest(weight=name):
                self.assertTrue(any(t is u for u in prefill.values()))

    def test_the_float_programs_are_left_alone(self):
        self.assertEqual(targets(self.float_programs['decode'])['aten.linear.default'], 15)

    def test_quantized_logits_stay_near_the_float_ones(self):
        import torch
        from src.export.frontend.export_graph.export import cache_spec

        spec = cache_spec(self.model, 32)
        args = (torch.tensor([[5]]), torch.tensor([0]), *spec.zeros())
        with torch.no_grad():
            expected = self.float_programs['decode'].module()(*args)[0]
            got = self.programs['decode'].module()(*args)[0]
        torch.testing.assert_close(got, expected, rtol=0.1, atol=0.1)

    def test_unknown_scheme_is_refused(self):
        from src.export.frontend.quantization import quantize

        with self.assertRaises(ValueError):
            quantize(self.float_programs, 'w4a4')


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class FunctionalCacheTest(unittest.TestCase):
    def test_chunked_prefill_and_decode_match_the_full_forward(self):
        import torch
        from src.export.frontend.export_graph.export import Step, cache_spec

        model = tiny_llama()
        ids = torch.randint(3, 120, (21,), generator=torch.Generator().manual_seed(3)).tolist()
        spec = cache_spec(model, 32)
        prefill, decode = Step(model, spec, True), Step(model, spec, False)
        kv, pos, chunk = spec.zeros(), 0, 8
        for start in range(0, 20, chunk):
            part = ids[start:min(start + chunk, 20)]
            out = prefill(torch.tensor([part + [0] * (chunk - len(part))]), torch.tensor([pos]),
                          torch.tensor([len(part) - 1]), *kv)
            logits, kv = out[0][0, -1], list(out[1:])
            pos += len(part)
        expected = model(torch.tensor([ids[:20]])).logits[0, -1]
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
        out = decode(torch.tensor([[ids[20]]]), torch.tensor([pos]), *kv)
        torch.testing.assert_close(out[0][0, -1], model(torch.tensor([ids])).logits[0, -1], rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
