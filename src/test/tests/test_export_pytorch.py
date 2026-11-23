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


def reference_rows(x):
    """The contract's A8: per token, scale = max(amax, EPS) * (1 / 127), round half even."""
    scale = (np.maximum(np.abs(x).max(-1), EPS) * np.float32(1.0 / 127.0)).astype(np.float32)
    return np.clip(np.round(x * (1.0 / scale)[..., None]), -127, 127).astype(np.int8), scale


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class W8A8NumericsTest(unittest.TestCase):
    def test_weights_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.frontend.quantization.fx.w8a8 import quantize_weight

        w = np.random.default_rng(0).normal(size=(48, 64)).astype(np.float32)
        q, s = quantize_weight(torch.from_numpy(w))
        rq, rs = reference_weight(w)
        np.testing.assert_array_equal(q.numpy(), rq)
        np.testing.assert_array_equal(s.numpy(), rs)

    def test_activations_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.frontend.quantization.fx.w8a8 import quantize_tokens

        x = (np.random.default_rng(1).normal(size=(5, 64)) * 3).astype(np.float32)
        q, s = quantize_tokens(torch.from_numpy(x))
        rq, rs = reference_rows(x)
        np.testing.assert_array_equal(q.numpy(), rq)
        np.testing.assert_array_equal(s.numpy(), rs)

    def test_linear_is_the_int8_contraction_rescaled(self):
        import torch
        from src.export.frontend.quantization.fx.w8a8 import linear, prepare_linear

        rng = np.random.default_rng(2)
        w = rng.normal(size=(32, 64)).astype(np.float32)
        x = rng.normal(size=(3, 64)).astype(np.float32)
        y = linear(torch.from_numpy(x), *prepare_linear(torch.from_numpy(w)), None).numpy()
        wq, ws = reference_weight(w)
        xq, xs = reference_rows(x)
        acc = xq.astype(np.int32) @ wq.astype(np.int32)
        np.testing.assert_allclose(y, acc.astype(np.float32) * xs[:, None] * ws[None, :], rtol=1e-6, atol=1e-6)

    def test_convolution_is_the_contraction_of_its_patches(self):
        import torch
        from src.export.frontend.quantization.fx.w8a8 import conv2d, prepare_conv2d

        torch.manual_seed(4)
        conv = torch.nn.Conv2d(5, 7, 3, stride=2, padding=1)
        x = torch.randn(1, 5, 9, 11)
        with torch.no_grad():
            y = conv2d(x, *prepare_conv2d(conv.weight), conv.bias, (3, 3), (2, 2), (1, 1))
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
            ep = quantize({'forward': torch.export.export(net, (x,), strict=False)}, 'w8a8')['forward']
            got = ep.module()(x)
        self.assertEqual(targets(ep)['aten.batch_norm.default'], 1)
        torch.testing.assert_close(got, expected, rtol=0.1, atol=0.1)

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
        from src.export.frontend.export_graph import export_llm
        from src.export.frontend.quantization import quantize

        cls.model = tiny_llama()
        cls.float_programs, _ = export_llm(cls.model, 32, 4)
        cls.programs = quantize(cls.float_programs, 'w8a8')

    def test_every_contraction_is_rewritten(self):
        # 2 layers x 7 projections + lm_head; the embedding becomes an INT8 lookup.
        for name, ep in self.programs.items():
            with self.subTest(program=name):
                ops = targets(ep)
                self.assertNotIn('aten.linear.default', ops)
                self.assertNotIn('aten.embedding.default', ops)
                self.assertEqual(ops['aten._int_mm.default'], 15)

    def test_no_float_weight_is_left(self):
        import torch

        for name, ep in self.programs.items():
            with self.subTest(program=name):
                self.assertFalse([n for n, t in ep.state_dict.items() if t.dtype == torch.float32 and t.dim() == 2])

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
