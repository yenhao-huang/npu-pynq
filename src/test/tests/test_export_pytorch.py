"""The PyTorch front end: W8A8 numerics, the load-time scheme check, and the
functional-cache export against transformers' own forward. Skipped where
torch, torchao or transformers are not installed."""

import importlib.util
import unittest

import numpy as np

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ('torch', 'torchao', 'transformers'))


EPS = 1e-8   # src/vendors/npu_pynq/spec/quant.py


def reference_weight(w_out_in):
    """The contract's W8: per output channel, scale = max(amax, EPS) / 127, round half even."""
    w = np.asarray(w_out_in, np.float32).T
    scale = (np.maximum(np.abs(w).max(0), EPS) / 127.0).astype(np.float32)
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale


def reference_rows(x):
    """The contract's A8: per row, scale = max(amax, EPS) * (1 / 127), round half even."""
    scale = (np.maximum(np.abs(x).max(-1), EPS) * np.float32(1.0 / 127.0)).astype(np.float32)
    return np.clip(np.round(x * (1.0 / scale)[..., None]), -127, 127).astype(np.int8), scale


@unittest.skipUnless(HAVE_TORCH, 'torch, torchao and transformers are required')
class W8A8NumericsTest(unittest.TestCase):
    def test_weights_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import quantize_weight

        w = np.random.default_rng(0).normal(size=(48, 64)).astype(np.float32)
        q, s = quantize_weight(torch.from_numpy(w))
        rq, rs = reference_weight(w)
        np.testing.assert_array_equal(q.numpy(), rq)
        np.testing.assert_array_equal(s.numpy(), rs)

    def test_activations_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import quantize_rows

        x = (np.random.default_rng(1).normal(size=(5, 64)) * 3).astype(np.float32)
        q, s = quantize_rows(torch.from_numpy(x))
        rq, rs = reference_rows(x)
        np.testing.assert_array_equal(q.numpy(), rq)
        np.testing.assert_array_equal(s.numpy(), rs)

    def test_linear_is_the_int8_contraction_rescaled(self):
        import torch
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import W8A8Linear

        rng = np.random.default_rng(2)
        layer = torch.nn.Linear(64, 32, bias=False)
        layer.weight.data = torch.from_numpy(rng.normal(size=(32, 64)).astype(np.float32))
        x = rng.normal(size=(3, 64)).astype(np.float32)
        y = W8A8Linear.from_float(layer)(torch.from_numpy(x)).detach().numpy()
        wq, ws = reference_weight(layer.weight.detach().numpy())
        xq, xs = reference_rows(x)
        acc = xq.astype(np.int32) @ wq.astype(np.int32)
        np.testing.assert_allclose(y, acc.astype(np.float32) * xs[:, None] * ws[None, :], rtol=1e-6, atol=1e-6)

    def test_convolution_is_the_contraction_of_its_patches(self):
        import torch
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import W8A8Conv2d

        torch.manual_seed(4)
        conv = torch.nn.Conv2d(5, 7, 3, stride=2, padding=1)
        x = torch.randn(1, 5, 9, 11)
        q = W8A8Conv2d.from_float(conv)
        y = q(x)
        self.assertEqual(tuple(y.shape), tuple(conv(x).shape))
        # The W8A8 result stays within quantization error of the float convolution.
        torch.testing.assert_close(y, conv(x), rtol=0.05, atol=0.05)

    def test_batchnorm_is_folded_before_quantization(self):
        import torch
        from src.export.frontend.load import check
        from src.export.frontend.quantization.torchao.config import quantize

        torch.manual_seed(5)
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
                                  torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Linear(4 * 6 * 6, 10)).eval()
        net[1].running_mean.uniform_(-1, 1)
        net[1].running_var.uniform_(0.5, 2)
        x = torch.randn(1, 3, 6, 6)
        expected = net(x)
        q = quantize(net, 'w8a8')
        check(q, 'w8a8')
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in q.modules()))
        torch.testing.assert_close(q(x), expected, rtol=0.1, atol=0.1)

def tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True)
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


@unittest.skipUnless(HAVE_TORCH, 'torch, torchao and transformers are required')
class LoadCheckTest(unittest.TestCase):
    def test_float_layers_are_rejected(self):
        from src.export.frontend.load import QuantizationError, check

        with self.assertRaises(QuantizationError):
            check(tiny_llama(), 'w8a8')

    def test_our_config_passes_the_check(self):
        from torchao.quantization import quantize_
        from src.export.frontend.load import check
        from src.export.frontend.quantization.torchao.config import NpuW8A8Config, npu_layers

        model = tiny_llama()
        quantize_(model, NpuW8A8Config(), filter_fn=npu_layers)
        check(model, 'w8a8')


@unittest.skipUnless(HAVE_TORCH, 'torch, torchao and transformers are required')
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
