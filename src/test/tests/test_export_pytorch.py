"""The PyTorch front end: W8A8 numerics, the load-time scheme check, and the
functional-cache export against transformers' own forward. Skipped where
torch, torchao or transformers are not installed."""

import importlib.util
import unittest

import numpy as np

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ('torch', 'torchao', 'transformers'))


@unittest.skipUnless(HAVE_TORCH, 'torch, torchao and transformers are required')
class W8A8NumericsTest(unittest.TestCase):
    def test_weights_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import quantize_weight
        from src.export.quantize.int8 import quantize_weight as reference

        w = np.random.default_rng(0).normal(size=(48, 64)).astype(np.float32)
        q, s = quantize_weight(torch.from_numpy(w))
        ref = reference(w, 8)
        np.testing.assert_array_equal(q.numpy(), ref.q)
        np.testing.assert_array_equal(s.numpy(), ref.scale)

    def test_activations_match_the_numeric_contract_bit_for_bit(self):
        import torch
        from src.export.compilers.mlir.ops.linear import ref_quantize_rows
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import quantize_rows

        x = (np.random.default_rng(1).normal(size=(5, 64)) * 3).astype(np.float32)
        q, s = quantize_rows(torch.from_numpy(x))
        rq, rs = ref_quantize_rows(x)
        np.testing.assert_array_equal(q.numpy(), rq)
        np.testing.assert_array_equal(s.numpy(), rs)

    def test_linear_matches_the_reference_linear(self):
        import torch
        from src.export.compilers.mlir.ops.linear import ref_linear
        from src.export.frontend.quantization.torchao.custom_quant.w8a8 import W8A8Linear
        from src.export.quantize.int8 import quantize_weight as reference

        rng = np.random.default_rng(2)
        layer = torch.nn.Linear(64, 32, bias=False)
        layer.weight.data = torch.from_numpy(rng.normal(size=(32, 64)).astype(np.float32))
        x = rng.normal(size=(3, 64)).astype(np.float32)
        y = W8A8Linear.from_float(layer)(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_allclose(y, ref_linear(x, reference(layer.weight.detach().numpy(), 8)), rtol=1e-6, atol=1e-6)


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
        from src.export.frontend.quantization.torchao.config import NpuW8A8Config, embedding_and_linear

        model = tiny_llama()
        quantize_(model, NpuW8A8Config(), filter_fn=embedding_and_linear)
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
