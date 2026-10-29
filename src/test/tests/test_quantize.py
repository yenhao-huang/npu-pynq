"""The quantize stage: weight plans, group selection, the AWQ hook, and the
preserve recipe for already-quantized sources."""

import unittest

import numpy as np

from src.export.ir.options import QuantRecipe
from src.export.ir.quantized import OnnxQuantWeight, QuantizedWeight
from src.export.quantize.int8 import quantize_weight
from src.export.quantize.plan import WeightSpec, run_plan
from src.export.quantize.preserve import dequantize, fuse_columns
from src.export.quantize.recipe import choose_group


def tensors(seed=0):
    rng = np.random.default_rng(seed)
    return {'a': rng.normal(size=(8, 16)).astype(np.float32), 'b': rng.normal(size=(4, 16)).astype(np.float32),
            'n': rng.normal(size=16).astype(np.float64), 'stack': rng.normal(size=(3, 8, 16)).astype(np.float32)}


class WeightPlanTest(unittest.TestCase):
    def test_kinds_store_what_the_operators_consume(self):
        t = tensors()
        recipe = QuantRecipe(bits=4, group=8, embed_bits=8)
        params = run_plan(t, [WeightSpec('fused', ('a', 'b'), 'linear'), WeightSpec('emb', ('a',), 'embed'),
                              WeightSpec('router', ('b',), 'fp32'), WeightSpec('norm', ('n',), 'float'),
                              WeightSpec('raw', ('n',), 'raw'),
                              WeightSpec('e1', ('stack',), 'linear', transform=lambda s: s[1])], recipe)
        expected = quantize_weight(np.concatenate([t['a'], t['b']]), 4, 8)
        np.testing.assert_array_equal(params['fused'].q, expected.q)
        np.testing.assert_array_equal(params['fused'].scale, expected.scale)
        self.assertEqual((params['emb'].bits, params['router'].bits), (8, 32))
        self.assertEqual(params['norm'].dtype, np.float32)
        self.assertIs(params['raw'], t['n'])
        np.testing.assert_array_equal(params['e1'].q, quantize_weight(t['stack'][1], 4, 8).q)

    def test_awq_hook_sees_each_layer_once_with_its_roles(self):
        t = tensors()
        calls = []

        def awq(layer, mats, norms):
            calls.append((layer, sorted(mats), sorted(norms)))
            return {k: v * 2 for k, v in mats.items()}, {k: v + 1 for k, v in norms.items()}

        specs = [WeightSpec('emb', ('a',), 'embed')]
        for i in range(2):
            specs += [WeightSpec(f'l{i}.qkv', ('a',), 'linear', layer=i, role='qkv'),
                      WeightSpec(f'l{i}.ln1', ('n',), 'float', layer=i, role='ln1'),
                      WeightSpec(f'l{i}.q_norm', ('n',), 'float', layer=i)]
        params = run_plan(t, specs, QuantRecipe(), awq)
        self.assertEqual(calls, [(0, ['qkv'], ['ln1']), (1, ['qkv'], ['ln1'])])
        np.testing.assert_array_equal(params['l0.qkv'].q, quantize_weight(t['a'] * 2).q)
        np.testing.assert_array_equal(params['l0.ln1'], (t['n'] + 1).astype(np.float32))
        np.testing.assert_array_equal(params['l0.q_norm'], t['n'].astype(np.float32))

    def test_stacked_source_is_read_once_per_layer(self):
        reads = []

        class Counting(dict):
            def __getitem__(self, name):
                reads.append(name)
                return super().__getitem__(name)

        t = Counting(tensors())
        specs = [WeightSpec(f'l0.e{e}', ('stack',), 'linear', transform=lambda s, e=e: s[e], layer=0)
                 for e in range(3)]
        run_plan(t, specs, QuantRecipe())
        self.assertEqual(reads, ['stack'])

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            WeightSpec('x', ('a',), 'int3')


class RecipeTest(unittest.TestCase):
    def test_group_divides_every_k(self):
        self.assertEqual(choose_group(128, [1024, 3072]), 128)
        self.assertEqual(choose_group(128, [576, 1536]), 64)
        self.assertEqual(choose_group(128, []), 128)
        with self.assertRaises(ValueError):
            choose_group(128, [100])


class PreserveTest(unittest.TestCase):
    def test_fused_columns_keep_each_projection_scale(self):
        t = {'p.weight_quantized': np.ones((4, 2), np.int8), 'p.weight_scale': np.float32(0.5),
             'q.weight_quantized': np.full((4, 3), 2, np.int8), 'q.weight_scale': np.float32(0.25)}
        fused = fuse_columns(t, ['p', 'q'])
        self.assertIsInstance(fused, OnnxQuantWeight)
        np.testing.assert_array_equal(fused.q, np.concatenate([t['p.weight_quantized'], t['q.weight_quantized']], 1))
        np.testing.assert_array_equal(fused.scale, [0.5, 0.5, 0.25, 0.25, 0.25])

    def test_dequantize_subtracts_the_zero_point(self):
        q = np.array([[0, 10]], np.uint8)
        np.testing.assert_array_equal(dequantize(q, np.uint8(5), np.float32(0.5)), [[-2.5, 2.5]])

    def test_quantized_weight_round_trips(self):
        w = np.random.default_rng(1).normal(size=(6, 16)).astype(np.float32)
        q = quantize_weight(w, 8)
        self.assertIsInstance(q, QuantizedWeight)
        np.testing.assert_allclose(q.dequantize(), w.T, atol=np.abs(w).max() / 127)


if __name__ == '__main__':
    unittest.main()
