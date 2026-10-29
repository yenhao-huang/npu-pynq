"""The three front-end stages chained, saved and replayed: each stage on its
own produces exactly what build_graph (and so export) compiles."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.export import cli, pipeline
from src.export.ir.options import LLMOptions
from src.export.model_arch.resnet18 import ResNetOptions
from src.test.vectors import generate_export_golden as golden


class StagedFrontEndTest(unittest.TestCase):
    def test_build_graph_reports_each_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            timings = {}
            graph, cfg = pipeline.build_graph('smollm', golden.smollm_source(Path(tmp)),
                                              golden.llm_options(), timings=timings)
            self.assertEqual(sorted(timings), ['load', 'quantize', 'to_mlir'])
            self.assertEqual(cfg.layers, 2)
            self.assertIn('decode', graph.entry_points)

    def test_saved_parameters_lower_to_the_same_graph(self):
        cases = [('smollm', golden.smollm_source, LLMOptions(chunk=3, max_seq=16, bits=4, group=8, kv='int8',
                                                              attention='online', layers=1)),
                 ('qwen3_moe', golden.qwen3_moe_source, golden.llm_options()),
                 ('qwen3_5', golden.qwen3_5_source, golden.llm_options())]
        for model, writer, opts in cases:
            with self.subTest(model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                source = writer(root)
                direct, _ = pipeline.build_graph(model, source, opts)
                recipe, shape, layers = opts.split()
                q = pipeline.quantize(pipeline.load(model, source, layers=layers), recipe)
                replayed = pipeline.to_mlir(pipeline.load_quantized(pipeline.save_quantized(q, root / 'q')), shape)
                self.assertEqual(golden.digest(replayed, root), golden.digest(direct, root))
                q.source.tensors.close()

    def test_resnet_parameters_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(sys.modules, {'torch': golden.fake_torch(golden.resnet18_state())}):
            root = Path(tmp).resolve()
            source = golden.resnet18_source(root)
            direct, cfg = pipeline.build_graph('resnet18', source, ResNetOptions(8))
            self.assertIsNone(cfg)
            q = pipeline.quantize(pipeline.load('resnet18', source), pipeline.split_options(ResNetOptions(8))[0])
            replayed = pipeline.to_mlir(pipeline.load_quantized(pipeline.save_quantized(q, root / 'q')),
                                        pipeline.split_options(ResNetOptions(8))[1])
            self.assertEqual(golden.digest(replayed, root), golden.digest(direct, root))

    def test_stage_commands_match_build_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = golden.smollm_source(root / 'model')
            stage = root / 'stage'
            with mock.patch('builtins.print'):
                cli.main(['load', 'smollm', '--source', str(source), '--out', str(stage / 'source.json')])
                cli.main(['quantize', str(stage / 'source.json'), '--bits', '4', '--group', '8',
                          '--out', str(stage / 'quantized')])
                cli.main(['lower', str(stage / 'quantized'), '--chunk', '3', '--max-seq', '16',
                          '--attention', 'naive', '--out', str(stage / 'graph')])
            direct, _ = pipeline.build_graph('smollm', source, golden.llm_options(bits=4))
            self.assertEqual((stage / 'graph' / 'model.mlir').read_text(), direct.module.render())
            graph = json.loads((stage / 'graph' / 'graph.json').read_text())
            self.assertEqual(graph['weights'], list(direct.weights))
            self.assertEqual(graph['meta']['options'], direct.meta['options'])
            record = json.loads((stage / 'source.json').read_text())
            self.assertEqual((record['arch'], record['format']), ('smollm', 'safetensors'))

    def test_lower_rejects_shapes_the_source_cannot_honour(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = golden.smollm_source(root)
            q = pipeline.quantize(pipeline.load('smollm', source), LLMOptions().split()[0])
            saved = pipeline.save_quantized(q, root / 'q')
            model = json.loads((saved / 'model.json').read_text())
            model['arch'] = 'smollm_onnx'
            (saved / 'model.json').write_text(json.dumps(model))
            with self.assertRaises(ValueError), mock.patch('builtins.print'):
                cli.main(['lower', str(saved), '--kv', 'int8', '--out', str(root / 'graph')])
            q.source.tensors.close()


if __name__ == '__main__':
    unittest.main()
