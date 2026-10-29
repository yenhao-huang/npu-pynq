"""The load stage: format detection, architecture choice, and reading a
source without changing any number."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.export.load import detect_format
from src.export.load.onnx import imported_node, verify_pin
from src.export.pipeline import architecture, load
from src.test.vectors import generate_export_golden as golden


class DetectFormatTest(unittest.TestCase):
    def test_paths_name_their_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'prepared').mkdir()
            (root / 'prepared' / 'model_int8.onnx').write_bytes(b'')
            (root / 'nested' / 'onnx').mkdir(parents=True)
            (root / 'nested' / 'onnx' / 'model_int8.onnx').write_bytes(b'')
            cases = {root / 'model.onnx': 'onnx', root / 'prepared': 'onnx', root / 'nested': 'onnx',
                     root / 'resnet.pth': 'torch', root / 'checkpoint': 'safetensors', None: 'safetensors'}
            for path, fmt in cases.items():
                with self.subTest(path=path):
                    self.assertEqual(detect_format(path), fmt)
            self.assertEqual(detect_format(root / 'model.onnx', 'safetensors'), 'safetensors')

    def test_architecture_follows_model_and_format(self):
        self.assertEqual(architecture('smollm', 'x.onnx')[0].name, 'smollm_onnx')
        self.assertEqual(architecture('smollm', 'checkpoint')[0].name, 'smollm')
        arch, fmt = architecture('resnet18', 'weights.bin')
        self.assertEqual((arch.name, fmt), ('resnet18', 'torch'))
        with self.assertRaisesRegex(ValueError, 'Unsupported Qwen3 source format: onnx'):
            architecture('qwen3', 'checkpoint', 'onnx')
        with self.assertRaises(KeyError):
            architecture('unknown', 'checkpoint')


class SafetensorsLoadTest(unittest.TestCase):
    def test_file_or_directory_reads_unchanged_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = golden.smollm_source(Path(tmp))
            for source in (root, root / 'model.safetensors'):
                with self.subTest(source=source.name):
                    src = load('smollm', source, layers=1)
                    self.assertEqual((src.arch, src.format, src.root, src.layers), ('smollm', 'safetensors', root, 1))
                    self.assertEqual(src.config['hidden_size'], 16)
                    self.assertEqual(src.tensors['model.norm.weight'].dtype, np.float32)
                    src.tensors.close()


class OnnxSourceChecksTest(unittest.TestCase):
    def test_pin_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, pin = Path(tmp) / 'model_int8.onnx', Path(tmp) / 'pin.json'
            model.write_bytes(b'not the pinned model')
            pin.write_text(json.dumps({'files': {'model_int8.onnx': {'sha256': '0' * 64}}}))
            with self.assertRaisesRegex(ValueError, 'pinned ONNX artifact'):
                verify_pin(model, pin)

    def test_imported_node_checks_the_operation(self):
        nodes = {'/a': {'op': 'onnx.Sigmoid'}}
        self.assertIs(imported_node(nodes, '/a', 'onnx.Sigmoid'), nodes['/a'])
        for name, op in (('/a', 'onnx.Custom'), ('/missing', 'onnx.Sigmoid')):
            with self.subTest(name=name), self.assertRaises(ValueError):
                imported_node(nodes, name, op)


if __name__ == '__main__':
    unittest.main()
