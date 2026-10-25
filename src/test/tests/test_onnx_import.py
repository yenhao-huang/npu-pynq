"""Source inventory and import failure contracts for ONNX-MLIR."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper

from src.export.compilers.onnx_mlir.support import inspect_source
from src.export.compilers.onnx_mlir.toolchain import Compiler, OnnxMlirError
from src.export.compilers.onnx_mlir.pipeline import prepare_import


def fixture(path, nodes=None):
    inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, ['tokens', 4])]
    outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, ['tokens', 4])]
    nodes = nodes or [helper.make_node('Sigmoid', ['input'], ['output'])]
    model = helper.make_model(helper.make_graph(nodes, 'fixture', inputs, outputs),
                              opset_imports=[helper.make_opsetid('', 14),
                                             helper.make_opsetid('com.microsoft', 1)])
    model.ir_version = 9
    onnx.save(model, path)


class OnnxImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.onnx'
        fixture(self.source)

    def test_inventory_is_deterministic_and_does_not_claim_compilation(self):
        report = inspect_source(self.source)
        self.assertEqual(report, inspect_source(self.source))
        self.assertEqual(report['graphs'][0]['inputs'][0]['shape'], ['tokens', 4])
        self.assertEqual(report['operators'][0]['op'], 'Sigmoid')
        self.assertEqual(report['operators'][0]['status'], 'unverified')
        self.assertFalse(report['compile_ready'])
        self.assertEqual(len(report['source']['sha256']), 64)
        json.dumps(report)

    def test_corrupt_source_does_not_create_output_or_report_success(self):
        self.source.write_bytes(b'not an ONNX protobuf')
        with self.assertRaisesRegex(ValueError, 'Cannot read ONNX source'):
            prepare_import(self.source, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())

    def test_native_override_does_not_claim_the_pinned_revision(self):
        def compile_fixture(*args, **kwargs):
            (self.root / 'out/model.onnx.mlir').write_text('module {}')
            return subprocess.CompletedProcess([], 0, stdout='')

        with patch('subprocess.run', side_effect=compile_fixture):
            report = prepare_import(self.source, self.root / 'out', compiler=Compiler(binary='native'))
        self.assertEqual(report['provenance'], 'unverified-native')
        self.assertNotIn('revision', report['toolchain'])

    def test_custom_domain_is_not_confused_with_standard_operator(self):
        fixture(self.source, [helper.make_node('Sigmoid', ['input'], ['output'],
                                              domain='com.microsoft')])
        operator = inspect_source(self.source)['operators'][0]
        self.assertEqual(operator['domain'], 'com.microsoft')
        self.assertEqual(operator['status'], 'requires_conversion')

    def test_unknown_standard_op_and_attributes_are_visible(self):
        fixture(self.source, [helper.make_node('SimplifiedLayerNormalization',
                         ['input'], ['output'], epsilon=1e-5, axis=-1)])
        report = inspect_source(self.source)
        self.assertEqual(report['operators'][0]['status'], 'requires_conversion')
        self.assertEqual(report['graphs'][0]['nodes'][0]['attributes']['axis'], -1)

    def test_nested_graph_custom_operator_is_reported(self):
        branch = helper.make_graph([helper.make_node('RotaryEmbedding', ['input'],
                   ['output'], domain='com.microsoft')], 'branch', [], [])
        fixture(self.source, [helper.make_node('If', ['condition'], ['output'],
                                              then_branch=branch, else_branch=branch)])
        report = inspect_source(self.source)
        rotary = next(entry for entry in report['operators'] if entry['op'] == 'RotaryEmbedding')
        self.assertEqual(rotary['count'], 2)
        self.assertEqual(len(report['graphs']), 3)

    def test_import_failure_preserves_inventory_and_diagnostics(self):
        failed = subprocess.CompletedProcess([], 1, stdout='unsupported operator')
        with patch('subprocess.run', return_value=failed):
            with self.assertRaisesRegex(OnnxMlirError, 'compiler.log'):
                prepare_import(self.source, self.root / 'out', compiler=Compiler(binary='compiler'))
        report = json.loads((self.root / 'out/import.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertTrue((self.root / 'out/inventory.json').is_file())
        self.assertIn('unsupported operator', (self.root / 'out/compiler.log').read_text())

    def test_missing_tool_is_actionable(self):
        with patch('subprocess.run', side_effect=FileNotFoundError('missing')):
            with self.assertRaisesRegex(OnnxMlirError, 'not found'):
                prepare_import(self.source, self.root / 'out', compiler=Compiler(binary='missing'))

    def test_successful_exit_without_ir_is_rejected(self):
        success = subprocess.CompletedProcess([], 0, stdout='')
        with patch('subprocess.run', return_value=success):
            with self.assertRaisesRegex(OnnxMlirError, 'MLIR'):
                prepare_import(self.source, self.root / 'out', compiler=Compiler(binary='compiler'))

    def test_nonempty_existing_directory_is_not_overwritten(self):
        output = self.root / 'out'
        output.mkdir()
        existing = output / 'model.onnx.mlir'
        existing.write_text('previous result')
        with self.assertRaisesRegex(ValueError, 'empty'):
            prepare_import(self.source, output)
        self.assertEqual(existing.read_text(), 'previous result')

    def test_docker_command_uses_pinned_image_and_readonly_input(self):
        command = Compiler().command(self.source, self.root / 'out')
        self.assertTrue(any('@sha256:' in arg for arg in command))
        self.assertIn('--network=none', command)
        self.assertTrue(any('target=/input,readonly' in arg for arg in command))
        self.assertIn('--EmitONNXIR', command)

    def test_timeout_removes_only_the_container_started_for_this_import(self):
        timeout = subprocess.TimeoutExpired('docker', 1, output=b'partial diagnostic')
        cleanup = subprocess.CompletedProcess([], 0, stdout='removed', stderr='')
        with patch('subprocess.run', side_effect=[timeout, cleanup]) as run:
            with self.assertRaisesRegex(OnnxMlirError, 'timed out'):
                prepare_import(self.source, self.root / 'out')
        import_command = run.call_args_list[0].args[0]
        container_name = import_command[import_command.index('--name') + 1]
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'rm', '--force', container_name])
        self.assertIn('partial diagnostic', (self.root / 'out/compiler.log').read_text())

    def test_external_tensor_data_is_inventoried_but_import_is_rejected(self):
        model = onnx.load(self.source)
        tensor = helper.make_tensor('weight', TensorProto.FLOAT, [1], [1.0])
        tensor.ClearField('float_data')
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key='location', value='weights.bin')
        model.graph.initializer.append(tensor)
        self.source.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(OnnxMlirError, 'External tensor data'):
            prepare_import(self.source, self.root / 'out')
        report = json.loads((self.root / 'out/inventory.json').read_text())
        self.assertEqual(report['external_data'][0]['external_data']['location'], 'weights.bin')

    @unittest.skipUnless(os.environ.get('NPU_TEST_ONNX_MLIR') == '1',
                         'Requires the pinned ONNX-MLIR Docker image')
    def test_real_sigmoid_import_preserves_dynamic_tensor_shape(self):
        report = prepare_import(self.source, self.root / 'out')
        self.assertEqual(report['status'], 'imported')
        self.assertFalse(report['compile_ready'])
        module = (self.root / 'out/model.onnx.mlir').read_text()
        self.assertIn('"onnx.Sigmoid"', module)
        self.assertIn('tensor<?x4xf32>', module)


if __name__ == '__main__':
    unittest.main()
