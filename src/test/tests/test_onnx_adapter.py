"""Identity validation and real MLIR parsing at the ONNX import boundary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.export.compilers.mlir import toolchain
from src.export.compilers.onnx_mlir import adapter


# Generic syntax emitted by the pinned ONNX-MLIR 0.5.1 Sigmoid fixture.
IMPORTED_SIGMOID = '''
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{onnx.name = "input"}],
    function_type = (tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main_graph"}> ({
  ^bb0(%arg0: tensor<?x4xf32>):
    %0 = "onnx.Sigmoid"(%arg0) {onnx_node_name = "sigmoid"} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    "func.return"(%0) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) {"onnx-mlir.compiler_version" = "onnx-mlir version 0.5.1 (20f2b50)"} : () -> ()
'''


class ImportIdentityTests(unittest.TestCase):
    def test_wrong_source_or_changed_ir_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.onnx'
            source.write_bytes(b'source')
            module = root / 'model.onnx.mlir'
            module.write_text(IMPORTED_SIGMOID)
            report = {'status': 'imported', 'source_sha256': adapter.file_digest(source),
                      'artifact': module.name, 'artifact_sha256': adapter.file_digest(module)}
            report_path = root / 'import.json'
            report_path.write_text(json.dumps(report))
            with patch.object(adapter, 'inspect_ir', return_value={'functions': [1]}) as inspect:
                graph, _, directory = adapter.load_imported(source, root)
                self.assertEqual(graph['functions'], [1])
                self.assertEqual(directory, root)
                self.assertEqual(inspect.call_count, 1)
                module.write_text(IMPORTED_SIGMOID + '\n// changed\n')
                with self.assertRaisesRegex(ValueError, 'artifact identity'):
                    adapter.load_imported(source, root)
                self.assertEqual(inspect.call_count, 1)
                source.write_bytes(b'other source')
                with self.assertRaisesRegex(ValueError, 'source SHA-256'):
                    adapter.load_imported(source, root)

    def test_real_parser_preserves_dynamic_shapes_names_and_ssa_edges(self):
        try:
            toolchain.llvm_bin()
        except toolchain.ToolchainError:
            self.skipTest('LLVM 22 with MLIR is required')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = root / 'model.mlir'
            module.write_text(IMPORTED_SIGMOID)
            graph = adapter.inspect_ir(module, root / 'graph.json')
            function = graph['functions'][0]
            node = function['nodes'][0]
            self.assertEqual(graph['mlir_major'], 22)
            self.assertEqual(function['inputs'][0]['name'], 'input')
            self.assertEqual(function['inputs'][0]['type'], 'tensor<?x4xf32>')
            self.assertEqual(node['op'], 'onnx.Sigmoid')
            self.assertEqual(node['inputs'], [function['inputs'][0]['id']])
            self.assertEqual(function['nodes'][-1]['inputs'], [node['outputs'][0]['id']])
            self.assertEqual(node['attributes']['onnx_node_name'], 'sigmoid')


if __name__ == '__main__':
    unittest.main()
