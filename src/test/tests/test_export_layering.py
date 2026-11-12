"""The export front end's stages import one way:

    every stage stands alone;  pipeline and cli above all

load builds a float model but never quantizes or exports it; quantization
takes a model from anywhere; export_graph never loads or lowers; to_mlir only
consumes ExportedPrograms. Only pipeline.py passes a model from one to the next."""

import ast
from pathlib import Path
import unittest

FRONTEND = Path(__file__).resolve().parents[2] / 'export' / 'frontend'
ABOVE = ('src.export.pipeline', 'src.export.cli')
STAGES = ('load', 'quantization', 'export_graph', 'to_mlir')

# stage -> front-end stages its modules must not import
FORBIDDEN = {
    'quantization': ('load', 'export_graph', 'to_mlir'),
    'load': ('quantization', 'export_graph', 'to_mlir'),
    'export_graph': ('load', 'quantization', 'to_mlir'),
    'to_mlir': ('load', 'quantization', 'export_graph'),
}


def imports(path: Path) -> list[str]:
    """Every module ``path`` imports, nested imports included."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names += [f'{node.module}.{alias.name}' for alias in node.names]
    return names


class ExportLayeringTest(unittest.TestCase):
    def test_stages_import_one_way(self):
        for stage, forbidden in FORBIDDEN.items():
            for path in sorted((FRONTEND / stage).rglob('*.py')):
                for module in imports(path):
                    with self.subTest(path=path.relative_to(FRONTEND).as_posix(), module=module):
                        self.assertFalse(module.startswith(ABOVE), f'{path.name} imports {module}')
                        for target in forbidden:
                            self.assertFalse(module.startswith(f'src.export.frontend.{target}'),
                                             f'{path.name} imports {module}')

    def test_quantization_knows_no_device(self):
        # docs/rules/lib/register_quant.md
        for path in sorted((FRONTEND / 'quantization').rglob('*.py')):
            with self.subTest(path=path.relative_to(FRONTEND).as_posix()):
                self.assertNotRegex(path.read_text(encoding='utf-8'), r'(?i)npu|src\.vendors|vendor')

    def test_an_npu_package_needs_a_scheme_the_npu_accepts(self):
        from src.export.pipeline import check_scheme

        check_scheme({'devices': ['cpu', 'npu'], 'quantization': 'w8a8'})
        check_scheme({'devices': ['cpu']})
        for scheme in (None, 'Int8WeightOnlyConfig'):
            with self.assertRaises(ValueError):
                check_scheme({'devices': ['cpu', 'npu'], 'quantization': scheme})
        check_scheme({'devices': ['cpu', 'risc_v_npu'], 'quantization': 'w8a8'})
        with self.assertRaises(ValueError):
            check_scheme({'devices': ['cpu', 'risc_v_npu'], 'quantization': None})

    def test_rules_cover_every_stage(self):
        stages = {p.name for p in FRONTEND.iterdir() if (p / '__init__.py').is_file()}
        self.assertEqual(set(FORBIDDEN), stages)
        self.assertEqual(set(STAGES), stages)

    def test_runtime_and_vendor_never_import_export(self):
        src = FRONTEND.parents[1]
        for root in (src / 'runtime', src / 'vendors'):
            for path in sorted(root.rglob('*.py')):
                for module in imports(path):
                    with self.subTest(path=path.relative_to(src).as_posix(), module=module):
                        self.assertFalse(module.startswith('src.export'), f'{path.name} imports {module}')


if __name__ == '__main__':
    unittest.main()
