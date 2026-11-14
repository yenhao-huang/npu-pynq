"""The export front end's stages import one way:

    every stage stands alone;  pipeline and cli above all

load builds a float model but never quantizes or exports it; quantization
takes a model from anywhere; export_graph never loads or lowers; to_mlir only
consumes ExportedPrograms. Only pipeline.py passes a model from one to the next."""

import ast
from pathlib import Path
import re
import unittest

FRONTEND = Path(__file__).resolve().parents[2] / 'export' / 'frontend'
EXPORT = FRONTEND.parent
CUSTOM_DIALECT = EXPORT.parent / 'vendors' / 'npu_pynq' / 'custom_dialect'
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


def link_libraries(cmake: Path, target: str) -> list[str]:
    """The libraries every target_link_libraries(``target`` ...) in ``cmake`` names."""
    text = re.sub(r'#.*', '', cmake.read_text(encoding='utf-8'))
    found = []
    for body in re.findall(rf'target_link_libraries\(\s*{re.escape(target)}\s+([^)]*)\)', text):
        found += [w for w in body.split() if w not in ('PRIVATE', 'PUBLIC', 'INTERFACE')]
    return found


class ExportToolLinkTest(unittest.TestCase):
    """docs/rules/lib/export_stages.md: the middle end links a device's claim rule
    (NpuMatch), never its dialect, lowering or encoder (NpuDialect)."""

    def test_middle_end_links_the_claim_rule_only(self):
        libraries = link_libraries(EXPORT / 'middle-end' / 'CMakeLists.txt', 'pynqo-middle-end')
        self.assertIn('NpuMatch', libraries)
        self.assertNotIn('NpuDialect', libraries)
        self.assertNotIn('MLIR', libraries)

    def test_claim_rule_brings_no_mlir_of_its_own(self):
        # Each consumer supplies MLIR in its own link style; NpuMatch links nothing.
        self.assertEqual(link_libraries(CUSTOM_DIALECT / 'CMakeLists.txt', 'NpuMatch'), [])
        self.assertIn('NpuMatch', link_libraries(CUSTOM_DIALECT / 'CMakeLists.txt', 'NpuDialect'))
        for path in (CUSTOM_DIALECT / 'lib' / 'Match.cpp', CUSTOM_DIALECT / 'include' / 'Npu' / 'NpuMatch.h'):
            with self.subTest(path=path.name):
                self.assertNotRegex(path.read_text(encoding='utf-8'), r'NpuDialect|NpuEncoder|\.inc"')

    def test_device_attributes_are_spelled_alike_in_both_tools(self):
        spelled = re.compile(r'"(pynqo\.\w+)"')
        middle = set(spelled.findall((EXPORT / 'middle-end' / 'linalg' / 'transforms' / 'Passes.h')
                                     .read_text(encoding='utf-8')))
        backend = set()
        for name in ('pynqo_backend.cpp', 'cpu/delegate_calls.cpp', 'delegates/npu/lower.cpp'):
            backend |= set(spelled.findall((EXPORT / 'backend' / name).read_text(encoding='utf-8')))
        self.assertEqual(middle, {'pynqo.devices', 'pynqo.device', 'pynqo.delegate_id'})
        self.assertEqual(backend, middle)


if __name__ == '__main__':
    unittest.main()
