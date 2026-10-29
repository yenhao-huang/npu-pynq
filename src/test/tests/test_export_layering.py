"""The export front end's stages import one way:

    ir <- load <- quantize <- model_arch -> compilers;  pipeline and cli above all

load reads sources and never quantizes; quantize never builds MLIR; the
MLIR operators consume quantized weights and never quantize; and model
architectures never import each other."""

import ast
from pathlib import Path
import unittest

EXPORT = Path(__file__).resolve().parents[2] / 'export'
ABOVE = ('pipeline', 'cli')

# package -> src.export subpackages its modules must not import
FORBIDDEN = {
    'ir': ('load', 'quantize', 'model_arch', 'delivery', *ABOVE),
    'load': ('quantize', 'model_arch', 'compilers.mlir', 'delivery', *ABOVE),
    'quantize': ('load', 'compilers', 'model_arch', 'delivery', *ABOVE),
    'compilers': ('load', 'quantize', 'model_arch', 'delivery', *ABOVE),
    'model_arch': ('delivery', *ABOVE),
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
        for package, forbidden in FORBIDDEN.items():
            for path in sorted((EXPORT / package).rglob('*.py')):
                for module in imports(path):
                    for target in forbidden:
                        with self.subTest(path=path.relative_to(EXPORT).as_posix(), module=module):
                            self.assertFalse(module == f'src.export.{target}' or
                                             module.startswith(f'src.export.{target}.'),
                                             f'{path.name} imports {module}')

    def test_architectures_never_import_each_other(self):
        for path in sorted((EXPORT / 'model_arch').glob('*.py')):
            if path.name in ('__init__.py', 'base.py'):
                continue
            for module in imports(path):
                if module.startswith('src.export.model_arch.'):
                    with self.subTest(path=path.name, module=module):
                        self.assertTrue(module.startswith('src.export.model_arch.base'), module)

    def test_rules_cover_every_stage_package(self):
        packages = {p.name for p in EXPORT.iterdir() if (p / '__init__.py').is_file()}
        self.assertLessEqual(set(FORBIDDEN), packages)


if __name__ == '__main__':
    unittest.main()
