"""The optional interpreter must be removable without affecting active code."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class BoundaryTests(unittest.TestCase):
    def test_active_python_has_no_jit_or_deprecate_imports(self):
        for base in ('src/export','src/runtime','src/vendors','src/test','examples','.github','tools'):
            for path in (ROOT/base).rglob('*.py'):
                if path.resolve() == Path(__file__).resolve(): continue
                tree = ast.parse(path.read_text(encoding='utf-8-sig'))
                for node in ast.walk(tree):
                    modules = ([a.name for a in node.names] if isinstance(node,ast.Import)
                               else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                    for module in modules:
                        self.assertFalse(module.startswith(('src.jit','src.deprecate')),str(path))

    def test_jit_has_no_imports_from_other_source_packages(self):
        for path in (ROOT/'src/deprecate/jit').rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node,ast.Import)
                           else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                for module in modules:
                    self.assertFalse(module.startswith('src.') and not module.startswith('src.deprecate.jit'),str(path))


if __name__ == '__main__': unittest.main()
