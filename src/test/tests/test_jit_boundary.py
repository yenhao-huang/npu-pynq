"""The optional interpreter must be removable without affecting active code."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class BoundaryTests(unittest.TestCase):

    def test_jit_has_no_imports_from_other_source_packages(self):
        for path in (ROOT/'src/jit').rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node,ast.Import)
                           else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                for module in modules:
                    self.assertFalse(module.startswith('src.') and not module.startswith('src.jit'),str(path))


if __name__ == '__main__': unittest.main()
