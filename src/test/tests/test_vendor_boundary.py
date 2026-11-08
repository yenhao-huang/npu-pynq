"""src/vendors sits below src/export and src/runtime: both import it, it imports neither."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.append(node.module or '')
    return modules


class VendorBoundaryTests(unittest.TestCase):
    def test_vendors_import_neither_export_nor_runtime(self):
        for path in (ROOT/'src/vendors').rglob('*.py'):
            for module in imported_modules(path):
                self.assertFalse(module.startswith(('src.export', 'src.runtime')), f'{path}: {module}')

    def test_vendor_parts_import_only_their_own_isa(self):
        # isa/ is the specification and imports no other part; compiler/ and
        # driver/ may use isa/ but never each other.
        allowed = {'isa': ('isa',), 'compiler': ('compiler', 'isa'), 'driver': ('driver', 'isa')}
        for vendor in (p for p in (ROOT/'src/vendors').iterdir() if (p/'isa').is_dir()):
            for part, ok in allowed.items():
                for path in (vendor/part).rglob('*.py'):
                    for module in imported_modules(path):
                        if module.startswith('src.'):
                            prefixes = tuple(f'src.vendors.{vendor.name}.{p}' for p in ok)
                            self.assertTrue(module.startswith(prefixes), f'{path}: {module}')


if __name__ == '__main__': unittest.main()
