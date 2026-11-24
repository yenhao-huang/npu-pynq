"""src/vendors sits below src/export and src/runtime: both import it, it imports neither.

``src/vendors/<vendor>/`` holds what one accelerator needs at run time:
``spec/`` (hardware facts, read by both export and runtime) and ``driver/``
(board side). Compile-time code for a device lives under
``src/export/devices/<device>/``."""
import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
VENDORS = ROOT / 'src/vendors'
DEVICES = ROOT / 'src/export/devices'
DELEGATES = ROOT / 'src/export/backend/delegates'

# What a vendor folder may hold: its two run-time parts, plus the package
# marker and a README. Anything else (a compiler, a dialect, a generator that
# writes into export) belongs under src/export/devices/<device>/.
VENDOR_DIRS = {'spec', 'driver'}
VENDOR_FILES = {'__init__.py', 'README.md'}
# A path or module name that reaches into the compile-time side.
COMPILE_TIME = re.compile(r'src[/\\.]export|export/devices|custom_dialect')
SKIP_DIRS = {'__pycache__', 'build'}
BINARY = {'.pyc', '.so', '.o', '.a', '.bit', '.hwh', '.elf', '.bin'}


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.append(node.module or '')
    return modules


def vendors() -> list[Path]:
    return sorted(p for p in VENDORS.iterdir() if p.is_dir() and p.name not in SKIP_DIRS)


def vendor_files() -> list[Path]:
    return sorted(p for p in VENDORS.rglob('*')
                  if p.is_file() and p.suffix not in BINARY
                  and not SKIP_DIRS & set(p.relative_to(VENDORS).parts))


class VendorBoundaryTests(unittest.TestCase):
    def test_vendors_import_neither_export_nor_runtime(self):
        for path in VENDORS.rglob('*.py'):
            for module in imported_modules(path):
                self.assertFalse(module.startswith(('src.export', 'src.runtime')), f'{path}: {module}')

    def test_vendor_parts_import_only_their_own_spec(self):
        # spec/ is the hardware contract and imports no other part; driver/
        # may use its own vendor's spec/ but nothing else under src/.
        allowed = {'spec': ('spec',), 'driver': ('driver', 'spec')}
        for vendor in vendors():
            for part, ok in allowed.items():
                for path in (vendor / part).rglob('*.py'):
                    for module in imported_modules(path):
                        if module.startswith('src.'):
                            prefixes = tuple(f'src.vendors.{vendor.name}.{p}' for p in ok)
                            self.assertTrue(module.startswith(prefixes), f'{path}: {module}')


class VendorHoldsRunTimeOnlyTests(unittest.TestCase):
    """The run-time / compile-time split between src/vendors and src/export/devices."""

    def test_a_vendor_holds_only_spec_and_driver(self):
        self.assertTrue(vendors(), 'no vendor under src/vendors')
        for vendor in vendors():
            with self.subTest(vendor=vendor.name):
                self.assertFalse((vendor / 'custom_dialect').exists(),
                                 f'move {vendor.name}/custom_dialect to src/export/devices/<device>/')
                self.assertTrue((vendor / 'spec').is_dir(), f'{vendor.name} has no spec/')
                dirs = {p.name for p in vendor.iterdir() if p.is_dir() and p.name not in SKIP_DIRS}
                files = {p.name for p in vendor.iterdir() if p.is_file()}
                self.assertLessEqual(dirs, VENDOR_DIRS,
                                     f'{vendor.name}: compile-time code goes to src/export/devices/')
                self.assertLessEqual(files, VENDOR_FILES, f'{vendor.name}: unexpected top-level file')

    def test_nothing_in_vendors_references_the_compile_time_side(self):
        self.maxDiff = None
        # No import, path, CMake add_subdirectory or generator output may point
        # from a vendor into src/export; the dependency runs export -> vendors.
        offenders = []
        for path in vendor_files():
            try:
                text = path.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:
                continue
            offenders += [f'{path.relative_to(ROOT).as_posix()}:{n}: {line.strip()}'
                          for n, line in enumerate(text.splitlines(), 1) if COMPILE_TIME.search(line)]
        self.assertEqual(offenders, [], 'src/vendors must not reference src/export or custom_dialect')

    def test_every_delegate_has_a_device_folder(self):
        delegates = sorted(p.name for p in DELEGATES.iterdir() if p.is_dir())
        self.assertTrue(delegates, 'no delegate under src/export/backend/delegates')
        for name in delegates:
            with self.subTest(device=name):
                self.assertTrue((DEVICES / name / 'CMakeLists.txt').is_file(),
                                f'src/export/devices/{name}/ is missing its compile-time code')


if __name__ == '__main__': unittest.main()
