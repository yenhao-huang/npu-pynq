"""The ResNet-18 release bundle: deterministic, sealed, verified on the board side."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from src.deploy import resnet18 as release
from src.runtime.share.bundle import verify_bundle
from src.test.tests.test_verify_overlay import HWH
from src.vendors.npu_pynq.driver.overlay.build import write_manifest


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / 'package'
        (self.package / 'arm-cortex-a9').mkdir(parents=True)
        for name in ('weights.bin', 'programs.txt', 'programs.bin', 'arm-cortex-a9/model.so'):
            (self.package / name).write_bytes(b'fixture')
        self.manifest = {'format': 'pynqo-package/2', 'model': 'resnet18',
                         'libraries': {'arm-cortex-a9': 'arm-cortex-a9/model.so'},
                         'delegates': [{'id': 0, 'device': 'npu', 'index': 0}],
                         'classifier': {'input': {'shape': [224, 224, 3]}, 'classes': 1000},
                         'npu_limits': {'rows': 2, 'columns': 2, 'max_k': 256}}
        (self.package / 'manifest.json').write_text(json.dumps(self.manifest))
        self.assets = self.root / 'model'
        self.assets.mkdir()
        for name in ('imagenet-classes.txt', 'demo-image.jpg'):
            (self.assets / name).write_bytes(b'fixture')
        self.overlay = self.root / 'artifacts'
        self.overlay.mkdir()
        (self.overlay / 'npu_matrix.bit').write_bytes(b'fixture')
        (self.overlay / 'npu_matrix.hwh').write_text(HWH)
        write_manifest(self.overlay, source_commit='a' * 40, vivado_version='fixture', array_size=2)

    def build(self, name):
        archive = self.root / f'{name}.zip'
        release.main([str(self.package), '--overlay', str(self.overlay), '--assets', str(self.assets),
                      '--out', str(self.root / name), '--release-tag', 'local-test',
                      '--source-commit', 'a' * 40, '--archive', str(archive)])
        return archive

    def test_deterministic_sealed_bundle_verifies_on_the_board_side(self):
        first, second = self.build('one'), self.build('two')
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            destination = self.root / 'unpacked'
            archive.extractall(destination)
        self.assertIn('package/arm-cortex-a9/model.so', names)
        self.assertIn('src/vendors/npu_pynq/driver/npu_rt/npu_rt.h', names)
        self.assertFalse(any(name.startswith('src/export') or name.endswith('.pth') for name in names))
        verify_bundle(destination)
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable, '-m', 'src.runtime.cli', 'classifier', '--package-root', '.',
                                 '--verify-only'], cwd=destination, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS: verified the release bundle', result.stdout)
        (destination / 'package/arm-cortex-a9/model.so').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_bundle(destination)

    def test_rejects_a_package_compiled_for_another_array(self):
        self.manifest['npu_limits']['rows'] = 16
        (self.package / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'different array'):
            self.build('bad')

    def test_rejects_a_package_without_the_board_library(self):
        self.manifest['libraries'] = {'x86_64-generic': 'x86_64-generic/model.so'}
        (self.package / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'arm-cortex-a9'):
            release.check(self.package, self.assets)


if __name__ == '__main__':
    unittest.main()
