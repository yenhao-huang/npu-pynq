import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from src.vendors.npu_pynq.driver.overlay.build import write_manifest
from src.test.tests.test_verify_overlay import HWH

ROOT = Path(__file__).resolve().parents[3]
from src.export.delivery import resnet18 as package


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root/'model'
        compiled = self.model/'aot'
        (compiled/'pynq').mkdir(parents=True)
        for name in ('weights.bin','programs.json','programs.bin','pynq/model.so'):
            (compiled/name).write_bytes(b'fixture')
        self.manifest = {'format':'npu-package/1','meta':{'family':'resnet18'},
                         'libraries':{'pynq':'pynq/model.so'},
                         'limits':{'rows':2,'columns':2,'max_k':256}}
        (compiled/'manifest.json').write_text(json.dumps(self.manifest))
        for name in ('imagenet-classes.txt','demo-image.jpg'):
            (self.model/name).write_bytes(b'fixture')
        self.artifacts = self.root/'artifacts'
        self.artifacts.mkdir()
        (self.artifacts/'npu_matrix.bit').write_bytes(b'fixture')
        (self.artifacts/'npu_matrix.hwh').write_text(HWH)
        write_manifest(self.artifacts,source_commit='a'*40,vivado_version='fixture',array_size=2)

    def test_deterministic_board_package_excludes_jit_and_checkpoints(self):
        first,second = self.root/'one.zip', self.root/'two.zip'
        for target in (first,second):
            package.build_archive(ROOT,self.model,self.artifacts,target,'local-test','a'*40)
        self.assertEqual(first.read_bytes(),second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            self.assertIn('model/aot/pynq/model.so',names)
            self.assertFalse(any('jit' in name or 'export' in name or name.endswith('.pth') for name in names))
            destination = self.root/'unpacked'
            archive.extractall(destination)
        package.verify_delivery(destination)
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        result = subprocess.run(
            [sys.executable, '-m', 'src.runtime.image_cli',
             '--package-root', '.', '--verify-only'],
            cwd=destination, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS: verified ResNet18 AOT delivery', result.stdout)
        (destination/'model/aot/pynq/model.so').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            package.verify_delivery(destination)

    def test_reject_model_path_traversal_and_geometry_mismatch(self):
        self.manifest['libraries']['pynq'] = '../outside.so'
        (self.model/'outside.so').write_bytes(b'not-a-library')
        path = self.model/'aot/manifest.json'
        path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError): package.validate_model(self.model)
        self.manifest['libraries']['pynq'] = 'pynq/model.so'
        self.manifest['limits']['rows'] = 16
        path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'geometry'):
            package.build_archive(ROOT,self.model,self.artifacts,self.root/'bad.zip','local-test','a'*40)


if __name__ == '__main__': unittest.main()
