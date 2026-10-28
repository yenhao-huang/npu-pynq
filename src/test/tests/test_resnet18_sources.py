"""Pinned assets and atomic downloads remain usable after source relocation."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.export.graphs.resnet18 import download_model as model
from src.export.graphs.resnet18 import download_demo_assets as demo
from src.export.graphs.resnet18 import download_gallery as gallery


METADATA = Path(model.__file__).parent


class Response(io.BytesIO):
    def geturl(self):
        return "https://download.pytorch.org/models/resnet18-f37072fd.pth"


class SourceTests(unittest.TestCase):
    def test_relocated_metadata_preserves_pins_and_gallery_licenses(self):
        source = model.load_source_metadata(METADATA / "model-source.json")
        self.assertEqual(source.sha256, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec")
        assets = demo.load_demo_metadata(METADATA / "demo-source.json")
        self.assertEqual(len(assets), 2)
        images = gallery.load_gallery_metadata(METADATA / "gallery-source.json")
        self.assertEqual(len(images), 5)
        self.assertTrue(all(image.license["spdx"] for image in images))

    def test_download_publishes_verified_bytes_and_rejects_overwrite(self):
        payload = b"checkpoint fixture"
        metadata = json.loads((METADATA / "model-source.json").read_bytes())
        metadata.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.json"
            path.write_bytes(model._canonical_json(metadata))
            opener = lambda *args, **kwargs: Response(payload)
            result = model.download_model(path, root / "model", opener=opener)
            self.assertEqual(result.read_bytes(), payload)
            with self.assertRaises(model.ModelDownloadError):
                model.download_model(path, result.parent, opener=opener)

    def test_corrupt_download_leaves_no_checkpoint_or_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(model.ModelDownloadError):
                model.download_model(METADATA / "model-source.json", root,
                                     opener=lambda *args, **kwargs: Response(b"corrupt"))
            self.assertEqual(list(root.iterdir()), [])
