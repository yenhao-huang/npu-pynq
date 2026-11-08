"""Pinned assets and atomic downloads remain usable after source relocation."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock

from src.deploy import resnet18_assets as source


METADATA = Path(source.__file__).parent


class Response(io.BytesIO):
    def geturl(self):
        return "https://download.pytorch.org/models/resnet18-f37072fd.pth"


class SourceTests(unittest.TestCase):
    def test_relocated_metadata_preserves_pins_and_gallery_licenses(self):
        pinned = source.load_source_metadata(METADATA / "resnet18-model-source.json")
        self.assertEqual(pinned.sha256, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec")
        assets = source.load_demo_metadata(METADATA / "resnet18-demo-source.json")
        self.assertEqual(len(assets), 2)
        images = source.load_gallery_metadata(METADATA / "resnet18-gallery-source.json")
        self.assertEqual(len(images), 5)
        self.assertTrue(all(image.license["spdx"] for image in images))

    def test_download_publishes_verified_bytes_and_rejects_overwrite(self):
        payload = b"checkpoint fixture"
        metadata = json.loads((METADATA / "resnet18-model-source.json").read_bytes())
        metadata.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.json"
            path.write_bytes(source._canonical_json(metadata))
            opener = lambda *args, **kwargs: Response(payload)
            result = source.download_model(path, root / "model", opener=opener)
            self.assertEqual(result.read_bytes(), payload)
            with self.assertRaises(source.SourceError):
                source.download_model(path, result.parent, opener=opener)

    def test_corrupt_download_leaves_no_checkpoint_or_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(source.SourceError):
                source.download_model(METADATA / "resnet18-model-source.json", root,
                                     opener=lambda *args, **kwargs: Response(b"corrupt"))
            self.assertEqual(list(root.iterdir()), [])

    def test_gallery_skips_verified_files_and_refuses_changed_ones(self):
        payload = b"gallery fixture"
        metadata = json.loads((METADATA / "resnet18-gallery-source.json").read_bytes())
        metadata["images"] = metadata["images"][:1]
        metadata["images"][0].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        url = metadata["images"][0]["url"]

        class GalleryResponse(io.BytesIO):
            def geturl(self):
                return url

        with tempfile.TemporaryDirectory() as directory, unittest.mock.patch("builtins.print"):
            root = Path(directory)
            path = root / "gallery.json"
            path.write_bytes(source._canonical_json(metadata))
            opener = lambda *args, **kwargs: GalleryResponse(payload)
            source.download_gallery(path, root, opener=opener)
            image = root / metadata["images"][0]["filename"]
            self.assertEqual(image.read_bytes(), payload)
            source.download_gallery(path, root, opener=lambda *a, **k: self.fail("verified file re-downloaded"))
            image.write_bytes(b"changed")
            with self.assertRaises(source.SourceError):
                source.download_gallery(path, root, opener=opener)

    def test_command_line_reports_a_failed_download(self):
        with tempfile.TemporaryDirectory() as directory, unittest.mock.patch("sys.stderr"):
            missing = Path(directory) / "missing.json"
            self.assertEqual(source.main(["model", "--metadata", str(missing), "--output-dir", directory]), 1)
