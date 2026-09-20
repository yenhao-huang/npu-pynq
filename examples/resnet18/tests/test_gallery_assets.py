"""Tests for the pinned live-demo gallery metadata and its fail-closed downloader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "resnet18"
SCRIPT_DIR = EXAMPLE_ROOT / "scripts"
GALLERY_SOURCE = EXAMPLE_ROOT / "gallery-source.json"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


gallery = load_module("download_gallery", SCRIPT_DIR / "download_gallery.py")


def canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_metadata(directory: Path, value: object) -> Path:
    path = directory / "gallery-source.json"
    path.write_text(canonical(value), encoding="utf-8")
    return path


def committed_metadata() -> dict:
    return json.loads(GALLERY_SOURCE.read_text(encoding="utf-8"))


class GalleryMetadataTests(unittest.TestCase):
    def test_committed_metadata_loads(self):
        images = gallery.load_gallery_metadata(GALLERY_SOURCE)
        self.assertEqual(len(images), 5)

    def test_committed_metadata_is_canonical(self):
        raw = GALLERY_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(raw, canonical(json.loads(raw)))

    def test_every_image_is_fully_pinned(self):
        for image in gallery.load_gallery_metadata(GALLERY_SOURCE):
            self.assertRegex(image.sha256, r"\A[0-9a-f]{64}\Z")
            self.assertGreater(image.byte_length, 0)
            self.assertTrue(image.url.startswith("https://upload.wikimedia.org/wikipedia/commons/"))
            self.assertTrue(image.expected_class.strip())
            for field in ("attribution", "notes", "source_page", "spdx"):
                self.assertTrue(image.license[field].strip(), field)

    def test_filenames_are_unique_safe_basenames(self):
        names = [image.filename for image in gallery.load_gallery_metadata(GALLERY_SOURCE)]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertEqual(name, Path(name).name)
            self.assertFalse(name.startswith("."))

    def test_non_canonical_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = directory / "gallery-source.json"
            path.write_text(json.dumps(committed_metadata(), indent=2), encoding="utf-8")
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_unknown_top_level_field_is_rejected(self):
        value = committed_metadata()
        value["extra"] = "no"
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_wrong_magic_is_rejected(self):
        value = committed_metadata()
        value["magic"] = "SOMETHING_ELSE"
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_unsupported_format_is_rejected(self):
        value = committed_metadata()
        value["format"] = {"major": 2, "minor": 0}
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_unapproved_host_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["url"] = "https://evil.example.com/wikipedia/commons/a/ab/x.jpg"
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_plain_http_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["url"] = value["images"][0]["url"].replace("https://", "http://", 1)
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_url_outside_the_approved_path_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["url"] = "https://upload.wikimedia.org/elsewhere/x.jpg"
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_traversing_filename_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["filename"] = "../escape.jpg"
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_short_digest_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["sha256"] = "abc123"
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_non_positive_length_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["bytes"] = 0
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_incomplete_licence_is_rejected(self):
        value = committed_metadata()
        del value["images"][0]["license"]["attribution"]
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_empty_expected_class_is_rejected(self):
        value = committed_metadata()
        value["images"][0]["expected_class"] = "   "
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_duplicate_filename_is_rejected(self):
        value = committed_metadata()
        value["images"][1]["filename"] = value["images"][0]["filename"]
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)

    def test_empty_image_list_is_rejected(self):
        value = committed_metadata()
        value["images"] = []
        with tempfile.TemporaryDirectory() as raw:
            path = write_metadata(Path(raw), value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.load_gallery_metadata(path)


class GalleryPublicationTests(unittest.TestCase):
    def _single_image_metadata(self, payload: bytes) -> dict:
        value = committed_metadata()
        entry = dict(value["images"][0])
        entry["bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        value["images"] = [entry]
        return value

    def test_existing_matching_file_is_kept(self):
        payload = b"pinned-bytes"
        value = self._single_image_metadata(payload)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = write_metadata(directory, value)
            destination = directory / value["images"][0]["filename"]
            destination.write_bytes(payload)
            published = gallery.download_gallery(path, directory)
            self.assertEqual(len(published), 1)
            self.assertEqual(destination.read_bytes(), payload)

    def test_existing_differing_file_is_never_overwritten(self):
        value = self._single_image_metadata(b"pinned-bytes")
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = write_metadata(directory, value)
            destination = directory / value["images"][0]["filename"]
            destination.write_bytes(b"something-else")
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.download_gallery(path, directory)
            self.assertEqual(destination.read_bytes(), b"something-else")

    def test_missing_model_workspace_is_rejected(self):
        value = self._single_image_metadata(b"pinned-bytes")
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            path = write_metadata(directory, value)
            with self.assertRaises(gallery.GalleryAssetError):
                gallery.download_gallery(path, directory / "absent")


if __name__ == "__main__":
    unittest.main()
