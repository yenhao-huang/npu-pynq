"""Tests for the pinned real-image demo assets, preparation, and notebook."""

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


class FakeResponse:
    def __init__(self, payload: bytes, final_url: str):
        self.payload = payload
        self.final_url = final_url
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def geturl(self) -> str:
        return self.final_url

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class DemoMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module(
            "resnet18_download_demo_assets", SCRIPT_DIR / "download_demo_assets.py"
        )
        cls.metadata_path = EXAMPLE_ROOT / "demo-source.json"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def raw(self) -> dict:
        return json.loads(self.metadata_path.read_bytes())

    def write_metadata(self, value: dict) -> Path:
        path = self.root / "demo-source.json"
        path.write_bytes(
            (
                json.dumps(
                    value, allow_nan=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
        )
        return path

    def test_committed_metadata_is_canonical_and_complete(self):
        raw = self.metadata_path.read_bytes()
        value = json.loads(raw)
        canonical = (
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.assertEqual(raw, canonical)
        assets = self.module.load_demo_metadata(self.metadata_path)
        self.assertEqual(
            sorted(asset.kind for asset in assets),
            ["class-names", "sample-image"],
        )
        sample = next(asset for asset in assets if asset.kind == "sample-image")
        self.assertTrue(sample.expected_class)
        for asset in assets:
            self.assertTrue(
                asset.url.startswith(
                    "https://raw.githubusercontent.com/pytorch/hub/"
                )
            )

    def test_every_pinned_asset_declares_a_license(self):
        for entry in self.raw()["assets"]:
            self.assertEqual(entry["license"]["spdx"], "BSD-3-Clause")
            self.assertTrue(entry["license"]["notes"].strip())

    def test_no_demo_asset_is_committed(self):
        for entry in self.raw()["assets"]:
            self.assertFalse((EXAMPLE_ROOT / entry["filename"]).exists())
            self.assertFalse((EXAMPLE_ROOT / "model" / entry["filename"]).is_symlink())
        tracked = {path.name for path in EXAMPLE_ROOT.glob("*")}
        self.assertNotIn("demo-image.jpg", tracked)

    def test_unapproved_host_is_rejected(self):
        value = self.raw()
        value["assets"][0]["url"] = "https://example.invalid/imagenet_classes.txt"
        with self.assertRaises(self.module.DemoAssetError):
            self.module.load_demo_metadata(self.write_metadata(value))

    def test_unpinned_revision_is_rejected(self):
        value = self.raw()
        value["assets"][0]["url"] = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with self.assertRaises(self.module.DemoAssetError):
            self.module.load_demo_metadata(self.write_metadata(value))

    def test_non_canonical_metadata_is_rejected(self):
        path = self.root / "demo-source.json"
        path.write_bytes(json.dumps(self.raw(), indent=2).encode("utf-8"))
        with self.assertRaises(self.module.DemoAssetError):
            self.module.load_demo_metadata(path)

    def test_missing_asset_kind_is_rejected(self):
        value = self.raw()
        value["assets"] = value["assets"][:1]
        with self.assertRaises(self.module.DemoAssetError):
            self.module.load_demo_metadata(self.write_metadata(value))

    def test_download_publishes_only_digest_matched_bytes(self):
        payload = {
            "imagenet-classes.txt": b"x" * 10,
            "demo-image.jpg": b"y" * 20,
        }
        value = self.raw()
        for entry in value["assets"]:
            data = payload[entry["filename"]]
            entry["bytes"] = len(data)
            entry["sha256"] = hashlib.sha256(data).hexdigest()
        metadata_path = self.write_metadata(value)

        def opener(request, timeout=None):
            name = request.full_url.rsplit("/", 1)[1]
            key = (
                "imagenet-classes.txt"
                if name == "imagenet_classes.txt"
                else "demo-image.jpg"
            )
            return FakeResponse(payload[key], request.full_url)

        published = self.module.download_demo_assets(
            metadata_path, self.root / "model", opener=opener
        )
        self.assertEqual(len(published), 2)
        for path in published:
            self.assertTrue(path.is_file())

    def test_digest_mismatch_leaves_no_file_behind(self):
        value = self.raw()
        value["assets"][0]["bytes"] = 4
        value["assets"][0]["sha256"] = "0" * 64
        metadata_path = self.write_metadata(value)

        def opener(request, timeout=None):
            return FakeResponse(b"abcd", request.full_url)

        destination = self.root / "model"
        with self.assertRaises(self.module.DemoAssetError):
            self.module.download_demo_assets(
                metadata_path, destination, opener=opener
            )
        self.assertEqual(list(destination.glob("*")), [])


class PreparationScriptTest(unittest.TestCase):
    def setUp(self):
        self.text = (SCRIPT_DIR / "prepare_demo_image.py").read_text(
            encoding="utf-8"
        )

    def test_preparation_never_claims_physical_board_evidence(self):
        self.assertNotIn("PASS [physical-pynq-z1]", self.text)
        self.assertIn('"physical_board": False', self.text)
        self.assertIn("compare_integer_captures", self.text)
        self.assertIn("execute_quantized_graph_reference", self.text)

    def test_image_is_selectable_without_editing_runtime_code(self):
        self.assertIn('"--image"', self.text)
        self.assertIn('"--expected-class"', self.text)
        runtime_sources = sorted(
            (REPOSITORY_ROOT / "src" / "runtime").glob("*.py")
        )
        self.assertTrue(runtime_sources)
        for path in runtime_sources:
            self.assertNotIn("demo-image", path.read_text(encoding="utf-8"))

    def test_record_pins_provenance_and_preprocessing(self):
        for field in (
            '"preprocessing": preprocessing_contract()',
            '"host_top_k"',
            '"captures"',
            '"expected"',
            '"integer_reference"',
        ):
            self.assertIn(field, self.text)

    def test_shared_helpers_are_not_duplicated(self):
        verifier = (SCRIPT_DIR / "verify_model.py").read_text(encoding="utf-8")
        for module_text in (self.text, verifier):
            self.assertIn("from host_reference import", module_text)
            self.assertNotIn("class HostMatrixBackend:", module_text)


class NotebookDemoTest(unittest.TestCase):
    """The deployed notebook is the demonstration path, not an acceptance path."""

    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(
            (EXAMPLE_ROOT / "resnet18.ipynb").read_text(encoding="utf-8")
        )
        cls.source = "\n".join(
            line
            for cell in cls.notebook["cells"]
            for line in cell.get("source", [])
        )

    def test_notebook_stays_output_free(self):
        code_cells = [
            cell for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(code_cells)
        for cell in code_cells:
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])

    def test_structure_alternates_and_every_code_cell_parses(self):
        import ast

        for index, cell in enumerate(self.notebook["cells"]):
            body = "".join(cell.get("source", []))
            if cell["cell_type"] == "code":
                self.assertFalse(
                    body.lstrip().startswith("#"),
                    f"cell {index} holds markdown in a code cell",
                )
                ast.parse(body)
            else:
                self.assertTrue(
                    body.lstrip().startswith("#"),
                    f"cell {index} is a markdown cell without a heading",
                )

    def test_six_steps_appear_once_each_in_order(self):
        headings = [
            "".join(cell["source"]).splitlines()[0]
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
            and "".join(cell["source"]).startswith("## ")
        ]
        self.assertEqual([head.split(".")[0] for head in headings], [f"## {n}" for n in range(1, 7)])

    def test_image_is_shown_before_inference(self):
        display = self.source.index("imshow(original)")
        preview = self.source.index("dequantized_preview(")
        inference = self.source.index("started = time.monotonic()")
        self.assertLess(display, inference)
        self.assertLess(preview, inference)

    def test_prediction_is_decoded_and_judged(self):
        self.assertIn("decode_logits(", self.source)
        self.assertIn("load_class_names(", self.source)
        self.assertIn("CORRECT", self.source)
        self.assertIn("INCORRECT", self.source)

    def test_runs_on_the_physical_overlay_only(self):
        self.assertIn("load_pynq_runtime(", self.source)
        self.assertIn("NPUModelRuntime(", self.source)
        self.assertIn("physical_jobs", self.source)
        self.assertIn("(8, 8, 256)", self.source)
        self.assertNotIn("HostMatrixBackend", self.source)

    def test_offers_the_gallery_and_an_upload(self):
        self.assertIn("gallery-source.json", self.source)
        self.assertIn("FileUpload(", self.source)
        self.assertIn("Dropdown(", self.source)

    def test_claims_no_acceptance_evidence(self):
        self.assertNotIn("human_approves", self.source)
        self.assertNotIn("notebook-evidence", self.source)
        self.assertNotIn("PASS [physical-pynq-z1]", self.source)


class RunbookTest(unittest.TestCase):
    def test_readme_orders_the_demo_preparation_before_deployment(self):
        readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
        commands = (
            "scripts/verify.ps1",
            "scripts/download_demo_assets.py",
            "scripts/prepare_demo_image.py",
            "build_overlay.tcl",
            "deploy_release.ps1",
        )
        positions = [readme.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("--expected-class", readme)
        self.assertIn("not ImageNet", readme)


if __name__ == "__main__":
    unittest.main()
