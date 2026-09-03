from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "resnet18"


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

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self.final_url

    def read(self, size: int = -1):
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = load_module(
            "resnet18_download_model",
            EXAMPLE_ROOT / "scripts" / "download_model.py",
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = b"pinned-real-resnet18-state"
        self.url = "https://download.pytorch.org/models/resnet18-test.pth"
        self.metadata = self.root / "source.json"
        self.metadata.write_text(
            json.dumps(
                {
                    "architecture": "resnet18",
                    "bytes": len(self.payload),
                    "filename": "resnet18-test.pth",
                    "format": {"major": 1, "minor": 0},
                    "license": {
                        "dataset_terms": "ImageNet-1K terms apply",
                        "source_project_spdx": "BSD-3-Clause",
                    },
                    "magic": "NPU_PRETRAINED_MODEL_SOURCE",
                    "provider": "torchvision",
                    "revision": "IMAGENET1K_V1",
                    "sha256": hashlib.sha256(self.payload).hexdigest(),
                    "url": self.url,
                },
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n",
            encoding="utf-8",
        )

    def opener(self, payload=None, final_url=None):
        expected_payload = self.payload if payload is None else payload
        expected_url = self.url if final_url is None else final_url

        def open_url(request, timeout):
            self.assertEqual(request.full_url, self.url)
            self.assertGreater(timeout, 0)
            return FakeResponse(expected_payload, expected_url)

        return open_url

    def test_atomic_verified_download_and_refuses_overwrite(self):
        destination = self.root / "model"
        result = self.downloader.download_model(
            self.metadata, destination, opener=self.opener()
        )
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertFalse((destination / ".resnet18-test.pth.tmp").exists())
        with self.assertRaisesRegex(self.downloader.ModelDownloadError, "exists"):
            self.downloader.download_model(
                self.metadata, destination, opener=self.opener()
            )

    def test_digest_and_redirect_host_mismatch_publish_nothing(self):
        for payload, final_url, message in (
            (b"substituted", self.url, "size|digest"),
            (self.payload, "https://example.com/model.pth", "host"),
        ):
            with self.subTest(message=message):
                destination = self.root / f"model-{message.replace('|', '-')}"
                with self.assertRaisesRegex(
                    self.downloader.ModelDownloadError, message
                ):
                    self.downloader.download_model(
                        self.metadata,
                        destination,
                        opener=self.opener(payload, final_url),
                    )
                self.assertFalse((destination / "resnet18-test.pth").exists())

    def test_metadata_is_canonical_and_complete(self):
        source = self.downloader.load_source_metadata(self.metadata)
        self.assertEqual(source.architecture, "resnet18")
        noncanonical = self.root / "noncanonical.json"
        noncanonical.write_text(
            json.dumps(json.loads(self.metadata.read_text()), indent=2),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            self.downloader.ModelDownloadError, "canonical"
        ):
            self.downloader.load_source_metadata(noncanonical)

    def test_model_workspace_is_documented_and_ignored(self):
        filetree = (REPOSITORY_ROOT / "docs" / "rules" / "filetree.md").read_text(
            encoding="utf-8"
        )
        ignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("examples/<example>/model/", filetree)
        self.assertIn("examples/*/model/*", ignore)
        self.assertTrue((EXAMPLE_ROOT / "model" / ".gitkeep").is_file())

    def test_human_runbook_and_output_free_notebook_follow_required_order(self):
        readme = (EXAMPLE_ROOT / "README.md").read_text(encoding="utf-8")
        commands = (
            "scripts/download_model.py",
            "scripts/convert_model.py",
            "scripts/verify.ps1",
            "build_overlay.tcl",
            "package_example.py",
            "run_on_board.py",
        )
        positions = [readme.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("PASS [real-model-host]", readme)
        self.assertIn("PASS [physical-pynq-z1]", readme)
        notebook = json.loads(
            (EXAMPLE_ROOT / "resnet18.ipynb").read_text(encoding="utf-8")
        )
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell["outputs"] == [] for cell in code_cells))
        self.assertTrue(
            all(cell["execution_count"] is None for cell in code_cells)
        )

    def test_only_physical_runner_owns_physical_pass_marker(self):
        verifier = (EXAMPLE_ROOT / "scripts" / "verify_model.py").read_text(
            encoding="utf-8"
        )
        board_runner = (EXAMPLE_ROOT / "run_on_board.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("PASS [physical-pynq-z1]", verifier)
        self.assertIn("PASS [physical-pynq-z1]", board_runner)
        self.assertIn("isinstance(physical, NPURuntime)", board_runner)
        self.assertIn("--expected-source-commit", board_runner)
        self.assertIn("physical-pynq-z1-development", board_runner)
        self.assertIn("--allow-source-mismatch", board_runner)
        deployment = (EXAMPLE_ROOT / "deploy_release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("source /etc/profile.d/pynq_venv.sh", deployment)
        self.assertIn("source /etc/profile.d/xrt_setup.sh", deployment)
        self.assertIn("Invoke-CheckedCommand -Command 'scp'", deployment)
        self.assertIn("overlay from this commit", deployment)
        self.assertIn("AllowArtifactCommitMismatch", deployment)


class ModelPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packager = load_module(
            "resnet18_package_example",
            EXAMPLE_ROOT / "package_example.py",
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.metadata_path = self.root / "model-source.json"
        self.files = {
            "resnet18-f37072fd.pth": b"checkpoint",
            "resnet18.npu.json": b"manifest\n",
            "resnet18.npu.bin": b"payload",
            "resnet18.validation.npy": b"input",
        }
        for name, data in self.files.items():
            (self.model_dir / name).write_bytes(data)
        metadata = {
            "architecture": "resnet18",
            "bytes": len(self.files["resnet18-f37072fd.pth"]),
            "filename": "resnet18-f37072fd.pth",
            "format": {"major": 1, "minor": 0},
            "license": {"source_project_spdx": "BSD-3-Clause"},
            "magic": "NPU_PRETRAINED_MODEL_SOURCE",
            "provider": "torchvision",
            "revision": "IMAGENET1K_V1",
            "sha256": hashlib.sha256(
                self.files["resnet18-f37072fd.pth"]
            ).hexdigest(),
            "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        }
        self._write_json(self.metadata_path, metadata)
        conversion = {
            "checkpoint": {
                "sha256": metadata["sha256"],
            },
            "evidence_type": "real-model-host",
            "input": {
                "sha256": self._digest("resnet18.validation.npy"),
            },
            "magic": "NPU_RESNET18_CONVERSION",
            "model": {
                "manifest_sha256": self._digest("resnet18.npu.json"),
                "payload_sha256": self._digest("resnet18.npu.bin"),
            },
        }
        conversion_path = self.model_dir / "resnet18.conversion.json"
        self._write_json(conversion_path, conversion)
        acceptance = {
            "conversion": {"sha256": self._path_digest(conversion_path)},
            "evidence_type": "real-model-host",
            "input": {"sha256": self._digest("resnet18.validation.npy")},
            "magic": "NPU_RESNET18_ACCEPTANCE",
            "model": {
                "manifest_sha256": self._digest("resnet18.npu.json"),
                "payload_sha256": self._digest("resnet18.npu.bin"),
            },
            "result": "pass",
            "runtime": {"physical_board": False},
            "source": {"sha256": metadata["sha256"]},
        }
        self._write_json(self.model_dir / "acceptance.json", acceptance)

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_bytes(
            (
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )

    @staticmethod
    def _path_digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _digest(self, name: str) -> str:
        return self._path_digest(self.model_dir / name)

    def _build(self, name: str):
        output = self.root / name
        self.packager.build_archive(
            model_dir=self.model_dir,
            source_metadata_path=self.metadata_path,
            output_archive=output,
        )
        return output

    def test_validated_archives_are_byte_identical_and_exclude_checkpoint(self):
        first = self._build("first.zip")
        second = self._build("second.zip")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            names = set(archive.namelist())
        self.assertIn("model/acceptance.json", names)
        self.assertIn("model/resnet18.npu.bin", names)
        self.assertNotIn("model/resnet18-f37072fd.pth", names)

    def test_missing_or_substituted_asset_publishes_no_archive(self):
        (self.model_dir / "resnet18.validation.npy").unlink()
        missing_output = self.root / "missing.zip"
        with self.assertRaisesRegex(
            self.packager.ResNet18PackageError, "incomplete"
        ):
            self.packager.build_archive(
                model_dir=self.model_dir,
                source_metadata_path=self.metadata_path,
                output_archive=missing_output,
            )
        self.assertFalse(missing_output.exists())

        (self.model_dir / "resnet18.validation.npy").write_bytes(b"substituted")
        substituted_output = self.root / "substituted.zip"
        with self.assertRaisesRegex(
            self.packager.ResNet18PackageError, "substituted"
        ):
            self.packager.build_archive(
                model_dir=self.model_dir,
                source_metadata_path=self.metadata_path,
                output_archive=substituted_output,
            )
        self.assertFalse(substituted_output.exists())


class BoardSourceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module(
            "resnet18_run_on_board",
            EXAMPLE_ROOT / "run_on_board.py",
        )

    def test_mismatch_requires_explicit_development_mode(self):
        artifact_commit = "a" * 40
        deployed_commit = "b" * 40
        overlay = {"source_commit": artifact_commit}
        with self.assertRaisesRegex(RuntimeError, "deployed source"):
            self.runner._source_binding(
                overlay, artifact_commit, deployed_commit, False
            )
        expected, deployed, mismatch = self.runner._source_binding(
            overlay, artifact_commit, deployed_commit, True
        )
        self.assertEqual(expected, artifact_commit)
        self.assertEqual(deployed, deployed_commit)
        self.assertTrue(mismatch)

    def test_matching_source_remains_trusted_mode(self):
        commit = "c" * 40
        expected, deployed, mismatch = self.runner._source_binding(
            {"source_commit": commit}, commit, commit, False
        )
        self.assertEqual(expected, commit)
        self.assertEqual(deployed, commit)
        self.assertFalse(mismatch)


if __name__ == "__main__":
    unittest.main()
