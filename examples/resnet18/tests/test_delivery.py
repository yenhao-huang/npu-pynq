from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np

from src.model.numeric import requantize_int32_to_int8


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "resnet18"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.export.resnet import export_model
from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES
from src.runtime.model import NPUModelRuntime, load_model_package
from src.runtime.verify_overlay import write_manifest
from src.test.tests.resnet18_fixture import (
    make_reduced_resnet18_graph,
    write_canonical_json,
    write_deterministic_npz,
)
from src.test.tests.test_verify_overlay import HWH


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MatrixFake:
    abi_major = REQUIRED_ABI_MAJOR
    capabilities = REQUIRED_CAPABILITIES
    max_m = 2
    max_n = 2
    max_k = 256

    def __init__(self, cycles=5):
        self.cycles = cycles

    def run_slices(self, a_tiles, b_tiles, *, bias, multipliers_q31, shifts,
                   output_zero_point, **_timeouts):
        if _timeouts.get("hardware_timeout_cycles") == 1:
            raise TimeoutError("injected physical accelerator timeout")
        self.last_metrics = SimpleNamespace(cycles=self.cycles * len(a_tiles))
        accumulator = sum(
            np.asarray(a, dtype=np.int64) @ np.asarray(b, dtype=np.int64)
            for a, b in zip(a_tiles, b_tiles)
        ) + bias.astype(np.int64)
        output = np.empty(accumulator.shape, dtype=np.int8)
        for row in range(output.shape[0]):
            for column in range(output.shape[1]):
                output[row, column] = requantize_int32_to_int8(
                    int(accumulator[row, column]), int(multipliers_q31[column]),
                    int(shifts[column]), output_zero_point
                )
        return output


class ResNet18DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.reports = self.root / "reports"
        self.bundle_root = self.root / "bundle"
        self.artifacts.mkdir()
        self.reports.mkdir()
        self.bundle_root.mkdir()
        (self.artifacts / "npu_matrix.bit").write_bytes(b"trusted-bit")
        (self.artifacts / "npu_matrix.hwh").write_text(HWH, encoding="utf-8")
        self.commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
        write_manifest(
            self.artifacts,
            source_commit=self.commit,
            vivado_version="2026.1",
        )
        self._write_reports()
        self.graph = make_reduced_resnet18_graph("delivery")
        package = export_model(self.graph, self.bundle_root / "model")
        model = load_model_package(package.manifest_path)
        inputs = np.stack(
            (
                np.full((1, 8, 8, 1), -1, dtype=np.int8),
                np.full((1, 8, 8, 1), 2, dtype=np.int8),
            )
        )
        runtime = NPUModelRuntime(MatrixFake(), model)
        outputs = np.stack([
            runtime.run({self.graph.inputs[0]: source}).outputs[self.graph.outputs[0]]
            for source in inputs
        ])
        corpus = self.bundle_root / "corpus.npz"
        write_deterministic_npz(
            corpus,
            {
                "expected_outputs": outputs,
                "inputs": inputs,
                "labels": np.argmax(outputs.reshape(2, -1), axis=1).astype(np.int64),
                "sample_ids": np.array(["a", "b"]),
            },
        )
        self.descriptor = self.bundle_root / "acceptance.json"
        write_canonical_json(
            self.descriptor,
            {
                "assets": {
                    "corpus": self.asset(corpus),
                    "model_manifest": self.asset(package.manifest_path),
                    "model_payload": self.asset(package.payload_path),
                },
                "capture_tensors": [],
                "class_count": 2,
                "format": {"major": 1, "minor": 0},
                "magic": "NPU_RESNET18_ACCEPTANCE",
                "reference": {
                    "framework": "fixture",
                    "model_id": "reduced-resnet18",
                    "preprocessing_id": "signed-int8-v1",
                    "version": "1",
                },
                "sample_count": 2,
                "thresholds": {
                    "exact_output_min": 1.0,
                    "require_cycles": True,
                    "top1_min": 1.0,
                },
            },
        )
        self.packager = load_module(
            "resnet18_package_example", EXAMPLE_ROOT / "package_example.py"
        )
        self.board = load_module(
            "resnet18_run_on_board", EXAMPLE_ROOT / "run_on_board.py"
        )

    def _write_reports(self):
        evidence = (
            "vivado=2026.1\n"
            "part=xc7z020clg400-1\n"
            "wns=0.250\n"
            "setup_failing_paths=0\n"
            "drc_errors=0\n"
            f"bit={self.artifacts / 'npu_matrix.bit'}\n"
            f"hwh={self.artifacts / 'npu_matrix.hwh'}\n"
            f"source_commit={self.commit}\n"
        )
        (self.reports / "build_evidence.txt").write_text(evidence, encoding="utf-8")
        for name in self.report_names()[1:]:
            (self.reports / name).write_text(f"trusted {name}\n", encoding="utf-8")

    @staticmethod
    def report_names():
        return (
            "build_evidence.txt", "drc_routed.rpt", "route_status.rpt",
            "timing_summary_routed.rpt", "utilization_impl.rpt",
            "utilization_synth.rpt",
        )

    @staticmethod
    def asset(path):
        return {
            "bytes": path.stat().st_size,
            "filename": path.name,
            "sha256": digest(path),
        }

    def build(self, path: Path):
        return self.packager.build_archive(
            repository_root=REPOSITORY_ROOT,
            artifact_dir=self.artifacts,
            report_dir=self.reports,
            descriptor_path=self.descriptor,
            output_archive=path,
            release_tag="v0.2.0",
            source_commit=self.commit,
        )

    def extract(self, archive: Path) -> Path:
        target = self.root / f"extract-{archive.stem}"
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(target)
        return target

    def test_archive_is_reproducible_allowlisted_and_path_free(self):
        first = self.root / "first.zip"
        second = self.root / "second.zip"
        manifest = self.build(first)
        self.build(second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            names = archive.namelist()
            self.assertEqual(names, sorted(names))
            self.assertIn("resnet18.ipynb", names)
            self.assertTrue(all(not name.startswith(("/", "\\")) for name in names))
            self.assertNotIn(str(REPOSITORY_ROOT).encode(), first.read_bytes())
        self.assertEqual(manifest["vivado_gates"]["drc_errors"], 0)
        report_records = json.loads(
            zipfile.ZipFile(first).read("reports/reports.manifest.json")
        )["files"]
        self.assertEqual(
            [record["path"] for record in report_records],
            sorted(self.report_names()),
        )

    def test_notebook_is_output_free_and_uses_public_runtime(self):
        notebook_path = EXAMPLE_ROOT / "resnet18.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertGreaterEqual(len(code_cells), 4)
        for cell in code_cells:
            self.assertEqual(cell.get("outputs"), [])
            self.assertIsNone(cell.get("execution_count"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "verify_artifacts",
            "load_acceptance_bundle",
            "load_model_package",
            "load_pynq_runtime",
            "NPUModelRuntime",
            "run_resnet18_acceptance",
            "physical_cycles",
        ):
            self.assertIn(required, source)
        for forbidden in (".mmio", "sendchannel", "recvchannel", "allocate("):
            self.assertNotIn(forbidden, source)

    def test_package_verification_precedes_board_execution(self):
        archive = self.root / "package.zip"
        self.build(archive)
        extracted = self.extract(archive)
        verified = self.board.verify_package_tree(
            extracted,
            archive_path=archive,
            expected_archive_sha256=digest(archive),
        )
        evidence = self.board.execute_board_acceptance(
            verified,
            MatrixFake(),
            evidence_path=extracted / "board-evidence.json",
        )
        self.assertEqual(evidence["evidence_type"], "board-acceptance")
        self.assertEqual(evidence["provenance"]["physical"]["limits"], [2, 2, 256])
        self.assertEqual(
            evidence["provenance"]["recovery_probe"]["kind"],
            "physical-accelerator-timeout",
        )
        self.assertEqual(len(evidence["provenance"]["reports"]), 6)
        self.assertGreater(evidence["performance"]["physical_cycles"], 0)

        unexpected = extracted / "unexpected.py"
        unexpected.write_text("pass\n", encoding="utf-8")
        with self.assertRaisesRegex(self.board.BoardAcceptanceError, "unexpected"):
            self.board.verify_package_tree(
                extracted,
                archive_path=archive,
                expected_archive_sha256=digest(archive),
            )

    def test_archive_or_report_gate_failure_publishes_nothing(self):
        archive = self.root / "package.zip"
        self.build(archive)
        extracted = self.extract(archive)
        with self.assertRaisesRegex(self.board.BoardAcceptanceError, "archive digest"):
            self.board.verify_package_tree(
                extracted,
                archive_path=archive,
                expected_archive_sha256="0" * 64,
            )
        (self.reports / "build_evidence.txt").write_text(
            (self.reports / "build_evidence.txt").read_text(encoding="utf-8").replace(
                "drc_errors=0", "drc_errors=1"
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(self.packager.ResNet18PackageError, "gates"):
            self.build(self.root / "failed.zip")
        self.assertFalse((self.root / "failed.zip").exists())

    def test_deployment_dry_run_has_no_network_effect(self):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        self.assertIsNotNone(powershell)
        archive = self.root / "package.zip"
        self.build(archive)
        result = subprocess.run(
            [
                str(powershell), "-NoProfile", "-File",
                str(EXAMPLE_ROOT / "deploy_release.ps1"),
                "-PackageArchive", str(archive),
                "-ReleaseTag", "v0.2.0",
                "-DeploymentId", "dry-run",
                "-EvidencePath", str(self.root / "evidence.json"),
                "-DryRun",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no network command was executed", result.stdout)
        self.assertFalse((self.root / "evidence.json").exists())

    def test_deployment_makes_root_evidence_readable_before_promotion(self):
        script = (EXAMPLE_ROOT / "deploy_release.ps1").read_text(encoding="utf-8")
        runner = script.index("--evidence board-evidence.json")
        readable = script.index("chmod 0644 board-evidence.json")
        promotion = script.index("mv '$remoteStaging' '$remoteDeployment'")
        self.assertLess(runner, readable)
        self.assertLess(readable, promotion)


if __name__ == "__main__":
    unittest.main()
