"""Verify and execute a standalone ResNet-18 package on PYNQ-Z1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
for import_root in (PACKAGE_ROOT, PACKAGE_ROOT.parents[1]):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES
from src.model.resnet18 import load_acceptance_bundle, validate_resnet18_topology
from src.runtime import NPUModelRuntime, load_model_package, load_pynq_runtime
from src.runtime.acceptance import run_resnet18_acceptance
from src.runtime.verify_overlay import verify_artifacts


PASS_MARKER = "PASS: Phase 2B ResNet-18 board acceptance"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class BoardAcceptanceError(RuntimeError):
    """Standalone package or physical acceptance did not pass."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BoardAcceptanceError(f"duplicate package JSON key {key!r}")
        result[key] = value
    return result


def _manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        data = path.read_bytes()
        value = json.loads(data, object_pairs_hook=_reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BoardAcceptanceError(f"package manifest is invalid: {error}") from error
    canonical = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if data != canonical:
        raise BoardAcceptanceError("package manifest is not canonical")
    if not isinstance(value, dict) or set(value) != {
        "files", "format", "magic", "release_tag", "source_commit",
        "target_part", "vivado_gates",
    }:
        raise BoardAcceptanceError("package manifest fields differ from the contract")
    if value["magic"] != "NPU_RESNET18_PACKAGE" or value["format"] != {
        "major": 1, "minor": 0,
    }:
        raise BoardAcceptanceError("package manifest format is unsupported")
    return value, hashlib.sha256(data).hexdigest()


def verify_package_tree(
    package_root: Path,
    *,
    archive_path: Path,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    """Verify every packaged byte before overlay programming or model execution."""

    package_root = package_root.resolve()
    archive_path = archive_path.resolve()
    expected = expected_archive_sha256.strip().lower()
    if SHA256_PATTERN.fullmatch(expected) is None:
        raise BoardAcceptanceError("expected archive SHA-256 is invalid")
    archive_digest = _sha256(archive_path)
    if archive_digest != expected:
        raise BoardAcceptanceError("package archive digest mismatch")
    manifest, manifest_digest = _manifest(package_root / "package.manifest.json")
    records = manifest["files"]
    if not isinstance(records, list):
        raise BoardAcceptanceError("package file records must be a list")
    expected_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"bytes", "path", "sha256"}:
            raise BoardAcceptanceError("package file record is malformed")
        relative = record["path"]
        if not isinstance(relative, str):
            raise BoardAcceptanceError("package file path must be a string")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise BoardAcceptanceError(f"unsafe package path: {relative!r}")
        if relative in expected_paths:
            raise BoardAcceptanceError("package manifest has duplicate paths")
        expected_paths.add(relative)
        path = package_root.joinpath(*pure.parts)
        if not path.is_file():
            raise BoardAcceptanceError(f"package file is missing: {relative}")
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise BoardAcceptanceError(f"package file differs: {relative}")
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != "package.manifest.json"
    }
    if actual_paths != expected_paths:
        raise BoardAcceptanceError("package tree contains missing or unexpected files")

    gates = manifest["vivado_gates"]
    if not isinstance(gates, dict) or not (
        gates.get("synthesis_complete") is True
        and gates.get("implementation_complete") is True
        and gates.get("setup_failing_paths") == 0
        and gates.get("drc_errors") == 0
        and gates.get("source_commit") == manifest["source_commit"]
        and gates.get("target_part") == manifest["target_part"]
        and isinstance(gates.get("wns"), (int, float))
        and gates["wns"] >= 0
    ):
        raise BoardAcceptanceError("trusted Vivado gates are incomplete")
    overlay = verify_artifacts(package_root / "artifacts")
    if (
        overlay.get("source_commit") != manifest["source_commit"]
        or overlay.get("target_part") != manifest["target_part"]
    ):
        raise BoardAcceptanceError("overlay provenance differs from package")
    reports, reports_digest = _manifest_like_reports(
        package_root / "reports" / "reports.manifest.json"
    )
    if reports.get("vivado_gates") != gates:
        raise BoardAcceptanceError("report manifest gates differ from package")
    descriptor_path = package_root / "acceptance" / "acceptance.json"
    preliminary = load_acceptance_bundle(descriptor_path)
    loaded = load_model_package(preliminary.model_manifest_path)
    bundle = load_acceptance_bundle(descriptor_path, graph=loaded.graph)
    validate_resnet18_topology(loaded.graph)
    return {
        "archive_sha256": archive_digest,
        "bundle": bundle,
        "manifest": manifest,
        "manifest_sha256": manifest_digest,
        "model": loaded,
        "overlay": overlay,
        "reports": reports,
        "reports_manifest_sha256": reports_digest,
    }


def _manifest_like_reports(path: Path) -> tuple[dict[str, Any], str]:
    try:
        data = path.read_bytes()
        value = json.loads(data, object_pairs_hook=_reject_duplicates)
        canonical = (
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise BoardAcceptanceError(f"report manifest is invalid: {error}") from error
    if data != canonical or not isinstance(value, dict) or set(value) != {
        "files", "vivado_gates",
    }:
        raise BoardAcceptanceError("report manifest is not canonical")
    return value, hashlib.sha256(data).hexdigest()


def execute_board_acceptance(
    verified: dict[str, Any],
    physical_runtime: Any,
    *,
    evidence_path: Path,
) -> dict[str, Any]:
    if (
        int(physical_runtime.abi_major) != REQUIRED_ABI_MAJOR
        or int(physical_runtime.capabilities) & REQUIRED_CAPABILITIES
        != REQUIRED_CAPABILITIES
    ):
        raise BoardAcceptanceError("runtime ABI or capabilities are incompatible")
    limits = [
        int(physical_runtime.max_m),
        int(physical_runtime.max_n),
        int(physical_runtime.max_k),
    ]
    if limits != [2, 2, 256]:
        raise BoardAcceptanceError(f"physical limits are incompatible: {limits}")
    runtime = NPUModelRuntime(physical_runtime, verified["model"])
    bundle = verified["bundle"]

    def recovery_probe() -> None:
        # Force a real accelerator timeout after valid DMA/MMIO preflight. A
        # one-cycle budget cannot complete even the smallest supported 2x2x1
        # job, so NPURuntime must execute its physical recovery path before the
        # acceptance runner submits the changed post-failure sample.
        probe_a = np.ones((2, 1), dtype=np.int8)
        probe_b = np.ones((1, 2), dtype=np.int8)
        physical_runtime.run(
            probe_a,
            probe_b,
            hardware_timeout_cycles=1,
            software_timeout=2.0,
        )

    overlay = verified["overlay"]
    provenance = {
        "archive_sha256": verified["archive_sha256"],
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "overlay": {
            "bit_sha256": overlay["bit"]["sha256"],
            "hwh_sha256": overlay["hwh"]["sha256"],
            "target_part": overlay["target_part"],
            "vivado_version": overlay.get("vivado_version", "unknown"),
        },
        "package_manifest_sha256": verified["manifest_sha256"],
        "reports": verified["reports"]["files"],
        "reports_manifest_sha256": verified["reports_manifest_sha256"],
        "physical": {
            "abi_major": int(physical_runtime.abi_major),
            "capabilities": int(physical_runtime.capabilities),
            "limits": limits,
        },
        "recovery_probe": {
            "hardware_timeout_cycles": 1,
            "kind": "physical-accelerator-timeout",
        },
        "release_tag": verified["manifest"]["release_tag"],
        "source_commit": verified["manifest"]["source_commit"],
        "vivado_gates": verified["manifest"]["vivado_gates"],
    }
    evidence = run_resnet18_acceptance(
        bundle,
        runtime,
        evidence_path=evidence_path,
        mode="board",
        recovery_probe=recovery_probe,
        provenance=provenance,
    )
    return dict(evidence)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    try:
        verified = verify_package_tree(
            arguments.package_root,
            archive_path=arguments.archive,
            expected_archive_sha256=arguments.archive_sha256,
        )
        if arguments.verify_only:
            print("PASS: standalone ResNet-18 package verification")
            return 0
        if arguments.evidence is None:
            raise BoardAcceptanceError("--evidence is required for execution")
        physical = load_pynq_runtime(
            arguments.package_root / "artifacts" / "npu_matrix.bit"
        )
        execute_board_acceptance(
            verified,
            physical,
            evidence_path=arguments.evidence,
        )
    except Exception as error:
        print(f"Phase 2B board acceptance failed: {error}", file=sys.stderr)
        return 1
    print(PASS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
