"""Run digest-bound ResNet-18 acceptance on a physical PYNQ-Z1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
# In a deployed standalone package this file sits at the package root; in the
# repository it sits under examples/resnet18/.
REPOSITORY_ROOT = (
    MODULE_DIR
    if (MODULE_DIR / "package.manifest.json").is_file()
    else MODULE_DIR.parents[2]
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from examples.resnet18.package_example import (
        DELIVERY_MAGIC,
        validate_workspace,
    )
except ModuleNotFoundError:  # deployed standalone package layout
    from package_example import DELIVERY_MAGIC, validate_workspace
from src.model.resnet18 import load_acceptance_bundle
from src.runtime import (
    NPURuntime,
    NPUModelRuntime,
    load_model_package,
    load_pynq_runtime,
    run_resnet18_acceptance,
)
from src.runtime.verify_overlay import verify_artifacts


PASS_MARKER = "PASS [physical-pynq-z1]: real ResNet-18 board acceptance"
DEVELOPMENT_PASS_MARKER = (
    "PASS [physical-pynq-z1-development]: real ResNet-18 board execution"
)
PACKAGE_PASS_MARKER = (
    "PASS [physical-pynq-z1]: standalone ResNet-18 release package accepted"
)
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_new(path: Path, value: object) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ValueError("board evidence must be a new file in an existing directory")
    data = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_binding(
    overlay: dict[str, object],
    expected_source_commit: str,
    deployed_source_commit: str,
    allow_source_mismatch: bool,
) -> tuple[str, str, bool]:
    expected = expected_source_commit.strip().lower()
    deployed = deployed_source_commit.strip().lower()
    if COMMIT_PATTERN.fullmatch(expected) is None:
        raise ValueError("expected source commit must be a full Git object ID")
    if COMMIT_PATTERN.fullmatch(deployed) is None:
        raise ValueError("deployed source commit must be a full Git object ID")
    if str(overlay.get("source_commit", "")).lower() != expected:
        raise RuntimeError("overlay source commit differs from its expected commit")
    mismatch = expected != deployed
    if mismatch and not allow_source_mismatch:
        raise RuntimeError("overlay source commit differs from deployed source")
    return expected, deployed, mismatch


def run_board(
    *,
    model_dir: Path,
    source_metadata_path: Path,
    artifact_dir: Path,
    expected_source_commit: str,
    deployed_source_commit: str,
    allow_source_mismatch: bool,
    evidence_path: Path,
    software_timeout: float,
    require_checkpoint: bool = True,
) -> dict[str, object]:
    validate_workspace(
        model_dir, source_metadata_path, require_checkpoint=require_checkpoint
    )
    model_dir = model_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    overlay = verify_artifacts(artifact_dir)
    expected_commit, deployed_commit, source_mismatch = _source_binding(
        overlay,
        expected_source_commit,
        deployed_source_commit,
        allow_source_mismatch,
    )
    physical = load_pynq_runtime(artifact_dir / "npu_matrix.bit")
    if not isinstance(physical, NPURuntime):
        raise RuntimeError("physical evidence requires the public NPURuntime")
    model = load_model_package(model_dir / "resnet18.npu.json")
    validation_input = np.load(
        model_dir / "resnet18.validation.npy", allow_pickle=False
    )
    acceptance = json.loads(
        (model_dir / "acceptance.json").read_text(encoding="utf-8")
    )
    result = NPUModelRuntime(physical, model).run(
        {"input": validation_input}, software_timeout=software_timeout
    )
    for name in model.graph.outputs:
        expected = acceptance["captures"][name]["sha256"]
        if _array_sha256(result.outputs[name]) != expected:
            raise RuntimeError(f"physical capture differs: {name}")
    evidence: dict[str, object] = {
        "captures": {
            name: {
                "sha256": _array_sha256(result.outputs[name]),
                "shape": list(result.outputs[name].shape),
            }
            for name in model.graph.outputs
        },
        "evidence_type": (
            "physical-pynq-z1-development"
            if source_mismatch
            else "physical-pynq-z1"
        ),
        "format": {"major": 1, "minor": 0},
        "host_acceptance_sha256": _sha256(model_dir / "acceptance.json"),
        "magic": "NPU_RESNET18_BOARD_ACCEPTANCE",
        "model_manifest_sha256": _sha256(model_dir / "resnet18.npu.json"),
        "overlay": {
            "bit_sha256": overlay["bit"]["sha256"],
            "deployed_source_commit": deployed_commit,
            "hwh_sha256": overlay["hwh"]["sha256"],
            "source_commit": overlay["source_commit"],
            "source_mismatch_allowed": source_mismatch,
            "target_part": overlay["target_part"],
        },
        "result": "pass",
        "runtime": {
            "abi_major": physical.abi_major,
            "capabilities": physical.capabilities,
            "mac_count": result.metrics.mac_count,
            "physical_jobs": result.metrics.physical_jobs,
            "physical_limits": [physical.max_m, physical.max_n, physical.max_k],
        },
    }
    _write_new(evidence_path.resolve(), evidence)
    return evidence


# --- Standalone package acceptance -----------------------------------------

GENERATED_PACKAGE_FILES = frozenset(
    {
        "board-evidence.json",
        "deployment.json",
        "image-acceptance.json",
    }
)
RECOVERY_PROBE_KIND = "physical-accelerator-timeout"


class BoardAcceptanceError(RuntimeError):
    """The deployed package or the board run failed a gate before evidence."""


@dataclass(frozen=True)
class VerifiedPackage:
    """One deployed package whose provenance was proved before execution."""

    root: Path
    manifest: dict[str, object]
    descriptor_path: Path | None
    artifact_dir: Path
    reports: tuple[dict[str, object], ...]
    archive_sha256: str


def _digest_or_fail(path: Path, label: str) -> str:
    if not path.is_file():
        raise BoardAcceptanceError(f"{label} is missing: {path.name}")
    return _sha256(path)


def _package_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BoardAcceptanceError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise BoardAcceptanceError(f"{label} must contain an object")
    return value


def _relative_entries(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def verify_package_tree(
    package_root: str | Path,
    *,
    archive_path: str | Path,
    expected_archive_sha256: str,
) -> VerifiedPackage:
    """Prove the deployed tree is exactly the published archive, or fail."""

    root = Path(package_root).resolve()
    archive = Path(archive_path).resolve()
    expected = str(expected_archive_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise BoardAcceptanceError("expected archive digest is malformed")
    if not archive.is_file():
        raise BoardAcceptanceError(f"release archive is missing: {archive.name}")
    if _sha256(archive) != expected:
        raise BoardAcceptanceError(
            "archive digest differs from the published release asset"
        )

    manifest = _package_json(root / "package.manifest.json", "package manifest")
    if manifest.get("magic") != DELIVERY_MAGIC:
        raise BoardAcceptanceError("package manifest magic is invalid")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise BoardAcceptanceError("package manifest lists no files")

    declared: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise BoardAcceptanceError("package manifest record is malformed")
        name = record.get("path")
        if not isinstance(name, str) or not name:
            raise BoardAcceptanceError("package manifest record has no path")
        pure = Path(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise BoardAcceptanceError(f"package entry escapes the package: {name}")
        target = root / name
        if _digest_or_fail(target, "package file") != record.get("sha256"):
            raise BoardAcceptanceError(f"package file was modified: {name}")
        if target.stat().st_size != record.get("bytes"):
            raise BoardAcceptanceError(f"package file size differs: {name}")
        declared.add(name)

    observed = _relative_entries(root) - {"package.manifest.json"}
    unexpected = sorted(observed - declared - GENERATED_PACKAGE_FILES)
    if unexpected:
        raise BoardAcceptanceError(
            f"unexpected file in the deployed package: {unexpected[0]}"
        )

    artifact_dir = root / "artifacts"
    overlay = verify_artifacts(artifact_dir)
    declared_overlay = manifest.get("overlay")
    if not isinstance(declared_overlay, dict):
        raise BoardAcceptanceError("package manifest has no overlay record")
    if (
        overlay["bit"]["sha256"] != declared_overlay.get("bit_sha256")
        or overlay["hwh"]["sha256"] != declared_overlay.get("hwh_sha256")
        or str(overlay.get("source_commit", "")).lower()
        != str(declared_overlay.get("source_commit", "")).lower()
    ):
        raise BoardAcceptanceError("deployed overlay differs from the package manifest")

    reports = _package_json(
        root / "reports" / "reports.manifest.json", "reports manifest"
    ).get("files")
    if not isinstance(reports, list) or not reports:
        raise BoardAcceptanceError("reports manifest lists no files")
    for record in reports:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise BoardAcceptanceError("reports manifest record is malformed")
        report = root / "reports" / str(record["path"])
        if _digest_or_fail(report, "build report") != record.get("sha256"):
            raise BoardAcceptanceError(f"build report was modified: {record['path']}")

    # A package is accepted from an external bundle when it carries one, and
    # otherwise from its model workspace. It must carry one or the other.
    bundle = root / "acceptance" / "acceptance.json"
    descriptor_path = bundle if bundle.is_file() else None
    if descriptor_path is None and not (root / "model" / "acceptance.json").is_file():
        raise BoardAcceptanceError(
            "package carries neither an acceptance bundle nor a model workspace"
        )

    return VerifiedPackage(
        root=root,
        manifest=manifest,
        descriptor_path=descriptor_path,
        artifact_dir=artifact_dir,
        reports=tuple(reports),
        archive_sha256=expected,
    )


def execute_board_acceptance(
    verified: VerifiedPackage,
    matrix_runtime: object,
    *,
    evidence_path: str | Path,
    repeat_count: int = 2,
    software_timeout: float = 86400.0,
) -> dict[str, object]:
    """Run the acceptance corpus on the verified package and publish evidence."""

    if not isinstance(verified, VerifiedPackage):
        raise TypeError("verified must come from verify_package_tree")
    if verified.descriptor_path is None:
        raise BoardAcceptanceError("this package carries no acceptance bundle")
    bundle = load_acceptance_bundle(verified.descriptor_path)
    model = load_model_package(bundle.model_manifest_path)
    runtime = NPUModelRuntime(matrix_runtime, model)

    def recovery_probe() -> None:
        operand = np.zeros((1, 1), dtype=np.int8)
        matrix_runtime.run(
            operand,
            operand,
            hardware_timeout_cycles=1,
            software_timeout=1.0,
        )

    provenance = {
        "archive": {
            "release_tag": verified.manifest.get("release_tag"),
            "sha256": verified.archive_sha256,
            "source_commit": verified.manifest.get("source_commit"),
        },
        "overlay": verified.manifest.get("overlay"),
        "physical": {
            "abi_major": getattr(matrix_runtime, "abi_major", None),
            "capabilities": getattr(matrix_runtime, "capabilities", None),
            "limits": [
                matrix_runtime.max_m,
                matrix_runtime.max_n,
                matrix_runtime.max_k,
            ],
            "runtime_class": type(matrix_runtime).__name__,
        },
        "recovery_probe": {"kind": RECOVERY_PROBE_KIND},
        "reports": [dict(record) for record in verified.reports],
        "vivado_gates": verified.manifest.get("vivado_gates"),
    }
    try:
        evidence = run_resnet18_acceptance(
            bundle,
            runtime,
            evidence_path=evidence_path,
            mode="board",
            repeat_count=repeat_count,
            recovery_probe=recovery_probe,
            provenance=provenance,
        )
    except Exception as error:
        raise BoardAcceptanceError(f"board acceptance failed: {error}") from error
    del software_timeout
    return dict(evidence)


def _accept_package(arguments: argparse.Namespace) -> int:
    """Accept one extracted standalone package on the physical PYNQ-Z1."""

    if arguments.package_archive is None or arguments.archive_sha256 is None:
        print(
            "package acceptance requires --package-archive and --archive-sha256",
            file=sys.stderr,
        )
        return 1
    try:
        verified = verify_package_tree(
            arguments.package_root,
            archive_path=arguments.package_archive,
            expected_archive_sha256=arguments.archive_sha256,
        )
        if (
            arguments.release_tag is not None
            and verified.manifest.get("release_tag") != arguments.release_tag
        ):
            raise BoardAcceptanceError(
                "package manifest names another release tag"
            )
        if verified.descriptor_path is None:
            # The real release: accept the model workspace the package carries.
            # verify_package_tree has already proved every file digest.
            commit = str(verified.manifest.get("source_commit", ""))
            run_board(
                model_dir=verified.root / "model",
                source_metadata_path=verified.root / "model-source.json",
                artifact_dir=verified.artifact_dir,
                expected_source_commit=commit,
                deployed_source_commit=commit,
                allow_source_mismatch=False,
                evidence_path=arguments.evidence,
                software_timeout=arguments.software_timeout,
                require_checkpoint=False,
            )
        else:
            physical = load_pynq_runtime(verified.artifact_dir / "npu_matrix.bit")
            if not isinstance(physical, NPURuntime):
                raise BoardAcceptanceError(
                    "physical evidence requires the public NPURuntime"
                )
            execute_board_acceptance(
                verified,
                physical,
                evidence_path=arguments.evidence,
                software_timeout=arguments.software_timeout,
            )
    except Exception as error:
        print(f"standalone package acceptance failed: {error}", file=sys.stderr)
        return 1
    print(PACKAGE_PASS_MARKER)
    return 0


def main() -> int:
    example_root = REPOSITORY_ROOT / "examples" / "resnet18"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=example_root / "model")
    parser.add_argument(
        "--source-metadata", type=Path, default=example_root / "model-source.json"
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--deployed-source-commit")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--software-timeout", type=float, default=86400.0)
    parser.add_argument(
        "--package-root",
        type=Path,
        help="accept one extracted standalone release package",
    )
    parser.add_argument("--package-archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--release-tag")
    arguments = parser.parse_args()
    if arguments.package_root is not None:
        return _accept_package(arguments)
    missing = [
        name
        for name in (
            "artifact_dir",
            "expected_source_commit",
            "deployed_source_commit",
        )
        if getattr(arguments, name) is None
    ]
    if missing:
        parser.error(f"--{missing[0].replace('_', '-')} is required")
    try:
        evidence = run_board(
            model_dir=arguments.model_dir,
            source_metadata_path=arguments.source_metadata,
            artifact_dir=arguments.artifact_dir,
            expected_source_commit=arguments.expected_source_commit,
            deployed_source_commit=arguments.deployed_source_commit,
            allow_source_mismatch=arguments.allow_source_mismatch,
            evidence_path=arguments.evidence,
            software_timeout=arguments.software_timeout,
        )
    except Exception as error:
        print(f"physical PYNQ-Z1 acceptance failed: {error}", file=sys.stderr)
        return 1
    print(
        DEVELOPMENT_PASS_MARKER
        if evidence["evidence_type"] == "physical-pynq-z1-development"
        else PASS_MARKER
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
