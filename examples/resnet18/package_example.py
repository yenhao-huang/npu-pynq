"""Build a deterministic standalone ResNet-18 acceptance archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model.resnet18 import load_acceptance_bundle, validate_resnet18_topology
from src.runtime.model import load_model_package
from src.runtime.verify_overlay import verify_artifacts


LABEL_PATTERN = re.compile(r"(?:v[0-9]+\.[0-9]+\.[0-9]+|local-[0-9a-fA-F]{8,64})")
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")
SOURCE_FILES = {
    "examples/resnet18/README.md": "README.md",
    "examples/resnet18/resnet18.ipynb": "resnet18.ipynb",
    "examples/resnet18/run_on_board.py": "run_on_board.py",
    "src/export/__init__.py": "src/export/__init__.py",
    "src/export/planner.py": "src/export/planner.py",
    "src/export/resnet.py": "src/export/resnet.py",
    "src/model/__init__.py": "src/model/__init__.py",
    "src/model/numeric.py": "src/model/numeric.py",
    "src/model/operators.py": "src/model/operators.py",
    "src/model/package.py": "src/model/package.py",
    "src/model/resnet.py": "src/model/resnet.py",
    "src/model/resnet18.py": "src/model/resnet18.py",
    "src/runtime/__init__.py": "src/runtime/__init__.py",
    "src/runtime/acceptance.py": "src/runtime/acceptance.py",
    "src/runtime/lowering.py": "src/runtime/lowering.py",
    "src/runtime/model.py": "src/runtime/model.py",
    "src/runtime/npu.py": "src/runtime/npu.py",
    "src/runtime/verify_overlay.py": "src/runtime/verify_overlay.py",
}
ARTIFACT_FILES = (
    "npu_matrix.bit",
    "npu_matrix.hwh",
    "npu_matrix.manifest.json",
)
REPORT_FILES = (
    "build_evidence.txt",
    "drc_routed.rpt",
    "route_status.rpt",
    "timing_summary_routed.rpt",
    "utilization_impl.rpt",
    "utilization_synth.rpt",
)
FIXED_DATE = (1980, 1, 1, 0, 0, 0)


class ResNet18PackageError(RuntimeError):
    """The requested standalone package is incomplete or untrusted."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validated_label(value: str) -> str:
    label = value.strip()
    if LABEL_PATTERN.fullmatch(label) is None:
        raise ResNet18PackageError("release tag must be semantic or local-<commit>")
    return label


def _validated_commit(value: str) -> str:
    commit = value.strip().lower()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise ResNet18PackageError("source commit must be a full hexadecimal id")
    return commit


def _build_gates(report_path: Path) -> dict[str, object]:
    try:
        pairs = {}
        for line in report_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in pairs:
                raise ValueError("malformed or duplicate record")
            pairs[key] = value
        required = {
            "vivado", "part", "wns", "setup_failing_paths", "drc_errors",
            "bit", "hwh", "source_commit",
        }
        if set(pairs) != required:
            raise ValueError("build evidence keys differ from the contract")
        wns = float(pairs["wns"])
        failing = int(pairs["setup_failing_paths"])
        drc_errors = int(pairs["drc_errors"])
    except (OSError, UnicodeError, ValueError) as error:
        raise ResNet18PackageError(f"build evidence is invalid: {error}") from error
    if not (math.isfinite(wns) and wns >= 0.0 and failing == 0 and drc_errors == 0):
        raise ResNet18PackageError("trusted implementation gates did not pass")
    return {
        "drc_errors": drc_errors,
        "implementation_complete": True,
        "source_commit": pairs["source_commit"].lower(),
        "setup_failing_paths": failing,
        "synthesis_complete": True,
        "target_part": pairs["part"],
        "vivado_version": pairs["vivado"],
        "wns": wns,
    }


def _record(path: str, data: bytes) -> dict[str, object]:
    return {"bytes": len(data), "path": path, "sha256": _sha256_bytes(data)}


def _source_from_commit(repository_root: Path, commit: str, source: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"{commit}:{source}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ResNet18PackageError(f"Git cannot verify source {source}: {error}") from error
    if result.returncode != 0:
        raise ResNet18PackageError(f"source is absent from commit: {source}")
    return result.stdout


def _entry(path: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != path:
        raise ResNet18PackageError(f"unsafe archive path: {path}")
    info = zipfile.ZipInfo(path, FIXED_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info, data


def build_archive(
    *,
    repository_root: Path,
    artifact_dir: Path,
    report_dir: Path,
    descriptor_path: Path,
    output_archive: Path,
    release_tag: str,
    source_commit: str,
) -> dict[str, object]:
    """Validate all external inputs and atomically publish one deterministic ZIP."""

    repository_root = repository_root.resolve()
    artifact_dir = artifact_dir.resolve()
    report_dir = report_dir.resolve()
    descriptor_path = descriptor_path.resolve()
    output_archive = output_archive.resolve()
    label = _validated_label(release_tag)
    commit = _validated_commit(source_commit)
    if output_archive.exists() or output_archive.suffix.lower() != ".zip":
        raise ResNet18PackageError("output archive must be a new .zip path")

    overlay = verify_artifacts(artifact_dir)
    if str(overlay.get("source_commit", "")).lower() != commit:
        raise ResNet18PackageError("overlay source commit differs from package commit")
    gates = _build_gates(report_dir / "build_evidence.txt")
    if gates["source_commit"] != commit:
        raise ResNet18PackageError("build evidence source commit differs")
    if gates["target_part"] != overlay["target_part"]:
        raise ResNet18PackageError("build evidence target part differs")
    model = load_model_package(
        load_acceptance_bundle(descriptor_path).model_manifest_path
    )
    bundle = load_acceptance_bundle(descriptor_path, graph=model.graph)
    validate_resnet18_topology(model.graph)

    sources: list[tuple[str, bytes]] = []
    for source, destination in sorted(SOURCE_FILES.items()):
        sources.append((destination, _source_from_commit(repository_root, commit, source)))
    for name in ARTIFACT_FILES:
        path = artifact_dir / name
        if not path.is_file():
            raise ResNet18PackageError(f"overlay artifact is missing: {name}")
        sources.append((f"artifacts/{name}", path.read_bytes()))
    report_records = []
    for name in REPORT_FILES:
        path = report_dir / name
        if not path.is_file():
            raise ResNet18PackageError(f"Vivado report is missing: {name}")
        report_records.append(_record(name, path.read_bytes()))
    report_manifest = {
        "files": sorted(report_records, key=lambda record: str(record["path"])),
        "vivado_gates": gates,
    }
    sources.append((
        "reports/reports.manifest.json",
        (
            json.dumps(
                report_manifest,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    ))
    sources.append(("acceptance/acceptance.json", descriptor_path.read_bytes()))
    for asset in bundle.descriptor.assets.values():
        sources.append((f"acceptance/{asset.filename}", asset.path.read_bytes()))

    names = [name for name, _data in sources]
    if len(names) != len(set(names)):
        raise ResNet18PackageError("package destinations collide")
    root_tokens = {
        str(repository_root).encode("utf-8"),
        repository_root.as_posix().encode("utf-8"),
    }
    for name, data in sources:
        if any(token and token in data for token in root_tokens):
            raise ResNet18PackageError(f"host path leaked into {name}")

    manifest = {
        "files": [_record(name, data) for name, data in sorted(sources)],
        "format": {"major": 1, "minor": 0},
        "magic": "NPU_RESNET18_PACKAGE",
        "release_tag": label,
        "source_commit": commit,
        "target_part": overlay["target_part"],
        "vivado_gates": gates,
    }
    manifest_data = (
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    all_entries = sorted([*sources, ("package.manifest.json", manifest_data)])
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_archive.with_name(f".{output_archive.name}.tmp")
    if temporary.exists():
        raise ResNet18PackageError("stale package temporary file exists")
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, data in all_entries:
                info, payload = _entry(name, data)
                archive.writestr(info, payload)
        temporary.replace(output_archive)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    build_archive(
        repository_root=arguments.repository_root,
        artifact_dir=arguments.artifact_dir,
        report_dir=arguments.report_dir,
        descriptor_path=arguments.descriptor,
        output_archive=arguments.output_archive,
        release_tag=arguments.release_tag,
        source_commit=arguments.source_commit,
    )
    print(f"PASS: standalone ResNet-18 package at {arguments.output_archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
