"""Build a deterministic validated ResNet-18 model archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile


MODULE_DIR = Path(__file__).resolve().parent
# In a deployed standalone package this file sits at the package root; in the
# repository it sits under examples/resnet18/.
REPOSITORY_ROOT = (
    MODULE_DIR
    if (MODULE_DIR / "package.manifest.json").is_file()
    else MODULE_DIR.parents[1]
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.runtime.verify_overlay import verify_artifacts

MODEL_FILENAMES = (
    "acceptance.json",
    "resnet18.conversion.json",
    "resnet18.npu.bin",
    "resnet18.npu.json",
    "resnet18.validation.npy",
)
FIXED_DATE = (1980, 1, 1, 0, 0, 0)


class ResNet18PackageError(RuntimeError):
    """The model workspace is incomplete, stale, or substituted."""


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResNet18PackageError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ResNet18PackageError(f"non-finite JSON value {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ResNet18PackageError(f"invalid {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ResNet18PackageError(f"{path.name} must contain an object")
    canonical = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ResNet18PackageError(f"{path.name} is not canonical JSON")
    return value


def _digest_record(record: object, key: str, label: str) -> str:
    if not isinstance(record, dict) or not isinstance(record.get(key), str):
        raise ResNet18PackageError(f"{label} digest record is missing")
    digest = record[key]
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ResNet18PackageError(f"{label} digest is malformed")
    return digest


def validate_workspace(
    model_dir: Path,
    source_metadata_path: Path,
    *,
    require_checkpoint: bool = True,
) -> list[Path]:
    """Return package assets only after validating the whole readiness boundary.

    The pinned TorchVision checkpoint is never redistributed, so a deployed
    release package does not carry it. There, pass ``require_checkpoint=False``:
    the host acceptance record must still name the checkpoint digest that the
    source metadata pins, which binds the package to the same source without
    shipping it.
    """

    model_dir = model_dir.resolve()
    metadata_path = source_metadata_path.resolve()
    if not model_dir.is_dir():
        raise ResNet18PackageError("model workspace is missing; run download first")
    required = [model_dir / name for name in MODEL_FILENAMES]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ResNet18PackageError(
            f"model workspace is incomplete: {missing[0]}; run conversion and validation"
        )
    source = _json(metadata_path)
    checkpoint_name = source.get("filename")
    if (
        not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise ResNet18PackageError("source checkpoint filename is invalid")
    checkpoint = model_dir / checkpoint_name
    if require_checkpoint:
        if not checkpoint.is_file():
            raise ResNet18PackageError("pinned source checkpoint is missing")
        if (
            checkpoint.stat().st_size != source.get("bytes")
            or _sha256(checkpoint) != source.get("sha256")
        ):
            raise ResNet18PackageError(
                "pinned source checkpoint differs from metadata"
            )
        checkpoint_sha256 = _sha256(checkpoint)
    else:
        checkpoint_sha256 = source.get("sha256")
        if not isinstance(checkpoint_sha256, str):
            raise ResNet18PackageError("source metadata pins no checkpoint digest")

    conversion_path = model_dir / "resnet18.conversion.json"
    acceptance_path = model_dir / "acceptance.json"
    conversion = _json(conversion_path)
    acceptance = _json(acceptance_path)
    if conversion.get("magic") != "NPU_RESNET18_CONVERSION":
        raise ResNet18PackageError("conversion provenance magic is invalid")
    if (
        acceptance.get("magic") != "NPU_RESNET18_ACCEPTANCE"
        or acceptance.get("result") != "pass"
    ):
        raise ResNet18PackageError("host acceptance descriptor did not pass")
    if acceptance.get("evidence_type") != "real-model-host":
        raise ResNet18PackageError("acceptance evidence type is invalid")
    runtime = acceptance.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("physical_board") is not False:
        raise ResNet18PackageError("host evidence must not claim a physical board")

    checks = (
        (
            model_dir / "resnet18.npu.json",
            _digest_record(
                conversion.get("model"),
                "manifest_sha256",
                "conversion manifest",
            ),
        ),
        (
            model_dir / "resnet18.npu.bin",
            _digest_record(conversion.get("model"), "payload_sha256", "conversion payload"),
        ),
        (
            model_dir / "resnet18.validation.npy",
            _digest_record(conversion.get("input"), "sha256", "conversion input"),
        ),
        (
            conversion_path,
            _digest_record(acceptance.get("conversion"), "sha256", "acceptance conversion"),
        ),
        (
            model_dir / "resnet18.npu.json",
            _digest_record(acceptance.get("model"), "manifest_sha256", "acceptance manifest"),
        ),
        (
            model_dir / "resnet18.npu.bin",
            _digest_record(acceptance.get("model"), "payload_sha256", "acceptance payload"),
        ),
        (
            model_dir / "resnet18.validation.npy",
            _digest_record(acceptance.get("input"), "sha256", "acceptance input"),
        ),
    )
    for path, expected in checks:
        if _sha256(path) != expected:
            raise ResNet18PackageError(f"stale or substituted asset: {path.name}")
    if (
        _digest_record(acceptance.get("source"), "sha256", "acceptance source")
        != checkpoint_sha256
    ):
        raise ResNet18PackageError("acceptance source differs from pinned checkpoint")
    return required


def _record(name: str, data: bytes) -> dict[str, object]:
    return {
        "bytes": len(data),
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_archive(output: Path, entries: list[tuple[str, bytes]]) -> None:
    """Write one deterministic ZIP atomically, or leave no output behind."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, data in sorted(entries):
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ResNet18PackageError(f"unsafe archive path: {name}")
                info = zipfile.ZipInfo(name, FIXED_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        if output.exists():
            raise ResNet18PackageError("output archive appeared during packaging")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_model_archive(
    *,
    model_dir: Path,
    source_metadata_path: Path,
    output_archive: Path,
) -> dict[str, object]:
    """Validate inputs, then atomically publish a deterministic model ZIP."""

    output = output_archive.resolve()
    if output.suffix.lower() != ".zip":
        raise ResNet18PackageError("output archive must use .zip")
    if output.exists():
        raise ResNet18PackageError("output archive already exists")
    assets = validate_workspace(model_dir, source_metadata_path)
    entries = [(f"model/{path.name}", path.read_bytes()) for path in assets]
    entries.append(("model-source.json", source_metadata_path.resolve().read_bytes()))
    entries.sort(key=lambda item: item[0])
    manifest: dict[str, object] = {
        "evidence_type": "real-model-host",
        "files": [_record(name, data) for name, data in entries],
        "format": {"major": 1, "minor": 0},
        "magic": "NPU_RESNET18_MODEL_PACKAGE",
    }
    manifest_data = (
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    entries.append(("package.manifest.json", manifest_data))
    _write_archive(output, entries)
    return manifest


# --- Standalone release delivery -------------------------------------------

DELIVERY_MAGIC = "NPU_RESNET18_RELEASE_PACKAGE"
RELEASE_TAG_PATTERN = re.compile(
    r"(?:v[0-9]+\.[0-9]+\.[0-9]+|local-[0-9a-fA-F]{8,64})"
)
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")

DELIVERY_SOURCE_FILES = {
    "examples/resnet18/resnet18.ipynb": "resnet18.ipynb",
    "examples/resnet18/README.md": "README.md",
    "examples/resnet18/QUICKSTART.md": "QUICKSTART.md",
    "examples/resnet18/run_on_board.py": "run_on_board.py",
    "examples/resnet18/accept_image_on_board.py": "accept_image_on_board.py",
    "examples/resnet18/package_example.py": "package_example.py",
    "examples/resnet18/demo-source.json": "demo-source.json",
    "examples/resnet18/gallery-source.json": "gallery-source.json",
    "examples/resnet18/model-source.json": "model-source.json",
    "src/export/__init__.py": "src/export/__init__.py",
    "src/export/imagenet.py": "src/export/imagenet.py",
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
DELIVERY_ARTIFACT_FILES = (
    "npu_matrix.bit",
    "npu_matrix.hwh",
    "npu_matrix.manifest.json",
)
DELIVERY_REPORT_FILES = (
    "build_evidence.txt",
    "drc_routed.rpt",
    "route_status.rpt",
    "timing_summary_routed.rpt",
    "utilization_impl.rpt",
    "utilization_synth.rpt",
)
DELIVERY_EVIDENCE_KEYS = (
    "bit",
    "drc_errors",
    "hwh",
    "part",
    "setup_failing_paths",
    "source_commit",
    "vivado",
    "wns",
)


def _validated_release_tag(value: str) -> str:
    tag = str(value).strip()
    if RELEASE_TAG_PATTERN.fullmatch(tag) is None:
        raise ResNet18PackageError(
            "release tag must match vMAJOR.MINOR.PATCH or local-<commit>"
        )
    return tag


def _validated_commit(value: str) -> str:
    commit = str(value).strip().lower()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise ResNet18PackageError("source commit must be a full hexadecimal object id")
    return commit


def parse_build_evidence(path: Path) -> dict[str, str]:
    """Return the key/value build evidence Vivado recorded for this overlay."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResNet18PackageError(f"build evidence is unreadable: {error}") from error
    values: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ResNet18PackageError(f"build evidence line is malformed: {line!r}")
        key = key.strip()
        if key in values:
            raise ResNet18PackageError(f"duplicate build evidence key {key!r}")
        values[key] = value.strip()
    missing = [key for key in DELIVERY_EVIDENCE_KEYS if key not in values]
    if missing:
        raise ResNet18PackageError(
            f"build evidence is incomplete: missing {missing[0]}"
        )
    return values


def evaluate_vivado_gates(evidence: dict[str, str], source_commit: str) -> dict:
    """Fail closed unless timing, DRC, and provenance gates all passed."""

    try:
        wns = float(evidence["wns"])
        setup_failing_paths = int(evidence["setup_failing_paths"])
        drc_errors = int(evidence["drc_errors"])
    except ValueError as error:
        raise ResNet18PackageError(
            f"vivado gates are not numeric: {error}"
        ) from error
    if evidence["source_commit"].strip().lower() != source_commit:
        raise ResNet18PackageError(
            "vivado gates reject this build: evidence names another source commit"
        )
    if drc_errors != 0:
        raise ResNet18PackageError(
            f"vivado gates reject this build: {drc_errors} DRC errors"
        )
    if setup_failing_paths != 0:
        raise ResNet18PackageError(
            f"vivado gates reject this build: {setup_failing_paths} failing setup paths"
        )
    if wns < 0.0:
        raise ResNet18PackageError(
            f"vivado gates reject this build: worst negative slack is {wns}"
        )
    return {
        "drc_errors": drc_errors,
        "part": evidence["part"],
        "setup_failing_paths": setup_failing_paths,
        "vivado": evidence["vivado"],
        "wns": wns,
    }


def _delivery_entries(
    *,
    repository_root: Path,
    artifact_dir: Path,
    report_dir: Path,
    descriptor_path: Path | None,
    model_dir: Path | None,
    source_metadata_path: Path | None,
) -> tuple[list[tuple[str, bytes]], list[dict[str, object]]]:
    sources: list[tuple[str, Path]] = [
        (destination, repository_root / source)
        for source, destination in DELIVERY_SOURCE_FILES.items()
    ]
    sources.extend(
        (f"artifacts/{name}", artifact_dir / name) for name in DELIVERY_ARTIFACT_FILES
    )
    sources.extend(
        (f"reports/{name}", report_dir / name) for name in DELIVERY_REPORT_FILES
    )

    if descriptor_path is not None:
        # An externally supplied acceptance bundle (a corpus with labels). The
        # real ResNet-18 release is accepted from its model workspace instead.
        bundle = _json(descriptor_path)
        assets = bundle.get("assets")
        if not isinstance(assets, dict):
            raise ResNet18PackageError("acceptance descriptor has no asset record")
        sources.append(("acceptance/acceptance.json", descriptor_path))
        descriptor_root = descriptor_path.parent
        for name in sorted(assets):
            record = assets[name]
            if not isinstance(record, dict) or not isinstance(
                record.get("filename"), str
            ):
                raise ResNet18PackageError(f"acceptance asset {name!r} is malformed")
            filename = record["filename"]
            pure = PurePosixPath(filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ResNet18PackageError(
                    f"acceptance asset {name!r} escapes the bundle"
                )
            sources.append((f"acceptance/{filename}", descriptor_root / filename))

    if model_dir is not None and source_metadata_path is not None:
        validate_workspace(model_dir, source_metadata_path)
        # Ship the metadata that was just validated, so the board checks the
        # workspace against the same pin rather than whatever the checkout holds.
        sources = [
            (destination, source_metadata_path.resolve())
            if destination == "model-source.json"
            else (destination, source)
            for destination, source in sources
        ]
        checkpoint = str(_json(source_metadata_path).get("filename"))
        for path in sorted(model_dir.resolve().iterdir()):
            if not path.is_file() or path.name == checkpoint:
                continue
            sources.append((f"model/{path.name}", path))

    missing = [
        destination for destination, source in sources if not source.is_file()
    ]
    if missing:
        raise ResNet18PackageError(
            f"required package inputs are missing: {missing[0]}"
        )
    seen: set[str] = set()
    entries: list[tuple[str, bytes]] = []
    for destination, source in sources:
        if destination in seen:
            raise ResNet18PackageError(f"duplicate package entry: {destination}")
        seen.add(destination)
        entries.append((destination, source.read_bytes()))
    entries.sort(key=lambda item: item[0])
    report_records = [
        _record(destination.split("/", 1)[1], data)
        for destination, data in entries
        if destination.startswith("reports/")
    ]
    return entries, sorted(report_records, key=lambda record: str(record["path"]))


def build_delivery_archive(
    *,
    repository_root: Path,
    artifact_dir: Path,
    report_dir: Path,
    output_archive: Path,
    release_tag: str,
    source_commit: str,
    descriptor_path: Path | None = None,
    model_dir: Path | None = None,
    source_metadata_path: Path | None = None,
    overlay_commit: str | None = None,
) -> dict[str, object]:
    """Publish one reproducible standalone board package, or publish nothing.

    ``overlay_commit`` names the commit the overlay was built from when a
    release reuses an earlier build of identical hardware sources; the caller
    proves that identity. By default the overlay must come from this commit.

    The board accepts a package from one of two sources: the real model
    workspace, whose host acceptance record binds the exact model, input and
    captures; or an externally supplied acceptance bundle. At least one must be
    present, or the package could never be accepted.
    """

    repository_root = repository_root.resolve()
    artifact_dir = artifact_dir.resolve()
    report_dir = report_dir.resolve()
    if descriptor_path is not None:
        descriptor_path = descriptor_path.resolve()
    if descriptor_path is None and model_dir is None:
        raise ResNet18PackageError(
            "a release package needs the model workspace or an acceptance bundle"
        )
    output = output_archive.resolve()
    if output.suffix.lower() != ".zip":
        raise ResNet18PackageError("output archive must use .zip")
    if output.exists():
        raise ResNet18PackageError("output archive already exists")
    tag = _validated_release_tag(release_tag)
    commit = _validated_commit(source_commit)
    hardware_commit = (
        commit if overlay_commit is None else _validated_commit(overlay_commit)
    )

    overlay = verify_artifacts(artifact_dir)
    if str(overlay.get("source_commit", "")).lower() != hardware_commit:
        raise ResNet18PackageError(
            "overlay manifest source commit differs from the release commit"
            if overlay_commit is None
            else "overlay manifest source commit differs from --overlay-commit"
        )
    gates = evaluate_vivado_gates(
        parse_build_evidence(report_dir / "build_evidence.txt"), hardware_commit
    )

    entries, report_records = _delivery_entries(
        repository_root=repository_root,
        artifact_dir=artifact_dir,
        report_dir=report_dir,
        descriptor_path=descriptor_path,
        model_dir=model_dir,
        source_metadata_path=source_metadata_path,
    )
    reports_manifest = (
        json.dumps(
            {"files": report_records},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    entries.append(("reports/reports.manifest.json", reports_manifest))
    entries.sort(key=lambda item: item[0])

    manifest: dict[str, object] = {
        "evidence_type": "real-model-host",
        "files": [_record(name, data) for name, data in entries],
        "format": {"major": 1, "minor": 0},
        "magic": DELIVERY_MAGIC,
        "acceptance_source": (
            "bundle" if descriptor_path is not None else "model-workspace"
        ),
        "model_workspace": any(
            name.startswith("model/") for name, _data in entries
        ),
        "overlay": {
            "bit_sha256": overlay["bit"]["sha256"],
            "hwh_sha256": overlay["hwh"]["sha256"],
            "source_commit": hardware_commit,
            "target_part": overlay["target_part"],
        },
        "release_tag": tag,
        "source_commit": commit,
        "vivado_gates": gates,
    }
    manifest_data = (
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    entries.append(("package.manifest.json", manifest_data))
    entries.sort(key=lambda item: item[0])
    for name, _data in entries:
        if str(repository_root) in name:
            raise ResNet18PackageError(f"package entry leaks a host path: {name}")
    _write_archive(output, entries)
    return manifest


def build_archive(**arguments: object) -> dict[str, object]:
    """Build the standalone board package, or the model-only archive."""

    delivery_keys = {
        "repository_root",
        "artifact_dir",
        "report_dir",
        "release_tag",
        "source_commit",
    }
    if (delivery_keys | {"descriptor_path"}) & set(arguments):
        missing = sorted(delivery_keys - set(arguments))
        if missing:
            raise ResNet18PackageError(
                f"standalone package requires {missing[0]}"
            )
        return build_delivery_archive(**arguments)  # type: ignore[arg-type]
    return build_model_archive(**arguments)  # type: ignore[arg-type]


def main() -> int:
    example_root = REPOSITORY_ROOT / "examples" / "resnet18"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=example_root / "model")
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=example_root / "model-source.json",
    )
    parser.add_argument(
        "--output-archive",
        type=Path,
        default=REPOSITORY_ROOT / "mount" / "resnet18" / "resnet18-model.zip",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="build the standalone board package from this checkout",
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--release-tag")
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--overlay-commit",
        help="the overlay was built from this earlier commit (reused build)",
    )
    parser.add_argument(
        "--without-model-workspace",
        action="store_true",
        help="omit the ignored model workspace from the standalone package",
    )
    arguments = parser.parse_args()

    if arguments.check_only:
        validate_workspace(arguments.model_dir, arguments.source_metadata)
        print("PASS [real-model-host]: model workspace is package-ready")
        return 0

    if arguments.repository_root is not None:
        missing = [
            name
            for name in ("artifact_dir", "report_dir")
            if getattr(arguments, name) is None
        ]
        if missing:
            parser.error(f"--{missing[0].replace('_', '-')} is required")
        artifact_dir = arguments.artifact_dir.resolve()
        overlay = verify_artifacts(artifact_dir)
        source_commit = arguments.source_commit or str(overlay["source_commit"])
        release_tag = arguments.release_tag or f"local-{source_commit[:8]}"
        build_delivery_archive(
            repository_root=arguments.repository_root,
            artifact_dir=artifact_dir,
            report_dir=arguments.report_dir,
            descriptor_path=arguments.descriptor,
            output_archive=arguments.output_archive,
            release_tag=release_tag,
            source_commit=source_commit,
            model_dir=None if arguments.without_model_workspace else arguments.model_dir,
            source_metadata_path=(
                None if arguments.without_model_workspace else arguments.source_metadata
            ),
            overlay_commit=arguments.overlay_commit,
        )
        print(
            "PASS [real-model-host]: standalone release package at "
            f"{arguments.output_archive.resolve()}"
        )
        return 0

    build_model_archive(
        model_dir=arguments.model_dir,
        source_metadata_path=arguments.source_metadata,
        output_archive=arguments.output_archive,
    )
    print(f"PASS [real-model-host]: model package at {arguments.output_archive.resolve()}")
    print("INFO: add trusted Vivado artifacts before physical PYNQ-Z1 acceptance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
