"""Build a deterministic validated ResNet-18 model archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
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


def validate_workspace(model_dir: Path, source_metadata_path: Path) -> list[Path]:
    """Return package assets only after validating the whole readiness boundary."""

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
    if not checkpoint.is_file():
        raise ResNet18PackageError("pinned source checkpoint is missing")
    if (
        checkpoint.stat().st_size != source.get("bytes")
        or _sha256(checkpoint) != source.get("sha256")
    ):
        raise ResNet18PackageError("pinned source checkpoint differs from metadata")

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
        != _sha256(checkpoint)
    ):
        raise ResNet18PackageError("acceptance source differs from pinned checkpoint")
    return required


def _record(name: str, data: bytes) -> dict[str, object]:
    return {
        "bytes": len(data),
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_archive(
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
    return manifest


def main() -> int:
    example_root = REPOSITORY_ROOT / "examples" / "resnet18"
    parser = argparse.ArgumentParser()
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
    arguments = parser.parse_args()
    build_archive(
        model_dir=arguments.model_dir,
        source_metadata_path=arguments.source_metadata,
        output_archive=arguments.output_archive,
    )
    print(f"PASS [real-model-host]: model package at {arguments.output_archive.resolve()}")
    print("INFO: add trusted Vivado artifacts before physical PYNQ-Z1 acceptance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
