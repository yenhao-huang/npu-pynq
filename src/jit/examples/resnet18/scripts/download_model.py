"""Download the one pinned TorchVision ResNet-18 checkpoint fail-closed."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping
import urllib.parse
import urllib.request


APPROVED_HOST = "download.pytorch.org"
MAGIC = "NPU_PRETRAINED_MODEL_SOURCE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FIELDS = {
    "architecture",
    "bytes",
    "filename",
    "format",
    "license",
    "magic",
    "provider",
    "revision",
    "sha256",
    "url",
}


class ModelDownloadError(RuntimeError):
    """The pinned source metadata or downloaded checkpoint is untrusted."""


@dataclass(frozen=True)
class ModelSource:
    architecture: str
    byte_length: int
    filename: str
    license: Mapping[str, str]
    provider: str
    revision: str
    sha256: str
    url: str


def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelDownloadError(f"duplicate metadata field: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() for character in value
    ):
        raise ModelDownloadError(f"{name} must be a non-empty identifier")
    return value


def _validated_url(value: object, filename: str) -> str:
    if not isinstance(value, str):
        raise ModelDownloadError("url must be a string")
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != APPROVED_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or Path(urllib.parse.unquote(parsed.path)).name != filename
        or parsed.query
        or parsed.fragment
    ):
        raise ModelDownloadError("url host/path must be an approved HTTPS model URL")
    return value


def load_source_metadata(path: str | Path) -> ModelSource:
    """Load exact canonical source metadata without accepting ambiguous JSON."""

    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelDownloadError(f"source metadata cannot be read: {error}") from error
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ModelDownloadError("source metadata fields do not match the contract")
    try:
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise ModelDownloadError(f"source metadata is invalid: {error}") from error
    if not raw.endswith((b"\n", b"\r\n")) or raw.rstrip(b"\r\n") != canonical[:-1]:
        raise ModelDownloadError("source metadata is not canonical")
    if value["magic"] != MAGIC or value["format"] != {"major": 1, "minor": 0}:
        raise ModelDownloadError("source metadata format is unsupported")
    filename = value["filename"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in (".", "..")
    ):
        raise ModelDownloadError("filename must be a safe basename")
    byte_length = value["bytes"]
    if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length <= 0:
        raise ModelDownloadError("bytes must be a positive integer")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
        raise ModelDownloadError("sha256 must be a lowercase full digest")
    license_value = value["license"]
    if (
        not isinstance(license_value, dict)
        or set(license_value) != {"dataset_terms", "source_project_spdx"}
        or not all(isinstance(item, str) and item for item in license_value.values())
    ):
        raise ModelDownloadError("license metadata is incomplete")
    return ModelSource(
        architecture=_require_identifier("architecture", value["architecture"]),
        byte_length=byte_length,
        filename=filename,
        license=MappingProxyType(dict(license_value)),
        provider=_require_identifier("provider", value["provider"]),
        revision=_require_identifier("revision", value["revision"]),
        sha256=sha256,
        url=_validated_url(value["url"], filename),
    )


def download_model(
    metadata_path: str | Path,
    destination_dir: str | Path,
    *,
    opener: Callable = urllib.request.urlopen,
    timeout: float = 60.0,
) -> Path:
    """Stream, authenticate, and atomically publish one pinned checkpoint."""

    source = load_source_metadata(metadata_path)
    destination_dir = Path(destination_dir)
    destination = destination_dir / source.filename
    temporary = destination_dir / f".{source.filename}.tmp"
    if destination.exists() or temporary.exists():
        raise ModelDownloadError(f"destination or temporary file already exists: {destination}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "npu-in-pynq-resnet18-downloader/1"},
    )
    digest = hashlib.sha256()
    count = 0
    try:
        with opener(request, timeout=timeout) as response:
            final_url = response.geturl()
            _validated_url(final_url, source.filename)
            with temporary.open("xb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > source.byte_length:
                        raise ModelDownloadError("download size exceeds pinned metadata")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if count != source.byte_length:
            raise ModelDownloadError(
                f"download size {count} differs from pinned {source.byte_length}"
            )
        if digest.hexdigest() != source.sha256:
            raise ModelDownloadError("download digest differs from pinned metadata")
        os.replace(temporary, destination)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if isinstance(error, ModelDownloadError):
            raise
        raise ModelDownloadError(f"checkpoint download failed: {error}") from error
    return destination


def main() -> int:
    example_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=example_root / "model-source.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=example_root / "model",
    )
    arguments = parser.parse_args()
    output = download_model(arguments.metadata, arguments.output_dir)
    print(f"PASS: verified pretrained ResNet-18 checkpoint at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
