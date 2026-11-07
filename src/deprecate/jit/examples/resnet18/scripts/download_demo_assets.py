"""Download the pinned ImageNet class list and sample image, fail-closed."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable
import urllib.parse
import urllib.request


APPROVED_HOST = "raw.githubusercontent.com"
APPROVED_REPOSITORY = "pytorch/hub"
MAGIC = "NPU_RESNET18_DEMO_SOURCE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
ASSET_KINDS = ("class-names", "sample-image")
ASSET_FIELDS = {"bytes", "filename", "kind", "license", "sha256", "url"}


class DemoAssetError(RuntimeError):
    """The pinned demo metadata or a downloaded asset is untrusted."""


@dataclass(frozen=True)
class DemoAsset:
    byte_length: int
    filename: str
    kind: str
    expected_class: str | None
    sha256: str
    url: str


def _object_without_duplicates(pairs):
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DemoAssetError(f"duplicate metadata field: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validated_url(value: object, filename: str, revision: str) -> str:
    if not isinstance(value, str):
        raise DemoAssetError("asset url must be a string")
    parsed = urllib.parse.urlparse(value)
    prefix = f"/{APPROVED_REPOSITORY}/{revision}/"
    if (
        parsed.scheme != "https"
        or parsed.hostname != APPROVED_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(prefix)
        or ".." in parsed.path.split("/")
    ):
        raise DemoAssetError("asset url must be a pinned approved HTTPS raw URL")
    return value


def load_demo_metadata(path: str | Path) -> tuple[DemoAsset, ...]:
    """Load exact canonical demo metadata without accepting ambiguous JSON."""

    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DemoAssetError(f"demo metadata cannot be read: {error}") from error
    if not isinstance(value, dict) or set(value) != {
        "assets",
        "format",
        "magic",
        "revision",
    }:
        raise DemoAssetError("demo metadata fields do not match the contract")
    try:
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise DemoAssetError(f"demo metadata is invalid: {error}") from error
    if raw != canonical:
        raise DemoAssetError("demo metadata is not canonical")
    if value["magic"] != MAGIC or value["format"] != {"major": 1, "minor": 0}:
        raise DemoAssetError("demo metadata format is unsupported")
    revision = value["revision"]
    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        raise DemoAssetError("revision must be a full lowercase commit id")
    entries = value["assets"]
    if not isinstance(entries, list) or not entries:
        raise DemoAssetError("demo metadata lists no assets")

    assets: list[DemoAsset] = []
    for entry in entries:
        if not isinstance(entry, dict) or not ASSET_FIELDS <= set(entry):
            raise DemoAssetError("asset record fields do not match the contract")
        if not set(entry) <= ASSET_FIELDS | {"expected_class"}:
            raise DemoAssetError("asset record holds unknown fields")
        filename = entry["filename"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in (".", "..")
        ):
            raise DemoAssetError("asset filename must be a safe basename")
        byte_length = entry["bytes"]
        if (
            isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length <= 0
        ):
            raise DemoAssetError("asset bytes must be a positive integer")
        digest = entry["sha256"]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise DemoAssetError("asset sha256 must be a lowercase full digest")
        kind = entry["kind"]
        if kind not in ASSET_KINDS:
            raise DemoAssetError(f"asset kind {kind!r} is not recognised")
        license_value = entry["license"]
        if (
            not isinstance(license_value, dict)
            or set(license_value) != {"notes", "spdx"}
            or not all(isinstance(item, str) and item for item in license_value.values())
        ):
            raise DemoAssetError("asset license metadata is incomplete")
        expected_class = entry.get("expected_class")
        if expected_class is not None and (
            not isinstance(expected_class, str) or not expected_class.strip()
        ):
            raise DemoAssetError("expected_class must be a non-empty label")
        assets.append(
            DemoAsset(
                byte_length=byte_length,
                filename=filename,
                kind=kind,
                expected_class=expected_class,
                sha256=digest,
                url=_validated_url(entry["url"], filename, revision),
            )
        )
    kinds = [asset.kind for asset in assets]
    if sorted(kinds) != sorted(set(kinds)) or set(kinds) != set(ASSET_KINDS):
        raise DemoAssetError("demo metadata must pin exactly one asset per kind")
    return tuple(assets)


def _download_one(
    asset: DemoAsset,
    destination_dir: Path,
    *,
    opener: Callable,
    timeout: float,
    revision: str,
) -> Path:
    destination = destination_dir / asset.filename
    temporary = destination_dir / f".{asset.filename}.tmp"
    if destination.exists() or temporary.exists():
        raise DemoAssetError(f"destination or temporary file already exists: {destination}")
    request = urllib.request.Request(
        asset.url, headers={"User-Agent": "npu-in-pynq-resnet18-demo/1"}
    )
    digest = hashlib.sha256()
    count = 0
    try:
        with opener(request, timeout=timeout) as response:
            _validated_url(response.geturl(), asset.filename, revision)
            with temporary.open("xb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > asset.byte_length:
                        raise DemoAssetError("download size exceeds pinned metadata")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if count != asset.byte_length:
            raise DemoAssetError(
                f"download size {count} differs from pinned {asset.byte_length}"
            )
        if digest.hexdigest() != asset.sha256:
            raise DemoAssetError("download digest differs from pinned metadata")
        os.replace(temporary, destination)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        if isinstance(error, DemoAssetError):
            raise
        raise DemoAssetError(f"demo asset download failed: {error}") from error
    return destination


def download_demo_assets(
    metadata_path: str | Path,
    destination_dir: str | Path,
    *,
    opener: Callable = urllib.request.urlopen,
    timeout: float = 60.0,
) -> list[Path]:
    """Stream, authenticate, and atomically publish every pinned demo asset."""

    metadata_path = Path(metadata_path)
    assets = load_demo_metadata(metadata_path)
    revision = json.loads(metadata_path.read_bytes())["revision"]
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    return [
        _download_one(
            asset,
            destination_dir,
            opener=opener,
            timeout=timeout,
            revision=revision,
        )
        for asset in assets
    ]


def main() -> int:
    example_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=example_root / "demo-source.json"
    )
    parser.add_argument("--output-dir", type=Path, default=example_root / "model")
    arguments = parser.parse_args()
    for path in download_demo_assets(arguments.metadata, arguments.output_dir):
        print(f"PASS: verified pinned demo asset at {path}")
    print("INFO: demo assets stay in the ignored model workspace; they are not committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
