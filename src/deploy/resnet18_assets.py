"""Download ResNet-18's pinned sources fail-closed into the ignored model
workspace (dist/resnet18/model):

    python -m src.deploy.resnet18_assets model     TorchVision checkpoint
    python -m src.deploy.resnet18_assets demo      ImageNet class list and sample image
    python -m src.deploy.resnet18_assets gallery   demo gallery photographs

Each file is pinned by URL, byte length and SHA-256 in a canonical JSON
record beside this module, streamed to a temporary file, verified, and only
then published. Nothing downloaded here is committed to the repository; no
gallery image is an ImageNet dataset image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import MappingProxyType
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
import urllib.parse
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1] / "dist/resnet18/model"
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
# Wikimedia requires an informative User-Agent with contact information, and
# refuses generic ones outright from cloud addresses such as CI runners.
USER_AGENT = "npu-in-pynq-resnet18-demo/1 (https://github.com/yenhao-huang/npu-pynq; release source fetch)"
# Wikimedia rate-limits bursts with 429; a short backoff clears it.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
ATTEMPTS = 5
CHUNK = 1024 * 1024


class SourceError(RuntimeError):
    """A pinned source record or a downloaded file is untrusted."""


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


@dataclass(frozen=True)
class DemoAsset:
    byte_length: int
    filename: str
    kind: str
    expected_class: str | None
    sha256: str
    url: str


@dataclass(frozen=True)
class GalleryImage:
    filename: str
    expected_class: str
    byte_length: int
    sha256: str
    url: str
    approved_host: str
    license: dict[str, str]


# Canonical records

def _object_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceError(f"duplicate metadata field: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read_record(path: str | Path, fields: set[str], magic: str, what: str) -> dict:
    """Exact canonical JSON with exactly ``fields``, version 1.0 and ``magic``."""
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceError(f"{what} metadata cannot be read: {error}") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise SourceError(f"{what} metadata fields do not match the contract")
    try:
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise SourceError(f"{what} metadata is invalid: {error}") from error
    if not raw.endswith((b"\n", b"\r\n")) or raw.rstrip(b"\r\n") != canonical[:-1]:
        raise SourceError(f"{what} metadata is not canonical")
    if value["magic"] != magic or value["format"] != {"major": 1, "minor": 0}:
        raise SourceError(f"{what} metadata format is unsupported")
    return value


def _basename(value: object) -> str:
    if not isinstance(value, str) or Path(value).name != value or value in ("", ".", "..") or value.startswith("."):
        raise SourceError("filename must be a safe basename")
    return value


def _byte_length(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceError("bytes must be a positive integer")
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise SourceError("sha256 must be a lowercase full digest")
    return value


def _label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceError("expected_class must be a non-empty label")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise SourceError(f"{name} must be a non-empty identifier")
    return value


def _license(value: object, fields: set[str]) -> dict[str, str]:
    if (not isinstance(value, dict) or set(value) != fields
            or not all(isinstance(item, str) and item.strip() for item in value.values())):
        raise SourceError("license metadata is incomplete")
    return dict(value)


def _https(value: object, host: str, path_ok: Callable[[str], bool]) -> str:
    """``value`` if it is a plain HTTPS URL on ``host`` whose path passes ``path_ok``."""
    if not isinstance(value, str):
        raise SourceError("url must be a string")
    parsed = urllib.parse.urlparse(value)
    if (parsed.scheme != "https" or parsed.hostname != host or parsed.username is not None
            or parsed.password is not None or parsed.port not in (None, 443) or parsed.query
            or parsed.fragment or not path_ok(parsed.path)):
        raise SourceError(f"url is not an approved pinned HTTPS URL: {value}")
    return value


def _model_url(value: object, filename: str) -> str:
    return _https(value, "download.pytorch.org", lambda path: Path(urllib.parse.unquote(path)).name == filename)


def _demo_url(value: object, revision: str) -> str:
    prefix = f"/pytorch/hub/{revision}/"
    return _https(value, "raw.githubusercontent.com",
                  lambda path: path.startswith(prefix) and ".." not in path.split("/"))


def _gallery_url(value: object, host: str) -> str:
    return _https(value, host, lambda path: path.startswith("/wikipedia/commons/"))


def load_source_metadata(path: str | Path = HERE / "resnet18-model-source.json") -> ModelSource:
    value = _read_record(path, {"architecture", "bytes", "filename", "format", "license", "magic", "provider",
                                "revision", "sha256", "url"}, "NPU_PRETRAINED_MODEL_SOURCE", "source")
    filename = _basename(value["filename"])
    return ModelSource(architecture=_identifier("architecture", value["architecture"]),
                       byte_length=_byte_length(value["bytes"]), filename=filename,
                       license=MappingProxyType(_license(value["license"], {"dataset_terms", "source_project_spdx"})),
                       provider=_identifier("provider", value["provider"]),
                       revision=_identifier("revision", value["revision"]), sha256=_sha256(value["sha256"]),
                       url=_model_url(value["url"], filename))


def load_demo_metadata(path: str | Path = HERE / "resnet18-demo-source.json") -> tuple[DemoAsset, ...]:
    value = _read_record(path, {"assets", "format", "magic", "revision"}, "NPU_RESNET18_DEMO_SOURCE", "demo")
    revision = value["revision"]
    if not isinstance(revision, str) or COMMIT.fullmatch(revision) is None:
        raise SourceError("revision must be a full lowercase commit id")
    entries = value["assets"]
    if not isinstance(entries, list) or not entries:
        raise SourceError("demo metadata lists no assets")
    fields = {"bytes", "filename", "kind", "license", "sha256", "url"}
    assets = []
    for entry in entries:
        if not isinstance(entry, dict) or not fields <= set(entry) <= fields | {"expected_class"}:
            raise SourceError("asset record fields do not match the contract")
        if entry["kind"] not in ("class-names", "sample-image"):
            raise SourceError(f"asset kind {entry['kind']!r} is not recognised")
        _license(entry["license"], {"notes", "spdx"})
        expected = entry.get("expected_class")
        assets.append(DemoAsset(byte_length=_byte_length(entry["bytes"]), filename=_basename(entry["filename"]),
                                kind=entry["kind"], expected_class=None if expected is None else _label(expected),
                                sha256=_sha256(entry["sha256"]), url=_demo_url(entry["url"], revision)))
    kinds = [asset.kind for asset in assets]
    if len(kinds) != len(set(kinds)) or set(kinds) != {"class-names", "sample-image"}:
        raise SourceError("demo metadata must pin exactly one asset per kind")
    return tuple(assets)


def load_gallery_metadata(path: str | Path = HERE / "resnet18-gallery-source.json") -> tuple[GalleryImage, ...]:
    value = _read_record(path, {"approved_host", "format", "images", "magic"}, "NPU_RESNET18_GALLERY_SOURCE",
                         "gallery")
    host = value["approved_host"]
    if not isinstance(host, str) or not host:
        raise SourceError("approved_host must be a non-empty string")
    entries = value["images"]
    if not isinstance(entries, list) or not entries:
        raise SourceError("gallery metadata lists no images")
    images = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"bytes", "expected_class", "filename", "license",
                                                         "sha256", "url"}:
            raise SourceError("image record fields do not match the contract")
        images.append(GalleryImage(filename=_basename(entry["filename"]), expected_class=_label(entry["expected_class"]),
                                   byte_length=_byte_length(entry["bytes"]), sha256=_sha256(entry["sha256"]),
                                   url=_gallery_url(entry["url"], host), approved_host=host,
                                   license=_license(entry["license"], {"attribution", "notes", "source_page", "spdx"})))
    names = [image.filename for image in images]
    if len(names) != len(set(names)):
        raise SourceError("gallery metadata repeats an image filename")
    return tuple(images)


# Downloads

def _open_with_retry(request: Request, opener: Callable, timeout: float, sleep=time.sleep):
    """Open a pinned URL, backing off on rate limits and transient failures."""
    delay = 2.0
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return opener(request, timeout=timeout)
        except HTTPError as error:
            if error.code not in RETRY_STATUSES or attempt == ATTEMPTS:
                raise SourceError(f"{request.full_url} answered HTTP {error.code} {error.reason}") from error
            retry_after = error.headers.get("Retry-After") if error.headers else None
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
        except URLError as error:
            if attempt == ATTEMPTS:
                raise SourceError(f"{request.full_url} is unreachable: {error.reason}") from error
            wait = delay
        print(f"INFO: retrying {request.full_url} in {wait:.0f}s (attempt {attempt})")
        sleep(min(wait, 60.0))
        delay *= 2
    raise AssertionError("unreachable")


def _publish(url: str, destination: Path, byte_length: int, sha256: str, check_url: Callable[[str], object],
             opener: Callable, timeout: float) -> Path:
    """Stream ``url`` to a temporary file, verify length and digest, then rename it."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists() or temporary.exists():
        raise SourceError(f"destination or temporary file already exists: {destination}")
    digest, count = hashlib.sha256(), 0
    try:
        with _open_with_retry(Request(url, headers={"User-Agent": USER_AGENT}), opener, timeout) as response:
            check_url(response.geturl())
            with temporary.open("xb") as stream:
                while chunk := response.read(CHUNK):
                    count += len(chunk)
                    if count > byte_length:
                        raise SourceError(f"download size exceeds pinned metadata: {destination.name}")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if count != byte_length:
            raise SourceError(f"download size {count} differs from pinned {byte_length}: {destination.name}")
        if digest.hexdigest() != sha256:
            raise SourceError(f"download digest differs from pinned metadata: {destination.name}")
        os.replace(temporary, destination)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        if isinstance(error, SourceError):
            raise
        raise SourceError(f"download of {destination.name} failed: {error}") from error
    return destination


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(metadata_path: str | Path, destination_dir: str | Path, *, opener: Callable = urlopen,
                   timeout: float = 60.0) -> Path:
    """Stream, authenticate, and atomically publish the pinned checkpoint."""
    source = load_source_metadata(metadata_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    return _publish(source.url, destination_dir / source.filename, source.byte_length, source.sha256,
                    lambda url: _model_url(url, source.filename), opener, timeout)


def download_demo_assets(metadata_path: str | Path, destination_dir: str | Path, *, opener: Callable = urlopen,
                         timeout: float = 60.0) -> list[Path]:
    """Stream, authenticate, and atomically publish every pinned demo asset."""
    assets = load_demo_metadata(metadata_path)
    revision = json.loads(Path(metadata_path).read_bytes())["revision"]
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    return [_publish(asset.url, destination_dir / asset.filename, asset.byte_length, asset.sha256,
                     lambda url: _demo_url(url, revision), opener, timeout) for asset in assets]


def download_gallery(metadata_path: str | Path, model_dir: str | Path, *, opener: Callable = urlopen,
                     timeout: float = 120.0) -> list[GalleryImage]:
    """Publish every gallery image not already present; an existing file must match its pin."""
    images = load_gallery_metadata(metadata_path)
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise SourceError(f"model workspace is missing: {model_dir}")
    for image in images:
        destination = model_dir / image.filename
        if destination.is_file():
            if _file_sha256(destination) != image.sha256:
                raise SourceError(f"existing file differs from pinned digest, refusing to overwrite: {destination}")
            print(f"SKIP: already verified {destination}")
            continue
        _publish(image.url, destination, image.byte_length, image.sha256,
                 lambda url, host=image.approved_host: _gallery_url(url, host), opener, timeout)
        print(f"PASS: downloaded and verified {destination}")
    return list(images)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", choices=("model", "demo", "gallery"))
    parser.add_argument("--metadata", type=Path, help="pin record (default: the one beside this module)")
    parser.add_argument("--output-dir", "--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)
    metadata = args.metadata or HERE / {"model": "resnet18-model-source.json", "demo": "resnet18-demo-source.json",
                                        "gallery": "resnet18-gallery-source.json"}[args.source]
    try:
        if args.source == "model":
            print(f"PASS: verified pretrained ResNet-18 checkpoint at {download_model(metadata, args.output_dir)}")
        elif args.source == "demo":
            for path in download_demo_assets(metadata, args.output_dir):
                print(f"PASS: verified pinned demo asset at {path}")
            print("INFO: demo assets stay in the ignored model workspace; they are not committed")
        else:
            images = download_gallery(metadata, args.output_dir)
            print(f"PASS [demo-gallery]: {len(images)} pinned images verified")
            for image in images:
                print(f"  {image.filename:<24} {image.expected_class:<12} "
                      f"{image.license['spdx']}  {image.license['attribution']}")
    except SourceError as error:
        print(f"{args.source} download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
