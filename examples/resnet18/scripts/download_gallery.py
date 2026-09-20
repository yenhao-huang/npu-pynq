"""Download the pinned demo gallery photographs into the ignored model workspace.

The gallery gives the live-demo notebook a few ready-made pictures so a
demonstration does not depend on someone having a suitable photo to hand. Every
image is pinned by URL, byte length, and SHA-256, carries its licence and
attribution, and is verified before it is published. Nothing here is an ImageNet
dataset image, and nothing here is committed to the repository.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

APPROVED_PATH_PREFIX = "/wikipedia/commons/"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
METADATA_FIELDS = {"approved_host", "format", "images", "magic"}
IMAGE_FIELDS = {"bytes", "expected_class", "filename", "license", "sha256", "url"}
LICENSE_FIELDS = {"attribution", "notes", "source_page", "spdx"}
MAGIC = "NPU_RESNET18_GALLERY_SOURCE"
USER_AGENT = "npu-in-pynq-resnet18-demo/1"
READ_CHUNK = 256 * 1024


class GalleryAssetError(ValueError):
    """Raised when gallery metadata or a download fails its pinned contract."""


@dataclass(frozen=True)
class GalleryImage:
    filename: str
    expected_class: str
    byte_length: int
    sha256: str
    url: str
    approved_host: str
    license: dict[str, str]


def _object_without_duplicates(pairs):
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise GalleryAssetError(f"duplicate metadata field: {key}")
        seen[key] = value
    return seen


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validated_url(value: object, approved_host: str) -> str:
    if not isinstance(value, str):
        raise GalleryAssetError("image url must be a string")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != approved_host
        or not parsed.path.startswith(APPROVED_PATH_PREFIX)
        or parsed.query
        or parsed.fragment
    ):
        raise GalleryAssetError(f"image url is not an approved pinned HTTPS URL: {value}")
    return value


def load_gallery_metadata(path: str | Path) -> tuple[GalleryImage, ...]:
    """Parse and fully validate the pinned gallery metadata."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise GalleryAssetError(f"gallery metadata cannot be read: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicates)
    except ValueError as error:
        raise GalleryAssetError(f"gallery metadata is invalid: {error}") from error
    if not isinstance(value, dict) or set(value) != METADATA_FIELDS:
        raise GalleryAssetError("gallery metadata fields do not match the contract")
    if _canonical_json(value) != raw:
        raise GalleryAssetError("gallery metadata is not canonical")
    if value["magic"] != MAGIC:
        raise GalleryAssetError("gallery metadata magic is unrecognised")
    if value["format"] != {"major": 1, "minor": 0}:
        raise GalleryAssetError("gallery metadata format is unsupported")
    approved_host = value["approved_host"]
    if not isinstance(approved_host, str) or not approved_host:
        raise GalleryAssetError("approved_host must be a non-empty string")

    entries = value["images"]
    if not isinstance(entries, list) or not entries:
        raise GalleryAssetError("gallery metadata lists no images")

    images: list[GalleryImage] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != IMAGE_FIELDS:
            raise GalleryAssetError("image record fields do not match the contract")
        filename = entry["filename"]
        if (
            not isinstance(filename, str)
            or filename != Path(filename).name
            or filename in {"", ".", ".."}
            or filename.startswith(".")
        ):
            raise GalleryAssetError("image filename must be a safe basename")
        byte_length = entry["bytes"]
        if not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length <= 0:
            raise GalleryAssetError("image bytes must be a positive integer")
        digest = entry["sha256"]
        if not isinstance(digest, str) or DIGEST_PATTERN.fullmatch(digest) is None:
            raise GalleryAssetError("image sha256 must be a lowercase full digest")
        expected_class = entry["expected_class"]
        if not isinstance(expected_class, str) or not expected_class.strip():
            raise GalleryAssetError("expected_class must be a non-empty label")
        licence = entry["license"]
        if not isinstance(licence, dict) or set(licence) != LICENSE_FIELDS:
            raise GalleryAssetError("image licence metadata is incomplete")
        if not all(isinstance(item, str) and item.strip() for item in licence.values()):
            raise GalleryAssetError("image licence fields must be non-empty strings")
        images.append(
            GalleryImage(
                filename=filename,
                expected_class=expected_class,
                byte_length=byte_length,
                sha256=digest,
                url=_validated_url(entry["url"], approved_host),
                approved_host=approved_host,
                license=dict(licence),
            )
        )

    names = [image.filename for image in images]
    if len(names) != len(set(names)):
        raise GalleryAssetError("gallery metadata repeats an image filename")
    return tuple(images)


def _download_one(image: GalleryImage, destination: Path) -> None:
    digest = hashlib.sha256()
    received = 0
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            request = Request(image.url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=120) as response:  # noqa: S310 - host is pinned
                _validated_url(response.geturl(), image.approved_host)
                while True:
                    chunk = response.read(READ_CHUNK)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > image.byte_length:
                        raise GalleryAssetError(
                            f"download exceeds pinned length: {image.filename}"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
            stream.flush()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if received != image.byte_length:
            raise GalleryAssetError(
                f"download length {received} differs from pinned {image.byte_length}: "
                f"{image.filename}"
            )
        if digest.hexdigest() != image.sha256:
            raise GalleryAssetError(f"download digest differs from pinned: {image.filename}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_status(destination: Path, image: GalleryImage) -> str:
    """Return "absent", "verified", or "differs" for what is already on disk."""

    if not destination.is_file():
        return "absent"
    return "verified" if _file_sha256(destination) == image.sha256 else "differs"


def download_gallery(metadata_path: Path, model_dir: Path) -> list[GalleryImage]:
    images = load_gallery_metadata(metadata_path)
    if not model_dir.is_dir():
        raise GalleryAssetError(f"model workspace is missing: {model_dir}")
    published: list[GalleryImage] = []
    for image in images:
        destination = model_dir / image.filename
        status = _existing_status(destination, image)
        if status == "differs":
            raise GalleryAssetError(
                f"existing file differs from pinned digest, refusing to overwrite: "
                f"{destination}"
            )
        if status == "verified":
            print(f"SKIP: already verified {destination}")
        else:
            _download_one(image, destination)
            print(f"PASS: downloaded and verified {destination}")
        published.append(image)
    return published


def main() -> int:
    example_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=example_root / "gallery-source.json")
    parser.add_argument("--model-dir", type=Path, default=example_root / "model")
    arguments = parser.parse_args()
    try:
        published = download_gallery(arguments.metadata, arguments.model_dir)
    except GalleryAssetError as error:
        print(f"gallery download failed: {error}", file=sys.stderr)
        return 1
    print(f"PASS [demo-gallery]: {len(published)} pinned images verified")
    for image in published:
        print(
            f"  {image.filename:<24} {image.expected_class:<12} "
            f"{image.license['spdx']}  {image.license['attribution']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
