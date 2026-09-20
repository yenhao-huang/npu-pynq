"""Download the pinned INT8 calibration images, fail-closed.

Activation scales are derived from a fixed set of real ImageNet photographs,
pinned by URL, byte length, and SHA-256 in
``examples/resnet18/calibration-source.json``. This script fetches them into the
ignored ``model/calibration/`` workspace and verifies every one. It never
overwrites an existing verified file and rejects any host, length, or digest
mismatch.

No image is committed to the repository; only the pinned metadata is. Users
must verify their own right to use ImageNet-derived data (see the licence field
in the source file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


MAGIC = "NPU_RESNET18_CALIBRATION_SOURCE"
APPROVED_HOST = "raw.githubusercontent.com"


class CalibrationDownloadError(RuntimeError):
    """The pinned calibration metadata or a downloaded image is untrusted."""


def _load_source(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationDownloadError(f"cannot read calibration source: {error}") from error
    if not isinstance(doc, dict) or doc.get("magic") != MAGIC:
        raise CalibrationDownloadError("calibration source has the wrong magic")
    images = doc.get("images")
    if not isinstance(images, list) or not images:
        raise CalibrationDownloadError("calibration source lists no images")
    return doc


def _fetch(url: str, timeout: float = 120.0) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != APPROVED_HOST:
        raise CalibrationDownloadError(f"refusing non-approved URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def main() -> int:
    example_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=example_root / "calibration-source.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=example_root / "model" / "calibration",
    )
    arguments = parser.parse_args()

    doc = _load_source(arguments.source)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"INFO: {doc['count']} pinned calibration images "
          f"({doc.get('calibration_id', 'unknown id')})", flush=True)

    verified = 0
    for entry in doc["images"]:
        filename = entry["filename"]
        target = arguments.output_dir / filename
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
                verified += 1
                continue
            raise CalibrationDownloadError(f"existing file fails digest: {filename}")
        data = _fetch(entry["url"])
        if len(data) != entry["bytes"]:
            raise CalibrationDownloadError(
                f"{filename}: length {len(data)} != pinned {entry['bytes']}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise CalibrationDownloadError(f"{filename}: digest mismatch")
        target.write_bytes(data)
        verified += 1

    print(f"PASS: {verified} calibration images verified in {arguments.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
