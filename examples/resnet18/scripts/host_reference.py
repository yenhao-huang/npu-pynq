"""Shared host-side validation helpers for the ResNet-18 example scripts.

Nothing here is physical-board evidence: the matrix backend runs in NumPy on
the conversion host and every marker it supports is a host marker.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES


class HostMatrixBackend:
    """Vectorized host matrix backend; never physical-board evidence."""

    abi_major = REQUIRED_ABI_MAJOR
    capabilities = REQUIRED_CAPABILITIES
    max_m = 256
    max_n = 512
    max_k = 4608

    def run(self, matrix_a: np.ndarray, matrix_b: np.ndarray, **_timeouts) -> np.ndarray:
        return (
            np.asarray(matrix_a, dtype=np.int32)
            @ np.asarray(matrix_b, dtype=np.int32)
        )


def file_sha256(path: Path) -> str:
    """Digest one file without holding it entirely in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Digest dtype, shape, and C-order bytes so captures compare exactly."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Read one JSON object, rejecting anything that is not a mapping."""

    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def canonical_write_new(path: Path, value: object) -> None:
    """Publish one canonical JSON document write-once and atomically."""

    path = Path(path)
    if path.exists():
        raise ValueError(f"validation output already exists: {path.name}")
    encoded = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            raise ValueError(f"validation output already exists: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_new_array(path: Path, array: np.ndarray) -> None:
    """Publish one .npy tensor write-once and atomically."""

    path = Path(path)
    if path.exists():
        raise ValueError(f"tensor output already exists: {path.name}")
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            raise ValueError(f"tensor output already exists: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
