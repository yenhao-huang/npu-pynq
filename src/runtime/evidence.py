"""Canonical JSON encoding and atomic publication for runtime evidence.

Acceptance certificates and benchmark records must be byte-reproducible and
must never be observed half-written, because both are compared across runs and
across commits. Both concerns live here so the two producers cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def canonical_json(value: Mapping[str, Any], error_type: type[Exception]) -> bytes:
    """Encode ``value`` as sorted, compact, NaN-free UTF-8 JSON with a newline."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise error_type(f"evidence is not canonical JSON: {error}") from error
    return (encoded + "\n").encode("utf-8")


def publish_atomic(path: Path, data: bytes, error_type: type[Exception]) -> None:
    """Replace ``path`` with ``data`` so no reader ever sees a partial file."""

    if not path.name or not path.parent.is_dir():
        raise error_type("evidence parent directory must exist")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise error_type(f"evidence publication failed: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
