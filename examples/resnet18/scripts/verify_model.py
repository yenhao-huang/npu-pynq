"""Validate the real converted model against an independent integer reference."""

from __future__ import annotations

import argparse
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

from src.export.torchvision_resnet18 import (
    REAL_MODEL_HOST_EVIDENCE_TYPE,
    compare_integer_captures,
)
from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES
from src.runtime.model import NPUModelRuntime, load_model_package
from src.test.model.quantized_graph_reference import (
    execute_quantized_graph_reference,
)


CAPTURE_NAMES = ("stem.relu", "layer1.1.relu", "logits")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _canonical_write_new(path: Path, value: object) -> None:
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


def validate_model(
    *,
    model_prefix: Path,
    checkpoint_path: Path,
    output_path: Path,
    software_timeout: float = 300.0,
) -> dict[str, object]:
    """Run reference and production host paths, then publish bound evidence."""

    prefix = model_prefix.resolve()
    checkpoint = checkpoint_path.resolve()
    output = output_path.resolve()
    manifest_path = Path(str(prefix) + ".npu.json")
    payload_path = Path(str(prefix) + ".npu.bin")
    input_path = prefix.with_name(f"{prefix.name}.validation.npy")
    conversion_path = prefix.with_name(f"{prefix.name}.conversion.json")
    required = (checkpoint, manifest_path, payload_path, input_path, conversion_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"validation input is missing: {missing[0]}")
    if output.exists():
        raise ValueError(f"validation output already exists: {output.name}")

    conversion = _read_json(conversion_path)
    if conversion.get("magic") != "NPU_RESNET18_CONVERSION":
        raise ValueError("conversion provenance magic is invalid")
    if conversion.get("evidence_type") != REAL_MODEL_HOST_EVIDENCE_TYPE:
        raise ValueError("conversion provenance evidence type is invalid")
    checkpoint_record = conversion.get("checkpoint")
    model_record = conversion.get("model")
    input_record = conversion.get("input")
    if not all(isinstance(item, dict) for item in (checkpoint_record, model_record, input_record)):
        raise ValueError("conversion provenance records are incomplete")
    digest_checks = (
        (checkpoint, checkpoint_record.get("sha256")),
        (manifest_path, model_record.get("manifest_sha256")),
        (payload_path, model_record.get("payload_sha256")),
        (input_path, input_record.get("sha256")),
    )
    for path, expected in digest_checks:
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"conversion provenance digest mismatch: {path.name}")

    model = load_model_package(manifest_path)
    if tuple(model.graph.outputs) != CAPTURE_NAMES:
        raise ValueError("converted graph does not expose required acceptance captures")
    validation_input = np.load(input_path, allow_pickle=False)
    if validation_input.dtype != np.int8 or validation_input.shape != (1, 224, 224, 3):
        raise ValueError("validation input must be signed INT8 (1, 224, 224, 3)")

    expected = execute_quantized_graph_reference(
        model.graph, model.constants, {"input": validation_input}
    )
    result = NPUModelRuntime(HostMatrixBackend(), model).run(
        {"input": validation_input}, software_timeout=software_timeout
    )
    compare_integer_captures(expected, result.outputs)

    evidence: dict[str, object] = {
        "captures": {
            name: {
                "dtype": "int8",
                "sha256": _array_sha256(result.outputs[name]),
                "shape": list(result.outputs[name].shape),
            }
            for name in CAPTURE_NAMES
        },
        "conversion": {
            "bytes": conversion_path.stat().st_size,
            "sha256": _sha256(conversion_path),
        },
        "evidence_type": REAL_MODEL_HOST_EVIDENCE_TYPE,
        "format": {"major": 1, "minor": 0},
        "input": {"bytes": input_path.stat().st_size, "sha256": _sha256(input_path)},
        "integer_reference": "independent-vectorized-v1",
        "magic": "NPU_RESNET18_ACCEPTANCE",
        "model": {
            "manifest_sha256": _sha256(manifest_path),
            "payload_sha256": _sha256(payload_path),
        },
        "result": "pass",
        "runtime": {
            "backend": "host-matrix",
            "mac_count": result.metrics.mac_count,
            "physical_board": False,
            "physical_jobs": result.metrics.physical_jobs,
        },
        "source": {
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        },
    }
    _canonical_write_new(output, evidence)
    return evidence


def main() -> int:
    model_dir = REPOSITORY_ROOT / "examples" / "resnet18" / "model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-prefix", type=Path, default=model_dir / "resnet18")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_dir / "resnet18-f37072fd.pth",
    )
    parser.add_argument("--output", type=Path, default=model_dir / "acceptance.json")
    parser.add_argument("--software-timeout", type=float, default=300.0)
    arguments = parser.parse_args()
    validate_model(
        model_prefix=arguments.model_prefix,
        checkpoint_path=arguments.checkpoint,
        output_path=arguments.output,
        software_timeout=arguments.software_timeout,
    )
    print(f"PASS [{REAL_MODEL_HOST_EVIDENCE_TYPE}]: {arguments.output.resolve()}")
    print("INFO: this is not physical PYNQ-Z1 evidence or ImageNet accuracy evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
