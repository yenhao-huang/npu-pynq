"""Validate the real converted model against an independent integer reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from host_reference import (
    HostMatrixBackend,
    REPOSITORY_ROOT,
    array_sha256,
    canonical_write_new,
    file_sha256,
    read_json,
)
from src.export.torchvision_resnet18 import (
    REAL_MODEL_HOST_EVIDENCE_TYPE,
    compare_integer_captures,
)
from src.runtime.model import NPUModelRuntime, load_model_package
from src.test.model.quantized_graph_reference import (
    execute_quantized_graph_reference,
)


CAPTURE_NAMES = ("stem.relu", "layer1.1.relu", "logits")


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

    conversion = read_json(conversion_path)
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
        if not isinstance(expected, str) or file_sha256(path) != expected:
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
                "sha256": array_sha256(result.outputs[name]),
                "shape": list(result.outputs[name].shape),
            }
            for name in CAPTURE_NAMES
        },
        "conversion": {
            "bytes": conversion_path.stat().st_size,
            "sha256": file_sha256(conversion_path),
        },
        "evidence_type": REAL_MODEL_HOST_EVIDENCE_TYPE,
        "format": {"major": 1, "minor": 0},
        "input": {"bytes": input_path.stat().st_size, "sha256": file_sha256(input_path)},
        "integer_reference": "independent-vectorized-v1",
        "magic": "NPU_RESNET18_ACCEPTANCE",
        "model": {
            "manifest_sha256": file_sha256(manifest_path),
            "payload_sha256": file_sha256(payload_path),
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
            "sha256": file_sha256(checkpoint),
        },
    }
    canonical_write_new(output, evidence)
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
