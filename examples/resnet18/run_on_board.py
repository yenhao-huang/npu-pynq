"""Run digest-bound ResNet-18 acceptance on a physical PYNQ-Z1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.resnet18.package_example import validate_workspace
from src.runtime import NPURuntime, NPUModelRuntime, load_model_package, load_pynq_runtime
from src.runtime.verify_overlay import verify_artifacts


PASS_MARKER = "PASS [physical-pynq-z1]: real ResNet-18 board acceptance"


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


def _write_new(path: Path, value: object) -> None:
    if path.exists() or not path.parent.is_dir():
        raise ValueError("board evidence must be a new file in an existing directory")
    data = (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_board(
    *,
    model_dir: Path,
    source_metadata_path: Path,
    artifact_dir: Path,
    evidence_path: Path,
    software_timeout: float,
) -> dict[str, object]:
    validate_workspace(model_dir, source_metadata_path)
    model_dir = model_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    overlay = verify_artifacts(artifact_dir)
    physical = load_pynq_runtime(artifact_dir / "npu_matrix.bit")
    if not isinstance(physical, NPURuntime):
        raise RuntimeError("physical evidence requires the public NPURuntime")
    model = load_model_package(model_dir / "resnet18.npu.json")
    validation_input = np.load(
        model_dir / "resnet18.validation.npy", allow_pickle=False
    )
    acceptance = json.loads(
        (model_dir / "acceptance.json").read_text(encoding="utf-8")
    )
    result = NPUModelRuntime(physical, model).run(
        {"input": validation_input}, software_timeout=software_timeout
    )
    for name in model.graph.outputs:
        expected = acceptance["captures"][name]["sha256"]
        if _array_sha256(result.outputs[name]) != expected:
            raise RuntimeError(f"physical capture differs: {name}")
    evidence: dict[str, object] = {
        "captures": {
            name: {
                "sha256": _array_sha256(result.outputs[name]),
                "shape": list(result.outputs[name].shape),
            }
            for name in model.graph.outputs
        },
        "evidence_type": "physical-pynq-z1",
        "format": {"major": 1, "minor": 0},
        "host_acceptance_sha256": _sha256(model_dir / "acceptance.json"),
        "magic": "NPU_RESNET18_BOARD_ACCEPTANCE",
        "model_manifest_sha256": _sha256(model_dir / "resnet18.npu.json"),
        "overlay": {
            "bit_sha256": overlay["bit"]["sha256"],
            "hwh_sha256": overlay["hwh"]["sha256"],
            "source_commit": overlay["source_commit"],
            "target_part": overlay["target_part"],
        },
        "result": "pass",
        "runtime": {
            "abi_major": physical.abi_major,
            "capabilities": physical.capabilities,
            "mac_count": result.metrics.mac_count,
            "physical_jobs": result.metrics.physical_jobs,
            "physical_limits": [physical.max_m, physical.max_n, physical.max_k],
        },
    }
    _write_new(evidence_path.resolve(), evidence)
    return evidence


def main() -> int:
    example_root = REPOSITORY_ROOT / "examples" / "resnet18"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=example_root / "model")
    parser.add_argument(
        "--source-metadata", type=Path, default=example_root / "model-source.json"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--software-timeout", type=float, default=86400.0)
    arguments = parser.parse_args()
    try:
        run_board(
            model_dir=arguments.model_dir,
            source_metadata_path=arguments.source_metadata,
            artifact_dir=arguments.artifact_dir,
            evidence_path=arguments.evidence,
            software_timeout=arguments.software_timeout,
        )
    except Exception as error:
        print(f"physical PYNQ-Z1 acceptance failed: {error}", file=sys.stderr)
        return 1
    print(PASS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
