"""Validate ResNet demo inputs without loading or executing hardware."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.resnet18.package_example import validate_workspace
from src.deprecate.runtime.model import load_model_package
from src.runtime.verify_overlay import verify_artifacts


def validate_demo(artifact_dir: Path, model_dir: Path) -> dict:
    overlay = verify_artifacts(artifact_dir)
    array_size = overlay.get("array_size")
    if array_size not in (8, 16):
        raise ValueError("ResNet demo requires an 8x8 or 16x16 overlay")
    validate_workspace(model_dir, ROOT / "examples/resnet18/model-source.json")
    model = load_model_package(model_dir / "resnet18.npu.json")
    return {
        "array_size": array_size,
        "artifact_source_commit": overlay["source_commit"],
        "model_commands": len(model.graph.commands),
        "model_outputs": list(model.graph.outputs),
        "hardware_executed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path,
                        default=ROOT / "build/vivado/npu_matrix_16x16/artifacts")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "examples/resnet18/model")
    args = parser.parse_args()
    print(json.dumps(validate_demo(args.artifact_dir, args.model_dir), indent=2))
    print("PASS: ResNet demo inputs verified; hardware not executed")
