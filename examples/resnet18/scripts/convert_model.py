"""Convert the pinned TorchVision checkpoint into a Phase 2A NPU package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.export.torchvision_resnet18 import convert_checkpoint


def main() -> int:
    example_root = Path(__file__).resolve().parents[1]
    model_dir = example_root / "model"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=model_dir / "resnet18-f37072fd.pth",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=model_dir / "resnet18",
    )
    arguments = parser.parse_args()
    result = convert_checkpoint(arguments.checkpoint, arguments.output_prefix)
    print(f"PASS: converted pretrained ResNet-18 to {result.package.manifest_path}")
    print(f"PASS: real-model host conversion evidence at {result.provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
