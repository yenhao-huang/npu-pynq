"""The load stage: find a model source and read it without changing any
number. Safetensors checkpoints, ONNX initializers and the pinned ONNX
preparation live here; detect.py decides which one a path is."""

from __future__ import annotations

from pathlib import Path

from src.export.load.onnx import OnnxWeights
from src.export.load.safetensors import SafeTensors


def load_weights(path: str | Path):
    """Use the same initializer interface for ONNX and safetensors sources."""
    path = Path(path)
    return OnnxWeights(path) if path.suffix == '.onnx' else SafeTensors(path)
