"""What the load stage hands to quantize: a model source, read and verified,
with every number exactly as the source stores it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceModel:
    arch: str                    # model_arch registry name, e.g. 'smollm' or 'smollm_onnx'
    format: str                  # 'safetensors' | 'onnx' | 'torch'
    source: str                  # as the caller named it; recorded as meta['source']
    root: Path                   # directory holding the tokenizer and config
    config: dict                 # raw config.json ({} when the source has none)
    tensors: Any                 # name -> np.ndarray: SafeTensors, OnnxWeights or a dict
    layers: int | None = None    # debug: keep only the first N decoder layers
    provenance: dict = field(default_factory=dict)   # digests, pins, imported ONNX graph
