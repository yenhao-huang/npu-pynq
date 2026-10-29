"""Read ONNX sources without changing their dtype or quantization:
initializers, the pin check, and the ONNX-MLIR import of the graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class OnnxWeights:
    """Read initializer arrays without changing their dtype or quantization."""
    def __init__(self, path: str | Path):
        import onnx
        from onnx import numpy_helper
        self.model = onnx.load(path)
        self.arrays = {t.name:numpy_helper.to_array(t) for t in self.model.graph.initializer}

    def keys(self):
        return list(self.arrays)

    def __contains__(self, name):
        return name in self.arrays

    def get(self, name):
        return self.arrays[name]

    def __getitem__(self, name):
        return self.get(name)


def verify_pin(onnx_path: str | Path, pin_file: str | Path, name: str = 'model_int8.onnx') -> str:
    """The file's SHA-256, after checking it is the one ``pin_file`` names."""
    digest = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
    pinned = json.loads(Path(pin_file).read_text())
    if digest != pinned['files'][name]['sha256']:
        raise ValueError('This frontend requires the pinned ONNX artifact')
    return digest


def import_graph(onnx_path: str | Path, import_dir=None) -> tuple[dict, dict, Path, dict]:
    """Import (or verify a prepared import of) one ONNX main graph through
    ONNX-MLIR: (graph, report, import directory, nodes by ONNX node name)."""
    from src.export.compilers.onnx_mlir import adapter

    imported, report, directory = adapter.load_imported(onnx_path, import_dir)
    if len(imported['functions']) != 1:
        raise ValueError('ONNX import requires one main graph')
    nodes = {node['attributes'].get('onnx_node_name'): node
             for node in imported['functions'][0]['nodes']
             if node['attributes'].get('onnx_node_name')}
    return imported, report, directory, nodes


def imported_node(nodes: dict, name: str, operation: str) -> dict:
    """The imported node ``name``, checked to be ``operation``."""
    node = nodes.get(name)
    if node is None or node['op'] != operation:
        raise ValueError('Imported ONNX operation mismatch: ' + name)
    return node
