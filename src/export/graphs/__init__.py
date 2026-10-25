"""Model frontends. Each model lives in its own package and builds its graph
from src/export/compilers/mlir/ops only; models never import each other."""

from importlib import import_module

MODELS = {
    "smollm": "src.export.graphs.smollm.model",
    "qwen3": "src.export.graphs.qwen3.model",
    "qwen3_moe": "src.export.graphs.qwen3_moe.model",
    "qwen3_5": "src.export.graphs.qwen3_5.model",
    "resnet18": "src.export.graphs.resnet18.model",
}


def load(name: str):
    return import_module(MODELS[name])
