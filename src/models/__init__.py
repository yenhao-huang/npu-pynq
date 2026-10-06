"""Model frontends. Each model lives in its own package and builds its graph
from src/compiler/ops only; models never import each other."""

from importlib import import_module

MODELS = {
    "smollm": "src.models.smollm.model",
    "qwen3": "src.models.qwen3.model",
    "qwen3_moe": "src.models.qwen3_moe.model",
    "qwen3_5": "src.models.qwen3_5.model",
    "resnet18": "src.models.resnet18.model",
}


def load(name: str):
    return import_module(MODELS[name])
