"""Model architectures: one module per model, each exposing ``ARCH``, an
``Architecture`` (base.py). Modules import only base.py and the stage
packages; they never import each other."""

from importlib import import_module

ARCHS = {
    "smollm": "src.export.model_arch.smollm",
    "smollm_onnx": "src.export.model_arch.smollm_onnx",
    "qwen3": "src.export.model_arch.qwen3",
    "qwen3_moe": "src.export.model_arch.qwen3_moe",
    "qwen3_5": "src.export.model_arch.qwen3_5",
    "resnet18": "src.export.model_arch.resnet18",
}

# (model name, source format) -> architecture, where one model has several.
BY_FORMAT = {("smollm", "onnx"): "smollm_onnx"}


def resolve(model: str, fmt: str) -> str:
    """The architecture that reads ``model`` from a source in ``fmt``."""
    if model not in ARCHS:
        raise KeyError(model)
    return BY_FORMAT.get((model, fmt), model)


def get(name: str):
    return import_module(ARCHS[name]).ARCH


def module(name: str):
    return import_module(ARCHS[name])
