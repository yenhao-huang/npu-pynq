"""Model architectures, each exposing ``ARCH``, an ``Architecture``
(base.py). A model is one module (qwen3.py) or, when it carries more than
its architecture, one directory (smollm/, resnet18/: arch.py plus source
pins and download scripts). Models import base.py, the stage packages and
their own directory; they never import each other."""

from importlib import import_module

ARCHS = {
    "smollm": "src.export.model_arch.smollm.arch",
    "smollm_onnx": "src.export.model_arch.smollm.onnx",
    "qwen3": "src.export.model_arch.qwen3",
    "qwen3_moe": "src.export.model_arch.qwen3_moe",
    "qwen3_5": "src.export.model_arch.qwen3_5",
    "resnet18": "src.export.model_arch.resnet18.arch",
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
