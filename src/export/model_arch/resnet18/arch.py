"""TorchVision ResNet-18 (IMAGENET1K_V1) on the compiler.

BatchNorm is folded into each convolution; every convolution is im2col on
the CPU plus a W8A8 linear layer on the NPU (rows = output pixels); ReLU,
residual adds, pooling and the classifier's bias stay on the CPU. One entry
point:

    forward(weights, image: memref<224x224x3xf32> (normalized NHWC), logits: memref<1x1000xf32>)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.export.compilers.mlir.builder import ModuleBuilder, memref_type
from src.export.compilers.mlir.ops import conv as C, core, linear as L
from src.export.compilers.mlir.ops.context import Ctx
from src.export.ir.graph import ModelGraph
from src.export.ir.options import QuantRecipe, ShapeOptions
from src.export.ir.data import QuantizedModel, SourceModel
from src.export.model_arch.base import Architecture
from src.export.quantize.plan import AwqHook, WeightSpec

DEFAULT_SOURCE = "dist/resnet18/model/resnet18-f37072fd.pth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class ResNetOptions:
    bits: int = 8


def _find_source(source: str) -> Path:
    p = Path(source)
    if p.exists():
        return p
    repo = Path(__file__).resolve().parents[4]
    if (repo / source).is_file():
        return repo / source
    raise FileNotFoundError(f"{source}: run python -m src.export.model_arch.resnet18.download model first")


def fold(w, g, beta, mean, var) -> tuple[np.ndarray, np.ndarray]:
    """One convolution with its BatchNorm folded in: (weight [out, kh*kw*in]
    in (ky, kx, c) order, bias [out])."""
    w = w.astype(np.float64)
    scale = g / np.sqrt(var + 1e-5)
    w = w * scale[:, None, None, None]
    b = beta - mean * scale
    return w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1).astype(np.float32), b.astype(np.float32)


def conv_names(state) -> list[str]:
    return [k[:-len(".weight")] for k in state.keys() if k.endswith(".weight") and state[k].ndim == 4]


def batchnorm(name: str) -> str:
    return name.replace("conv", "bn") if "conv" in name else name[:-1] + "1"  # downsample.0 -> downsample.1


def folded_convs(state: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (weight [out, kh*kw*in] in (ky, kx, c) order, bias [out])."""
    out = {}
    for name in conv_names(state):
        bn = batchnorm(name)
        out[name] = fold(state[name + ".weight"], state[bn + ".weight"], state[bn + ".bias"],
                         state[bn + ".running_mean"], state[bn + ".running_var"])
    return out


class ResNet18(Architecture):
    name = "resnet18"
    title = "ResNet-18"
    formats = ("torch",)
    default_source = DEFAULT_SOURCE

    def default_options(self) -> ResNetOptions:
        return ResNetOptions()

    def validate(self, recipe: QuantRecipe, shape: ShapeOptions, awq: AwqHook | None) -> None:
        if recipe.bits not in (8, 32):
            raise ValueError("ResNet18 supports per-channel INT8 or float32 weights")

    def describe_options(self, q: QuantizedModel, shape: ShapeOptions) -> dict:
        return {"bits": q.recipe.bits}

    def load(self, source: str, fmt: str, layers: int | None, **kw) -> SourceModel:
        import torch

        path = _find_source(source)
        state = {k: v.detach().numpy()
                 for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        return SourceModel(self.name, fmt, source, path.parent, {}, state)

    def parse_config(self, src: SourceModel) -> None:
        return None

    def config(self, src: SourceModel) -> None:
        return None

    def weight_plan(self, src: SourceModel, cfg: None) -> list[WeightSpec]:
        specs = []
        for name in conv_names(src.tensors):
            bn = batchnorm(name)
            sources = (name + ".weight", bn + ".weight", bn + ".bias", bn + ".running_mean", bn + ".running_var")
            specs += [WeightSpec(name, sources, "linear", transform=lambda *a: fold(*a)[0]),
                      WeightSpec(name + ".bias", sources, "raw", transform=lambda *a: fold(*a)[1])]
        return specs + [WeightSpec("fc", ("fc.weight",), "linear"), WeightSpec("fc.bias", ("fc.bias",), "float")]

    def to_mlir(self, q: QuantizedModel, shape: ShapeOptions) -> ModelGraph:
        p = q.params
        mod = ModuleBuilder()
        f = mod.func("forward", [("ws", "memref<?xi8>"), ("image", memref_type([224, 224, 3], "f32")),
                                 ("logits", memref_type([1, 1000], "f32"))])
        graph = ModelGraph("resnet18", mod)
        ctx = Ctx(f, graph, f.arg("ws"))
        b = f

        def conv(x, h, w, c, name, k, stride, pad, relu):
            cols, oh, ow = C.im2col(ctx, x, h, w, c, k, stride, pad)
            y = L.linear(ctx, cols, name, p[name], p[name + ".bias"])
            return (core.relu(b, y) if relu else y), oh, ow

        x = b.reshape(b.to_tensor(f.arg("image")), [224 * 224, 3])
        x, h, w = conv(x, 224, 224, 3, "conv1", 7, 2, 3, True)
        x, h, w = C.maxpool(ctx, x, h, w, 64, 3, 2, 1)
        c = 64
        for stage, out_c in enumerate((64, 128, 256, 512), start=1):
            for block in range(2):
                pre = f"layer{stage}.{block}."
                stride = 2 if (block == 0 and stage > 1) else 1
                y, oh, ow = conv(x, h, w, c, pre + "conv1", 3, stride, 1, True)
                y, oh, ow = conv(y, oh, ow, out_c, pre + "conv2", 3, 1, 1, False)
                if pre + "downsample.0" in p:
                    skip, _, _ = conv(x, h, w, c, pre + "downsample.0", 1, stride, 0, False)
                else:
                    skip = x
                x = core.relu(b, core.add(b, y, skip))
                h, w, c = oh, ow, out_c
        pooled = C.global_avgpool(ctx, x)
        b.store(L.linear(ctx, pooled, "fc", p["fc"], p["fc.bias"]), f.arg("logits"))
        f.emit("return")
        graph.entry_points = {"forward": ["weights", "image", "logits"]}
        graph.meta.update({"family": "resnet18", "source": q.source.source, "options": self.describe_options(q, shape),
                           "classes": 1000,
                           "input": {"shape": [224, 224, 3], "layout": "NHWC", "resize_shorter": 256,
                                     "mean": MEAN.tolist(), "std": STD.tolist()}})
        return graph


def preprocess(image) -> np.ndarray:
    """PIL image -> normalized NHWC float32 [224, 224, 3] (resize 256, center crop 224)."""
    from src.runtime.core.image import preprocess as prepare
    return prepare(image, {"shape": [224, 224, 3], "layout": "NHWC", "resize_shorter": 256,
                           "mean": MEAN.tolist(), "std": STD.tolist()})


def classify(model, image: np.ndarray) -> np.ndarray:
    logits = np.zeros((1, 1000), np.float32)
    model.call("forward", "weights", image, logits)
    return logits[0]


ARCH = ResNet18()
