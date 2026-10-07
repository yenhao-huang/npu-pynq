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

from src.compiler.export import ModelGraph
from src.compiler.mlir_builder import ModuleBuilder, memref_type
from src.compiler.ops import conv as C, core, linear as L
from src.compiler.ops.context import Ctx

DEFAULT_SOURCE = "examples/resnet18/model/resnet18-f37072fd.pth"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class ResNetOptions:
    bits: int = 8


def _find_source(source: str) -> Path:
    p = Path(source)
    if p.exists():
        return p
    repo = Path(__file__).resolve().parents[3]
    for root in (repo, repo.parent.parent / "npu-pynq"):
        if (root / source).exists():
            return root / source
    raise FileNotFoundError(f"{source}: run examples/resnet18/scripts/download_model.py first")


def folded_convs(state: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (weight [out, kh*kw*in] in (ky, kx, c) order, bias [out])."""
    out = {}
    convs = [k[:-len(".weight")] for k in state if k.endswith(".weight") and state[k].ndim == 4]
    for name in convs:
        w = state[name + ".weight"].astype(np.float64)
        bn = name.replace("conv", "bn") if "conv" in name else name[:-1] + "1"  # downsample.0 -> downsample.1
        g, beta = state[bn + ".weight"], state[bn + ".bias"]
        mean, var = state[bn + ".running_mean"], state[bn + ".running_var"]
        scale = g / np.sqrt(var + 1e-5)
        w = w * scale[:, None, None, None]
        b = beta - mean * scale
        out[name] = (w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1).astype(np.float32), b.astype(np.float32))
    return out


def build(source: str = DEFAULT_SOURCE, opts=None):
    import torch

    bits = getattr(opts, "bits", 8) if opts is not None else 8
    path = _find_source(source)
    state = {k: v.detach().numpy() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    convs = folded_convs(state)
    mod = ModuleBuilder()
    f = mod.func("forward", [("ws", "memref<?xi8>"), ("image", memref_type([224, 224, 3], "f32")),
                             ("logits", memref_type([1, 1000], "f32"))])
    graph = ModelGraph("resnet18", mod)
    ctx = Ctx(f, graph, f.arg("ws"))
    b = f

    def conv(x, h, w, c, name, k, stride, pad, relu):
        wt, bias = convs[name]
        cols, oh, ow = C.im2col(ctx, x, h, w, c, k, stride, pad)
        y = L.linear(ctx, cols, name, L.quantize_weight(wt, bits), bias)
        return (core.relu(b, y) if relu else y), oh, ow

    x = b.reshape(b.to_tensor(f.arg("image")), [224 * 224, 3])
    x, h, w = conv(x, 224, 224, 3, "conv1", 7, 2, 3, True)
    x, h, w = C.maxpool(ctx, x, h, w, 64, 3, 2, 1)
    c = 64
    for stage, out_c in enumerate((64, 128, 256, 512), start=1):
        for block in range(2):
            p = f"layer{stage}.{block}."
            stride = 2 if (block == 0 and stage > 1) else 1
            y, oh, ow = conv(x, h, w, c, p + "conv1", 3, stride, 1, True)
            y, oh, ow = conv(y, oh, ow, out_c, p + "conv2", 3, 1, 1, False)
            if p + "downsample.0" in convs:
                skip, _, _ = conv(x, h, w, c, p + "downsample.0", 1, stride, 0, False)
            else:
                skip = x
            x = core.relu(b, core.add(b, y, skip))
            h, w, c = oh, ow, out_c
    pooled = C.global_avgpool(ctx, x)
    fc = L.quantize_weight(state["fc.weight"], bits)
    b.store(L.linear(ctx, pooled, "fc", fc, state["fc.bias"].astype(np.float32)), f.arg("logits"))
    f.emit("return")
    graph.entry_points = {"forward": ["weights", "image", "logits"]}
    graph.meta.update({"family": "resnet18", "source": str(source), "options": {"bits": bits},
                       "input": {"shape": [224, 224, 3], "layout": "NHWC", "mean": MEAN.tolist(), "std": STD.tolist()}})
    return graph, None


def preprocess(image) -> np.ndarray:
    """PIL image -> normalized NHWC float32 [224, 224, 3] (resize 256, center crop 224)."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    s = 256 / min(w, h)
    image = image.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    x = np.asarray(image.crop((left, top, left + 224, top + 224)), np.float32) / 255.0
    return np.ascontiguousarray((x - MEAN) / STD, np.float32)


def classify(model, image: np.ndarray) -> np.ndarray:
    logits = np.zeros((1, 1000), np.float32)
    model.call("forward", "weights", image, logits)
    return logits[0]
