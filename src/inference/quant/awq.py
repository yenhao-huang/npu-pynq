"""AWQ: activation-aware per-channel scaling before quantization.

For a linear layer y = x W^T with calibration inputs X, AWQ (Lin et al.,
2023) picks a per-input-channel scale s = mean|X|^alpha and computes
y = (x / s) (W s)^T: salient input channels get more weight precision. alpha
is searched on a grid to minimize the error of the *quantized* product. Here
the objective quantizes the activations as well (per-token INT8, what the NPU
consumes), so the same search also migrates activation outliers into the
weights.

The inverse scale is folded into whatever produces the layer's input, so the
transformed model is exactly equivalent before quantization:

    qkv      <- RMSNorm (ln1) weight
    gate_up  <- RMSNorm (ln2) weight
    o        <- v rows of qkv (one scale per kv head channel, shared by the
                query heads of a GQA group)
    down     <- up rows of gate_up (silu(gate) * up is linear in up)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.compiler.ops.linear import EPS

CALIB_TEXT = (
    "Large language models are trained on vast amounts of text to predict the next token. "
    "When deployed on small devices, their weights are often quantized to eight or four bits, "
    "which reduces memory traffic and lets integer arithmetic units do most of the work.\n"
    "The river wound through the valley, past fields of barley and old stone farmhouses, "
    "until it reached the town where the market was held every Saturday morning.\n"
    "import numpy as np\n\ndef softmax(x):\n    e = np.exp(x - x.max())\n    return e / e.sum()\n\n"
    "for i in range(10):\n    print(i, softmax(np.arange(i + 1)))\n"
    "Q: What is the boiling point of water at sea level? A: One hundred degrees Celsius.\n"
    "<|im_start|>user\nWrite a haiku about autumn.<|im_end|>\n<|im_start|>assistant\n"
    "Crimson leaves drifting\nwhispers of the cooling wind\nthe year exhales slow<|im_end|>\n"
)


@dataclass
class FoldSpec:
    """Where a model's linear inputs come from (HF module paths, ``{i}`` = layer)."""

    qkv_input: str      # a module whose input is the qkv projections' input
    o_input: str
    gate_up_input: str
    down_input: str
    q_dim: int
    kv_dim: int
    inter: int
    heads: int
    kv_heads: int
    head_dim: int


def _quant_weight(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Fake-quantize a [out, in] weight like src/compiler/ops/linear.py."""
    if bits == 8:
        s = w.abs().amax(1, keepdim=True).clamp(min=EPS) / 127.0
        return torch.clamp(torch.round(w / s), -127, 127) * s
    out, k = w.shape
    g = w.reshape(out, k // group, group)
    s = g.abs().amax(-1, keepdim=True).clamp(min=EPS) / 7.0
    return (torch.clamp(torch.round(g / s), -8, 7) * s).reshape(out, k)


def _quant_act(x: torch.Tensor) -> torch.Tensor:
    s = x.abs().amax(-1, keepdim=True).clamp(min=EPS) / 127.0
    return torch.clamp(torch.round(x / s), -127, 127) * s


def search_scale(w: np.ndarray, x: torch.Tensor, bits: int, group: int, tie=None, grid: int = 20) -> np.ndarray:
    """Best per-input-channel scale for weight ``w`` [out, K] on inputs ``x`` [T, K]."""
    wt = torch.from_numpy(np.ascontiguousarray(w, np.float32))
    y = x @ wt.T
    x_mean = x.abs().mean(0).clamp(min=1e-5)
    best_err, best = None, torch.ones_like(x_mean)
    for alpha in np.linspace(0.0, 1.0, grid + 1):
        s = x_mean ** float(alpha)
        if tie is not None:
            s = tie(s)
        s = s / torch.sqrt(s.max() * s.min())
        yq = _quant_act(x / s) @ _quant_weight(wt * s, bits, group).T
        err = float(((yq - y) ** 2).mean())
        if best_err is None or err < best_err:
            best_err, best = err, s
    return best.numpy().astype(np.float32)


def capture_inputs(model: torch.nn.Module, ids: list[int], spec: FoldSpec, layers: int) -> dict:
    """Inputs of every linear group, per layer, from the FP32 model."""
    inputs: dict[tuple[int, str], torch.Tensor] = {}
    hooks = []
    modules = dict(model.named_modules())
    for i in range(layers):
        for group, path in (("qkv", spec.qkv_input), ("o", spec.o_input),
                            ("gate_up", spec.gate_up_input), ("down", spec.down_input)):
            def hook(_mod, args, key=(i, group)):
                inputs[key] = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            hooks.append(modules[path.format(i=i)].register_forward_pre_hook(hook))
    with torch.no_grad():
        model(torch.tensor([ids]))
    for h in hooks:
        h.remove()
    return inputs


def make_hook(model: torch.nn.Module, ids: list[int], spec: FoldSpec, layers: int, bits: int, group: int,
              log: list | None = None):
    """Return ``awq(layer, mats, norms) -> (mats, norms)`` for a model's weight builder."""
    inputs = capture_inputs(model, ids, spec, layers)
    group_heads = spec.heads // spec.kv_heads
    D = spec.head_dim

    def tie_heads(s: torch.Tensor) -> torch.Tensor:
        # o's input channel (h, d) is v's (h // group, d): average within groups.
        kv = s.reshape(spec.kv_heads, group_heads, D).mean(1, keepdim=True)
        return kv.expand(spec.kv_heads, group_heads, D).reshape(-1)

    def awq(layer: int, mats: dict, norms: dict) -> tuple[dict, dict]:
        mats = {k: v.copy() for k, v in mats.items()}
        norms = {k: v.copy() for k, v in norms.items()}
        v0 = spec.q_dim + spec.kv_dim
        s = search_scale(mats["o"], inputs[(layer, "o")], bits, group, tie=tie_heads)
        mats["o"] *= s[None, :]
        s_kv = s.reshape(spec.kv_heads, group_heads, D)[:, 0, :].reshape(-1)
        mats["qkv"][v0:v0 + spec.kv_dim] /= s_kv[:, None]
        s = search_scale(mats["down"], inputs[(layer, "down")], bits, group)
        mats["down"] *= s[None, :]
        mats["gate_up"][spec.inter:] /= s[:, None]
        x_qkv = inputs[(layer, "qkv")]
        s = search_scale(mats["qkv"], x_qkv, bits, group)
        mats["qkv"] *= s[None, :]
        norms["ln1"] = norms["ln1"] / s
        s = search_scale(mats["gate_up"], inputs[(layer, "gate_up")], bits, group)
        mats["gate_up"] *= s[None, :]
        norms["ln2"] = norms["ln2"] / s
        if log is not None:
            log.append(layer)
        return mats, norms

    return awq


def hook_for(root, tokenizer_ids: list[int], spec: FoldSpec, layers: int, bits: int, group: int):
    """Load the FP32 Hugging Face model at ``root`` and calibrate on ``tokenizer_ids``."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32).eval()
    try:
        return make_hook(model, tokenizer_ids, spec, layers, bits, group)
    finally:
        del model


def calibration_ids(root) -> list[int]:
    from src.inference.tokenizer import Tokenizer

    return Tokenizer(root).encode(CALIB_TEXT)
