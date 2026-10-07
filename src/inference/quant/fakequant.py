"""Hugging Face models with the compiler's quantization simulated in PyTorch.

Every nn.Linear in the decoder becomes a fake-quantized linear with exactly
the numerics of src/compiler/ops/linear.py: per-row symmetric INT8
activations and the package's INT8/INT4 weights, the product taken in
integers and rescaled. The tied embedding/lm_head uses its INT8 copy. A
compiled package should agree with this model to float rounding; any larger
difference is a compiler bug, and the difference between this model and the
FP32 one is the quantization error itself.
"""

from __future__ import annotations

import numpy as np
import torch

from src.compiler.ops.linear import EPS, QuantizedWeight, quantize_weight


class FakeQuantLinear(torch.nn.Module):
    def __init__(self, w: QuantizedWeight, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.bits = w.bits
        self.group = w.group
        self.register_buffer("q", torch.from_numpy(w.q.astype(np.float32)))       # [K, N]
        self.register_buffer("scale", torch.from_numpy(w.scale))
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).float()
        if self.bits == 32:
            y = x2 @ self.q
        else:
            s = torch.clamp(x2.abs().amax(-1), min=EPS) * (1.0 / 127.0)
            xq = torch.clamp(torch.round(x2 * (1.0 / s)[:, None]), -127, 127)
            if self.bits == 8:
                y = (xq.double() @ self.q.double()).float() * s[:, None] * self.scale[None, :]
            else:
                g = self.group
                k = xq.shape[1]
                parts = torch.einsum("mcj,cjn->cmn", xq.reshape(-1, k // g, g).double(),
                                     self.q.reshape(k // g, g, -1).double()).float()
                y = (parts * self.scale[:, None, :]).sum(0) * s[:, None]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], -1)


def quantize_model(model: torch.nn.Module, bits: int = 8, group: int = 128, embed_bits: int = 8,
                   weights: dict[str, QuantizedWeight] | None = None) -> torch.nn.Module:
    """Swap the decoder's linears (and lm_head) for fake-quantized ones in place.

    ``weights`` overrides the quantized weight for a module name (e.g. AWQ)."""
    weights = weights or {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, torch.nn.Linear) and "visual" not in full:
                is_head = full.endswith("lm_head")
                w = weights.get(full) or quantize_weight(child.weight.detach().float().numpy(),
                                                         embed_bits if is_head else bits, group)
                setattr(module, child_name, FakeQuantLinear(w, child.bias))
    if embed_bits != 32:
        emb = model.get_input_embeddings()
        w = quantize_weight(emb.weight.detach().float().numpy(), embed_bits, group)
        emb.weight.data = torch.from_numpy(w.dequantize().T.copy())
    return model
