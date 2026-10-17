"""Quantization quality of W8A8 / W4A8 with and without AWQ, in memory.

The compiler reproduces fake-quantized Transformers numerics (quant_eval.py),
so the quantization schemes are compared on the fake-quantized model:

    python exp/1007_sw_stack/host/awq_eval.py smollm [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.compiler.ops.linear import quantize_weight  # noqa: E402
from src.inference.quant import awq as AWQ  # noqa: E402
from src.inference.quant.evaluate import EVAL_TEXT, compare_logits  # noqa: E402
from src.inference.quant.fakequant import quantize_model  # noqa: E402
from src.inference.tokenizer import Tokenizer  # noqa: E402
from src.inference.weights import resolve_hf  # noqa: E402
from src.inference.graph import choose_group  # noqa: E402
from src.models import load  # noqa: E402


def awq_weights(model, hook, spec: AWQ.FoldSpec, layers: int, bits: int, group: int) -> dict:
    """Apply the AWQ transform to an HF model in place; return its quantized linears."""
    out = {}
    q, kv = spec.q_dim, spec.kv_dim
    for i in range(layers):
        L = model.model.layers[i]
        a, m = L.self_attn, L.mlp
        mats = {"qkv": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach().numpy(),
                "o": a.o_proj.weight.detach().numpy(),
                "gate_up": torch.cat([m.gate_proj.weight, m.up_proj.weight]).detach().numpy(),
                "down": m.down_proj.weight.detach().numpy()}
        norms = {"ln1": L.input_layernorm.weight.detach().numpy(), "ln2": L.post_attention_layernorm.weight.detach().numpy()}
        mats, norms = hook(i, mats, norms)
        L.input_layernorm.weight.data = torch.from_numpy(norms["ln1"])
        L.post_attention_layernorm.weight.data = torch.from_numpy(norms["ln2"])
        p = f"model.layers.{i}."
        parts = {"self_attn.q_proj": mats["qkv"][:q], "self_attn.k_proj": mats["qkv"][q:q + kv],
                 "self_attn.v_proj": mats["qkv"][q + kv:], "self_attn.o_proj": mats["o"],
                 "mlp.gate_proj": mats["gate_up"][:spec.inter], "mlp.up_proj": mats["gate_up"][spec.inter:],
                 "mlp.down_proj": mats["down"]}
        for name, w in parts.items():
            out[p + name] = quantize_weight(w, bits, group)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--json")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM

    module = load(args.model)
    root = resolve_hf(module.DEFAULT_SOURCE)
    cfg = module.Config(json.loads((root / "config.json").read_text()))
    spec = module.awq_spec(cfg)
    group = choose_group(128, [cfg.hidden, cfg.inter, cfg.heads * cfg.head_dim])
    ids = Tokenizer(root).encode(EVAL_TEXT)[: args.tokens]
    fresh = lambda: AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32).eval()  # noqa: E731
    with torch.no_grad():
        ref = fresh()(torch.tensor([ids])).logits[0].numpy()
    results = {}
    for bits in (8, 4):
        for use_awq in (False, True):
            model = fresh()
            weights = None
            if use_awq:
                hook = AWQ.make_hook(model, AWQ.calibration_ids(root), spec, cfg.layers, bits, group)
                weights = awq_weights(model, hook, spec, cfg.layers, bits, group)
            quantize_model(model, bits, group, 8, weights)
            with torch.no_grad():
                out = model(torch.tensor([ids])).logits[0].numpy()
            key = f"W{bits}A8{'+AWQ' if use_awq else ''}"
            results[key] = compare_logits(ref, out, ids)
            print(key, json.dumps(results[key]), flush=True)
    report = {"group": group, "model": args.model, "source": module.DEFAULT_SOURCE, "eval_tokens": len(ids), "results": results}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
