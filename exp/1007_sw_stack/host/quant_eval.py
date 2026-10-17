"""Score a compiled LLM package against FP32 and fake-quantized Hugging Face models.

    python exp/1007_sw_stack/host/quant_eval.py build/pkg/smollm [--tokens 200] [--json out.json]

``vs_fakequant`` checks the compiler (same quantization simulated in torch);
``vs_fp32`` measures the quantization itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.inference.engine import LLMEngine  # noqa: E402
from src.inference.quant.evaluate import EVAL_TEXT, compare_logits  # noqa: E402
from src.inference.weights import resolve_hf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--json")
    ap.add_argument("--no-fakequant", action="store_true")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from src.inference.quant.fakequant import quantize_model

    engine = LLMEngine(args.package, "sim")
    opts = engine.meta["options"]
    ids = engine.tokenizer.encode(EVAL_TEXT)[: args.tokens]
    ours = np.stack(engine.score(ids))
    root = resolve_hf(engine.meta["source"])
    load = lambda: AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32).eval()  # noqa: E731
    with torch.no_grad():
        fp32 = load()(torch.tensor([ids])).logits[0].numpy()
    report = {"package": args.package, "options": opts, "tokens": len(ids),
              "vs_fp32": compare_logits(fp32, ours, ids)}
    if not args.no_fakequant and not opts.get("awq"):
        fq = quantize_model(load(), opts["bits"], opts["group"], opts["embed_bits"])
        with torch.no_grad():
            ref = fq(torch.tensor([ids])).logits[0].numpy()
        report["vs_fakequant"] = compare_logits(ref, ours, ids)
        report["vs_fakequant"]["max_abs_logit_diff"] = float(np.abs(ref - ours).max())
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
