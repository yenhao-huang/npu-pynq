"""Compare an exported LLM package with Hugging Face Transformers (FP32) on the host.

    python exp/1007_sw_stack/host/compare_hf.py build/pkg/smollm [--prompt ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.inference.engine import LLMEngine  # noqa: E402
from src.inference.weights import resolve_hf  # noqa: E402

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ", "In a distant galaxy, scientists discovered"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--json")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    engine = LLMEngine(args.package, "sim")
    source = engine.meta["source"]
    hf = AutoModelForCausalLM.from_pretrained(resolve_hf(source), torch_dtype=torch.float32)
    hf.eval()
    report = {"package": args.package, "options": engine.meta["options"], "prompts": []}
    for prompt in PROMPTS:
        ids = engine.tokenizer.encode(prompt)
        with torch.no_grad():
            ref = hf(torch.tensor([ids])).logits[0].numpy()
        ours_prefill = engine.prefill(ids).copy() if not engine.reset() else None
        engine.reset()
        decode_logits = [engine.decode(t).copy() for t in ids]
        last = ref[-1]
        cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
        per_pos_top1 = float(np.mean([np.argmax(d) == np.argmax(r) for d, r in zip(decode_logits, ref)]))
        t = time.perf_counter()
        out_ids, text, stats = engine.generate(ids, args.max_new)
        with torch.no_grad():
            hf_out = hf.generate(torch.tensor([ids]), max_new_tokens=args.max_new, do_sample=False)[0, len(ids):]
        hf_ids = [int(x) for x in hf_out if int(x) not in engine.eos]
        n = min(len(out_ids), len(hf_ids))
        agree = next((i for i in range(n) if out_ids[i] != hf_ids[i]), n)
        entry = {"prompt": prompt, "tokens": len(ids),
                 "prefill_cosine": cos(ours_prefill, last), "decode_cosine": cos(decode_logits[-1], last),
                 "prefill_vs_decode_max_abs": float(np.abs(ours_prefill - decode_logits[-1]).max()),
                 "top1_match_every_position": per_pos_top1,
                 "greedy_tokens_agreeing_with_hf": f"{agree}/{n}",
                 "ours": text, "hf": engine.tokenizer.decode(hf_ids), **{"stats": stats.as_dict()}}
        report["prompts"].append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
