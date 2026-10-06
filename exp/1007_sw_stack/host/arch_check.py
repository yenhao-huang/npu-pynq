"""Architecture check: an FP32 build truncated to N layers against Transformers.

    python exp/1007_sw_stack/host/arch_check.py qwen3 --layers 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.compiler.export import export  # noqa: E402
from src.inference.engine import LLMEngine  # noqa: E402
from src.inference.graph import LLMOptions  # noqa: E402
from src.inference.weights import resolve_hf  # noqa: E402
from src.models import load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--source")
    ap.add_argument("--bits", type=int, default=32)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    module = load(args.model)
    source = args.source or module.DEFAULT_SOURCE
    tmp = Path(tempfile.mkdtemp())
    try:
        graph, _ = module.build(source, LLMOptions(bits=args.bits, embed_bits=args.bits, layers=args.layers, max_seq=64))
        export(graph, tmp / "pkg", targets=("host",), keep_ir=False)
        engine = LLMEngine(tmp / "pkg", "sim")
        ids = (engine.tokenizer.encode("The quick brown fox jumps over the lazy dog because")
               if engine.tokenizer else [3 + 37 * i % 1000 for i in range(20)])
        decode = np.stack(engine.score(ids))
        engine.reset()
        prefill_last = engine.prefill(ids).copy()
        hf = AutoModelForCausalLM.from_pretrained(resolve_hf(source), dtype=torch.float32,
                                                  num_hidden_layers=args.layers).eval()
        with torch.no_grad():
            ref = hf(torch.tensor([ids])).logits[0].numpy()
        rel = float(np.abs(decode - ref).max() / np.abs(ref).max())
        out = {"model": args.model, "layers": args.layers, "tokens": len(ids),
               "max_rel_err_decode": rel,
               "max_rel_err_prefill_last": float(np.abs(prefill_last - ref[-1]).max() / np.abs(ref).max()),
               "top1_agreement": float(np.mean(decode.argmax(-1) == ref.argmax(-1)))}
        print(json.dumps(out))
        return 0 if rel < 1e-3 or args.bits != 32 else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(main())
