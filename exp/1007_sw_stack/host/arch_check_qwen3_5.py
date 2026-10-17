"""FP32 Qwen3.5 (text) truncated to N layers against Transformers' Qwen3_5ForCausalLM.

    python exp/1007_sw_stack/host/arch_check_qwen3_5.py --layers 4
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.compiler.export import export  # noqa: E402
from src.inference.engine import LLMEngine  # noqa: E402
from src.inference.graph import LLMOptions  # noqa: E402
from src.inference.weights import resolve_hf  # noqa: E402
from src.models.qwen3_5 import model as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--bits", type=int, default=32)
args = ap.parse_args()
from transformers import AutoConfig, Qwen3_5ForCausalLM  # noqa: E402

root = resolve_hf(M.DEFAULT_SOURCE)
tmp = Path(tempfile.mkdtemp())
try:
    graph, _ = M.build(opts=LLMOptions(bits=args.bits, embed_bits=args.bits, layers=args.layers, max_seq=64))
    export(graph, tmp / "pkg", targets=("host",), keep_ir=False)
    engine = LLMEngine(tmp / "pkg", "sim")
    ids = engine.tokenizer.encode("The quick brown fox jumps over the lazy dog because it was")
    decode = np.stack(engine.score(ids))
    engine.reset()
    prefill = engine.prefill(ids).copy()
    text = AutoConfig.from_pretrained(root).text_config
    text.num_hidden_layers = args.layers
    text.layer_types = text.layer_types[: args.layers]
    hf = Qwen3_5ForCausalLM.from_pretrained(root, config=text, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()
    out = {"layers": args.layers, "bits": args.bits, "tokens": len(ids),
           "max_rel_err_decode": float(np.abs(decode - ref).max() / np.abs(ref).max()),
           "max_rel_err_prefill_last": float(np.abs(prefill - ref[-1]).max() / np.abs(ref).max()),
           "top1_agreement": float(np.mean(decode.argmax(-1) == ref.argmax(-1)))}
    print(json.dumps(out))
finally:
    shutil.rmtree(tmp)
