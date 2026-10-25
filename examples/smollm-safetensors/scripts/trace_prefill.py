"""Capture the full model's first-step logits without patching its runtime."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.runtime.core.engine import LLMEngine
    engine = LLMEngine(args.package, backend='sim', stream_weights=True)
    ids = [1,4093,198,5162,253,8510,563,2333,101,2,198,1,520,9531,198]
    logits = engine.prefill(ids).copy()
    np.save(args.out, logits)
    order = np.argsort(logits.reshape(-1))[-10:][::-1]
    print(json.dumps({'min':float(logits.min()),'max':float(logits.max()),
                      'top10':[(int(i),float(logits.reshape(-1)[i])) for i in order]}), flush=True)


if __name__ == '__main__':
    main()
