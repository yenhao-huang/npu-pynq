"""ResNet-18 package vs TorchVision FP32 on the pinned gallery images.

    python exp/1007_sw_stack/host/resnet_eval.py build/pkg/resnet18 [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.models.resnet18 import model as R  # noqa: E402
from src.runtime.compiled import CompiledModel  # noqa: E402


def gallery(cache: Path) -> list[tuple[str, Path]]:
    spec = json.loads((REPO / "examples/resnet18/gallery-source.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for item in spec["images"]:
        path = cache / item["filename"]
        for attempt in range(6):
            if path.exists():
                break
            try:
                req = urllib.request.Request(item["url"], headers={"User-Agent": "npu-pynq-exp/1.0 (research)"})
                path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            except urllib.error.HTTPError as error:  # Wikimedia rate-limits bursts (429)
                if error.code != 429 or attempt == 5:
                    raise
                time.sleep(float(error.headers.get("Retry-After") or 10 * 2 ** attempt))
        time.sleep(2)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"{path} does not match its pinned SHA-256")
        out.append((item["expected_class"], path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--images", default=str(REPO / "build/gallery"))
    ap.add_argument("--json")
    args = ap.parse_args()
    import torch
    import torchvision
    from PIL import Image

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    labels = weights.meta["categories"]
    ref = torchvision.models.resnet18()
    ref.load_state_dict(torch.load(R._find_source(R.DEFAULT_SOURCE), weights_only=True))
    ref.eval()
    model = CompiledModel(args.package, "sim")
    rows = []
    for expected, path in gallery(Path(args.images)):
        x = R.preprocess(Image.open(path))
        t = time.perf_counter()
        ours = R.classify(model, x)
        seconds = time.perf_counter() - t
        with torch.no_grad():
            fp = ref(torch.from_numpy(x).permute(2, 0, 1)[None])[0].numpy()
        top5 = [labels[i] for i in np.argsort(-ours)[:5]]
        rows.append({"image": path.name, "expected": expected, "top5": top5,
                     "top1_correct": expected.lower() in top5[0].lower(),
                     "top1_matches_fp32": int(np.argmax(ours)) == int(np.argmax(fp)),
                     "top5_overlap_fp32": len(set(np.argsort(-ours)[:5]) & set(np.argsort(-fp)[:5])),
                     "logit_cosine_fp32": float(ours @ fp / np.linalg.norm(ours) / np.linalg.norm(fp)),
                     "host_seconds": seconds})
        print(json.dumps(rows[-1]))
    report = {"package": args.package, "images": rows, "npu": model.stats()}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
