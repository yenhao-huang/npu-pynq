"""Download the INT8 evaluation image corpus into the ignored model workspace.

The corpus is a public GitHub-hosted sample of ImageNet-1K, one image per
class (1000 images total). It is used by ``eval_int8_accuracy.py`` to measure
accuracy on real photographs.

This is **not** the ImageNet-1K validation set. It is a curated one-per-class
sample distributed for visualization; absolute accuracy figures obtained here
are not comparable to published validation numbers. Only same-image deltas
between FP32 and INT8 are meaningful.

The corpus is downloaded because ``image-net.org`` is outside the environment
network allowlist while ``raw.githubusercontent.com`` is inside it.

Source: EliSchwartz/imagenet-sample-images, BSD-3-Clause repository licence.
Every image name embeds its WordNet id, which pins the ImageNet class label
per the mapping from raghakot/keras-vis' ``imagenet_class_index.json``.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/"
    "raghakot/keras-vis/master/resources/imagenet_class_index.json"
)
IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "EliSchwartz/imagenet-sample-images/master/"
)


def _fetch(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def main() -> int:
    example_root = Path(__file__).resolve().parents[4]
    corpus_dir = example_root / "model" / "imagenet-calibration"
    index_path = example_root / "model" / "imagenet-calibration.index.tsv"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    print("INFO: fetching ImageNet class index", flush=True)
    class_index = json.loads(_fetch(CLASS_INDEX_URL).decode("utf-8"))
    entries = [
        (index, class_index[str(index)][0], class_index[str(index)][1])
        for index in range(1000)
    ]
    index_path.write_text(
        "".join(
            f"{index}\t{wnid}\t{name}\t{wnid}_{name}.JPEG\n"
            for index, wnid, name in entries
        ),
        encoding="utf-8",
    )
    print(f"PASS: wrote {index_path.name} with {len(entries)} entries", flush=True)

    def _download(entry):
        index, wnid, name = entry
        filename = f"{wnid}_{name}.JPEG"
        target = corpus_dir / filename
        if target.exists() and target.stat().st_size > 1024:
            return filename, "cached"
        try:
            data = _fetch(IMAGE_BASE_URL + urllib.parse.quote(filename))
        except Exception as error:
            return filename, f"error:{type(error).__name__}:{error}"
        if len(data) < 512:
            return filename, f"short:{len(data)}"
        target.write_bytes(data)
        return filename, "ok"

    failed = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for filename, status in pool.map(_download, entries):
            completed += 1
            if status not in ("ok", "cached"):
                failed.append((filename, status))
            if completed % 100 == 0:
                print(f"  {completed}/1000  failures={len(failed)}", flush=True)

    if failed:
        print(f"FAIL: {len(failed)} downloads failed:", flush=True)
        for filename, status in failed[:10]:
            print(f"  {filename}: {status}")
        return 1
    print(f"PASS: downloaded 1000 images into {corpus_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
