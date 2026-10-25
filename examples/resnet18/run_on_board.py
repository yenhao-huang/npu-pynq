"""Run a previously compiled ResNet18 package on an image and record block times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR if (MODULE_DIR/'src/runtime').is_dir() else MODULE_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime.core.image import ImageEngine, preprocess, top_predictions


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--package-archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--release-tag")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--backend", choices=("sim", "pynq"), default="pynq")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--expected-class")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    if args.package_root:
        from package_example import verify_delivery, sha256_file
        manifest = verify_delivery(args.package_root)
        if args.release_tag and args.release_tag != manifest['release_tag']:
            raise ValueError('Release tag mismatch')
        if args.package_archive:
            digest = sha256_file(args.package_archive)
            if digest != args.archive_sha256:
                raise ValueError('Archive SHA256 mismatch')
        root = args.package_root
        args.package = args.package or root/'model/aot'
        args.image = args.image or root/'model/demo-image.jpg'
        args.labels = args.labels or root/'model/imagenet-classes.txt'
        args.overlay = args.overlay or root/'artifacts/npu_matrix.bit'
    if args.verify_only:
        if not args.package_root:
            parser.error('--verify-only requires --package-root')
        print('PASS: verified ResNet18 AOT delivery')
        return 0
    if not all((args.package,args.image,args.labels)):
        parser.error('--package, --image and --labels are required outside a delivery')
    started = time.perf_counter()
    times = {}
    t = time.perf_counter()
    overlay = None
    if args.backend == "pynq":
        if args.overlay is None:
            parser.error("--overlay is required for pynq")
        from pynq import Overlay
        overlay = Overlay(str(args.overlay))
    times["overlay"] = time.perf_counter() - t
    t = time.perf_counter()
    engine = ImageEngine(args.package, args.backend, overlay=overlay)
    times["load_package"] = time.perf_counter() - t
    t = time.perf_counter()
    from PIL import Image
    with Image.open(args.image) as image:
        tensor = preprocess(image, engine.contract)
    times["preprocess"] = time.perf_counter() - t
    t = time.perf_counter()
    logits = engine.predict(tensor)
    times["inference"] = time.perf_counter() - t
    t = time.perf_counter()
    labels = args.labels.read_text(encoding="utf-8").splitlines()
    predictions = top_predictions(logits, labels)
    times["prediction"] = time.perf_counter() - t
    times["total"] = time.perf_counter() - started
    if args.backend == "pynq" and engine.last_stats["jobs"] <= 0:
        raise RuntimeError("No physical NPU jobs executed")
    record = {"backend": args.backend, "package": str(args.package),
              "image": str(args.image), "block_seconds": times,
              "npu": engine.last_stats, "top5": [vars(p) for p in predictions],
              "expected_class": args.expected_class,
              "correct": predictions[0].name == args.expected_class if args.expected_class else None}
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if args.expected_class and not record['correct']:
        raise RuntimeError('Image prediction does not match the expected class')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
