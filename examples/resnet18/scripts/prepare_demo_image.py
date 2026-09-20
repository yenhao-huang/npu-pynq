"""Turn one real image into the ResNet-18 demo input and host reference record.

The image is preprocessed once, on the conversion host, so the board reloads a
byte-identical signed-INT8 tensor instead of depending on its image library.
This publishes host evidence only; the physical PASS belongs to the notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from host_reference import (
    HostMatrixBackend,
    REPOSITORY_ROOT,
    array_sha256,
    canonical_write_new,
    file_sha256,
    read_json,
    write_new_array,
)
from src.export.imagenet import (
    ImageNetInputError,
    decode_logits,
    input_scale,
    load_class_names,
    load_rgb_image,
    preprocess_image,
    preprocessing_contract,
    resolve_class_index,
)
from src.export.torchvision_resnet18 import (
    REAL_MODEL_HOST_EVIDENCE_TYPE,
    compare_integer_captures,
)
from src.runtime.model import NPUModelRuntime, load_model_package
from src.test.model.quantized_graph_reference import (
    execute_quantized_graph_reference,
)
from download_demo_assets import load_demo_metadata

CAPTURE_NAMES = ("stem.relu", "layer1.1.relu", "logits")
DEMO_MAGIC = "NPU_RESNET18_DEMO_INPUT"
TOP_K = 5


def _pinned_sample(metadata_path: Path) -> tuple[str, str]:
    """Return the pinned sample image filename and its expected class label."""

    for asset in load_demo_metadata(metadata_path):
        if asset.kind == "sample-image":
            if not asset.expected_class:
                raise ImageNetInputError("pinned sample image has no expected class")
            return asset.filename, asset.expected_class
    raise ImageNetInputError("demo metadata pins no sample image")


def prepare_demo_image(
    *,
    model_dir: Path,
    demo_metadata_path: Path,
    image_path: Path | None,
    expected_class: str | None,
    output_prefix: Path,
    software_timeout: float = 600.0,
) -> dict[str, object]:
    """Preprocess one image, run the host reference, and publish the record."""

    model_dir = model_dir.resolve()
    prefix = output_prefix.resolve()
    manifest_path = model_dir / "resnet18.npu.json"
    payload_path = model_dir / "resnet18.npu.bin"
    classes_path = model_dir / "imagenet-classes.txt"
    input_path = prefix.with_name(f"{prefix.name}.npy")
    record_path = prefix.with_name(f"{prefix.name}.json")
    for path in (manifest_path, payload_path, classes_path):
        if not path.is_file():
            raise ImageNetInputError(f"demo input is missing: {path.name}")
    existing = [path.name for path in (input_path, record_path) if path.exists()]
    if existing:
        raise ImageNetInputError(f"demo output already exists: {existing[0]}")

    if image_path is None:
        sample_name, sample_class = _pinned_sample(demo_metadata_path)
        image_path = model_dir / sample_name
        provenance = "pinned-sample"
        if expected_class is None:
            expected_class = sample_class
    else:
        image_path = Path(image_path).resolve()
        provenance = "user-provided"
    if not image_path.is_file():
        raise ImageNetInputError(f"image file is missing: {image_path}")

    names = load_class_names(classes_path)
    manifest = read_json(manifest_path)
    scale = input_scale(manifest, "input")
    logit_scale = input_scale(manifest, "logits")

    image = load_rgb_image(image_path)
    _crop, quantized = preprocess_image(image, scale)
    if quantized.dtype != np.int8 or quantized.shape != (1, 224, 224, 3):
        raise ImageNetInputError("preprocessed input must be int8 (1, 224, 224, 3)")

    model = load_model_package(manifest_path)
    if tuple(model.graph.outputs) != CAPTURE_NAMES:
        raise ImageNetInputError("model does not expose the required demo captures")
    reference = execute_quantized_graph_reference(
        model.graph, model.constants, {"input": quantized}
    )
    result = NPUModelRuntime(HostMatrixBackend(), model).run(
        {"input": quantized}, software_timeout=software_timeout
    )
    compare_integer_captures(reference, result.outputs)
    predictions = decode_logits(result.outputs["logits"], logit_scale, names, top_k=TOP_K)

    expected_index = (
        resolve_class_index(expected_class, names) if expected_class is not None else None
    )
    write_new_array(input_path, quantized)
    record: dict[str, object] = {
        "captures": {
            name: {
                "dtype": "int8",
                "sha256": array_sha256(result.outputs[name]),
                "shape": list(result.outputs[name].shape),
            }
            for name in CAPTURE_NAMES
        },
        "classes": {
            "count": len(names),
            "file": classes_path.name,
            "sha256": file_sha256(classes_path),
        },
        "evidence_type": REAL_MODEL_HOST_EVIDENCE_TYPE,
        "expected": (
            None
            if expected_index is None
            else {"index": expected_index, "name": names[expected_index]}
        ),
        "format": {"major": 1, "minor": 0},
        "host_top_k": [prediction.as_record() for prediction in predictions],
        "image": {
            "bytes": image_path.stat().st_size,
            "file": image_path.name,
            "height": int(image.shape[0]),
            "provenance": provenance,
            "sha256": file_sha256(image_path),
            "width": int(image.shape[1]),
        },
        "input": {
            "bytes": input_path.stat().st_size,
            "dtype": "int8",
            "file": input_path.name,
            "scale": scale,
            "sha256": file_sha256(input_path),
            "shape": list(quantized.shape),
        },
        "integer_reference": "independent-vectorized-v1",
        "logit_scale": logit_scale,
        "magic": DEMO_MAGIC,
        "model": {
            "manifest_sha256": file_sha256(manifest_path),
            "payload_sha256": file_sha256(payload_path),
        },
        "preprocessing": preprocessing_contract(),
        "result": "pass",
        "runtime": {
            "backend": "host-matrix",
            "mac_count": result.metrics.mac_count,
            "physical_board": False,
            "physical_jobs": result.metrics.physical_jobs,
        },
    }
    try:
        canonical_write_new(record_path, record)
    except Exception:
        if input_path.exists():
            input_path.unlink()
        raise
    return record


def main() -> int:
    example_root = REPOSITORY_ROOT / "examples" / "resnet18"
    model_dir = example_root / "model"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument(
        "--demo-metadata", type=Path, default=example_root / "demo-source.json"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="real image to classify; defaults to the pinned sample image",
    )
    parser.add_argument(
        "--expected-class",
        default=None,
        help="ImageNet class index or exact class name the image should predict",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=model_dir / "resnet18.demo"
    )
    parser.add_argument("--software-timeout", type=float, default=600.0)
    arguments = parser.parse_args()
    record = prepare_demo_image(
        model_dir=arguments.model_dir,
        demo_metadata_path=arguments.demo_metadata,
        image_path=arguments.image,
        expected_class=arguments.expected_class,
        output_prefix=arguments.output_prefix,
        software_timeout=arguments.software_timeout,
    )
    top = record["host_top_k"][0]
    expected = record["expected"]
    print(f"PASS [{REAL_MODEL_HOST_EVIDENCE_TYPE}]: demo input at {arguments.output_prefix}.npy")
    print(f"INFO: host top-1 is {top['name']} (index {top['index']}, p={top['probability']:.4f})")
    if expected is not None:
        verdict = "CORRECT" if top["index"] == expected["index"] else "INCORRECT"
        print(f"INFO: expected {expected['name']} (index {expected['index']}) -> {verdict}")
    print("INFO: this is host evidence; the notebook owns the physical PYNQ-Z1 result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
