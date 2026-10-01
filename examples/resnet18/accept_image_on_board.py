"""Accept one real photograph on the physical PYNQ-Z1 from a release package.

This is the automated counterpart of the notebook demonstration. It runs the
pinned demo image through the deployed overlay, proves every capture against
the host record byte for byte, decodes the top-5 ImageNet prediction, and
publishes evidence that names the release package, the source and artifact
versions, the input image, and the inference result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
# In a deployed standalone package this file sits at the package root; in the
# repository it sits under examples/resnet18/.
REPOSITORY_ROOT = (
    MODULE_DIR
    if (MODULE_DIR / "package.manifest.json").is_file()
    else MODULE_DIR.parents[1]
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

try:
    from examples.resnet18.run_on_board import _array_sha256, _sha256, _write_new
except ModuleNotFoundError:  # deployed standalone package layout
    from run_on_board import _array_sha256, _sha256, _write_new

from src.export.imagenet import decode_logits, input_scale, load_class_names
from src.runtime import (
    NPURuntime,
    NPUModelRuntime,
    load_model_package,
    load_pynq_runtime,
)
from src.runtime.verify_overlay import verify_artifacts


PASS_MARKER = "PASS [physical-pynq-z1]: real image classified on the board"
EVIDENCE_MAGIC = "NPU_RESNET18_IMAGE_ACCEPTANCE"
DEMO_MAGIC = "NPU_RESNET18_DEMO_INPUT"
TOP_K = 5


class ImageAcceptanceError(RuntimeError):
    """A real-image board acceptance gate failed; no evidence was published."""


def _json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ImageAcceptanceError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ImageAcceptanceError(f"{label} must contain an object")
    return value


def _expect_digest(path: Path, expected: object, label: str) -> str:
    if not path.is_file():
        raise ImageAcceptanceError(f"{label} is missing: {path.name}")
    actual = _sha256(path)
    if actual != expected:
        raise ImageAcceptanceError(f"{label} differs from the host record")
    return actual


def accept_image(
    *,
    package_root: Path,
    evidence_path: Path,
    software_timeout: float = 86400.0,
) -> dict[str, object]:
    """Classify the pinned demo image on the board and publish the evidence."""

    root = Path(package_root).resolve()
    package = _json(root / "package.manifest.json", "package manifest")
    model_dir = root / "model"
    artifact_dir = root / "artifacts"

    record = _json(model_dir / "resnet18.demo.json", "demo record")
    if record.get("magic") != DEMO_MAGIC:
        raise ImageAcceptanceError("demo record magic is invalid")
    if record.get("result") != "pass":
        raise ImageAcceptanceError("demo record does not carry a host pass")

    overlay = verify_artifacts(artifact_dir)
    declared = package.get("overlay")
    if not isinstance(declared, dict):
        raise ImageAcceptanceError("package manifest has no overlay record")
    if (
        overlay["bit"]["sha256"] != declared.get("bit_sha256")
        or overlay["hwh"]["sha256"] != declared.get("hwh_sha256")
    ):
        raise ImageAcceptanceError("deployed overlay differs from the package manifest")

    manifest_path = model_dir / "resnet18.npu.json"
    payload_path = model_dir / "resnet18.npu.bin"
    model_record = record.get("model")
    if not isinstance(model_record, dict):
        raise ImageAcceptanceError("demo record has no model digests")
    _expect_digest(manifest_path, model_record.get("manifest_sha256"), "model manifest")
    _expect_digest(payload_path, model_record.get("payload_sha256"), "model payload")

    input_record = record.get("input")
    if not isinstance(input_record, dict) or not isinstance(
        input_record.get("file"), str
    ):
        raise ImageAcceptanceError("demo record has no input tensor")
    input_path = model_dir / str(input_record["file"])
    _expect_digest(input_path, input_record.get("sha256"), "demo input tensor")

    classes_record = record.get("classes")
    if not isinstance(classes_record, dict) or not isinstance(
        classes_record.get("file"), str
    ):
        raise ImageAcceptanceError("demo record has no class list")
    classes_path = model_dir / str(classes_record["file"])
    _expect_digest(classes_path, classes_record.get("sha256"), "ImageNet class list")

    physical = load_pynq_runtime(artifact_dir / "npu_matrix.bit")
    if not isinstance(physical, NPURuntime):
        raise ImageAcceptanceError("physical evidence requires the public NPURuntime")

    model = load_model_package(manifest_path)
    demo_input = np.load(input_path, allow_pickle=False)
    started = time.monotonic()
    result = NPUModelRuntime(physical, model).run(
        {"input": demo_input}, software_timeout=software_timeout
    )
    elapsed = time.monotonic() - started
    if result.metrics.physical_jobs <= 0:
        raise ImageAcceptanceError("no physical job ran; this was not the board")

    captures = record.get("captures")
    if not isinstance(captures, dict) or not captures:
        raise ImageAcceptanceError("demo record has no host captures")
    for name, capture in sorted(captures.items()):
        if name not in result.outputs:
            raise ImageAcceptanceError(f"model did not produce capture {name!r}")
        if not isinstance(capture, dict):
            raise ImageAcceptanceError(f"capture record {name!r} is malformed")
        if _array_sha256(result.outputs[name]) != capture.get("sha256"):
            raise ImageAcceptanceError(f"board capture differs from host: {name}")

    manifest = _json(manifest_path, "model manifest")
    logit_scale = input_scale(manifest, "logits")
    class_names = load_class_names(classes_path)
    predictions = decode_logits(
        result.outputs["logits"], logit_scale, class_names, top_k=TOP_K
    )
    expected = record.get("expected")
    verdict = "undeclared"
    if isinstance(expected, dict):
        verdict = (
            "correct" if predictions[0].name == expected.get("name") else "incorrect"
        )
        if verdict == "incorrect":
            raise ImageAcceptanceError(
                f"board predicted {predictions[0].name!r}, expected "
                f"{expected.get('name')!r}"
            )

    image_record = record.get("image")
    if not isinstance(image_record, dict):
        raise ImageAcceptanceError("demo record has no image provenance")
    evidence: dict[str, object] = {
        "evidence_type": "physical-pynq-z1",
        "expected": expected,
        "format": {"major": 1, "minor": 0},
        "image": {
            "bytes": image_record.get("bytes"),
            "file": image_record.get("file"),
            "provenance": image_record.get("provenance"),
            "sha256": image_record.get("sha256"),
        },
        "input": {
            "file": input_record.get("file"),
            "scale": input_record.get("scale"),
            "sha256": input_record.get("sha256"),
        },
        "magic": EVIDENCE_MAGIC,
        "model": {
            "manifest_sha256": model_record.get("manifest_sha256"),
            "payload_sha256": model_record.get("payload_sha256"),
        },
        "overlay": {
            "bit_sha256": overlay["bit"]["sha256"],
            "hwh_sha256": overlay["hwh"]["sha256"],
            "source_commit": str(overlay["source_commit"]).lower(),
            "target_part": overlay["target_part"],
        },
        "package": {
            "release_tag": package.get("release_tag"),
            "source_commit": package.get("source_commit"),
        },
        "result": "pass",
        "runtime": {
            "abi_major": physical.abi_major,
            "capabilities": physical.capabilities,
            "elapsed_seconds": elapsed,
            "mac_count": result.metrics.mac_count,
            "physical_jobs": result.metrics.physical_jobs,
            "physical_limits": [physical.max_m, physical.max_n, physical.max_k],
        },
        "top_k": [prediction.as_record() for prediction in predictions],
        "verdict": verdict,
    }
    _write_new(Path(evidence_path).resolve(), evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=MODULE_DIR)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--software-timeout", type=float, default=86400.0)
    arguments = parser.parse_args()
    try:
        evidence = accept_image(
            package_root=arguments.package_root,
            evidence_path=arguments.evidence,
            software_timeout=arguments.software_timeout,
        )
    except Exception as error:
        print(f"real-image board acceptance failed: {error}", file=sys.stderr)
        return 1
    top = evidence["top_k"][0]
    print(PASS_MARKER)
    print(f"predicted: {top['name']}   ({top['probability']:.2%})")
    print(f"verdict:   {evidence['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
