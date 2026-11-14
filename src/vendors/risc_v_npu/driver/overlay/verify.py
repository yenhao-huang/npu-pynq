"""Verify that a generated CVA6 NPU BIT/HWH pair matches its build manifest.

Modelled on src/vendors/npu_pynq/driver/overlay/verify.py (copied, not
imported). Differences: the artifacts are ``npu_accelerator_cva6.*`` so they
can sit beside the CUSTOM build's; the HWH must show the accelerator built
with ``SCALAR_IMPL = "CVA6"``; and the manifest must record capabilities that
include bit 7 (``RISCV_SCALAR_CORE``). ``verify_device`` checks the same bit
on the loaded overlay's CSR window. Runs on the board before an overlay is
loaded and in CD before a release; the manifest is written by build.py.

    python src/vendors/risc_v_npu/driver/overlay/verify.py ARTIFACT_DIR
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Callable
import xml.etree.ElementTree as ET

try:
    from src.vendors.risc_v_npu.spec.abi import (
        ABI_MAGIC, ABI_MAJOR, REQUIRED_CAPABILITIES, Capability, Register,
    )
except ImportError:  # run as a script from build_overlay.tcl, without the repository on sys.path
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        '_risc_v_npu_abi', Path(__file__).resolve().parents[2] / 'spec' / 'abi.py')
    _abi = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_abi)
    ABI_MAGIC, ABI_MAJOR = _abi.ABI_MAGIC, _abi.ABI_MAJOR
    REQUIRED_CAPABILITIES, Capability, Register = _abi.REQUIRED_CAPABILITIES, _abi.Capability, _abi.Register


ARTIFACT_STEM = "npu_accelerator_cva6"
SCALAR_IMPL = "CVA6"

EXPECTED_METADATA = {
    "accelerator": {
        "instance": "npu_accelerator_0",
        "parameters": {
            "ROWS": "2",
            "COLUMNS": "2",
            "MAX_K": "256",
            "IN_BYTES": "8",
            "SCALAR_IMPL": SCALAR_IMPL,
            "C_BASEADDR": "0x43C00000",
            "C_HIGHADDR": "0x43C0FFFF",
        },
    },
}


class OverlayVerificationError(RuntimeError):
    """Generated artifacts or their provenance do not satisfy the contract."""


def _expected_metadata(array_size: int) -> dict[str, object]:
    if array_size not in (2, 8, 16):
        raise OverlayVerificationError(
            f"unsupported array size {array_size}: expected 2, 8, or 16"
        )
    expected = copy.deepcopy(EXPECTED_METADATA)
    parameters = expected["accelerator"]["parameters"]
    parameters["ROWS"] = str(array_size)
    parameters["COLUMNS"] = str(array_size)
    return expected


def _validated_commit(value: str) -> str:
    commit = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", commit) is None:
        raise OverlayVerificationError(
            "source_commit must be a full hexadecimal Git object id"
        )
    return commit.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_parameters(hwh_path: Path, instance: str) -> dict[str, str]:
    root = ET.parse(hwh_path).getroot()
    for module in root.iter("MODULE"):
        if module.attrib.get("INSTANCE") == instance:
            return {
                parameter.attrib["NAME"].upper(): parameter.attrib.get("VALUE", "")
                for parameter in module.iter("PARAMETER")
                if "NAME" in parameter.attrib
            }
    raise OverlayVerificationError(f"HWH is missing module {instance}")


def _parameter_value(value: str) -> str:
    """Vivado may quote string parameters ("CVA6"); compare without quotes."""
    return value.strip().strip('"').lower()


def inspect_hwh(hwh_path: Path, *, array_size: int = 2) -> dict[str, object]:
    observed: dict[str, object] = {}
    text = hwh_path.read_text(encoding="utf-8")
    for role, expectation in _expected_metadata(array_size).items():
        instance = str(expectation["instance"])
        parameters = _module_parameters(hwh_path, instance)
        expected_parameters = dict(expectation["parameters"])
        for name, expected in expected_parameters.items():
            actual = parameters.get(name)
            if actual is None or _parameter_value(actual) != _parameter_value(expected):
                raise OverlayVerificationError(
                    f"{instance} {name} expected {expected}, found {actual!r}"
                )
        observed[role] = {"instance": instance, "parameters": expected_parameters}
    required_connections = (
        'INSTANCE="npu_accelerator_0" PORT="m_axi_araddr"',
        'INSTANCE="npu_accelerator_0" PORT="m_axi_awaddr"',
        'INSTANCE="npu_accelerator_0" PORT="irq"',
        'SLAVEBUSINTERFACE="S_AXI_HP0"',
        'VALUE="100000000"',
    )
    missing = [connection for connection in required_connections if connection not in text]
    if missing:
        raise OverlayVerificationError(f"HWH is missing connectivity metadata: {missing}")
    return observed


def _artifact_paths(artifact_dir: Path) -> tuple[Path, Path, Path]:
    bit_path = artifact_dir / f"{ARTIFACT_STEM}.bit"
    hwh_path = artifact_dir / f"{ARTIFACT_STEM}.hwh"
    manifest_path = artifact_dir / f"{ARTIFACT_STEM}.manifest.json"
    for path in (bit_path, hwh_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise OverlayVerificationError(f"missing or empty artifact: {path}")
    return bit_path, hwh_path, manifest_path


def check_capabilities(capabilities: int) -> None:
    """The word reports the CVA6 scalar core and everything librv_npu_rt needs."""
    if isinstance(capabilities, bool) or not isinstance(capabilities, int):
        raise OverlayVerificationError("capabilities must be an integer")
    reported = Capability(capabilities & 0xFFFFFFFF)
    if not reported & Capability.RISCV_SCALAR_CORE:
        raise OverlayVerificationError(
            f"capabilities 0x{capabilities:08X} lack bit 7 (RISCV_SCALAR_CORE): not a CVA6 build"
        )
    missing = Capability(REQUIRED_CAPABILITIES) & ~reported
    if missing:
        raise OverlayVerificationError(f"missing required capabilities: {missing!s}")


def verify_artifacts(artifact_dir: Path) -> dict[str, object]:
    bit_path, hwh_path, manifest_path = _artifact_paths(artifact_dir)
    if not manifest_path.is_file():
        raise OverlayVerificationError("overlay provenance manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("target_part") != "xc7z020clg400-1":
        raise OverlayVerificationError("unsupported manifest schema or target part")
    if manifest.get("scalar_impl") != SCALAR_IMPL:
        raise OverlayVerificationError(
            f"manifest scalar_impl is {manifest.get('scalar_impl')!r}, expected {SCALAR_IMPL!r}"
        )
    check_capabilities(manifest.get("capabilities"))
    _validated_commit(str(manifest.get("source_commit", "")))
    for label, path in (("bit", bit_path), ("hwh", hwh_path)):
        record = manifest.get(label)
        if not isinstance(record, dict):
            raise OverlayVerificationError(f"manifest lacks {label} record")
        if record.get("name") != path.name or record.get("size") != path.stat().st_size:
            raise OverlayVerificationError(f"{label} name or size does not match manifest")
        if record.get("sha256") != _sha256(path):
            raise OverlayVerificationError(f"{label} hash does not match manifest")
    array_size = manifest.get("array_size", 2)
    if isinstance(array_size, bool) or not isinstance(array_size, int):
        raise OverlayVerificationError("manifest array_size must be 2, 8, or 16")
    observed = inspect_hwh(hwh_path, array_size=array_size)
    if manifest.get("metadata") != observed:
        raise OverlayVerificationError("HWH metadata differs from the build manifest")
    return manifest


def verify_device(read: Callable[[int], int]) -> int:
    """Check a loaded overlay through ``read(offset)`` on its CSR window
    (e.g. ``overlay.npu_accelerator_0.read``): magic, ABI major and bit 7.
    Returns the capability word."""
    magic = int(read(Register.MAGIC))
    if magic != ABI_MAGIC:
        raise OverlayVerificationError(f"unexpected ABI magic 0x{magic:08X}")
    major = int(read(Register.VERSION)) >> 16
    if major != ABI_MAJOR:
        raise OverlayVerificationError(f"unsupported ABI major {major}; expected {ABI_MAJOR}")
    capabilities = int(read(Register.CAPABILITIES))
    check_capabilities(capabilities)
    return capabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    arguments = parser.parse_args()
    verify_artifacts(arguments.artifact_dir.resolve())
    print("PASS: npu_accelerator (CVA6) BIT/HWH provenance and metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
