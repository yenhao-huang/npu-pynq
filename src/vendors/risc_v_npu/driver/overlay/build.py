"""Write the provenance manifest for a freshly built CVA6 NPU BIT/HWH pair.

Modelled on src/vendors/npu_pynq/driver/overlay/build.py. The CVA6 build of
build_overlay.tcl (a hardware follow-up of plan 2026-1112-003; not wired yet)
is to run this right after Vivado copies ``npu_accelerator_cva6.bit`` and
``.hwh``. It records the source commit, tool version, array size, file hashes,
the HWH metadata, ``scalar_impl = "CVA6"`` and the capability word the build
reports (with bit 7, RISCV_SCALAR_CORE), then verifies the result with
verify.py. Standard library only.

    python src/vendors/risc_v_npu/driver/overlay/build.py --source-commit SHA --vivado-version V \
        --array-size 16 ARTIFACT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

try:
    from .verify import (
        REQUIRED_CAPABILITIES, SCALAR_IMPL, _artifact_paths, _sha256, _validated_commit,
        check_capabilities, inspect_hwh, verify_artifacts,
    )
except ImportError:  # run as a script by build_overlay.tcl
    from verify import (
        REQUIRED_CAPABILITIES, SCALAR_IMPL, _artifact_paths, _sha256, _validated_commit,
        check_capabilities, inspect_hwh, verify_artifacts,
    )

#: What csr.sv reports with SCALAR_IMPL = "CVA6": the CUSTOM build's 0x7B
#: plus bit 7 (CAP_SCALAR_CVA6).
CVA6_CAPABILITIES = 0x7B | (1 << 7)


def write_manifest(
    artifact_dir: Path,
    *,
    source_commit: str,
    vivado_version: str,
    array_size: int = 2,
    capabilities: int = CVA6_CAPABILITIES,
) -> dict[str, object]:
    bit_path, hwh_path, manifest_path = _artifact_paths(artifact_dir)
    check_capabilities(capabilities)
    metadata = inspect_hwh(hwh_path, array_size=array_size)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_commit": _validated_commit(source_commit),
        "vivado_version": vivado_version.strip(),
        "target_part": "xc7z020clg400-1",
        "array_size": array_size,
        "scalar_impl": SCALAR_IMPL,
        "capabilities": capabilities,
        "required_capabilities": int(REQUIRED_CAPABILITIES),
        "bit": {
            "name": bit_path.name,
            "size": bit_path.stat().st_size,
            "mtime_ns": bit_path.stat().st_mtime_ns,
            "sha256": _sha256(bit_path),
        },
        "hwh": {
            "name": hwh_path.name,
            "size": hwh_path.stat().st_size,
            "mtime_ns": hwh_path.stat().st_mtime_ns,
            "sha256": _sha256(hwh_path),
        },
        "metadata": metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _default_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--source-commit", default=_default_commit())
    parser.add_argument("--vivado-version", default="unknown")
    parser.add_argument("--array-size", type=int, choices=(2, 8, 16), default=16)
    arguments = parser.parse_args()
    artifact_dir = arguments.artifact_dir.resolve()
    write_manifest(
        artifact_dir,
        source_commit=arguments.source_commit,
        vivado_version=arguments.vivado_version,
        array_size=arguments.array_size,
    )
    verify_artifacts(artifact_dir)
    print("PASS: npu_accelerator (CVA6) BIT/HWH provenance and metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
