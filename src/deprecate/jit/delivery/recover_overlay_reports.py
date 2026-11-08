"""Recover the Vivado reports of a reused overlay from that build's own package.

An overlay reused from an earlier CD run arrives as that run's ``npu-build``
artifact. Runs before the artifact carried every report held only
``build_evidence.txt``, but the ResNet-18 package that same run built carries
all of them under ``reports/``, with digests in its manifest.

This copies the missing reports out of that package, and only if the package
provably belongs to this overlay build: its overlay record names the same BIT
and HWH digests, and its copy of ``build_evidence.txt`` is byte-identical to
the one beside the overlay. Every recovered report must match the digest the
package recorded. Otherwise nothing is written and the exit status is 1, so the
caller builds the overlay instead.

    python .github/cd/recover_overlay_reports.py \\
        --package-zip build/candidate/npu-resnet18-vX.Y.Z.zip \\
        --artifact-dir build/vivado/npu_matrix_16x16/artifacts \\
        --report-dir build/vivado/npu_matrix_16x16/reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

REQUIRED_REPORTS = (
    "build_evidence.txt",
    "drc_routed.rpt",
    "route_status.rpt",
    "timing_summary_routed.rpt",
    "utilization_impl.rpt",
    "utilization_synth.rpt",
)


class RecoveryError(RuntimeError):
    """The package cannot be shown to belong to this overlay build."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recover(package_zip: Path, artifact_dir: Path, report_dir: Path) -> list[str]:
    """Write the missing reports into report_dir; return the names written."""

    missing = [name for name in REQUIRED_REPORTS if not (report_dir / name).is_file()]
    if not missing:
        return []
    if "build_evidence.txt" in missing:
        raise RecoveryError("the overlay has no build_evidence.txt to bind to")
    try:
        bundle = zipfile.ZipFile(package_zip)
    except (OSError, zipfile.BadZipFile) as error:
        raise RecoveryError(f"cannot open the candidate package: {error}") from error
    with bundle:
        try:
            manifest = json.loads(bundle.read("package.manifest.json"))
            reports = json.loads(bundle.read("reports/reports.manifest.json"))
        except (KeyError, ValueError) as error:
            raise RecoveryError(f"candidate package manifest unreadable: {error}") from error

        overlay = manifest.get("overlay") or {}
        for key, filename in (("bit_sha256", "npu_matrix.bit"), ("hwh_sha256", "npu_matrix.hwh")):
            local = artifact_dir / filename
            if not local.is_file():
                raise RecoveryError(f"overlay file is missing: {filename}")
            if overlay.get(key) != _sha256(local.read_bytes()):
                raise RecoveryError(f"candidate package was built from another overlay ({filename})")

        recorded = {
            str(record.get("path")): record.get("sha256")
            for record in reports.get("files", [])
            if isinstance(record, dict)
        }
        evidence = (report_dir / "build_evidence.txt").read_bytes()
        if recorded.get("build_evidence.txt") != _sha256(evidence):
            raise RecoveryError("candidate package carries another build's evidence")

        payloads: dict[str, bytes] = {}
        for name in missing:
            try:
                data = bundle.read(f"reports/{name}")
            except KeyError as error:
                raise RecoveryError(f"candidate package lacks reports/{name}") from error
            if recorded.get(name) != _sha256(data):
                raise RecoveryError(f"reports/{name} differs from its recorded digest")
            payloads[name] = data

    # Every check passed before anything is written.
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (report_dir / name).write_bytes(data)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-zip", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        written = recover(arguments.package_zip, arguments.artifact_dir, arguments.report_dir)
    except RecoveryError as error:
        print(f"cannot recover the overlay reports: {error}", file=sys.stderr)
        return 1
    if written:
        print("recovered from the candidate package: " + ", ".join(written))
    else:
        print("all overlay reports present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
