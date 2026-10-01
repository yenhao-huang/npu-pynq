"""Rehearse a release's package and board path without Vivado or a board.

Builds the real ResNet-18 release package from a validated model workspace,
with placeholder overlay artifacts, extracts it the way the board does, and
runs CD's two board steps from inside the extracted package: the deploy check
(run_on_board.py --verify-only) and the real-image inference, against a NumPy
8 x 8 NPU that enforces the hardware tile limits. Everything but the FPGA runs:
tree verification, overlay binding, 135,290 tiled jobs, capture comparison
against the host record, demo decoding and evidence writing.

A full CD run spends two to three hours in Vivado before any of this executes,
so run this first whenever packaging, the board scripts or the model code
change. It takes about five minutes.

    python .codex/skills/deploy/release-npu-pynq/references/scripts/rehearse_release.py \\
        --model-dir examples/resnet18/model

Run it from the repository root. It writes only under --work-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BOARD_DRIVER = textwrap.dedent(
    '''
    import json, sys
    from pathlib import Path
    import numpy as np

    root = Path.cwd()
    sys.path.insert(0, str(root))
    import run_on_board as board
    import accept_image_on_board as image
    from src.runtime import NPURuntime
    from src.runtime.npu import PhysicalJobMetrics
    from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES

    class NumpyNPU(NPURuntime):
        """The 8 x 8 overlay's limits, with NumPy doing the arithmetic."""
        def __init__(self):
            self.max_m, self.max_n, self.max_k = 8, 8, 256
            self.abi_major, self.capabilities = REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES
            self.last_metrics = None
        def run(self, a, b, *, hardware_timeout_cycles=1_000_000, software_timeout=1.0):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape[0] > 8 or b.shape[1] > 8 or a.shape[1] > 256:
                raise ValueError(f"tile exceeds the 8x8 overlay: {a.shape} x {b.shape}")
            self.last_metrics = PhysicalJobMetrics(cycles=a.shape[1] + 16)
            return a.astype(np.int32) @ b.astype(np.int32)

    npu = NumpyNPU()
    board.load_pynq_runtime = lambda bit: npu
    image.load_pynq_runtime = lambda bit: npu
    # CD's one board check: the real-image inference.
    result = image.accept_image(package_root=root, evidence_path=root / "image-acceptance.json")
    top, runtime = result["top_k"][0], result["runtime"]
    print(f"PASS image acceptance: {top['name']} ({top['probability']:.2%}), verdict {result['verdict']}, "
          f"{runtime['physical_jobs']:,} jobs, {runtime['mac_count']:,} MACs")
    '''
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("build/rehearsal"))
    parser.add_argument("--release-tag", default="v0.0.0")
    arguments = parser.parse_args()

    repository = Path.cwd().resolve()
    sys.path.insert(0, str(repository))
    from src.runtime.verify_overlay import write_manifest
    from src.test.tests.test_verify_overlay import HWH

    packager = _load("rehearsal_packager", repository / "examples/resnet18/package_example.py")
    work = arguments.work_dir.resolve()
    shutil.rmtree(work, ignore_errors=True)
    artifacts, reports, staging = work / "artifacts", work / "reports", work / "staging"
    for folder in (artifacts, reports, staging):
        folder.mkdir(parents=True)

    # Placeholder overlay: real formats, fake bitstream. The NumPy NPU stands in for it.
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (artifacts / "npu_matrix.bit").write_bytes(b"rehearsal placeholder bitstream")
    (artifacts / "npu_matrix.hwh").write_text(HWH, encoding="utf-8")
    write_manifest(artifacts, source_commit=commit, vivado_version="rehearsal")
    (reports / "build_evidence.txt").write_text(
        "vivado=rehearsal\npart=xc7z020clg400-1\nrows=8\ncolumns=8\nwns=0.000\n"
        f"setup_failing_paths=0\ndrc_errors=0\nbit=b\nhwh=h\nsource_commit={commit}\n",
        encoding="utf-8",
    )
    for name in packager.DELIVERY_REPORT_FILES[1:]:
        (reports / name).write_text("rehearsal\n", encoding="utf-8")

    archive = work / f"npu-resnet18-{arguments.release_tag}.zip"
    packager.build_delivery_archive(
        repository_root=repository, artifact_dir=artifacts, report_dir=reports,
        output_archive=archive, release_tag=arguments.release_tag, source_commit=commit,
        model_dir=arguments.model_dir, source_metadata_path=repository / "examples/resnet18/model-source.json",
    )
    print(f"PASS package: {archive.stat().st_size // 1024} KiB")

    # Exactly what the board does: copy, extract with Python, verify, then classify.
    shutil.copy2(archive, staging / archive.name)
    subprocess.run([sys.executable, "-m", "zipfile", "-e", archive.name, "package"], cwd=staging, check=True)
    package = staging / "package"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    # The deploy step's check, exactly as resnet18_deploy.ps1 runs it.
    check = subprocess.run(
        [sys.executable, "-B", "run_on_board.py", "--package-root", ".",
         "--package-archive", f"../{archive.name}", "--archive-sha256", digest,
         "--release-tag", arguments.release_tag, "--verify-only"],
        cwd=package, env=environment, text=True,
    )
    if check.returncode:
        return check.returncode
    return subprocess.run([sys.executable, "-c", BOARD_DRIVER], cwd=package, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
