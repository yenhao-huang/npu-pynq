"""The NPU passes in ``npu-opt``: ``partition`` picks which contractions run
on the NPU, ``lower`` turns them into ISA programs and C runtime calls.

The LLVM toolchain belongs to src/export, which passes in how to find LLVM
(``llvm_bin``) and how to run a tool (``run``); this module imports neither.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[4]
NPU_OPT_SOURCE = Path(__file__).resolve().parent / "mlir"
NPU_OPT_BUILD = Path(os.environ.get("NPU_OPT_BUILD", str(REPO / "build/npu-opt")))
RUNTIME_C = Path(__file__).resolve().parents[1] / "driver/npu_rt.c"


class NpuOpt:
    def __init__(self, llvm_bin: Callable[[], Path], run: Callable[[list[str], str], None]) -> None:
        self.llvm_bin = llvm_bin
        self.run = run

    def npu_opt(self) -> str:
        """Build npu-opt on first use (CMake against the LLVM found above)."""
        exe = NPU_OPT_BUILD / "npu-opt"
        sources = list(NPU_OPT_SOURCE.rglob("*.cpp")) + list(NPU_OPT_SOURCE.rglob("*.td")) + \
            list(NPU_OPT_SOURCE.rglob("*.h")) + [NPU_OPT_SOURCE / "CMakeLists.txt"]
        if exe.exists() and exe.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
            return str(exe)
        bin_dir = self.llvm_bin()
        mlir_dir = bin_dir.parent / "lib/cmake/mlir"
        configure = ["cmake", "-S", str(NPU_OPT_SOURCE), "-B", str(NPU_OPT_BUILD),
                     f"-DMLIR_DIR={mlir_dir}", "-DCMAKE_BUILD_TYPE=Release"]
        if (bin_dir / "clang++").exists():
            configure += [f"-DCMAKE_CXX_COMPILER={bin_dir / 'clang++'}", f"-DCMAKE_C_COMPILER={bin_dir / 'clang'}"]
        self.run(configure, "configuring npu-opt")
        jobs = int(os.environ.get("NPU_BUILD_JOBS", str(os.cpu_count() or 4)))
        self.run(["cmake", "--build", str(NPU_OPT_BUILD), "-j", str(jobs)], "building npu-opt")
        return str(exe)

    def partition(self, module: Path, out: Path, report: Path, max_k: int = 256) -> None:
        self.run([self.npu_opt(), str(module), f"--npu-partition=report={report} max-k={max_k}", "-o", str(out)],
                 "npu-partition")

    def lower(self, module: Path, out: Path, weights: Path, programs: Path, rows: int = 16,
              columns: int = 16, max_k: int = 256, segment_bytes: int = 4 << 20) -> None:
        self.run([self.npu_opt(), str(module),
                  f"--npu-lower=weights={weights} programs={programs} rows={rows} columns={columns} max-k={max_k}"
                  f" segment-bytes={segment_bytes}",
                  "--canonicalize", "-o", str(out)], "npu-lower")
