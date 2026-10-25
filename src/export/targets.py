"""Compilation targets. A vendor's compiler backend lives under
src/vendors/<vendor>/compiler and gets one entry here; ``host`` needs no vendor."""

from src.export.compilers.mlir.toolchain import Target, host_target, llvm_bin, run
from src.vendors.npu_pynq.compiler import passes as npu_pynq_passes, target as npu_pynq_target

TARGETS = {"pynq": Target(**npu_pynq_target.CPU)}

# npu-opt, built and run with this package's LLVM toolchain.
npu_opt = npu_pynq_passes.NpuOpt(llvm_bin, run)


def target(name: str) -> Target:
    return host_target() if name == "host" else TARGETS[name]
