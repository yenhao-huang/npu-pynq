"""MLIR/LLVM toolchain: tool discovery, targets, and the lowering pipeline.

``partition`` and ``lower`` run the NPU passes in ``npu-opt``; ``codegen``
runs the upstream pipeline (bufferization, LLVM dialect, LLVM IR, llc) and
links the CPU code with the C runtime into one shared library.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
NPU_OPT_SOURCE = REPO / "src/compiler/mlir"
NPU_OPT_BUILD = REPO / "build/npu-opt"
RUNTIME_C = REPO / "src/runtime/c/npu_rt.c"

LLVM_CANDIDATES = (
    os.environ.get("NPU_LLVM_BIN", ""),
    "/opt/homebrew/opt/llvm/bin",
    "/usr/lib/llvm-22/bin",
    "/usr/local/opt/llvm/bin",
)


class ToolchainError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    """Where the CPU part of a model runs."""

    name: str
    triple: str
    cpu: str
    features: str
    index_bits: int
    library: str  # file name of the linked model
    float_abi: str = ""

    @property
    def llc_args(self) -> list[str]:
        args = [f"-mtriple={self.triple}", "-relocation-model=pic", "-O3"]
        if self.cpu:
            args.append(f"-mcpu={self.cpu}")
        if self.features:
            args.append(f"-mattr={self.features}")
        if self.float_abi:
            args.append(f"-float-abi={self.float_abi}")
        return args


# The PYNQ-Z1's Cortex-A9 runs 32-bit hard-float Linux with NEON.
PYNQ = Target("pynq", "armv7a-unknown-linux-gnueabihf", "cortex-a9", "+neon,+vfp3",
              32, "model.so", float_abi="hard")


def host_target() -> Target:
    if sys.platform == "darwin":
        import platform
        arch = "arm64" if platform.machine() == "arm64" else "x86_64"
        return Target("host", f"{arch}-apple-macosx13.0", "", "", 64, "model.dylib")
    import platform
    arch = platform.machine()
    return Target("host", f"{arch}-unknown-linux-gnu", "", "", 64, "model.so")


TARGETS = {"pynq": PYNQ}


def target(name: str) -> Target:
    return host_target() if name == "host" else TARGETS[name]


def llvm_bin() -> Path:
    for candidate in LLVM_CANDIDATES:
        if candidate and (Path(candidate) / "mlir-opt").exists():
            return Path(candidate)
    found = shutil.which("mlir-opt")
    if found:
        return Path(found).parent
    raise ToolchainError("no LLVM 22 with MLIR found; set NPU_LLVM_BIN")


def tool(name: str) -> str:
    path = llvm_bin() / name
    if path.exists():
        return str(path)
    found = shutil.which(name) or shutil.which(f"{name}-22")
    if not found:
        raise ToolchainError(f"{name} not found")
    return found


def run(args: list[str], what: str) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ToolchainError(f"{what} failed ({' '.join(args[:2])} ...):\n{proc.stderr[-4000:]}")


def npu_opt() -> str:
    """Build npu-opt on first use (CMake against the LLVM found above)."""
    exe = NPU_OPT_BUILD / "npu-opt"
    sources = list(NPU_OPT_SOURCE.rglob("*.cpp")) + list(NPU_OPT_SOURCE.rglob("*.td")) + \
        list(NPU_OPT_SOURCE.rglob("*.h")) + [NPU_OPT_SOURCE / "CMakeLists.txt"]
    if exe.exists() and exe.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return str(exe)
    bin_dir = llvm_bin()
    mlir_dir = bin_dir.parent / "lib/cmake/mlir"
    configure = ["cmake", "-S", str(NPU_OPT_SOURCE), "-B", str(NPU_OPT_BUILD),
                 f"-DMLIR_DIR={mlir_dir}", "-DCMAKE_BUILD_TYPE=Release"]
    if (bin_dir / "clang++").exists():
        configure += [f"-DCMAKE_CXX_COMPILER={bin_dir / 'clang++'}", f"-DCMAKE_C_COMPILER={bin_dir / 'clang'}"]
    run(configure, "configuring npu-opt")
    run(["cmake", "--build", str(NPU_OPT_BUILD), "-j", str(os.cpu_count() or 4)], "building npu-opt")
    return str(exe)


def lld() -> list[str]:
    for name in ("ld.lld", "ld.lld-22"):
        found = shutil.which(name)
        if found:
            return [found]
    candidate = llvm_bin() / "ld.lld"
    if candidate.exists():
        return [str(candidate)]
    try:
        import ziglang  # noqa: F401  (LLVM's lld, packaged as a wheel)
    except ImportError as error:
        raise ToolchainError("no ld.lld; install lld or the ziglang wheel") from error
    return [sys.executable, "-m", "ziglang", "ld.lld"]


def cpu_pipeline(index_bits: int) -> list[str]:
    """Upstream passes from linalg-on-tensors (with NPU calls) to the LLVM dialect."""
    ib = f"index-bitwidth={index_bits}"
    return [
        "--canonicalize", "--cse",
        "--linalg-fuse-elementwise-ops",
        "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
        "--canonicalize",
        "--buffer-deallocation-pipeline",
        "--convert-bufferization-to-memref",
        "--convert-linalg-to-loops",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-scf-to-cf",
        f"--finalize-memref-to-llvm={ib}",
        "--convert-math-to-llvm",
        f"--convert-arith-to-llvm={ib}",
        f"--convert-cf-to-llvm={ib}",
        f"--convert-func-to-llvm={ib}",
        f"--convert-index-to-llvm={ib}",
        "--reconcile-unrealized-casts",
    ]


def partition(module: Path, out: Path, report: Path, max_k: int = 256) -> None:
    run([npu_opt(), str(module), f"--npu-partition=report={report} max-k={max_k}", "-o", str(out)],
        "npu-partition")


def lower(module: Path, out: Path, weights: Path, programs: Path, rows: int = 16,
          columns: int = 16, max_k: int = 256, segment_bytes: int = 4 << 20) -> None:
    run([npu_opt(), str(module),
         f"--npu-lower=weights={weights} programs={programs} rows={rows} columns={columns} max-k={max_k}"
         f" segment-bytes={segment_bytes}",
         "--canonicalize", "-o", str(out)], "npu-lower")


def codegen(module: Path, workdir: Path, tgt: Target) -> Path:
    """CPU code for ``tgt``: MLIR -> LLVM IR -> object -> shared library."""
    workdir.mkdir(parents=True, exist_ok=True)
    llvm_mlir = workdir / f"{tgt.name}.llvm.mlir"
    ll = workdir / f"{tgt.name}.ll"
    obj = workdir / f"{tgt.name}.model.o"
    rt_obj = workdir / f"{tgt.name}.npu_rt.o"
    run([tool("mlir-opt"), str(module), *cpu_pipeline(tgt.index_bits), "-o", str(llvm_mlir)],
        "mlir-opt lowering")
    run([tool("mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir), "-o", str(ll)], "mlir-translate")
    run([tool("llc"), *tgt.llc_args, "-filetype=obj", str(ll), "-o", str(obj)], "llc")
    clang = [tool("clang"), f"--target={tgt.triple}", "-O2", "-fPIC", "-ffreestanding",
             "-fno-builtin-memcpy", "-c", str(RUNTIME_C), "-o", str(rt_obj)]
    if tgt.cpu:
        clang.insert(3, f"-mcpu={tgt.cpu}")
    if tgt.float_abi:
        clang.insert(3, f"-mfloat-abi={tgt.float_abi}")
    run(clang, "compiling the NPU runtime")
    library = workdir / tgt.library
    if "apple" in tgt.triple:
        run([tool("clang"), f"--target={tgt.triple}", "-dynamiclib", "-o", str(library),
             str(obj), str(rt_obj)], "linking")
    elif tgt.name == "host":
        run([tool("clang"), "-shared", "-o", str(library), str(obj), str(rt_obj), "-lm"], "linking")
    else:
        run([*lld(), "-shared", "--hash-style=both", "-soname", tgt.library, "-o", str(library),
             str(obj), str(rt_obj)], "linking")
    return library
