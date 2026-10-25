"""Load a package compiled by src/export and call its entry points.

A package holds the CPU code for each target (one shared library that also
contains the C NPU runtime), the ISA programs and their relocations, and the
weight blob. ``Package`` loads the library, lets src/vendors/npu_pynq/driver/placement.py
put weights, programs and I/O tiles where the NPU can reach, and calls entry
points through MLIR's C interface (``_mlir_ciface_<name>`` taking memref
descriptors).

Backends:

``sim``   host memory; the runtime interprets programs (src/vendors/npu_pynq/isa semantics)
``pynq``  the FPGA overlay; programs, weights and tiles in CMA buffers
"""

from __future__ import annotations

import ctypes
import json
import hashlib
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np

from src.vendors.npu_pynq.driver.abi import NpuStats, descriptor
from src.vendors.npu_pynq.driver.placement import place


class RuntimeFault(RuntimeError):
    pass


class Package:
    def __init__(self, package: str | Path, backend: str = "sim", *, overlay: Any = None,
                 stream_weights: bool | None = None, target: str | None = None) -> None:
        self.root = Path(package)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        meta = self.manifest['meta']
        for name, expected in meta.get('native_kernel_hashes', {}).items():
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Native kernel identity differs: ' + name)
        self.hooks = meta.get('runtime_hooks', {})
        # Packages exported before runtime hooks became part of ModelGraph.
        if not self.hooks and meta.get('native_float_runtime') == 'onnxruntime-1.20.1':
            self.hooks = {'preload':['libonnxruntime.so.1.20.1'],
                'initialize':'official_compiled_init', 'initialize_path':'float-ops',
                'reset':'official_compiled_clear', 'status':'official_compiled_status',
                'error':'official_ort_error'}
        self.dependencies = [ctypes.CDLL(str((self.root / name).resolve()), mode=ctypes.RTLD_GLOBAL)
                             for name in self.hooks.get('preload', ())]
        self.backend = backend
        target = target or ("pynq" if backend == "pynq" else "host")
        if target not in self.manifest['libraries'] and len(self.manifest['libraries']) == 1:
            target = next(iter(self.manifest['libraries']))
        library = self.root / self.manifest["libraries"][target]
        if target == "pynq":
            # The model leaves libm/libgcc symbols for the loader to resolve.
            for dep in ("libm.so.6", "libgcc_s.so.1"):
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(str(library))
        self.lib.npu_rt_stats.restype = ctypes.POINTER(NpuStats)
        placed = place(self.lib, self.root, self.manifest, backend, overlay, stream_weights)
        self.weights, self.io, self.stage, self.words = placed.weights, placed.io, placed.stage, placed.words
        self.stream_weights, self.tasks = placed.stream_weights, placed.tasks
        self._records, self._segments = placed.records, placed.segments
        if self.hooks:
            error = getattr(self.lib, self.hooks['error'])
            error.restype = ctypes.c_char_p
            self.native_error = error
            self.native_status = getattr(self.lib, self.hooks['status'])
            self.native_status.restype = ctypes.c_int
            initialize = getattr(self.lib, self.hooks['initialize'])
            initialize.argtypes = [ctypes.c_char_p]
            initialize.restype = ctypes.c_int
            if initialize(str((self.root / self.hooks['initialize_path']).resolve()).encode()):
                raise RuntimeFault(error().decode())

    def reset(self):
        """Reset optional native operator state before a new generation."""
        if self.hooks:
            getattr(self.lib, self.hooks['reset'])()

    def stats(self) -> dict[str, int]:
        s = self.lib.npu_rt_stats().contents
        return {"calls": s.calls, "jobs": s.jobs, "cycles": s.cycles, "macs": s.macs,
                "polls": s.polls, "error": s.error, "error_program": s.error_program}

    # ---------------------------------------------------------------- calls
    @property
    def weight_arena(self) -> np.ndarray:
        return self.weights.array

    def call(self, name: str, *args: Any) -> None:
        """Call ``name`` through its C interface. Arrays pass as memrefs, the
        string "weights" as the weight arena, ints as index scalars."""
        self.reset()
        fn = getattr(self.lib, f"_mlir_ciface_{name}")
        c_args = []
        keep = []
        for arg in args:
            if isinstance(arg, str) and arg == "weights":
                arg = self.weights.array
            if isinstance(arg, np.ndarray):
                desc = descriptor(arg)
                keep.append(desc)
                c_args.append(ctypes.byref(desc))
            elif isinstance(arg, (int, np.integer)):
                c_args.append(ctypes.c_ssize_t(int(arg)))
            else:
                raise TypeError(f"unsupported argument {type(arg)}")
        fn.restype = None
        fn(*c_args)
        if self.hooks and self.native_status():
            raise RuntimeFault(self.native_error().decode())
        stats = self.lib.npu_rt_stats().contents
        if stats.error:
            raise RuntimeFault(f"NPU task {stats.error_program} failed with ISA error 0x{stats.error:02x}")


def platform_backend() -> str:
    return "pynq" if sys.platform.startswith("linux") and Path("/etc/profile.d/pynq_venv.sh").exists() else "sim"


def onnx_asset_dir(root: str | Path, name: str) -> Path:
    """Resolve current asset names, accepting pre-rename local model stores."""
    root = Path(root)
    current = root / name
    legacy = {'onnx':'official-onnx', 'onnx-graph':'official-graph',
              'onnx-ops':'official-ort-ops', 'onnx-runtime':'official-ort'}
    previous = root / legacy.get(name, name)
    return previous if not current.exists() and previous.exists() else current


def model_env(name: str):
    """Read canonical model settings, with old notebook settings as fallback."""
    legacy = {'NPU_ONNX_MODEL':'NPU_OFFICIAL_MODEL',
              'NPU_COMPILED_PACKAGE':'NPU_OFFICIAL_COMPILED'}
    return os.environ.get(name, os.environ.get(legacy.get(name, name)))
