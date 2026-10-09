"""Load and call a model compiled by src/compiler on the host or the PYNQ-Z1.

A package holds the CPU code for each target (one shared library that also
contains the C NPU runtime), the ISA programs and their relocations, and the
weight blob. ``CompiledModel`` places the weights, programs and I/O tiles in
memory the NPU can reach, relocates each program's weight addresses, hands
the runtime those addresses, and calls entry points through MLIR's C
interface (``_mlir_ciface_<name>`` taking memref descriptors).

Backends:

``sim``   host memory; the runtime interprets programs (src/isa semantics)
``pynq``  the FPGA overlay; programs, weights and tiles in CMA buffers
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

MASK32 = 0xFFFFFFFF
# Pretend physical addresses for the simulator; any distinct 32-bit ranges do.
SIM_WEIGHTS_PHYS = 0x10000000
SIM_IO_PHYS = 0x60000000
SIM_STAGE_PHYS = 0x70000000
SIM_WORDS_PHYS = 0x78000000


TASK_FIELDS = ("m", "n", "k", "tm", "tn", "tk", "a_bytes", "c_base", "c_bytes")


class NpuProgram(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in (*TASK_FIELDS, "first_segment", "segments")]


class NpuSegment(ctypes.Structure):
    _fields_ = [("word_offset", ctypes.c_int32), ("length", ctypes.c_int32), ("b_bytes", ctypes.c_int32),
                ("packed", ctypes.c_int32), ("stage_src", ctypes.c_void_p)]


class NpuRegion(ctypes.Structure):
    _fields_ = [("phys", ctypes.c_uint32), ("host", ctypes.c_void_p), ("size", ctypes.c_uint32)]


class NpuStats(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int32), ("mmio", ctypes.c_void_p), ("io", ctypes.c_void_p),
        ("io_phys", ctypes.c_uint32), ("io_size", ctypes.c_uint32),
        ("words", ctypes.c_void_p), ("words_phys", ctypes.c_uint32), ("stage", ctypes.c_void_p),
        ("programs", ctypes.c_void_p), ("count", ctypes.c_int32),
        ("segments", ctypes.c_void_p), ("nsegments", ctypes.c_int32),
        ("regions", NpuRegion * 4), ("nregions", ctypes.c_int32),
        ("calls", ctypes.c_int64), ("jobs", ctypes.c_int64), ("cycles", ctypes.c_int64),
        ("macs", ctypes.c_int64), ("polls", ctypes.c_int64),
        ("error", ctypes.c_int32), ("error_program", ctypes.c_int32),
    ]


class RuntimeFault(RuntimeError):
    pass


_DESCRIPTORS: dict[int, type] = {}


def descriptor_type(rank: int) -> type:
    """ctypes layout of an MLIR ranked memref descriptor (index = intptr_t)."""
    if rank not in _DESCRIPTORS:
        fields = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                  ("offset", ctypes.c_ssize_t)]
        if rank:
            fields += [("sizes", ctypes.c_ssize_t * rank), ("strides", ctypes.c_ssize_t * rank)]
        _DESCRIPTORS[rank] = type(f"MemRef{rank}D", (ctypes.Structure,), {"_fields_": fields})
    return _DESCRIPTORS[rank]


def descriptor(array: np.ndarray) -> ctypes.Structure:
    if not array.flags.c_contiguous:
        raise ValueError("memref arguments must be C-contiguous")
    desc = descriptor_type(array.ndim)()
    desc.allocated = desc.aligned = array.ctypes.data
    desc.offset = 0
    if array.ndim:
        for i, (size, stride) in enumerate(zip(array.shape, array.strides)):
            desc.sizes[i] = size
            desc.strides[i] = stride // array.itemsize
    return desc


class MappedWeights:
    """weights.bin mapped read-only: pages load on demand and are never copied."""

    def __init__(self, path: Path) -> None:
        self.array = np.memmap(path, dtype=np.uint8, mode="r") if path.stat().st_size else np.zeros(8, np.uint8)
        self.physical_address = 0

    @property
    def address(self) -> int:
        return self.array.ctypes.data

    def flush(self) -> None:
        pass


class HostBuffer:
    """Page-aligned host memory standing in for a CMA buffer."""

    def __init__(self, nbytes: int, phys: int) -> None:
        self._raw = np.zeros(nbytes + 4096, np.uint8)
        start = (-self._raw.ctypes.data) % 4096
        self.array = self._raw[start:start + nbytes]
        self.physical_address = phys

    @property
    def address(self) -> int:
        return self.array.ctypes.data

    def flush(self) -> None:
        pass


class CmaBuffer:
    """A pynq.allocate buffer; uncached ones need no flush/invalidate."""

    def __init__(self, nbytes: int, cacheable: bool) -> None:
        from pynq import allocate
        self.buffer = allocate(shape=(max(nbytes, 8),), dtype=np.uint8, cacheable=cacheable)
        self.array = np.asarray(self.buffer)[:nbytes]
        self.physical_address = int(self.buffer.physical_address)

    @property
    def address(self) -> int:
        return self.array.ctypes.data

    def flush(self) -> None:
        self.buffer.flush()


class CompiledModel:
    def __init__(self, package: str | Path, backend: str = "sim", *, overlay: Any = None,
                 stream_weights: bool | None = None) -> None:
        self.root = Path(package)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.backend = backend
        target = "pynq" if backend == "pynq" else "host"
        library = self.root / self.manifest["libraries"][target]
        if backend == "pynq":
            # The model leaves libm/libgcc symbols for the loader to resolve.
            for dep in ("libm.so.6", "libgcc_s.so.1"):
                ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(str(library))
        self.lib.npu_rt_stats.restype = ctypes.POINTER(NpuStats)
        self._setup_npu(overlay, stream_weights)

    # ------------------------------------------------------------------ npu
    def _setup_npu(self, overlay: Any, stream_weights: bool | None) -> None:
        table = json.loads((self.root / "programs.json").read_text())
        words = np.fromfile(self.root / "programs.bin", dtype="<u8")
        weights_path = self.root / "weights.bin"
        tasks = table["tasks"]
        segments = [g for t in tasks for g in t["segments"]]
        layouts = self.manifest.get("weight_layouts", {})
        packed = {t["weight"] for t in tasks if layouts.get(t["weight"], "").startswith("npu_b4")}
        io_bytes = max([t["c_base"] + t["c_bytes"] for t in tasks], default=64)
        stage_bytes = max([g["b_bytes"] for g in segments], default=8)
        weight_bytes = weights_path.stat().st_size
        if packed:
            stream_weights = True  # packed INT4 blocks are widened into the stage window

        if self.backend == "pynq":
            mmio_ip = getattr(overlay, "npu_accelerator_0")
            mmio_addr = mmio_ip.mmio.array.ctypes.data
            if stream_weights is None:
                stream_weights = weight_bytes > self.manifest.get("resident_limit", 96 << 20)
            if stream_weights:
                # Weights stay in the page cache; each segment's block is copied
                # into a CMA stage window just before its program runs.
                self.weights = MappedWeights(weights_path)
            else:
                self.weights = CmaBuffer(weight_bytes, cacheable=True)
                self.weights.array[:] = np.fromfile(weights_path, dtype=np.uint8)
            self.io = CmaBuffer(io_bytes, cacheable=False)
            self.stage = CmaBuffer(stage_bytes, cacheable=False) if stream_weights else None
            self.words = CmaBuffer(words.nbytes, cacheable=True)
        else:
            mmio_addr = None
            stream_weights = bool(stream_weights)
            if stream_weights:
                self.weights = MappedWeights(weights_path)
            else:
                self.weights = HostBuffer(weight_bytes, SIM_WEIGHTS_PHYS)
                self.weights.array[:] = np.fromfile(weights_path, dtype=np.uint8)
            self.io = HostBuffer(io_bytes, SIM_IO_PHYS)
            self.stage = HostBuffer(stage_bytes, SIM_STAGE_PHYS) if stream_weights else None
            self.words = HostBuffer(words.nbytes, SIM_WORDS_PHYS)
        self.stream_weights = stream_weights
        self.weights.flush()

        # ADDR_B words hold offsets inside their segment's weight block; make
        # them relative to DATA_BASE (the I/O arena), wrapping at 32 bits.
        relocated = words.copy()
        io_phys = self.io.physical_address
        records = (NpuProgram * max(len(tasks), 1))()
        seg_records = (NpuSegment * max(len(segments), 1))()
        cursor = 0
        for i, t in enumerate(tasks):
            rec = records[i]
            for name in TASK_FIELDS:
                setattr(rec, name, int(t[name]))
            rec.first_segment = cursor
            rec.segments = len(t["segments"])
            is_packed = t["weight"] in packed
            for g in t["segments"]:
                block = (self.stage.physical_address if stream_weights
                         else self.weights.physical_address + g["weight_offset"])
                delta = (block - io_phys) & MASK32
                for r in g["relocations"]:
                    index = g["word_offset"] + r
                    word = int(relocated[index])
                    relocated[index] = np.uint64((word & ~MASK32 & 0xFFFFFFFFFFFFFFFF) | (((word & MASK32) + delta) & MASK32))
                sr = seg_records[cursor]
                sr.word_offset, sr.length, sr.b_bytes = g["word_offset"], g["length"], g["b_bytes"]
                sr.packed = 1 if is_packed else 0
                src = (t["weight_offset"] + (g["weight_offset"] - t["weight_offset"]) // 2) if is_packed else g["weight_offset"]
                sr.stage_src = (self.weights.address + src) if stream_weights else None
                cursor += 1
        self.words.array[:] = relocated.view(np.uint8)
        self.words.flush()
        self._records, self._segments = records, seg_records
        self.tasks = tasks

        lib = self.lib
        lib.npu_rt_init.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
        lib.npu_rt_add_region.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        lib.npu_rt_init(1 if self.backend == "pynq" else 0, mmio_addr, self.io.address, io_phys,
                        self.io.array.nbytes, self.words.address, self.words.physical_address,
                        self.stage.address if self.stage else None,
                        ctypes.addressof(records), len(tasks), ctypes.addressof(seg_records), len(segments))
        if self.backend != "pynq":
            if not stream_weights:
                lib.npu_rt_add_region(self.weights.physical_address, self.weights.address,
                                      self.weights.array.nbytes)
            if self.stage:
                lib.npu_rt_add_region(self.stage.physical_address, self.stage.address,
                                      self.stage.array.nbytes)

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
        stats = self.lib.npu_rt_stats().contents
        if stats.error:
            raise RuntimeFault(f"NPU task {stats.error_program} failed with ISA error 0x{stats.error:02x}")


def platform_backend() -> str:
    return "pynq" if sys.platform.startswith("linux") and Path("/etc/profile.d/pynq_venv.sh").exists() else "sim"
