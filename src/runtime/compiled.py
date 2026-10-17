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
import hashlib
from types import SimpleNamespace
from pathlib import Path
import sys
from typing import Any
import numpy as np
import ctypes as c


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
        self._setup_npu(overlay, stream_weights)
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
        if stream_weights is None:
            stream_weights = weight_bytes > self.manifest.get('resident_limit', 96 << 20)
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


ORT_OPS = {'SimplifiedLayerNormalization', 'SkipSimplifiedLayerNormalization',
           'DynamicQuantizeLinear', 'RotaryEmbedding', 'GroupQueryAttention', 'Sigmoid'}


class OrtOperators:
    def __init__(self, directory):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.sessions = {key: ort.InferenceSession(str(Path(directory)/(key+'.onnx')),opts,
                                                  providers=['CPUExecutionProvider']) for key in ORT_OPS}

    def __call__(self, op, args):
        session = self.sessions[op]
        inputs = {'x'+str(i):np.asarray(v) for i,v in enumerate(args) if v is not None}
        values = session.run(None, inputs)
        # Restore omitted optional slots so graph-output positions remain unchanged.
        slots = {'y'+str(i):v for i,v in zip([int(o.name[1:]) for o in session.get_outputs()],values)}
        maximum = max(int(n[1:]) for n in slots)
        result = tuple(slots.get('y'+str(i)) for i in range(maximum+1))
        return result[0] if len(result)==1 else result


class Output(c.Structure):
    _fields_ = [('data',c.c_void_p),('dimensions',c.c_int64*8),('rank',c.c_size_t),
                ('type',c.c_int),('value',c.c_void_p)]


class NativeOrtOperators:
    def __init__(self,directory,library):
        self.lib = c.CDLL(str(library))
        prefix = 'onnx_ort_' if hasattr(self.lib, 'onnx_ort_version') else 'official_ort_'
        self.bridge = SimpleNamespace(**{name:getattr(self.lib, prefix+name) for name in
            ('version','output_size','create','error','run','free_outputs','close')})
        self.bridge.version.restype=c.c_char_p
        self.version=self.bridge.version().decode()
        assert self.version=='1.20.1', 'Unexpected ONNX Runtime version: '+self.version
        self.bridge.output_size.restype = c.c_size_t
        assert self.bridge.output_size()==c.sizeof(Output), 'ORT bridge struct ABI mismatch'
        self.bridge.create.argtypes = [c.c_char_p]
        self.bridge.create.restype = c.c_void_p
        self.bridge.error.restype = c.c_char_p
        self.bridge.run.argtypes = [c.c_void_p,c.c_size_t,c.POINTER(c.c_char_p),
            c.POINTER(c.c_void_p),c.POINTER(c.c_size_t),c.POINTER(c.POINTER(c.c_int64)),
            c.POINTER(c.c_size_t),c.POINTER(c.c_int),c.c_size_t,c.POINTER(c.c_char_p),c.POINTER(Output)]
        self.bridge.free_outputs.argtypes = [c.c_size_t,c.POINTER(Output)]
        self.bridge.close.argtypes = [c.c_void_p]
        self.sessions = {}
        for op in ORT_OPS:
            session = self.bridge.create(str(Path(directory)/(op+'.onnx')).encode())
            if not session:
                self.close()
                raise RuntimeError(self.bridge.error().decode())
            self.sessions[op] = session

    def __call__(self,op,args):
        arrays = [(i,np.ascontiguousarray(v) if np.asarray(v).ndim else np.asarray(v))
                  for i,v in enumerate(args) if v is not None]
        n = len(arrays)
        names = (c.c_char_p*n)(*[('x'+str(i)).encode() for i,_ in arrays])
        data = (c.c_void_p*n)(*[v.ctypes.data for _,v in arrays])
        sizes = (c.c_size_t*n)(*[v.nbytes for _,v in arrays])
        storage = [(c.c_int64*v.ndim)(*v.shape) for _,v in arrays]
        shapes = (c.POINTER(c.c_int64)*n)(*[c.cast(s,c.POINTER(c.c_int64)) for s in storage])
        ranks = (c.c_size_t*n)(*[v.ndim for _,v in arrays])
        codes = {np.dtype('float32'):1,np.dtype('int64'):7,np.dtype('int32'):6,np.dtype('uint8'):2}
        types = (c.c_int*n)(*[codes[v.dtype] for _,v in arrays])
        indices = [0,3] if op=='SkipSimplifiedLayerNormalization' else [0,1,2] if op in ('DynamicQuantizeLinear','GroupQueryAttention') else [0]
        out_names = (c.c_char_p*len(indices))(*[('y'+str(i)).encode() for i in indices])
        outputs = (Output*len(indices))()
        failed = self.bridge.run(self.sessions[op],n,names,data,sizes,shapes,ranks,types,len(indices),out_names,outputs)
        try:
            if failed: raise RuntimeError(self.bridge.error().decode())
            values = [None]*(max(indices)+1)
            kinds = {1:c.c_float,2:c.c_uint8,6:c.c_int32,7:c.c_int64}
            for slot,item in zip(indices,outputs):
                shape = tuple(item.dimensions[:item.rank])
                length = int(np.prod(shape))
                pointer = c.cast(item.data,c.POINTER(kinds[item.type]))
                values[slot] = np.ctypeslib.as_array(pointer,shape=(length,)).reshape(shape).copy()
            return values[0] if len(values)==1 else tuple(values)
        finally:
            self.bridge.free_outputs(len(indices),outputs)

    def close(self):
        for session in self.sessions.values(): self.bridge.close(session)
        self.sessions.clear()
