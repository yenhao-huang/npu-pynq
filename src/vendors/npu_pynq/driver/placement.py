"""Place a package's weights, ISA programs and I/O tiles where the NPU can reach.

``place`` allocates the buffers for the chosen backend, relocates each
program's weight addresses against the I/O arena, and hands the C runtime
(npu_rt.c) the program and segment tables.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np

from src.vendors.npu_pynq.driver.abi import TASK_FIELDS, NpuProgram, NpuSegment
from src.vendors.npu_pynq.driver.memory import CmaBuffer, HostBuffer, MappedWeights
from src.vendors.npu_pynq.driver.isa import IsaDevice


MASK32 = 0xFFFFFFFF
# Pretend physical addresses for the simulator; any distinct 32-bit ranges do.
SIM_WEIGHTS_PHYS = 0x10000000
SIM_IO_PHYS = 0x60000000
SIM_STAGE_PHYS = 0x70000000
SIM_WORDS_PHYS = 0x78000000


def place(lib: ctypes.CDLL, root: Path, manifest: dict, backend: str, overlay: Any,
          stream_weights: bool | None) -> SimpleNamespace:
    """Allocate NPU memory, relocate programs and initialize ``lib``'s runtime.

    Returns the buffers (weights, io, stage, words), the resolved
    ``stream_weights`` flag, the task table, and the ctypes records the C
    runtime keeps pointers to (they must outlive the library calls).
    """
    table = json.loads((root / "programs.json").read_text())
    words = np.fromfile(root / "programs.bin", dtype="<u8")
    weights_path = root / "weights.bin"
    tasks = table["tasks"]
    segments = [g for t in tasks for g in t["segments"]]
    layouts = manifest.get("weight_layouts", {})
    packed = {t["weight"] for t in tasks if layouts.get(t["weight"], "").startswith("npu_b4")}
    io_bytes = max([t["c_base"] + t["c_bytes"] for t in tasks], default=64)
    stage_bytes = max([g["b_bytes"] for g in segments], default=8)
    weight_bytes = weights_path.stat().st_size
    if stream_weights is None:
        stream_weights = weight_bytes > manifest.get('resident_limit', 96 << 20)
    if packed:
        stream_weights = True  # packed INT4 blocks are widened into the stage window

    if backend == "pynq":
        device = IsaDevice(overlay)
        if manifest.get('limits') != vars(device.limits):
            raise ValueError('Compiled package geometry differs from the ISA overlay')
        mmio_ip = getattr(overlay, "npu_accelerator_0")
        mmio_addr = mmio_ip.mmio.array.ctypes.data
        if stream_weights is None:
            stream_weights = weight_bytes > manifest.get("resident_limit", 96 << 20)
        if stream_weights:
            # Weights stay in the page cache; each segment's block is copied
            # into a CMA stage window just before its program runs.
            weights = MappedWeights(weights_path)
        else:
            weights = CmaBuffer(weight_bytes, cacheable=True)
            weights.array[:] = np.fromfile(weights_path, dtype=np.uint8)
        io = CmaBuffer(io_bytes, cacheable=False)
        stage = CmaBuffer(stage_bytes, cacheable=False) if stream_weights else None
        code = CmaBuffer(words.nbytes, cacheable=True)
    else:
        mmio_addr = None
        stream_weights = bool(stream_weights)
        if stream_weights:
            weights = MappedWeights(weights_path)
        else:
            weights = HostBuffer(weight_bytes, SIM_WEIGHTS_PHYS)
            weights.array[:] = np.fromfile(weights_path, dtype=np.uint8)
        io = HostBuffer(io_bytes, SIM_IO_PHYS)
        stage = HostBuffer(stage_bytes, SIM_STAGE_PHYS) if stream_weights else None
        code = HostBuffer(words.nbytes, SIM_WORDS_PHYS)
    weights.flush()

    # ADDR_B words hold offsets inside their segment's weight block; make
    # them relative to DATA_BASE (the I/O arena), wrapping at 32 bits.
    relocated = words.copy()
    io_phys = io.physical_address
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
            block = (stage.physical_address if stream_weights
                     else weights.physical_address + g["weight_offset"])
            delta = (block - io_phys) & MASK32
            for r in g["relocations"]:
                index = g["word_offset"] + r
                word = int(relocated[index])
                relocated[index] = np.uint64((word & ~MASK32 & 0xFFFFFFFFFFFFFFFF) | (((word & MASK32) + delta) & MASK32))
            sr = seg_records[cursor]
            sr.word_offset, sr.length, sr.b_bytes = g["word_offset"], g["length"], g["b_bytes"]
            sr.packed = 1 if is_packed else 0
            src = (t["weight_offset"] + (g["weight_offset"] - t["weight_offset"]) // 2) if is_packed else g["weight_offset"]
            sr.stage_src = (weights.address + src) if stream_weights else None
            cursor += 1
    code.array[:] = relocated.view(np.uint8)
    code.flush()

    lib.npu_rt_init.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
    lib.npu_rt_add_region.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    lib.npu_rt_init(1 if backend == "pynq" else 0, mmio_addr, io.address, io_phys,
                    io.array.nbytes, code.address, code.physical_address,
                    stage.address if stage else None,
                    ctypes.addressof(records), len(tasks), ctypes.addressof(seg_records), len(segments))
    if backend != "pynq":
        if not stream_weights:
            lib.npu_rt_add_region(weights.physical_address, weights.address,
                                  weights.array.nbytes)
        if stage:
            lib.npu_rt_add_region(stage.physical_address, stage.address,
                                  stage.array.nbytes)
    return SimpleNamespace(weights=weights, io=io, stage=stage, words=code,
                           stream_weights=stream_weights, tasks=tasks,
                           records=records, segments=seg_records)
