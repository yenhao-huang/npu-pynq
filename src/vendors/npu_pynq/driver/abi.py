"""ctypes mirrors of the C NPU runtime (npu_rt.c) and MLIR memref descriptors.

Field order and types must match the structs in npu_rt.c exactly; the
compiled model's entry points take memref descriptors through MLIR's C
interface (``_mlir_ciface_<name>``).
"""

from __future__ import annotations

import ctypes
import numpy as np


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
