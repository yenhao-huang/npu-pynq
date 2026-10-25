"""ONNX Runtime float operators for the Python-driven execution paths.

``OrtOperators`` runs each operator through the onnxruntime Python package;
``NativeOrtOperators`` runs the same per-operator sessions through the ARM C
bridge (onnx_ort_bridge.c) via ctypes. Both take ``(op, args)`` and return the
operator's outputs. Compiled models call the C kernels in onnx_ops.c instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import ctypes as c
import numpy as np


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
