"""Independent ONNX Runtime operators for host-side source-graph references.

Compiled packages do not import this module or require ONNX Runtime. The board
reference can evaluate floating operators with NumPy and integer work on NPU.
"""

from __future__ import annotations

from pathlib import Path
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
