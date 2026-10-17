"""Physical integer and full-graph smoke probes, with durable evidence."""
from pathlib import Path
import sys
import argparse
import time
import numpy as np

deploy = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(deploy.parents[2]))
from pynq import Overlay
from official_npu import NpuGemm
from official_engine import ONNXReferenceEngine
from acceptance import save, memory

parser=argparse.ArgumentParser()
parser.add_argument('--arithmetic-only',action='store_true')
parser.add_argument('--native-ort',action='store_true')
args=parser.parse_args()

overlay = Overlay(str(deploy / "hardware/overlay/npu_matrix.bit"), download=False)
npu = NpuGemm(overlay)
rng = np.random.default_rng(126)
for shape in ((1,576,192), (19,576,576), (1,1536,576)):
    m,k,n = shape
    a = rng.integers(-128,128,(m,k),dtype=np.int8)
    b = rng.integers(-128,128,(k,n),dtype=np.int8)
    start = time.monotonic()
    actual = npu(a,b)
    np.testing.assert_array_equal(actual, a.astype(np.int32) @ b.astype(np.int32))
    print("INTEGER PASS",shape,time.monotonic()-start, npu.stats, flush=True)
if args.arithmetic_only:
    npu.close()
    sys.exit(0)
float_ops=None
if args.native_ort:
    from onnx_ort_ctypes import NativeOrtOperators
    float_ops=NativeOrtOperators(deploy/'packages/onnx-ops',deploy/'packages/onnx-runtime/libonnx_ort_bridge.so')
engine = ONNXReferenceEngine(deploy,npu,float_ops)
start = time.monotonic()
ids = engine.tokenizer.encode(engine.tokenizer.chat([{"role":"user","content":"Explain why the sky is blue in simple words."}]))
logits = engine.prefill(ids)
out = deploy / ("runs/official-native-probe" if args.native_ort else "runs/official-probe")
out.mkdir(parents=True,exist_ok=True)
np.save(out / "prefill.npy",logits)
save(out / "probe.json",{"seconds":time.monotonic()-start,"prompt_ids":ids,
                          "native_ort":args.native_ort,
                          "first_token":int(logits[0].argmax()),"npu":npu.stats,"memory":memory()})
print("PREFILL",time.monotonic()-start,engine.tokenizer.decode([int(logits[0].argmax())]), npu.stats, flush=True)
npu.close()
