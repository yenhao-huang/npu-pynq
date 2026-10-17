"""Compare complete graph logits/KV against the unchanged official ORT model."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort
from official_runtime import ONNXGraph
from official_ort_ops import OrtOperators

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parents[2]))
from official_tokenizer import Tokenizer

package = root / "packages/onnx"
tokenizer = Tokenizer(package)
prompt = "Explain why the sky is blue in simple words."
ids = tokenizer.encode(tokenizer.chat([{"role": "user", "content": prompt}]))
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(str(package / "model_int8.onnx"), options, providers=["CPUExecutionProvider"])
adapter = ONNXGraph(root / "packages/onnx-graph", float_ops=OrtOperators(root / "packages/onnx-ops"))
past = {f"past_key_values.{i}.{kind}": np.empty((1, 3, 0, 64), np.float32)
        for i in range(30) for kind in ("key", "value")}
rows, report, generated = [], [], []
adapter_past = dict(past)
for step in range(24):
    length = next(iter(past.values())).shape[2]
    tokens = ids if step == 0 else [generated[-1]]
    inputs = {"input_ids": np.asarray([tokens], dtype=np.int64),
              "attention_mask": np.ones((1, length + len(tokens)), np.int64),
              "position_ids": np.arange(length, length + len(tokens), dtype=np.int64)[None], **past}
    start = time.monotonic()
    official = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
    ort_seconds = time.monotonic() - start
    start = time.monotonic()
    actual = adapter.run({**inputs, **adapter_past})
    errors = {k: float(np.max(np.abs(actual[k] - official[k]))) for k in official}
    row = official["logits"][0, -1]
    next_id = int(row.argmax())
    generated.append(next_id)
    rows.append(row.copy())
    entry = {"step": step, "ort_seconds": ort_seconds, "adapter_seconds": time.monotonic()-start,
             "logits_max_error": errors["logits"], "kv_max_error": max(v for k,v in errors.items() if k != "logits"),
             "last_logits_max_error": float(np.max(np.abs(actual["logits"][0,-1] - row))),
             "official_token": next_id, "adapter_token": int(actual["logits"][0,-1].argmax())}
    report.append(entry)
    print(json.dumps(entry), flush=True)
    # Preserve failures for diagnosis rather than stopping before saving evidence.
    past = {k.replace("present.", "past_key_values."): v for k,v in official.items() if k != "logits"}
    adapter_past = {k.replace("present.", "past_key_values."): v for k,v in actual.items() if k != "logits"}
out = root / "runs/official-reference"
out.mkdir(parents=True, exist_ok=True)
(out / "comparison.json").write_text(json.dumps(report, indent=2))
np.savez(out / "reference.npz", prompt_ids=ids, ids=generated, logits=np.stack(rows))
import hashlib
summary={'source_sha256':hashlib.sha256((package/'model_int8.onnx').read_bytes()).hexdigest(),
         'tokenizer_sha256':hashlib.sha256((package/'tokenizer.json').read_bytes()).hexdigest(),
         'tokenizer_config_sha256':hashlib.sha256((package/'tokenizer_config.json').read_bytes()).hexdigest(),
         'prompt_ids':ids,'prompt':prompt,'official_default_system':True,
         'onnxruntime_version':ort.__version__,'steps':len(report),
         'tokens_equal':all(i['official_token']==i['adapter_token'] for i in report),
         'logits_max_abs_error':max(i['logits_max_error'] for i in report),
         'kv_max_abs_error':max(i['kv_max_error'] for i in report)}
summary['pass']=summary['tokens_equal'] and summary['logits_max_abs_error']<1e-3 and summary['kv_max_abs_error']<1e-3
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print("Reference saved; token matches:", sum(e["official_token"] == e["adapter_token"] for e in report), "/", len(report), tokenizer.decode(generated))
for item in report:
    assert item['logits_max_error'] < 1e-3 and item['kv_max_error'] < 1e-3, item
    assert item['official_token'] == item['adapter_token'], item
