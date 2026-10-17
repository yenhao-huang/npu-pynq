"""Inspect the pinned official graph and preserve its exact node contracts."""
from collections import Counter
import json
from pathlib import Path
import onnx
from onnx import helper

root = Path(__file__).resolve().parents[1]
model = onnx.load(root / "packages/onnx/model_int8.onnx")
nodes = []
for n in model.graph.node:
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    attrs = {k: v.decode() if isinstance(v, bytes) else v for k, v in attrs.items()}
    nodes.append({"name": n.name, "op": n.op_type, "domain": n.domain,
                  "inputs": list(n.input), "outputs": list(n.output), "attrs": attrs})
report = {"ir_version": model.ir_version, "opsets": {o.domain: o.version for o in model.opset_import},
          "operators": dict(Counter((n["domain"] + ":" + n["op"]) for n in nodes)),
          "inputs": [str(i) for i in model.graph.input], "outputs": [str(i) for i in model.graph.output],
          "initializers": {i.name: {"shape": list(i.dims), "dtype": i.data_type} for i in model.graph.initializer}}
out = root / "runs/official-inspect"
out.mkdir(parents=True, exist_ok=True)
(out / "graph.json").write_text(json.dumps(nodes, indent=2))
(out / "summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("operators", "opsets")}, indent=2))
print("INPUTS", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model.graph.input][:8])
print("FIRST", json.dumps(nodes[:15], indent=2))
print("CUSTOM", json.dumps([n for n in nodes if n["domain"]][:8], indent=2))
print("LAST", json.dumps(nodes[-5:], indent=2))
