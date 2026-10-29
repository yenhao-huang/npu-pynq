"""Prepare a pinned ONNX model for export and for the CPU reference.

    fetch            download the artifacts a pin file names, verifying size and SHA-256
    export_graph     split the ONNX model into graph.json + tensors.bin
    verify_export    check every initializer survived the split unchanged
    export_operators one ONNX Runtime session per float operator contract

The pin file belongs to the model (src/export/model_arch/<model>/onnx-source.json)
and is passed in by the caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from src.runtime.core.package import onnx_asset_dir


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(model_dir, manifest):
    source = json.loads(Path(manifest).read_text(encoding="utf-8"))
    package = Path(model_dir) / "onnx"
    package.mkdir(parents=True, exist_ok=True)

    def valid(path, spec):
        return path.is_file() and path.stat().st_size == spec["bytes"] and sha256(path) == spec["sha256"]

    for name, spec in source["files"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid artifact basename: " + name)
        dest = package / name
        if valid(dest, spec):
            print("Verified cache", name, flush=True)
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(spec["url"], timeout=120) as response, part.open("wb") as stream:
            for block in iter(lambda: response.read(1 << 20), b""):
                stream.write(block)
        if not valid(part, spec):
            raise ValueError("ONNX artifact identity mismatch: " + name)
        part.replace(dest)
        print("Verified download", name, flush=True)
    (package / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return package


def export_graph(model_dir, manifest):
    import onnx
    from onnx import helper, numpy_helper
    model_dir = Path(model_dir)
    source = onnx_asset_dir(model_dir, "onnx") / "model_int8.onnx"
    pinned = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if sha256(source) != pinned["files"]["model_int8.onnx"]["sha256"]:
        raise ValueError("Model differs from the pinned pinned INT8 artifact")
    out = onnx_asset_dir(model_dir, "onnx-graph")
    out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    tensors = {}
    with (out / "tensors.bin").open("wb") as stream:
        for tensor in model.graph.initializer:
            array = numpy_helper.to_array(tensor)
            stream.write(bytes((-stream.tell()) % 64))
            tensors[tensor.name] = {"offset": stream.tell(), "dtype": array.dtype.str,
                                    "shape": list(array.shape)}
            stream.write(array.tobytes())
    nodes = [{"name": node.name, "op": node.op_type, "inputs": list(node.input),
              "outputs": list(node.output),
              "attrs": {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}}
             for node in model.graph.node]
    manifest = {"source_sha256": sha256(source), "tensors_sha256": sha256(out / "tensors.bin"),
                "tensors": tensors, "nodes": nodes,
                "inputs": [value.name for value in model.graph.input],
                "outputs": [value.name for value in model.graph.output]}
    (out / "graph.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


def verify_export(model_dir):
    import onnx
    from onnx import numpy_helper
    from src.test.model.onnx_reference import ONNXGraph
    model_dir = Path(model_dir)
    source = onnx_asset_dir(model_dir, "onnx") / "model_int8.onnx"
    graph = ONNXGraph(onnx_asset_dir(model_dir, "onnx-graph"))
    if graph.manifest["source_sha256"] != sha256(source):
        raise ValueError("Exported source hash differs")
    if graph.manifest["tensors_sha256"] != sha256(onnx_asset_dir(model_dir, "onnx-graph") / "tensors.bin"):
        raise ValueError("Exported tensor hash differs")
    count = 0
    for tensor in onnx.load(source).graph.initializer:
        original = numpy_helper.to_array(tensor)
        actual = graph.weights[tensor.name]
        if (original.dtype != actual.dtype or original.shape != actual.shape or
                original.tobytes() != actual.tobytes()):
            raise ValueError("Initializer differs: " + tensor.name)
        count += 1
    return {"pass": True, "initializers_checked": count,
            "source_sha256": graph.manifest["source_sha256"],
            "tensors_sha256": graph.manifest["tensors_sha256"]}


def export_operators(model_dir):
    from src.test.model.onnx_reference import ONNXGraph
    model_dir = Path(model_dir)
    export_sessions(ONNXGraph(onnx_asset_dir(model_dir, "onnx-graph")), onnx_asset_dir(model_dir, "onnx-ops"))


def export_sessions(graph, output):
    from src.runtime.cpu.ort import ORT_OPS
    import onnx
    from onnx import helper, TensorProto
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sessions = {}
    # Build one shape-polymorphic session per operator/attribute contract.
    for node in graph.manifest['nodes']:
        if node['op'] not in ORT_OPS:
            continue
        key = node['op']
        if key in sessions:
            assert sessions[key] == node['attrs']
            continue
        sessions[key] = node['attrs']
        count = len(node['inputs'])
        names = ['x' + str(i) if n else '' for i,n in enumerate(node['inputs'])]
        dtype = [TensorProto.FLOAT] * count
        if key == 'RotaryEmbedding': dtype[1] = TensorProto.INT64
        if key == 'GroupQueryAttention': dtype[5] = dtype[6] = TensorProto.INT32
        outputs = ['y'+str(i) if n else '' for i,n in enumerate(node['outputs'])]
        types = [TensorProto.FLOAT] * len(outputs)
        if key == 'DynamicQuantizeLinear': types = [TensorProto.UINT8,TensorProto.FLOAT,TensorProto.UINT8]
        domain = 'com.microsoft' if key in ('SkipSimplifiedLayerNormalization','RotaryEmbedding','GroupQueryAttention') else ''
        op = helper.make_node(key,names,outputs,domain=domain,**node['attrs'])
        ins = [helper.make_tensor_value_info(n,dtype[i],None) for i,n in enumerate(names) if n]
        outs = [helper.make_tensor_value_info(n,types[i],None) for i,n in enumerate(outputs) if n]
        model = helper.make_model(helper.make_graph([op],key,ins,outs),
                                  opset_imports=[helper.make_opsetid('',14),helper.make_opsetid('com.microsoft',1)])
        model.ir_version = 9
        onnx.save(model, output / (key+'.onnx'))
