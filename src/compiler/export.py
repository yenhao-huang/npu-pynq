"""Compile a model graph into a package for the host and the PYNQ-Z1.

    model frontend -> MLIR (linalg on tensors, npu.weight)
      npu-opt --npu-partition        which contractions become NPU tasks
      pack weights                   B tiles for NPU-read weights, dense otherwise
      npu-opt --npu-lower            ISA programs + runtime calls
      mlir-opt / mlir-translate / llc / link, once per target

The package directory is self-contained: ``CompiledModel`` (src/runtime)
loads it on either platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence
import numpy as np
from src.compiler import toolchain
from src.compiler.mlir_builder import ModuleBuilder
from src.isa import isa, layout
from src.inference.weights import onnx_asset_dir
import hashlib
import urllib.request


FORMAT = "npu-package/1"
WEIGHT_ALIGN = 64


@dataclass
class ModelGraph:
    """What a model frontend hands to the compiler."""

    name: str
    module: ModuleBuilder
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    entry_points: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Path] = field(default_factory=dict)  # copied verbatim (tokenizer, ...)

    def add_weight(self, name: str, array: np.ndarray) -> str:
        if name in self.weights:
            raise ValueError(f"duplicate weight {name}")
        self.weights[name] = np.ascontiguousarray(array)
        return name


def pack_weights(graph: ModelGraph, report: dict, limits: isa.Limits) -> tuple[bytes, dict, dict]:
    """Return (blob, {name: offset}, {name: layout}) with NPU weights tiled."""
    npu_layout: dict[str, tuple[int, int, int]] = {}
    for task in report["npu_tasks"]:
        key = (task["k"], task["n"], task["tk"])
        prev = npu_layout.setdefault(task["weight"], key)
        if prev != key:
            raise ValueError(f"weight {task['weight']} used by NPU tasks with different tiling")
    for name, (k, n, tk) in graph.meta.get("require_npu_layout", {}).items():
        if npu_layout.get(name) != (k, n, tk):
            raise ValueError(f"{name} is read in NPU layout (k={k}, n={n}, tk={tk}) but the partition "
                             f"stores it as {npu_layout.get(name, 'dense')}")
    packed = set(graph.meta.get("packed_int4", []))
    chunks: list[bytes] = []
    offsets: dict[str, int] = {}
    layouts: dict[str, str] = {}
    cursor = 0
    for name, array in graph.weights.items():
        if name in npu_layout:
            k, n, tk = npu_layout[name]
            if array.shape != (k, n) or array.dtype != np.int8:
                raise ValueError(f"NPU weight {name} must be int8 {k}x{n}, got {array.dtype} {array.shape}")
            plan = layout.plan_gemm(1, n, k, limits, tk)
            tiles = layout.pack_b(plan, array)
            if name in packed:
                if tiles.min() < -8 or tiles.max() > 7:
                    raise ValueError(f"{name} is marked INT4 but holds values outside [-8, 7]")
                u = tiles.view(np.uint8) & 0x0F
                data = (u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()
                layouts[name] = f"npu_b4(k={k},n={n},tk={tk})"
            else:
                data = tiles.tobytes()
                layouts[name] = f"npu_b(k={k},n={n},tk={tk})"
        else:
            data = np.ascontiguousarray(array).tobytes()
            layouts[name] = f"dense({array.dtype},{'x'.join(map(str, array.shape))})"
        pad = (-cursor) % WEIGHT_ALIGN
        chunks.append(b"\0" * pad)
        cursor += pad
        offsets[name] = cursor
        chunks.append(data)
        cursor += len(data)
    return b"".join(chunks), offsets, layouts


def export(graph: ModelGraph | str | Path | None, out: str | Path, targets: Sequence[str] | None = None,
           limits: isa.Limits = isa.Limits(), keep_ir: bool = True, segment_bytes: int = 4 << 20,
           extra_objects: dict[str, Sequence[Path]] | None = None, *, model: str | None = None,
           opts=None, source_format: str = 'auto', prefill_lengths=(36, 40, 42),
           native_object: str | Path | None = None) -> Path:
    """Compile a graph or model source through the same partition/lower pipeline.

    Model frontends own input formats; the compiler always consumes ModelGraph.
    Existing graph callers retain their original API and default CPU targets.
    """
    frontend_seconds = None
    if not isinstance(graph, ModelGraph):
        from src.models import load
        if model is None:
            raise ValueError('A model frontend is required when exporting a source')
        kwargs = {} if opts is None else {'opts': opts}
        if graph is not None:
            kwargs['source'] = graph
        if source_format != 'auto':
            kwargs['source_format'] = source_format
        source_path = Path(graph) if graph is not None else None
        if source_format == 'onnx' or (source_path is not None and
                (source_path.suffix == '.onnx' or (source_path/'model_int8.onnx').is_file()
                 or (onnx_asset_dir(source_path, 'onnx')/'model_int8.onnx').is_file())):
            kwargs['prefill_lengths'] = prefill_lengths
        start = time.monotonic()
        graph, _ = load(model).build(**kwargs)
        frontend_seconds = time.monotonic() - start
    supported = graph.meta.get('targets', ('host', 'pynq'))
    targets = tuple(targets) if targets is not None else tuple(supported)
    if not targets or any(t not in supported for t in targets):
        raise ValueError(f'Unsupported targets {targets}; this graph supports {supported}')
    objects = dict(extra_objects or {})
    if native_object is not None:
        objects['pynq'] = (*objects.get('pynq', ()), Path(native_object))
    for target in graph.meta.get('requires_native_object', ()):
        if target in targets and not objects.get(target):
            raise ValueError(f'{target} requires a native kernel object')
    for paths in objects.values():
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    out = Path(out)
    assets = list(graph.files.values()) + [p for paths in objects.values() for p in paths]
    if any(out.resolve() == Path(p).resolve() or out.resolve() in Path(p).resolve().parents
           for p in assets):
        raise ValueError('The output directory must not contain input assets')
    if out.exists():
        shutil.rmtree(out)
    work = out / "ir"
    work.mkdir(parents=True)
    timings: dict[str, float] = {}
    if frontend_seconds is not None:
        timings['frontend'] = frontend_seconds

    # Write the model graph as MLIR, then choose which matmuls run on the NPU.
    t = time.monotonic()
    source = work / "model.mlir"
    source.write_text(graph.module.render())

    parted = work / "partitioned.mlir"
    report_path = out / "partition.json"
    toolchain.partition(source, parted, report_path, max_k=limits.max_k)
    report = json.loads(report_path.read_text())
    timings["partition"] = time.monotonic() - t

    # Pack NPU weights as tiles and record every weight's offset in the arena.
    t = time.monotonic()
    blob, offsets, layouts = pack_weights(graph, report, limits)
    (out / "weights.bin").write_bytes(blob)
    table = work / "weights.json"
    table.write_text(json.dumps(offsets))
    timings["pack_weights"] = time.monotonic() - t

    # Turn NPU tasks into ISA programs and calls to the C runtime.
    t = time.monotonic()
    lowered = work / "lowered.mlir"
    toolchain.lower(parted, lowered, table, out / "programs", limits.rows, limits.columns, limits.max_k,
                    segment_bytes)
    timings["npu_lower"] = time.monotonic() - t

    # Compile the remaining CPU operations for each requested platform.
    libraries = {}
    for name in targets:
        t = time.monotonic()
        tgt = toolchain.target(name)
        lib = toolchain.codegen(lowered, work / name, tgt, objects.get(name, ()))
        dest = out / name / tgt.library
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(lib, dest)
        libraries[name] = f"{name}/{tgt.library}"
        timings[f"codegen_{name}"] = time.monotonic() - t

    # Bundle the compiled libraries, weights, programs, and model metadata.
    for dest, src in graph.files.items():
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dest)
    programs = json.loads((out / "programs.json").read_text())
    manifest = {
        "format": FORMAT,
        "model": graph.name,
        "libraries": libraries,
        "entry_points": graph.entry_points,
        "meta": graph.meta,
        "weights": {"file": "weights.bin", "bytes": len(blob), "count": len(offsets)},
        "weight_layouts": layouts,
        "weight_offsets": offsets,
        "npu": {"tasks": len(programs["tasks"]),
                "segments": sum(len(t["segments"]) for t in programs["tasks"]),
                "program_words": programs["words"],
                "jobs_total": sum(t["jobs"] for t in programs["tasks"])},
        "cpu_linalg_ops": report["cpu_linalg_ops"],
        "limits": {"rows": limits.rows, "columns": limits.columns, "max_k": limits.max_k},
        "compile_seconds": timings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not keep_ir:
        shutil.rmtree(work)
    else:
        for name in targets:
            for f in (work / name).glob("*.o"):
                f.unlink()
    return out


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(model_dir, manifest=None):
    manifest = Path(manifest) if manifest else Path(__file__).with_name("onnx-source.json")
    source = json.loads(manifest.read_text(encoding="utf-8"))
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


def export_graph(model_dir):
    import onnx
    from onnx import helper, numpy_helper
    model_dir = Path(model_dir)
    source = onnx_asset_dir(model_dir, "onnx") / "model_int8.onnx"
    pinned = json.loads(Path(__file__).with_name("onnx-source.json").read_text(encoding="utf-8"))
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
    from src.inference.graph import ONNXGraph
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
    from src.inference.graph import ONNXGraph
    model_dir = Path(model_dir)
    export_sessions(ONNXGraph(onnx_asset_dir(model_dir, "onnx-graph")), onnx_asset_dir(model_dir, "onnx-ops"))


def export_sessions(graph, output):
    from src.runtime.compiled import ORT_OPS
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


def attach_float_runtime(graph, root, source_model):
    """Attach native operator sessions to a frontend's ordinary ModelGraph."""
    root = Path(root)
    from types import SimpleNamespace
    from onnx import helper, TensorProto, save
    opdir = root/'onnx-compiled-ops'
    nodes = [{'op':node.op_type,'inputs':list(node.input),'outputs':list(node.output),
              'attrs':{a.name:helper.get_attribute_value(a) for a in node.attribute}}
             for node in source_model.graph.node]
    export_sessions(SimpleNamespace(manifest={'nodes':nodes}),opdir)
    model = helper.make_model(helper.make_graph(
        [helper.make_node('MatMul',['x0','x1'],['y0'])],'MatMul',
        [helper.make_tensor_value_info('x0',TensorProto.FLOAT,None),
         helper.make_tensor_value_info('x1',TensorProto.FLOAT,None)],
        [helper.make_tensor_value_info('y0',TensorProto.FLOAT,None)]),
        opset_imports=[helper.make_opsetid('',14)])
    model.ir_version=9
    save(model,opdir/'MatMul.onnx')
    hashes={}
    for path in sorted(opdir.glob('*.onnx')):
        dest='float-ops/'+path.name
        graph.files[dest]=path
        hashes[dest]=hashlib.sha256(path.read_bytes()).hexdigest()
    runtime=onnx_asset_dir(root, 'onnx-runtime')/'libonnxruntime.so.1.20.1'
    graph.files[runtime.name]=runtime
    hashes[runtime.name]=hashlib.sha256(runtime.read_bytes()).hexdigest()
    graph.meta['native_kernel_hashes']=hashes
    graph.meta['compiled_pipeline']='ONNX frontend -> MLIR -> npu-partition -> npu-lower -> LLVM ARM'
    graph.meta['targets'] = ['pynq']
    graph.meta['requires_native_object'] = ['pynq']
    graph.meta['runtime_hooks'] = {
        'preload': [runtime.name], 'initialize': 'onnx_compiled_init',
        'initialize_path': 'float-ops', 'reset': 'onnx_compiled_clear',
        'status': 'onnx_compiled_status', 'error': 'onnx_ort_error'}
