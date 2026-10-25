"""Inventory ONNX without loading external tensor data or claiming support."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(source: str | Path) -> dict:
    import onnx
    from onnx import AttributeProto, TensorProto, helper
    from google.protobuf.message import DecodeError

    source = Path(source).expanduser().resolve(strict=True)
    try:
        model = onnx.load(source, load_external_data=False)
    except (DecodeError, ValueError) as error:
        raise ValueError(f'Cannot read ONNX source {source}: {error}') from error
    graphs = []
    counts = Counter()
    external = []

    def tensor_info(tensor):
        descriptor = {'name': tensor.name, 'dtype': TensorProto.DataType.Name(tensor.data_type),
                      'shape': list(tensor.dims)}
        if tensor.external_data:
            descriptor['external_data'] = {entry.key: entry.value for entry in tensor.external_data}
            external.append(descriptor)
        return descriptor

    def value_info(value):
        if not value.type.HasField('tensor_type'):
            return {'name': value.name, 'type': str(value.type)}
        tensor_type = value.type.tensor_type
        shape = None
        if tensor_type.HasField('shape'):
            shape = [dimension.dim_value if dimension.HasField('dim_value') else
                     dimension.dim_param if dimension.HasField('dim_param') else None
                     for dimension in tensor_type.shape.dim]
        return {'name': value.name, 'dtype': TensorProto.DataType.Name(tensor_type.elem_type),
                'shape': shape}

    def attribute_info(attribute):
        if attribute.type == AttributeProto.GRAPH:
            return {'graph': attribute.g.name}
        if attribute.type == AttributeProto.GRAPHS:
            return {'graphs': [graph.name for graph in attribute.graphs]}
        if attribute.type == AttributeProto.TENSOR:
            descriptor = tensor_info(attribute.t)
            descriptor['sha256'] = hashlib.sha256(attribute.t.SerializeToString()).hexdigest()
            return descriptor
        if attribute.type == AttributeProto.TENSORS:
            return [tensor_info(tensor) for tensor in attribute.tensors]
        if attribute.type == AttributeProto.STRING:
            return attribute.s.decode('utf-8', errors='backslashreplace')
        if attribute.type == AttributeProto.STRINGS:
            return [value.decode('utf-8', errors='backslashreplace') for value in attribute.strings]
        value = helper.get_attribute_value(attribute)
        if isinstance(value, (str, int, float, list)) or value is None:
            return value
        return {'type': AttributeProto.AttributeType.Name(attribute.type), 'value': str(value)}

    def visit_nodes(nodes, opsets, scope):
        descriptors = []
        for index, node in enumerate(nodes):
            domain = '' if node.domain == 'ai.onnx' else node.domain
            version = opsets.get(domain)
            counts[(domain, node.op_type, version)] += 1
            descriptors.append({'name': node.name, 'op': node.op_type, 'domain': domain,
                                'opset': version, 'inputs': list(node.input),
                                'outputs': list(node.output),
                                'attributes': {attribute.name: attribute_info(attribute)
                                               for attribute in node.attribute}})
            for attribute in node.attribute:
                subgraphs = ([attribute.g] if attribute.type == AttributeProto.GRAPH else
                             list(attribute.graphs) if attribute.type == AttributeProto.GRAPHS else [])
                for branch_index, subgraph in enumerate(subgraphs):
                    visit_graph(subgraph, opsets, f'{scope}/{index}/{attribute.name}/{branch_index}')
        return descriptors

    def visit_graph(graph, opsets, scope):
        descriptor = {'scope': scope, 'name': graph.name,
                      'inputs': [value_info(value) for value in graph.input],
                      'outputs': [value_info(value) for value in graph.output],
                      'values': [value_info(value) for value in graph.value_info],
                      'initializers': [tensor_info(tensor) for tensor in graph.initializer],
                      'sparse_initializers': [{'values': tensor_info(tensor.values),
                                               'indices': tensor_info(tensor.indices),
                                               'shape': list(tensor.dims)}
                                              for tensor in graph.sparse_initializer]}
        graphs.append(descriptor)
        descriptor['nodes'] = visit_nodes(graph.node, opsets, scope)

    def imports(entries):
        return {'' if entry.domain == 'ai.onnx' else entry.domain: entry.version for entry in entries}

    opsets = imports(model.opset_import)
    visit_graph(model.graph, opsets, 'main')
    for index, function in enumerate(model.functions):
        scope = f'function/{function.domain}/{function.name}/{index}'
        descriptor = {'scope': scope, 'name': function.name, 'opsets': imports(function.opset_import)}
        graphs.append(descriptor)
        descriptor['nodes'] = visit_nodes(function.node, descriptor['opsets'], scope)

    operators = []
    for (domain, operation, version), count in sorted(counts.items(), key=lambda item: str(item[0])):
        known_schema = False
        if version is not None:
            try:
                onnx.defs.get_schema(operation, version, domain)
                known_schema = True
            except onnx.defs.SchemaError:
                pass
        requires_conversion = bool(domain) or not known_schema
        operators.append({'domain': domain, 'op': operation, 'opset': version, 'count': count,
                          'status': 'requires_conversion' if requires_conversion else 'unverified',
                          'reason': 'No verified lowering for this domain/operator/opset.' if requires_conversion
                          else 'ONNX schema exists; compiler and target support remain unverified.'})
    return {'format': 'onnx-mlir-inventory/1', 'scope': 'inventory-only',
            'source': {'sha256': file_hash(source), 'bytes': source.stat().st_size},
            'ir_version': model.ir_version, 'opsets': opsets, 'operators': operators,
            'graphs': graphs, 'external_data': external, 'compile_ready': False}
