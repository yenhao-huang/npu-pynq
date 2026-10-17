"""Find the first onnx-graph numerical divergence using ORT intermediates."""
exec(open(str(__import__('pathlib').Path(__file__).with_name('official_reference.py')), encoding='utf-8').read().split('rows, report, generated')[0])
import onnx
model = onnx.load(package / 'model_int8.onnx')
names = []
for node in model.graph.node[:1500]:
    for name in node.output:
        if name:
            names.append(name)
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
inputs = {'input_ids': np.asarray([ids], np.int64), 'attention_mask': np.ones((1,len(ids)), np.int64),
          'position_ids': np.arange(len(ids), dtype=np.int64)[None], **past}
expected = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
env = {**adapter.weights, **inputs, **expected}
for node in adapter.manifest['nodes'][:1500]:
    if node['op'] != 'MatMulInteger':
        continue
    got = adapter.op(node['op'], [env[n] if n else None for n in node['inputs']], node['attrs'])
    name = node['outputs'][0]
    wanted = expected[name]
    error = float(np.max(np.abs(wanted.astype(np.float64) - got.astype(np.float64))))
    if error:
        print(name, wanted.shape, error)
        print('EXPECTED', wanted.flatten()[:8], 'ACTUAL', got.flatten()[:8])
        break
else:
    print('All isolated integer nodes match ORT exactly')
