"""Check every operator against ORT with identical, captured operator inputs."""
from collections import Counter
import json
import onnx
exec(open(str(__import__('pathlib').Path(__file__).with_name('official_reference.py')), encoding='utf-8').read().split('rows, report, generated')[0])

model = onnx.load(package / 'model_int8.onnx')
original_names = set(o.name for o in model.graph.output)
for node in model.graph.node:
    for name in node.output:
        if name and name not in original_names:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
inputs = {'input_ids': np.asarray([ids],np.int64), 'attention_mask': np.ones((1,len(ids)),np.int64),
          'position_ids': np.arange(len(ids),dtype=np.int64)[None], **past}
expected = dict(zip([o.name for o in session.get_outputs()], session.run(None,inputs)))
env = {**adapter.weights, **inputs, **expected}
report = []
from official_runtime import Dequantized
for node in adapter.manifest['nodes']:
    result = adapter.op(node['op'], [env[n] if n else None for n in node['inputs']],node['attrs'])
    if not isinstance(result,tuple): result = (result,)
    for name, actual in zip(node['outputs'], result):
        if not name: continue
        if isinstance(actual,Dequantized): actual = actual.materialize()
        wanted = expected[name]
        exact = np.array_equal(actual,wanted)
        # The 576-term FP32 output contraction has ordinary accumulation-order error.
        atol = 1e-4 if node['op'] == 'MatMul' else 1e-5
        close = bool(np.allclose(actual,wanted,rtol=1e-5,atol=atol)) if wanted.dtype.kind == 'f' else exact
        report.append({'name':name,'op':node['op'],'exact':exact,'pass':close,
                       'max_abs_error':float(np.max(np.abs(actual.astype(np.float64)-wanted.astype(np.float64))))})
out = root / 'runs/official-operators'
out.mkdir(parents=True,exist_ok=True)
summary = {'pass':all(r['pass'] for r in report),'nodes':len(adapter.manifest['nodes']),
           'outputs_checked':len(report),'operators':dict(Counter(r['op'] for r in report)),
           'failures':[r for r in report if not r['pass']]}
(out/'results.json').write_text(json.dumps({'summary':summary,'results':report},indent=2))
print(json.dumps(summary,indent=2))
