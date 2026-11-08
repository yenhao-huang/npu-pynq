"""Audit saved outputs and physical evidence of the compiled board notebook."""
import argparse
import json
from pathlib import Path
import nbformat
from acceptance import save, sha256, MAX_NEW
from official_acceptance import SOURCE_SHA256, compare

p=argparse.ArgumentParser()
p.add_argument('--run-id',required=True)
a=p.parse_args()
deploy=Path(__file__).resolve().parents[1]
run=deploy/'runs'/a.run_id
def read(name): return json.loads((run/name).read_text())
nb=nbformat.read(run/'smollm_board_onnx_quant.executed.ipynb',as_version=4)
code=[c for c in nb.cells if c.cell_type=='code']
outputs=[o for c in code for o in c.outputs]
load=read('load.json'); result=read('generation.json')
execution=read('notebook-audit.json')
comparison=compare(run/'generation.npz',run/'cpu-reference.npz')
full=compare(run/'generation.npz',deploy/'data/official-arm-oracle/reference.npz')
repeat=compare(run/'generation.npz',run/'repeat.npz'); cpu=read('cpu-runtime.json')
package=deploy.parents[2]/'model-assets/official-compiled'
manifest=json.loads((package/'manifest.json').read_text()); meta=manifest['meta']
arm=deploy/'data/official-arm-oracle'
arm_info=json.loads((arm/'summary.json').read_text())
gates={
 'eight_code_cells':len(code)==8,
 'executed_in_order':[c.execution_count for c in code]==list(range(1,9)),
 'no_errors':not any(o.output_type=='error' for o in outputs),
 'executor_success':execution['success'] and execution['executed_code_cells']==8,
 'all_cells_have_outputs':all(c.outputs for c in code),
 'saved_four_pngs':sum('image/png' in o.get('data',{}) for o in outputs)==4,
 'official_source':meta['source_sha256']==SOURCE_SHA256,
 'full_layers':meta['layers']==30,
 'quantization_preserved':meta['onnx_quantization_unchanged'],
 'compiled_pipeline':load['compiled_pipeline']==meta['compiled_pipeline'],
 'native_float_version':load['onnxruntime_version']=='1.20.1',
 'unchanged_arm_oracle':arm_info['unchanged_onnx'] and
     arm_info['source_sha256']==SOURCE_SHA256 and 'ARMv7' in arm_info['architecture'],
 'arm_reference_identity':sha256(arm/'reference.npz')==arm_info['reference_sha256'],
 'same_arm_runtime_binary':sha256(package/'libonnxruntime.so.1.20.1')==arm_info['runtime_sha256'],
 'arm_prompt':result['prompt_ids']==arm_info['prompt_ids'],
 'runtime_physical':load['runtime_backend']==1 and load['runtime_mmio'],
 'compiled_library':load['library_sha256']==sha256(package/manifest['libraries']['pynq']),
 'compiled_programs':load['programs_sha256']==sha256(package/'programs.bin'),
 'compiled_weights':load['weights_sha256']==sha256(package/'weights.bin'),
 'cpu_runtime_separate':cpu['backend']==0 and not cpu['mmio'],
 'cpu_runtime_no_fault':cpu['stats']['error']==0,
 'generated_24':len(result['ids'])==MAX_NEW,
 'physical_fused_contractions':result['npu']['calls']==MAX_NEW*120,
 'physical_jobs':result['npu']['jobs']==842400,
 'physical_cycles':result['npu']['cycles']>0,
 'physical_integer_probes':len(read('gemm.json'))==3 and all(t['exact'] for t in read('gemm.json')),
 'cpu_tokens':comparison['token_ids_equal'],
 'cpu_logits_exact':comparison['max_abs_error']==0,
 'full_onnx_tokens':full['token_ids_equal'],
 'full_onnx_logits':full['logits_allclose'],
 'repeat_tokens':repeat['token_ids_equal'],
 'repeat_logits_exact':repeat['max_abs_error']==0,
 'interactive':len(read('interactive.json')['ids'])>1,
 'notebook_verdict':read('verdict.json')['pass'],
 'named_notebook_has_outputs':sha256(deploy/'smollm_board_onnx_quant.ipynb')==sha256(run/'smollm_board_onnx_quant.executed.ipynb'),
}
report={'pass':all(gates.values()),'gates':gates,'passed':sum(gates.values()),
 'total':len(gates),'notebook_sha256':sha256(deploy/'smollm_board_onnx_quant.ipynb'),
 'full_arm_comparison':full,'cpu_comparison':comparison,'repeat_comparison':repeat,
 'timing':{k:result[k] for k in ('ttft_seconds','tpot_seconds','decode_tokens_per_second')},
 'physical_npu':result['npu']}
save(run/'compiled-audit.json',report)
print(json.dumps(report,indent=2))
assert report['pass'],report
