"""Build the official-model notebook, with complete executable acceptance gates."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# Official SmolLM2-135M-Instruct INT8 on PYNQ-Z1
The source is the unchanged official `onnx/model_int8.onnx` (137,147,867 bytes).
Its 30 layers, INT8 projection weights, dynamic UINT8 activation scales/zero points,
RoPE, attention and FP32 output head are retained. Integer projections execute on
the physical NPU. Native ONNX Runtime executes the floating-point operators.
Run all cells in order with the root PYNQ Python kernel; saved evidence lives in `runs/`.
The CPU reference uses identical native ORT operators and exact CPU integer GEMMs.
The separate host oracle verifies this execution against the unchanged full ONNX.
''')
code('''
import gc, html, json, os, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown
candidates = []
for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    candidates.extend((parent, parent / 'exp/1007_sw_stack/deploy'))
DEPLOY = next(p for p in candidates if (p/'scripts/official_engine.py').is_file()
              and (p/'packages/onnx-graph/graph.json').is_file())
REPO = DEPLOY.parents[2]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(DEPLOY/'scripts'))
from acceptance import memory, save, sha256, PROMPT, MAX_NEW
from official_acceptance import SOURCE_SHA256, trace, compare
from official_engine import ONNXReferenceEngine
from onnx_ort_ctypes import NativeOrtOperators
from official_npu import NpuGemm
assert os.geteuid()==0, 'Use the authorized root PYNQ kernel.'
assert (DEPLOY/'packages/onnx-runtime/libonnx_ort_bridge.so').is_file(), 'Official ARM runtime deployment is not complete.'
RUN = DEPLOY/'runs'/os.environ.get('NPU_RUN_ID',time.strftime('official-manual-%Y%m%d-%H%M%S'))
RUN.mkdir(parents=True,exist_ok=True)
manifest = json.loads((DEPLOY/'packages/onnx-graph/graph.json').read_text())
assert sha256(DEPLOY/'packages/onnx/model_int8.onnx') == manifest['source_sha256'] == SOURCE_SHA256
assert sha256(DEPLOY/'packages/onnx-graph/tensors.bin') == manifest['tensors_sha256']
oracle=json.loads((DEPLOY/'data/official-oracle/summary.json').read_text())
assert oracle['pass'] and oracle['source_sha256']==SOURCE_SHA256 and oracle['steps']==MAX_NEW
assert sha256(DEPLOY/'packages/onnx/tokenizer.json')==oracle['tokenizer_sha256']
assert sha256(DEPLOY/'packages/onnx/tokenizer_config.json')==oracle['tokenizer_config_sha256']
info = {'source_sha256':SOURCE_SHA256,'official_bytes':(DEPLOY/'packages/onnx/model_int8.onnx').stat().st_size,
        'nodes':len(manifest['nodes']),'layers':sum(n['op']=='GroupQueryAttention' for n in manifest['nodes']),
        'integer_projections':sum(n['op']=='MatMulInteger' for n in manifest['nodes']),'memory':memory()}
assert info['layers']==30 and info['integer_projections']==210
save(RUN/'package.json',info); display(info,oracle)
def metrics_table(rows):
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    display(HTML('<table><tr><th>Metric</th><th>Value</th><th>Unit</th></tr>'+body+'</table>'))
''')
md('## 1. Load the FPGA and verify signed integer arithmetic')
code('''
from pynq import Overlay
from src.runtime.isa_runtime import REG_ISA_STATUS, STATUS_RUNNING, STATUS_BUSY
existing = Overlay(str(DEPLOY/'hardware/overlay/npu_matrix.bit'),download=False)
assert not existing.npu_accelerator_0.read(REG_ISA_STATUS) & (STATUS_RUNNING|STATUS_BUSY), 'NPU is busy.'
overlay = Overlay(str(DEPLOY/'hardware/overlay/npu_matrix.bit'))
npu = NpuGemm(overlay)
rng=np.random.default_rng(126); arithmetic=[]
for m,k,n in ((1,576,192),(19,576,576),(1,1536,576)):
    a=rng.integers(-128,128,(m,k),dtype=np.int8); b=rng.integers(-128,128,(k,n),dtype=np.int8)
    before=dict(npu.stats); start=time.monotonic()
    actual=npu(a,b); expected=a.astype(np.int32)@b.astype(np.int32)
    np.testing.assert_array_equal(actual,expected)
    arithmetic.append({'shape':[m,k,n],'exact':True,'seconds':time.monotonic()-start,
                       'jobs':npu.stats['jobs']-before['jobs'],'cycles':npu.stats['cycles']-before['cycles']})
save(RUN/'gemm.json',arithmetic); display(arithmetic)
metrics_table([(str(c['shape']),'PASS',str(c['jobs'])+' physical jobs') for c in arithmetic])
''')
md('## 2. Load original official tensors and native ORT floating-point kernels')
code('''
start=time.monotonic()
float_ops=NativeOrtOperators(DEPLOY/'packages/onnx-ops',DEPLOY/'packages/onnx-runtime/libonnx_ort_bridge.so')
assert float_ops.version==oracle['onnxruntime_version']
engine=ONNXReferenceEngine(DEPLOY,npu,float_ops)
load_result={'seconds':time.monotonic()-start,'memory':memory(),'max_seq':engine.max_seq,
             'onnxruntime_version':float_ops.version,
             'weight_tensors':len(engine.graph.weights),'integer_projections':info['integer_projections'],'cma_bytes':(4<<20)+4096*8}
save(RUN/'load.json',load_result); display(load_result)
metrics_table([('Model load',round(load_result['seconds'],3),'seconds'),
               ('Official ONNX',round(info['official_bytes']/2**20,2),'MiB'),('CMA window',4.03125,'MiB')])
''')
md('## 3. Prefill and one-token decode')
code('''
engine.reset()
prompt_ids=engine.tokenizer.encode(engine.tokenizer.chat([{'role':'user','content':PROMPT}]))
assert prompt_ids==oracle['prompt_ids'], 'Board tokenizer or official chat template differs from the oracle.'
start=time.monotonic(); engine.prefill(prompt_ids); prefill_seconds=time.monotonic()-start
token=int(engine.logits[0].argmax()); start=time.monotonic(); row=engine.decode(token)[0].copy()
smoke={'prefill_seconds':prefill_seconds,'decode_seconds':time.monotonic()-start,
       'first_token':token,'text':engine.tokenizer.decode([token]),'finite':bool(np.isfinite(row).all())}
assert smoke['finite']; save(RUN/'smoke.json',smoke); display(smoke)
top=np.argsort(row)[-10:][::-1]
plt.figure(figsize=(10,3)); plt.bar(range(10),row[top])
plt.xticks(range(10),[engine.tokenizer.decode([int(t)]) for t in top],rotation=30)
plt.title('Official-model decode logits'); plt.show()
''')
md('## 4. Generate 24 tokens and measure physical execution')
code('''
view=display(Markdown('Generating…'),display_id=True)
result=trace(engine,RUN,display_token=lambda text:view.update(Markdown(text)))
display(result)
metrics_table([('Prompt',len(result['prompt_ids']),'tokens'),('Generated',len(result['ids']),'tokens'),
               ('TTFT',round(result['ttft_seconds'],3),'seconds'),('TPOT',round(result['tpot_seconds'],3),'seconds/token'),
               ('Decode throughput',round(result['decode_tokens_per_second'],3),'tokens/s'),
               ('Physical NPU jobs',result['npu']['jobs'],'jobs'),('Physical NPU cycles',result['npu']['cycles'],'cycles')])
fig,axes=plt.subplots(1,2,figsize=(12,3))
axes[0].plot(result['per_token_seconds'],marker='o'); axes[0].set(xlabel='Decode step',ylabel='Seconds',title='Decode latency')
axes[1].bar(['Prefill','Decode'],[result['prefill_seconds'],result['decode_seconds']]); axes[1].set(ylabel='Seconds',title='Generation timing')
plt.tight_layout(); plt.show()
snapshots=[info['memory'],load_result['memory'],result['memory']]; labels=['Before load','Model loaded','After generation']
fig,axes=plt.subplots(1,2,figsize=(12,3))
axes[0].bar(labels,[s.get('VmRSS_kib',0)/1024 for s in snapshots]); axes[0].set(ylabel='MiB',title='Resident memory')
axes[1].bar(labels,[s.get('CmaFree_kib',0)/1024 for s in snapshots]); axes[1].set(ylabel='MiB',title='Free CMA')
plt.tight_layout(); plt.show()
''')
md('## 5. Compare every logit and token against the same ARM CPU graph')
code('''
cpu=ONNXReferenceEngine(DEPLOY,float_ops=float_ops)
cpu_result=trace(cpu,RUN,'cpu-reference')
comparison=compare(RUN/'generation.npz',RUN/'cpu-reference.npz')
save(RUN/'comparison.json',comparison); display(comparison)
assert comparison['token_ids_equal'] and comparison['logits_allclose'], comparison
del cpu; gc.collect()
plt.figure(figsize=(9,3)); plt.plot(comparison['per_step_max_abs_error'],marker='o')
plt.xlabel('Token step'); plt.ylabel('Maximum absolute logit error'); plt.title('Physical NPU versus ARM CPU'); plt.show()
''')
md('## 6. Repeat from reset and report the gates')
code('''
repeat=trace(engine,RUN,'repeat')
repeat_comparison=compare(RUN/'repeat.npz',RUN/'generation.npz')
from src.isa.layout import plan_gemm
expected_jobs=0
for node in manifest['nodes']:
    if node['op']=='MatMulInteger':
        k,n=engine.graph.weights[node['inputs'][1]].shape
        expected_jobs+=plan_gemm(len(result['prompt_ids']),n,k,npu.device.limits).jobs
        expected_jobs+=(MAX_NEW-1)*plan_gemm(1,n,k,npu.device.limits).jobs
verdict={'official_source':info['source_sha256']==SOURCE_SHA256,'full_30_layers':info['layers']==30,
         'unchanged_onnx_oracle':oracle['pass'],
         'integer_exact':all(c['exact'] for c in arithmetic),'generated_24':len(result['ids'])==MAX_NEW,
         'hardware_active':result['npu']['jobs']>0 and result['npu']['cycles']>0,
         'all_integer_nodes_physical':result['npu']['calls']==MAX_NEW*info['integer_projections'],
         'all_integer_jobs_physical':result['npu']['jobs']==expected_jobs,
         'cpu_tokens':comparison['token_ids_equal'],'cpu_logits':comparison['logits_allclose'],
         'repeat_tokens':repeat_comparison['token_ids_equal'],'repeat_logits':repeat_comparison['logits_allclose']}
verdict['pass']=all(verdict.values()); save(RUN/'verdict.json',verdict)
save(RUN/'repeat-comparison.json',repeat_comparison); display(verdict,repeat_comparison)
assert verdict['pass'], 'Inspect the saved acceptance evidence.'
''')
md('## 7. Edit your prompt and run interactively\n`max_new_tokens` is an upper bound. Generation stops at EOS or the configured context limit.')
code('''
prompt = 'Write a paragraph about a neural processing unit (NPU).'
ids, text, stats = engine.generate(
    prompt, max_new_tokens=32, chat=True,
    on_token=lambda _, piece: print(piece,end='',flush=True),
)
print()
assert len(ids)>1 and text.strip(), 'The interactive example did not produce meaningful output.'
display(stats)
save(RUN/'interactive.json',{'prompt':prompt,'ids':ids,'text':text,'stats':stats})
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (PYNQ)','language':'python','name':'python3'},'language_info':{'name':'python'}})
nbf.validate(nb)
nbf.write(nb,root/'smollm_board_onnx_quant.ipynb')
