"""Run every named official-model cell and always save its actual outputs."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from acceptance import save

p=argparse.ArgumentParser()
p.add_argument('--run-id',required=True)
p.add_argument('--timeout',type=int,default=1800)
a=p.parse_args()
deploy=Path(__file__).resolve().parents[1]
run=deploy/'runs'/a.run_id
run.mkdir(parents=True,exist_ok=True)
for key in ('NPU_DEPLOY','NPU_REPO'): os.environ.pop(key,None)
os.environ['NPU_RUN_ID']=a.run_id
os.environ.setdefault('XILINX_XRT','/usr')
source=deploy/'smollm_board_onnx_quant.ipynb'
nb=nbformat.read(source,as_version=4)
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.execution_count=None
        cell.outputs=[]
def cell_start(cell,cell_index,**kwargs):
    print('CELL START',cell_index,cell['source'].splitlines()[0],flush=True)
class AuditedNotebookClient(NotebookClient):
    async def async_execute_cell(self,cell,cell_index,*args,**kwargs):
        result=await super().async_execute_cell(cell,cell_index,*args,**kwargs)
        if cell.cell_type=='code':
            print('CELL DONE',cell_index,'execution_count',cell.execution_count,flush=True)
            nbformat.write(self.nb,run/'smollm_board_onnx_quant.checkpoint.ipynb')
        return result

client=AuditedNotebookClient(nb,timeout=a.timeout,kernel_name='python3',
                      resources={'metadata':{'path':str(run)}},
                      on_cell_execute=cell_start)
success=False
try:
    client.execute()
    success=True
finally:
    executed=run/'smollm_board_onnx_quant.executed.ipynb'
    nbformat.write(nb,executed)
    code=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    audit={'success':success,'code_cells':len(code),'executed_code_cells':sum(c.execution_count is not None for c in code),
           'errors':[{'ename':o.ename,'evalue':o.evalue} for o in errors]}
    save(run/'notebook-audit.json',audit)
    if success and len(code)==audit['executed_code_cells'] and not errors:
        # The user's exact URL opens the verified output-bearing notebook.
        nbformat.write(nb,source)
    print('NOTEBOOK AUDIT',audit,flush=True)
