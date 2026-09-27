"""Checkpointed architecture studies with source-bound verification gates."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
from typing import Literal
import pydantic
from pydantic import BaseModel, ConfigDict, Field
from ....dispatch import dispatch
from ....errors import InvalidInput
from ....process import run as run_process
from ....registry import Op, backend
from ...exploration_common import ExplorationInput, Identifier, fingerprint
from ...synth.exploration import Result
from .. import CATEGORY


GENERATORS=Literal['synth_prefix_adder','synth_csd_multiplier','synth_mcm','synth_adder_tree','synth_popcount','synth_priority_encoder','synth_leading_zero','synth_barrel_shifter','synth_onehot_mux','synth_argmax_tree']


class Case(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name: Identifier
    generator: GENERATORS
    baseline: dict
    candidate: dict
    objective: Literal['area','throughput']


class SweepIn(ExplorationInput):
    backend: str | None = Field(default='architecture_study',description='Checkpointed verification and physical implementation backend.')
    study_name: Identifier = Field(description='Namespace beneath .ic/studies in the caller workspace.')
    cases: list[Case] = Field(min_length=1,max_length=100,description='Predeclared configurations, architectures and optimization objectives.')
    repeats: int = Field(default=3,ge=1,le=10,description='Number of paired physical runs per configuration.')
    period_ns: float = Field(default=5,gt=0,allow_inf_nan=False,description='Shared physical implementation clock period, ns.')
    device: str = Field(default='xc7z020clg400-1',pattern=r'^[A-Za-z0-9_-]+$',description='Matched FPGA part.')
    optimization_mode: Literal['Default','Basic'] = Field(default='Default',description='Matched Vivado optimization stage for every physical pair.')
    formal_container: str | None = Field(default=None,pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$',description='Existing container providing Yosys, or null for local Yosys.')
    formal_timeout_s: int = Field(default=120,ge=1,le=1800,description='Per-case SAT time limit.')
    verify_only: bool = Field(default=False,description='Run correctness gates without physical implementation.')
    require_proof: bool = Field(default=True,description='Skip implementation when SAT proof is not obtained; false permits diagnostic measurements only.')


@contextmanager
def study_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            if path.stat().st_size==0:
                stream.write(b'0'); stream.flush(); stream.seek(0)
            try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc: raise InvalidInput('This study is already running') from exc
        else:
            import fcntl
            try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc: raise InvalidInput('This study is already running') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream,fcntl.LOCK_UN)


def identity(params,ctx):
    result={'python':platform.python_version(),'pydantic':pydantic.__version__}
    commands={'iverilog':['iverilog','-V'],'yosys':['yosys','-V']}
    if params.formal_container:
        commands['yosys']=['docker','exec',params.formal_container,'yosys','-V']
    if not params.verify_only:
        commands['vivado']=[shutil.which('vivado') or 'vivado','-version']
    for name,argv in commands.items():
        run=run_process(argv,cwd=ctx.cwd,log_path=ctx.run.artifacts/(name+'-version.log'),timeout_s=30)
        lines=[line.strip() for line in run.text().splitlines() if re.search(r'^Icarus Verilog version|^Yosys |^Vivado v|^SW Build|^IP Build',line,re.I)]
        # Vivado 2026.1 Windows launcher returns 1 for a successful -version.
        vivado_version_exit=(name=='vivado' and run.exit_code==1
                             and any(re.match(r'^vivado v',line,re.I) for line in lines)
                             and any(line.startswith('SW Build ') for line in lines))
        if run.timed_out or (run.exit_code and not vivado_version_exit):
            raise InvalidInput(f'Cannot establish {name} version for study provenance')
        if not lines: raise InvalidInput(f'Missing {name} version identifier')
        result[name]={'command':argv,'version':lines}
    return result


def checkpoint_key(params,environment):
    digest=hashlib.sha256(json.dumps({'params':params.model_dump(),'environment':environment},sort_keys=True).encode())
    root=Path(__file__).resolve().parents[3]
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


@backend('pipeline','architecture_study')
class Sweep:
    def architecture_sweep(self,p,ctx):
        if len({c.name for c in p.cases})!=len(p.cases):
            raise InvalidInput('Case names must be unique')
        environment=identity(p,ctx)
        key=checkpoint_key(p,environment)
        directory=ctx.cwd/'.ic'/'studies'/p.study_name/key
        directory.mkdir(parents=True,exist_ok=True)
        children=[]
        def invoke(label,op,payload):
            path=directory/(label+'.json')
            if path.exists():
                envelope=json.loads(path.read_text())
                if envelope['operation']!=op or envelope['payload']!=payload:
                    raise InvalidInput('Checkpoint input mismatch')
                result=envelope['result']
            else:
                result=dispatch(op,payload,cwd=ctx.cwd,store=ctx.store)
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(operation=op,payload=payload,result=result),indent=2)+'\n',encoding='utf-8')
                temporary.replace(path)
            if result.get('run_id'): children.append(result['run_id'])
            return result
        cases=[]
        with study_lock(directory/'active.lock'):
            for case in p.cases:
                row={'name':case.name,'objective':case.objective,'generator':case.generator,'records':[],'accepted':False}
                designs=[]
                for label,params in [('baseline',case.baseline),('candidate',case.candidate)]:
                    result=invoke(case.name+'-'+label+'-generate',case.generator,params)
                    if not result['ok']: raise InvalidInput('Generator failed')
                    designs.append(result['data'])
                a,b=designs
                if any(a[k]!=b[k] for k in ('input_width','output_width','latency_cycles','initiation_interval')):
                    raise InvalidInput('Generated interfaces or cycle contracts differ')
                paths=[[Path(d['core_path']),Path(d['wrapper_path'])] for d in designs]
                if any(not path.exists() for pair in paths for path in pair):
                    raise InvalidInput('Checkpoint source artifact missing; do not reuse this study namespace')
                hashes=[fingerprint(pair,'registered_dut') for pair in paths]
                if hashes != [d.get('source_sha256') for d in designs]:
                    raise InvalidInput('Generated source artifacts changed since generation')
                payload=dict(reference_files=[a['core_path']],candidate_files=[b['core_path']],top='dut',input_width=a['input_width'],output_width=a['output_width'])
                check=invoke(case.name+'-vectors','vector_equivalence',dict(payload,random_vectors=8192,combinational_contract=True))
                proof=invoke(case.name+'-proof','yosys_equivalence',dict(payload,container=p.formal_container,timeout_s=p.formal_timeout_s))
                row.update(simulation=check,formal=proof)
                if not check['ok'] or (p.require_proof and not proof['ok']):
                    row['status']='correctness_not_established'; cases.append(row); continue
                core_hashes=[fingerprint([pair[0]],'dut') for pair in paths]
                for check_result in (check,proof):
                    if [check_result['data'].get(k) for k in ('reference_sha256','candidate_sha256')]!=core_hashes:
                        raise InvalidInput('Correctness evidence source mismatch')
                if p.verify_only:
                    row['status']='verified' if proof['ok'] else 'simulation_only'; cases.append(row); continue
                failed=False
                for repeat in range(p.repeats):
                    pair=[]
                    for i,label in enumerate(('baseline','candidate')):
                        result=invoke(f'{case.name}-{label}-r{repeat}','clocked_ppa',dict(files=[str(path) for path in paths[i]],top='registered_dut',name=case.name+'-'+label,period_ns=p.period_ns,device=p.device,optimization_mode=p.optimization_mode,latency_cycles=designs[i]['latency_cycles'],initiation_interval=designs[i]['initiation_interval']))
                        if not result['ok']:
                            row['failure']=result; failed=True; break
                        record=result['data']['record']
                        if record['source_sha256']!=hashes[i] or fingerprint(paths[i],'registered_dut')!=hashes[i]:
                            raise InvalidInput('Source changed between verification and implementation')
                        pair.append(record)
                    row['records'].append(pair)
                    if failed: break
                row['status']='measurement_failed' if failed else ('measured' if proof['ok'] else 'diagnostic_unproved')
                cases.append(row)
            summary=dict(checkpoint_key=key,checkpoint_directory=str(directory),environment=environment,cases=cases,children=children)
            (ctx.run.artifacts/'study.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        return Result(ok=all(c['status'] in ('verified','measured') for c in cases),data=summary,note='All predeclared cases retained. Measurements are not automatic acceptance; assess paired gains, resources and full diversity gates separately.')


CATEGORY.ops.append(Op('architecture_sweep',SweepIn,Result,'Run source-bound architecture comparisons with formal/simulation gates and environment-aware checkpoints.',long_running=True))
