"""Reproducible host-side Docker evaluation: verify -> synthesize -> STA.

Requires Docker CLI, the existing Verilator/Yosys container, and openroad/orfs.
No third-party Python packages are needed. Commands and hashes are persisted.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from generate import generate

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
LINUX=os.environ.get('EVAL_LINUX_ROOT','/workspace/npu/npu_repo_in_pynq/exp/fp32_multipier')
PDK=Path(os.environ.get('EVAL_PDK',str(REPO/'external/OpenROAD-flow-scripts/flow/platforms/asap7'))).resolve()
CONTAINER=os.environ.get('EVAL_CONTAINER','codex-sandbox-agent-workspace')
IMAGE=os.environ.get('OPENROAD_IMAGE','openroad/orfs@sha256:7f9c688e71cd49379b9e9a45a4558724e3eda501eaaaf11bc9b77382a3fe647b')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(args, log, timeout=600):
    start=time.monotonic()
    with open(log,'w',encoding='utf-8') as f:
        f.write(json.dumps(args)+'\n');f.flush()
        proc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if proc.returncode: raise RuntimeError(f'exit={proc.returncode}: {log}')
    return time.monotonic()-start

def docker(args): return ['docker','exec','-w',LINUX,CONTAINER,*args]

def cells(text):
    for m in re.finditer(r'\bcell\s*\([^)]*\)\s*\{',text):
        depth=1;i=m.end()
        while depth:
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            i+=1
        yield text[m.start():i]

def prepare():
    target=ROOT/'build/pdk';target.mkdir(parents=True,exist_ok=True)
    files=[]
    for group in ['AO','INVBUF','OA','SIMPLE']:
        files += list((PDK/'lib/NLDM').glob(f'asap7sc7p5t_{group}_RVT_TT_*.lib.gz'))
    if len(files)!=4:raise RuntimeError('Expected four ASAP7 RVT TT combinational libraries')
    texts=[gzip.open(p,'rt').read() for p in files]
    first=texts[0]; merged=first[:first.rfind('}')]+ '\n'+'\n'.join(c for s in texts[1:] for c in cells(s))+'\n}\n'
    lib=target/'asap7_rvt_tt.lib'
    if lib.exists() and lib.read_text()!=merged:
        raise RuntimeError('Cached mapping library differs from configured PDK; use a fresh build directory')
    if not lib.exists():lib.write_text(merged)
    (target/'sources.json').write_text(json.dumps({str(p.relative_to(PDK)):digest(p) for p in files},indent=2))
    return lib

def evaluate(name,mode,mul,lzd,delay,exhaustive=False,fast=False,constrained=False,source=None):
    dest=ROOT/'runs'/name
    if dest.exists(): raise RuntimeError(f'Run already exists: {dest}; choose a fresh name')
    dest.mkdir(parents=True)
    rel=f'runs/{name}';dl=f'{LINUX}/{rel}'
    meta={'name':name,'mode':mode,'mul':mul,'lzd':lzd,'target_delay_ps':delay,'status':'running','started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'image':IMAGE,'metric':'post-mapping zero-wire-load ASAP7 RVT TT','exhaustive':exhaustive,'abc_fast':fast,'abc_constrained':constrained,'generator_sha256':digest(ROOT/'generate.py'),'checker_sha256':digest(ROOT/'test/check.cpp')}
    def save(): (dest/'metrics.json').write_text(json.dumps(meta,indent=2))
    save()
    try:
        lib=prepare()
        rtl=dest/'design.sv'
        if source:
            import shutil
            shutil.copyfile(source,rtl)
        else:rtl.write_text(generate(mode,mul,lzd))
        meta['rtl_sha256']=digest(rtl);meta['lib_sha256']=digest(lib)
        (dest/'obj').mkdir()
        run(docker(['verilator','--cc','--exe','--build','-j','2','-Wno-fatal','--top-module','fp12_mul','--Mdir',f'{dl}/obj',f'{dl}/design.sv',f'{LINUX}/test/check.cpp']),dest/'compile.log')
        run(docker([f'{dl}/obj/Vfp12_mul']+(['--exhaustive'] if exhaustive else [])),dest/'simulation.log')
        meta['checks']=int(re.search(r'PASS checks=(\d+)',(dest/'simulation.log').read_text()).group(1));save()
        liblinux=f'{LINUX}/build/pdk/asap7_rvt_tt.lib'
        (dest/'abc.constr').write_text('set_driving_cell BUFx2_ASAP7_75t_R\nset_load 3.898\n')
        constr=f'-constr {dl}/abc.constr ' if constrained else ''
        synth=f'''read_verilog -sv {dl}/design.sv
hierarchy -check -top fp12_mul
synth -top fp12_mul -noabc
abc {'-fast ' if fast else ''}-liberty {liblinux} {constr}-D {delay}
clean
read_liberty -lib {liblinux}
check -assert
stat -liberty {liblinux}
write_verilog -noattr -noexpr {dl}/mapped.v
write_json {dl}/mapped.json
'''
        (dest/'synth.ys').write_text(synth)
        run(docker(['yosys','-s',f'{dl}/synth.ys']),dest/'synthesis.log')
        txt=(dest/'synthesis.log').read_text();match=re.search(r'Chip area for module.*?:\s*([\d.]+)',txt)
        if not match:raise RuntimeError('Missing mapped area')
        meta['area_um2']=float(match.group(1));save()
        # Functional cell models are generated from the exact mapping Liberty.
        (dest/'cells.ys').write_text(f'read_liberty {liblinux}\nwrite_verilog -noattr {dl}/cells.v\n')
        run(docker(['yosys','-s',f'{dl}/cells.ys']),dest/'cells.log')
        (dest/'gate_obj').mkdir()
        run(docker(['verilator','--cc','--exe','--build','-j','2','-Wno-fatal','--top-module','fp12_mul','--Mdir',f'{dl}/gate_obj',f'{dl}/mapped.v',f'{dl}/cells.v',f'{LINUX}/test/check.cpp']),dest/'gate_compile.log')
        run(docker([f'{dl}/gate_obj/Vfp12_mul']+(['--exhaustive'] if exhaustive else [])),dest/'gate_simulation.log')
        meta['gate_checks']=int(re.search(r'PASS checks=(\d+)',(dest/'gate_simulation.log').read_text()).group(1));save()
        from sta import measure
        meta.update(measure(dest,run,IMAGE))
        meta['adp_um2_ps']=meta['area_um2']*meta['delay_ps'];meta['meets_mapping_target']=meta['delay_ps']<=delay;meta['mapped_sha256']=digest(dest/'mapped.v');meta['status']='pass'
    except Exception as e:
        meta['status']='failed';meta['error']=str(e)
    meta['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save();print(json.dumps(meta),flush=True)
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--mode',default='baseline',choices=['baseline','slices','compact','dual','prenorm','packed','parallel','onehot']);p.add_argument('--mul',default='operator',choices=['operator','rows','split','csa']);p.add_argument('--lzd',default='priority',choices=['priority','case']);p.add_argument('--delay',type=int,default=1200);p.add_argument('--exhaustive',action='store_true');p.add_argument('--fast',action='store_true');p.add_argument('--constrained',action='store_true');p.add_argument('--rtl',type=Path,help='Evaluate an existing fp12_mul RTL snapshot')
    a=p.parse_args();r=evaluate(a.name,a.mode,a.mul,a.lzd,a.delay,a.exhaustive,a.fast,a.constrained,a.rtl);raise SystemExit(0 if r['status']=='pass' else 1)
