"""Yosys netlist attribution, electrical-load counts and memory mapping checks."""
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Literal
from pydantic import Field
from ...errors import InvalidInput
from ...process import run as run_process
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, fingerprint, sources
from . import CATEGORY
from .exploration import Result


class ProfileIn(ExplorationInput):
    backend: str | None = Field(default='netlist_analysis',description='Yosys netlist profiling backend.')
    files: list[str] = Field(min_length=1,description='Self-contained RTL source files.')
    top: Identifier = Field(description='Top module to flatten and inspect.')
    mapping: Literal['generic','xc7'] = Field(default='generic',description='Coarse operators/memories or Xilinx 7-series mapped estimate.')
    container: str | None = Field(default=None,pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$',description='Existing Yosys Docker container; null selects local Yosys.')
    timeout_s: int = Field(default=120,ge=1,le=1800,description='Overall synthesis time limit, including technology mapping.')


class NetlistIn(ExplorationInput):
    backend: str | None = Field(default='netlist_analysis',description='Read-only netlist analysis backend.')
    netlist_file: str = Field(description='JSON netlist path returned by netlist_profile.')
    top: Identifier = Field(description='Module to inspect.')
    expected_sha256: str = Field(pattern=r'^[0-9a-f]{64}$',description='Expected JSON digest from the producer; changed files are rejected.')


class FanoutIn(NetlistIn):
    threshold: int = Field(default=16,ge=1,description='Minimum connected sink pins to flag.')
    limit: int = Field(default=30,ge=1,le=500,description='Maximum high-fanout nets to return.')


class ConeIn(NetlistIn):
    endpoints: list[str] = Field(default=['y'],min_length=1,max_length=100,description='Named output ports or internal nets whose logic fan-in should be inspected.')
    limit: int = Field(default=100,ge=1,le=1000,description='Maximum cell names to return; full cone counts remain available.')


def read_netlist(path,top,expected=None):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>64_000_000:
        raise InvalidInput('Netlist must be an existing JSON file below 64 MB')
    raw=path.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    if expected is not None and sha!=expected: raise InvalidInput('Netlist digest mismatch')
    try:
        document=json.loads(raw)
        module=document['modules'][top]
        if not isinstance(module['cells'],dict) or not isinstance(module['ports'],dict): raise ValueError()
        for cell in module['cells'].values():
            if not isinstance(cell['type'],str) or not isinstance(cell['connections'],dict) or not isinstance(cell['port_directions'],dict): raise ValueError()
    except (ValueError,KeyError,TypeError) as exc:
        raise InvalidInput('Malformed Yosys module JSON') from exc
    return module,sha


def number(value):
    if isinstance(value,int) and not isinstance(value,bool) and value>=0: return value
    if isinstance(value,str) and re.fullmatch('[01]+',value): return int(value,2)
    raise InvalidInput('Memory dimensions must be defined binary integers')


@backend('synth','netlist_analysis')
class NetlistAnalysis:
    def netlist_profile(self,p,ctx):
        files=sources(p.files,ctx.cwd)
        if any('`include' in path.read_text(encoding='utf-8') for path in files):
            raise InvalidInput('Inline includes for complete synthesis source identity')
        source_sha=fingerprint(files,p.top)
        stage=ctx.run.work/'netlist';stage.mkdir()
        names=[]
        for index,path in enumerate(files):
            name=f'source{index}.sv';(stage/name).write_bytes(path.read_bytes());names.append(name)
        flow=(f'synth_xilinx -family xc7 -top {p.top} -flatten' if p.mapping=='xc7' else
              f'hierarchy -check -top {p.top}; flatten; proc; opt; memory_collect; opt')
        (stage/'profile.ys').write_text('read_verilog -sv '+' '.join(names)+'\n'+flow+'\ncheck -assert\nwrite_json netlist.json\n',encoding='utf-8')
        prefix=[]
        if p.container:
            remote='/tmp/ic-netlist-'+ctx.run.run_id
            for label,argv in [('mkdir',['docker','exec',p.container,'mkdir',remote]),('copy',['docker','cp',str(stage)+'/.',p.container+':'+remote])]:
                result=run_process(argv,cwd=stage,log_path=ctx.run.artifacts/(label+'.log'),timeout_s=30)
                if result.exit_code or result.timed_out:
                    return Result(ok=False,data={'log':ctx.run.handle(label+'.log')},note='Container staging failed; no netlist estimate.')
            prefix=['docker','exec','-w',remote,p.container]
        elif not shutil.which('yosys'):
            raise InvalidInput('Local Yosys unavailable; provide an existing container')
        version=run_process(prefix+['yosys','-V'],cwd=stage,log_path=ctx.run.artifacts/'version.log',timeout_s=30)
        guard=['timeout','--signal=TERM','--kill-after=5s',str(p.timeout_s)] if p.container else []
        result=run_process(prefix+guard+['yosys','-s','profile.ys'],cwd=stage,log_path=ctx.run.artifacts/'netlist.log',timeout_s=p.timeout_s+15)
        if result.exit_code or result.timed_out:
            return Result(ok=False,data={'log':ctx.run.handle('netlist.log'),'exit_code':result.exit_code,'timed_out':result.timed_out or result.exit_code in (124,137)},note='Synthesis failed or timed out; no estimate.')
        artifact=ctx.run.artifacts/'netlist.json'
        if p.container:
            copied=run_process(['docker','cp',p.container+':'+remote+'/netlist.json',str(artifact)],cwd=stage,log_path=ctx.run.artifacts/'fetch.log',timeout_s=30)
            if copied.exit_code or copied.timed_out: return Result(ok=False,data={'log':ctx.run.handle('fetch.log')},note='Cannot retrieve generated netlist.')
        else:
            shutil.copyfile(stage/'netlist.json',artifact)
        if fingerprint(files,p.top)!=source_sha: return Result(ok=False,data={},note='Source changed; estimate discarded.')
        module,sha=read_netlist(artifact,p.top)
        histogram=Counter(cell['type'] for cell in module['cells'].values())
        widths=defaultdict(int)
        for cell in module['cells'].values():
            for name,direction in cell['port_directions'].items():
                if direction=='output': widths[cell['type']]+=len(cell['connections'][name])
        return Result(data=dict(source_sha256=source_sha,netlist_sha256=sha,netlist=ctx.run.handle(artifact.name),netlist_file=str(ctx.run.final_artifact(artifact.name)),top=p.top,mapping=p.mapping,engine=version.text().strip(),cell_count=sum(histogram.values()),cell_types=dict(histogram),output_bits_by_cell_type=dict(widths),log=ctx.run.handle('netlist.log')),note='Yosys structural estimate, not routed Vivado area or timing. Generic cells remain word-level operators; xc7 cells are mapped estimates.')

    def fanout_analysis(self,p,ctx):
        module,sha=read_netlist(ctx.cwd/p.netlist_file,p.top,p.expected_sha256)
        sinks=defaultdict(list);drivers=defaultdict(list);aliases=defaultdict(list)
        for name,net in module.get('netnames',{}).items():
            for index,bit in enumerate(net['bits']):
                if isinstance(bit,int): aliases[bit].append(f'{name}[{index}]')
        for name,cell in module['cells'].items():
            for port,bits in cell['connections'].items():
                direction=cell['port_directions'].get(port)
                if direction not in ('input','output'): raise InvalidInput('Unsupported cell port direction')
                target=sinks if direction=='input' else drivers
                for index,bit in enumerate(bits):
                    if isinstance(bit,int): target[bit].append(f'{name}.{port}[{index}]')
        for name,port in module['ports'].items():
            target=sinks if port['direction']=='output' else drivers
            for index,bit in enumerate(port['bits']):
                if isinstance(bit,int): target[bit].append(f'port:{name}[{index}]')
        rows=[dict(bit=bit,aliases=aliases[bit][:8],sink_pins=len(pins),drivers=drivers[bit][:8],sample_sinks=pins[:12]) for bit,pins in sinks.items() if len(pins)>=p.threshold]
        rows.sort(key=lambda row:(-row['sink_pins'],row['bit']))
        return Result(data=dict(netlist_sha256=sha,total_flagged=len(rows),nets=rows[:p.limit],truncated=len(rows)>p.limit),note='Counts connected sink pins, including clock/reset/enable and output ports; not electrical capacitance or routed delay. Constant connections are excluded.')


    def critical_cone(self,p,ctx):
        module,sha=read_netlist(ctx.cwd/p.netlist_file,p.top,p.expected_sha256)
        cells=module['cells'];drivers=defaultdict(set)
        for name,cell in cells.items():
            for port,direction in cell['port_directions'].items():
                if direction=='output':
                    for bit in cell['connections'][port]:
                        if isinstance(bit,int): drivers[bit].add(name)
        bits=[]
        for endpoint in p.endpoints:
            net=module['ports'].get(endpoint,module.get('netnames',{}).get(endpoint))
            if net is None: raise InvalidInput('Unknown cone endpoint: '+endpoint)
            bits.extend(bit for bit in net['bits'] if isinstance(bit,int))
        pending=list(bits);seen=set();cone=set();boundaries=set()
        while pending:
            bit=pending.pop()
            if bit in seen: continue
            seen.add(bit)
            for name in drivers[bit]:
                kind=cells[name]['type']
                if re.search(r'ff|latch|mem',kind,re.I) or kind.startswith(('FD','RAM','SRL')):
                    boundaries.add(name);continue
                if name in cone: continue
                cone.add(name)
                for port,direction in cells[name]['port_directions'].items():
                    if direction=='input': pending.extend(bit for bit in cells[name]['connections'][port] if isinstance(bit,int))
        dependencies={name:set() for name in cone};successors=defaultdict(set)
        for name in cone:
            for port,direction in cells[name]['port_directions'].items():
                if direction=='input':
                    for bit in cells[name]['connections'][port]:
                        dependencies[name].update(drivers.get(bit,set()) & cone)
            for upstream in dependencies[name]: successors[upstream].add(name)
        degrees={name:len(deps) for name,deps in dependencies.items()}
        ready=deque(sorted(name for name,degree in degrees.items() if degree==0))
        depth={};predecessor={}
        while ready:
            name=ready.popleft()
            previous=max(dependencies[name],key=lambda n:(depth[n],n),default=None)
            predecessor[name]=previous;depth[name]=1+(depth[previous] if previous else 0)
            for downstream in sorted(successors[name]):
                degrees[downstream]-=1
                if degrees[downstream]==0: ready.append(downstream)
        if len(depth)!=len(cone): raise InvalidInput('Combinational cycle prevents a finite cone-depth estimate')
        tip=max(depth,key=lambda n:(depth[n],n),default=None)
        path=[]
        while tip is not None:
            path.append(tip);tip=predecessor[tip]
        path.reverse()
        return Result(data=dict(netlist_sha256=sha,endpoints=p.endpoints,combinational_cells=len(cone),logic_levels=max(depth.values(),default=0),cell_types=dict(Counter(cells[name]['type'] for name in cone)),boundary_cells=sorted(boundaries)[:p.limit],longest_structural_path=path[:p.limit],path_truncated=len(path)>p.limit,multiple_driver_bits=[bit for bit in sorted(seen) if len(drivers[bit])>1]),note='Logical fan-in and unweighted cell depth for selected endpoints, not routed timing criticality. Flip-flops, latches and memories are opaque boundaries; memory read timing requires separate physical evidence.')

    def memory_inference(self,p,ctx):
        module,sha=read_netlist(ctx.cwd/p.netlist_file,p.top,p.expected_sha256)
        groups=Counter(); logical_bits=0; logical=[];unknown=[]
        for name,cell in module['cells'].items():
            kind=cell['type']
            if kind in ('$mem','$mem_v2'):
                try:
                    params=cell['parameters'];width=number(params['WIDTH']);depth=number(params['SIZE'])
                    reads=number(params['RD_PORTS']);writes=number(params['WR_PORTS'])
                    if width==0 or depth==0: raise ValueError()
                except (KeyError,TypeError,ValueError) as exc:
                    raise InvalidInput('Malformed memory dimensions') from exc
                logical_bits+=width*depth
                logical.append(dict(name=name,width=width,depth=depth,read_ports=reads,write_ports=writes))
            elif kind.startswith('RAMB18'): groups['bram18_primitives']+=1
            elif kind.startswith('RAMB36'): groups['bram36_primitives']+=1
            elif re.match(r'^(RAM(16|32|64|128|256)|RAM[DS](32|64))',kind): groups['distributed_ram_primitives']+=1
            elif kind.startswith(('SRL16','SRLC32')): groups['shift_register_primitives']+=1
            elif kind.startswith(('FDRE','FDSE','FDCE','FDPE','$_DFF','$dff','$adff','$sdff')):
                groups['register_output_bits']+=sum(len(cell['connections'][port]) for port,direction in cell['port_directions'].items() if direction=='output')
            elif 'mem' in kind.lower() or kind.startswith('RAM'): unknown.append(dict(name=name,type=kind))
        return Result(data=dict(netlist_sha256=sha,logical_memory_bits=logical_bits,logical_memories=logical,physical_primitives=dict(groups),unclassified=unknown),note='Reports logical memories and mapped primitive classes separately. Registers may store control or datapath state; they are not automatically FIFO storage. No Vivado inference or power claim.')


CATEGORY.ops.append(Op('netlist_profile',ProfileIn,Result,'Synthesize a source-bound generic or xc7 netlist and attribute operator/resource structure.',long_running=True))
CATEGORY.ops.append(Op('fanout_analysis',FanoutIn,Result,'Locate high-load nets with driver and sink-pin attribution in a verified netlist.'))
CATEGORY.ops.append(Op('memory_inference',NetlistIn,Result,'Distinguish logical memories, RAM primitives, shift registers and flip-flop storage estimates.'))

CATEGORY.ops.append(Op('critical_cone',ConeIn,Result,'Extract endpoint fan-in, state boundaries and longest structural logic paths for targeted rewriting.'))
