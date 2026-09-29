"""Exact affine GF(2) equivalence for a conservative Yosys operator subset."""
from collections import defaultdict, deque
import hashlib
import json
from pydantic import Field
from ...dispatch import dispatch
from ...errors import InvalidInput
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, Identifier, sources, fingerprint
from ..synth.exploration import Result
from ..synth.netlist import read_netlist, number
from . import CATEGORY


class AffineIn(ExplorationInput):
    backend: str | None = Field(default='affine_equivalence',description='Exact two-state affine netlist equivalence backend.')
    reference_files: list[str] = Field(min_length=1,description='Self-contained reference RTL sources.')
    candidate_files: list[str] = Field(min_length=1,description='Self-contained candidate RTL sources.')
    top: Identifier = Field(default='dut',description='Shared module name with complete packed x/y ports.')
    input_width: int = Field(ge=1,le=4096,description='Complete x input width.')
    output_width: int = Field(ge=1,le=4096,description='Complete y output width.')
    container: str | None = Field(default=None,pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$',description='Existing Yosys Docker container; null uses local Yosys.')
    timeout_s: int = Field(default=120,ge=1,le=600,description='Maximum runtime per source self-check or netlist synthesis process.')


def affine_rows(module,inputs,outputs):
    """Map each output to input coefficients plus a constant bit at index inputs."""
    ports=module['ports'];cells=module['cells'];bias=1<<inputs
    if set(ports)!={'x','y'} or ports['x']['direction']!='input' or ports['y']['direction']!='output':
        raise InvalidInput('Affine proof requires only complete x input and y output ports')
    x=ports['x']['bits'];y=ports['y']['bits']
    if len(x)!=inputs or len(y)!=outputs or len(set(x))!=inputs or any(type(bit) is not int for bit in x):
        raise InvalidInput('Affine proof interface mismatch or aliased input bits')
    if len(cells)>50000: raise InvalidInput('Affine proof exceeds 50000-cell bound')
    values={bit:1<<index for index,bit in enumerate(x)};drivers=set(x)
    binary={'$xor','$xnor','$and','$or'}
    unary={'$not','$pos','$reduce_xor','$reduce_xnor'}
    pending={}
    for name,cell in cells.items():
        kind=cell['type'];expected={'A','B','Y'} if kind in binary else {'A','Y'}
        if kind not in binary|unary: raise InvalidInput('Unsupported or stateful affine cell: '+kind)
        connection=cell['connections']
        if set(connection)!=expected or cell['port_directions']!={port:('output' if port=='Y' else 'input') for port in expected}:
            raise InvalidInput('Malformed affine cell interface')
        for bits in connection.values():
            if not isinstance(bits,list) or any(not (type(bit) is int or bit in ('0','1')) for bit in bits):
                raise InvalidInput('Affine proof rejects unknown or malformed bits')
        for bit in connection['Y']:
            if type(bit) is not int or bit in drivers: raise InvalidInput('Multiple or constant-driven affine output')
            drivers.add(bit)
        pending[name]=cell
    def get(bit):
        if bit=='0': return 0
        if bit=='1': return bias
        if type(bit) is not int: raise InvalidInput('Unknown output bit in affine proof')
        return values[bit]
    def resize(bits,width,sign):
        row=[get(bit) for bit in bits]
        if not row: raise InvalidInput('Empty affine operand')
        return (row+[(row[-1] if sign else 0)]*max(0,width-len(row)))[:width]
    missing={};users=defaultdict(set)
    for name,cell in pending.items():
        needed={bit for port,bits in cell['connections'].items() if port!='Y' for bit in bits if type(bit) is int and bit not in values}
        if not needed <= drivers: raise InvalidInput('Undriven input prevents affine proof')
        missing[name]=needed
        for bit in needed: users[bit].add(name)
    ready=deque(sorted(name for name,needed in missing.items() if not needed))
    while ready:
        name=ready.popleft();cell=pending[name]
        con=cell['connections'];kind=cell['type'];width=len(con['Y']);params=cell.get('parameters',{})
        try:
            a_signed=bool(number(params.get('A_SIGNED',0)))
            b_signed=bool(number(params.get('B_SIGNED',0)))
            a=resize(con['A'],width,a_signed and (b_signed if kind in binary else True))
            b=resize(con['B'],width,a_signed and b_signed) if kind in binary else []
            if kind in ('$reduce_xor','$reduce_xnor'):
                parity=0
                for bit in con['A']: parity ^= get(bit)
                if kind=='$reduce_xnor': parity ^= bias
                result=[parity]+[0]*(width-1)
            elif kind=='$pos': result=a
            elif kind=='$not': result=[value^bias for value in a]
            else:
                result=[]
                for left,right in zip(a,b):
                    if kind in ('$xor','$xnor'): value=left^right^(bias if kind=='$xnor' else 0)
                    elif left==right: value=left
                    elif kind=='$and' and (left in (0,bias) or right in (0,bias)):
                        value=0 if left==0 or right==0 else (right if left==bias else left)
                    elif kind=='$or' and (left in (0,bias) or right in (0,bias)):
                        value=bias if left==bias or right==bias else (right if left==0 else left)
                    else: raise InvalidInput('Nonlinear AND/OR cannot receive an affine proof')
                    result.append(value)
        except KeyError as exc:
            raise InvalidInput('Unresolved affine operand') from exc
        values.update(zip(con['Y'],result));del pending[name]
        for bit in con['Y']:
            for user in sorted(users[bit]):
                missing[user].remove(bit)
                if not missing[user]: ready.append(user)
    if pending: raise InvalidInput('Cycle prevents affine proof')
    try: return [get(bit) for bit in y]
    except KeyError as exc: raise InvalidInput('Undriven output prevents affine proof') from exc


def evaluate(rows,value,inputs):
    augmented=value|(1<<inputs)
    return sum(((row&augmented).bit_count()&1)<<bit for bit,row in enumerate(rows))


@backend('debug','affine_equivalence')
class Affine:
    def gf2_equivalence(self,p,ctx):
        designs=[sources(p.reference_files,ctx.cwd),sources(p.candidate_files,ctx.cwd)]
        hashes=[fingerprint(paths,p.top) for paths in designs]
        profiles=[];matrices=[];source_guards=[]
        for paths,sha in zip(designs,hashes):
            # Optimized affine logic alone cannot establish source semantics:
            # e.g. an out-of-range select XOR itself may disappear. Reuse the
            # SAT backend's pre-optimization total-binary/interface checks on
            # each source before trusting its simplified affine expression.
            guard=dispatch('yosys_equivalence',dict(reference_files=[str(path) for path in paths],candidate_files=[str(path) for path in paths],top=p.top,input_width=p.input_width,output_width=p.output_width,container=p.container,timeout_s=p.timeout_s),cwd=ctx.cwd,store=ctx.store)
            source_guards.append(guard)
            data=guard.get('data',{})
            if (not guard.get('ok') or not data.get('proved')
                    or not data.get('interface_complete') or not data.get('abstraction_defined')
                    or any(data.get(role+'_sha256')!=sha for role in ('reference','candidate'))):
                return Result(ok=False,data=dict(proved=False,source_guards=source_guards,profiles=profiles),note='Original source semantics or complete interface not established; no affine proof.')
            profile=dispatch('netlist_profile',dict(files=[str(path) for path in paths],top=p.top,mapping='generic',container=p.container,timeout_s=p.timeout_s),cwd=ctx.cwd,store=ctx.store)
            profiles.append(profile)
            if not profile['ok']: return Result(ok=False,data=dict(source_guards=source_guards,profiles=profiles),note='Netlist synthesis failed; no affine proof.')
            data=profile['data']
            if data['source_sha256']!=sha: raise InvalidInput('Affine proof source identity changed')
            module,_=read_netlist(data['netlist_file'],p.top,data['netlist_sha256'])
            matrices.append(affine_rows(module,p.input_width,p.output_width))
        if hashes!=[fingerprint(paths,p.top) for paths in designs]: raise InvalidInput('Source changed during affine proof')
        mismatch=next((index for index,(a,b) in enumerate(zip(*matrices)) if a!=b),None)
        counterexample=None
        if mismatch is not None:
            difference=matrices[0][mismatch]^matrices[1][mismatch]
            value=0 if difference>>p.input_width&1 else difference & -difference
            counterexample=dict(x=hex(value),reference_y=hex(evaluate(matrices[0],value,p.input_width)),candidate_y=hex(evaluate(matrices[1],value,p.input_width)))
        certificate=dict(source_sha256=hashes,input_width=p.input_width,output_width=p.output_width,rows=[[hex(row) for row in matrix] for matrix in matrices],netlist_sha256=[profile['data']['netlist_sha256'] for profile in profiles])
        raw=json.dumps(certificate,sort_keys=True).encode();(ctx.run.artifacts/'affine.json').write_bytes(raw)
        return Result(ok=mismatch is None,data=dict(proved=mismatch is None,method='exact_affine_gf2',reference_sha256=hashes[0],candidate_sha256=hashes[1],interface_complete=True,abstraction_defined=True,certificate=ctx.run.handle('affine.json'),certificate_sha256=hashlib.sha256(raw).hexdigest(),counterexample=counterexample,profiles=profiles,source_guards=source_guards),note='Exact two-state equivalence of affine output expressions derived from synthesized netlists, after separate source-semantic and complete-interface guards. Unsupported/nonlinear/stateful/unknown cells are rejected. The cross-design proof is algebraic; source self-checks do not establish cross-design equivalence.')


CATEGORY.ops.append(Op('gf2_equivalence',AffineIn,Result,'Prove complete affine binary interfaces using source-bound netlist coefficient propagation.',long_running=True))
