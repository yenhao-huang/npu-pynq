"""Parallel CRC and autonomous LFSR jump networks over GF(2)."""
from collections import Counter
from itertools import combinations
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from . import CATEGORY
from .exploration import Result
from .networks import RTL, emit


class LinearIn(ExplorationInput):
    backend: str | None = Field(default='linear_rtl',description='Binary linear recurrence backend.')
    width: int = Field(default=32,ge=8,le=64,description='State width; the leading polynomial coefficient is implicit.')
    polynomial: int = Field(default=0x04C11DB7,ge=1,description='Low coefficients of a monic odd polynomial, strictly below 2**width.')
    architecture: Literal['unrolled','matrix','shared'] = Field(default='shared',description='Unrolled bit steps, independent balanced matrix rows, or greedy shared XOR pairs.')

    @model_validator(mode='after')
    def valid_polynomial(self):
        if self.polynomial>=1<<self.width or not self.polynomial&1:
            raise ValueError('polynomial must be odd and fit the declared state width')
        return self


class CRCIn(LinearIn):
    data_width: int = Field(default=64,ge=8,le=256,description='Data bits consumed MSB first in one combinational update.')


class JumpIn(LinearIn):
    steps: int = Field(default=128,ge=1,le=256,description='Autonomous left-shift recurrence steps advanced per update.')


def symbolic_step(state,polynomial,incoming=0):
    feedback=state[-1]^incoming
    return [((state[bit-1] if bit else 0) ^ (feedback if (polynomial>>bit)&1 else 0)) for bit in range(len(state))]


def compose(left,right):
    result=[]
    for row in left:
        value=0
        while row:
            selected=row & -row
            value ^= right[selected.bit_length()-1];row^=selected
        result.append(value)
    return result


def jump_matrix(width,polynomial,steps):
    identity=[1<<bit for bit in range(width)]
    transform=symbolic_step(identity,polynomial);result=identity
    while steps:
        if steps&1: result=compose(transform,result)
        steps>>=1
        if steps: transform=compose(transform,transform)
    return result


def xor_network(rtl,matrix,shared):
    rows=[set(bit for bit in range(rtl.inputs) if row>>bit&1) for row in matrix]
    signals={bit:f'x[{bit}]' for bit in range(rtl.inputs)}
    next_id=rtl.inputs;shared_nodes=0
    if shared:
        while True:
            counts=Counter(pair for row in rows for pair in combinations(sorted(row),2))
            if not counts: break
            pair,count=min(counts.items(),key=lambda item:(-item[1],item[0]))
            if count<2: break
            signals[next_id]=rtl.wire(1,f'{signals[pair[0]]} ^ {signals[pair[1]]}')
            for row in rows:
                if pair[0] in row and pair[1] in row:
                    row.difference_update(pair);row.add(next_id)
            next_id+=1;shared_nodes+=1
    outputs=[]
    for row in rows:
        terms=[signals[bit] for bit in sorted(row)]
        if not terms: outputs.append("1'b0");continue
        while len(terms)>1:
            terms=[rtl.wire(1,f'{terms[i]} ^ {terms[i+1]}') if i+1<len(terms) else terms[i] for i in range(0,len(terms),2)]
        outputs.append(terms[0])
    return '{'+','.join(reversed(outputs))+'}',shared_nodes


def unroll(rtl,width,polynomial,steps,data_width=0):
    state=rtl.wire(width,f'x[{width-1}:0]')
    for step in range(steps):
        feedback=f'{state}[{width-1}]'
        if data_width: feedback=rtl.wire(1,f'{feedback} ^ x[{width+data_width-1-step}]')
        state=rtl.wire(width,f"{{{state}[{width-2}:0],1'b0}} ^ ({{{width}{{{feedback}}}}} & {width}'d{polynomial})")
    return state


@backend('synth','linear_rtl')
class Linear:
    def synth_crc_parallel(self,p,ctx):
        rtl=RTL(p.width+p.data_width,p.width)
        matrix=[1<<bit for bit in range(p.width)]
        for bit in reversed(range(p.data_width)):
            matrix=symbolic_step(matrix,p.polynomial,1<<(p.width+bit))
        if p.architecture=='unrolled': result=unroll(rtl,p.width,p.polynomial,p.data_width,p.data_width);count=0
        else: result,count=xor_network(rtl,matrix,p.architecture=='shared')
        out=emit(ctx,rtl,result,p.architecture,'CRC state update, MSB-first and non-reflected: low input bits are current state, high bits are data. No implicit initial/final XOR or augmentation. Every binary starting state is valid.')
        out.data.update(state_width=p.width,data_width=p.data_width,polynomial=p.polynomial,shared_xor_nodes=count,matrix_rows=[hex(row) for row in matrix])
        return out

    def synth_lfsr_jump(self,p,ctx):
        rtl=RTL(p.width,p.width);matrix=jump_matrix(p.width,p.polynomial,p.steps)
        if p.architecture=='unrolled': result=unroll(rtl,p.width,p.polynomial,p.steps);count=0
        else: result,count=xor_network(rtl,matrix,p.architecture=='shared')
        out=emit(ctx,rtl,result,p.architecture,'Autonomous MSB-feedback left-shift recurrence; matrix exponentiation advances the exact declared number of steps. Zero is a valid fixed state. No maximal-period or randomness claim.')
        out.data.update(state_width=p.width,steps=p.steps,polynomial=p.polynomial,shared_xor_nodes=count,matrix_rows=[hex(row) for row in matrix])
        return out


CATEGORY.ops.append(Op('synth_crc_parallel',CRCIn,Result,'Generate MSB-first parallel CRC transforms with explicit state/data packing and XOR sharing.'))
CATEGORY.ops.append(Op('synth_lfsr_jump',JumpIn,Result,'Generate exact LFSR jump networks using GF(2) matrix exponentiation and shared XORs.'))
