"""Prefix addition and constant-coefficient arithmetic graph generation."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from . import CATEGORY
from .exploration import Result
from .networks import RTL, emit


class PrefixIn(ExplorationInput):
    backend: str | None = Field(default='arithmetic_rtl',description='Structured arithmetic generator backend.')
    width: int = Field(default=32,ge=8,le=128,description='Unsigned operand width; result includes carry-out.')
    architecture: Literal['native','kogge_stone','sklansky'] = Field(default='kogge_stone',description='Native addition or explicit parallel-prefix topology.')


class ConstantIn(ExplorationInput):
    backend: str | None = Field(default='arithmetic_rtl',description='Constant arithmetic graph generator backend.')
    width: int = Field(default=16,ge=8,le=64,description='Unsigned multiplicand width.')
    constant: int = Field(default=255,ge=1,le=65535,description='Positive compile-time multiplier.')
    architecture: Literal['native','binary','csd'] = Field(default='csd',description='Multiply operator, binary shift/add expansion, or canonical signed-digit expansion.')


class McmIn(ExplorationInput):
    backend: str | None = Field(default='arithmetic_rtl',description='Shared multiple-constant arithmetic graph backend.')
    width: int = Field(default=16,ge=8,le=64,description='Unsigned multiplicand width.')
    constants: list[int] = Field(default=[45,51,85,255],min_length=2,max_length=16,description='Distinct positive coefficients up to 65535, in packed output order.')
    architecture: Literal['independent','shared'] = Field(default='shared',description='Separate binary expansions or a memoized factored adder graph.')

    @model_validator(mode='after')
    def coefficients(self):
        if len(set(self.constants))!=len(self.constants) or any(c<1 or c>65535 for c in self.constants):
            raise ValueError('constants must be distinct integers in 1..65535')
        return self


def signed_digits(constant):
    result=[]
    bit=0
    while constant:
        digit=2-(constant%4) if constant&1 else 0
        if digit: result.append((bit,digit))
        constant=(constant-digit)//2
        bit+=1
    return result


def shifted_sum(rtl,wide,constant,width,csd):
    digits=signed_digits(constant) if csd else [(i,1) for i in range(constant.bit_length()) if constant&(1<<i)]
    value=f"{width}'d0"
    for bit,digit in reversed(digits):
        term=f'({wide} << {bit})'
        value=rtl.wire(width,f'{value} {"+" if digit>0 else "-"} {term}')
    return value


@backend('synth','arithmetic_rtl')
class Arithmetic:
    def synth_prefix_adder(self,p,ctx):
        width=p.width
        rtl=RTL(2*width+1,width+1)
        a=rtl.wire(width,f'x[{width-1}:0]')
        b=rtl.wire(width,f'x[{width} +: {width}]')
        cin=f'x[{2*width}]'
        if p.architecture=='native':
            result=f"{{1'b0,{a}}} + {{1'b0,{b}}} + {cin}"
        else:
            propagate=[rtl.wire(1,f'{a}[{i}] ^ {b}[{i}]') for i in range(width)]
            generate=[rtl.wire(1,f'{a}[{i}] & {b}[{i}]') for i in range(width)]
            original=propagate[:]
            distance=1
            while distance<width:
                next_p,next_g=propagate[:],generate[:]
                for i in range(width):
                    j=i-distance if p.architecture=='kogge_stone' else (i//(2*distance))*2*distance+distance-1
                    active=i>=distance if p.architecture=='kogge_stone' else bool(i&distance)
                    if active:
                        next_g[i]=rtl.wire(1,f'{generate[i]} | ({propagate[i]} & {generate[j]})')
                        next_p[i]=rtl.wire(1,f'{propagate[i]} & {propagate[j]}')
                propagate,generate=next_p,next_g
                distance*=2
            carries=[cin]+[rtl.wire(1,f'{generate[i]} | ({propagate[i]} & {cin})') for i in range(width)]
            sums=[rtl.wire(1,f'{original[i]} ^ {carries[i]}') for i in range(width)]
            result='{'+','.join([carries[-1]]+list(reversed(sums)))+'}'
        return emit(ctx,rtl,result,p.architecture,'Unsigned a+b+carry-in: a in low word, b in next word, carry-in in highest input bit; full-width sum.')

    def synth_csd_multiplier(self,p,ctx):
        width=p.width+p.constant.bit_length()
        rtl=RTL(p.width,width)
        wide=rtl.wire(width,'x')
        result=f"{wide} * {p.constant.bit_length()}'d{p.constant}" if p.architecture=='native' else shifted_sum(rtl,wide,p.constant,width,p.architecture=='csd')
        return emit(ctx,rtl,result,p.architecture,'Exact unsigned constant product; signed-digit intermediate subtraction is modulo a width large enough for the full final product.')

    def synth_mcm(self,p,ctx):
        width=p.width+max(p.constants).bit_length()
        rtl=RTL(p.width,width*len(p.constants))
        wide=rtl.wire(width,'x')
        memo={1:wide}
        def build(coefficient):
            if coefficient in memo: return memo[coefficient]
            # Powers of two are wiring; factors of 2^k +/- 1 need one adder.
            if coefficient&(coefficient-1)==0:
                value=rtl.wire(width,f'{wide} << {coefficient.bit_length()-1}')
            else:
                options=[]
                for shift in range(1,coefficient.bit_length()+1):
                    for sign in (-1,1):
                        factor=(1<<shift)+sign
                        if factor>1 and coefficient%factor==0:
                            part=coefficient//factor
                            cost=0 if part in memo else max(0,len(signed_digits(part))-1)
                            options.append((cost,part,shift,sign))
                if options:
                    _,part,shift,sign=min(options)
                    node=build(part)
                    value=rtl.wire(width,f'({node} << {shift}) {"+" if sign>0 else "-"} {node}')
                else:
                    value=shifted_sum(rtl,wide,coefficient,width,True)
            memo[coefficient]=value
            return value
        results=[build(c) if p.architecture=='shared' else shifted_sum(rtl,wide,c,width,False) for c in p.constants]
        return emit(ctx,rtl,'{'+','.join(reversed(results))+'}',p.architecture,'Exact unsigned products packed coefficient-first in low output words; shared graph uses memoized shift/add factorization, not a globally optimal solver.')


CATEGORY.ops.append(Op('synth_prefix_adder',PrefixIn,Result,'Generate native, Kogge-Stone or Sklansky unsigned addition networks.'))
CATEGORY.ops.append(Op('synth_csd_multiplier',ConstantIn,Result,'Generate exact native, binary or canonical signed-digit constant multiplication.'))
CATEGORY.ops.append(Op('synth_mcm',McmIn,Result,'Generate independently expanded or shared factored multiple-constant products.'))
