"""Signed FIR-window and vector-product datapaths with exact accumulation."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from . import CATEGORY
from .exploration import Result
from .networks import RTL, emit


class FIRIn(ExplorationInput):
    backend: str | None = Field(default='datapath_rtl',description='Signed arithmetic datapath backend.')
    width: int = Field(default=16,ge=8,le=64,description='Signed sample width; packed oldest-to-newest in increasing lane order.')
    coefficients: list[int] = Field(default=[3,5,7,11,11,7,5,3],min_length=3,max_length=32,description='Signed constant coefficients, each within -32767..32767.')
    architecture: Literal['direct','symmetric','direct_serial'] = Field(default='symmetric',description='Balanced independent products, symmetric preaddition, or serial independent products for ablation.')
    multiplier_mapping: Literal['auto','logic'] = Field(default='logic',description='Allow synthesis inference or prohibit DSP use; match policy between compared designs.')

    @model_validator(mode='after')
    def valid_coefficients(self):
        if not any(self.coefficients) or any(abs(c)>32767 for c in self.coefficients):
            raise ValueError('coefficients must include a nonzero value and lie within -32767..32767')
        if self.architecture=='symmetric' and self.coefficients!=self.coefficients[::-1]:
            raise ValueError('symmetric architecture requires palindromic coefficients')
        return self


class DotIn(ExplorationInput):
    backend: str | None = Field(default='datapath_rtl',description='Signed arithmetic datapath backend.')
    width: int = Field(default=16,ge=8,le=32,description='Signed multiplicand width.')
    lanes: int = Field(default=8,ge=2,le=32,description='Number of packed signed operand pairs.')
    architecture: Literal['serial','balanced','compressor'] = Field(default='compressor',description='Product reduction with serial carry propagation, a balanced tree, or 3:2 carry-save stages.')
    multiplier_mapping: Literal['auto','logic'] = Field(default='logic',description='Allow DSP inference or request logic mapping; match policy within comparisons.')


class SaturatingIn(ExplorationInput):
    backend: str | None = Field(default='datapath_rtl',description='Saturating arithmetic datapath backend.')
    width: int = Field(default=32,ge=8,le=128,description='Operand and saturated result width.')
    architecture: Literal['dual','shared'] = Field(default='shared',description='Independent add/subtract paths or a shared conditional-inversion adder.')
    signed: bool = Field(default=True,description='Signed two-complement saturation or unsigned clamping.')


def extend(rtl,value,width,output_width):
    if output_width==width: return value
    return rtl.wire(output_width,f'{{{{{output_width-width}{{{value}[{width-1}]}}}},{value}}}')


def sum_terms(rtl,terms,width,architecture):
    if architecture=='serial':
        total=terms[0]
        for term in terms[1:]: total=rtl.wire(width,f'{total} + {term}')
        return total
    while len(terms)>2:
        step=3 if architecture=='compressor' else 2
        following=[]
        for start in range(0,len(terms),step):
            group=terms[start:start+step]
            if len(group)<step: following.extend(group)
            elif step==2: following.append(rtl.wire(width,' + '.join(group)))
            else:
                a,b,c=group
                following.extend([rtl.wire(width,f'{a} ^ {b} ^ {c}'),rtl.wire(width,f'(({a} & {b}) | ({a} & {c}) | ({b} & {c})) << 1')])
        terms=following
    return rtl.wire(width,' + '.join(terms)) if len(terms)>1 else terms[0]


def mapping_policy(rtl,policy):
    if policy=='logic': rtl.lines[0]='(* use_dsp = "no" *) '+rtl.lines[0]


@backend('synth','datapath_rtl')
class Datapaths:
    def synth_saturating_alu(self,p,ctx):
        width=p.width;rtl=RTL(2*width+1,width+1)
        a=rtl.wire(width,f'x[{width-1}:0]')
        b=rtl.wire(width,f'x[{width} +: {width}]')
        sub=f'x[{2*width}]'
        if p.architecture=='dual':
            a_ext=extend(rtl,a,width,width+1) if p.signed else rtl.wire(width+1,a)
            b_ext=extend(rtl,b,width,width+1) if p.signed else rtl.wire(width+1,b)
            plus=rtl.wire(width+1,f'{a_ext} + {b_ext}')
            minus=rtl.wire(width+1,f'{a_ext} - {b_ext}')
            total=rtl.wire(width+1,f'{sub} ? {minus} : {plus}')
            if p.signed:
                high=rtl.wire(1,f"$signed({total}) > {width+1}'sd{(1<<(width-1))-1}")
                low=rtl.wire(1,f"$signed({total}) < (-{width+1}'sd{1<<(width-1)})")
                overflow=rtl.wire(1,f'{high} | {low}')
            else:
                overflow=f'{total}[{width}]'
            raw=f'{total}[{width-1}:0]'
        else:
            inverted=rtl.wire(width,f'{b} ^ {{{width}{{{sub}}}}}')
            total=rtl.wire(width+1,f"{{1'b0,{a}}} + {{1'b0,{inverted}}} + {sub}")
            raw=rtl.wire(width,f'{total}[{width-1}:0]')
            overflow=rtl.wire(1,f'({a}[{width-1}] ^ {raw}[{width-1}]) & ~({a}[{width-1}] ^ {b}[{width-1}] ^ {sub})' if p.signed else f'{total}[{width}] ^ {sub}')
        limit=f"{a}[{width-1}] ? {width}'d{1<<(width-1)} : {width}'d{(1<<(width-1))-1}" if p.signed else f"{sub} ? {width}'d0 : {width}'d{(1<<width)-1}"
        saturated=rtl.wire(width,f'{overflow} ? ({limit}) : {raw}')
        output=emit(ctx,rtl,'{'+overflow+','+saturated+'}',p.architecture,'Packed input a, b, subtract flag. Low output word is saturated result; high bit indicates mathematical overflow/underflow. Both addition and subtraction are supported.')
        output.data.update(signed=p.signed,operand_width=width)
        return output

    def synth_fir(self,p,ctx):
        taps=len(p.coefficients)
        output_width=p.width+sum(abs(c) for c in p.coefficients).bit_length()
        rtl=RTL(p.width*taps,output_width);mapping_policy(rtl,p.multiplier_mapping)
        samples=[rtl.wire(p.width,f'x[{i*p.width} +: {p.width}]') for i in range(taps)]
        terms=[];products=0
        indices=range((taps+1)//2) if p.architecture=='symmetric' else range(taps)
        for index in indices:
            coefficient=p.coefficients[index]
            if coefficient==0: continue
            sample=samples[index];sample_width=p.width
            if p.architecture=='symmetric' and index!=taps-1-index:
                left=extend(rtl,sample,p.width,p.width+1)
                right=extend(rtl,samples[taps-1-index],p.width,p.width+1)
                sample=rtl.wire(p.width+1,f'{left} + {right}');sample_width+=1
            sample=extend(rtl,sample,sample_width,output_width)
            literal=f"{output_width}'sd{abs(coefficient)}"
            if coefficient<0: literal=f'(-{literal})'
            terms.append(rtl.wire(output_width,f'$signed({sample}) * {literal}'));products+=1
        result=sum_terms(rtl,terms,output_width,'serial' if p.architecture=='direct_serial' else 'balanced')
        output=emit(ctx,rtl,result,p.architecture,'Exact signed FIR-window dot sum; sample history is supplied externally. This is the arithmetic kernel, not a streaming delay-line/protocol implementation. No rounding or saturation.')
        output.data.update(taps=taps,coefficients=p.coefficients,signed=True,multiplier_nodes=products,multiplier_mapping=p.multiplier_mapping)
        return output

    def synth_dot_product(self,p,ctx):
        product_width=2*p.width
        output_width=product_width+(p.lanes-1).bit_length()
        rtl=RTL(2*p.width*p.lanes,output_width);mapping_policy(rtl,p.multiplier_mapping)
        terms=[]
        for lane in range(p.lanes):
            a=rtl.wire(p.width,f'x[{2*lane*p.width} +: {p.width}]')
            b=rtl.wire(p.width,f'x[{(2*lane+1)*p.width} +: {p.width}]')
            product=rtl.wire(product_width,f'$signed({a}) * $signed({b})')
            terms.append(extend(rtl,product,product_width,output_width))
        result=sum_terms(rtl,terms,output_width,p.architecture)
        output=emit(ctx,rtl,result,p.architecture,'Exact signed vector dot product. Lane i packs a_i then b_i in increasing words; full products are sign extended before accumulation. No rounding, truncation or saturation.')
        output.data.update(lanes=p.lanes,signed=True,multiplier_mapping=p.multiplier_mapping)
        return output


CATEGORY.ops.append(Op('synth_fir',FIRIn,Result,'Generate signed FIR-window datapaths with verified coefficient symmetry and full-precision preaddition.'))
CATEGORY.ops.append(Op('synth_dot_product',DotIn,Result,'Generate full-precision signed vector products with balanced or carry-save accumulation.'))
CATEGORY.ops.append(Op('synth_saturating_alu',SaturatingIn,Result,'Generate signed or unsigned saturating add/subtract with explicit overflow and shared arithmetic.'))
