"""Signed full-product multiplication with explicit Booth partial products."""
from typing import Literal
from pydantic import Field
from ...registry import Op, backend
from ..exploration_common import ExplorationInput
from . import CATEGORY
from .exploration import Result
from .networks import RTL, emit


class BoothIn(ExplorationInput):
    backend: str | None = Field(default='booth_rtl',description='Signed Booth recoding backend.')
    width: int = Field(default=16,ge=8,le=64,description='Signed operand width, including odd widths.')
    architecture: Literal['native','radix2','radix4'] = Field(default='radix4',description='Native signed product, adjacent-bit Booth rows, or paired Booth rows.')


@backend('synth','booth_rtl')
class Booth:
    def synth_booth_multiplier(self,p,ctx):
        w=p.width;bits=2*w
        rtl=RTL(bits,bits)
        rtl.lines[0]='(* use_dsp = "no" *) '+rtl.lines[0]
        rtl.lines.extend([f'wire signed [{w-1}:0] a=x[{w-1}:0];',f'wire signed [{w-1}:0] b=x[{w} +: {w}];'])
        if p.architecture=='native':
            product=rtl.wire(bits,'$signed(a) * $signed(b)')
        else:
            extended=rtl.wire(bits,f'{{{{{w}{{a[{w-1}]}}}},a}}')
            def bit(index): return "1'b0" if index<0 else f'b[{min(index,w-1)}]'
            terms=[];step=1 if p.architecture=='radix2' else 2
            for index in range(0,w,step):
                if step==1:
                    selector=f'{{{bit(index)},{bit(index-1)}}}'
                    term=rtl.wire(bits,f"({selector}==2'b01) ? {extended} : (({selector}==2'b10) ? -{extended} : {bits}'d0)")
                else:
                    selector=rtl.wire(3,f'{{{bit(index+1)},{bit(index)},{bit(index-1)}}}')
                    term=rtl.wire(bits,f"({selector}==3'b001 || {selector}==3'b010) ? {extended} : ({selector}==3'b011) ? ({extended} << 1) : ({selector}==3'b100) ? -({extended} << 1) : ({selector}==3'b101 || {selector}==3'b110) ? -{extended} : {bits}'d0")
                terms.append(rtl.wire(bits,f'{term} << {index}'))
            while len(terms)>1:
                terms=[rtl.wire(bits,f'{terms[i]} + {terms[i+1]}') if i+1<len(terms) else terms[i] for i in range(0,len(terms),2)]
            product=terms[0]
        return emit(ctx,rtl,product,p.architecture,'Exact signed two\'s-complement product. Packed input a low, b high; full 2*width-bit result. Operands extend before negation, including the most-negative value. Logic-only mapping request; no DSP or PPA improvement assumed.')


CATEGORY.ops.append(Op('synth_booth_multiplier',BoothIn,Result,'Generate signed full-product Booth partial-product architectures with exact extrema semantics.'))
