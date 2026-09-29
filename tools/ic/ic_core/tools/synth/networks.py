"""Structured selection, shift and count networks with exact total semantics."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class NetworkIn(ExplorationInput):
    backend: str | None = Field(default='network_rtl', description='Structured network generator backend.')
    width: int = Field(default=64, ge=8, le=1024, description='Input word width; must be a power of two.')
    architecture: Literal['linear','tree'] = Field(default='tree', description='Natural serial reference or hierarchical candidate.')

    @model_validator(mode='after')
    def power_of_two(self):
        if self.width & (self.width-1):
            raise ValueError('width must be a power of two')
        return self


class LeadingIn(NetworkIn):
    architecture: Literal['linear','tree','binary_search'] = Field(default='tree',description='Serial priority, recursive counts, or binary-search narrowing network.')


class LaneIn(NetworkIn):
    width: int = Field(default=16, ge=8, le=64, description='Unsigned lane word width; power of two.')
    lanes: int = Field(default=16, ge=2, le=64, description='Number of lanes; power of two.')

    @model_validator(mode='after')
    def lane_power_of_two(self):
        if self.lanes & (self.lanes-1):
            raise ValueError('lanes must be a power of two')
        return self


class RTL:
    def __init__(self, inputs, outputs):
        self.inputs,self.outputs=inputs,outputs
        self.lines=[f'module dut(input [{inputs-1}:0] x, output [{outputs-1}:0] y);']
        self.index=0

    def wire(self, width, expression):
        name=f'n{self.index}'
        self.index+=1
        self.lines.append(f'wire [{width-1}:0] {name} = {expression};')
        return name

    def finish(self, expression):
        return '\n'.join(self.lines+[f'assign y={expression};','endmodule'])+'\n'


def emit(ctx, rtl, expression, architecture, semantics):
    core=ctx.run.artifacts/'core.sv'
    core.write_text(rtl.finish(expression),encoding='utf-8')
    wrapper=ctx.run.artifacts/'registered.sv'
    wrapper.write_text(f'''module registered_dut(input clk, input [{rtl.inputs-1}:0] x, output reg [{rtl.outputs-1}:0] y);
reg [{rtl.inputs-1}:0] x_q;
wire [{rtl.outputs-1}:0] result;
dut core(.x(x_q),.y(result));
always @(posedge clk) begin x_q<=x; y<=result; end
endmodule
''',encoding='utf-8')
    return Result(data=dict(core=ctx.run.handle(core.name),wrapper=ctx.run.handle(wrapper.name),
                           core_path=str(ctx.run.final_artifact(core.name)),wrapper_path=str(ctx.run.final_artifact(wrapper.name)),
                           source_sha256=fingerprint([core,wrapper],'registered_dut'), input_width=rtl.inputs,output_width=rtl.outputs,latency_cycles=1,initiation_interval=1,architecture=architecture),
                  note=semantics+' Registered boundaries are identical. PPA benefit requires measurement.')


@backend('synth','network_rtl')
class Networks:
    def synth_popcount(self,p,ctx):
        result_width=p.width.bit_length()
        rtl=RTL(p.width,result_width)
        if p.architecture=='linear':
            count=f"{result_width}'d0"
            for i in range(p.width):
                count=rtl.wire(result_width,f'{count} + x[{i}]')
        else:
            terms=[(f'x[{i}]',1) for i in range(p.width)]
            while len(terms)>1:
                terms=[(rtl.wire(a[1]+1,f"{{1'b0,{a[0]}}} + {{1'b0,{b[0]}}}"),a[1]+1) for a,b in zip(terms[::2],terms[1::2])]
            count=terms[0][0]
        return emit(ctx,rtl,count,p.architecture,'Exact population count of all input bits.')

    def synth_priority_encoder(self,p,ctx):
        bits=(p.width-1).bit_length()
        rtl=RTL(p.width,bits+1)
        if p.architecture=='linear':
            result=f"{bits+1}'d0"
            for i in range(p.width):
                result=rtl.wire(bits+1,f"x[{i}] ? {bits+1}'d{(1<<bits)|i} : {result}")
        else:
            terms=[(f'x[{i}]',f"{bits}'d{i}") for i in range(p.width)]
            while len(terms)>1:
                next_terms=[]
                for lo,hi in zip(terms[::2],terms[1::2]):
                    valid=rtl.wire(1,f'{lo[0]} | {hi[0]}')
                    index=rtl.wire(bits,f'{hi[0]} ? {hi[1]} : {lo[1]}')
                    next_terms.append((valid,index))
                terms=next_terms
            valid,index=terms[0]
            result=f"{{{valid},({valid} ? {index} : {bits}'d0)}}"
        return emit(ctx,rtl,result,p.architecture,'Highest asserted index with valid in the MSB; zero input returns zero.')

    def synth_leading_zero(self,p,ctx):
        bits=p.width.bit_length()
        rtl=RTL(p.width,bits)
        if p.architecture=='linear':
            result=f"{bits}'d{p.width}"
            for i in range(p.width):
                result=rtl.wire(bits,f"x[{i}] ? {bits}'d{p.width-1-i} : {result}")
        elif p.architecture=='binary_search':
            segment='x'
            size=p.width
            decisions=[]
            while size>1:
                half=size//2
                empty=rtl.wire(1,f'~|{segment}[{size-1}:{half}]')
                decisions.append(empty)
                segment=rtl.wire(half,f'{empty} ? {segment}[{half-1}:0] : {segment}[{size-1}:{half}]')
                size=half
            encoded="{1'b0,"+','.join(decisions)+'}'
            result=f"(~|x) ? {bits}'d{p.width} : {encoded}"
        else:
            def count(start,size):
                if size==1:
                    return rtl.wire(1,f'~x[{start}]')
                half=size//2
                high=count(start+half,half)
                low=count(start,half)
                high_zero=rtl.wire(1,f'~|x[{start+half} +: {half}]')
                return rtl.wire(size.bit_length(),f"{high_zero} ? ({size.bit_length()}'d{half} + {low}) : {high}")
            result=count(0,p.width)
        return emit(ctx,rtl,result,p.architecture,'Leading-zero count, returning word width for zero.')

    def synth_barrel_shifter(self,p,ctx):
        bits=(p.width-1).bit_length()
        rtl=RTL(p.width+bits,p.width)
        amount=rtl.wire(bits,f'x[{p.width} +: {bits}]')
        data=rtl.wire(p.width,f'x[{p.width-1}:0]')
        if p.architecture=='linear':
            # A full decoded case is a common mux-based shift implementation.
            rtl.lines += [f'reg [{p.width-1}:0] shifted;', 'always @* begin', 'case ('+amount+')']
            rtl.lines += [f"{bits}'d{i}: shifted={data} << {i};" for i in range(p.width)]
            rtl.lines += ["default: shifted='0;",'endcase','end']
            result='shifted'
        else:
            result=data
            for bit in range(bits):
                result=rtl.wire(p.width,f'{amount}[{bit}] ? ({result} << {1<<bit}) : {result}')
        return emit(ctx,rtl,result,p.architecture,'Logical left shift modulo word width; high input bits encode shift amount.')

    def synth_onehot_mux(self,p,ctx):
        rtl=RTL(p.width*p.lanes+p.lanes,p.width)
        terms=[rtl.wire(p.width,f"x[{i*p.width} +: {p.width}] & {{{p.width}{{x[{p.width*p.lanes+i}]}}}}") for i in range(p.lanes)]
        if p.architecture=='linear':
            result=terms[0]
            for term in terms[1:]:
                result=rtl.wire(p.width,f'{result} | {term}')
        else:
            while len(terms)>1:
                terms=[rtl.wire(p.width,f'{a} | {b}') for a,b in zip(terms[::2],terms[1::2])]
            result=terms[0]
        return emit(ctx,rtl,result,p.architecture,'OR of selected lanes; one-hot selects one word, multi-hot returns bitwise OR, zero-hot returns zero.')

    def synth_argmax_tree(self,p,ctx):
        bits=(p.lanes-1).bit_length()
        rtl=RTL(p.width*p.lanes,p.width+bits)
        terms=[(f'x[{i*p.width} +: {p.width}]',f"{bits}'d{i}") for i in range(p.lanes)]
        def combine(a,b):
            left=rtl.wire(1,f'{a[0]} >= {b[0]}')
            value=rtl.wire(p.width,f'{left} ? {a[0]} : {b[0]}')
            index=rtl.wire(bits,f'{left} ? {a[1]} : {b[1]}')
            return value,index
        if p.architecture=='linear':
            result=terms[0]
            for term in terms[1:]:
                result=combine(result,term)
        else:
            while len(terms)>1:
                terms=[combine(a,b) for a,b in zip(terms[::2],terms[1::2])]
            result=terms[0]
        return emit(ctx,rtl,'{'+','.join(result)+'}',p.architecture,'Unsigned maximum followed by index in low bits; equal maxima choose lowest lane index.')


for name,model,summary in [
    ('synth_popcount',NetworkIn,'Generate exact serial or width-aware tree population counts.'),
    ('synth_priority_encoder',NetworkIn,'Generate highest-priority valid/index selection networks.'),
    ('synth_leading_zero',LeadingIn,'Generate exact hierarchical leading-zero counters.'),
    ('synth_barrel_shifter',NetworkIn,'Generate decoded or logarithmic staged logical-shift networks.'),
    ('synth_onehot_mux',LaneIn,'Generate total-semantics masked lane-selection networks.'),
    ('synth_argmax_tree',LaneIn,'Generate stable unsigned argmax comparison tournaments.')]:
    CATEGORY.ops.append(Op(name,model,Result,summary))
