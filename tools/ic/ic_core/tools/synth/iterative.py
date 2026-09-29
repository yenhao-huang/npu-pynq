"""Ready/valid exact unsigned multiply and restoring divide architectures."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class IterativeIn(ExplorationInput):
    backend: str | None = Field(default='iterative_rtl',description='Multi-cycle arithmetic generator backend.')
    width: int = Field(default=16,ge=8,le=64,description='Unsigned operand width; output contains two words.')
    architecture: Literal['parallel','serial','radix4'] = Field(default='serial',description='Direct combinational result register, one bit per working cycle, or two bit steps per working cycle.')

    @model_validator(mode='after')
    def radix_width(self):
        if self.architecture=='radix4' and self.width%2: raise ValueError('radix4 requires an even operand width')
        return self


def generate(p,operation,ctx):
    w=p.width;step=2 if p.architecture=='radix4' else 1
    iterations=w//step;latency=1 if p.architecture=='parallel' else iterations+1
    lines=[f'''(* use_dsp = "no" *) module arithmetic_dut(input clk,input rst,
input valid_in,output ready_in,input [{2*w-1}:0] data_in,
output valid_out,input ready_out,output reg [{2*w-1}:0] data_out);
reg busy,valid_q;
assign valid_out=!rst && valid_q;
assign ready_in=!rst && !busy && (!valid_q || ready_out);
wire [{w-1}:0] a=data_in[{w-1}:0];
wire [{w-1}:0] b=data_in[{w} +: {w}];''']
    if p.architecture=='parallel':
        expression='a*b' if operation=='multiply' else f"(b==0) ? {{a,{w}'d{(1<<w)-1}}} : {{(a % b),(a / b)}}"
        body=f'if(ready_in && valid_in) begin data_out<={expression};valid_q<=1;end'
        reset=''
    else:
        counter_width=max(1,(iterations-1).bit_length())
        lines.append(f'reg [{counter_width-1}:0] remaining;')
        if operation=='multiply':
            lines += [f'reg [{2*w-1}:0] accumulator,multiplicand;',f'reg [{w-1}:0] multiplier;']
            acc,mc,mp='accumulator','multiplicand','multiplier'
            for index in range(step):
                lines += [f'wire [{2*w-1}:0] acc{index}={mp}[0] ? ({acc}+{mc}) : {acc};',
                          f'wire [{2*w-1}:0] mc{index}={mc} << 1;',f'wire [{w-1}:0] mp{index}={mp} >> 1;']
                acc,mc,mp=f'acc{index}',f'mc{index}',f'mp{index}'
            start='accumulator<=0;multiplicand<=a;multiplier<=b;'
            update=f'accumulator<={acc};multiplicand<={mc};multiplier<={mp};'
            result=acc
            reset='accumulator<=0;multiplicand<=0;multiplier<=0;'
        else:
            lines += [f'reg [{w-1}:0] dividend,divisor,quotient;',f'reg [{w}:0] remainder;']
            rem,dvd,quo='remainder','dividend','quotient'
            for index in range(step):
                lines += [f"wire [{w}:0] trial{index}={{{rem}[{w-1}:0],{dvd}[{w-1}]}};",
                          f"wire take{index}=trial{index}>={{1'b0,divisor}};",
                          f'wire [{w}:0] rem{index}=take{index} ? (trial{index}-divisor) : trial{index};',
                          f"wire [{w-1}:0] dvd{index}={{{dvd}[{w-2}:0],1'b0}};",
                          f'wire [{w-1}:0] quo{index}={{{quo}[{w-2}:0],take{index}}};']
                rem,dvd,quo=f'rem{index}',f'dvd{index}',f'quo{index}'
            start='dividend<=a;divisor<=b;quotient<=0;remainder<=0;'
            update=f'dividend<={dvd};quotient<={quo};remainder<={rem};'
            result=f'{{{rem}[{w-1}:0],{quo}}}'
            reset='dividend<=0;divisor<=0;quotient<=0;remainder<=0;'
        body=f'''if(ready_in && valid_in) begin
  {start} remaining<={counter_width}'d{iterations-1};busy<=1;
end else if(busy) begin
  {update}
  if(remaining==0) begin data_out<={result};valid_q<=1;busy<=0;end
  else remaining<=remaining-1'b1;
end'''
        reset+='remaining<=0;'
    lines.append(f'''always @(posedge clk) begin
  if(rst) begin busy<=0;valid_q<=0;data_out<=0;{reset}end
  else begin
    if(valid_q && ready_out) valid_q<=0;
    {body}
  end
end
endmodule
''')
    core=ctx.run.artifacts/'arithmetic.sv';core.write_text('\n'.join(lines),encoding='utf-8')
    fixture=ctx.run.artifacts/'arithmetic_timing.sv'
    fixture.write_text(f'''module arithmetic_timing(input clk,input rst,
input valid_in,output reg ready_in,input [{2*w-1}:0] data_in,
output reg valid_out,input ready_out,output reg [{2*w-1}:0] data_out);
reg rst_q,valid_q,ready_q;
reg [{2*w-1}:0] data_q;
wire core_ready,core_valid;
wire [{2*w-1}:0] core_data;
arithmetic_dut core(.clk(clk),.rst(rst_q),.valid_in(valid_q),.ready_in(core_ready),.data_in(data_q),.valid_out(core_valid),.ready_out(ready_q),.data_out(core_data));
always @(posedge clk) begin
rst_q<=rst;valid_q<=valid_in;ready_q<=ready_out;data_q<=data_in;
ready_in<=core_ready;valid_out<=core_valid;data_out<=core_data;
end
endmodule
''',encoding='utf-8')
    return Result(data=dict(files=[str(ctx.run.final_artifact(core.name))],top='arithmetic_dut',source_sha256=fingerprint([core],'arithmetic_dut'),
        timing_files=[str(ctx.run.final_artifact(core.name)),str(ctx.run.final_artifact(fixture.name))],timing_top='arithmetic_timing',timing_sha256=fingerprint([core,fixture],'arithmetic_timing'),
        width=w,operation=operation,architecture=p.architecture,latency_cycles=latency,initiation_interval=latency,fixture_observation_delay_cycles=2),
        note='Exact unsigned arithmetic with synchronous reset cancelling in-flight work. Inputs pack a then b. Multiply returns the full product; divide packs quotient low and remainder high, with divide-by-zero quotient all ones and remainder a. Output holds under backpressure. Fixture is a delayed observation environment, not an external handshake adapter.')


@backend('synth','iterative_rtl')
class Iterative:
    def synth_serial_multiplier(self,p,ctx): return generate(p,'multiply',ctx)
    def synth_divider(self,p,ctx): return generate(p,'divide',ctx)


CATEGORY.ops.append(Op('synth_serial_multiplier',IterativeIn,Result,'Generate full-product ready/valid multipliers with parallel or iterative resource sharing.'))
CATEGORY.ops.append(Op('synth_divider',IterativeIn,Result,'Generate exact quotient/remainder dividers with parallel or one/two-bit restoring iterations.'))
