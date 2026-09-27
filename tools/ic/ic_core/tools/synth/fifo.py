"""Streaming FIFO alternatives with identical transaction semantics."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class FifoIn(ExplorationInput):
    backend: str | None = Field(default='fifo_rtl',description='Streaming FIFO generator backend.')
    width: int = Field(default=16,ge=8,le=64,description='Transaction data width.')
    depth: int = Field(default=64,ge=4,le=256,description='FIFO capacity; power of two.')
    architecture: Literal['shift','circular'] = Field(default='circular',description='Shift-on-pop storage or circular pointer storage.')

    memory_style: Literal['auto','distributed','block'] = Field(default='auto',description='Circular-memory inference policy; shift architecture requires auto.')

    @model_validator(mode='after')
    def capacity(self):
        if self.architecture=='shift' and self.memory_style!='auto': raise ValueError('shift storage does not accept a RAM inference policy')
        if self.depth & (self.depth-1): raise ValueError('depth must be a power of two')
        return self


@backend('synth','fifo_rtl')
class FIFO:
    def synth_fifo(self,p,ctx):
        bits=(p.depth-1).bit_length()
        attribute='' if p.memory_style=='auto' else f'(* ram_style = "{p.memory_style}" *) '
        lines=[f'''module fifo_dut(input clk, input rst,
input valid_in, output ready_in, input [{p.width-1}:0] data_in,
output valid_out, input ready_out, output [{p.width-1}:0] data_out);
reg [{bits}:0] count;
{attribute}reg [{p.width-1}:0] mem [0:{p.depth-1}];
assign valid_out=!rst && (count!=0);
wire pop=valid_out && ready_out;
assign ready_in=!rst && ((count < {bits+1}'d{p.depth}) || pop);
wire push=valid_in && ready_in;
always @(posedge clk) begin
  if(rst) count<=0;
  else case ({{push,pop}})
    2'b10: count<=count+1'b1;
    2'b01: count<=count-1'b1;
    default: count<=count;
  endcase
end
''']
        if p.architecture=='shift':
            lines += ['assign data_out=mem[0];','integer i;','always @(posedge clk) begin',
                      '  if(!rst) begin',f'    if(pop) for(i=0;i<{p.depth-1};i=i+1) mem[i]<=mem[i+1];',
                      "    if(push) mem[count-pop]<=data_in;",'  end','end']
        else:
            lines += [f'reg [{bits-1}:0] read_ptr,write_ptr;',
                      'assign data_out=mem[read_ptr];','always @(posedge clk) begin',
                      '  if(rst) begin read_ptr<=0; write_ptr<=0; end',
                      '  else begin',
                      '    if(push) begin mem[write_ptr]<=data_in; write_ptr<=write_ptr+1\'b1; end',
                      "    if(pop) read_ptr<=read_ptr+1'b1;",'  end','end']
        lines+=['endmodule']
        path=ctx.run.artifacts/'fifo.sv'
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        timing=ctx.run.artifacts/'fifo_timing.sv'
        timing.write_text(f"""module fifo_timing(input clk, input rst,
input valid_in, output reg ready_in, input [{p.width-1}:0] data_in,
output reg valid_out, input ready_out, output reg [{p.width-1}:0] data_out);
reg rst_q,valid_q,ready_q;
reg [{p.width-1}:0] data_q;
wire core_ready,core_valid;
wire [{p.width-1}:0] core_data;
fifo_dut core(.clk(clk),.rst(rst_q),.valid_in(valid_q),.ready_in(core_ready),.data_in(data_q),.valid_out(core_valid),.ready_out(ready_q),.data_out(core_data));
always @(posedge clk) begin
  rst_q<=rst; valid_q<=valid_in; ready_q<=ready_out; data_q<=data_in;
  ready_in<=core_ready; valid_out<=core_valid; data_out<=core_data;
end
endmodule
""",encoding='utf-8')
        return Result(data=dict(timing_files=[str(ctx.run.final_artifact(path.name)),str(ctx.run.final_artifact(timing.name))],
                               timing_top='fifo_timing',timing_sha256=fingerprint([path,timing],'fifo_timing'),fixture_observation_delay_cycles=2,
                               files=[str(ctx.run.final_artifact(path.name))],source=ctx.run.handle(path.name),source_sha256=fingerprint([path],'fifo_dut'),top='fifo_dut',width=p.width,depth=p.depth,minimum_latency_cycles=1,latency_kind='occupancy_and_backpressure_dependent',initiation_interval=1,architecture=p.architecture,memory_style=p.memory_style),note='Synchronous active-high reset flushes occupancy. No fall-through. Full FIFO may replace a popped word in the same cycle. Invalid output data is unspecified. Storage bits are not reset.')


CATEGORY.ops.append(Op('synth_fifo',FifoIn,Result,'Generate shift or circular FIFO storage with full-throughput simultaneous push/pop.'))
