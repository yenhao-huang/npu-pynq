"""Elastic chains and two-slot stages that cut combinational ready propagation."""
from typing import Literal
from pydantic import Field
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class SkidIn(ExplorationInput):
    backend: str | None = Field(default='skid_rtl',description='Backpressure pipeline generator.')
    width: int = Field(default=32,ge=8,le=64,description='Payload word width.')
    stages: int = Field(default=8,ge=2,le=32,description='Number of registered forwarding stages.')
    architecture: Literal['elastic','skid'] = Field(default='skid',description='One-slot elastic stages or two-slot stages with readiness derived only from local registered occupancy.')


@backend('synth','skid_rtl')
class Skid:
    def synth_skid_buffer(self,p,ctx):
        w,s=p.width,p.stages
        ports=f'input clk,input rst,input valid_in,output ready_in,input [{w-1}:0] data_in,output valid_out,input ready_out,output [{w-1}:0] data_out'
        if p.architecture=='elastic':
            body=f'''reg occupied;
reg [{w-1}:0] payload;
assign ready_in=!rst && (!occupied || ready_out);
assign valid_out=!rst && occupied;
assign data_out=payload;
always @(posedge clk) begin
if(rst) begin occupied<=0;payload<=0;end
else if(ready_in) begin occupied<=valid_in;if(valid_in) payload<=data_in;end
end'''
        else:
            body=f'''reg [1:0] count;
reg [{w-1}:0] head,spill;
wire push=valid_in && ready_in;
wire pop=valid_out && ready_out;
assign ready_in=!rst && (count<2);
assign valid_out=!rst && (count!=0);
assign data_out=head;
always @(posedge clk) begin
if(rst) begin count<=0;head<=0;spill<=0;end
else begin
case({{push,pop}})
2'b10: begin if(count==0) head<=data_in;else spill<=data_in;count<=count+1'b1;end
2'b01: begin if(count==2) head<=spill;count<=count-1'b1;end
2'b11: head<=data_in;
default: begin end
endcase
end
end'''
        core=ctx.run.artifacts/'stream.sv'
        lines=[f'module stream_stage({ports});\n{body}\nendmodule',f'module stream_dut({ports});',f'wire [{s}:0] v,r;',f'wire [{w-1}:0] d[0:{s}];',
            'assign v[0]=valid_in;assign d[0]=data_in;assign ready_in=r[0];',f'assign valid_out=v[{s}];assign data_out=d[{s}];assign r[{s}]=ready_out;']
        for i in range(s):
            lines.append(f'stream_stage stage{i}(.clk(clk),.rst(rst),.valid_in(v[{i}]),.ready_in(r[{i}]),.data_in(d[{i}]),.valid_out(v[{i+1}]),.ready_out(r[{i+1}]),.data_out(d[{i+1}]));')
        core.write_text('\n'.join(lines+['endmodule\n']))
        fixture=ctx.run.artifacts/'stream_timing.sv'
        fixture.write_text(f'''module stream_timing(input clk,input rst,input valid_in,output reg ready_in,input [{w-1}:0] data_in,output reg valid_out,input ready_out,output reg [{w-1}:0] data_out);
reg rst_q,valid_q,ready_q;
reg [{w-1}:0] data_q;
wire core_ready,core_valid;
wire [{w-1}:0] core_data;
stream_dut core(.clk(clk),.rst(rst_q),.valid_in(valid_q),.ready_in(core_ready),.data_in(data_q),.valid_out(core_valid),.ready_out(ready_q),.data_out(core_data));
always @(posedge clk) begin
rst_q<=rst;valid_q<=valid_in;ready_q<=ready_out;data_q<=data_in;
ready_in<=core_ready;valid_out<=core_valid;data_out<=core_data;
end
endmodule
''')
        return Result(data=dict(files=[str(ctx.run.final_artifact(core.name))],top='stream_dut',source_sha256=fingerprint([core],'stream_dut'),
            timing_files=[str(ctx.run.final_artifact(core.name)),str(ctx.run.final_artifact(fixture.name))],timing_top='stream_timing',timing_sha256=fingerprint([core,fixture],'stream_timing'),
            width=w,stages=s,architecture=p.architecture,capacity=s*(2 if p.architecture=='skid' else 1),minimum_latency_cycles=s,initiation_interval=1,fixture_observation_delay_cycles=2),
            note='Both pipelines preserve the accepted transaction stream and hold output under backpressure. Reset discards all pending data. Skid stages break downstream combinational readiness at the cost of two storage slots and a recovery bubble from full occupancy. Capacities differ and are explicit. No-stall latency is stages; fixture observations add two cycles. II=1 requires sustained no-stall checking, not Fmax alone.')


CATEGORY.ops.append(Op('synth_skid_buffer',SkidIn,Result,'Generate elastic and skid-buffer pipelines to explore backpressure timing and storage tradeoffs.'))
