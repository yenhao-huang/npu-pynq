"""Packed matrix batches implemented by a parallel bank or a 2D MAC wavefront."""
from typing import Literal
from pydantic import Field
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class SystolicIn(ExplorationInput):
    backend: str | None = Field(default='systolic_rtl',description='Signed spatial matrix-tile generator.')
    width: int = Field(default=16,ge=8,le=32,description='Signed element width.')
    size: int = Field(default=2,ge=2,le=4,description='Square matrix dimension and spatial PE rows/columns.')
    architecture: Literal['parallel','systolic'] = Field(default='systolic',description='All products in parallel or output-stationary nearest-neighbor PE wavefront.')


@backend('synth','systolic_rtl')
class Systolic:
    def synth_systolic_tile(self,p,ctx):
        w,n=p.width,p.size
        acc=2*w+(n-1).bit_length();ib=2*n*n*w;ob=n*n*acc
        latency=1 if p.architecture=='parallel' else 3*n-1
        lines=[f'''(* use_dsp = "yes" *) module matrix_dut(input clk,input rst,
input valid_in,output ready_in,input [{ib-1}:0] data_in,
output valid_out,input ready_out,output reg [{ob-1}:0] data_out);
reg busy,valid_q;
assign valid_out=!rst && valid_q;
assign ready_in=!rst && !busy && (!valid_q || ready_out);''']
        if p.architecture=='parallel':
            for i in range(n):
                for j in range(n):
                    terms=[]
                    for k in range(n):
                        name=f'p_{i}_{j}_{k}'
                        lines.append(f'wire signed [{2*w-1}:0] {name}=$signed(data_in[{(i*n+k)*w} +: {w}]) * $signed(data_in[{(n*n+k*n+j)*w} +: {w}]);')
                        terms.append(f'$signed({{{{{acc-2*w}{{{name}[{2*w-1}]}}}},{name}}})')
                    lines.append(f'wire signed [{acc-1}:0] result_{i}_{j}='+ ' + '.join(terms)+';')
            results='\n'.join(f'data_out[{(i*n+j)*acc} +: {acc}]<=result_{i}_{j};' for i in range(n) for j in range(n))
            lines.append(f'''always @(posedge clk) begin
if(rst) begin busy<=0;valid_q<=0;data_out<=0;end
else begin
if(valid_q && ready_out) valid_q<=0;
if(valid_in && ready_in) begin valid_q<=1;{results} end
end
end''')
        else:
            lines.extend([f'reg [{ib-1}:0] operands;',f'reg [{(3*n-2).bit_length()-1}:0] phase;'])
            for i in range(n):
                for j in range(n):
                    lines.append(f'reg signed [{w-1}:0] a_{i}_{j},b_{i}_{j};\nreg signed [{acc-1}:0] sum_{i}_{j};')
            for i in range(n):
                a=' : '.join(f'(phase=={i+k}) ? $signed(operands[{(i*n+k)*w} +: {w}])' for k in range(n))+f" : {w}'sd0"
                b=' : '.join(f'(phase=={i+k}) ? $signed(operands[{(n*n+k*n+i)*w} +: {w}])' for k in range(n))+f" : {w}'sd0"
                lines.append(f'wire signed [{w-1}:0] left_{i}={a};\nwire signed [{w-1}:0] above_{i}={b};')
            reset=[];work=[];capture=[]
            for i in range(n):
                for j in range(n):
                    a=f'left_{i}' if j==0 else f'a_{i}_{j-1}'
                    b=f'above_{j}' if i==0 else f'b_{i-1}_{j}'
                    prod=f'product_{i}_{j}'
                    lines.append(f'wire signed [{2*w-1}:0] {prod}={a} * {b};')
                    lines.append(f'wire signed [{acc-1}:0] next_{i}_{j}=sum_{i}_{j} + $signed({{{{{acc-2*w}{{{prod}[{2*w-1}]}}}},{prod}}});')
                    reset.append(f'a_{i}_{j}<=0;b_{i}_{j}<=0;sum_{i}_{j}<=0;')
                    work.append(f'a_{i}_{j}<={a};b_{i}_{j}<={b};sum_{i}_{j}<=next_{i}_{j};')
                    capture.append(f'data_out[{(i*n+j)*acc} +: {acc}]<=next_{i}_{j};')
            clear='\n'.join(reset)
            lines.append(f'''always @(posedge clk) begin
if(rst) begin busy<=0;valid_q<=0;data_out<=0;operands<=0;phase<=0;{clear} end
else begin
if(valid_q && ready_out) valid_q<=0;
if(valid_in && ready_in) begin busy<=1;valid_q<=0;operands<=data_in;phase<=0;{clear} end
else if(busy) begin
{chr(10).join(work)}
if(phase=={3*n-3}) begin busy<=0;valid_q<=1;{chr(10).join(capture)} end
else phase<=phase+1'b1;
end
end
end''')
        core=ctx.run.artifacts/'matrix.sv';core.write_text('\n'.join(lines+['endmodule\n']))
        fixture=ctx.run.artifacts/'matrix_timing.sv'
        fixture.write_text(f'''module matrix_timing(input clk,input rst,input valid_in,output reg ready_in,
input [{ib-1}:0] data_in,output reg valid_out,input ready_out,output reg [{ob-1}:0] data_out);
reg rst_q,valid_q,ready_q;
reg [{ib-1}:0] data_q;
wire core_ready,core_valid;
wire [{ob-1}:0] core_data;
matrix_dut core(.clk(clk),.rst(rst_q),.valid_in(valid_q),.ready_in(core_ready),.data_in(data_q),.valid_out(core_valid),.ready_out(ready_q),.data_out(core_data));
always @(posedge clk) begin
rst_q<=rst;valid_q<=valid_in;ready_q<=ready_out;data_q<=data_in;
ready_in<=core_ready;valid_out<=core_valid;data_out<=core_data;
end
endmodule
''')
        return Result(data=dict(files=[str(ctx.run.final_artifact(core.name))],top='matrix_dut',source_sha256=fingerprint([core],'matrix_dut'),
            timing_files=[str(ctx.run.final_artifact(core.name)),str(ctx.run.final_artifact(fixture.name))],timing_top='matrix_timing',timing_sha256=fingerprint([core,fixture],'matrix_timing'),
            width=w,size=n,operation='matrix',architecture=p.architecture,input_width=ib,output_width=ob,accumulator_width=acc,
            latency_cycles=latency,initiation_interval=latency,fixture_observation_delay_cycles=2),
            note='Row-major A occupies low input bits, followed by row-major B; row-major exact signed C=A*B occupies output bits. Systolic PEs forward A right and B down, with skewed boundary injection and local accumulation. One batch is outstanding; reset cancels it and output is held during backpressure. Latency/II include operand capture and wavefront drain. Fixture adds two observation cycles, not an external protocol adapter. DSP mapping costs and throughput must be measured; no area or speed gain is assumed.')


CATEGORY.ops.append(Op('synth_systolic_tile',SystolicIn,Result,'Generate signed 2D matrix wavefronts and parallel matrix baselines with explicit batch cycle contracts.'))
