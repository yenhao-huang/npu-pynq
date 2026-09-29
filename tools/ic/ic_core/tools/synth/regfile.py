"""Two-read, one-write register storage with explicit bank-conflict semantics."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class RegfileIn(ExplorationInput):
    backend: str | None = Field(default='banked_regfile',description='Conflict-aware register-file generator.')
    width: int = Field(default=16,ge=8,le=64,description='Data word width.')
    depth: int = Field(default=64,ge=16,le=256,description='Power-of-two number of logical words.')
    banks: int = Field(default=2,ge=2,le=8,description='Power-of-two bank count; low address bits select bank.')
    architecture: Literal['registers','banked'] = Field(default='banked',description='Resettable flip-flop words or banked distributed RAM with reset validity bits.')

    @model_validator(mode='after')
    def geometry(self):
        if self.depth&(self.depth-1) or self.banks&(self.banks-1) or self.depth<2*self.banks:
            raise ValueError('power-of-two geometry requires at least two rows per bank')
        return self


@backend('synth','banked_regfile')
class Regfile:
    def synth_banked_regfile(self,p,ctx):
        w,d,b=p.width,p.depth,p.banks;a=(d-1).bit_length();bb=(b-1).bit_length()
        inputs=w+3*a+3;outputs=2*w+3
        lines=[f'''module regfile_dut(input clk,input rst,input [{inputs-1}:0] x,output reg [{outputs-1}:0] y);
wire [{w-1}:0] wd=x[{w-1}:0];
wire [{a-1}:0] wa=x[{w} +: {a}],ra0=x[{w+a} +: {a}],ra1=x[{w+2*a} +: {a}];
wire we=x[{w+3*a}],re0=x[{w+3*a+1}],re1=x[{w+3*a+2}];
wire conflict=re0 && re1 && (ra0[{bb-1}:0]==ra1[{bb-1}:0]);
wire accept0=re0,accept1=re1 && !conflict;''']
        if p.architecture=='registers':
            lines.append(f'''(* ram_style="registers" *) reg [{w-1}:0] mem[0:{d-1}];
integer i;
always @(posedge clk) begin
if(rst) for(i=0;i<{d};i=i+1) mem[i]<=0;
else if(we) mem[wa]<=wd;
end
wire [{w-1}:0] read0=mem[ra0],read1=mem[ra1];''')
        else:
            lines.append(f'''reg [{d-1}:0] initialized;
always @(posedge clk) begin
if(rst) initialized<=0;
else if(we) initialized[wa]<=1'b1;
end''')
            for bank in range(b):
                lines.append(f'''(* ram_style="distributed" *) reg [{w-1}:0] bank{bank}[0:{d//b-1}];
wire [{a-bb-1}:0] row{bank}=(re0 && ra0[{bb-1}:0]=={bb}'d{bank}) ? ra0[{a-1}:{bb}] : ra1[{a-1}:{bb}];
wire [{w-1}:0] value{bank}=bank{bank}[row{bank}];
always @(posedge clk) if(!rst && we && wa[{bb-1}:0]=={bb}'d{bank}) bank{bank}[wa[{a-1}:{bb}]]<=wd;''')
            for port in (0,1):
                terms=[f"(ra{port}[{bb-1}:0]=={bb}'d{bank}) ? value{bank} : " for bank in range(b)]
                lines.append(f"wire [{w-1}:0] read{port}=initialized[ra{port}] ? ("+''.join(terms)+f"{w}'d0) : {w}'d0;")
        lines.append(f'''always @(posedge clk) begin
if(rst) y<=0;
else y<={{conflict,accept1,accept0,(accept1 ? read1 : {w}'d0),(accept0 ? read0 : {w}'d0)}};
end
endmodule
''')
        core=ctx.run.artifacts/'regfile.sv';core.write_text('\n'.join(lines))
        fixture=ctx.run.artifacts/'regfile_timing.sv'
        fixture.write_text(f'''module regfile_timing(input clk,input rst,input [{inputs-1}:0] x,output reg [{outputs-1}:0] y);
reg rst_q;
reg [{inputs-1}:0] x_q;
wire [{outputs-1}:0] result;
regfile_dut core(.clk(clk),.rst(rst_q),.x(x_q),.y(result));
always @(posedge clk) begin rst_q<=rst;x_q<=x;y<=result;end
endmodule
''')
        return Result(data=dict(files=[str(ctx.run.final_artifact(core.name))],top='regfile_dut',source_sha256=fingerprint([core],'regfile_dut'),
            timing_files=[str(ctx.run.final_artifact(core.name)),str(ctx.run.final_artifact(fixture.name))],timing_top='regfile_timing',timing_sha256=fingerprint([core,fixture],'regfile_timing'),
            width=w,depth=d,banks=b,input_width=inputs,output_width=outputs,architecture=p.architecture,latency_cycles=1,initiation_interval=1,fixture_observation_delay_cycles=2),
            note='One command per cycle contains one write and up to two reads. Both architectures enforce low-address bank conflicts: read0 wins, read1 is rejected and returns zero even for identical addresses. Reads observe old data on write collision. Reset makes all logical words zero; RAM bits use validity masking. Output packs read0/read1 words then accepted0/accepted1/conflict flags. II counts command batches, not guaranteed read completions.')


CATEGORY.ops.append(Op('synth_banked_regfile',RegfileIn,Result,'Generate conflict-aware two-read/one-write storage with register or banked LUTRAM implementations.'))
