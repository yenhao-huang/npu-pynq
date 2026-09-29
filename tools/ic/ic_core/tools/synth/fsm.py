"""Clock-enable phase sequencers with explicit state encodings."""
from typing import Literal
from pydantic import Field, model_validator
from ...registry import Op, backend
from ..exploration_common import ExplorationInput, fingerprint
from . import CATEGORY
from .exploration import Result


class FSMIn(ExplorationInput):
    backend: str | None = Field(default='phase_fsm',description='Cyclic phase controller generator.')
    states: int = Field(default=64,ge=4,le=256,description='Power-of-two number of distinct output phases.')
    encoding: Literal['binary','onehot'] = Field(default='onehot',description='Preserved state representation; alternatives count as one operation.')

    @model_validator(mode='after')
    def power_two(self):
        if self.states & (self.states-1): raise ValueError('states must be a power of two')
        return self


@backend('synth','phase_fsm')
class FSM:
    def synth_fsm(self,p,ctx):
        n=p.states;bits=(n-1).bit_length()
        if p.encoding=='binary':
            body=f'''(* fsm_encoding = "none" *) reg [{bits-1}:0] state;
always @(posedge clk) begin
if(rst) state<=0;
else if(advance) state<=state+1'b1;
end
assign phases={n}'d1 << state;'''
        else:
            body=f'''(* fsm_encoding = "none", shreg_extract = "no" *) reg [{n-1}:0] state;
always @(posedge clk) begin
if(rst) state<={n}'d1;
else if(advance) state<={{state[{n-2}:0],state[{n-1}]}};
end
assign phases=state;'''
        core=ctx.run.artifacts/'phase.sv'
        core.write_text(f'module phase_dut(input clk,input rst,input advance,output [{n-1}:0] phases);\n{body}\nendmodule\n')
        fixture=ctx.run.artifacts/'phase_timing.sv'
        fixture.write_text(f'''module phase_timing(input clk,input rst,input advance,output reg [{n-1}:0] phases);
reg rst_q,advance_q;
wire [{n-1}:0] core_phases;
phase_dut core(.clk(clk),.rst(rst_q),.advance(advance_q),.phases(core_phases));
always @(posedge clk) begin rst_q<=rst;advance_q<=advance;phases<=core_phases;end
endmodule
''')
        return Result(data=dict(files=[str(ctx.run.final_artifact(core.name))],top='phase_dut',source_sha256=fingerprint([core],'phase_dut'),
            timing_files=[str(ctx.run.final_artifact(core.name)),str(ctx.run.final_artifact(fixture.name))],timing_top='phase_timing',timing_sha256=fingerprint([core,fixture],'phase_timing'),
            states=n,encoding=p.encoding,latency_cycles=1,initiation_interval=1,fixture_observation_delay_cycles=2),
            note='Synchronous reset selects phase zero. Each asserted advance moves one phase modulo states; deasserted advance holds. Output is one bit per phase. Initial state before the first reset edge and fault recovery from illegal internal states are outside the contract. This bounded cyclic controller is not a general arbitrary-transition FSM compiler.')


CATEGORY.ops.append(Op('synth_fsm',FSMIn,Result,'Generate equivalent cyclic phase controllers with binary or one-hot state storage.'))
