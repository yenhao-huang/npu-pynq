"""Real bypassed-DSP positive and active unclocked-DSP negative audit controls."""
import json
import os
from pathlib import Path
from ic_core import dispatch
from ic_core.tools.synth.clocked import DSP48E1_ACTIVE_REGS

here=Path(__file__).resolve().parent
os.chdir(here.parents[2])
out=here/'output'/'dsp';out.mkdir(parents=True,exist_ok=True)
rows=[]
for label,preg,expected in [('bypassed',0,True),('active_unclocked',1,False)]:
    parameters={key:0 for key in DSP48E1_ACTIVE_REGS}
    parameters.update(PREG=preg,ADREG=1,DREG=1)
    attrs=', '.join(f'.{key}({value})' for key,value in parameters.items())
    source=out/(label+'.sv')
    source.write_text(f'''module dsp_probe(input clk,input aux,input [95:0] x,output reg [47:0] y);
reg [95:0] q;
wire [47:0] p;
always @(posedge clk) begin q<=x;y<=p+q[47:0];end
(* DONT_TOUCH="yes" *) DSP48E1 #({attrs},.USE_DPORT("FALSE"),.USE_MULT("NONE")) core (
.A(q[95:66]),.B(q[65:48]),.C(q[47:0]),.D(25'b0),.P(p),
.ACIN(30'b0),.BCIN(18'b0),.PCIN(48'b0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
.ALUMODE(4'b0),.INMODE(5'b0),.OPMODE(7'b0110011),.CARRYINSEL(3'b0),.CARRYIN(1'b0),
.CLK({'aux' if preg else "1'b0"}),
.CEA1(1'b0),.CEA2(1'b0),.CEB1(1'b0),.CEB2(1'b0),.CEC(1'b0),.CED(1'b0),.CEAD(1'b0),
.CEM(1'b0),.CEP(1'b1),.CEALUMODE(1'b0),.CECTRL(1'b0),.CEINMODE(1'b0),.CECARRYIN(1'b0),
.RSTA(1'b0),.RSTB(1'b0),.RSTC(1'b0),.RSTD(1'b0),.RSTM(1'b0),.RSTP(1'b0),
.RSTALUMODE(1'b0),.RSTCTRL(1'b0),.RSTINMODE(1'b0),.RSTALLCARRYIN(1'b0));
endmodule
''')
    files=[str(source.resolve())]
    measured=dispatch('clocked_ppa',dict(files=files,top='dsp_probe',name=label,period_ns=5,optimization_mode='Basic',latency_cycles=2,initiation_interval=1))
    audit=dispatch('timing_constraint_audit',dict(record=measured['data']['record'],files=files,top='dsp_probe')) if measured['ok'] else None
    rows.append(dict(label=label,expected=expected,measurement=measured,audit=audit))
    (out/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(label,measured['ok'],None if audit is None else audit['ok'],flush=True)
    assert measured['ok'] and audit['ok']==expected
    if expected:assert len(audit['data']['inactive_dsp48e1_cells'])==1
