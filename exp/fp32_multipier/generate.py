"""Generate equivalent FP12 microarchitectures; every candidate is verified."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def multiplier(kind):
    if kind == 'operator':
        return 'assign p = ma * mb;'
    if kind == 'rows':
        rows = [f'wire [13:0] r{i} = ({{7\'b0,ma}} << {i}) & {{14{{mb[{i}]}}}};' for i in range(7)]
        return '\n'.join(rows) + '\nassign p = ' + '+'.join(f'r{i}' for i in range(7)) + ';'
    if kind == 'split':
        return "wire [9:0] lo = ma * mb[2:0]; wire [10:0] hi = ma * mb[6:3]; assign p = {4'b0,lo} + {hi,3'b0};"
    rows = [f'wire [13:0] r{i} = ({{7\'b0,ma}} << {i}) & {{14{{mb[{i}]}}}};' for i in range(7)]
    active = [f'r{i}' for i in range(7)]
    idx = 0
    while len(active)>2:
        nxt=[]
        for j in range(0,len(active),3):
            group=active[j:j+3]
            if len(group)<3: nxt+=group; continue
            a,b,c=group
            rows += [f'wire [13:0] s{idx} = {a}^{b}^{c};',f'wire [13:0] c{idx} = (({a}&{b})|({a}&{c})|({b}&{c}))<<1;']
            nxt += [f's{idx}',f'c{idx}']; idx+=1
        active=nxt
    rows.append(f'assign p = {active[0]} + {active[1]};')
    return '\n'.join(rows)

def generate(mode='baseline', mul='operator', lzd='priority'):
    if mode=='prenorm':
        from architectures import prenorm
        return prenorm(mul)
    # Baseline: unnormalized significands, dynamic bidirectional quantization.
    prefix = '''module fp12_mul(input wire [11:0] a,b, output reg [11:0] y);
wire signbit=a[11]^b[11];
wire [4:0] ea=a[10:6], eb=b[10:6];
wire [5:0] fa=a[5:0], fb=b[5:0];
wire [5:0] ax=ea==0?6'd1:{1'b0,ea}, bx=eb==0?6'd1:{1'b0,eb};
wire [6:0] es={1'b0,ax}+{1'b0,bx};
wire [6:0] ma={ea!=0,fa}, mb={eb!=0,fb};
wire [13:0] p;
'''+multiplier(mul)+'\n'
    if lzd=='priority':
        lead='k=0;\n'+ '\n'.join(f'if (p[{i}]) k=4\'d{i};' for i in range(14))
    else:
        lead="k=0; casez(p)\n"+'\n'.join(f"14'b{'0'*(13-i)}1{'?'*i}: k=4'd{i};" for i in reversed(range(14)))+"\ndefault:k=0; endcase"
    if mode=='onehot':
        from architectures import onehot
        return onehot(prefix,lead)
    if mode=='parallel':
        from architectures import packed
        code=packed(prefix,lead)
        old="ex=$signed(es)+$signed({3'b0,k})-7'sd27;"
        new="case(k)\n"+''.join(f"4'd{i}:ex=$signed(es)-7'sd{27-i};\n" for i in range(6,14))+"default:ex=-7'sd25;endcase"
        return code.replace(old,new)
    if mode=='packed':
        from architectures import packed
        return packed(prefix,lead)
    if mode=='dual':
        from architectures import dual
        return dual(prefix,lead)
    body='''reg [3:0] k;
reg signed [8:0] ex;
reg signed [8:0] sh;
reg [14:0] q;
reg [14:0] remv;
reg [14:0] halfway;
reg [10:0] magnitude;
always @* begin
'''+lead+'''
ex=$signed({2'b0,es})+$signed({5'b0,k})-9'sd27;
sh=ex<1 ? 9'sd22-$signed({2'b0,es}) : $signed({5'b0,k})-9'sd6;
q=0; remv=0; halfway=0;
'''
    if mode=='baseline':
        body+='''if(sh>14) q=0; else if(sh>0) begin
q={1'b0,p} >> sh;
remv={1'b0,p} & ((15'd1<<sh)-15'd1);
halfway=15'd1<<(sh-1);
if(remv>halfway || (remv==halfway && q[0])) q=q+15'd1;
end else q={1'b0,p}<<(-sh);
'''
    else:
        # Constant slices eliminate variable masks and comparators.
        body+='case(sh)\n'
        for shift in range(-6,15):
            key=f"9'sd{shift}" if shift>=0 else f"-9'sd{-shift}"
            if shift<=0: expr=f"{{1'b0,p}} << {-shift}"
            elif shift<=14:
                kept=f"p[{shift}]" if shift<14 else "1'b0"
                sticky=f'(|p[{shift-2}:0])' if shift>1 else "1'b0"
                expr=f"({{1'b0,p}} >> {shift}) + {{14'b0,(p[{shift-1}] & ({sticky}|{kept}))}}"
            body+=f'{key}: q={expr};\n'
        body+='default:q=0; endcase\n'
    body+='''if(ex<1) ex=1;
if(q>=128) begin q=q>>1; ex=ex+1; end
if(ex>=31) magnitude=11'h7c0;
else if(ex==1 && q<64) magnitude=q[10:0];
else magnitude={ex[4:0],q[5:0]};
y={signbit,magnitude};
if((a[10:0]==0)||(b[10:0]==0)) y={signbit,11'b0};
if(ea==31 || eb==31) y={signbit,11'h7c0};
if((ea==31 && fa!=0)||(eb==31 && fb!=0)||
(ea==31 && b[10:0]==0)||(eb==31 && a[10:0]==0)) y=12'h7e0;
end
endmodule
'''
    if mode == 'compact':
        body=body.replace('reg signed [8:0]', 'reg signed [6:0]').replace("9'sd", "7'sd").replace("{2'b0,es}","es").replace("{5'b0,k}","{3'b0,k}")
        body=body.replace('reg [14:0] q;', 'reg [7:0] q;').replace("q=q+15'd1", "q=q+8'd1").replace('magnitude=q[10:0]',"magnitude={3'b0,q}")
    return prefix+body

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','slices','compact','dual','prenorm','packed','parallel','onehot'],default='baseline');p.add_argument('--mul',choices=['operator','rows','split','csa'],default='operator');p.add_argument('--lzd',choices=['priority','case'],default='priority');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(generate(a.mode,a.mul,a.lzd))
