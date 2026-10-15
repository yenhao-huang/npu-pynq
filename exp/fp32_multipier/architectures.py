"""Separate normal and subnormal rounding paths to shorten shift selection."""
def rounded(shift):
    if shift==0:return "{1'b0,p[6:0]}"
    sticky=f'(|p[{shift-2}:0])' if shift>1 else "1'b0"
    lsb=f'p[{shift}]' if shift<14 else "1'b0"
    return f"8'((p >> {shift}) + (p[{shift-1}] & ({sticky}|{lsb})))"

def dual(prefix,lead):
    s=prefix+'''reg [3:0] k;
reg signed [6:0] ex;
reg [7:0] nq,sq,q;
reg [10:0] magnitude;
always @* begin
'''+lead+'''\nex=$signed(es)+$signed({3'b0,k})-7'sd27;
nq=0; sq=0;
case(k)
'''
    for k in range(6,14):s+=f"4'd{k}: nq={rounded(k-6)};\n"
    s+='default:nq=0; endcase\ncase(es)\n'
    for es in range(8,23):s+=f"7'd{es}: sq={rounded(22-es)};\n"
    s+='''default:sq=0; endcase
q=ex<1?sq:nq;
if(ex<1) ex=1;
if(q[7]) begin q=q>>1; ex=ex+1; end
if(ex>=31) magnitude=11'h7c0;
else if(ex==1 && !q[6]) magnitude={5'b0,q[5:0]};
else magnitude={ex[4:0],q[5:0]};
y={signbit,magnitude};
if(a[10:0]==0 || b[10:0]==0) y={signbit,11'b0};
if(ea==31 || eb==31) y={signbit,11'h7c0};
if((ea==31 && fa!=0)||(eb==31 && fb!=0)||
(ea==31 && b[10:0]==0)||(eb==31 && a[10:0]==0)) y=12'h7e0;
end
endmodule
'''
    return s

def prenorm(mul):
    from generate import multiplier
    s='''module fp12_mul(input wire [11:0] a,b,output reg [11:0] y);
wire signbit=a[11]^b[11];
wire [4:0] ea=a[10:6],eb=b[10:6];
wire [5:0] fa=a[5:0],fb=b[5:0];
wire [6:0] rawa={ea!=0,fa},rawb={eb!=0,fb};
wire [5:0] ax=ea==0?6'd1:{1'b0,ea},bx=eb==0?6'd1:{1'b0,eb};
wire [6:0] es={1'b0,ax}+{1'b0,bx};
reg [6:0] ma,mb;
reg [2:0] la,lb;
wire [13:0] p;
'''+multiplier(mul)+'\nalways @* begin\n'
    for v in ['a','b']:
        s+=f'casez(raw{v})\n'
        for n in range(7):
            bits='0'*n+'1'+'?'*(6-n)
            shifted=f'raw{v}' if n==0 else f"{{raw{v}[{6-n}:0],{n}'b0}}"
            s+=f"7'b{bits}: begin l{v}=3'd{n}; m{v}={shifted}; end\n"
        s+=f"default:begin l{v}=3'd6;m{v}=0;end endcase\n"
    s+='''end
reg signed [6:0] base,ex;
reg [5:0] sh;
reg [7:0] q;
reg [10:0] mag;
always @* begin
base=$signed(es)-7'sd15-$signed({4'b0,la})-$signed({4'b0,lb});
ex=base+$signed({6'b0,p[13]});
sh=ex<1 ? 6'(7-base) : (p[13]?6'd7:6'd6);
q=0;
case(sh)
'''
    for sh in range(6,15):s+=f"6'd{sh}:q={rounded(sh)};\n"
    s+='''default:q=0;endcase
if(ex<1) ex=1;
if(q[7]) begin q=q>>1;ex=ex+1;end
if(ex>=31) mag=11'h7c0;
else if(ex==1 && !q[6]) mag={5'b0,q[5:0]};
else mag={ex[4:0],q[5:0]};
y={signbit,mag};
if(a[10:0]==0 || b[10:0]==0) y={signbit,11'b0};
if(ea==31 || eb==31) y={signbit,11'h7c0};
if((ea==31 && fa!=0)||(eb==31 && fb!=0)||
(ea==31 && b[10:0]==0)||(eb==31 && a[10:0]==0)) y=12'h7e0;
end
endmodule
'''
    return s

def round_fields(shift):
    if shift==0:return "{p[5:0],1'b0}"
    sticky=f'(|p[{shift-2}:0])' if shift>1 else "1'b0"
    lsb=f'p[{shift}]' if shift<14 else "1'b0"
    return f"{{6'(p >> {shift}), (p[{shift-1}] & ({sticky}|{lsb}))}}"

def packed(prefix,lead):
    s=prefix+'''reg [3:0] k;
reg signed [6:0] ex;
reg [6:0] nf,sf,f;
reg [10:0] mag,unrounded;
always @* begin
'''+lead+'''\nex=$signed(es)+$signed({3'b0,k})-7'sd27;
nf=0;sf=0;
case(k)
'''
    for k in range(6,14):s+=f"4'd{k}:nf={round_fields(k-6)};\n"
    s+='default:nf=0;endcase\ncase(es)\n'
    for es in range(8,23):s+=f"7'd{es}:sf={round_fields(22-es)};\n"
    s+='''default:sf=0;endcase
f=ex<1?sf:nf;
unrounded={ex<1?5'b0:ex[4:0],f[6:1]};
mag=unrounded+{10'b0,f[0]};
if(ex>=31)mag=11'h7c0;
y={signbit,mag};
if(a[10:0]==0 || b[10:0]==0)y={signbit,11'b0};
if(ea==31 || eb==31)y={signbit,11'h7c0};
if((ea==31 && fa!=0)||(eb==31 && fb!=0)||
(ea==31 && b[10:0]==0)||(eb==31 && a[10:0]==0))y=12'h7e0;
end
endmodule
'''
    return s

def onehot(prefix,lead):
    s=prefix+'\n'
    masks=[];words=[];rounds=[];overs=[]
    for k in range(6,14):
        hi=f'!(|p[13:{k+1}])' if k<13 else "1'b1"
        s+=f"wire hit{k}=p[{k}] & {hi};\nwire valid{k}=hit{k} & (es>=7'd{28-k});\nwire [4:0] e{k}=5'(es-7'd{27-k});\nwire [6:0] f{k}={round_fields(k-6)};\n"
        masks.append(f'valid{k}');words.append("({11{valid%d}} & {e%d,f%d[6:1]})" % (k,k,k));rounds.append(f'(valid{k}&f{k}[0])');overs.append(f"(hit{k}&(es>=7'd{58-k}))")
    s+='wire normal='+'|'.join(masks)+';\nwire [10:0] nw='+'|'.join(words)+';\nwire nr='+'|'.join(rounds)+';\nwire overflow='+'|'.join(overs)+';\n'
    s+='''reg [6:0] sf;
reg [10:0] unrounded,mag;
reg inc;
always @* begin
sf=0;
case(es)
'''
    for es in range(8,23):s+=f"7'd{es}:sf={round_fields(22-es)};\n"
    s+='''default:sf=0;endcase
unrounded=normal?nw:{5'b0,sf[6:1]};
inc=normal?nr:sf[0];
mag=unrounded+{10'b0,inc};
if(overflow)mag=11'h7c0;
y={signbit,mag};
if(a[10:0]==0 || b[10:0]==0)y={signbit,11'b0};
if(ea==31 || eb==31)y={signbit,11'h7c0};
if((ea==31 && fa!=0)||(eb==31 && fb!=0)||
(ea==31 && b[10:0]==0)||(eb==31 && a[10:0]==0))y=12'h7e0;
end
endmodule
'''
    return s
