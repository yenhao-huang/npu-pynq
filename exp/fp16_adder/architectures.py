"""Compact binary16 alignment/addition/normalization architectures."""

def aligned(lzd='priority', shared=False):
    lead='shift=13;\n'+'\n'.join(f"if(z[{i}]) shift=4'd{13-i};" for i in range(14))
    if lzd=='case':
        lead="casez(z[13:0])\n"+'\n'.join(f"14'b{'0'*(13-i)}1{'?'*i}:shift=4'd{13-i};" for i in reversed(range(14)))+"\ndefault:shift=13; endcase"
    arithmetic="wire [14:0] z=opsub?({1'b0,h}-{1'b0,t}):({1'b0,h}+{1'b0,t});"
    if shared:
        arithmetic="wire [14:0] z={1'b0,h}+({1'b0,t}^{15{opsub}})+opsub;"
    return '''module fpadd_fp16(input wire [15:0] a,b,output reg [15:0] y);
wire swap=a[14:0]<b[14:0];
wire [15:0] big=swap?b:a, lower=swap?a:b;
wire signbit=big[15], opsub=a[15]^b[15];
wire [4:0] be=big[14:10]==0?5'd1:big[14:10];
wire [4:0] se=lower[14:10]==0?5'd1:lower[14:10];
wire [4:0] diff=be-se;
wire [13:0] h={big[14:10]!=0,big[9:0],3'b0};
wire [13:0] l={lower[14:10]!=0,lower[9:0],3'b0};
wire [13:0] shifted=l>>diff;
wire sticky=|(l & ((14'd1<<diff)-14'd1));
wire [13:0] t=shifted|{13'b0,sticky};
'''+arithmetic+'''
reg [3:0] shift;
reg [13:0] norm;
reg [5:0] ex;
reg [11:0] rounded;
always @* begin
'''+lead+'''
ex={1'b0,be};
if(shift>=be) shift=be-1;
if(z[14]) begin norm={z[14:2],z[1]|z[0]};ex=ex+1;end
else begin norm=z[13:0]<<shift;ex=ex-{2'b0,shift};end
rounded={1'b0,norm[13:3]}+{11'b0,(norm[2]&(norm[1]|norm[0]|norm[3]))};
if(rounded[11]) begin rounded=rounded>>1;ex=ex+1;end
if(ex>=31) y={signbit,15'h7c00};
else if(!rounded[10]) y={signbit,5'b0,rounded[9:0]};
else y={signbit,ex[4:0],rounded[9:0]};
if(z==0) y={a[15]&b[15],15'b0};
if(a[14:10]==31) y={a[15],15'h7c00};
if(b[14:10]==31) y={b[15],15'h7c00};
if((a[14:10]==31 && a[9:0]!=0)||(b[14:10]==31 && b[9:0]!=0)||
(a[14:10]==31 && b[14:10]==31 && opsub)) y=16'h7e00;
end
endmodule
'''


def staged(lzd='priority',shared=True):
    code=aligned(lzd,shared)
    old="""wire [13:0] shifted=l>>diff;
wire sticky=|(l & ((14'd1<<diff)-14'd1));
wire [13:0] t=shifted|{13'b0,sticky};"""
    lines=[]; prev='l'
    for j in range(5):
        s=1<<j
        value=f"{{{s}'b0,{prev}[13:{s+1}],(|{prev}[{s}:0])}}" if s<13 else f"{{13'b0,(|{prev})}}"
        lines.append(f'wire [13:0] j{j}=diff[{j}]?{value}:{prev};')
        prev=f'j{j}'
    return code.replace(old,'\n'.join(lines)+'\nwire [13:0] t=j4;')

def normalize_tree(lzd='priority'):
    code=staged(lzd,True)
    start=code.index('reg [3:0] shift;')
    end=code.index('rounded={1\'b0,norm[13:3]}')
    prefix=code[:start]
    lines=[];prev='z[13:0]';exp='be'
    for j,s in enumerate([8,4,2,1]):
        lines.append(f'wire n{j}=(|{prev}[13:{14-s}])==0 && {exp}>5\'d{s};' if j else f"wire n{j}=(|z[13:{14-s}])==0 && be>5'd{s};")
        lines.append(f"wire [13:0] v{j}=n{j}?({prev}<<{s}):{prev};")
        lines.append(f"wire [4:0] e{j}=n{j}?{exp}-5'd{s}:{exp};")
        prev=f'v{j}';exp=f'e{j}'
    return prefix+'\n'.join(lines)+'''
reg [13:0] norm;
reg [5:0] ex;
reg [11:0] rounded;
always @* begin
if(z[14]) begin norm={z[14:2],z[1]|z[0]};ex={1'b0,be}+1;end
else begin norm=v3;ex={1'b0,e3};end
'''+code[end:]

def packed(lzd='priority',parallel=False):
    code=staged(lzd,True)
    start=code.index("rounded={1'b0,norm[13:3]}")
    end=code.index('if(z==0)')
    code=code[:start]+'''y={signbit,((norm[13]?{ex[4:0],norm[12:3]}:{5'b0,norm[12:3]})+
{14'b0,(norm[2]&(norm[1]|norm[0]|norm[3]))})};
if(ex>=31) y={signbit,15'h7c00};
'''+code[end:]
    if parallel:
        code=code.replace('wire [4:0] diff=be-se;', '''wire [4:0] ae=a[14:10]==0?5'd1:a[14:10];
wire [4:0] ce=b[14:10]==0?5'd1:b[14:10];
wire [4:0] diff=swap?(ce-ae):(ae-ce);''')
    return code

def sentinel(lzd='case',parallel_exp=False):
    code=packed(lzd,True)
    code=code.replace('reg [3:0] shift;', "wire [13:0] leadbits=z[13:0]|(14'h2000>>(be-5'd1));\nreg [3:0] shift;")
    code=code.replace('casez(z[13:0])','casez(leadbits)')
    for i in range(14): code=code.replace(f'if(z[{i}]) shift=',f'if(leadbits[{i}]) shift=')
    code=code.replace('if(shift>=be) shift=be-1;','')
    if parallel_exp:
        code=code.replace("ex=ex-{2'b0,shift};", "case(shift)\n"+'\n'.join(f"4'd{i}:ex={{1'b0,be}}-6'd{i};" for i in range(14))+"\ndefault:ex=1;endcase\n")
    return code

def prefix(lzd='case',use_sentinel=False):
    code=sentinel(lzd) if use_sentinel else packed(lzd,True)
    old="wire [14:0] z={1'b0,h}+({1'b0,t}^{15{opsub}})+opsub;"
    lines=["wire [14:0] u={1'b0,h},v={1'b0,t}^{15{opsub}};",'wire [14:0] p0=u^v,g0=u&v;']
    for j,s in enumerate([1,2,4,8],1):
        prev=j-1
        lines.append(f"wire [14:0] g{j}=g{prev}|(p{prev}&(g{prev}<<{s}));")
        lines.append(f"wire [14:0] p{j}=p{prev}&((p{prev}<<{s})|15'd{(1<<s)-1});")
    lines.append("wire [14:0] z=p0^{(g4[13:0]|(p4[13:0]&{14{opsub}})),opsub};")
    return code.replace(old,'\n'.join(lines))

def round_prefix(lzd='case',pre=True):
    code=prefix(lzd,True) if pre else sentinel(lzd)
    code=code.replace('reg [11:0] rounded;', 'reg [14:0] encoded;\nreg increment;')
    start=code.index('y={signbit,((norm[13]?')
    end=code.index('if(ex>=31)',start)
    out="encoded=norm[13]?{ex[4:0],norm[12:3]}:{5'b0,norm[12:3]};\nincrement=norm[2]&(norm[1]|norm[0]|norm[3]);\ny[15]=signbit;\ny[0]=encoded[0]^increment;\n"
    out+='\n'.join(f'y[{i}]=encoded[{i}]^(increment & (&encoded[{i-1}:0]));' for i in range(1,15))+'\n'
    return code[:start]+out+code[end:]
