module fp12_mul(input wire [11:0] a,b, output reg [11:0] y);
wire signbit=a[11]^b[11];
wire [4:0] ea=a[10:6], eb=b[10:6];
wire [5:0] fa=a[5:0], fb=b[5:0];
wire [5:0] ax=ea==0?6'd1:{1'b0,ea}, bx=eb==0?6'd1:{1'b0,eb};
wire [6:0] es={1'b0,ax}+{1'b0,bx};
wire [6:0] ma={ea!=0,fa}, mb={eb!=0,fb};
wire [13:0] p;
assign p = ma * mb;

wire hit6=p[6] & !(|p[13:7]);
wire valid6=hit6 & (es>=7'd22);
wire [4:0] e6=5'(es-7'd21);
wire [6:0] f6={p[5:0],1'b0};
wire hit7=p[7] & !(|p[13:8]);
wire valid7=hit7 & (es>=7'd21);
wire [4:0] e7=5'(es-7'd20);
wire [6:0] f7={6'(p >> 1), (p[0] & (1'b0|p[1]))};
wire hit8=p[8] & !(|p[13:9]);
wire valid8=hit8 & (es>=7'd20);
wire [4:0] e8=5'(es-7'd19);
wire [6:0] f8={6'(p >> 2), (p[1] & ((|p[0:0])|p[2]))};
wire hit9=p[9] & !(|p[13:10]);
wire valid9=hit9 & (es>=7'd19);
wire [4:0] e9=5'(es-7'd18);
wire [6:0] f9={6'(p >> 3), (p[2] & ((|p[1:0])|p[3]))};
wire hit10=p[10] & !(|p[13:11]);
wire valid10=hit10 & (es>=7'd18);
wire [4:0] e10=5'(es-7'd17);
wire [6:0] f10={6'(p >> 4), (p[3] & ((|p[2:0])|p[4]))};
wire hit11=p[11] & !(|p[13:12]);
wire valid11=hit11 & (es>=7'd17);
wire [4:0] e11=5'(es-7'd16);
wire [6:0] f11={6'(p >> 5), (p[4] & ((|p[3:0])|p[5]))};
wire hit12=p[12] & !(|p[13:13]);
wire valid12=hit12 & (es>=7'd16);
wire [4:0] e12=5'(es-7'd15);
wire [6:0] f12={6'(p >> 6), (p[5] & ((|p[4:0])|p[6]))};
wire hit13=p[13] & 1'b1;
wire valid13=hit13 & (es>=7'd15);
wire [4:0] e13=5'(es-7'd14);
wire [6:0] f13={6'(p >> 7), (p[6] & ((|p[5:0])|p[7]))};
wire normal=valid6|valid7|valid8|valid9|valid10|valid11|valid12|valid13;
wire [10:0] nw=({11{valid6}} & {e6,f6[6:1]})|({11{valid7}} & {e7,f7[6:1]})|({11{valid8}} & {e8,f8[6:1]})|({11{valid9}} & {e9,f9[6:1]})|({11{valid10}} & {e10,f10[6:1]})|({11{valid11}} & {e11,f11[6:1]})|({11{valid12}} & {e12,f12[6:1]})|({11{valid13}} & {e13,f13[6:1]});
wire nr=(valid6&f6[0])|(valid7&f7[0])|(valid8&f8[0])|(valid9&f9[0])|(valid10&f10[0])|(valid11&f11[0])|(valid12&f12[0])|(valid13&f13[0]);
wire overflow=(hit6&(es>=7'd52))|(hit7&(es>=7'd51))|(hit8&(es>=7'd50))|(hit9&(es>=7'd49))|(hit10&(es>=7'd48))|(hit11&(es>=7'd47))|(hit12&(es>=7'd46))|(hit13&(es>=7'd45));
reg [6:0] sf;
reg [10:0] unrounded,mag;
reg inc;
always @* begin
sf=0;
case(es)
7'd8:sf={6'(p >> 14), (p[13] & ((|p[12:0])|1'b0))};
7'd9:sf={6'(p >> 13), (p[12] & ((|p[11:0])|p[13]))};
7'd10:sf={6'(p >> 12), (p[11] & ((|p[10:0])|p[12]))};
7'd11:sf={6'(p >> 11), (p[10] & ((|p[9:0])|p[11]))};
7'd12:sf={6'(p >> 10), (p[9] & ((|p[8:0])|p[10]))};
7'd13:sf={6'(p >> 9), (p[8] & ((|p[7:0])|p[9]))};
7'd14:sf={6'(p >> 8), (p[7] & ((|p[6:0])|p[8]))};
7'd15:sf={6'(p >> 7), (p[6] & ((|p[5:0])|p[7]))};
7'd16:sf={6'(p >> 6), (p[5] & ((|p[4:0])|p[6]))};
7'd17:sf={6'(p >> 5), (p[4] & ((|p[3:0])|p[5]))};
7'd18:sf={6'(p >> 4), (p[3] & ((|p[2:0])|p[4]))};
7'd19:sf={6'(p >> 3), (p[2] & ((|p[1:0])|p[3]))};
7'd20:sf={6'(p >> 2), (p[1] & ((|p[0:0])|p[2]))};
7'd21:sf={6'(p >> 1), (p[0] & (1'b0|p[1]))};
7'd22:sf={p[5:0],1'b0};
default:sf=0;endcase
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
