module fp12_mul(input wire [11:0] a,b,output reg [11:0] y);
wire signbit=a[11]^b[11];
wire [4:0] ea=a[10:6],eb=b[10:6];
wire [5:0] fa=a[5:0],fb=b[5:0];
wire [6:0] rawa={ea!=0,fa},rawb={eb!=0,fb};
wire [5:0] ax=ea==0?6'd1:{1'b0,ea},bx=eb==0?6'd1:{1'b0,eb};
wire [6:0] es={1'b0,ax}+{1'b0,bx};
reg [6:0] ma,mb;
reg [2:0] la,lb;
wire [13:0] p;
assign p = ma * mb;
always @* begin
casez(rawa)
7'b1??????: begin la=3'd0; ma=rawa; end
7'b01?????: begin la=3'd1; ma={rawa[5:0],1'b0}; end
7'b001????: begin la=3'd2; ma={rawa[4:0],2'b0}; end
7'b0001???: begin la=3'd3; ma={rawa[3:0],3'b0}; end
7'b00001??: begin la=3'd4; ma={rawa[2:0],4'b0}; end
7'b000001?: begin la=3'd5; ma={rawa[1:0],5'b0}; end
7'b0000001: begin la=3'd6; ma={rawa[0:0],6'b0}; end
default:begin la=3'd6;ma=0;end endcase
casez(rawb)
7'b1??????: begin lb=3'd0; mb=rawb; end
7'b01?????: begin lb=3'd1; mb={rawb[5:0],1'b0}; end
7'b001????: begin lb=3'd2; mb={rawb[4:0],2'b0}; end
7'b0001???: begin lb=3'd3; mb={rawb[3:0],3'b0}; end
7'b00001??: begin lb=3'd4; mb={rawb[2:0],4'b0}; end
7'b000001?: begin lb=3'd5; mb={rawb[1:0],5'b0}; end
7'b0000001: begin lb=3'd6; mb={rawb[0:0],6'b0}; end
default:begin lb=3'd6;mb=0;end endcase
end
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
6'd6:q=8'((p >> 6) + (p[5] & ((|p[4:0])|p[6])));
6'd7:q=8'((p >> 7) + (p[6] & ((|p[5:0])|p[7])));
6'd8:q=8'((p >> 8) + (p[7] & ((|p[6:0])|p[8])));
6'd9:q=8'((p >> 9) + (p[8] & ((|p[7:0])|p[9])));
6'd10:q=8'((p >> 10) + (p[9] & ((|p[8:0])|p[10])));
6'd11:q=8'((p >> 11) + (p[10] & ((|p[9:0])|p[11])));
6'd12:q=8'((p >> 12) + (p[11] & ((|p[10:0])|p[12])));
6'd13:q=8'((p >> 13) + (p[12] & ((|p[11:0])|p[13])));
6'd14:q=8'((p >> 14) + (p[13] & ((|p[12:0])|1'b0)));
default:q=0;endcase
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
