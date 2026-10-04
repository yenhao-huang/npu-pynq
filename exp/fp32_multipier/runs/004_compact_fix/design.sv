module fp12_mul(input wire [11:0] a,b, output reg [11:0] y);
wire signbit=a[11]^b[11];
wire [4:0] ea=a[10:6], eb=b[10:6];
wire [5:0] fa=a[5:0], fb=b[5:0];
wire [5:0] ax=ea==0?6'd1:{1'b0,ea}, bx=eb==0?6'd1:{1'b0,eb};
wire [6:0] es={1'b0,ax}+{1'b0,bx};
wire [6:0] ma={ea!=0,fa}, mb={eb!=0,fb};
wire [13:0] p;
assign p = ma * mb;
reg [3:0] k;
reg signed [6:0] ex;
reg signed [6:0] sh;
reg [7:0] q;
reg [14:0] remv;
reg [14:0] halfway;
reg [10:0] magnitude;
always @* begin
k=0;
if (p[0]) k=4'd0;
if (p[1]) k=4'd1;
if (p[2]) k=4'd2;
if (p[3]) k=4'd3;
if (p[4]) k=4'd4;
if (p[5]) k=4'd5;
if (p[6]) k=4'd6;
if (p[7]) k=4'd7;
if (p[8]) k=4'd8;
if (p[9]) k=4'd9;
if (p[10]) k=4'd10;
if (p[11]) k=4'd11;
if (p[12]) k=4'd12;
if (p[13]) k=4'd13;
ex=$signed(es)+$signed({3'b0,k})-7'sd27;
sh=ex<1 ? 7'sd22-$signed(es) : $signed({3'b0,k})-7'sd6;
q=0; remv=0; halfway=0;
case(sh)
-7'sd6: q={1'b0,p} << 6;
-7'sd5: q={1'b0,p} << 5;
-7'sd4: q={1'b0,p} << 4;
-7'sd3: q={1'b0,p} << 3;
-7'sd2: q={1'b0,p} << 2;
-7'sd1: q={1'b0,p} << 1;
7'sd0: q={1'b0,p} << 0;
7'sd1: q=({1'b0,p} >> 1) + {14'b0,(p[0] & (1'b0|p[1]))};
7'sd2: q=({1'b0,p} >> 2) + {14'b0,(p[1] & ((|p[0:0])|p[2]))};
7'sd3: q=({1'b0,p} >> 3) + {14'b0,(p[2] & ((|p[1:0])|p[3]))};
7'sd4: q=({1'b0,p} >> 4) + {14'b0,(p[3] & ((|p[2:0])|p[4]))};
7'sd5: q=({1'b0,p} >> 5) + {14'b0,(p[4] & ((|p[3:0])|p[5]))};
7'sd6: q=({1'b0,p} >> 6) + {14'b0,(p[5] & ((|p[4:0])|p[6]))};
7'sd7: q=({1'b0,p} >> 7) + {14'b0,(p[6] & ((|p[5:0])|p[7]))};
7'sd8: q=({1'b0,p} >> 8) + {14'b0,(p[7] & ((|p[6:0])|p[8]))};
7'sd9: q=({1'b0,p} >> 9) + {14'b0,(p[8] & ((|p[7:0])|p[9]))};
7'sd10: q=({1'b0,p} >> 10) + {14'b0,(p[9] & ((|p[8:0])|p[10]))};
7'sd11: q=({1'b0,p} >> 11) + {14'b0,(p[10] & ((|p[9:0])|p[11]))};
7'sd12: q=({1'b0,p} >> 12) + {14'b0,(p[11] & ((|p[10:0])|p[12]))};
7'sd13: q=({1'b0,p} >> 13) + {14'b0,(p[12] & ((|p[11:0])|p[13]))};
7'sd14: q=({1'b0,p} >> 14) + {14'b0,(p[13] & ((|p[12:0])|1'b0))};
default:q=0; endcase
if(ex<1) ex=1;
if(q>=128) begin q=q>>1; ex=ex+1; end
if(ex>=31) magnitude=11'h7c0;
else if(ex==1 && q<64) magnitude={3'b0,q};
else magnitude={ex[4:0],q[5:0]};
y={signbit,magnitude};
if((a[10:0]==0)||(b[10:0]==0)) y={signbit,11'b0};
if(ea==31 || eb==31) y={signbit,11'h7c0};
if((ea==31 && fa!=0)||(eb==31 && fb!=0)||
(ea==31 && b[10:0]==0)||(eb==31 && a[10:0]==0)) y=12'h7e0;
end
endmodule
