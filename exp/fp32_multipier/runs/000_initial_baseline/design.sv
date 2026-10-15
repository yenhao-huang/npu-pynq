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
reg signed [8:0] ex;
reg signed [8:0] sh;
reg [14:0] q;
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
ex=$signed({2'b0,es})+$signed({5'b0,k})-9'sd27;
sh=ex<1 ? 9'sd22-$signed({2'b0,es}) : $signed({5'b0,k})-9'sd6;
q=0; remv=0; halfway=0;
if(sh>0) begin
q={1'b0,p} >> sh;
remv={1'b0,p} & ((15'd1<<sh)-15'd1);
halfway=15'd1<<(sh-1);
if(remv>halfway || (remv==halfway && q[0])) q=q+15'd1;
end else q={1'b0,p}<<(-sh);
if(ex<1) ex=1;
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
