module fp12_mul(input wire [11:0] a,b, output reg [11:0] y);
wire signbit=a[11]^b[11];
wire [4:0] ea=a[10:6], eb=b[10:6];
wire [5:0] fa=a[5:0], fb=b[5:0];
wire [5:0] ax=ea==0?6'd1:{1'b0,ea}, bx=eb==0?6'd1:{1'b0,eb};
wire [6:0] es={1'b0,ax}+{1'b0,bx};
wire [6:0] ma={ea!=0,fa}, mb={eb!=0,fb};
wire [13:0] p;
wire [13:0] r0 = ({7'b0,ma} << 0) & {14{mb[0]}};
wire [13:0] r1 = ({7'b0,ma} << 1) & {14{mb[1]}};
wire [13:0] r2 = ({7'b0,ma} << 2) & {14{mb[2]}};
wire [13:0] r3 = ({7'b0,ma} << 3) & {14{mb[3]}};
wire [13:0] r4 = ({7'b0,ma} << 4) & {14{mb[4]}};
wire [13:0] r5 = ({7'b0,ma} << 5) & {14{mb[5]}};
wire [13:0] r6 = ({7'b0,ma} << 6) & {14{mb[6]}};
wire [13:0] s0 = r0^r1^r2;
wire [13:0] c0 = ((r0&r1)|(r0&r2)|(r1&r2))<<1;
wire [13:0] s1 = r3^r4^r5;
wire [13:0] c1 = ((r3&r4)|(r3&r5)|(r4&r5))<<1;
wire [13:0] s2 = s0^c0^s1;
wire [13:0] c2 = ((s0&c0)|(s0&s1)|(c0&s1))<<1;
wire [13:0] s3 = s2^c2^c1;
wire [13:0] c3 = ((s2&c2)|(s2&c1)|(c2&c1))<<1;
wire [13:0] s4 = s3^c3^r6;
wire [13:0] c4 = ((s3&c3)|(s3&r6)|(c3&r6))<<1;
assign p = s4 + c4;
reg [3:0] k;
reg signed [8:0] ex;
reg signed [8:0] sh;
reg [14:0] q;
reg [14:0] remv;
reg [14:0] halfway;
reg [10:0] magnitude;
always @* begin
k=0; casez(p)
14'b1?????????????: k=4'd13;
14'b01????????????: k=4'd12;
14'b001???????????: k=4'd11;
14'b0001??????????: k=4'd10;
14'b00001?????????: k=4'd9;
14'b000001????????: k=4'd8;
14'b0000001???????: k=4'd7;
14'b00000001??????: k=4'd6;
14'b000000001?????: k=4'd5;
14'b0000000001????: k=4'd4;
14'b00000000001???: k=4'd3;
14'b000000000001??: k=4'd2;
14'b0000000000001?: k=4'd1;
14'b00000000000001: k=4'd0;
default:k=0; endcase
ex=$signed({2'b0,es})+$signed({5'b0,k})-9'sd27;
sh=ex<1 ? 9'sd22-$signed({2'b0,es}) : $signed({5'b0,k})-9'sd6;
q=0; remv=0; halfway=0;
if(sh>14) q=0; else if(sh>0) begin
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
