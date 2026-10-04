module fpadd_fp16(input wire [15:0] a,b,output reg [15:0] y);
wire swap=a[14:0]<b[14:0];
wire [15:0] big=swap?b:a, small=swap?a:b;
wire signbit=big[15], opsub=a[15]^b[15];
wire [4:0] be=big[14:10]==0?5'd1:big[14:10];
wire [4:0] se=small[14:10]==0?5'd1:small[14:10];
wire [4:0] diff=be-se;
wire [13:0] h={big[14:10]!=0,big[9:0],3'b0};
wire [13:0] l={small[14:10]!=0,small[9:0],3'b0};
wire [13:0] shifted=l>>diff;
wire sticky=|(l & ((14'd1<<diff)-14'd1));
wire [13:0] t=shifted|{13'b0,sticky};
wire [14:0] z=opsub?({1'b0,h}-{1'b0,t}):({1'b0,h}+{1'b0,t});
reg [3:0] shift;
reg [13:0] norm;
reg [5:0] ex;
reg [11:0] rounded;
always @* begin
shift=13;
if(z[0]) shift=4'd13;
if(z[1]) shift=4'd12;
if(z[2]) shift=4'd11;
if(z[3]) shift=4'd10;
if(z[4]) shift=4'd9;
if(z[5]) shift=4'd8;
if(z[6]) shift=4'd7;
if(z[7]) shift=4'd6;
if(z[8]) shift=4'd5;
if(z[9]) shift=4'd4;
if(z[10]) shift=4'd3;
if(z[11]) shift=4'd2;
if(z[12]) shift=4'd1;
if(z[13]) shift=4'd0;
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
