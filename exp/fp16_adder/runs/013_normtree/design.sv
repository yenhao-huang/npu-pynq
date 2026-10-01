module fpadd_fp16(input wire [15:0] a,b,output reg [15:0] y);
wire swap=a[14:0]<b[14:0];
wire [15:0] big=swap?b:a, lower=swap?a:b;
wire signbit=big[15], opsub=a[15]^b[15];
wire [4:0] be=big[14:10]==0?5'd1:big[14:10];
wire [4:0] se=lower[14:10]==0?5'd1:lower[14:10];
wire [4:0] diff=be-se;
wire [13:0] h={big[14:10]!=0,big[9:0],3'b0};
wire [13:0] l={lower[14:10]!=0,lower[9:0],3'b0};
wire [13:0] j0=diff[0]?{1'b0,l[13:2],(|l[1:0])}:l;
wire [13:0] j1=diff[1]?{2'b0,j0[13:3],(|j0[2:0])}:j0;
wire [13:0] j2=diff[2]?{4'b0,j1[13:5],(|j1[4:0])}:j1;
wire [13:0] j3=diff[3]?{8'b0,j2[13:9],(|j2[8:0])}:j2;
wire [13:0] j4=diff[4]?{13'b0,(|j3)}:j3;
wire [13:0] t=j4;
wire [14:0] z={1'b0,h}+({1'b0,t}^{15{opsub}})+opsub;
wire n0=(|z[13:6])==0 && be>5'd8;
wire [13:0] v0=n0?(z[13:0]<<8):z[13:0];
wire [4:0] e0=n0?be-5'd8:be;
wire n1=(|v0[13:10])==0 && e0>5'd4;
wire [13:0] v1=n1?(v0<<4):v0;
wire [4:0] e1=n1?e0-5'd4:e0;
wire n2=(|v1[13:12])==0 && e1>5'd2;
wire [13:0] v2=n2?(v1<<2):v1;
wire [4:0] e2=n2?e1-5'd2:e1;
wire n3=(|v2[13:13])==0 && e2>5'd1;
wire [13:0] v3=n3?(v2<<1):v2;
wire [4:0] e3=n3?e2-5'd1:e2;
reg [13:0] norm;
reg [5:0] ex;
reg [11:0] rounded;
always @* begin
if(z[14]) begin norm={z[14:2],z[1]|z[0]};ex={1'b0,be}+1;end
else begin norm=v3;ex={1'b0,e3};end
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
