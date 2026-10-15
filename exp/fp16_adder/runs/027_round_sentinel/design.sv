module fpadd_fp16(input wire [15:0] a,b,output reg [15:0] y);
wire swap=a[14:0]<b[14:0];
wire [15:0] big=swap?b:a, lower=swap?a:b;
wire signbit=big[15], opsub=a[15]^b[15];
wire [4:0] be=big[14:10]==0?5'd1:big[14:10];
wire [4:0] se=lower[14:10]==0?5'd1:lower[14:10];
wire [4:0] ae=a[14:10]==0?5'd1:a[14:10];
wire [4:0] ce=b[14:10]==0?5'd1:b[14:10];
wire [4:0] diff=swap?(ce-ae):(ae-ce);
wire [13:0] h={big[14:10]!=0,big[9:0],3'b0};
wire [13:0] l={lower[14:10]!=0,lower[9:0],3'b0};
wire [13:0] j0=diff[0]?{1'b0,l[13:2],(|l[1:0])}:l;
wire [13:0] j1=diff[1]?{2'b0,j0[13:3],(|j0[2:0])}:j0;
wire [13:0] j2=diff[2]?{4'b0,j1[13:5],(|j1[4:0])}:j1;
wire [13:0] j3=diff[3]?{8'b0,j2[13:9],(|j2[8:0])}:j2;
wire [13:0] j4=diff[4]?{13'b0,(|j3)}:j3;
wire [13:0] t=j4;
wire [14:0] z={1'b0,h}+({1'b0,t}^{15{opsub}})+opsub;
wire [13:0] leadbits=z[13:0]|(14'h2000>>(be-5'd1));
reg [3:0] shift;
reg [13:0] norm;
reg [5:0] ex;
reg [14:0] encoded;
reg increment;
always @* begin
casez(leadbits)
14'b1?????????????:shift=4'd0;
14'b01????????????:shift=4'd1;
14'b001???????????:shift=4'd2;
14'b0001??????????:shift=4'd3;
14'b00001?????????:shift=4'd4;
14'b000001????????:shift=4'd5;
14'b0000001???????:shift=4'd6;
14'b00000001??????:shift=4'd7;
14'b000000001?????:shift=4'd8;
14'b0000000001????:shift=4'd9;
14'b00000000001???:shift=4'd10;
14'b000000000001??:shift=4'd11;
14'b0000000000001?:shift=4'd12;
14'b00000000000001:shift=4'd13;
default:shift=13; endcase
ex={1'b0,be};

if(z[14]) begin norm={z[14:2],z[1]|z[0]};ex=ex+1;end
else begin norm=z[13:0]<<shift;ex=ex-{2'b0,shift};end
encoded=norm[13]?{ex[4:0],norm[12:3]}:{5'b0,norm[12:3]};
increment=norm[2]&(norm[1]|norm[0]|norm[3]);
y[15]=signbit;
y[0]=encoded[0]^increment;
y[1]=encoded[1]^(increment & (&encoded[0:0]));
y[2]=encoded[2]^(increment & (&encoded[1:0]));
y[3]=encoded[3]^(increment & (&encoded[2:0]));
y[4]=encoded[4]^(increment & (&encoded[3:0]));
y[5]=encoded[5]^(increment & (&encoded[4:0]));
y[6]=encoded[6]^(increment & (&encoded[5:0]));
y[7]=encoded[7]^(increment & (&encoded[6:0]));
y[8]=encoded[8]^(increment & (&encoded[7:0]));
y[9]=encoded[9]^(increment & (&encoded[8:0]));
y[10]=encoded[10]^(increment & (&encoded[9:0]));
y[11]=encoded[11]^(increment & (&encoded[10:0]));
y[12]=encoded[12]^(increment & (&encoded[11:0]));
y[13]=encoded[13]^(increment & (&encoded[12:0]));
y[14]=encoded[14]^(increment & (&encoded[13:0]));
if(ex>=31) y={signbit,15'h7c00};
if(z==0) y={a[15]&b[15],15'b0};
if(a[14:10]==31) y={a[15],15'h7c00};
if(b[14:10]==31) y={b[15],15'h7c00};
if((a[14:10]==31 && a[9:0]!=0)||(b[14:10]==31 && b[9:0]!=0)||
(a[14:10]==31 && b[14:10]==31 && opsub)) y=16'h7e00;
end
endmodule
