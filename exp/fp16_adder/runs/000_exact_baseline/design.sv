module fpadd_fp16(input wire [15:0] a,b, output reg [15:0] y);
wire [4:0] ea=a[14:10], eb=b[14:10];
wire [10:0] ma={ea!=0,a[9:0]}, mb={eb!=0,b[9:0]};
wire [4:0] sa=ea==0?5'd0:ea-5'd1, sb=eb==0?5'd0:eb-5'd1;
wire [40:0] va={30'b0,ma}<<sa, vb={30'b0,mb}<<sb;
wire swap=va<vb;
wire [40:0] hi=swap?vb:va, lo=swap?va:vb;
wire signbit=swap?b[15]:a[15];
wire [41:0] mag=(a[15]==b[15])?({1'b0,hi}+{1'b0,lo}):({1'b0,hi}-{1'b0,lo});
reg [5:0] k,sh;
reg [41:0] q,remv,halfway;
reg [5:0] ex;
integer i;
always @* begin
k=0;
for(i=0;i<42;i=i+1) if(mag[i]) k=i;
sh=k>10?k-10:0;
q=mag>>sh;
remv=mag&((42'd1<<sh)-1);
halfway=sh==0?0:(42'd1<<(sh-1));
if(sh!=0 && (remv>halfway || (remv==halfway && q[0]))) q=q+1;
ex=sh+1;
if(q>=2048) begin q=q>>1; ex=ex+1; end
if(ex>=31) y={signbit,15'h7c00};
else if(q<1024) y={signbit,5'b0,q[9:0]};
else y={signbit,ex[4:0],q[9:0]};
if(mag==0) y={(a[15]&b[15]),15'b0};
if(ea==31) y={a[15],15'h7c00};
if(eb==31) y={b[15],15'h7c00};
if((ea==31 && a[9:0]!=0)||(eb==31 && b[9:0]!=0)||
(ea==31 && eb==31 && a[15]!=b[15])) y=16'h7e00;
end
endmodule
