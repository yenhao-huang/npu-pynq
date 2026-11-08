module fsm(input wire IN, CLK, RST, output wire MATCH);
  reg [2:0] s;
  always @(posedge CLK or posedge RST) begin
    if (RST) s <= 3'b001;
    else begin
      s[0] <= ~((s[2]) | (s[0] & IN));
      s[1] <= (IN);
      s[2] <= (s[1]) | (IN);
    end
  end
  assign MATCH = ~RST & IN & ((s[0] & s[1]));
endmodule
