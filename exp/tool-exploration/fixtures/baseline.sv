module dut(input logic [3:0] a, b, c, input logic sel, output logic [3:0] y);
  assign y = sel ? (a + b) : (a + c);
endmodule
