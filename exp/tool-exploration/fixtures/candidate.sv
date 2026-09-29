module dut(input logic [3:0] a, b, c, input logic sel, output logic [3:0] y);
  assign y = a + (sel ? b : c);
endmodule
