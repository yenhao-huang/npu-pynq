`timescale 1ns/1ps
module tb_counter;
    logic clk = 0;
    logic rst_n;
    logic enable;
    logic [7:0] ref_count;
    logic [7:0] dut_count;

    counter_ref    u_ref (.clk(clk), .rst_n(rst_n), .enable(enable), .count(ref_count));
    counter_broken u_dut (.clk(clk), .rst_n(rst_n), .enable(enable), .count(dut_count));

    always #5 clk = ~clk;

    int mismatches = 0;

    initial begin
        rst_n  = 0;
        enable = 0;
        @(posedge clk);
        @(posedge clk);
        rst_n  = 1;
        enable = 1;
        repeat (20) begin
            @(posedge clk);
            #1;
            if (ref_count !== dut_count) mismatches++;
        end
        if (mismatches == 0) $display("PASS tb_counter");
        else                 $display("FAIL tb_counter: %0d mismatching cycles", mismatches);
        $finish;
    end
endmodule
