`timescale 1ns/1ps

module tb_npu_pe;
    logic clk;
    logic rst_n;
    logic clear;
    logic enable;
    logic signed [7:0] a_in;
    logic signed [7:0] b_in;
    logic a_valid_in;
    logic b_valid_in;
    logic signed [7:0] a_out;
    logic signed [7:0] b_out;
    logic a_valid_out;
    logic b_valid_out;
    logic signed [31:0] accumulator;

    npu_pe dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_in(a_in),
        .b_in(b_in),
        .a_valid_in(a_valid_in),
        .b_valid_in(b_valid_in),
        .a_out(a_out),
        .b_out(b_out),
        .a_valid_out(a_valid_out),
        .b_valid_out(b_valid_out),
        .accumulator(accumulator)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task automatic step(
        input logic signed [7:0] next_a,
        input logic signed [7:0] next_b,
        input logic next_a_valid,
        input logic next_b_valid,
        input logic next_enable,
        input logic next_clear
    );
        begin
            @(negedge clk);
            a_in = next_a;
            b_in = next_b;
            a_valid_in = next_a_valid;
            b_valid_in = next_b_valid;
            enable = next_enable;
            clear = next_clear;
            @(posedge clk);
            #1;
        end
    endtask

    task automatic check(input logic condition, input string message);
        begin
            if (!condition) begin
                $display("FAIL tb_npu_pe: %s", message);
                $fatal(1);
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        enable = 1'b0;
        a_in = '0;
        b_in = '0;
        a_valid_in = 1'b0;
        b_valid_in = 1'b0;

        #1;
        check(accumulator === 32'sd0, "asynchronous reset clears accumulator");
        check(a_out === 8'sd0 && b_out === 8'sd0,
              "asynchronous reset clears forwarded operands");
        check(!a_valid_out && !b_valid_out,
              "asynchronous reset clears forwarded valids");

        @(negedge clk);
        rst_n = 1'b1;

        step(-8'sd128, 8'sd127, 1'b1, 1'b1, 1'b1, 1'b0);
        check(accumulator === -32'sd16256, "signed endpoint MAC");
        check(a_out === -8'sd128 && b_out === 8'sd127,
              "signed operands forward exactly");
        check(a_valid_out && b_valid_out, "both valids forward");

        step(8'sd5, 8'sd6, 1'b1, 1'b0, 1'b1, 1'b0);
        check(accumulator === -32'sd16256,
              "one invalid operand does not accumulate");
        check(a_out === 8'sd5 && b_out === 8'sd6,
              "independent-valid operands still forward");
        check(a_valid_out && !b_valid_out, "valids forward independently");

        step(8'sd99, -8'sd77, 1'b1, 1'b1, 1'b0, 1'b0);
        check(accumulator === -32'sd16256, "stall holds accumulator");
        check(a_out === 8'sd5 && b_out === 8'sd6,
              "stall holds forwarded operands");
        check(a_valid_out && !b_valid_out, "stall holds forwarded valids");

        step(8'sd1, 8'sd1, 1'b1, 1'b1, 1'b0, 1'b1);
        check(accumulator === 32'sd0, "clear wins over disabled enable");
        check(a_out === 8'sd0 && b_out === 8'sd0,
              "clear zeros forwarded operands");
        check(!a_valid_out && !b_valid_out, "clear zeros forwarded valids");

        @(negedge clk);
        dut.accumulator = 32'sh7ffffffe;
        a_in = 8'sd1;
        b_in = 8'sd2;
        a_valid_in = 1'b1;
        b_valid_in = 1'b1;
        enable = 1'b1;
        clear = 1'b0;
        @(posedge clk);
        #1;
        check(accumulator === 32'sh7fffffff, "positive overflow saturates");

        @(negedge clk);
        dut.accumulator = -32'sd2147483647;
        a_in = -8'sd1;
        b_in = 8'sd2;
        a_valid_in = 1'b1;
        b_valid_in = 1'b1;
        enable = 1'b1;
        clear = 1'b0;
        @(posedge clk);
        #1;
        check(accumulator === -32'sd2147483648, "negative overflow saturates");

        step(8'sd3, 8'sd4, 1'b1, 1'b1, 1'b1, 1'b0);
        #1;
        rst_n = 1'b0;
        #1;
        check(accumulator === 32'sd0, "mid-job asynchronous reset");
        check(a_out === 8'sd0 && b_out === 8'sd0,
              "mid-job reset clears pipeline");
        check(!a_valid_out && !b_valid_out,
              "mid-job reset clears valid pipeline");

        $display("PASS tb_npu_pe");
        $finish;
    end
endmodule
