`timescale 1ns/1ps

// IEEE 754 binary16 adder/subtractor for the vector unit.
//
// y = a + b (a - b when sub=1), round-to-nearest-even, subnormal inputs and
// outputs, overflow to +-Inf, canonical NaN 0x7E00 for every NaN result
// (including Inf - Inf), exact zero is +0 except (-0) + (-0) = -0.
// Fixed latency LAT_ADD = 3 (out_valid = in_valid delayed by 3 clocks), fully
// pipelined: one operation per clock. See npu_fp_add_core for the pipeline.
module npu_fp16_add (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic        sub,
    output logic        out_valid,
    output logic [15:0] y
);
    npu_fp_add_core #(
        .EW(5),
        .MW(10)
    ) core (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .a(a),
        .b(b),
        .sub(sub),
        .out_valid(out_valid),
        .y(y)
    );
endmodule
