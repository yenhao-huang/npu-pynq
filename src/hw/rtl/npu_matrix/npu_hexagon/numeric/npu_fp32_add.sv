`timescale 1ns/1ps

// IEEE 754 binary32 adder/subtractor, used for FP32 accumulation in the
// tensor PE's FP16 mode (npu_pe_mp).
//
// y = a + b (a - b when sub=1), round-to-nearest-even, subnormal inputs and
// outputs, overflow to +-Inf, canonical NaN 0x7FC00000 for every NaN result
// (including Inf - Inf), exact zero is +0 except (-0) + (-0) = -0.
// Fixed latency 3 (out_valid = in_valid delayed by 3 clocks), fully
// pipelined: one operation per clock. See npu_fp_add_core for the pipeline.
// SUBNORMALS = 0 (npu_pe_mp only) drops gradual underflow for callers whose
// operands are never subnormal and whose sums never underflow; see
// npu_fp_add_core.
//
// Timing note (Yosys netlist, rough 7-series -1 model): the full-IEEE
// configuration (SUBNORMALS = 1) has its worst path in stage 2 (27-bit add,
// leading-zero count, then the compare that limits the normalisation shift
// at the subnormal exponent), ~9.2 ns. A one-hot limit marker decoded in
// stage 1 and ORed into the leading-zero count removes the compare (~7.5 ns)
// but costs ~+100 LUT and +27 FF here (+20%). Not applied: nothing in the
// design instantiates SUBNORMALS = 1 (npu_pe_mp uses SUBNORMALS = 0, ~7 ns),
// and 9.2 ns meets 100 MHz; revisit if a full-IEEE FP32 adder is ever placed
// on a critical path. With SUBNORMALS = 0 the core defers the last two
// alignment shift places to stage 2 (see npu_fp_add_core): ~8.2 ns (stage 1)
// and ~7.9 ns (stage 2) behind npu_systolic_array_mp's shared-adder operand
// select, ~6.5 / 7.0 ns stand-alone.
module npu_fp32_add #(
    parameter integer SUBNORMALS = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic        sub,
    output logic        out_valid,
    output logic [31:0] y
);
    npu_fp_add_core #(
        .EW(8),
        .MW(23),
        .SUBNORMALS(SUBNORMALS)
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
