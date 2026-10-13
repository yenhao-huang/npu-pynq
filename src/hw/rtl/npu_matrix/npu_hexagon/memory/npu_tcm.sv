`timescale 1ns/1ps

// Tightly coupled memory (TCM): a true dual-port scratchpad of DEPTH words of
// WIDTH bits, shared by the vector unit, the scalar core and the TCM DMA.
//
// Contract, identical for port A and port B:
// - gnt = req: every request is granted in the cycle it is made.
// - addr must be below DEPTH (only relevant when DEPTH is not a power of
//   two).
// - Write: req && we stores the bytes of wdata whose wstrb bit is set.
// - Read: rdata holds mem[addr] exactly one cycle after a request. Every
//   request (read or write) updates rdata; rdata holds its value while req
//   is low.
// - Read-during-write on the same port is READ_FIRST: rdata returns the word
//   as it was before the write (all bytes, regardless of wstrb).
// - Cross-port, same address, same cycle:
//   * write on one port, read on the other: the reading port returns the old
//     word (READ_FIRST, common clock; this is what the RTL simulates and what
//     7-series BRAM in READ_FIRST mode returns with a common clock). Clients
//     should still avoid relying on it.
//   * write on both ports: UNDEFINED (the memory contents of that word are
//     unpredictable on hardware). Clients must avoid it.
//
// Implementation follows the Vivado UG901 "byte-write true dual-port,
// read-first" template (one clocked process per port, per-byte write enables,
// no reset and no output register), so Vivado infers RAMB36E1/RAMB18E1 in TDP
// mode with byte-wide write enables (1024 x 64 = 2 x RAMB36). rst_n is part of
// the uniform module interface but is intentionally unused: an asynchronous
// reset on rdata would prevent block-RAM inference.
module npu_tcm #(
    parameter integer WIDTH = 64,
    parameter integer DEPTH = 1024,
    parameter integer AW = (DEPTH > 1) ? $clog2(DEPTH) : 1
) (
    input  logic               clk,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic               rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    // Port A
    input  logic               a_req,
    input  logic               a_we,
    input  logic [AW-1:0]      a_addr,
    input  logic [WIDTH-1:0]   a_wdata,
    input  logic [WIDTH/8-1:0] a_wstrb,
    output logic               a_gnt,
    output logic [WIDTH-1:0]   a_rdata,
    // Port B
    input  logic               b_req,
    input  logic               b_we,
    input  logic [AW-1:0]      b_addr,
    input  logic [WIDTH-1:0]   b_wdata,
    input  logic [WIDTH/8-1:0] b_wstrb,
    output logic               b_gnt,
    output logic [WIDTH-1:0]   b_rdata
);
    localparam integer NB = WIDTH / 8;

    (* ram_style = "block" *) logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (WIDTH <= 0 || (WIDTH % 8) != 0 || DEPTH <= 1)
            $fatal(1, "npu_tcm requires WIDTH a multiple of 8 and DEPTH > 1");
        if (AW < $clog2(DEPTH))
            $fatal(1, "npu_tcm AW must address all DEPTH words");
    end

    assign a_gnt = a_req;
    assign b_gnt = b_req;

    // Port A: byte writes, read-first.
    always @(posedge clk) begin
        if (a_req) begin
            if (a_we) begin
                for (int i = 0; i < NB; i++) begin
                    if (a_wstrb[i])
                        mem[a_addr][i*8 +: 8] <= a_wdata[i*8 +: 8];
                end
            end
            a_rdata <= mem[a_addr];
        end
    end

    // Port B: byte writes, read-first.
    always @(posedge clk) begin
        if (b_req) begin
            if (b_we) begin
                for (int i = 0; i < NB; i++) begin
                    if (b_wstrb[i])
                        mem[b_addr][i*8 +: 8] <= b_wdata[i*8 +: 8];
                end
            end
            b_rdata <= mem[b_addr];
        end
    end
endmodule
