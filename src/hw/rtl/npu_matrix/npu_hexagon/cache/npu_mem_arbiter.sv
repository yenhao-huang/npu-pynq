`timescale 1ns/1ps

// Two line-request masters (port 0 = L1 I$ side, port 1 = L1 D$ side) onto
// one line-request port (the L2 CPU side), round robin, one transaction at
// a time.
//
// Line protocol (same on all three ports, shared with npu_cache and
// npu_axi4_line_bridge):
// - A request transfers on *_mem_req_valid && *_mem_req_ready; the address is
//   line aligned. One request is outstanding per requester, and the requester
//   keeps addr / we / wdata (whole line) stable from the request until its
//   response is complete.
// - A write completes with exactly one *_mem_resp_valid pulse (rdata
//   don't-care). A read completes with exactly RBEATS = LINE_BYTES*8 / BEAT_W
//   *_mem_resp_valid pulses, each carrying the next BEAT_W bits of the line
//   (beat 0 = lowest address) on *_mem_resp_rdata; beats may have gaps and
//   are always accepted (no ready).
//
// Operation: in IDLE the arbiter picks a requesting master (when both
// request, the one not granted last time) and registers the grant. In REQ the
// granted master's request is forwarded unchanged (valid, payload and ready
// are a mux on the registered grant, so the forwarded payload never switches
// while it is waiting). After the downstream handshake, WAIT routes the
// response pulses to the granted master and counts them (one for a write,
// RBEATS for a read, from the held m_mem_req_we), then returns to IDLE.
// A master therefore waits at most one transaction of the other master
// (no starvation). Responses return in request order trivially.
// The response data bus is broadcast to both masters (no mux); only the
// valid pulses are routed.
//
// Latency: one cycle from s*_mem_req_valid to m_mem_req_valid (grant
// register); responses are routed combinationally (zero added cycles).
//
// Combinational port-to-port paths (they chain across modules), each one
// LUT level (a 2:1 mux on grant_q, or an AND with registered state):
//   s*_mem_req_addr / we / wdata -> m_mem_req_addr / we / wdata;
//   m_mem_req_ready -> s*_mem_req_ready; m_mem_resp_valid ->
//   s*_mem_resp_valid; m_mem_resp_rdata -> s*_mem_resp_rdata (wires).
// No valid depends on a ready or a response in the same cycle
// (m_mem_req_valid is the registered state), and npu_cache has no
// port-to-port paths, so L1 -> arbiter -> L2 cannot form a loop. In the
// hierarchy the longest chains through it are the L2 data RAM output ->
// RESP_W beat mux -> s*_mem_resp_rdata -> L1 data RAM write data, and an L1
// request register -> m_mem_req_addr -> L2 tag / data RAM read address.
// The always_comb below writes each output exactly once per evaluation (no
// default-then-override), so an event-driven simulator never sees glitches
// on these pass-throughs; npu_cache follows the same rule (see its header:
// a glitching cache output looped forever through this module in Icarus).
//
// P0_READ_ONLY = 1 declares that master 0 never writes (the I$): its we and
// wdata are then ignored and the forwarded wdata is always master 1's, which
// removes the LINE_BYTES*8-bit write-data mux.
module npu_mem_arbiter #(
    parameter integer LINE_BYTES = 32,
    parameter integer BEAT_W = 32,       // read response beat width
    parameter integer P0_READ_ONLY = 0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    // Master 0 (I$ side)
    input  logic                    s0_mem_req_valid,
    output logic                    s0_mem_req_ready,
    input  logic [31:0]             s0_mem_req_addr,
    input  logic                    s0_mem_req_we,
    input  logic [LINE_BYTES*8-1:0] s0_mem_req_wdata,
    output logic                    s0_mem_resp_valid,
    output logic [BEAT_W-1:0]       s0_mem_resp_rdata,
    // Master 1 (D$ side)
    input  logic                    s1_mem_req_valid,
    output logic                    s1_mem_req_ready,
    input  logic [31:0]             s1_mem_req_addr,
    input  logic                    s1_mem_req_we,
    input  logic [LINE_BYTES*8-1:0] s1_mem_req_wdata,
    output logic                    s1_mem_resp_valid,
    output logic [BEAT_W-1:0]       s1_mem_resp_rdata,
    // Downstream
    output logic                    m_mem_req_valid,
    input  logic                    m_mem_req_ready,
    output logic [31:0]             m_mem_req_addr,
    output logic                    m_mem_req_we,
    output logic [LINE_BYTES*8-1:0] m_mem_req_wdata,
    input  logic                    m_mem_resp_valid,
    input  logic [BEAT_W-1:0]       m_mem_resp_rdata
);
    localparam integer RBEATS = LINE_BYTES * 8 / BEAT_W;
    localparam integer BC_W = (RBEATS > 1) ? $clog2(RBEATS) : 1;

    initial begin
        if (BEAT_W < 8 || BEAT_W > LINE_BYTES * 8 || RBEATS * BEAT_W != LINE_BYTES * 8)
            $fatal(1, "npu_mem_arbiter BEAT_W must divide the line");
    end

    localparam logic [1:0] A_IDLE = 2'd0;
    localparam logic [1:0] A_REQ  = 2'd1;
    localparam logic [1:0] A_WAIT = 2'd2;

    logic [1:0] state_q;
    logic       grant_q;    // master of the current transaction
    logic       last_q;     // master granted last
    logic [BC_W-1:0] beat_q;  // read beats received

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= A_IDLE;
            grant_q <= 1'b0;
            last_q <= 1'b1;
            beat_q <= '0;
        end else begin
            case (state_q)
                A_IDLE: begin
                    if (s0_mem_req_valid || s1_mem_req_valid) begin
                        // Both requesting: take the one not granted last time
                        if (s0_mem_req_valid && s1_mem_req_valid)
                            grant_q <= ~last_q;
                        else
                            grant_q <= s1_mem_req_valid;
                        state_q <= A_REQ;
                    end
                end
                A_REQ: begin
                    beat_q <= '0;
                    if (m_mem_req_ready) begin
                        last_q <= grant_q;
                        state_q <= A_WAIT;
                    end
                end
                A_WAIT: begin
                    if (m_mem_resp_valid) begin
                        beat_q <= beat_q + 1'b1;
                        if (m_mem_req_we || RBEATS == 1 || beat_q == BC_W'(RBEATS - 1))
                            state_q <= A_IDLE;
                    end
                end
                default: state_q <= A_IDLE;
            endcase
        end
    end

    always_comb begin
        m_mem_req_valid = (state_q == A_REQ);
        m_mem_req_addr = grant_q ? s1_mem_req_addr : s0_mem_req_addr;
        if (P0_READ_ONLY != 0) begin
            m_mem_req_we = grant_q && s1_mem_req_we;
            m_mem_req_wdata = s1_mem_req_wdata;
        end else begin
            m_mem_req_we = grant_q ? s1_mem_req_we : s0_mem_req_we;
            m_mem_req_wdata = grant_q ? s1_mem_req_wdata : s0_mem_req_wdata;
        end
        s0_mem_req_ready = (state_q == A_REQ) && !grant_q && m_mem_req_ready;
        s1_mem_req_ready = (state_q == A_REQ) && grant_q && m_mem_req_ready;
        s0_mem_resp_valid = (state_q == A_WAIT) && !grant_q && m_mem_resp_valid;
        s1_mem_resp_valid = (state_q == A_WAIT) && grant_q && m_mem_resp_valid;
        s0_mem_resp_rdata = m_mem_resp_rdata;
        s1_mem_resp_rdata = m_mem_resp_rdata;
    end

    // Master 0's write fields are unused when it is declared read-only
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_p0;
    /* verilator lint_on UNUSEDSIGNAL */
    assign unused_p0 = s0_mem_req_we ^ (^s0_mem_req_wdata);
endmodule
