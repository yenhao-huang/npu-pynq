`timescale 1ns/1ps

// Line-request port (L2 memory side) -> AXI4 master, 64-bit data.
//
// Line protocol on the slave side (shared with npu_cache and
// npu_mem_arbiter), with 64-bit read beats:
// - A request transfers on mem_req_valid && mem_req_ready; mem_req_ready is
//   high only while the bridge is idle (one transaction at a time). The
//   requester keeps mem_req_addr, mem_req_we and mem_req_wdata (whole line)
//   stable from the request until its response is complete: the bridge reads
//   the address directly from this input instead of registering it, and
//   loads each write beat from mem_req_wdata into the W data register.
// - A read completes with exactly BEATS = LINE_BYTES/8 mem_resp_valid pulses,
//   one per AXI R beat, each carrying that beat on mem_resp_rdata[63:0]
//   (beat 0 = lowest address) in its cycle. Both are registers (one cycle
//   after the R handshake); beats have gaps when the slave's R has gaps.
// - A write completes with exactly one mem_resp_valid pulse (a register),
//   one cycle after the AXI B handshake; mem_resp_rdata is don't-care.
//
// AXI4: one request = one INCR burst of BEATS beats,
// arsize/awsize = 3 (8 bytes), arcache/awcache = 4'b0011, prot 0, single ID
// (no ID signals; the interconnect uses ID 0). The address is forced line
// aligned, so a burst never crosses a 4 KB boundary. Reads: AR, then the R
// beats are forwarded; rready is high for the whole transaction. Writes: AW
// and the W beats are issued together (W may lead AW, as AXI allows), wstrb
// all ones, wlast on the last beat; bready rises once AW and all W beats are
// done. A read completes on the R beat with rlast. m_axi_wdata is a register:
// beat 0 is loaded at the request handshake and beat k+1 at the handshake of
// beat k, so W leaves the bridge with the same timing as an unregistered mux
// (wvalid the cycle after acceptance) while the L2 data RAM output (the
// requester's mem_req_wdata) ends at this register instead of running through
// the AXI arbiter and interconnect into the PS HP port.
//
// Combinational port-to-port paths (they chain across modules): only
// mem_req_addr -> m_axi_araddr / m_axi_awaddr (wires; the AXI arbiter
// registers AR/AW in its slices). Every other output is a register or
// decodes registers only (no AXI ready or valid input reaches an output in
// the same cycle).
//
// error: one-cycle pulse (a register) for every R beat with RRESP != OKAY
// and every B response with BRESP != OKAY, in the same cycle as the
// mem_resp_valid pulse of that beat / response. The owner of the status
// register keeps the sticky bit (npu_hx_regs STATUS.bus_error, cleared by
// CONTROL.CLEAR); a flag that only reset clears could not implement that.
// The transaction still completes normally (read data is whatever the slave
// returned), so the cache side never hangs.
//
// Latency with a zero-wait slave (ready always high, R/B one cycle after the
// address/last W): request acceptance to the last read beat = BEATS + 2,
// write = BEATS + 2 cycles (6 for 32-byte lines); the first read beat
// arrives 3 cycles after acceptance.
module npu_axi4_line_bridge #(
    parameter integer LINE_BYTES = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    // Line-request slave port
    input  logic                    mem_req_valid,
    output logic                    mem_req_ready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0]             mem_req_addr,   // low OFF_W bits ignored
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                    mem_req_we,
    input  logic [LINE_BYTES*8-1:0] mem_req_wdata,
    output logic                    mem_resp_valid,
    output logic [63:0]             mem_resp_rdata,
    output logic                    error,
    // AXI4 master, read channels
    output logic [31:0]             m_axi_araddr,
    output logic [7:0]              m_axi_arlen,
    output logic [2:0]              m_axi_arsize,
    output logic [1:0]              m_axi_arburst,
    output logic [3:0]              m_axi_arcache,
    output logic [2:0]              m_axi_arprot,
    output logic                    m_axi_arvalid,
    input  logic                    m_axi_arready,
    input  logic [63:0]             m_axi_rdata,
    input  logic [1:0]              m_axi_rresp,
    input  logic                    m_axi_rlast,
    input  logic                    m_axi_rvalid,
    output logic                    m_axi_rready,
    // AXI4 master, write channels
    output logic [31:0]             m_axi_awaddr,
    output logic [7:0]              m_axi_awlen,
    output logic [2:0]              m_axi_awsize,
    output logic [1:0]              m_axi_awburst,
    output logic [3:0]              m_axi_awcache,
    output logic [2:0]              m_axi_awprot,
    output logic                    m_axi_awvalid,
    input  logic                    m_axi_awready,
    output logic [63:0]             m_axi_wdata,
    output logic [7:0]              m_axi_wstrb,
    output logic                    m_axi_wlast,
    output logic                    m_axi_wvalid,
    input  logic                    m_axi_wready,
    input  logic [1:0]              m_axi_bresp,
    input  logic                    m_axi_bvalid,
    output logic                    m_axi_bready
);
    localparam integer BEATS = LINE_BYTES / 8;
    localparam integer OFF_W = $clog2(LINE_BYTES);
    localparam integer BW = (BEATS > 1) ? $clog2(BEATS) : 1;

    localparam logic [1:0] B_IDLE = 2'd0;
    localparam logic [1:0] B_RD   = 2'd1;
    localparam logic [1:0] B_WR   = 2'd2;

    initial begin
        if (BEATS < 1 || BEATS > 256 || (1 << OFF_W) != LINE_BYTES || LINE_BYTES < 8)
            $fatal(1, "npu_axi4_line_bridge LINE_BYTES must be a power of two, 8..2048");
    end

    logic [1:0]        state_q;
    logic              ax_pend_q;     // AR or AW not yet accepted
    logic              w_pend_q;      // W beats remaining
    logic [BW-1:0]     beat_q;        // next write beat to load (0 when idle)
    logic [BW-1:0]     beat_inc;
    logic [63:0]       rbeat_q;
    logic [63:0]       wbeat_q;       // W data register
    logic              wload;
    logic [31:0]       line_addr;
    logic              r_hs, w_hs, b_hs;

    assign line_addr = {mem_req_addr[31:OFF_W], {OFF_W{1'b0}}};
    assign r_hs = m_axi_rvalid && m_axi_rready;
    assign w_hs = m_axi_wvalid && m_axi_wready;
    assign b_hs = m_axi_bvalid && m_axi_bready;
    assign beat_inc = (BEATS == 1) ? '0 : BW'(beat_q + 1'b1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= B_IDLE;
            ax_pend_q <= 1'b0;
            w_pend_q <= 1'b0;
            beat_q <= '0;
            mem_resp_valid <= 1'b0;
            error <= 1'b0;
        end else begin
            mem_resp_valid <= 1'b0;
            error <= 1'b0;
            case (state_q)
                B_IDLE: begin
                    if (mem_req_valid) begin
                        ax_pend_q <= 1'b1;
                        w_pend_q <= mem_req_we;
                        if (mem_req_we)
                            beat_q <= beat_inc;            // beat 0 loaded now
                        state_q <= mem_req_we ? B_WR : B_RD;
                    end
                end
                B_RD: begin
                    if (m_axi_arready)
                        ax_pend_q <= 1'b0;
                    if (r_hs) begin
                        mem_resp_valid <= 1'b1;
                        if (m_axi_rresp != 2'b00)
                            error <= 1'b1;
                        if (m_axi_rlast)
                            state_q <= B_IDLE;
                    end
                end
                B_WR: begin
                    if (m_axi_awready)
                        ax_pend_q <= 1'b0;
                    if (w_hs) begin
                        // beat_q wrapped to 0 once the last beat was loaded:
                        // this handshake sent it (beat_q stays 0 for idle)
                        if (beat_q == '0)
                            w_pend_q <= 1'b0;
                        else
                            beat_q <= beat_inc;
                    end
                    if (b_hs) begin
                        if (m_axi_bresp != 2'b00)
                            error <= 1'b1;
                        mem_resp_valid <= 1'b1;
                        state_q <= B_IDLE;
                    end
                end
                default: state_q <= B_IDLE;
            endcase
        end
    end

    // Read beat register (no reset: qualified by mem_resp_valid)
    always_ff @(posedge clk) begin
        if (r_hs)
            rbeat_q <= m_axi_rdata;
    end
    assign mem_resp_rdata = rbeat_q;

    // W data register (no reset: qualified by m_axi_wvalid). Loads beat
    // beat_q (a register, so the line-to-beat mux is one LUT level): beat 0 at
    // a write request handshake, then the next beat at each W handshake (the
    // load at the last handshake is harmless: wvalid is low by then).
    assign wload = (state_q == B_IDLE) ? (mem_req_valid && mem_req_we) : w_hs;
    always_ff @(posedge clk) begin
        if (wload)
            wbeat_q <= mem_req_wdata[beat_q*64 +: 64];
    end

    always_comb begin
        mem_req_ready = (state_q == B_IDLE);

        m_axi_araddr = line_addr;
        m_axi_arlen = 8'(BEATS - 1);
        m_axi_arsize = 3'd3;
        m_axi_arburst = 2'b01;
        m_axi_arcache = 4'b0011;
        m_axi_arprot = 3'b000;
        m_axi_arvalid = (state_q == B_RD) && ax_pend_q;
        m_axi_rready = (state_q == B_RD);

        m_axi_awaddr = line_addr;
        m_axi_awlen = 8'(BEATS - 1);
        m_axi_awsize = 3'd3;
        m_axi_awburst = 2'b01;
        m_axi_awcache = 4'b0011;
        m_axi_awprot = 3'b000;
        m_axi_awvalid = (state_q == B_WR) && ax_pend_q;
        m_axi_wdata = wbeat_q;
        m_axi_wstrb = 8'hFF;
        m_axi_wlast = (beat_q == '0);           // the loaded beat is the last
        m_axi_wvalid = (state_q == B_WR) && w_pend_q;
        m_axi_bready = (state_q == B_WR) && !ax_pend_q && !w_pend_q;
    end
endmodule
