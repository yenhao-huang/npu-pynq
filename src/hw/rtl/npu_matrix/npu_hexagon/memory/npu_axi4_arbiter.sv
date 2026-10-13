`timescale 1ns/1ps

// Two-master AXI4 arbiter: master 0 (cache line bridge) and master 1 (TCM
// DMA) share one AXI4 master port. 32-bit addresses, 64-bit data, a single
// ID (no ID signals), so the downstream returns R bursts in AR order and B
// responses in AW order.
//
// Ports: s0_axi_* (slave port 0), s1_axi_* (slave port 1) and m_axi_* (master
// port) use the AXI signal names of the masters they connect to
// (npu_axi4_line_bridge and npu_tcm_dma both use m_axi_*), so integration is
// a name-for-name hookup. (Renamed from s0_*/s1_*/m_* in review round 2,
// before any instantiation.)
//
// Read path: round-robin AR arbitration into a registered AR slice (the AR
// address/control seen downstream is a register and stays stable while
// m_axi_arvalid waits for m_axi_arready). The slice loads only while it is
// empty, so s*_axi_arready depends on registers and the masters' own ARVALIDs
// but never on m_axi_arready: no combinational path crosses the arbiter from
// the downstream AR ready to a master's burst-issue logic, and m_axi_arready
// drives a single flop. The granted master's index is pushed into an
// in-order tag FIFO (TAG_DEPTH entries); R beats go to the master at the FIFO
// head, which is popped on the RLAST handshake. A full tag FIFO stalls AR,
// bounding the number of outstanding read bursts to TAG_DEPTH.
//
// Write path, independent of the read path: round-robin AW arbitration into
// a registered AW slice that, like the AR slice, loads only while empty
// (s*_axi_awready does not depend on m_axi_awready). The granted index is
// pushed into a W-order FIFO and a B-order FIFO. The W channel is locked to
// the master at the W-order FIFO head until its WLAST handshake; B responses
// go to the master at the B-order FIFO head. W of the granted master may
// reach the downstream before its AW handshake (allowed by AXI; the data
// still follows AW order). Deadlock freedom: m_axi_awvalid never waits for
// W, and W of a burst is routed as soon as its AW is in the slice, so a
// downstream that waits for AW before WREADY and one that waits for W before
// AWREADY both make progress. Masters must follow the AXI rule that AWVALID
// does not wait for a W handshake (W before AW is fine: it is held until
// that master's AW is granted).
//
// Round-robin: when both masters request in the same cycle, the master that
// was not granted last wins; a lone requester is granted as soon as the
// slice is empty and the tag FIFO(s) have room.
// Latency: one cycle from s*_axi_arvalid/awvalid to m_axi_arvalid/awvalid
// (slice empty); R, W and B pass through combinationally (no added latency).
// Throughput: one AR and one AW every two cycles (a slice is free again the
// cycle after its handshake). Data throughput is unaffected for bursts of two
// or more beats: the line bridge issues one 4-beat burst at a time and the
// DMA issues up to 16-beat bursts with up to 4 in flight; only a stream of
// single-beat bursts would be limited to one beat every two cycles. (Changed
// in review round 3 from a slice that also loaded during its handshake, which
// put m_axi_arready/awready combinationally on s*_axi_arready/awready and on
// the payload register enables.)
//
// Combinational port-to-port paths (they chain across modules in the
// subsystem), each one LUT level from the input (a 2:1 mux, or an AND with
// registered slice/tag-FIFO state):
//   s0/s1_axi_arvalid -> s0/s1_axi_arready, s0/s1_axi_awvalid ->
//   s0/s1_axi_awready (the grant; AXI allows ready to depend on valid);
//   m_axi_rvalid/rdata/rresp/rlast -> s*_axi_r*; s*_axi_rready ->
//   m_axi_rready; s*_axi_wvalid/wdata/wstrb/wlast -> m_axi_w*; m_axi_wready
//   -> s*_axi_wready; m_axi_bvalid/bresp -> s*_axi_b*; s*_axi_bready ->
//   m_axi_bready.
// m_axi_arready and m_axi_awready reach no output.
//
// Reset: only valid/priority/pointer flops are reset; the AR/AW payload
// registers and the tag storage are not (X until first use, always
// qualified by m_axi_arvalid/awvalid or by a non-empty tag FIFO).
module npu_axi4_arbiter #(
    parameter integer TAG_DEPTH = 4    // power of two, >= 2
) (
    input  logic        clk,
    input  logic        rst_n,
    // Slave port 0 (cache line bridge)
    input  logic [31:0] s0_axi_araddr,
    input  logic [7:0]  s0_axi_arlen,
    input  logic [2:0]  s0_axi_arsize,
    input  logic [1:0]  s0_axi_arburst,
    input  logic [3:0]  s0_axi_arcache,
    input  logic [2:0]  s0_axi_arprot,
    input  logic        s0_axi_arvalid,
    output logic        s0_axi_arready,
    output logic [63:0] s0_axi_rdata,
    output logic [1:0]  s0_axi_rresp,
    output logic        s0_axi_rlast,
    output logic        s0_axi_rvalid,
    input  logic        s0_axi_rready,
    input  logic [31:0] s0_axi_awaddr,
    input  logic [7:0]  s0_axi_awlen,
    input  logic [2:0]  s0_axi_awsize,
    input  logic [1:0]  s0_axi_awburst,
    input  logic [3:0]  s0_axi_awcache,
    input  logic [2:0]  s0_axi_awprot,
    input  logic        s0_axi_awvalid,
    output logic        s0_axi_awready,
    input  logic [63:0] s0_axi_wdata,
    input  logic [7:0]  s0_axi_wstrb,
    input  logic        s0_axi_wlast,
    input  logic        s0_axi_wvalid,
    output logic        s0_axi_wready,
    output logic [1:0]  s0_axi_bresp,
    output logic        s0_axi_bvalid,
    input  logic        s0_axi_bready,
    // Slave port 1 (TCM DMA)
    input  logic [31:0] s1_axi_araddr,
    input  logic [7:0]  s1_axi_arlen,
    input  logic [2:0]  s1_axi_arsize,
    input  logic [1:0]  s1_axi_arburst,
    input  logic [3:0]  s1_axi_arcache,
    input  logic [2:0]  s1_axi_arprot,
    input  logic        s1_axi_arvalid,
    output logic        s1_axi_arready,
    output logic [63:0] s1_axi_rdata,
    output logic [1:0]  s1_axi_rresp,
    output logic        s1_axi_rlast,
    output logic        s1_axi_rvalid,
    input  logic        s1_axi_rready,
    input  logic [31:0] s1_axi_awaddr,
    input  logic [7:0]  s1_axi_awlen,
    input  logic [2:0]  s1_axi_awsize,
    input  logic [1:0]  s1_axi_awburst,
    input  logic [3:0]  s1_axi_awcache,
    input  logic [2:0]  s1_axi_awprot,
    input  logic        s1_axi_awvalid,
    output logic        s1_axi_awready,
    input  logic [63:0] s1_axi_wdata,
    input  logic [7:0]  s1_axi_wstrb,
    input  logic        s1_axi_wlast,
    input  logic        s1_axi_wvalid,
    output logic        s1_axi_wready,
    output logic [1:0]  s1_axi_bresp,
    output logic        s1_axi_bvalid,
    input  logic        s1_axi_bready,
    // Master port
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic [3:0]  m_axi_arcache,
    output logic [2:0]  m_axi_arprot,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic [3:0]  m_axi_awcache,
    output logic [2:0]  m_axi_awprot,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready
);
    // ------------------------------------------------------------ read path
    logic ar_prio, ar_sel, ar_load;
    logic rq_full, rq_empty, rq_head;
    logic r_pop;

    always_comb begin
        // Grant: the other master than last time when both request.
        ar_sel = (s0_axi_arvalid && s1_axi_arvalid) ? ar_prio : s1_axi_arvalid;
        // Load only into an empty slice (no m_axi_arready in the ready path).
        ar_load = (s0_axi_arvalid || s1_axi_arvalid) && !rq_full && !m_axi_arvalid;
        s0_axi_arready = ar_load && !ar_sel;
        s1_axi_arready = ar_load && ar_sel;

        m_axi_rready = !rq_empty && (rq_head ? s1_axi_rready : s0_axi_rready);
        r_pop = m_axi_rvalid && m_axi_rready && m_axi_rlast;
        s0_axi_rvalid = m_axi_rvalid && !rq_empty && !rq_head;
        s1_axi_rvalid = m_axi_rvalid && !rq_empty && rq_head;
        s0_axi_rdata = m_axi_rdata;
        s1_axi_rdata = m_axi_rdata;
        s0_axi_rresp = m_axi_rresp;
        s1_axi_rresp = m_axi_rresp;
        s0_axi_rlast = m_axi_rlast;
        s1_axi_rlast = m_axi_rlast;
    end

    npu_axi4_arbiter_tagq #(.DEPTH(TAG_DEPTH)) read_tags (
        .clk(clk), .rst_n(rst_n),
        .push(ar_load), .push_tag(ar_sel),
        .pop(r_pop), .head(rq_head), .empty(rq_empty), .full(rq_full)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_prio <= 1'b0;
            m_axi_arvalid <= 1'b0;
        end else begin
            if (ar_load) begin
                ar_prio <= !ar_sel;
                m_axi_arvalid <= 1'b1;
            end else if (m_axi_arready) begin
                m_axi_arvalid <= 1'b0;
            end
        end
    end

    // AR payload: qualified by m_axi_arvalid, so no reset (saves the reset
    // net and per-flop inverters; X until the first grant).
    always_ff @(posedge clk) begin
        if (ar_load) begin
            m_axi_araddr <= ar_sel ? s1_axi_araddr : s0_axi_araddr;
            m_axi_arlen <= ar_sel ? s1_axi_arlen : s0_axi_arlen;
            m_axi_arsize <= ar_sel ? s1_axi_arsize : s0_axi_arsize;
            m_axi_arburst <= ar_sel ? s1_axi_arburst : s0_axi_arburst;
            m_axi_arcache <= ar_sel ? s1_axi_arcache : s0_axi_arcache;
            m_axi_arprot <= ar_sel ? s1_axi_arprot : s0_axi_arprot;
        end
    end

    // ----------------------------------------------------------- write path
    logic aw_prio, aw_sel, aw_load;
    logic wq_full, wq_empty, wq_head;
    logic bq_full, bq_empty, bq_head;
    logic w_pop, b_pop;

    always_comb begin
        aw_sel = (s0_axi_awvalid && s1_axi_awvalid) ? aw_prio : s1_axi_awvalid;
        aw_load = (s0_axi_awvalid || s1_axi_awvalid) && !wq_full && !bq_full &&
                  !m_axi_awvalid;
        s0_axi_awready = aw_load && !aw_sel;
        s1_axi_awready = aw_load && aw_sel;

        // W locked to the master at the W-order head until WLAST.
        m_axi_wvalid = !wq_empty && (wq_head ? s1_axi_wvalid : s0_axi_wvalid);
        m_axi_wdata = wq_head ? s1_axi_wdata : s0_axi_wdata;
        m_axi_wstrb = wq_head ? s1_axi_wstrb : s0_axi_wstrb;
        m_axi_wlast = wq_head ? s1_axi_wlast : s0_axi_wlast;
        s0_axi_wready = !wq_empty && !wq_head && m_axi_wready;
        s1_axi_wready = !wq_empty && wq_head && m_axi_wready;
        w_pop = m_axi_wvalid && m_axi_wready && m_axi_wlast;

        m_axi_bready = !bq_empty && (bq_head ? s1_axi_bready : s0_axi_bready);
        s0_axi_bvalid = m_axi_bvalid && !bq_empty && !bq_head;
        s1_axi_bvalid = m_axi_bvalid && !bq_empty && bq_head;
        s0_axi_bresp = m_axi_bresp;
        s1_axi_bresp = m_axi_bresp;
        b_pop = m_axi_bvalid && m_axi_bready;
    end

    npu_axi4_arbiter_tagq #(.DEPTH(TAG_DEPTH)) w_tags (
        .clk(clk), .rst_n(rst_n),
        .push(aw_load), .push_tag(aw_sel),
        .pop(w_pop), .head(wq_head), .empty(wq_empty), .full(wq_full)
    );

    npu_axi4_arbiter_tagq #(.DEPTH(TAG_DEPTH)) b_tags (
        .clk(clk), .rst_n(rst_n),
        .push(aw_load), .push_tag(aw_sel),
        .pop(b_pop), .head(bq_head), .empty(bq_empty), .full(bq_full)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_prio <= 1'b0;
            m_axi_awvalid <= 1'b0;
        end else begin
            if (aw_load) begin
                aw_prio <= !aw_sel;
                m_axi_awvalid <= 1'b1;
            end else if (m_axi_awready) begin
                m_axi_awvalid <= 1'b0;
            end
        end
    end

    // AW payload: qualified by m_axi_awvalid, so no reset (saves the reset
    // net and per-flop inverters; X until the first grant).
    always_ff @(posedge clk) begin
        if (aw_load) begin
            m_axi_awaddr <= aw_sel ? s1_axi_awaddr : s0_axi_awaddr;
            m_axi_awlen <= aw_sel ? s1_axi_awlen : s0_axi_awlen;
            m_axi_awsize <= aw_sel ? s1_axi_awsize : s0_axi_awsize;
            m_axi_awburst <= aw_sel ? s1_axi_awburst : s0_axi_awburst;
            m_axi_awcache <= aw_sel ? s1_axi_awcache : s0_axi_awcache;
            m_axi_awprot <= aw_sel ? s1_axi_awprot : s0_axi_awprot;
        end
    end
endmodule

// One-bit in-order tag FIFO (registers). head is valid when !empty (X after
// reset until the first push: consumers must qualify it with !empty).
module npu_axi4_arbiter_tagq #(
    parameter integer DEPTH = 4
) (
    input  logic clk,
    input  logic rst_n,
    input  logic push,
    input  logic push_tag,
    input  logic pop,
    output logic head,
    output logic empty,
    output logic full
);
    localparam integer PW = $clog2(DEPTH);

    logic [DEPTH-1:0] tags;
    logic [PW:0] wp, rp;

    initial begin
        if (DEPTH < 2 || (1 << PW) != DEPTH)
            $fatal(1, "npu_axi4_arbiter TAG_DEPTH must be a power of two >= 2");
    end

    assign empty = (wp == rp);
    assign full = (wp[PW] != rp[PW]) && (wp[PW-1:0] == rp[PW-1:0]);
    assign head = tags[rp[PW-1:0]];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wp <= '0;
            rp <= '0;
        end else begin
            if (push && !full)
                wp <= wp + 1'b1;
            if (pop && !empty)
                rp <= rp + 1'b1;
        end
    end

    // Tag storage: only read at the head while !empty, so no reset. Explicit
    // per-entry write enables (an indexed bit write made Yosys build a
    // 32-bit shifter with carry chains).
    always_ff @(posedge clk) begin
        for (int i = 0; i < DEPTH; i++) begin
            if (push && !full && wp[PW-1:0] == PW'(i))
                tags[i] <= push_tag;
        end
    end
endmodule
