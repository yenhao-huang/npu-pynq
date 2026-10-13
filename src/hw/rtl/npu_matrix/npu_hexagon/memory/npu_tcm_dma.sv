`timescale 1ns/1ps

// TCM DMA: block copies between DDR (AXI4 master, 64-bit data) and one TCM
// port. One TCM word (64 bits) is one AXI beat.
//
// Descriptor, sampled when start is high and busy is low (start while busy
// is ignored):
//   dir      0 = DDR -> TCM, 1 = TCM -> DDR
//   ddr_addr byte address, must be 8-byte aligned
//   tcm_addr TCM word address (TCM_AW bits; an integrator that feeds it
//            from a wider register, e.g. DMA_TCM_ADDR, must flag set upper
//            bits itself, since truncation would pass the range check)
//   words    number of 64-bit words; 0 = nothing to do
// Status:
//   busy     high while a transfer is in progress (registered)
//   done     one-cycle pulse; exactly one pulse per accepted start, including
//            words = 0 and rejected descriptors (both pulse the cycle after
//            start without raising busy)
//   error    sticky until the next accepted start. Set by a misaligned
//            ddr_addr or tcm_addr + words > TCM_DEPTH (nothing is transferred),
//            or by RRESP/BRESP != OKAY. After a response error no new burst is
//            issued; bursts already issued are completed (R beats drained, W
//            beats sent, B collected) before done. The drained R beats are
//            still written to the TCM, so after an error the destination
//            range holds undefined data; nothing outside the range is written.
//
// AXI4: INCR bursts of up to 16 beats that never cross a 4 KB boundary,
// arsize/awsize = 3, cache 4'b0011, prot 0, single ID. A shared address
// generator issues up to MAX_OUT bursts ahead of the data (AR for reads,
// AW for writes), so a zero-latency slave sees ~1 beat/cycle in both
// directions. rready and bready are high whenever the DMA can accept; W may
// be presented before its AW handshake completes (allowed by AXI), and AW
// never waits for W, so the DMA works with slaves that order AW/W either way.
//
// DDR -> TCM: each R beat is registered once and then written to the TCM
// (one cycle later, all byte strobes set). rready is low only while that
// register is full and the TCM port does not grant.
// TCM -> DDR: TCM words are prefetched into a FIFO_DEPTH-entry FIFO
// (distributed RAM); W beats stream from the FIFO head and WREADY low simply
// holds the head. A credit counter (entries + reads in flight) keeps the FIFO
// from overflowing given the TCM's one-cycle read latency.
//
// The TCM port uses the npu_tcm port protocol (req/we/addr/wdata/wstrb/gnt,
// rdata one cycle after a granted read). Its signals carry the tcm_port_
// prefix because tcm_addr is the descriptor field named by the architecture.
// gnt may be low (shared port); req never depends combinationally on gnt.
// Port sharing (architecture: port B, fixed priority DMA > scalar > host):
// req is high only when the access can complete and make progress, i.e.
// DDR -> TCM while one received R beat waits to be written, TCM -> DDR while
// words remain and a FIFO slot is free (credit). A stalled R or W channel
// therefore drops req within FIFO_DEPTH accesses, and every granted access
// moves one descriptor word, so one transfer holds the port for at most
// `words` (<= TCM_DEPTH) granted cycles; lower-priority clients wait no
// longer than that. With a slave that streams one beat per cycle the DMA
// does use the port on every cycle of the transfer.
//
// Reset: control, valid, pointer and occupancy flops have the async reset;
// pure datapath registers (DDR/TCM addresses, remaining-word counts, AR/AW
// payload, R data, FIFO) do not. They are X until the first descriptor but
// are never observed unqualified (see the datapath always_ff below).
module npu_tcm_dma #(
    parameter integer TCM_DEPTH = 1024,
    parameter integer TCM_AW = (TCM_DEPTH > 1) ? $clog2(TCM_DEPTH) : 1,
    parameter integer FIFO_DEPTH = 8,   // power of two, >= 2 (>= 4 for 1 W beat/cycle)
    parameter integer MAX_OUT = 4       // bursts in flight, 1..4 (>= 2 for 1 beat/cycle)
) (
    input  logic              clk,
    input  logic              rst_n,
    // Descriptor and status
    input  logic              start,
    input  logic              dir,
    input  logic [31:0]       ddr_addr,
    input  logic [TCM_AW-1:0] tcm_addr,
    input  logic [15:0]       words,
    output logic              busy,
    output logic              done,
    output logic              error,
    // TCM port
    output logic              tcm_port_req,
    output logic              tcm_port_we,
    output logic [TCM_AW-1:0] tcm_port_addr,
    output logic [63:0]       tcm_port_wdata,
    output logic [7:0]        tcm_port_wstrb,
    input  logic              tcm_port_gnt,
    input  logic [63:0]       tcm_port_rdata,
    // AXI4 master, read channels
    output logic [31:0]       m_axi_araddr,
    output logic [7:0]        m_axi_arlen,
    output logic [2:0]        m_axi_arsize,
    output logic [1:0]        m_axi_arburst,
    output logic [3:0]        m_axi_arcache,
    output logic [2:0]        m_axi_arprot,
    output logic              m_axi_arvalid,
    input  logic              m_axi_arready,
    input  logic [63:0]       m_axi_rdata,
    input  logic [1:0]        m_axi_rresp,
    input  logic              m_axi_rlast,
    input  logic              m_axi_rvalid,
    output logic              m_axi_rready,
    // AXI4 master, write channels
    output logic [31:0]       m_axi_awaddr,
    output logic [7:0]        m_axi_awlen,
    output logic [2:0]        m_axi_awsize,
    output logic [1:0]        m_axi_awburst,
    output logic [3:0]        m_axi_awcache,
    output logic [2:0]        m_axi_awprot,
    output logic              m_axi_awvalid,
    input  logic              m_axi_awready,
    output logic [63:0]       m_axi_wdata,
    output logic [7:0]        m_axi_wstrb,
    output logic              m_axi_wlast,
    output logic              m_axi_wvalid,
    input  logic              m_axi_wready,
    input  logic [1:0]        m_axi_bresp,
    input  logic              m_axi_bvalid,
    output logic              m_axi_bready
);
    localparam integer FAW = $clog2(FIFO_DEPTH);
    // Width of the remaining-word counters: an accepted descriptor has
    // tcm_addr + words <= TCM_DEPTH, so words fits in clog2(TCM_DEPTH + 1)
    // bits (11 for 1024); at least 5 for the burst-length logic.
    localparam integer LW0 = $clog2(TCM_DEPTH + 1);
    localparam integer LW = (LW0 < 5) ? 5 : ((LW0 > 16) ? 16 : LW0);

    initial begin
        if (FIFO_DEPTH < 2 || (1 << FAW) != FIFO_DEPTH)
            $fatal(1, "npu_tcm_dma FIFO_DEPTH must be a power of two >= 2");
        if (MAX_OUT < 1 || MAX_OUT > 4)
            $fatal(1, "npu_tcm_dma MAX_OUT must be 1..4");
        if (TCM_DEPTH < 2 || TCM_DEPTH > 65536 || TCM_AW > 16)
            $fatal(1, "npu_tcm_dma TCM_DEPTH must be 2..65536 and TCM_AW <= 16 (17-bit range check)");
        if (TCM_AW < $clog2(TCM_DEPTH))
            $fatal(1, "npu_tcm_dma TCM_AW must address all TCM_DEPTH words");
    end

    // ------------------------------------------------------------ control
    logic running, dir_q, err_q, done_q;
    logic desc_bad;
    logic accept;           // start sampled
    logic finish;           // running transfer completes this cycle

    // Part selects use continuous assignments (Icarus limitation in always_*).
    assign desc_bad = (ddr_addr[2:0] != 3'd0) ||
                      ({1'b0, words} + 17'(tcm_addr) > 17'(TCM_DEPTH));
    assign accept = start && !running;

    // ------------------------------------------------- burst generator
    // gen_addr/gen_rem describe what is left to issue. The burst length is
    // min(16, gen_rem, words to the next 4 KB boundary). The boundary limit
    // is below 16 only in the last 128 bytes of a page.
    // gen_addr only matters while words remain, i.e. after a burst of
    // to_bnd beats (any shorter burst is the last one). So its next value is
    // gen_addr + 8 * to_bnd = the next 128-byte step, or the next page start
    // in the last 128 bytes of a page: {gen_addr[31:7] + 1, low bits}. This
    // keeps the burst-length compare out of the address carry chain.
    // Addresses are kept as 8-byte word addresses ([31:3]): a running
    // transfer has an aligned ddr_addr, so AxADDR[2:0] is constant 0.
    logic [31:3] gen_addr;
    logic [31:3] gen_addr_nx;   // gen_addr after a to_bnd-beat burst
    logic [LW-1:0] gen_rem;
    logic [15:0] gen_rem16;     // zero-extended view
    logic [4:0]  to_bnd, beats;
    logic        ax_valid;
    logic [31:3] ax_addr;
    logic [3:0]  ax_len;
    logic        ax_ready;
    logic        issue;
    logic [2:0]  out_bursts;    // issued (incl. ax register) and not finished
    logic        burst_end;     // RLAST beat accepted / B accepted

    assign to_bnd = (&gen_addr[11:7]) ? (5'd16 - {1'b0, gen_addr[6:3]}) : 5'd16;
    assign gen_addr_nx = {gen_addr[31:7] + 25'd1,
                          (&gen_addr[11:7]) ? 4'd0 : gen_addr[6:3]};
    assign gen_rem16 = 16'(gen_rem);
    assign beats = (gen_rem16[15:5] == 11'd0 && gen_rem16[4:0] < to_bnd) ?
                   gen_rem16[4:0] : to_bnd;

    always_comb begin
        ax_ready = dir_q ? m_axi_awready : m_axi_arready;
        issue = running && !err_q && gen_rem != '0 &&
                (!ax_valid || ax_ready) && out_bursts < 3'(MAX_OUT);
        burst_end = dir_q ? (m_axi_bvalid && m_axi_bready)
                          : (m_axi_rvalid && m_axi_rready && m_axi_rlast);
    end

    // --------------------------------------------- DDR -> TCM (R side)
    logic        rbuf_valid;
    logic [63:0] rbuf_data;
    logic [TCM_AW-1:0] tptr;    // next TCM word (written or read)
    logic        r_hs;

    // ------------------------------------------- TCM -> DDR (W side)
    logic [63:0] fifo_mem [0:FIFO_DEPTH-1];
    logic [FAW:0] fifo_wp, fifo_rp;
    logic [FAW:0] credits;      // FIFO entries + TCM reads in flight
    logic        fifo_empty;
    logic        rd_inflight;
    logic [LW-1:0] rd_rem;      // TCM words left to read
    logic        rd_issue;
    logic [3:0]  wlen_q [0:3];  // lengths of issued write bursts, in order
    logic [1:0]  wlen_wp, wlen_rp;
    logic [2:0]  wlen_cnt;
    logic [3:0]  w_beat;
    logic        w_hs;

    always_comb begin
        r_hs = m_axi_rvalid && m_axi_rready;
        fifo_empty = (fifo_wp == fifo_rp);
        rd_issue = running && dir_q && rd_rem != '0 &&
                   credits != (FAW+1)'(FIFO_DEPTH);
        w_hs = m_axi_wvalid && m_axi_wready;
    end

    // TCM port
    always_comb begin
        tcm_port_req = dir_q ? rd_issue : rbuf_valid;
        tcm_port_we = !dir_q;
        tcm_port_addr = tptr;
        tcm_port_wdata = rbuf_data;
        tcm_port_wstrb = 8'hFF;
    end

    // AXI outputs
    assign m_axi_wdata = fifo_mem[fifo_rp[FAW-1:0]];

    always_comb begin
        m_axi_araddr = {ax_addr, 3'b000};
        m_axi_arlen = {4'd0, ax_len};
        m_axi_arsize = 3'd3;
        m_axi_arburst = 2'b01;
        m_axi_arcache = 4'b0011;
        m_axi_arprot = 3'b000;
        m_axi_arvalid = ax_valid && !dir_q;
        m_axi_rready = !rbuf_valid || tcm_port_gnt;

        m_axi_awaddr = {ax_addr, 3'b000};
        m_axi_awlen = {4'd0, ax_len};
        m_axi_awsize = 3'd3;
        m_axi_awburst = 2'b01;
        m_axi_awcache = 4'b0011;
        m_axi_awprot = 3'b000;
        m_axi_awvalid = ax_valid && dir_q;
        m_axi_wstrb = 8'hFF;
        m_axi_wlast = (w_beat == wlen_q[wlen_rp]);
        m_axi_wvalid = dir_q && wlen_cnt != 3'd0 && !fifo_empty;
        m_axi_bready = 1'b1;

        finish = running && (gen_rem == '0 || err_q) &&
                 out_bursts == 3'd0 && !rbuf_valid;
        busy = running;
        done = done_q;
        error = err_q;
    end

    // FIFO storage and burst-length queue: no reset (distributed RAM).
    always_ff @(posedge clk) begin
        if (rd_inflight)
            fifo_mem[fifo_wp[FAW-1:0]] <= tcm_port_rdata;
        if (issue && dir_q)
            wlen_q[wlen_wp] <= beats[3:0] - 4'd1;
    end

    // Datapath registers: no reset. Each is loaded at accept (or with the
    // handshake that makes it valid) and only observed while a reset control
    // bit qualifies it: gen_addr/gen_rem/rd_rem/tptr while running (and
    // tcm_port_addr only with tcm_port_req), ax_addr/ax_len with ax_valid,
    // rbuf_data with rbuf_valid. accept and issue are mutually exclusive
    // (accept needs !running, issue needs running). A rejected descriptor
    // never sets running, so its (possibly truncated) words value loaded
    // into gen_rem/rd_rem is never observed.
    always_ff @(posedge clk) begin
        if (accept) begin
            gen_addr <= ddr_addr[31:3];
            gen_rem <= LW'(words);
            rd_rem <= LW'(words);
            tptr <= tcm_addr;
        end else begin
            if (issue) begin
                gen_addr <= gen_addr_nx;  // stale after the last burst (unused)
                gen_rem <= gen_rem - LW'(beats);
            end
            if (tcm_port_req && tcm_port_gnt)
                tptr <= tptr + 1'b1;
            if (rd_issue && tcm_port_gnt)
                rd_rem <= rd_rem - 1'b1;
        end
        if (issue) begin
            ax_addr <= gen_addr;
            ax_len <= beats[3:0] - 4'd1;
        end
        if (r_hs)
            rbuf_data <= m_axi_rdata;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            dir_q <= 1'b0;
            err_q <= 1'b0;
            done_q <= 1'b0;
            ax_valid <= 1'b0;
            out_bursts <= '0;
            rbuf_valid <= 1'b0;
            fifo_wp <= '0;
            fifo_rp <= '0;
            credits <= '0;
            rd_inflight <= 1'b0;
            wlen_wp <= '0;
            wlen_rp <= '0;
            wlen_cnt <= '0;
            w_beat <= '0;
        end else begin
            done_q <= 1'b0;
            if (accept) begin
                dir_q <= dir;
                err_q <= desc_bad;
                fifo_wp <= '0;
                fifo_rp <= '0;
                credits <= '0;
                rd_inflight <= 1'b0;
                wlen_wp <= '0;
                wlen_rp <= '0;
                wlen_cnt <= '0;
                w_beat <= '0;
                if (desc_bad || words == 16'd0)
                    done_q <= 1'b1;
                else
                    running <= 1'b1;
            end else begin
                if (finish) begin
                    running <= 1'b0;
                    done_q <= 1'b1;
                end

                // Address generator and AR/AW register.
                if (issue) begin
                    ax_valid <= 1'b1;
                end else if (ax_ready) begin
                    ax_valid <= 1'b0;
                end
                out_bursts <= out_bursts + {2'd0, issue} - {2'd0, burst_end};

                // Response errors.
                if ((!dir_q && r_hs && m_axi_rresp != 2'b00) ||
                    (dir_q && m_axi_bvalid && m_axi_bresp != 2'b00))
                    err_q <= 1'b1;

                // DDR -> TCM: register each R beat, write it when granted.
                if (!dir_q) begin
                    if (r_hs)
                        rbuf_valid <= 1'b1;
                    else if (tcm_port_gnt)
                        rbuf_valid <= 1'b0;
                end

                // TCM -> DDR: prefetch reads, FIFO, W bursts.
                rd_inflight <= rd_issue && tcm_port_gnt;
                if (rd_inflight)
                    fifo_wp <= fifo_wp + 1'b1;
                if (w_hs)
                    fifo_rp <= fifo_rp + 1'b1;
                credits <= credits + (FAW+1)'(rd_issue && tcm_port_gnt) -
                           (FAW+1)'(w_hs);
                if (issue && dir_q)
                    wlen_wp <= wlen_wp + 1'b1;
                if (w_hs) begin
                    if (m_axi_wlast) begin
                        w_beat <= '0;
                        wlen_rp <= wlen_rp + 1'b1;
                    end else begin
                        w_beat <= w_beat + 1'b1;
                    end
                end
                wlen_cnt <= wlen_cnt + {2'd0, issue && dir_q} -
                            {2'd0, w_hs && m_axi_wlast};
            end
        end
    end
endmodule
