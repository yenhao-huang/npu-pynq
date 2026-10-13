`timescale 1ns/1ps

// AXI4 slave memory model for tb_npu_tcm_dma: a 64-bit word array at BASE,
// random ready/valid timing per channel (percent per cycle, 100 = zero
// latency), in-order R and B, optional error response err_code on one burst
// (on its first or its last R beat). It checks
// generic AXI4 rules: payload stability while valid waits for ready, INCR,
// size 3, len <= 15, 8-byte alignment, no 4 KB crossing, address in the
// window, WLAST on the last beat. W beats are buffered; with w_early they
// are accepted before their AW, otherwise WREADY waits for an AW. With
// aw_after_w (use with w_early) AWREADY additionally waits until a W beat of
// the next burst has arrived: a slave that wants W before AW (legal AXI).
module tb_npu_tcm_dma_axi_mem #(
    parameter logic [31:0] BASE = 32'h3000_0000,
    parameter integer WORDS = 8192
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] ar_pct, r_pct, aw_pct, w_pct, b_pct,
    input  logic        w_early,       // accept W before its AW
    input  logic        aw_after_w,    // AWREADY only after W of that burst
    input  logic signed [31:0] r_err_burst,   // AR index with SLVERR, -1 none
    input  logic signed [31:0] b_err_burst,   // AW index with SLVERR, -1 none
    input  logic        r_err_last,    // SLVERR on the last beat, else the first
    input  logic [1:0]  err_code,      // RRESP/BRESP of the failing burst
    input  logic [31:0] araddr,
    input  logic [7:0]  arlen,
    input  logic [2:0]  arsize,
    input  logic [1:0]  arburst,
    input  logic        arvalid,
    output logic        arready,
    output logic [63:0] rdata,
    output logic [1:0]  rresp,
    output logic        rlast,
    output logic        rvalid,
    input  logic        rready,
    input  logic [31:0] awaddr,
    input  logic [7:0]  awlen,
    input  logic [2:0]  awsize,
    input  logic [1:0]  awburst,
    input  logic        awvalid,
    output logic        awready,
    input  logic [63:0] wdata,
    input  logic [7:0]  wstrb,
    input  logic        wlast,
    input  logic        wvalid,
    output logic        wready,
    output logic [1:0]  bresp,
    output logic        bvalid,
    input  logic        bready,
    output integer      n_ar,
    output integer      n_aw,
    output integer      n_rbeats,
    output integer      n_wbeats
);
    logic [63:0] mem [0:WORDS-1];
    integer seed = 32'h1360_0001;

    logic [31:0] arq_addr [0:15];
    logic [7:0]  arq_len [0:15];
    logic        arq_err [0:15];
    integer arq_wp, arq_rp, arq_n, r_beat;
    logic [31:0] awq_addr [0:15];
    logic [7:0]  awq_len [0:15];
    logic        awq_err [0:15];
    integer awq_wp, awq_rp, awq_n, w_beat;
    logic [1:0]  bq_resp [0:15];
    integer bq_wp, bq_rp, bq_n;
    logic [63:0] wb_data [0:63];
    logic [7:0]  wb_strb [0:63];
    logic        wb_last [0:63];
    integer wb_wp, wb_rp, wb_n;

    logic ar_wait, aw_wait, w_wait;
    logic [31:0] ar_addr_h, aw_addr_h;
    logic [7:0]  ar_len_h, aw_len_h;
    logic [63:0] w_data_h;
    logic [7:0]  w_strb_h;
    logic        w_last_h;

    function automatic integer rnd100();
        rnd100 = ($random(seed) & 32'h7fff_ffff) % 100;
    endfunction

    function automatic integer widx(input logic [31:0] a);
        if (a < BASE || a >= BASE + WORDS * 8) begin
            $display("MISMATCH tb_npu_tcm_dma: address %h outside the memory model", a);
            $fatal(1);
        end
        widx = (a - BASE) >> 3;
    endfunction

    task automatic check_burst(input logic [31:0] a, input logic [7:0] len,
                               input logic [2:0] size, input logic [1:0] burst);
        integer dummy;
        if (len > 8'd15 || size != 3'd3 || burst != 2'b01 || a[2:0] != 3'd0 ||
            ({20'd0, a[11:0]} + ({24'd0, len} + 32'd1) * 32'd8) > 32'd4096) begin
            $display("MISMATCH tb_npu_tcm_dma: illegal burst addr=%h len=%0d size=%0d burst=%0d",
                     a, len, size, burst);
            $fatal(1);
        end
        dummy = widx(a);
        dummy = widx(a + {21'd0, len, 3'd0});
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            arready <= 1'b0; rvalid <= 1'b0; rdata <= '0; rresp <= '0; rlast <= 1'b0;
            awready <= 1'b0; wready <= 1'b0; bvalid <= 1'b0; bresp <= '0;
            arq_wp = 0; arq_rp = 0; arq_n = 0; r_beat = 0;
            awq_wp = 0; awq_rp = 0; awq_n = 0; w_beat = 0;
            bq_wp = 0; bq_rp = 0; bq_n = 0;
            wb_wp = 0; wb_rp = 0; wb_n = 0;
            ar_wait = 1'b0; aw_wait = 1'b0; w_wait = 1'b0;
            n_ar = 0; n_aw = 0; n_rbeats = 0; n_wbeats = 0;
        end else begin
            // Master-side stability: payload held while valid waits.
            if (ar_wait && (!arvalid || araddr != ar_addr_h || arlen != ar_len_h)) begin
                $display("MISMATCH tb_npu_tcm_dma: AR changed while waiting for ARREADY");
                $fatal(1);
            end
            if (aw_wait && (!awvalid || awaddr != aw_addr_h || awlen != aw_len_h)) begin
                $display("MISMATCH tb_npu_tcm_dma: AW changed while waiting for AWREADY");
                $fatal(1);
            end
            if (w_wait && (!wvalid || wdata != w_data_h || wstrb != w_strb_h ||
                           wlast != w_last_h)) begin
                $display("MISMATCH tb_npu_tcm_dma: W changed while waiting for WREADY");
                $fatal(1);
            end
            ar_wait = arvalid && !arready; ar_addr_h = araddr; ar_len_h = arlen;
            aw_wait = awvalid && !awready; aw_addr_h = awaddr; aw_len_h = awlen;
            w_wait = wvalid && !wready; w_data_h = wdata; w_strb_h = wstrb;
            w_last_h = wlast;

            // AR
            if (arvalid && arready) begin
                check_burst(araddr, arlen, arsize, arburst);
                arq_addr[arq_wp] = araddr;
                arq_len[arq_wp] = arlen;
                arq_err[arq_wp] = (n_ar == r_err_burst);
                arq_wp = (arq_wp + 1) % 16;
                arq_n = arq_n + 1;
                n_ar = n_ar + 1;
            end
            // R
            if (rvalid && rready) begin
                n_rbeats = n_rbeats + 1;
                if (r_beat == arq_len[arq_rp]) begin
                    r_beat = 0;
                    arq_rp = (arq_rp + 1) % 16;
                    arq_n = arq_n - 1;
                end else begin
                    r_beat = r_beat + 1;
                end
            end
            if (!rvalid || rready) begin
                if (arq_n > 0 && rnd100() < r_pct) begin
                    rvalid <= 1'b1;
                    rdata <= mem[widx(arq_addr[arq_rp] + r_beat * 8)];
                    rlast <= (r_beat == arq_len[arq_rp]);
                    rresp <= (arq_err[arq_rp] &&
                              (r_err_last ? (r_beat == arq_len[arq_rp]) : (r_beat == 0))) ?
                             err_code : 2'b00;
                end else begin
                    rvalid <= 1'b0;
                end
            end
            arready <= (arq_n < 14) && (rnd100() < ar_pct);

            // AW
            if (awvalid && awready) begin
                check_burst(awaddr, awlen, awsize, awburst);
                awq_addr[awq_wp] = awaddr;
                awq_len[awq_wp] = awlen;
                awq_err[awq_wp] = (n_aw == b_err_burst);
                awq_wp = (awq_wp + 1) % 16;
                awq_n = awq_n + 1;
                n_aw = n_aw + 1;
            end
            // W: beats are buffered, so W may arrive before its AW when
            // w_early is set; otherwise WREADY waits for an AW.
            if (wvalid && wready) begin
                n_wbeats = n_wbeats + 1;
                wb_data[wb_wp] = wdata;
                wb_strb[wb_wp] = wstrb;
                wb_last[wb_wp] = wlast;
                wb_wp = (wb_wp + 1) % 64;
                wb_n = wb_n + 1;
            end
            while (wb_n > 0 && awq_n > 0) begin
                for (int i = 0; i < 8; i++) begin
                    if (wb_strb[wb_rp][i])
                        mem[widx(awq_addr[awq_rp] + w_beat * 8)][i*8 +: 8] = wb_data[wb_rp][i*8 +: 8];
                end
                if (wb_last[wb_rp] != (w_beat == awq_len[awq_rp])) begin
                    $display("MISMATCH tb_npu_tcm_dma: WLAST at beat %0d of len %0d",
                             w_beat, awq_len[awq_rp]);
                    $fatal(1);
                end
                if (wb_last[wb_rp]) begin
                    bq_resp[bq_wp] = awq_err[awq_rp] ? err_code : 2'b00;
                    bq_wp = (bq_wp + 1) % 16;
                    bq_n = bq_n + 1;
                    awq_rp = (awq_rp + 1) % 16;
                    awq_n = awq_n - 1;
                    w_beat = 0;
                end else begin
                    w_beat = w_beat + 1;
                end
                wb_rp = (wb_rp + 1) % 64;
                wb_n = wb_n - 1;
            end
            wready <= (w_early ? (wb_n < 60) : (awq_n > 0)) && (rnd100() < w_pct);
            // After the loop above, wb_n > 0 means unmatched W beats of the
            // next burst are buffered (awq_n is then 0).
            awready <= (awq_n < 14) && (bq_n < 14) && (!aw_after_w || wb_n > 0) &&
                       (rnd100() < aw_pct);
            // B
            if (bvalid && bready) begin
                bq_rp = (bq_rp + 1) % 16;
                bq_n = bq_n - 1;
            end
            if (!bvalid || bready) begin
                if (bq_n > 0 && rnd100() < b_pct) begin
                    bvalid <= 1'b1;
                    bresp <= bq_resp[bq_rp];
                end else begin
                    bvalid <= 1'b0;
                end
            end
        end
    end
endmodule

// npu_tcm_dma + npu_tcm (port A) + the AXI memory model. Port A of the TCM
// is driven by the DMA through an optional random grant gate (models a
// shared TCM port). The TB checks: descriptor validation, done pulse count,
// exact burst sequence (address and length = min(16, remaining, to 4 KB)),
// AXI attributes, data round trips DDR -> TCM -> DDR with full-array
// comparison against reference copies, response errors (early and late;
// nothing outside the descriptor range may change), slaves that order AW and
// W either way (WREADY waits for AW, or AWREADY waits for W), start-while-
// busy, a start held high across an aborted write (accepted in the done
// cycle), SLVERR/DECERR/EXOKAY all treated as errors, reset, and the
// throughput of a long transfer with a zero-latency slave.
// TCM port: while the DMA is not granted, a competing client reads random
// addresses on the same port (rdata changes under the DMA, as on the shared
// port B). Every granted DMA access must be the next word of the descriptor
// range, and must make progress: a TCM write only with a received R beat
// pending, at most FIFO_DEPTH TCM reads ahead of the W beats. Directed stalls
// of W and of R check that the DMA drops tcm_port_req (no port hogging).
// No X on handshakes or on valid payloads (datapath flops are not reset).
module tb_npu_tcm_dma;
    localparam integer DEPTH = 1024;
    localparam integer AW = 10;
    localparam logic [31:0] BASE = 32'h3000_0000;
    localparam integer DWORDS = 8192;      // 64 KB DDR window
    localparam integer MAX_OUT = 4;
    localparam integer FIFO_DEPTH = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic start = 1'b0, dir = 1'b0;
    logic [31:0] ddr_addr = '0;
    logic [AW-1:0] tcm_addr = '0;
    logic [15:0] words = '0;
    logic busy, done, error;

    logic tp_req, tp_we, tp_gnt;
    logic [AW-1:0] tp_addr;
    logic [63:0] tp_wdata, tp_rdata;
    logic [7:0] tp_wstrb;
    logic gnt_ok = 1'b1;

    logic [31:0] araddr, awaddr;
    logic [7:0] arlen, awlen, wstrb;
    logic [2:0] arsize, awsize, arprot, awprot;
    logic [1:0] arburst, awburst, rresp, bresp;
    logic [3:0] arcache, awcache;
    logic arvalid, arready, rlast, rvalid, rready;
    logic awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [63:0] rdata, wdata;

    logic w_early = 1'b0;
    logic aw_after_w = 1'b0;
    logic [31:0] ar_pct = 100, r_pct = 100, aw_pct = 100, w_pct = 100, b_pct = 100;
    logic signed [31:0] r_err_burst = -1, b_err_burst = -1;
    logic r_err_last = 1'b0;
    logic [1:0] err_code = 2'b10;
    integer n_ar, n_aw, n_rbeats, n_wbeats;

    npu_tcm_dma #(.TCM_DEPTH(DEPTH), .FIFO_DEPTH(FIFO_DEPTH), .MAX_OUT(MAX_OUT)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .dir(dir), .ddr_addr(ddr_addr), .tcm_addr(tcm_addr),
        .words(words), .busy(busy), .done(done), .error(error),
        .tcm_port_req(tp_req), .tcm_port_we(tp_we), .tcm_port_addr(tp_addr),
        .tcm_port_wdata(tp_wdata), .tcm_port_wstrb(tp_wstrb),
        .tcm_port_gnt(tp_gnt), .tcm_port_rdata(tp_rdata),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize),
        .m_axi_arburst(arburst), .m_axi_arcache(arcache), .m_axi_arprot(arprot),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready),
        .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast),
        .m_axi_rvalid(rvalid), .m_axi_rready(rready),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize),
        .m_axi_awburst(awburst), .m_axi_awcache(awcache), .m_axi_awprot(awprot),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bvalid(bvalid), .m_axi_bready(bready)
    );

    // Shared TCM port: the DMA is granted when gnt_ok; otherwise a competing
    // client may read a random address (comp_on), which changes rdata.
    logic tcm_a_req, tcm_a_we, comp_on = 1'b0;
    logic [AW-1:0] tcm_a_addr, comp_addr = '0;
    logic [7:0] tcm_a_wstrb;
    logic tcm_a_gnt_unused, tcm_b_gnt_unused;
    logic [63:0] tcm_b_rdata_unused;
    assign tp_gnt = tp_req && gnt_ok;
    assign tcm_a_req = tp_gnt || comp_on;
    assign tcm_a_we = tp_gnt && tp_we;
    assign tcm_a_addr = tp_gnt ? tp_addr : comp_addr;
    assign tcm_a_wstrb = tp_gnt ? tp_wstrb : 8'h00;

    npu_tcm #(.WIDTH(64), .DEPTH(DEPTH)) u_tcm (
        .clk(clk), .rst_n(rst_n),
        .a_req(tcm_a_req), .a_we(tcm_a_we), .a_addr(tcm_a_addr), .a_wdata(tp_wdata),
        .a_wstrb(tcm_a_wstrb), .a_gnt(tcm_a_gnt_unused), .a_rdata(tp_rdata),
        .b_req(1'b0), .b_we(1'b0), .b_addr(10'd0), .b_wdata(64'd0), .b_wstrb(8'd0),
        .b_gnt(tcm_b_gnt_unused), .b_rdata(tcm_b_rdata_unused)
    );

    tb_npu_tcm_dma_axi_mem #(.BASE(BASE), .WORDS(DWORDS)) ddr (
        .clk(clk), .rst_n(rst_n),
        .ar_pct(ar_pct), .r_pct(r_pct), .aw_pct(aw_pct), .w_pct(w_pct), .b_pct(b_pct),
        .w_early(w_early), .aw_after_w(aw_after_w),
        .r_err_burst(r_err_burst), .b_err_burst(b_err_burst), .r_err_last(r_err_last),
        .err_code(err_code),
        .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst),
        .arvalid(arvalid), .arready(arready),
        .rdata(rdata), .rresp(rresp), .rlast(rlast), .rvalid(rvalid), .rready(rready),
        .awaddr(awaddr), .awlen(awlen), .awsize(awsize), .awburst(awburst),
        .awvalid(awvalid), .awready(awready),
        .wdata(wdata), .wstrb(wstrb), .wlast(wlast), .wvalid(wvalid), .wready(wready),
        .bresp(bresp), .bvalid(bvalid), .bready(bready),
        .n_ar(n_ar), .n_aw(n_aw), .n_rbeats(n_rbeats), .n_wbeats(n_wbeats)
    );

    // ----------------------------------------------------------- monitors
    integer seed = 136;
    integer gnt_pct = 100;
    integer done_count = 0;
    logic done_prev = 1'b0;
    // Expected burst sequence for the current transfer.
    logic [31:0] exp_ar_addr = '0, exp_aw_addr = '0;
    integer exp_ar_rem = 0, exp_aw_rem = 0;
    integer cross_4k = 0, bursts_total = 0;
    // Current transfer and its progress counters (TCM port monitor).
    logic cur_dir = 1'b0;
    integer cur_ta = 0, cur_n = 0;
    integer t_wr = 0, t_rd = 0, r_acc = 0, w_acc = 0;
    logic stall_win = 1'b0;
    integer stall_req = 0;

    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    function automatic integer exp_len(input logic [31:0] a, input integer rem);
        integer to_bnd;
        to_bnd = (4096 - a[11:0]) / 8;
        exp_len = rem;
        if (exp_len > 16) exp_len = 16;
        if (exp_len > to_bnd) exp_len = to_bnd;
    endfunction

    always @(posedge clk) begin
        if (rst_n) begin
            if (done) done_count = done_count + 1;
            if (done && done_prev) begin
                $display("MISMATCH tb_npu_tcm_dma: done high for two cycles");
                $fatal(1);
            end
            if (arvalid && arready) begin
                if (exp_ar_rem <= 0 || araddr != exp_ar_addr ||
                    arlen != 8'(exp_len(exp_ar_addr, exp_ar_rem) - 1) ||
                    arcache != 4'b0011 || arprot != 3'b000) begin
                    $display("MISMATCH tb_npu_tcm_dma: AR addr=%h len=%0d, expected addr=%h len=%0d rem=%0d",
                             araddr, arlen, exp_ar_addr, exp_len(exp_ar_addr, exp_ar_rem) - 1,
                             exp_ar_rem);
                    $fatal(1);
                end
                if (araddr[11:0] + (arlen + 1) * 8 == 4096 && exp_ar_rem > arlen + 1)
                    cross_4k = cross_4k + 1;
                exp_ar_addr = exp_ar_addr + (arlen + 1) * 8;
                exp_ar_rem = exp_ar_rem - (arlen + 1);
                bursts_total = bursts_total + 1;
            end
            if (awvalid && awready) begin
                if (exp_aw_rem <= 0 || awaddr != exp_aw_addr ||
                    awlen != 8'(exp_len(exp_aw_addr, exp_aw_rem) - 1) ||
                    awcache != 4'b0011 || awprot != 3'b000) begin
                    $display("MISMATCH tb_npu_tcm_dma: AW addr=%h len=%0d, expected addr=%h len=%0d rem=%0d",
                             awaddr, awlen, exp_aw_addr, exp_len(exp_aw_addr, exp_aw_rem) - 1,
                             exp_aw_rem);
                    $fatal(1);
                end
                if (awaddr[11:0] + (awlen + 1) * 8 == 4096 && exp_aw_rem > awlen + 1)
                    cross_4k = cross_4k + 1;
                exp_aw_addr = exp_aw_addr + (awlen + 1) * 8;
                exp_aw_rem = exp_aw_rem - (awlen + 1);
                bursts_total = bursts_total + 1;
            end
            if (wvalid && wstrb != 8'hFF) begin
                $display("MISMATCH tb_npu_tcm_dma: partial WSTRB %h", wstrb);
                $fatal(1);
            end
            if (tp_req && !busy) begin
                $display("MISMATCH tb_npu_tcm_dma: TCM access while not busy");
                $fatal(1);
            end
            if (tp_req && tp_we && tp_wstrb != 8'hFF) begin
                $display("MISMATCH tb_npu_tcm_dma: partial TCM write strobe");
                $fatal(1);
            end
            // No X on handshakes, nor on a payload while it is valid.
            if ((^{busy, done, error, tp_req, arvalid, rready, awvalid, wvalid,
                   bready}) === 1'bx)
                fail("X on a status or handshake output");
            if (tp_req && (^{tp_we, tp_addr}) === 1'bx)
                fail("X on the TCM request");
            if (tp_req && tp_we && (^{tp_wdata, tp_wstrb}) === 1'bx)
                fail("X on the TCM write data");
            if (arvalid && (^{araddr, arlen, arsize, arburst, arcache, arprot}) === 1'bx)
                fail("X on the AR payload while ARVALID");
            if (awvalid && (^{awaddr, awlen, awsize, awburst, awcache, awprot}) === 1'bx)
                fail("X on the AW payload while AWVALID");
            if (wvalid && (^{wdata, wstrb, wlast}) === 1'bx)
                fail("X on the W payload while WVALID");

            // TCM port progress (pre-update counts). The k-th granted access
            // of a transfer is word cur_ta + k in the descriptor direction.
            if (tp_gnt) begin
                if (tp_we == cur_dir)
                    fail("TCM access in the wrong direction");
                if (tp_addr != AW'(cur_ta + (cur_dir ? t_rd : t_wr)) ||
                    (cur_dir ? t_rd : t_wr) >= cur_n) begin
                    $display("MISMATCH tb_npu_tcm_dma: TCM access %0d at %0d, descriptor ta=%0d n=%0d",
                             cur_dir ? t_rd : t_wr, tp_addr, cur_ta, cur_n);
                    $fatal(1);
                end
            end
            // DDR -> TCM: req only with a received R beat not yet written.
            if (tp_req && tp_we && r_acc - t_wr < 1)
                fail("TCM write requested without a pending R beat");
            // TCM -> DDR: req only while a FIFO slot is free.
            if (tp_req && !tp_we && t_rd - w_acc >= FIFO_DEPTH)
                fail("TCM read requested with the FIFO full (port hogging)");
            if (tp_gnt && tp_we) t_wr = t_wr + 1;
            if (tp_gnt && !tp_we) t_rd = t_rd + 1;
            if (rvalid && rready) r_acc = r_acc + 1;
            if (wvalid && wready) w_acc = w_acc + 1;
            if (r_acc - t_wr > 1) fail("R accepted while the R register is full");
            if (t_rd - w_acc > FIFO_DEPTH) fail("TCM read-ahead exceeds FIFO_DEPTH");
            if (stall_win && tp_req) stall_req = stall_req + 1;
            // An accepted start begins a new transfer.
            if (start && !busy) begin
                cur_dir = dir; cur_ta = tcm_addr; cur_n = words;
                t_wr = 0; t_rd = 0; r_acc = 0; w_acc = 0;
            end
        end else begin
            t_wr = 0; t_rd = 0; r_acc = 0; w_acc = 0; cur_n = 0;
        end
        done_prev = done;
        gnt_ok <= (rnd(100) < gnt_pct);
        comp_on <= (rnd(100) < 50);
        comp_addr <= AW'(rnd(DEPTH));
    end

    // --------------------------------------------------- reference copies
    logic [63:0] ref_tcm [0:DEPTH-1];
    logic [63:0] ref_ddr [0:DWORDS-1];
    integer last_cycles;

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_tcm_dma: %s", msg);
        $fatal(1);
    endtask

    task automatic compare_all(input string what);
        for (int i = 0; i < DEPTH; i++) begin
            if (u_tcm.mem[i] !== ref_tcm[i]) begin
                $display("MISMATCH tb_npu_tcm_dma: %s TCM[%0d] = %h, expected %h",
                         what, i, u_tcm.mem[i], ref_tcm[i]);
                $fatal(1);
            end
        end
        for (int i = 0; i < DWORDS; i++) begin
            if (ddr.mem[i] !== ref_ddr[i]) begin
                $display("MISMATCH tb_npu_tcm_dma: %s DDR word %0d = %h, expected %h",
                         what, i, ddr.mem[i], ref_ddr[i]);
                $fatal(1);
            end
        end
    endtask

    // Run one descriptor and wait for its done pulse. bad = descriptor is
    // invalid (no AXI traffic, done the cycle after start).
    task automatic run(input logic d, input logic [31:0] da, input logic [AW-1:0] ta,
                       input logic [15:0] n, input logic bad, input logic exp_err);
        integer d0, ar0, aw0, cycles;
        d0 = done_count;
        ar0 = n_ar;
        aw0 = n_aw;
        exp_ar_addr = da; exp_aw_addr = da;
        exp_ar_rem = (!bad && !d) ? n : 0;
        exp_aw_rem = (!bad && d) ? n : 0;
        @(negedge clk);
        start = 1'b1; dir = d; ddr_addr = da; tcm_addr = ta; words = n;
        @(negedge clk);
        start = 1'b0; dir = $random(seed); ddr_addr = $random(seed);
        tcm_addr = $random(seed); words = $random(seed);
        cycles = 1;
        if (bad || n == 0) begin
            if (!done || busy || error !== bad)
                fail("rejected/empty descriptor must pulse done next cycle without busy");
        end else if (!busy) begin
            fail("busy not raised");
        end
        while (done_count == d0) begin
            if (cycles > 40000) fail("timeout waiting for done (deadlock)");
            // Start while busy is ignored.
            if (cycles == 3 && busy) begin
                start = 1'b1; dir = !d; ddr_addr = 32'h0000_0003; words = 16'hFFFF;
            end else begin
                start = 1'b0;
            end
            @(negedge clk);
            cycles = cycles + 1;
        end
        start = 1'b0;
        last_cycles = cycles;
        if (busy) fail("busy still high after done");
        if (error !== exp_err) begin
            $display("MISMATCH tb_npu_tcm_dma: error=%b expected %b (dir=%0d addr=%h n=%0d)",
                     error, exp_err, d, da, n);
            $fatal(1);
        end
        if (!exp_err && (exp_ar_rem != 0 || exp_aw_rem != 0))
            fail("transfer finished before all bursts were issued");
        if (bad || n == 0) begin
            if (n_ar != ar0 || n_aw != aw0) fail("AXI traffic for a rejected/empty descriptor");
        end
        repeat (3) @(negedge clk);
        if (done_count != d0 + 1) fail("done pulse count");
        if (!error && exp_err) fail("error not sticky");
    endtask

    // Two TCM -> DDR descriptors back to back: the first is expected to end
    // with a response error (set b_err_burst before the call); the second is
    // held on start while the first is busy and must be accepted in the
    // cycle the first one's done pulses, then complete without error.
    task automatic run_b2b(input logic [31:0] da1, input logic [AW-1:0] ta1,
                           input logic [15:0] n1, input logic [31:0] da2,
                           input logic [AW-1:0] ta2, input logic [15:0] n2);
        integer d0, cycles;
        d0 = done_count;
        exp_aw_addr = da1; exp_aw_rem = n1; exp_ar_rem = 0;
        @(negedge clk);
        start = 1'b1; dir = 1'b1; ddr_addr = da1; tcm_addr = ta1; words = n1;
        @(negedge clk);
        if (!busy) fail("busy not raised");
        ddr_addr = da2; tcm_addr = ta2; words = n2;   // start stays high
        cycles = 1;
        while (!done) begin
            if (cycles > 40000) fail("timeout waiting for done (deadlock)");
            @(negedge clk);
            cycles = cycles + 1;
        end
        if (error !== 1'b1) fail("aborted write did not raise error");
        // start is still high: the second descriptor is accepted at the
        // next edge, so its bursts follow.
        exp_aw_addr = da2; exp_aw_rem = n2;
        @(negedge clk);
        start = 1'b0; dir = $random(seed); ddr_addr = $random(seed);
        tcm_addr = $random(seed); words = $random(seed);
        if (busy !== 1'b1 || error !== 1'b0)
            fail("start held high was not accepted in the done cycle");
        while (done_count == d0 + 1) begin
            if (cycles > 80000) fail("timeout waiting for done (deadlock)");
            @(negedge clk);
            cycles = cycles + 1;
        end
        if (busy || error !== 1'b0) fail("back-to-back transfer ended busy or with error");
        if (exp_aw_rem != 0) fail("transfer finished before all bursts were issued");
        repeat (3) @(negedge clk);
        if (done_count != d0 + 2) fail("done pulse count");
    endtask

    // Random data into a DDR region (model and reference).
    task automatic fill_ddr(input logic [31:0] da, input integer n);
        integer w;
        for (int i = 0; i < n; i++) begin
            w = ((da - BASE) >> 3) + i;
            ddr.mem[w] = {$random(seed), $random(seed)};
            ref_ddr[w] = ddr.mem[w];
        end
    endtask

    task automatic set_profile(input integer p);
        w_early = p[0];
        aw_after_w = (p == 3);   // slave that waits for W before AW
        case (p)
            0: begin ar_pct = 100; r_pct = 100; aw_pct = 100; w_pct = 100; b_pct = 100; gnt_pct = 100; end
            1: begin ar_pct = 50; r_pct = 60; aw_pct = 50; w_pct = 60; b_pct = 50; gnt_pct = 100; end
            2: begin ar_pct = 20; r_pct = 25; aw_pct = 20; w_pct = 25; b_pct = 20; gnt_pct = 60; end
            3: begin ar_pct = 100; r_pct = 30; aw_pct = 100; w_pct = 30; b_pct = 100; gnt_pct = 100; end
            default: begin ar_pct = 70; r_pct = 100; aw_pct = 70; w_pct = 100; b_pct = 40; gnt_pct = 50; end
        endcase
    endtask

    function automatic logic [31:0] rand_ddr(input integer n, input logic xing);
        integer page, k, off;
        if (xing && n > 1) begin
            // Start k words before a 4 KB boundary inside the window.
            page = 1 + rnd(DWORDS * 8 / 4096 - 1);
            k = 1 + rnd(n - 1);
            rand_ddr = BASE + page * 4096 - k * 8;
        end else begin
            off = rnd(DWORDS - n + 1);
            rand_ddr = BASE + off * 8;
        end
        if (rand_ddr + n * 8 > BASE + DWORDS * 8)
            rand_ddr = BASE + (DWORDS - n) * 8;
    endfunction

    integer n, iter, rt_count, rd_cycles_1k, wr_cycles_1k;
    logic [31:0] src, dst;
    logic [AW-1:0] ta;

    initial begin
        // Initial contents: DDR random, TCM all zero via backdoor.
        for (int i = 0; i < DWORDS; i++) begin
            ddr.mem[i] = {$random(seed), $random(seed)};
            ref_ddr[i] = ddr.mem[i];
        end
        for (int i = 0; i < DEPTH; i++) begin
            u_tcm.mem[i] = '0;
            ref_tcm[i] = '0;
        end

        // ------------------------------------------------ reset behaviour
        repeat (3) @(negedge clk);
        if (busy !== 1'b0 || done !== 1'b0 || error !== 1'b0 || arvalid !== 1'b0 ||
            awvalid !== 1'b0 || wvalid !== 1'b0 || tp_req !== 1'b0)
            fail("outputs not idle during reset");
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // ------------------------------------------------- directed cases
        set_profile(0);
        run(1'b0, BASE, 10'd0, 16'd0, 1'b0, 1'b0);              // empty
        run(1'b0, BASE + 4, 10'd0, 16'd4, 1'b1, 1'b1);          // misaligned
        run(1'b1, BASE + 1, 10'd0, 16'd4, 1'b1, 1'b1);          // misaligned
        run(1'b0, BASE, 10'd1000, 16'd25, 1'b1, 1'b1);          // past TCM end
        run(1'b1, BASE, 10'd0, 16'd1025, 1'b1, 1'b1);           // longer than TCM
        run(1'b0, BASE, 10'd5, 16'hFFFF, 1'b1, 1'b1);           // longer than TCM
        // Exactly to the end of the TCM, 1 word, 16 words, 17 words.
        run(1'b0, BASE + 8 * 100, 10'd1000, 16'd24, 1'b0, 1'b0);
        for (int i = 0; i < 24; i++) ref_tcm[1000 + i] = ref_ddr[100 + i];
        run(1'b0, BASE + 8 * 7, 10'd3, 16'd1, 1'b0, 1'b0);
        ref_tcm[3] = ref_ddr[7];
        run(1'b0, BASE + 8 * 32, 10'd16, 16'd16, 1'b0, 1'b0);
        for (int i = 0; i < 16; i++) ref_tcm[16 + i] = ref_ddr[32 + i];
        run(1'b1, BASE + 8 * 4000, 10'd15, 16'd17, 1'b0, 1'b0);
        for (int i = 0; i < 17; i++) ref_ddr[4000 + i] = ref_tcm[15 + i];
        // A burst ending exactly on a 4 KB boundary, then one crossing it.
        run(1'b0, BASE + 4096 - 8 * 16, 10'd200, 16'd16, 1'b0, 1'b0);
        for (int i = 0; i < 16; i++) ref_tcm[200 + i] = ref_ddr[512 - 16 + i];
        run(1'b1, BASE + 2 * 4096 - 8 * 3, 10'd200, 16'd40, 1'b0, 1'b0);
        for (int i = 0; i < 40; i++) ref_ddr[1024 - 3 + i] = ref_tcm[200 + i];
        compare_all("directed");

        // ------------------------------------------- random round trips
        rt_count = 0;
        for (iter = 0; iter < 90; iter++) begin
            set_profile(iter % 5);
            n = 1 + rnd(300);
            if (iter % 7 == 0) n = 280 + rnd(21);
            ta = rnd(DEPTH - n + 1);
            src = rand_ddr(n, (iter % 3) == 0);
            dst = rand_ddr(n, (iter % 3) == 1);
            fill_ddr(src, n);
            run(1'b0, src, ta, n[15:0], 1'b0, 1'b0);
            for (int i = 0; i < n; i++) ref_tcm[ta + i] = ref_ddr[((src - BASE) >> 3) + i];
            run(1'b1, dst, ta, n[15:0], 1'b0, 1'b0);
            for (int i = 0; i < n; i++) ref_ddr[((dst - BASE) >> 3) + i] = ref_tcm[ta + i];
            compare_all("round trip");
            rt_count = rt_count + 1;
        end

        // ----------------------------------------------- response errors
        // Early errors check that issuing stops; late errors (near the end
        // of the transfer) check that the drain stays inside the range.
        // Every response other than OKAY is an error: SLVERR, DECERR and
        // EXOKAY (illegal for a normal access, still not OKAY).
        for (int p = 0; p < 6; p++) begin
            set_profile(p % 3);
            err_code = (p % 3 == 0) ? 2'b10 : ((p % 3 == 1) ? 2'b11 : 2'b01);
            // RRESP error on burst 3 (or 12) of a 200-word read (13 bursts),
            // on the first or (odd p) the last beat of that burst.
            r_err_burst = n_ar + ((p < 3) ? 2 : 11);
            r_err_last = p[0];
            src = BASE + 8 * 64;
            run(1'b0, src, 10'd100, 16'd200, 1'b0, 1'b1);
            // At most MAX_OUT bursts may follow the failing one.
            if (n_ar - (r_err_burst + 1) > MAX_OUT) fail("bursts issued after RRESP error");
            r_err_burst = -1;
            // The destination range is undefined after an aborted read, but
            // no word outside it may change and DDR is untouched.
            for (int i = 100; i < 300; i++) ref_tcm[i] = u_tcm.mem[i];
            compare_all("outside an aborted read");
            // BRESP error on burst 2 (or 9) of a 150-word write (10 bursts).
            b_err_burst = n_aw + ((p < 3) ? 1 : 8);
            dst = BASE + 8 * 5000;
            run(1'b1, dst, 10'd300, 16'd150, 1'b0, 1'b1);
            if (n_aw - (b_err_burst + 1) > MAX_OUT) fail("bursts issued after BRESP error");
            b_err_burst = -1;
            // Same for an aborted write: only [dst, dst + 150 words) may change.
            for (int i = 5000; i < 5150; i++) ref_ddr[i] = ddr.mem[i];
            compare_all("outside an aborted write");
            // Error clears on the next start; a clean transfer works.
            fill_ddr(BASE + 8 * 6000, 33);
            run(1'b0, BASE + 8 * 6000, 10'd700, 16'd33, 1'b0, 1'b0);
            for (int i = 0; i < 33; i++) ref_tcm[700 + i] = ref_ddr[6000 + i];
            compare_all("after error");
        end

        // ------------------- back-to-back start after an aborted write
        // The next descriptor is held on start while the aborted write
        // drains, so it is accepted in the cycle done pulses. State left by
        // the aborted transfer (prefetched FIFO words, a TCM read in flight)
        // must not leak into it.
        for (int p = 0; p < 8; p++) begin
            set_profile(p % 3);
            err_code = 2'b10;
            b_err_burst = n_aw + 1 + (p % 4) * 2;
            run_b2b(BASE + 8 * 5000, 10'd300, 16'd150,
                    BASE + 8 * 5400, AW'(400 + 16 * p), 16'(37 + p));
            b_err_burst = -1;
            for (int i = 5000; i < 5150; i++) ref_ddr[i] = ddr.mem[i];
            for (int i = 0; i < 37 + p; i++) ref_ddr[5400 + i] = ref_tcm[400 + 16 * p + i];
            compare_all("back-to-back after an aborted write");
        end
        err_code = 2'b10;

        // ------------------------------------- throughput, zero latency
        set_profile(0);
        fill_ddr(BASE, 1024);
        run(1'b0, BASE, 10'd0, 16'd1024, 1'b0, 1'b0);
        rd_cycles_1k = last_cycles;
        for (int i = 0; i < 1024; i++) ref_tcm[i] = ref_ddr[i];
        run(1'b1, BASE + 8 * 4096, 10'd0, 16'd1024, 1'b0, 1'b0);
        wr_cycles_1k = last_cycles;
        for (int i = 0; i < 1024; i++) ref_ddr[4096 + i] = ref_tcm[i];
        compare_all("throughput");
        if (rd_cycles_1k > 1040 || wr_cycles_1k > 1040)
            fail("long-transfer throughput below 1024/1040 beats/cycle");

        // ---------------------------- stalled channel: TCM port released
        // W stalled in the middle of a TCM -> DDR transfer: after at most
        // FIFO_DEPTH more TCM reads the DMA must drop tcm_port_req for the
        // rest of the stall (the monitor also checks the FIFO bound).
        set_profile(0);
        fork
            run(1'b1, BASE + 8 * 2000, 10'd100, 16'd300, 1'b0, 1'b0);
            begin
                repeat (40) @(negedge clk);
                w_pct = 0;
                repeat (FIFO_DEPTH + 6) @(negedge clk);
                stall_req = 0; stall_win = 1'b1;
                repeat (100) @(negedge clk);
                stall_win = 1'b0; w_pct = 100;
                if (!busy) fail("W stall window ended after the transfer");
                if (stall_req != 0) fail("tcm_port_req held while W is stalled");
            end
        join
        for (int i = 0; i < 300; i++) ref_ddr[2000 + i] = ref_tcm[100 + i];
        // R stalled in the middle of a DDR -> TCM transfer: the pending
        // beat is written and tcm_port_req drops.
        fill_ddr(BASE + 8 * 2500, 300);
        fork
            run(1'b0, BASE + 8 * 2500, 10'd400, 16'd300, 1'b0, 1'b0);
            begin
                repeat (40) @(negedge clk);
                r_pct = 0;
                repeat (6) @(negedge clk);
                stall_req = 0; stall_win = 1'b1;
                repeat (100) @(negedge clk);
                stall_win = 1'b0; r_pct = 100;
                if (!busy) fail("R stall window ended after the transfer");
                if (stall_req != 0) fail("tcm_port_req held while R is stalled");
            end
        join
        for (int i = 0; i < 300; i++) ref_tcm[400 + i] = ref_ddr[2500 + i];
        compare_all("stalled channels");

        // ------------------------------------------- reset mid-transfer
        exp_ar_addr = BASE; exp_ar_rem = 500; exp_aw_rem = 0;
        @(negedge clk);
        start = 1'b1; dir = 1'b0; ddr_addr = BASE; tcm_addr = '0; words = 16'd500;
        @(negedge clk);
        start = 1'b0;
        repeat (20) @(negedge clk);
        rst_n = 1'b0;
        @(negedge clk);
        if (busy !== 1'b0 || done !== 1'b0 || arvalid !== 1'b0 || tp_req !== 1'b0)
            fail("reset did not return to idle");
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        for (int i = 0; i < DEPTH; i++) ref_tcm[i] = u_tcm.mem[i];
        fill_ddr(BASE + 8 * 300, 77);
        run(1'b0, BASE + 8 * 300, 10'd900, 16'd77, 1'b0, 1'b0);
        for (int i = 0; i < 77; i++) ref_tcm[900 + i] = ref_ddr[300 + i];
        compare_all("after reset");

        $display("PASS tb_npu_tcm_dma round_trips=%0d bursts=%0d crossings_4k=%0d done_pulses=%0d read_1024w=%0d_cycles write_1024w=%0d_cycles",
                 rt_count, bursts_total, cross_4k, done_count, rd_cycles_1k, wr_cycles_1k);
        $finish;
    end
endmodule
