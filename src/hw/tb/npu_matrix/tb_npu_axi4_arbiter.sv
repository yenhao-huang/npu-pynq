`timescale 1ns/1ps

// AXI4 slave memory model for tb_npu_axi4_arbiter (derived from the one in
// tb_npu_tcm_dma): a 64-bit word array at BASE, random ready/valid timing per
// channel (percent per cycle, 100 = zero latency), in-order R and B. RRESP
// and BRESP carry a routing tag computed from the address A of the R beat or
// of the write burst's start: {A[3], A[14] ^ A[4]} (A[14] is set in master 1
// regions), so a misrouted, stuck or tied-off response bit is visible.
// It checks
// generic AXI4 rules: payload stability while valid waits for ready, INCR,
// size 3, len <= 15, 8-byte alignment, no 4 KB crossing, address in the
// window, WLAST on the last beat. W beats are buffered; with w_early they
// are accepted before their AW, otherwise WREADY waits for an AW. With
// aw_after_w (use with w_early) AWREADY additionally waits until a W beat of
// the next burst has arrived: a slave that wants W before AW (legal AXI).
module tb_npu_axi4_arbiter_mem #(
    parameter logic [31:0] BASE = 32'h3000_0000,
    parameter integer WORDS = 8192
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [31:0] ar_pct, r_pct, aw_pct, w_pct, b_pct,
    input  logic        w_early,       // accept W before its AW
    input  logic        aw_after_w,    // AWREADY only after W of that burst
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
    logic        arq_tag [0:15];   // routing tag (address bit 14)
    integer arq_wp, arq_rp, arq_n, r_beat;
    logic [31:0] awq_addr [0:15];
    logic [7:0]  awq_len [0:15];
    logic        awq_tag [0:15];
    integer awq_wp, awq_rp, awq_n, w_beat;
    logic [1:0]  bq_resp [0:15];
    integer bq_wp, bq_rp, bq_n;
    logic [63:0] wb_data [0:63];
    logic [7:0]  wb_strb [0:63];
    logic        wb_last [0:63];
    integer wb_wp, wb_rp, wb_n;

    logic ar_wait, aw_wait, w_wait;
    logic [31:0] ar_beat_a;
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
            $display("MISMATCH tb_npu_axi4_arbiter: address %h outside the memory model", a);
            $fatal(1);
        end
        widx = (a - BASE) >> 3;
    endfunction

    task automatic check_burst(input logic [31:0] a, input logic [7:0] len,
                               input logic [2:0] size, input logic [1:0] burst);
        integer dummy;
        if (len > 8'd15 || size != 3'd3 || burst != 2'b01 || a[2:0] != 3'd0 ||
            ({20'd0, a[11:0]} + ({24'd0, len} + 32'd1) * 32'd8) > 32'd4096) begin
            $display("MISMATCH tb_npu_axi4_arbiter: illegal burst addr=%h len=%0d size=%0d burst=%0d",
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
                $display("MISMATCH tb_npu_axi4_arbiter: AR changed while waiting for ARREADY");
                $fatal(1);
            end
            if (aw_wait && (!awvalid || awaddr != aw_addr_h || awlen != aw_len_h)) begin
                $display("MISMATCH tb_npu_axi4_arbiter: AW changed while waiting for AWREADY");
                $fatal(1);
            end
            if (w_wait && (!wvalid || wdata != w_data_h || wstrb != w_strb_h ||
                           wlast != w_last_h)) begin
                $display("MISMATCH tb_npu_axi4_arbiter: W changed while waiting for WREADY");
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
                arq_tag[arq_wp] = araddr[14];
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
                    ar_beat_a = arq_addr[arq_rp] + r_beat * 8;
                    rdata <= mem[widx(ar_beat_a)];
                    rlast <= (r_beat == arq_len[arq_rp]);
                    rresp <= {ar_beat_a[3], arq_tag[arq_rp] ^ ar_beat_a[4]};
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
                awq_tag[awq_wp] = awaddr[14];
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
                    $display("MISMATCH tb_npu_axi4_arbiter: WLAST at beat %0d of len %0d",
                             w_beat, awq_len[awq_rp]);
                    $fatal(1);
                end
                if (wb_last[wb_rp]) begin
                    bq_resp[bq_wp] = {awq_addr[awq_rp][3],
                                      awq_tag[awq_rp] ^ awq_addr[awq_rp][4]};
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

// Random AXI4 master for tb_npu_axi4_arbiter. Reads random INCR bursts
// (1..16 beats, no 4 KB crossing) from its read region, whose words hold
// pat(word index), and checks every R beat (data, RLAST, RRESP routing tag,
// payload stability). Writes random bursts with random data and strobes to
// its write region and keeps a reference copy; W of a burst may start before
// its AW handshake. Checks B count, BRESP routing tag and B stability.
module tb_npu_axi4_arbiter_master #(
    parameter integer ID = 0,
    parameter logic [31:0] RD_BASE = 32'h0,
    parameter logic [31:0] WR_BASE = 32'h8000,
    parameter integer REGION_WORDS = 2048
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        go,
    input  logic [31:0] n_rd,
    input  logic [31:0] n_wr,
    input  logic [31:0] arv_pct, rr_pct, awv_pct, wv_pct, br_pct,
    output logic        finished,
    output logic [31:0] araddr,
    output logic [7:0]  arlen,
    output logic [2:0]  arsize,
    output logic [1:0]  arburst,
    output logic [3:0]  arcache,
    output logic [2:0]  arprot,
    output logic        arvalid,
    input  logic        arready,
    input  logic [63:0] rdata,
    input  logic [1:0]  rresp,
    input  logic        rlast,
    input  logic        rvalid,
    output logic        rready,
    output logic [31:0] awaddr,
    output logic [7:0]  awlen,
    output logic [2:0]  awsize,
    output logic [1:0]  awburst,
    output logic [3:0]  awcache,
    output logic [2:0]  awprot,
    output logic        awvalid,
    input  logic        awready,
    output logic [63:0] wdata,
    output logic [7:0]  wstrb,
    output logic        wlast,
    output logic        wvalid,
    input  logic        wready,
    input  logic [1:0]  bresp,
    input  logic        bvalid,
    output logic        bready,
    output integer      n_rbeats,
    output integer      n_wbeats
);
    localparam logic [1:0] TAG = (ID != 0) ? 2'b01 : 2'b00;
    localparam logic [3:0] CACHE = (ID != 0) ? 4'b1111 : 4'b0011;
    localparam logic [2:0] PROT = (ID != 0) ? 3'b010 : 3'b000;

    integer seed = 32'h0A4B_0000 + ID;
    logic [63:0] ref_mem [0:REGION_WORDS-1];

    // Outstanding reads (accepted ARs), in order.
    logic [31:0] rq_addr [0:63];
    logic [7:0]  rq_len [0:63];
    integer rq_wp, rq_rp, rq_n, r_beat;
    // Write bursts generated (AW presented), in order, for W generation.
    logic [31:0] wq_addr [0:63];
    logic [7:0]  wq_len [0:63];
    integer wq_wp, wq_rp, wq_n, w_beat;
    integer ar_issued, rd_done, aw_issued, aw_acc, b_seen;
    // Accepted AW addresses, in order, for the expected BRESP tag.
    logic [31:0] bq_addr [0:63];
    logic r_wait, b_wait;
    logic [63:0] r_data_h;
    logic r_last_h;
    logic [1:0] r_resp_h, b_resp_h;

    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    function automatic logic [63:0] pat(input logic [31:0] a);
        pat = {a ^ 32'hA5C3_0000, ~a + 32'h1357_9BDF};
    endfunction

    // Random burst in [base, base + REGION_WORDS words), no 4 KB crossing.
    task automatic new_burst(input logic [31:0] base, output logic [31:0] a,
                             output logic [7:0] len);
        integer off, l, to_bnd;
        l = 1 + rnd(16);
        if (rnd(4) == 0) l = 16;
        off = rnd(REGION_WORDS - 16);
        a = base + off * 8;
        to_bnd = (4096 - a[11:0]) / 8;
        if (l > to_bnd) l = to_bnd;
        len = l - 1;
    endtask

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_axi4_arbiter: master %0d %s", ID, msg);
        $fatal(1);
    endtask

    always_comb finished = (rd_done == n_rd) && (b_seen == n_wr) &&
                           (aw_issued == n_wr) && !arvalid && !awvalid;

    always @(posedge clk or negedge rst_n) begin : proc
        logic [31:0] a;
        logic [7:0] l;
        logic [63:0] d;
        logic [7:0] s;
        integer widx;
        if (!rst_n) begin
            arvalid <= 1'b0; araddr <= '0; arlen <= '0;
            arsize <= 3'd3; arburst <= 2'b01; arcache <= CACHE; arprot <= PROT;
            awvalid <= 1'b0; awaddr <= '0; awlen <= '0;
            awsize <= 3'd3; awburst <= 2'b01; awcache <= CACHE; awprot <= PROT;
            wvalid <= 1'b0; wdata <= '0; wstrb <= '0; wlast <= 1'b0;
            rready <= 1'b0; bready <= 1'b0;
            rq_wp = 0; rq_rp = 0; rq_n = 0; r_beat = 0;
            wq_wp = 0; wq_rp = 0; wq_n = 0; w_beat = 0;
            ar_issued = 0; rd_done = 0; aw_issued = 0; aw_acc = 0; b_seen = 0;
            r_wait = 1'b0; b_wait = 1'b0;
            n_rbeats = 0; n_wbeats = 0;
        end else begin
            // R/B stability towards this master.
            if (r_wait && (!rvalid || rdata != r_data_h || rlast != r_last_h ||
                           rresp != r_resp_h))
                fail("R changed while waiting for RREADY");
            if (b_wait && (!bvalid || bresp != b_resp_h))
                fail("B changed while waiting for BREADY");
            r_wait = rvalid && !rready; r_data_h = rdata; r_last_h = rlast;
            r_resp_h = rresp;
            b_wait = bvalid && !bready; b_resp_h = bresp;

            // AR
            if (arvalid && arready) begin
                rq_addr[rq_wp] = araddr;
                rq_len[rq_wp] = arlen;
                rq_wp = (rq_wp + 1) % 64;
                rq_n = rq_n + 1;
            end
            if (!arvalid || arready) begin
                if (go && ar_issued < n_rd && rq_n < 60 && rnd(100) < arv_pct) begin
                    new_burst(RD_BASE, a, l);
                    arvalid <= 1'b1; araddr <= a; arlen <= l;
                    ar_issued = ar_issued + 1;
                end else begin
                    arvalid <= 1'b0;
                end
            end
            // R
            if (rvalid && rready) begin
                if (rq_n == 0) fail("R beat without an outstanding read");
                a = rq_addr[rq_rp] + r_beat * 8;
                if (rdata !== pat(a) || rresp !== {a[3], TAG[0] ^ a[4]} ||
                    rlast !== (r_beat == rq_len[rq_rp])) begin
                    $display("MISMATCH tb_npu_axi4_arbiter: master %0d R addr=%h data=%h exp=%h resp=%0d last=%b beat=%0d len=%0d",
                             ID, a, rdata, pat(a), rresp, rlast, r_beat, rq_len[rq_rp]);
                    $fatal(1);
                end
                n_rbeats = n_rbeats + 1;
                if (rlast) begin
                    r_beat = 0;
                    rq_rp = (rq_rp + 1) % 64;
                    rq_n = rq_n - 1;
                    rd_done = rd_done + 1;
                end else begin
                    r_beat = r_beat + 1;
                end
            end
            rready <= rnd(100) < rr_pct;

            // AW (the burst is queued for W as soon as it is presented)
            if (awvalid && awready) begin
                bq_addr[aw_acc % 64] = awaddr;
                aw_acc = aw_acc + 1;
            end
            if (!awvalid || awready) begin
                if (go && aw_issued < n_wr && wq_n < 60 && rnd(100) < awv_pct) begin
                    new_burst(WR_BASE, a, l);
                    awvalid <= 1'b1; awaddr <= a; awlen <= l;
                    wq_addr[wq_wp] = a;
                    wq_len[wq_wp] = l;
                    wq_wp = (wq_wp + 1) % 64;
                    wq_n = wq_n + 1;
                    aw_issued = aw_issued + 1;
                end else begin
                    awvalid <= 1'b0;
                end
            end
            // W
            if (wvalid && wready) begin
                widx = (wq_addr[wq_rp] - WR_BASE) / 8 + w_beat;
                for (int i = 0; i < 8; i++)
                    if (wstrb[i]) ref_mem[widx][i*8 +: 8] = wdata[i*8 +: 8];
                n_wbeats = n_wbeats + 1;
                if (wlast) begin
                    w_beat = 0;
                    wq_rp = (wq_rp + 1) % 64;
                    wq_n = wq_n - 1;
                end else begin
                    w_beat = w_beat + 1;
                end
            end
            if (!wvalid || wready) begin
                if (wq_n > 0 && rnd(100) < wv_pct) begin
                    d = {$random(seed), $random(seed)};
                    s = (rnd(3) == 0) ? 8'hFF : $random(seed);
                    wvalid <= 1'b1; wdata <= d; wstrb <= s;
                    wlast <= (w_beat == wq_len[wq_rp]);
                end else begin
                    wvalid <= 1'b0;
                end
            end
            // B
            if (bvalid && bready) begin
                if (bresp !== {bq_addr[b_seen % 64][3], TAG[0] ^ bq_addr[b_seen % 64][4]})
                    fail("BRESP routed to the wrong master or not passed through");
                b_seen = b_seen + 1;
                if (b_seen > aw_acc) fail("B before its AW was accepted");
            end
            bready <= rnd(100) < br_pct;
        end
    end
endmodule

// Two random masters -> npu_axi4_arbiter -> one memory model. Checks data
// integrity (read pattern, write regions vs. master reference copies),
// routing (R/B tags and data), AXI stability on every channel, strict
// round-robin on AR and AW when both masters request, a bounded wait
// (no starvation), no deadlock (phase timeout) with a downstream that waits
// for AW before W and one that waits for W before AW, no X on handshakes or
// on valid payloads (payload registers are not reset), no AR/AW accepted
// while the corresponding slice is occupied (the ready path must not depend
// on m_axi_arready/awready), saturated R throughput >= 95%, and reset between
// phases.
module tb_npu_axi4_arbiter;
    localparam integer WORDS = 8192;
    localparam integer RW = 2048;          // words per master region (16 KB)

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic go = 1'b0;
    logic [31:0] n_rd0 = 0, n_wr0 = 0, n_rd1 = 0, n_wr1 = 0;
    logic [31:0] m0_pct [0:4];
    logic [31:0] m1_pct [0:4];
    logic w_early = 1'b0;
    logic aw_after_w = 1'b0;
    logic [31:0] ar_pct = 100, r_pct = 100, aw_pct = 100, w_pct = 100, b_pct = 100;
    logic fin0, fin1;
    integer rb0, wb0, rb1, wb1, mem_nar, mem_naw, mem_nrb, mem_nwb;

    logic [63:0] s0_axi_rdata, s0_axi_wdata;
    logic [31:0] s0_axi_araddr, s0_axi_awaddr;
    logic [7:0] s0_axi_arlen, s0_axi_awlen, s0_axi_wstrb;
    logic [3:0] s0_axi_arcache, s0_axi_awcache;
    logic [2:0] s0_axi_arsize, s0_axi_arprot, s0_axi_awsize, s0_axi_awprot;
    logic [1:0] s0_axi_arburst, s0_axi_rresp, s0_axi_awburst, s0_axi_bresp;
    logic s0_axi_arvalid, s0_axi_arready, s0_axi_rlast, s0_axi_rvalid, s0_axi_rready,
        s0_axi_awvalid, s0_axi_awready, s0_axi_wlast, s0_axi_wvalid, s0_axi_wready,
        s0_axi_bvalid, s0_axi_bready;
    logic [63:0] s1_axi_rdata, s1_axi_wdata;
    logic [31:0] s1_axi_araddr, s1_axi_awaddr;
    logic [7:0] s1_axi_arlen, s1_axi_awlen, s1_axi_wstrb;
    logic [3:0] s1_axi_arcache, s1_axi_awcache;
    logic [2:0] s1_axi_arsize, s1_axi_arprot, s1_axi_awsize, s1_axi_awprot;
    logic [1:0] s1_axi_arburst, s1_axi_rresp, s1_axi_awburst, s1_axi_bresp;
    logic s1_axi_arvalid, s1_axi_arready, s1_axi_rlast, s1_axi_rvalid, s1_axi_rready,
        s1_axi_awvalid, s1_axi_awready, s1_axi_wlast, s1_axi_wvalid, s1_axi_wready,
        s1_axi_bvalid, s1_axi_bready;
    logic [63:0] m_axi_rdata, m_axi_wdata;
    logic [31:0] m_axi_araddr, m_axi_awaddr;
    logic [7:0] m_axi_arlen, m_axi_awlen, m_axi_wstrb;
    logic [3:0] m_axi_arcache, m_axi_awcache;
    logic [2:0] m_axi_arsize, m_axi_arprot, m_axi_awsize, m_axi_awprot;
    logic [1:0] m_axi_arburst, m_axi_rresp, m_axi_awburst, m_axi_bresp;
    logic m_axi_arvalid, m_axi_arready, m_axi_rlast, m_axi_rvalid, m_axi_rready, m_axi_awvalid,
        m_axi_awready, m_axi_wlast, m_axi_wvalid, m_axi_wready, m_axi_bvalid, m_axi_bready;

    npu_axi4_arbiter dut (
        .clk(clk), .rst_n(rst_n),
        .s0_axi_araddr(s0_axi_araddr), .s0_axi_arlen(s0_axi_arlen),
        .s0_axi_arsize(s0_axi_arsize), .s0_axi_arburst(s0_axi_arburst),
        .s0_axi_arcache(s0_axi_arcache), .s0_axi_arprot(s0_axi_arprot),
        .s0_axi_arvalid(s0_axi_arvalid), .s0_axi_arready(s0_axi_arready),
        .s0_axi_rdata(s0_axi_rdata), .s0_axi_rresp(s0_axi_rresp), .s0_axi_rlast(s0_axi_rlast),
        .s0_axi_rvalid(s0_axi_rvalid), .s0_axi_rready(s0_axi_rready),
        .s0_axi_awaddr(s0_axi_awaddr), .s0_axi_awlen(s0_axi_awlen),
        .s0_axi_awsize(s0_axi_awsize), .s0_axi_awburst(s0_axi_awburst),
        .s0_axi_awcache(s0_axi_awcache), .s0_axi_awprot(s0_axi_awprot),
        .s0_axi_awvalid(s0_axi_awvalid), .s0_axi_awready(s0_axi_awready),
        .s0_axi_wdata(s0_axi_wdata), .s0_axi_wstrb(s0_axi_wstrb), .s0_axi_wlast(s0_axi_wlast),
        .s0_axi_wvalid(s0_axi_wvalid), .s0_axi_wready(s0_axi_wready),
        .s0_axi_bresp(s0_axi_bresp), .s0_axi_bvalid(s0_axi_bvalid),
        .s0_axi_bready(s0_axi_bready),
        .s1_axi_araddr(s1_axi_araddr), .s1_axi_arlen(s1_axi_arlen),
        .s1_axi_arsize(s1_axi_arsize), .s1_axi_arburst(s1_axi_arburst),
        .s1_axi_arcache(s1_axi_arcache), .s1_axi_arprot(s1_axi_arprot),
        .s1_axi_arvalid(s1_axi_arvalid), .s1_axi_arready(s1_axi_arready),
        .s1_axi_rdata(s1_axi_rdata), .s1_axi_rresp(s1_axi_rresp), .s1_axi_rlast(s1_axi_rlast),
        .s1_axi_rvalid(s1_axi_rvalid), .s1_axi_rready(s1_axi_rready),
        .s1_axi_awaddr(s1_axi_awaddr), .s1_axi_awlen(s1_axi_awlen),
        .s1_axi_awsize(s1_axi_awsize), .s1_axi_awburst(s1_axi_awburst),
        .s1_axi_awcache(s1_axi_awcache), .s1_axi_awprot(s1_axi_awprot),
        .s1_axi_awvalid(s1_axi_awvalid), .s1_axi_awready(s1_axi_awready),
        .s1_axi_wdata(s1_axi_wdata), .s1_axi_wstrb(s1_axi_wstrb), .s1_axi_wlast(s1_axi_wlast),
        .s1_axi_wvalid(s1_axi_wvalid), .s1_axi_wready(s1_axi_wready),
        .s1_axi_bresp(s1_axi_bresp), .s1_axi_bvalid(s1_axi_bvalid),
        .s1_axi_bready(s1_axi_bready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arcache(m_axi_arcache),
        .m_axi_arprot(m_axi_arprot), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awcache(m_axi_awcache),
        .m_axi_awprot(m_axi_awprot), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready)
    );

    tb_npu_axi4_arbiter_master #(.ID(0), .RD_BASE(32'h0000), .WR_BASE(32'h8000),
                                 .REGION_WORDS(RW)) m0 (
        .clk(clk), .rst_n(rst_n), .go(go), .n_rd(n_rd0), .n_wr(n_wr0),
        .arv_pct(m0_pct[0]), .rr_pct(m0_pct[1]), .awv_pct(m0_pct[2]),
        .wv_pct(m0_pct[3]), .br_pct(m0_pct[4]), .finished(fin0),
        .araddr(s0_axi_araddr), .arlen(s0_axi_arlen), .arsize(s0_axi_arsize), .arburst(s0_axi_arburst),
        .arcache(s0_axi_arcache), .arprot(s0_axi_arprot), .arvalid(s0_axi_arvalid), .arready(s0_axi_arready),
        .rdata(s0_axi_rdata), .rresp(s0_axi_rresp), .rlast(s0_axi_rlast), .rvalid(s0_axi_rvalid),
        .rready(s0_axi_rready),
        .awaddr(s0_axi_awaddr), .awlen(s0_axi_awlen), .awsize(s0_axi_awsize), .awburst(s0_axi_awburst),
        .awcache(s0_axi_awcache), .awprot(s0_axi_awprot), .awvalid(s0_axi_awvalid), .awready(s0_axi_awready),
        .wdata(s0_axi_wdata), .wstrb(s0_axi_wstrb), .wlast(s0_axi_wlast), .wvalid(s0_axi_wvalid),
        .wready(s0_axi_wready), .bresp(s0_axi_bresp), .bvalid(s0_axi_bvalid), .bready(s0_axi_bready),
        .n_rbeats(rb0), .n_wbeats(wb0)
    );

    tb_npu_axi4_arbiter_master #(.ID(1), .RD_BASE(32'h4000), .WR_BASE(32'hC000),
                                 .REGION_WORDS(RW)) m1 (
        .clk(clk), .rst_n(rst_n), .go(go), .n_rd(n_rd1), .n_wr(n_wr1),
        .arv_pct(m1_pct[0]), .rr_pct(m1_pct[1]), .awv_pct(m1_pct[2]),
        .wv_pct(m1_pct[3]), .br_pct(m1_pct[4]), .finished(fin1),
        .araddr(s1_axi_araddr), .arlen(s1_axi_arlen), .arsize(s1_axi_arsize), .arburst(s1_axi_arburst),
        .arcache(s1_axi_arcache), .arprot(s1_axi_arprot), .arvalid(s1_axi_arvalid), .arready(s1_axi_arready),
        .rdata(s1_axi_rdata), .rresp(s1_axi_rresp), .rlast(s1_axi_rlast), .rvalid(s1_axi_rvalid),
        .rready(s1_axi_rready),
        .awaddr(s1_axi_awaddr), .awlen(s1_axi_awlen), .awsize(s1_axi_awsize), .awburst(s1_axi_awburst),
        .awcache(s1_axi_awcache), .awprot(s1_axi_awprot), .awvalid(s1_axi_awvalid), .awready(s1_axi_awready),
        .wdata(s1_axi_wdata), .wstrb(s1_axi_wstrb), .wlast(s1_axi_wlast), .wvalid(s1_axi_wvalid),
        .wready(s1_axi_wready), .bresp(s1_axi_bresp), .bvalid(s1_axi_bvalid), .bready(s1_axi_bready),
        .n_rbeats(rb1), .n_wbeats(wb1)
    );

    tb_npu_axi4_arbiter_mem #(.BASE(32'h0), .WORDS(WORDS)) mem (
        .clk(clk), .rst_n(rst_n),
        .ar_pct(ar_pct), .r_pct(r_pct), .aw_pct(aw_pct), .w_pct(w_pct), .b_pct(b_pct),
        .w_early(w_early), .aw_after_w(aw_after_w),
        .araddr(m_axi_araddr), .arlen(m_axi_arlen), .arsize(m_axi_arsize), .arburst(m_axi_arburst),
        .arvalid(m_axi_arvalid), .arready(m_axi_arready),
        .rdata(m_axi_rdata), .rresp(m_axi_rresp), .rlast(m_axi_rlast), .rvalid(m_axi_rvalid),
        .rready(m_axi_rready),
        .awaddr(m_axi_awaddr), .awlen(m_axi_awlen), .awsize(m_axi_awsize), .awburst(m_axi_awburst),
        .awvalid(m_axi_awvalid), .awready(m_axi_awready),
        .wdata(m_axi_wdata), .wstrb(m_axi_wstrb), .wlast(m_axi_wlast), .wvalid(m_axi_wvalid),
        .wready(m_axi_wready), .bresp(m_axi_bresp), .bvalid(m_axi_bvalid), .bready(m_axi_bready),
        .n_ar(mem_nar), .n_aw(mem_naw), .n_rbeats(mem_nrb), .n_wbeats(mem_nwb)
    );

    // ------------------------------------------------------------ monitors
    logic ar_last = 1'b0, aw_last = 1'b0;
    integer ar_wait0 = 0, ar_wait1 = 0, aw_wait0 = 0, aw_wait1 = 0, max_wait = 0;
    integer gr_ar0 = 0, gr_ar1 = 0, gr_aw0 = 0, gr_aw1 = 0, both_ar = 0, both_aw = 0;

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_axi4_arbiter: %s", msg);
        $fatal(1);
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            // After reset master 0 has priority: as if master 1 won last.
            ar_last = 1'b1;
            aw_last = 1'b1;
        end else begin
            if (s0_axi_arready && s1_axi_arready) fail("AR granted to both masters");
            if (s0_axi_awready && s1_axi_awready) fail("AW granted to both masters");
            if (s0_axi_wready && s1_axi_wready) fail("W open to both masters");
            // Timing contract: a slice loads only while empty, so a master's
            // AR/AW ready never depends on m_axi_arready/awready.
            if ((s0_axi_arready || s1_axi_arready) && m_axi_arvalid)
                fail("AR accepted while the AR slice is occupied");
            if ((s0_axi_awready || s1_axi_awready) && m_axi_awvalid)
                fail("AW accepted while the AW slice is occupied");
            // No X on handshake signals, nor on a payload while it is valid
            // (the payload registers have no reset).
            if ((^{s0_axi_arready, s1_axi_arready, s0_axi_awready, s1_axi_awready,
                   s0_axi_wready, s1_axi_wready, s0_axi_rvalid, s1_axi_rvalid,
                   s0_axi_bvalid, s1_axi_bvalid, m_axi_arvalid, m_axi_awvalid,
                   m_axi_wvalid, m_axi_rready, m_axi_bready}) === 1'bx)
                fail("X on a handshake signal");
            if (m_axi_arvalid && (^{m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst,
                                    m_axi_arcache, m_axi_arprot}) === 1'bx)
                fail("X on the AR payload while ARVALID");
            if (m_axi_awvalid && (^{m_axi_awaddr, m_axi_awlen, m_axi_awsize, m_axi_awburst,
                                    m_axi_awcache, m_axi_awprot}) === 1'bx)
                fail("X on the AW payload while AWVALID");
            if (m_axi_wvalid && (^{m_axi_wdata, m_axi_wstrb, m_axi_wlast}) === 1'bx)
                fail("X on the W payload while WVALID");
            if (s0_axi_rvalid && s1_axi_rvalid) fail("R to both masters");
            if (s0_axi_bvalid && s1_axi_bvalid) fail("B to both masters");
            // Downstream attributes follow the master (by address region).
            if (m_axi_arvalid && m_axi_arready &&
                (m_axi_arcache != (m_axi_araddr[14] ? 4'b1111 : 4'b0011) ||
                 m_axi_arprot != (m_axi_araddr[14] ? 3'b010 : 3'b000)))
                fail("AR attributes mixed between masters");
            if (m_axi_awvalid && m_axi_awready &&
                (m_axi_awcache != (m_axi_awaddr[14] ? 4'b1111 : 4'b0011) ||
                 m_axi_awprot != (m_axi_awaddr[14] ? 3'b010 : 3'b000)))
                fail("AW attributes mixed between masters");
            // Strict round robin when both request.
            if ((s0_axi_arvalid && s0_axi_arready) || (s1_axi_arvalid && s1_axi_arready)) begin
                if (s0_axi_arvalid && s1_axi_arvalid) begin
                    both_ar = both_ar + 1;
                    if (s1_axi_arready == ar_last) fail("AR round robin violated");
                end
                ar_last = s1_axi_arready;
                if (s0_axi_arready) gr_ar0 = gr_ar0 + 1; else gr_ar1 = gr_ar1 + 1;
            end
            if ((s0_axi_awvalid && s0_axi_awready) || (s1_axi_awvalid && s1_axi_awready)) begin
                if (s0_axi_awvalid && s1_axi_awvalid) begin
                    both_aw = both_aw + 1;
                    if (s1_axi_awready == aw_last) fail("AW round robin violated");
                end
                aw_last = s1_axi_awready;
                if (s0_axi_awready) gr_aw0 = gr_aw0 + 1; else gr_aw1 = gr_aw1 + 1;
            end
            // Waiting time of a pending request (starvation bound).
            ar_wait0 = (s0_axi_arvalid && !s0_axi_arready) ? ar_wait0 + 1 : 0;
            ar_wait1 = (s1_axi_arvalid && !s1_axi_arready) ? ar_wait1 + 1 : 0;
            aw_wait0 = (s0_axi_awvalid && !s0_axi_awready) ? aw_wait0 + 1 : 0;
            aw_wait1 = (s1_axi_awvalid && !s1_axi_awready) ? aw_wait1 + 1 : 0;
            if (ar_wait0 > max_wait) max_wait = ar_wait0;
            if (ar_wait1 > max_wait) max_wait = ar_wait1;
            if (aw_wait0 > max_wait) max_wait = aw_wait0;
            if (aw_wait1 > max_wait) max_wait = aw_wait1;
            if (max_wait > 2000) fail("a master waited more than 2000 cycles");
        end
    end

    // --------------------------------------------------------------- phases
    integer phase_cycles, total_r, total_w, cyc_b, rbeats_b;

    task automatic set_pcts(input integer m0a, m0b, m0c, m0d, m0e,
                            input integer m1a, m1b, m1c, m1d, m1e);
        m0_pct[0] = m0a; m0_pct[1] = m0b; m0_pct[2] = m0c; m0_pct[3] = m0d; m0_pct[4] = m0e;
        m1_pct[0] = m1a; m1_pct[1] = m1b; m1_pct[2] = m1c; m1_pct[3] = m1d; m1_pct[4] = m1e;
    endtask

    task automatic run_phase(input string name, input integer r0, w0, r1, w1);
        integer r0b, r1b, w0b, w1b;
        // Reset between phases (quiescent): every block returns to idle.
        rst_n = 1'b0;
        go = 1'b0;
        repeat (2) @(negedge clk);
        if (m_axi_arvalid !== 1'b0 || m_axi_awvalid !== 1'b0 || m_axi_wvalid !== 1'b0 ||
            m_axi_rready !== 1'b0 || m_axi_bready !== 1'b0 || s0_axi_rvalid !== 1'b0 ||
            s1_axi_bvalid !== 1'b0)
            fail("arbiter outputs not idle in reset");
        n_rd0 = r0; n_wr0 = w0; n_rd1 = r1; n_wr1 = w1;
        rst_n = 1'b1;
        @(negedge clk);
        go = 1'b1;
        phase_cycles = 0;
        while (!(fin0 && fin1)) begin
            @(negedge clk);
            phase_cycles = phase_cycles + 1;
            if (phase_cycles > 200000) begin
                $display("MISMATCH tb_npu_axi4_arbiter: phase %s deadlock (fin0=%b fin1=%b)",
                         name, fin0, fin1);
                $fatal(1);
            end
        end
        repeat (5) @(negedge clk);
        if (mem_nrb != rb0 + rb1 || mem_nwb != wb0 + wb1)
            fail("beat counts differ between masters and memory");
        if (m_axi_rvalid || m_axi_bvalid) fail("responses left over at the end of a phase");
        for (int i = 0; i < RW; i++) begin
            if (mem.mem[4096 + i] !== m0.ref_mem[i]) begin
                $display("MISMATCH tb_npu_axi4_arbiter: %s master 0 write word %0d = %h, expected %h",
                         name, i, mem.mem[4096 + i], m0.ref_mem[i]);
                $fatal(1);
            end
            if (mem.mem[6144 + i] !== m1.ref_mem[i]) begin
                $display("MISMATCH tb_npu_axi4_arbiter: %s master 1 write word %0d = %h, expected %h",
                         name, i, mem.mem[6144 + i], m1.ref_mem[i]);
                $fatal(1);
            end
        end
        total_r = total_r + rb0 + rb1;
        total_w = total_w + wb0 + wb1;
    endtask

    initial begin
        // Read regions hold the pattern, write regions random data (both
        // in the memory model and in the master reference copies).
        for (int i = 0; i < 4096; i++) mem.mem[i] = m0.pat(i * 8);
        for (int i = 0; i < RW; i++) begin
            mem.mem[4096 + i] = {$random, $random};
            m0.ref_mem[i] = mem.mem[4096 + i];
            mem.mem[6144 + i] = {$random, $random};
            m1.ref_mem[i] = mem.mem[6144 + i];
        end
        total_r = 0; total_w = 0;

        // A: random timing everywhere.
        ar_pct = 60; r_pct = 70; aw_pct = 60; w_pct = 70; b_pct = 60;
        set_pcts(60, 70, 60, 80, 70, 70, 60, 70, 60, 80);
        w_early = 1'b1;
        run_phase("random", 250, 250, 250, 250);
        // B: zero-latency memory, saturating masters (fairness, throughput).
        ar_pct = 100; r_pct = 100; aw_pct = 100; w_pct = 100; b_pct = 100;
        set_pcts(100, 100, 100, 100, 100, 100, 100, 100, 100, 100);
        gr_ar0 = 0; gr_ar1 = 0; gr_aw0 = 0; gr_aw1 = 0;
        w_early = 1'b0;
        run_phase("saturated", 200, 200, 200, 200);
        cyc_b = phase_cycles;
        rbeats_b = rb0 + rb1;
        if (gr_ar0 != 200 || gr_ar1 != 200) fail("AR grant counts");
        // R stays saturated (bursts average ~10 beats, so one AR every two
        // cycles is enough): at least 95% of the phase cycles carry an R beat.
        if (rbeats_b * 100 < cyc_b * 95) fail("saturated R throughput below 95%");
        // C: one master at a time.
        ar_pct = 80; r_pct = 90; aw_pct = 80; w_pct = 90; b_pct = 80;
        set_pcts(70, 80, 70, 80, 80, 70, 80, 70, 80, 80);
        w_early = 1'b1;
        run_phase("only master 1", 0, 0, 150, 150);
        run_phase("only master 0", 150, 150, 0, 0);
        // D: heavy backpressure on both sides.
        ar_pct = 20; r_pct = 25; aw_pct = 20; w_pct = 25; b_pct = 20;
        set_pcts(30, 25, 30, 30, 25, 25, 30, 25, 30, 30);
        w_early = 1'b0;
        run_phase("backpressure", 120, 120, 120, 120);
        // E: asymmetric masters: reads-only vs writes-only, slow master 1.
        ar_pct = 90; r_pct = 90; aw_pct = 90; w_pct = 90; b_pct = 90;
        set_pcts(100, 100, 100, 100, 100, 40, 10, 40, 15, 10);
        w_early = 1'b1;
        run_phase("asymmetric", 200, 0, 0, 200);
        run_phase("mixed", 100, 200, 200, 100);
        // F: slow AW acceptance, fast early W: masters present W before
        // their AW is granted while the W-order FIFO is empty.
        w_early = 1'b1;
        ar_pct = 15; r_pct = 100; aw_pct = 15; w_pct = 100; b_pct = 100;
        set_pcts(100, 100, 100, 100, 100, 100, 100, 100, 100, 100);
        run_phase("early W", 100, 200, 100, 200);
        // G: the downstream waits for W before it accepts AW (legal AXI),
        // the opposite order to the WREADY-waits-for-AW phases above. The
        // W channel must reach the downstream before the AW handshake.
        aw_after_w = 1'b1;
        ar_pct = 70; r_pct = 70; aw_pct = 70; w_pct = 70; b_pct = 70;
        set_pcts(80, 80, 80, 60, 80, 80, 80, 80, 60, 80);
        run_phase("AW after W", 150, 200, 150, 200);
        aw_after_w = 1'b0;

        $display("PASS tb_npu_axi4_arbiter r_beats=%0d w_beats=%0d rr_ar_contended=%0d rr_aw_contended=%0d max_wait=%0d saturated_r_beats=%0d_in_%0d_cycles",
                 total_r, total_w, both_ar, both_aw, max_wait, rbeats_b, cyc_b);
        $finish;
    end
endmodule
