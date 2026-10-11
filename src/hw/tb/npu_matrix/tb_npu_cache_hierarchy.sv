`timescale 1ns/1ps

// Integration test of the cache hierarchy:
//   L1 I$ (npu_cache, 1 KB, read-only) --\
//                                         npu_mem_arbiter -> L2 (npu_cache,
//   L1 D$ (npu_cache, 1 KB, write-back) -/   8 KB, 2-way, CPU_W = 256,
//                                            CPU_HOLD = 1, RESP_W = 32,
//                                            MEM_W = 64)
//   -> npu_axi4_line_bridge -> AXI4 slave memory model (random timing).
// Line protocol read beats: 32 bits from the L2 to the L1s (through the
// arbiter), 64 bits from the bridge to the L2.
// The I side reads a code region, the D side reads and writes (random byte
// strobes) a disjoint data region, concurrently. Every load is checked
// against a flat reference. Random flush pulses hit all three caches during
// traffic; at quiescent points L1 I$, L1 D$ then L2 are flushed, the AXI
// memory is compared with the reference, and then the memory changes behind
// the (now invalid) caches in the lines of the last I and D accesses, as the
// host does between FLUSH and START: a cache level that kept a stale line
// returns old data on the next access. Also checks the end-to-end zero-wait
// latencies (L1 miss / L2 hit, L1 miss / L2 miss, L1 dirty miss / L2 hit).
// A second, independent harness (tb_npu_cache_hierarchy_comb, below) drives
// the L1s from always_comb requesters as the scalar core does and catches a
// zero-time evaluation loop through L1 D$ -> arbiter -> L2 (review round 3).

module tb_npu_cache_hierarchy;
    localparam integer LW = 256;
    localparam logic [31:0] CODE_BASE = 32'h0000_4000;
    localparam integer CODE_SIZE = 8192;
    localparam logic [31:0] DATA_BASE = 32'h0100_0000;
    localparam integer DATA_SIZE = 16384;
    localparam integer NOPS = 3000;          // per CPU port

    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst_n;

    // L1 I$
    logic          i_req_valid, i_req_ready, i_resp_valid, i_flush, i_flush_done;
    logic [31:0]   i_req_addr, i_resp_rdata, i_hits, i_misses;
    logic          i_mreq_valid, i_mreq_ready, i_mreq_we, i_mresp_valid;
    logic [31:0]   i_mreq_addr;
    logic [LW-1:0] i_mreq_wdata;
    logic [31:0]   i_mresp_rdata;
    // L1 D$
    logic          d_req_valid, d_req_ready, d_req_we, d_resp_valid, d_flush, d_flush_done;
    logic [31:0]   d_req_addr, d_req_wdata, d_resp_rdata, d_hits, d_misses;
    logic [3:0]    d_req_wstrb;
    logic          d_mreq_valid, d_mreq_ready, d_mreq_we, d_mresp_valid;
    logic [31:0]   d_mreq_addr;
    logic [LW-1:0] d_mreq_wdata;
    logic [31:0]   d_mresp_rdata;
    // L2
    logic          l2_req_valid, l2_req_ready, l2_req_we, l2_resp_valid, l2_flush, l2_flush_done;
    logic [31:0]   l2_req_addr, l2_hits, l2_misses;
    logic [LW-1:0] l2_req_wdata;
    logic [31:0]   l2_resp_rdata;
    logic          l2_mreq_valid, l2_mreq_ready, l2_mreq_we, l2_mresp_valid;
    logic [31:0]   l2_mreq_addr;
    logic [LW-1:0] l2_mreq_wdata;
    logic [63:0]   l2_mresp_rdata;
    logic          bridge_error;
    // AXI
    logic [31:0]   araddr, awaddr;
    logic [7:0]    arlen, awlen;
    logic [2:0]    arsize, awsize, arprot, awprot;
    logic [1:0]    arburst, awburst, rresp, bresp;
    logic [3:0]    arcache, awcache;
    logic          arvalid, arready, rlast, rvalid, rready;
    logic          awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [63:0]   rdata, wdata;
    logic [7:0]    wstrb;

    npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .READ_ONLY(1)) l1i (
        .clk(clk), .rst_n(rst_n),
        .req_valid(i_req_valid), .req_ready(i_req_ready), .req_addr(i_req_addr),
        .req_we(1'b0), .req_wstrb(4'b0), .req_wdata(32'b0),
        .resp_valid(i_resp_valid), .resp_rdata(i_resp_rdata),
        .flush(i_flush), .flush_done(i_flush_done),
        .mem_req_valid(i_mreq_valid), .mem_req_ready(i_mreq_ready),
        .mem_req_addr(i_mreq_addr), .mem_req_we(i_mreq_we), .mem_req_wdata(i_mreq_wdata),
        .mem_resp_valid(i_mresp_valid), .mem_resp_rdata(i_mresp_rdata),
        .hits(i_hits), .misses(i_misses)
    );

    npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .WRITE_BACK(1)) l1d (
        .clk(clk), .rst_n(rst_n),
        .req_valid(d_req_valid), .req_ready(d_req_ready), .req_addr(d_req_addr),
        .req_we(d_req_we), .req_wstrb(d_req_wstrb), .req_wdata(d_req_wdata),
        .resp_valid(d_resp_valid), .resp_rdata(d_resp_rdata),
        .flush(d_flush), .flush_done(d_flush_done),
        .mem_req_valid(d_mreq_valid), .mem_req_ready(d_mreq_ready),
        .mem_req_addr(d_mreq_addr), .mem_req_we(d_mreq_we), .mem_req_wdata(d_mreq_wdata),
        .mem_resp_valid(d_mresp_valid), .mem_resp_rdata(d_mresp_rdata),
        .hits(d_hits), .misses(d_misses)
    );

    npu_mem_arbiter #(.LINE_BYTES(32), .BEAT_W(32), .P0_READ_ONLY(1)) arb (
        .clk(clk), .rst_n(rst_n),
        .s0_mem_req_valid(i_mreq_valid), .s0_mem_req_ready(i_mreq_ready),
        .s0_mem_req_addr(i_mreq_addr), .s0_mem_req_we(i_mreq_we),
        .s0_mem_req_wdata(i_mreq_wdata), .s0_mem_resp_valid(i_mresp_valid),
        .s0_mem_resp_rdata(i_mresp_rdata),
        .s1_mem_req_valid(d_mreq_valid), .s1_mem_req_ready(d_mreq_ready),
        .s1_mem_req_addr(d_mreq_addr), .s1_mem_req_we(d_mreq_we),
        .s1_mem_req_wdata(d_mreq_wdata), .s1_mem_resp_valid(d_mresp_valid),
        .s1_mem_resp_rdata(d_mresp_rdata),
        .m_mem_req_valid(l2_req_valid), .m_mem_req_ready(l2_req_ready),
        .m_mem_req_addr(l2_req_addr), .m_mem_req_we(l2_req_we),
        .m_mem_req_wdata(l2_req_wdata), .m_mem_resp_valid(l2_resp_valid),
        .m_mem_resp_rdata(l2_resp_rdata)
    );

    npu_cache #(.SIZE_BYTES(8192), .LINE_BYTES(32), .WAYS(2), .CPU_W(256),
                .CPU_HOLD(1), .MEM_W(64), .RESP_W(32)) l2 (
        .clk(clk), .rst_n(rst_n),
        .req_valid(l2_req_valid), .req_ready(l2_req_ready), .req_addr(l2_req_addr),
        .req_we(l2_req_we), .req_wstrb({32{1'b1}}), .req_wdata(l2_req_wdata),
        .resp_valid(l2_resp_valid), .resp_rdata(l2_resp_rdata),
        .flush(l2_flush), .flush_done(l2_flush_done),
        .mem_req_valid(l2_mreq_valid), .mem_req_ready(l2_mreq_ready),
        .mem_req_addr(l2_mreq_addr), .mem_req_we(l2_mreq_we), .mem_req_wdata(l2_mreq_wdata),
        .mem_resp_valid(l2_mresp_valid), .mem_resp_rdata(l2_mresp_rdata),
        .hits(l2_hits), .misses(l2_misses)
    );

    npu_axi4_line_bridge #(.LINE_BYTES(32)) bridge (
        .clk(clk), .rst_n(rst_n),
        .mem_req_valid(l2_mreq_valid), .mem_req_ready(l2_mreq_ready),
        .mem_req_addr(l2_mreq_addr), .mem_req_we(l2_mreq_we),
        .mem_req_wdata(l2_mreq_wdata), .mem_resp_valid(l2_mresp_valid),
        .mem_resp_rdata(l2_mresp_rdata), .error(bridge_error),
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

    logic comb_done;
    tb_npu_cache_hierarchy_comb u_comb (.clk(clk), .done(comb_done));

    integer seed = 2024;

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_cache_hierarchy: %s (t=%0t)", msg, $time);
        $fatal(1, "tb_npu_cache_hierarchy");
    endtask

    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    // ------------------------------------------------------------------
    // AXI4 slave memory: code region then data region in one byte array
    // ------------------------------------------------------------------
    localparam integer MEMB = CODE_SIZE + DATA_SIZE;
    logic [7:0] axmem [0:MEMB-1];
    logic [7:0] refm [0:MEMB-1];
    logic       zero_wait;
    integer     ready_pct;

    function automatic integer moff(input logic [31:0] a);
        if (a >= CODE_BASE && a < CODE_BASE + CODE_SIZE)
            moff = a - CODE_BASE;
        else if (a >= DATA_BASE && a < DATA_BASE + DATA_SIZE)
            moff = CODE_SIZE + (a - DATA_BASE);
        else
            moff = -1;
    endfunction

    logic        r_active, aw_got, b_pend;
    integer      r_beat, w_cnt, b_delay, ax_rd, ax_wr;
    logic [31:0] r_addr, aw_addr;
    logic [63:0] w_buf [0:3];

    always @(posedge clk) begin
        if (!rst_n) begin
            rvalid <= 1'b0;
            bvalid <= 1'b0;
            r_active = 1'b0;
            aw_got = 1'b0;
            b_pend = 1'b0;
            w_cnt = 0;
        end else begin
            if (arvalid && arready) begin
                if (moff(araddr) < 0 || araddr[4:0] != 0 || arlen != 3 || arsize != 3 ||
                    arburst != 2'b01 || arcache != 4'b0011 || arprot != 0 || r_active)
                    fail("AR burst");
                r_active = 1'b1;
                r_addr = araddr;
                r_beat = 0;
                ax_rd = ax_rd + 1;
            end
            if (rvalid && rready) begin
                r_beat = r_beat + 1;
                if (rlast) r_active = 1'b0;
            end
            if (r_active && (!rvalid || rready) && (zero_wait || rnd(100) < ready_pct)) begin
                rvalid <= 1'b1;
                for (int b = 0; b < 8; b++)
                    rdata[b*8 +: 8] <= axmem[moff(r_addr) + 8 * r_beat + b];
                rlast <= (r_beat == 3);
                rresp <= 2'b00;
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
            end
            if (awvalid && awready) begin
                if (moff(awaddr) < 0 || awaddr[4:0] != 0 || awlen != 3 || awsize != 3 ||
                    awburst != 2'b01 || awcache != 4'b0011 || awprot != 0 || aw_got)
                    fail("AW burst");
                if (awaddr < DATA_BASE) fail("write-back into the code region");
                aw_got = 1'b1;
                aw_addr = awaddr;
            end
            if (wvalid && wready) begin
                if (w_cnt >= 4 || wstrb != 8'hFF || wlast != (w_cnt == 3)) fail("W beat");
                w_buf[w_cnt] = wdata;
                w_cnt = w_cnt + 1;
            end
            if (!b_pend && aw_got && w_cnt == 4) begin
                b_pend = 1'b1;
                b_delay = zero_wait ? 0 : rnd(4);
            end
            if (bvalid && bready) begin
                bvalid <= 1'b0;
                for (int i = 0; i < 4; i++)
                    for (int b = 0; b < 8; b++)
                        axmem[moff(aw_addr) + 8 * i + b] = w_buf[i][b*8 +: 8];
                b_pend = 1'b0;
                aw_got = 1'b0;
                w_cnt = 0;
                ax_wr = ax_wr + 1;
            end else if (b_pend && !bvalid) begin
                if (b_delay == 0) begin
                    bvalid <= 1'b1;
                    bresp <= 2'b00;
                end else begin
                    b_delay = b_delay - 1;
                end
            end
        end
        arready <= zero_wait || rnd(100) < ready_pct;
        awready <= zero_wait || rnd(100) < ready_pct;
        wready <= zero_wait || rnd(100) < ready_pct;
    end

    // ------------------------------------------------------------------
    // CPU drivers
    // ------------------------------------------------------------------
    integer i_lat, d_lat;
    integer flush_req [0:2];      // flush pulses issued: I, D, L2
    integer flush_got [0:2];      // flush_done pulses seen
    logic   quiesce;              // CPUs pause while set

    always @(posedge clk) begin
        if (rst_n) begin
            if (i_flush_done) flush_got[0] = flush_got[0] + 1;
            if (d_flush_done) flush_got[1] = flush_got[1] + 1;
            if (l2_flush_done) flush_got[2] = flush_got[2] + 1;
            for (int k = 0; k < 3; k++)
                if (flush_got[k] > flush_req[k]) fail("flush_done without a flush");
            if (bridge_error) fail("bridge flag set");
        end
    end

    integer last_i, last_d;       // offsets of the last I fetch / D access

    task automatic i_fetch(input integer off);
        logic [31:0] exp;
        integer lat;
        last_i = off;
        i_req_valid = 1'b1;
        i_req_addr = CODE_BASE + off;
        while (!i_req_ready) @(negedge clk);
        @(negedge clk);
        i_req_valid = 1'b0;
        i_req_addr = $random(seed);
        for (int b = 0; b < 4; b++)
            exp[b*8 +: 8] = refm[off + b];
        lat = 1;
        while (!i_resp_valid) begin
            @(negedge clk);
            lat = lat + 1;
            if (lat > 5000) fail("I-side response missing");
        end
        if (i_resp_rdata !== exp) begin
            $display("I addr %h got %h exp %h", CODE_BASE + off, i_resp_rdata, exp);
            fail("instruction fetch data");
        end
        i_lat = lat;
        @(negedge clk);
    endtask

    task automatic d_access(input integer off, input logic we, input logic [3:0] strb,
                            input logic [31:0] wd);
        logic [31:0] exp;
        integer lat, o;
        o = CODE_SIZE + off;
        last_d = o;
        d_req_valid = 1'b1;
        d_req_addr = DATA_BASE + off;
        d_req_we = we;
        d_req_wstrb = strb;
        d_req_wdata = wd;
        while (!d_req_ready) @(negedge clk);
        @(negedge clk);
        d_req_valid = 1'b0;
        d_req_addr = $random(seed);
        d_req_wdata = $random(seed);
        if (we) begin
            for (int b = 0; b < 4; b++)
                if (strb[b]) refm[o + b] = wd[b*8 +: 8];
        end
        for (int b = 0; b < 4; b++)
            exp[b*8 +: 8] = refm[o + b];
        lat = 1;
        while (!d_resp_valid) begin
            @(negedge clk);
            lat = lat + 1;
            if (lat > 5000) fail("D-side response missing");
        end
        if (!we && d_resp_rdata !== exp) begin
            $display("D addr %h got %h exp %h", DATA_BASE + off, d_resp_rdata, exp);
            fail("load data");
        end
        d_lat = lat;
        @(negedge clk);
    endtask

    task automatic wait_flush(input integer k);
        integer n;
        n = 0;
        while (flush_got[k] < flush_req[k]) begin
            @(negedge clk);
            n = n + 1;
            if (n > 50000) fail("flush_done missing");
        end
    endtask

    // One-cycle flush pulse; a previous flush of the same cache must be done
    task automatic pulse_flush(input integer k);
        wait_flush(k);
        @(negedge clk);
        flush_req[k] = flush_req[k] + 1;
        case (k)
            0: i_flush = 1'b1;
            1: d_flush = 1'b1;
            default: l2_flush = 1'b1;
        endcase
        @(negedge clk);
        case (k)
            0: i_flush = 1'b0;
            1: d_flush = 1'b0;
            default: l2_flush = 1'b0;
        endcase
    endtask


    // Flush L1 I$, L1 D$ then L2, compare the AXI memory with the reference,
    // then change the memory behind the invalidated caches (host write)
    task automatic flush_and_check;
        integer o;
        pulse_flush(0);
        pulse_flush(1);
        wait_flush(1);
        pulse_flush(2);
        wait_flush(2);
        wait_flush(0);
        @(negedge clk);
        for (int i = 0; i < MEMB; i++)
            if (axmem[i] !== refm[i]) begin
                $display("memory offset %0d axi %h ref %h", i, axmem[i], refm[i]);
                fail("AXI memory differs from the reference after flush");
            end
        for (int k = 0; k < 8; k++) begin
            o = ((k < 4) ? last_i : last_d) & ~31;
            o = o + ((k % 4 == 0) ? 0 : rnd(32));
            axmem[o] = axmem[o] ^ 8'(1 + rnd(255));
            refm[o] = axmem[o];
        end
    endtask

    integer n_i, n_d, n_dw, n_chk, lat_l2hit, lat_l2miss, lat_l2dirty;

    task automatic i_stream;
        integer pc;
        pc = 0;
        for (int n = 0; n < NOPS; n++) begin
            while (quiesce) @(negedge clk);
            // mostly sequential fetch with random jumps
            if (rnd(8) == 0)
                pc = rnd(CODE_SIZE) & ~3;
            else
                pc = (pc + 4) % CODE_SIZE;
            i_fetch(pc);
            n_i++;
            if (rnd(400) == 0) begin
                pulse_flush(0);
                wait_flush(0);
            end
            repeat (rnd(2)) @(negedge clk);
        end
    endtask

    task automatic d_stream;
        integer off;
        logic [3:0] strb;
        for (int n = 0; n < NOPS; n++) begin
            while (quiesce) @(negedge clk);
            if (rnd(2) == 0)
                off = rnd(1024);              // hot window
            else
                off = rnd(DATA_SIZE);
            off = off & ~3;
            if (rnd(2) == 0) begin
                strb = rnd(16);
                d_access(off, 1'b1, strb, $random(seed));
                n_dw++;
            end else begin
                d_access(off, 1'b0, 4'b0, 32'b0);
            end
            n_d++;
            if (rnd(300) == 0)
                pulse_flush(1 + rnd(2));      // L1 D$ or L2 flush during traffic
            repeat (rnd(3)) @(negedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        i_req_valid = 1'b0;
        i_req_addr = '0;
        d_req_valid = 1'b0;
        d_req_addr = '0;
        d_req_we = 1'b0;
        d_req_wstrb = '0;
        d_req_wdata = '0;
        i_flush = 1'b0;
        d_flush = 1'b0;
        l2_flush = 1'b0;
        quiesce = 1'b0;
        zero_wait = 1'b1;
        ready_pct = 100;
        ax_rd = 0;
        ax_wr = 0;
        n_i = 0;
        n_d = 0;
        n_dw = 0;
        n_chk = 0;
        last_i = 0;
        last_d = 0;
        for (int k = 0; k < 3; k++) begin
            flush_req[k] = 0;
            flush_got[k] = 0;
        end
        for (int i = 0; i < MEMB; i++) begin
            axmem[i] = $random(seed);
            refm[i] = axmem[i];
        end
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (300) @(negedge clk);           // tag sweeps (L2: 128 cycles)

        // End-to-end zero-wait latencies on the D side
        d_access(0, 1'b0, 4'b0, 32'b0);        // L1 miss, L2 miss
        lat_l2miss = d_lat;
        d_access(1024, 1'b0, 4'b0, 32'b0);     // evicts line 0 from L1 (clean)
        d_access(0, 1'b0, 4'b0, 32'b0);        // L1 miss, L2 hit
        lat_l2hit = d_lat;
        d_access(0, 1'b1, 4'hF, 32'h1234_5678);   // dirty in L1
        d_access(1024, 1'b0, 4'b0, 32'b0);     // L1 dirty eviction, L2 hit
        lat_l2dirty = d_lat;
        $display("INFO tb_npu_cache_hierarchy: zero-wait L1 hit=1, L1 miss/L2 hit=%0d, L1 miss/L2 miss=%0d, L1 dirty miss/L2 hit=%0d cycles",
                 lat_l2hit, lat_l2miss, lat_l2dirty);
        // L1 clean miss 12 with the L2 answering in 3 (hit) / 10 + bridge
        // (miss); a dirty L1 victim adds its write-back (L2 write hit)
        if (lat_l2hit != 15 || lat_l2miss != 24 || lat_l2dirty != 28)
            fail("end-to-end zero-wait latency differs from the npu_cache header values");
        i_fetch(0);                            // line 0 cached in L1 I$ and L2
        d_access(0, 1'b0, 4'b0, 32'b0);
        flush_and_check();                     // ... then changed in memory
        n_chk++;
        i_fetch(0);                            // must see the new bytes
        i_fetch(4);
        d_access(0, 1'b0, 4'b0, 32'b0);
        d_access(4, 1'b0, 4'b0, 32'b0);

        // Random concurrent traffic with periodic quiescent checks
        zero_wait = 1'b0;
        ready_pct = 50;
        fork
            i_stream();
            d_stream();
            begin
                while (n_i < NOPS || n_d < NOPS) begin
                    repeat (3000) @(negedge clk);
                    ready_pct = 20 + rnd(81);
                    zero_wait = rnd(4) == 0;
                    if (n_i < NOPS && n_d < NOPS) begin
                        quiesce = 1'b1;
                        // let both outstanding requests and flushes finish
                        repeat (200) @(negedge clk);
                        wait_flush(0);
                        wait_flush(1);
                        wait_flush(2);
                        flush_and_check();
                        n_chk++;
                        quiesce = 1'b0;
                    end
                end
            end
        join
        wait_flush(0);
        wait_flush(1);
        wait_flush(2);
        flush_and_check();
        n_chk++;
        $display("INFO tb_npu_cache_hierarchy: %0d fetches, %0d data accesses (%0d stores), %0d AXI reads, %0d AXI writes, %0d memory checks",
                 n_i, n_d, n_dw, ax_rd, ax_wr, n_chk);
        $display("INFO tb_npu_cache_hierarchy: hits/misses L1I %0d/%0d L1D %0d/%0d L2 %0d/%0d",
                 i_hits, i_misses, d_hits, d_misses, l2_hits, l2_misses);
        wait (comb_done);
        $display("PASS tb_npu_cache_hierarchy %0d fetches %0d data accesses, combinational-requester harness", n_i, n_d);
        $finish;
    end

    initial begin
        #50_000_000;
        $display("MISMATCH tb_npu_cache_hierarchy: timeout");
        $fatal(1, "timeout");
    end
endmodule

// Combinational-requester harness (review round 3 regression): the L1 I$ and
// L1 D$ (CPU_HOLD = 1, as in npu_hx_subsystem) are driven by requesters that
// compute req_valid / req_addr / req_we / req_wdata in an always_comb with
// default-then-override assignments and present the next request in the
// cycle of a read response, as the scalar core does. Behind them:
// npu_mem_arbiter -> L2 (the npu_hx_subsystem configuration) -> a line
// memory with random timing. An event-driven simulator re-runs such a block
// whenever an input changes and passes its intermediate defaults on, so if
// an npu_cache output were written more than once per evaluation, every
// re-run would glitch it and the requester and the cache would wake each
// other forever within one time step (seen in npu_hx_subsystem on the first
// L1 D$ dirty write-back). A counter of requester evaluations per time step
// turns such a loop into a MISMATCH instead of a hang. The sequence starts
// with L1 D$ dirty write-backs issued while the L2 is idle and the I side is
// quiet, then runs random concurrent traffic; every read is checked.
module tb_npu_cache_hierarchy_comb #(
    parameter integer NOPS = 600
) (
    input  logic clk,
    output logic done
);
    localparam integer LW = 256;
    localparam integer DREG = 4096;          // D region bytes (4 L1 D$ ways)
    localparam integer IREG = 2048;          // I region bytes
    localparam logic [31:0] IBASE = 32'h0000_8000;

    logic rst_n;
    logic          i_req_valid, i_req_ready, i_resp_valid, i_fd;
    logic [31:0]   i_req_addr, i_resp_rdata, i_h, i_m;
    logic          d_req_valid, d_req_ready, d_req_we, d_resp_valid, d_fd;
    logic [31:0]   d_req_addr, d_req_wdata, d_resp_rdata, d_h, d_m;
    logic [3:0]    d_req_wstrb;
    logic          i_mv, i_mr, i_mwe, i_mrv, d_mv, d_mr, d_mwe, d_mrv;
    logic [31:0]   i_ma, d_ma, i_mrd, d_mrd;
    logic [LW-1:0] i_mwd, d_mwd;
    logic          a_v, a_r, a_we, a_rv;
    logic [31:0]   a_addr, a_rd;
    logic [LW-1:0] a_wd;
    logic          l2_mv, l2_mr, l2_mwe, l2_mrv, l2_fd;
    logic [31:0]   l2_ma, l2_h, l2_m;
    logic [LW-1:0] l2_mwd;
    logic [63:0]   l2_mrd;

    npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .READ_ONLY(1)) l1i (
        .clk(clk), .rst_n(rst_n),
        .req_valid(i_req_valid), .req_ready(i_req_ready), .req_addr(i_req_addr),
        .req_we(1'b0), .req_wstrb(4'b0), .req_wdata(32'b0),
        .resp_valid(i_resp_valid), .resp_rdata(i_resp_rdata), .flush(1'b0), .flush_done(i_fd),
        .mem_req_valid(i_mv), .mem_req_ready(i_mr), .mem_req_addr(i_ma), .mem_req_we(i_mwe),
        .mem_req_wdata(i_mwd), .mem_resp_valid(i_mrv), .mem_resp_rdata(i_mrd),
        .hits(i_h), .misses(i_m));
    npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(32), .WRITE_BACK(1), .CPU_HOLD(1)) l1d (
        .clk(clk), .rst_n(rst_n),
        .req_valid(d_req_valid), .req_ready(d_req_ready), .req_addr(d_req_addr),
        .req_we(d_req_we), .req_wstrb(d_req_wstrb), .req_wdata(d_req_wdata),
        .resp_valid(d_resp_valid), .resp_rdata(d_resp_rdata), .flush(1'b0), .flush_done(d_fd),
        .mem_req_valid(d_mv), .mem_req_ready(d_mr), .mem_req_addr(d_ma), .mem_req_we(d_mwe),
        .mem_req_wdata(d_mwd), .mem_resp_valid(d_mrv), .mem_resp_rdata(d_mrd),
        .hits(d_h), .misses(d_m));
    npu_mem_arbiter #(.LINE_BYTES(32), .BEAT_W(32), .P0_READ_ONLY(1)) arb (
        .clk(clk), .rst_n(rst_n),
        .s0_mem_req_valid(i_mv), .s0_mem_req_ready(i_mr), .s0_mem_req_addr(i_ma),
        .s0_mem_req_we(i_mwe), .s0_mem_req_wdata(i_mwd), .s0_mem_resp_valid(i_mrv),
        .s0_mem_resp_rdata(i_mrd),
        .s1_mem_req_valid(d_mv), .s1_mem_req_ready(d_mr), .s1_mem_req_addr(d_ma),
        .s1_mem_req_we(d_mwe), .s1_mem_req_wdata(d_mwd), .s1_mem_resp_valid(d_mrv),
        .s1_mem_resp_rdata(d_mrd),
        .m_mem_req_valid(a_v), .m_mem_req_ready(a_r), .m_mem_req_addr(a_addr),
        .m_mem_req_we(a_we), .m_mem_req_wdata(a_wd), .m_mem_resp_valid(a_rv),
        .m_mem_resp_rdata(a_rd));
    npu_cache #(.SIZE_BYTES(8192), .LINE_BYTES(32), .WAYS(2), .CPU_W(LW), .WRITE_BACK(1),
                .CPU_HOLD(1), .MEM_W(64), .RESP_W(32)) l2 (
        .clk(clk), .rst_n(rst_n),
        .req_valid(a_v), .req_ready(a_r), .req_addr(a_addr), .req_we(a_we),
        .req_wstrb({(LW/8){1'b1}}), .req_wdata(a_wd), .resp_valid(a_rv), .resp_rdata(a_rd),
        .flush(1'b0), .flush_done(l2_fd),
        .mem_req_valid(l2_mv), .mem_req_ready(l2_mr), .mem_req_addr(l2_ma), .mem_req_we(l2_mwe),
        .mem_req_wdata(l2_mwd), .mem_resp_valid(l2_mrv), .mem_resp_rdata(l2_mrd),
        .hits(l2_h), .misses(l2_m));

    integer seed = 909;
    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction
    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_cache_hierarchy comb: %s (t=%0t)", msg, $time);
        $fatal(1, "tb_npu_cache_hierarchy comb");
    endtask

    // ---------------- line memory (64-bit beats, random timing) ----------
    logic [7:0] mem [0:65535];
    logic       busy, mwe;
    integer     beat, mdel, ready_pct, n_l2wr, n_l1wb_idle;
    logic [15:0] ma;
    always @(posedge clk) begin
        l2_mrv <= 1'b0;
        l2_mr <= rnd(100) < ready_pct;
        if (!rst_n) begin
            busy = 1'b0;
        end else if (l2_mv && l2_mr) begin
            if (busy) fail("second L2 memory request");
            busy = 1'b1; beat = 0; ma = l2_ma[15:0]; mwe = l2_mwe; mdel = rnd(3);
            if (mwe) begin
                for (int b = 0; b < 32; b++) mem[ma + b] = l2_mwd[b*8 +: 8];
                n_l2wr++;
            end
        end else if (busy) begin
            if (mdel > 0) begin
                mdel--;
            end else begin
                l2_mrv <= 1'b1;
                for (int b = 0; b < 8; b++) l2_mrd[b*8 +: 8] <= mem[ma + beat * 8 + b];
                beat++;
                if (mwe || beat == 4) busy = 1'b0;
                else mdel = rnd(2);
            end
        end
        // L1 D$ write-back accepted by an idle L2 (the regression scenario)
        if (rst_n && a_v && a_r && a_we && arb.grant_q && l2.state_q == 4'd1)
            n_l1wb_idle++;
    end

    // ---------------- operation lists ----------------
    logic [31:0] d_op_addr [0:NOPS-1];
    logic        d_op_we   [0:NOPS-1];
    logic [3:0]  d_op_strb [0:NOPS-1];
    logic [31:0] d_op_wdata[0:NOPS-1];
    logic [31:0] i_op_addr [0:NOPS-1];
    logic [7:0]  refm [0:DREG-1];
    logic [31:0] d_exp, i_exp;

    // ---------------- requesters (registered state) ----------------
    integer d_idx_q, i_idx_q;     // next operation to present
    logic   d_busy_q, d_we_q, i_busy_q;
    // Requester evaluations, counted inside the always_comb blocks (an
    // event-driven loop may re-run them without changing any net value)
    integer d_evals = 0, i_evals = 0, d_base, i_base;
    time    eval_t;

    always @(d_evals, i_evals) begin
        if ($time != eval_t) begin
            eval_t = $time;
            d_base = d_evals;
            i_base = i_evals;
        end
        if (d_evals - d_base > 500 || i_evals - i_base > 500)
            fail("zero-time evaluation loop through the requesters, caches and arbiter");
    end

    // D requester: default-then-override, next request in the read response
    // cycle (CPU_HOLD: after a write only once its response is over)
    always_comb begin
        d_evals = d_evals + 1;
        d_req_valid = 1'b0;
        d_req_addr = 32'hFFFF_FFF0;
        d_req_we = 1'b0;
        d_req_wstrb = 4'h0;
        d_req_wdata = 32'h0;
        if (d_busy_q) begin
            d_req_we = d_we_q;               // CPU_HOLD: the held write
            d_req_wstrb = d_op_strb[d_idx_q - 1];
            d_req_wdata = d_op_wdata[d_idx_q - 1];
        end
        if (rst_n && d_idx_q < NOPS && (!d_busy_q || (d_resp_valid && !d_we_q))) begin
            d_req_valid = 1'b1;
            d_req_addr = d_op_addr[d_idx_q];
            d_req_we = d_op_we[d_idx_q];
            d_req_wstrb = d_op_strb[d_idx_q];
            d_req_wdata = d_op_wdata[d_idx_q];
        end
    end
    // I requester: starts once the directed D sequence is done
    always_comb begin
        i_evals = i_evals + 1;
        i_req_valid = 1'b0;
        i_req_addr = 32'hFFFF_FFF0;
        if (rst_n && i_idx_q < NOPS && d_idx_q >= 6 && (!i_busy_q || i_resp_valid)) begin
            i_req_valid = 1'b1;
            i_req_addr = i_op_addr[i_idx_q];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            d_idx_q <= 0; d_busy_q <= 1'b0; d_we_q <= 1'b0;
            i_idx_q <= 0; i_busy_q <= 1'b0;
        end else begin
            if (d_resp_valid) begin
                if (!d_busy_q) fail("D response without a request");
                if (!d_we_q && d_resp_rdata !== d_exp) begin
                    $display("D read %h got %h exp %h", d_op_addr[d_idx_q - 1], d_resp_rdata, d_exp);
                    fail("D read data");
                end
                d_busy_q <= 1'b0;
            end
            if (d_req_valid && d_req_ready) begin
                if (d_busy_q) fail("D accepted while busy");
                if (d_req_we)
                    for (int b = 0; b < 4; b++)
                        if (d_req_wstrb[b]) refm[d_req_addr[11:0] + b] = d_req_wdata[b*8 +: 8];
                for (int b = 0; b < 4; b++) d_exp[b*8 +: 8] = refm[d_req_addr[11:2] * 4 + b];
                d_idx_q <= d_idx_q + 1; d_busy_q <= 1'b1; d_we_q <= d_req_we;
            end
            if (i_resp_valid) begin
                if (!i_busy_q) fail("I response without a request");
                if (i_resp_rdata !== i_exp) fail("I read data");
                i_busy_q <= 1'b0;
            end
            if (i_req_valid && i_req_ready) begin
                if (i_busy_q) fail("I accepted while busy");
                for (int b = 0; b < 4; b++) i_exp[b*8 +: 8] = mem[i_req_addr[15:2] * 4 + b];
                i_idx_q <= i_idx_q + 1; i_busy_q <= 1'b1;
            end
        end
    end

    initial begin
        done = 1'b0;
        eval_t = 0;
        d_base = 0;
        i_base = 0;
        n_l2wr = 0;
        n_l1wb_idle = 0;
        ready_pct = 100;
        for (int i = 0; i < 65536; i++) mem[i] = $random(seed);
        for (int i = 0; i < DREG; i++) refm[i] = mem[i];
        // directed: dirty line 0, conflict 0x400 (dirty write-back of line 0
        // with the L2 idle), back to 0 (write-back of 0x400), two reads
        d_op_addr[0] = 32'h000; d_op_we[0] = 1'b1;
        d_op_addr[1] = 32'h400; d_op_we[1] = 1'b1;
        d_op_addr[2] = 32'h000; d_op_we[2] = 1'b0;
        d_op_addr[3] = 32'h800; d_op_we[3] = 1'b0;
        d_op_addr[4] = 32'h404; d_op_we[4] = 1'b0;
        d_op_addr[5] = 32'h004; d_op_we[5] = 1'b0;
        for (int k = 0; k < 6; k++) begin
            d_op_strb[k] = 4'hF;
            d_op_wdata[k] = $random(seed);
        end
        for (int k = 6; k < NOPS; k++) begin
            d_op_addr[k] = rnd(DREG) & ~3;
            d_op_we[k] = rnd(2);
            d_op_strb[k] = rnd(16);
            d_op_wdata[k] = $random(seed);
        end
        for (int k = 0; k < NOPS; k++)
            i_op_addr[k] = IBASE + (rnd(IREG) & ~3);
        rst_n = 1'b0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        while (d_idx_q < 6 || d_busy_q) @(negedge clk);
        ready_pct = 60;
        while (d_idx_q < NOPS || i_idx_q < NOPS || d_busy_q || i_busy_q) @(negedge clk);
        if (n_l1wb_idle == 0) fail("no L1 D$ write-back reached an idle L2");
        $display("INFO tb_npu_cache_hierarchy comb: %0d D + %0d I requests from combinational requesters, %0d L1 D$ write-backs into an idle L2",
                 NOPS, NOPS, n_l1wb_idle);
        done = 1'b1;
    end
endmodule
