`timescale 1ns/1ps

// Self-checking test of npu_cache in eight configurations, each in its own
// harness (own reset, CPU driver, flat reference memory and behavioural
// line-protocol memory model):
//   l1d : 1 KB, 32 B lines, direct mapped, write-back      (L1 D$)
//   l1i : 1 KB, 32 B lines, direct mapped, read-only       (L1 I$)
//   l2  : 8 KB, 32 B lines, 2-way LRU, CPU_W = 256, write-back (L2)
//   l1wt: 1 KB, 32 B lines, direct mapped, write-through  (WRITE_BACK = 0)
//   l2hold: the L2 configuration with CPU_HOLD = 1 (write data held by the
//         driver until resp_valid, as npu_mem_arbiter does)
//   dm4k: 4 KB direct mapped, write-back, CPU_HOLD = 1: 128 sets, so the tag
//         RAM is a BRAM and the compare is registered (TREG with one way)
//   w2s : 1 KB 2-way write-back with 32-bit words (2-way LRU with multi-word
//         eviction from the hit / victim way; LUTRAM tags, no TREG)
//   m16 : 1 KB direct mapped write-back, 64-bit words, 16-bit memory read
//         beats and 16-bit CPU read beats (MEM_W = RESP_W = 16): several
//         refill beats per data word and several data words per line, and
//         the read beats of a direct-mapped cache leave from state RESP
// The L1 configurations use 32-bit memory read beats (MEM_W = CPU_W); both L2
// configurations use 64-bit memory read beats and answer CPU reads with
// eight 32-bit beats (MEM_W = 64, RESP_W = 32), as in the hierarchy.
// Each harness runs: reset behaviour; zero-wait latency measurement; a
// directed hit/miss counter sequence (conflict misses, dirty eviction, LRU for
// 2-way, full-line write allocate for L2); random read/write traffic with
// random byte strobes, memory backpressure and response delays, random
// flushes (some issued while a request is outstanding) each followed by a
// backing-memory check; a mid-run reset; a final flush and memory check.
// The random phase also presents the next request in the cycle of the last
// response pulse (while req_ready is low), sends a second flush pulse while a
// flush is running (it must merge: one flush_done), and runs the memory side
// in a mode where mem_req_ready rises only after mem_req_valid (a valid that
// waited for ready would deadlock). The l1d and l2hold harnesses start at
// address 0, so tag 0 (the tag of an invalidated entry) is exercised.
// After every completed flush the memory side changes bytes in the lines of
// the last eight accesses behind the cache (as the host does between FLUSH
// and the next start); the reference follows, so a flush that does not
// invalidate a line is caught by the next read of it (directed: line 0 is
// re-read right after the first flush).
// Protocol checks run every cycle: one outstanding memory request, line
// alignment, request payload stable until the response is complete (all read
// beats delivered, with random gaps between beats), one resp_valid per
// accepted write and RRB consecutive resp_valid beats per accepted read, no
// request accepted while a response is incomplete, flush_done only after a
// flush. Zero-wait hit and miss latencies are checked against the values in
// the npu_cache header.

module tb_npu_cache_harness #(
    parameter NAME = "l1d",
    parameter integer SIZE_BYTES = 1024,
    parameter integer WAYS = 1,
    parameter integer CPU_W = 32,
    parameter integer WRITE_BACK = 1,
    parameter integer READ_ONLY = 0,
    parameter integer CPU_HOLD = 0,
    parameter integer MEM_W = CPU_W,
    parameter integer RESP_W = CPU_W,
    parameter logic [31:0] BASE = 32'h0010_0000,
    parameter integer REGION = 4096,       // bytes of address space exercised
    parameter integer NOPS = 3000,
    parameter integer SEED = 1
) (
    input  logic clk,
    input  logic start,
    output logic done
);
    localparam integer LINE_BYTES = 32;
    localparam integer LINE_W = LINE_BYTES * 8;
    localparam integer SB = CPU_W / 8;
    localparam integer SETS = SIZE_BYTES / (LINE_BYTES * WAYS);
    localparam integer WAY_STRIDE = SETS * LINE_BYTES;   // same set, next tag
    localparam integer MB = MEM_W / 8;                    // bytes per read beat
    localparam integer RBEATS = LINE_W / MEM_W;           // read beats per line
    localparam integer RRB = CPU_W / RESP_W;              // CPU read beats
    localparam integer WPL = LINE_W / CPU_W;
    localparam integer TREG = (SETS > 64) ? 1 : 0;        // BRAM tag RAM
    localparam integer R1 = (WAYS == 1 && RRB > 1) ? 1 : 0;  // beats from RESP

    logic              rst_n;
    logic              req_valid, req_ready, req_we;
    logic [31:0]       req_addr;
    logic [SB-1:0]     req_wstrb;
    logic [CPU_W-1:0]  req_wdata;
    logic              resp_valid;
    logic [RESP_W-1:0] resp_rdata;
    logic              flush, flush_done;
    logic              mem_req_valid, mem_req_ready, mem_req_we;
    logic [31:0]       mem_req_addr;
    logic [LINE_W-1:0] mem_req_wdata;
    logic              mem_resp_valid;
    logic [MEM_W-1:0]  mem_resp_rdata;
    logic [31:0]       hits, misses;

    npu_cache #(
        .SIZE_BYTES(SIZE_BYTES), .LINE_BYTES(LINE_BYTES), .WAYS(WAYS),
        .CPU_W(CPU_W), .WRITE_BACK(WRITE_BACK), .READ_ONLY(READ_ONLY),
        .CPU_HOLD(CPU_HOLD), .MEM_W(MEM_W), .RESP_W(RESP_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .req_valid(req_valid), .req_ready(req_ready), .req_addr(req_addr),
        .req_we(req_we), .req_wstrb(req_wstrb), .req_wdata(req_wdata),
        .resp_valid(resp_valid), .resp_rdata(resp_rdata),
        .flush(flush), .flush_done(flush_done),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_we(mem_req_we),
        .mem_req_wdata(mem_req_wdata), .mem_resp_valid(mem_resp_valid),
        .mem_resp_rdata(mem_resp_rdata), .hits(hits), .misses(misses)
    );

    integer seed = SEED;
    logic [7:0] bmem [0:REGION-1];      // backing memory (memory-side view)
    logic [7:0] refm [0:REGION-1];      // reference (CPU view)

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_cache %s: %s (t=%0t)", NAME, msg, $time);
        $fatal(1, "tb_npu_cache %s", NAME);
    endtask

    function automatic integer rnd(input integer n);   // 0..n-1
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    // ------------------------------------------------------------------
    // Memory-side model
    // ------------------------------------------------------------------
    logic              zero_wait;       // ready = 1, response the next cycle
    integer            ready_pct;       // mem_req_ready probability (random mode)
    integer            max_delay;
    logic              m_busy;
    integer            m_delay;
    logic [31:0]       m_addr;
    logic              m_we;
    logic [LINE_W-1:0] m_wdata;
    integer            mem_rd_cnt, mem_wr_cnt;

    integer            m_beat;          // next read beat
    integer            gap_pct;         // chance of a gap before a read beat
    logic              ready_after_valid;  // mem_req_ready only after mem_req_valid

    // One memory step: the write response, or the next read beat
    task automatic mem_step;
        integer off;
        off = m_addr - BASE;
        if (m_we) begin
            for (int b = 0; b < LINE_BYTES; b++)
                bmem[off + b] = m_wdata[b*8 +: 8];
            mem_wr_cnt = mem_wr_cnt + 1;
            mem_resp_valid <= 1'b1;
            m_busy <= 1'b0;
        end else begin
            for (int b = 0; b < MB; b++)
                mem_resp_rdata[b*8 +: 8] <= bmem[off + m_beat * MB + b];
            mem_resp_valid <= 1'b1;
            m_beat = m_beat + 1;
            if (m_beat == RBEATS) begin
                mem_rd_cnt = mem_rd_cnt + 1;
                m_busy <= 1'b0;
            end else begin
                // 0 or 1: next beat in the next cycle; n > 1: n - 1 idle cycles
                m_delay = (zero_wait || rnd(100) >= gap_pct) ? 0 : rnd(max_delay + 2);
            end
        end
    endtask

    always @(posedge clk) begin
        mem_resp_valid <= 1'b0;
        mem_resp_rdata <= {MEM_W/8{8'($random(seed))}};   // garbage when not valid
        if (!rst_n) begin
            m_busy <= 1'b0;
        end else if (mem_req_valid && mem_req_ready) begin
            if (m_busy) fail("second memory request while one is outstanding");
            if (mem_req_addr[4:0] != 5'd0) fail("memory request not line aligned");
            if (mem_req_addr < BASE || mem_req_addr >= BASE + REGION)
                fail("memory request outside the exercised region");
            if (READ_ONLY != 0 && mem_req_we) fail("read-only cache issued a write");
            m_addr = mem_req_addr;
            m_we = mem_req_we;
            m_wdata = mem_req_wdata;
            m_beat = 0;
            m_busy <= 1'b1;
            m_delay = zero_wait ? 0 : rnd(max_delay + 1);
            if (m_delay == 0)
                mem_step();
        end else if (m_busy) begin
            if (mem_req_valid) fail("mem_req_valid while a request is outstanding");
            if (mem_req_addr !== m_addr || mem_req_we !== m_we ||
                (m_we && mem_req_wdata !== m_wdata))
                fail("memory request payload changed before its response");
            m_delay = m_delay - 1;
            if (m_delay <= 0)
                mem_step();
        end
        if (ready_after_valid)
            mem_req_ready <= mem_req_valid && !mem_req_ready && (rnd(100) < ready_pct);
        else
            mem_req_ready <= zero_wait ? 1'b1 : (rnd(100) < ready_pct);
    end

    // ------------------------------------------------------------------
    // CPU-side monitor
    // ------------------------------------------------------------------
    // Response monitor: an accepted write expects one resp_valid, an
    // accepted read RRB resp_valid beats in consecutive cycles
    integer acc_cnt, resp_cnt, flush_cnt, flush_done_cnt;
    integer mon_left;                 // resp_valid pulses still expected
    logic   mon_beats;                // between the first and last read beat
    always @(posedge clk) begin
        if (!rst_n) begin
            acc_cnt <= 0;
            resp_cnt <= 0;
            mon_left = 0;
            mon_beats = 1'b0;
        end else begin
            if (resp_valid) begin
                if (mon_left == 0) fail("resp_valid without an accepted request");
                mon_left = mon_left - 1;
                mon_beats = (mon_left != 0);
                if (mon_left == 0)
                    resp_cnt <= resp_cnt + 1;
            end else if (mon_beats) begin
                fail("read response beats not in consecutive cycles");
            end
            if (req_valid && req_ready) begin
                if (mon_left != 0) fail("request accepted while a response is incomplete");
                acc_cnt <= acc_cnt + 1;
                mon_left = req_we ? 1 : RRB;
            end
            if (flush_done) begin
                if (flush_done_cnt >= flush_cnt) fail("flush_done without a flush");
                flush_done_cnt = flush_done_cnt + 1;
            end
        end
    end

    // ------------------------------------------------------------------
    // CPU driver: all inputs change at negedge, outputs sampled at negedge
    // ------------------------------------------------------------------
    integer last_lat;
    logic [CPU_W-1:0] last_rdata;
    integer recent [0:7];   // offsets of the last eight accesses
    integer recent_wp;
    logic early_next;       // return in the last response cycle (next request early)

    task automatic cpu_access(input logic [31:0] addr, input logic we,
                              input logic [SB-1:0] strb, input logic [CPU_W-1:0] wdata,
                              input integer flush_at);
        integer off, lat;
        logic [CPU_W-1:0] exp;
        off = (addr - BASE) & ~(SB - 1);
        recent[recent_wp] = off;
        recent_wp = (recent_wp + 1) % 8;
        req_valid = 1'b1;
        req_addr = addr;
        req_we = we;
        req_wstrb = strb;
        req_wdata = wdata;
        while (!req_ready) @(negedge clk);
        @(negedge clk);                       // accepted at the posedge
        req_valid = 1'b0;
        req_addr = $random(seed);             // junk: the request was registered
        req_we = 1'b0;
        if (CPU_HOLD == 0) begin
            req_wstrb = $random(seed);
            req_wdata = {CPU_W/32{$random(seed)}};
        end
        // Reference update and expected read data (one request outstanding)
        if (we && READ_ONLY == 0) begin
            for (int b = 0; b < SB; b++)
                if (strb[b]) refm[off + b] = wdata[b*8 +: 8];
        end
        for (int b = 0; b < SB; b++)
            exp[b*8 +: 8] = refm[off + b];
        lat = 1;
        while (!resp_valid) begin
            if (lat == flush_at) begin
                flush = 1'b1;
                flush_cnt = flush_cnt + 1;
            end
            @(negedge clk);
            flush = 1'b0;
            lat = lat + 1;
            if (lat > 2000) fail("no response");
        end
        last_lat = lat;
        if (!we) begin
            // RRB beats, one per cycle, beat k = bits [k*RESP_W +: RESP_W]
            for (int k = 0; k < RRB; k++) begin
                if (k > 0) begin
                    @(negedge clk);
                    if (!resp_valid) fail("read beat missing");
                end
                if (resp_rdata !== exp[k*RESP_W +: RESP_W]) begin
                    $display("addr %h beat %0d got %h exp %h", addr, k, resp_rdata,
                             exp[k*RESP_W +: RESP_W]);
                    fail("read data");
                end
                last_rdata[k*RESP_W +: RESP_W] = resp_rdata;
            end
        end
        if (early_next && !(CPU_HOLD != 0 && we)) begin
            // the caller presents the next request in this response cycle
            // (not after a CPU_HOLD write: its data is held through resp_valid)
            early_next = 1'b0;
        end else begin
            early_next = 1'b0;
            @(negedge clk);
            req_wstrb = $random(seed);        // CPU_HOLD: held until resp_valid
            req_wdata = {CPU_W/32{$random(seed)}};
            if (resp_valid) fail("extra resp_valid after the response");
        end
    endtask

    task automatic rd(input integer off);
        cpu_access(BASE + off, 1'b0, '0, '0, 0);
    endtask
    task automatic wr(input integer off, input logic [CPU_W-1:0] d);
        cpu_access(BASE + off, 1'b1, '1, d, 0);
    endtask

    task automatic do_flush;
        integer n;
        @(negedge clk);
        flush = 1'b1;
        flush_cnt = flush_cnt + 1;
        @(negedge clk);
        flush = 1'b0;
        wait_flush();
    endtask

    // Second pulse while the first flush is running: merged, one flush_done
    // (a re-armed flush would pulse flush_done again within the wait below)
    task automatic do_double_flush;
        @(negedge clk);
        flush = 1'b1;
        flush_cnt = flush_cnt + 1;
        @(negedge clk);
        flush = 1'b0;
        repeat (2 + rnd(SETS - 4)) @(negedge clk);
        flush = 1'b1;
        @(negedge clk);
        flush = 1'b0;
        wait_flush();
        repeat (4 * SETS + 64) @(negedge clk);
    endtask

    task automatic wait_flush;
        integer n;
        n = 0;
        while (flush_done_cnt < flush_cnt) begin
            @(negedge clk);
            n = n + 1;
            if (n > 20000) fail("flush_done missing");
        end
        @(negedge clk);
        if (READ_ONLY == 0)
            check_backing();
        host_write();
    endtask

    // Memory changed behind the flushed (all invalid) cache: a byte of each
    // recently accessed word and another byte of its line
    task automatic host_write;
        integer o;
        for (int i = 0; i < 8; i++) begin
            o = recent[i] + rnd(SB);
            bmem[o] = bmem[o] ^ 8'(1 + rnd(255));
            refm[o] = bmem[o];
            o = (recent[i] & ~(LINE_BYTES - 1)) + rnd(LINE_BYTES);
            bmem[o] = bmem[o] ^ 8'(1 + rnd(255));
            refm[o] = bmem[o];
        end
    endtask

    task automatic check_backing;
        for (int i = 0; i < REGION; i++)
            if (bmem[i] !== refm[i]) begin
                $display("offset %0d mem %h ref %h", i, bmem[i], refm[i]);
                fail("backing memory differs from reference after flush");
            end
    endtask

    task automatic expect_counts(input integer h, input integer m, input integer mr,
                                 input integer mw, input string what);
        if (hits !== h || misses !== m || mem_rd_cnt != mr || mem_wr_cnt != mw) begin
            $display("%s: hits %0d/%0d misses %0d/%0d mem rd %0d/%0d wr %0d/%0d", what,
                     hits, h, misses, m, mem_rd_cnt, mr, mem_wr_cnt, mw);
            fail("counter check");
        end
    endtask

    task automatic do_reset;
        @(negedge clk);
        rst_n = 1'b0;
        repeat (3) @(negedge clk);
        if (req_ready || resp_valid || mem_req_valid || flush_done)
            fail("outputs active during reset");
        if (hits !== 0 || misses !== 0) fail("counters not cleared by reset");
        rst_n = 1'b1;
        mem_rd_cnt = 0;
        mem_wr_cnt = 0;
        // tag sweep: SETS cycles with req_ready low
        for (int i = 0; i < SETS - 1; i++) begin
            @(negedge clk);
            if (req_ready) fail("req_ready during the reset sweep");
        end
        while (!req_ready) @(negedge clk);
    endtask

    function automatic logic [CPU_W-1:0] rnd_word();
        logic [CPU_W-1:0] w;
        for (int i = 0; i < CPU_W / 32; i++)
            w[i*32 +: 32] = $random(seed);
        rnd_word = w;
    endfunction

    // ------------------------------------------------------------------
    // Test sequence
    // ------------------------------------------------------------------
    integer h0, m0, r0, w0, op, off, lat_hit, lat_mc, lat_md, lat_wh, lat_wm;
    integer n_rd, n_wr, n_fl;
    logic [SB-1:0] strb;
    logic early;

    initial begin
        rst_n = 1'b0;
        req_valid = 1'b0;
        req_addr = '0;
        req_we = 1'b0;
        req_wstrb = '0;
        req_wdata = '0;
        flush = 1'b0;
        zero_wait = 1'b1;
        ready_pct = 100;
        max_delay = 0;
        gap_pct = 0;
        ready_after_valid = 1'b0;
        early_next = 1'b0;
        done = 1'b0;
        flush_cnt = 0;
        flush_done_cnt = 0;
        mem_rd_cnt = 0;
        mem_wr_cnt = 0;
        lat_md = 0;
        lat_wh = 0;
        lat_wm = 0;
        for (int i = 0; i < REGION; i++) begin
            bmem[i] = $random(seed);
            refm[i] = bmem[i];
        end
        for (int i = 0; i < 8; i++)
            recent[i] = 0;
        recent_wp = 0;
        wait (start);
        do_reset();

        // ---------------- zero-wait latencies ----------------
        rd(0);
        lat_mc = last_lat;
        rd(0);
        lat_hit = last_lat;
        if (READ_ONLY == 0) begin
            wr(0, rnd_word());
            lat_wh = last_lat;
            if (WRITE_BACK != 0) begin
                // dirty line 0 must be evicted by a conflicting miss
                if (WAYS == 2) begin
                    wr(WAY_STRIDE, rnd_word());      // fills the other way, dirty
                    rd(0);                           // way of line 0 becomes MRU
                    rd(2 * WAY_STRIDE);              // evicts the LRU (dirty) way
                end else begin
                    rd(WAY_STRIDE);
                end
                lat_md = last_lat;
            end else begin
                wr(WAY_STRIDE, rnd_word());          // write-through write miss
                lat_wm = last_lat;
            end
        end
        // line 0 is cached (or was evicted clean); after the flush the memory
        // side changes it, and the read must miss and return the new bytes
        rd(0);
        for (int i = 0; i < 8; i++)
            recent[i] = 0;
        $display("INFO tb_npu_cache %s: zero-wait latency hit=%0d clean_miss=%0d dirty_miss=%0d write_hit=%0d wt_write_miss=%0d (cycles from acceptance to resp_valid)",
                 NAME, lat_hit, lat_mc, lat_md, lat_wh, lat_wm);
        // expected zero-wait latencies (npu_cache header): read hit 1 (+1 for
        // 2 ways, +1 for a BRAM tag RAM: TREG, +R1), write hit 1 + TREG,
        // clean read miss RBEATS + 4 (+1 for 2 ways, +TREG, +R1), dirty
        // + WPL + 3; write-through write hit WPL + 5 + TREG, miss RBEATS + 5
        // + TREG
        if (lat_hit != WAYS + TREG + R1) fail("hit latency");
        if (lat_mc != RBEATS + 4 + WAYS - 1 + TREG + R1) fail("clean miss latency");
        if (WRITE_BACK != 0 && READ_ONLY == 0 && lat_wh != 1 + TREG)
            fail("write hit latency");
        if (WRITE_BACK != 0 && READ_ONLY == 0 &&
            lat_md != RBEATS + 4 + WAYS - 1 + TREG + R1 + WPL + 3)
            fail("dirty miss latency");
        if (WRITE_BACK == 0 && READ_ONLY == 0 &&
            (lat_wh != WPL + 5 + TREG || lat_wm != RBEATS + 5 + TREG))
            fail("write-through write latency");
        do_flush();
        m0 = misses;
        rd(0);
        if (misses !== m0 + 1) fail("read after flush did not miss");
        do_flush();             // empty cache for the directed sequence

        // ---------------- directed counter sequence ----------------
        h0 = hits; m0 = misses; r0 = mem_rd_cnt; w0 = mem_wr_cnt;
        if (WAYS == 1) begin
            rd(64);                     // miss
            rd(64);                     // hit
            rd(64 + LINE_BYTES - SB);   // hit, same line
            expect_counts(h0 + 2, m0 + 1, r0 + 1, w0, "dm reads");
            cpu_access(BASE + 64, 1'b1, '1, rnd_word(), 0);   // write
            if (READ_ONLY != 0) begin
                expect_counts(h0 + 2, m0 + 1, r0 + 1, w0, "ro write ignored");
                rd(64);
                expect_counts(h0 + 3, m0 + 1, r0 + 1, w0, "ro read after write");
            end else if (WRITE_BACK != 0) begin
                expect_counts(h0 + 3, m0 + 1, r0 + 1, w0, "wb write hit");
                rd(64 + WAY_STRIDE);    // conflict miss, dirty eviction
                expect_counts(h0 + 3, m0 + 2, r0 + 2, w0 + 1, "dirty eviction");
                rd(64);                 // conflict miss, clean victim
                expect_counts(h0 + 3, m0 + 3, r0 + 3, w0 + 1, "clean eviction");
                cpu_access(BASE + 64 + WAY_STRIDE + 4, 1'b1, 4'b0110, rnd_word(), 0);
                expect_counts(h0 + 3, m0 + 4, r0 + 4, w0 + 1, "write allocate");
                rd(64 + WAY_STRIDE);    // hit on the allocated line
                expect_counts(h0 + 4, m0 + 4, r0 + 4, w0 + 1, "hit after allocate");
            end else begin
                // write-through: hit writes the line through
                expect_counts(h0 + 3, m0 + 1, r0 + 1, w0 + 1, "wt write hit");
                cpu_access(BASE + 64 + WAY_STRIDE, 1'b1, 4'b1001, rnd_word(), 0);
                expect_counts(h0 + 3, m0 + 2, r0 + 2, w0 + 2, "wt write miss");
                rd(64);                 // no allocation: still a hit
                expect_counts(h0 + 4, m0 + 2, r0 + 2, w0 + 2, "wt no allocate");
            end
        end else begin
            // 2-way LRU: A, B, C map to the same set
            rd(0);                      // miss -> way 0
            rd(WAY_STRIDE);             // miss -> way 1
            rd(0);                      // hit, B becomes LRU
            rd(2 * WAY_STRIDE);         // miss, evicts B
            rd(0);                      // hit (A kept)
            expect_counts(h0 + 2, m0 + 3, r0 + 3, w0, "lru 1");
            rd(WAY_STRIDE);             // miss, evicts C (A is MRU)
            rd(0);                      // hit
            expect_counts(h0 + 3, m0 + 4, r0 + 4, w0, "lru 2");
            rd(2 * WAY_STRIDE);         // miss, C was evicted; evicts B
            expect_counts(h0 + 3, m0 + 5, r0 + 5, w0, "lru 3");
            if (WRITE_BACK != 0 && READ_ONLY == 0 && WPL == 1) begin
                // full-line write miss allocates without a refill; A is LRU now
                wr(3 * WAY_STRIDE, rnd_word());
                expect_counts(h0 + 3, m0 + 6, r0 + 5, w0, "full-line allocate");
                rd(3 * WAY_STRIDE);
                expect_counts(h0 + 4, m0 + 6, r0 + 5, w0, "hit after allocate");
                // partial write miss: refill then merge
                cpu_access(BASE + 4 * WAY_STRIDE, 1'b1, 32'h0000_F00F, rnd_word(), 0);
                expect_counts(h0 + 4, m0 + 7, r0 + 6, w0, "partial write allocate");
                rd(5 * WAY_STRIDE);     // evicts the dirty LRU line (3 * stride)
                expect_counts(h0 + 4, m0 + 8, r0 + 7, w0 + 1, "dirty eviction");
            end
        end
        // counter saturation: preload both counters near the maximum
        @(negedge clk);
        dut.hits_q = 32'hFFFF_FFFE;
        dut.misses_q = 32'hFFFF_FFFE;
        for (int i = 0; i < 3; i++)
            rd((8 + i) * LINE_BYTES);   // three misses in three sets
        repeat (3) rd(8 * LINE_BYTES);  // three hits
        if (hits !== 32'hFFFF_FFFF || misses !== 32'hFFFF_FFFF)
            fail("counters do not saturate");
        if (READ_ONLY == 0)
            for (int i = 0; i < 2 * SETS; i++)          // dirty lines for the flush
                wr((i * LINE_BYTES) % REGION, rnd_word());
        do_double_flush();

        // ---------------- random traffic ----------------
        n_rd = 0; n_wr = 0; n_fl = 0;
        for (op = 0; op < NOPS; op++) begin
            if (op % 500 == 0) begin
                // change the memory timing mode every 500 operations
                zero_wait = (op % 1500) == 0;
                ready_pct = 30 + rnd(70);
                max_delay = rnd(8);
                gap_pct = rnd(60);
                ready_after_valid = !zero_wait && rnd(3) == 0;
            end
            if (op == NOPS / 2) begin
                do_flush();             // memory == reference, then reset
                n_fl++;
                do_reset();
            end
            // half of the accesses go to a hot window of half the cache size
            if (rnd(2) == 0)
                off = rnd(SIZE_BYTES / 2);
            else
                off = rnd(REGION);
            off = off & ~(SB - 1);
            // a third of the requests are followed by the next one presented
            // in their last response cycle
            early = (rnd(3) == 0);
            early_next = early;
            if (rnd(2) == 0) begin
                case (rnd(4))
                    0: strb = '1;
                    1: strb = '0;
                    default: begin
                        for (int b = 0; b < SB; b++) strb[b] = rnd(2);
                    end
                endcase
                // a write sometimes carries a flush pulse while outstanding
                cpu_access(BASE + off + rnd(SB), 1'b1, strb, rnd_word(),
                           (rnd(150) == 0) ? 1 + rnd(4) : 0);
                n_wr++;
            end else begin
                cpu_access(BASE + off + rnd(SB), 1'b0, '0, '0,
                           (rnd(150) == 0) ? 1 + rnd(4) : 0);
                n_rd++;
            end
            if (flush_done_cnt < flush_cnt) begin
                wait_flush();
                n_fl++;
                early = 1'b0;
            end
            if (rnd(300) == 0) begin
                early = 1'b0;
                if (rnd(2) == 0)
                    do_flush();
                else
                    do_double_flush();
                n_fl++;
            end
            if (!early)
                repeat (rnd(3)) @(negedge clk);
        end
        do_flush();
        if (acc_cnt != resp_cnt) fail("accepted requests and responses differ");
        $display("INFO tb_npu_cache %s: %0d reads, %0d writes, %0d flushes, hits=%0d misses=%0d since the last reset",
                 NAME, n_rd, n_wr, n_fl + 1, hits, misses);
        done = 1'b1;
    end
endmodule

module tb_npu_cache;
    logic clk = 1'b0;
    always #5 clk = ~clk;

    logic d_l1d, d_l1i, d_l2, d_wt, d_l2h, d_dm4k, d_w2s, d_m16;

    tb_npu_cache_harness #(.NAME("l1d"), .SIZE_BYTES(1024), .WAYS(1), .CPU_W(32),
        .WRITE_BACK(1), .READ_ONLY(0), .BASE(32'h0000_0000), .REGION(4096),
        .NOPS(4000), .SEED(11)) h_l1d (.clk(clk), .start(1'b1), .done(d_l1d));
    tb_npu_cache_harness #(.NAME("l1i"), .SIZE_BYTES(1024), .WAYS(1), .CPU_W(32),
        .WRITE_BACK(1), .READ_ONLY(1), .BASE(32'h3FFF_0000), .REGION(4096),
        .NOPS(3000), .SEED(22)) h_l1i (.clk(clk), .start(1'b1), .done(d_l1i));
    tb_npu_cache_harness #(.NAME("l2"), .SIZE_BYTES(8192), .WAYS(2), .CPU_W(256),
        .WRITE_BACK(1), .READ_ONLY(0), .MEM_W(64), .RESP_W(32), .BASE(32'h1234_0000), .REGION(32768),
        .NOPS(3000), .SEED(33)) h_l2 (.clk(clk), .start(1'b1), .done(d_l2));
    tb_npu_cache_harness #(.NAME("l1wt"), .SIZE_BYTES(1024), .WAYS(1), .CPU_W(32),
        .WRITE_BACK(0), .READ_ONLY(0), .BASE(32'h0000_8000), .REGION(4096),
        .NOPS(2000), .SEED(44)) h_wt (.clk(clk), .start(1'b1), .done(d_wt));

    tb_npu_cache_harness #(.NAME("l2hold"), .SIZE_BYTES(8192), .WAYS(2), .CPU_W(256),
        .WRITE_BACK(1), .READ_ONLY(0), .CPU_HOLD(1), .MEM_W(64), .RESP_W(32), .BASE(32'h0000_0000), .REGION(32768),
        .NOPS(1500), .SEED(55)) h_l2h (.clk(clk), .start(1'b1), .done(d_l2h));

    tb_npu_cache_harness #(.NAME("dm4k"), .SIZE_BYTES(4096), .WAYS(1), .CPU_W(32),
        .WRITE_BACK(1), .READ_ONLY(0), .CPU_HOLD(1), .BASE(32'h0040_0000), .REGION(16384),
        .NOPS(1200), .SEED(66)) h_dm4k (.clk(clk), .start(1'b1), .done(d_dm4k));
    tb_npu_cache_harness #(.NAME("w2s"), .SIZE_BYTES(1024), .WAYS(2), .CPU_W(32),
        .WRITE_BACK(1), .READ_ONLY(0), .BASE(32'h0080_0000), .REGION(4096),
        .NOPS(1500), .SEED(77)) h_w2s (.clk(clk), .start(1'b1), .done(d_w2s));
    tb_npu_cache_harness #(.NAME("m16"), .SIZE_BYTES(1024), .WAYS(1), .CPU_W(64),
        .WRITE_BACK(1), .READ_ONLY(0), .MEM_W(16), .RESP_W(16), .BASE(32'h00C0_0000),
        .REGION(4096), .NOPS(1200), .SEED(88)) h_m16 (.clk(clk), .start(1'b1), .done(d_m16));

    initial begin
        wait (d_l1d && d_l1i && d_l2 && d_wt && d_l2h && d_dm4k && d_w2s && d_m16);
        $display("PASS tb_npu_cache l1d l1i l2 l1wt l2hold dm4k w2s m16");
        $finish;
    end

    initial begin
        #50_000_000;
        $display("MISMATCH tb_npu_cache: timeout");
        $fatal(1, "timeout");
    end
endmodule
