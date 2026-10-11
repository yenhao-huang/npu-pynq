`timescale 1ns/1ps

// Parameterised cache core, used as L1 I$ (READ_ONLY), L1 D$ (write-back)
// and the unified L2 (2-way, CPU_W = one line).
//
// Geometry: SETS = SIZE_BYTES / (LINE_BYTES * WAYS), all powers of two.
// WAYS = 1 (direct mapped) or 2 (LRU, one bit per set). Addresses are byte
// addresses; a CPU access is one naturally aligned CPU_W-bit word, so
// req_addr bits below log2(CPU_W/8) are ignored.
//
// CPU side (valid/ready, one request outstanding):
// - A request transfers on req_valid && req_ready. req_ready is high only in
//   the idle state (registered state decode), so the cache accepts at most one
//   request every two cycles.
// - Every accepted write gets exactly one resp_valid pulse (resp_rdata
//   don't-care). Every accepted read gets RRB = CPU_W / RESP_W resp_valid
//   pulses in consecutive cycles; pulse k carries bits [k*RESP_W +: RESP_W]
//   of the word on resp_rdata, valid only in that cycle. RESP_W = CPU_W (the
//   default, both L1s) is the plain one-pulse word protocol; the L2 uses
//   RESP_W = 32 so that its read responses are the 32-bit beats of the line
//   protocol below (it is the line-protocol slave behind npu_mem_arbiter).
// - Hit latency: WAYS = 1 answers 1 cycle after acceptance (tag and data RAMs
//   are read in the accept cycle; resp_valid is the tag compare, straight
//   from the tag RAM output register; with RRB > 1 the beats start one
//   cycle later, from state RESP). WAYS = 2 answers 2 cycles after
//   acceptance (tag compare first, then the data RAM is read from the hit way
//   only, so both ways share one data RAM). A tag RAM of more than 64 sets
//   is a BRAM (the L2): its compare result is registered first (state
//   TAGCMP), so the slow BRAM clock-to-out is followed only by the compare
//   and the data RAM address, write enables, tag write data and resp_valid
//   all start at that register: + 1 cycle (TREG = 1) on every lookup except
//   the replay after a refill, which is known to hit.
// - Miss: optional write-back of a dirty victim, line refill (each read beat
//   is written into the data RAM as it arrives), then the request is
//   replayed through the lookup (the replay is not counted). With a
//   zero-wait memory side (mem_req_ready = 1, the RBEATS read beats in the
//   RBEATS cycles after the request handshake, the write response the cycle
//   after it), cycles from acceptance to the first resp_valid, with
//   WPL = LINE_BYTES*8 / CPU_W, RBEATS = LINE_BYTES*8 / MEM_W and
//   R1 = 1 if WAYS = 1 and RRB > 1 (the beats leave from state RESP), else 0:
//     read hit = 1 + (WAYS - 1) + TREG + R1, write hit = 1 + TREG,
//     clean read miss = RBEATS + 4 + (WAYS - 1) + TREG + R1,
//     clean partial write miss = RBEATS + 4 + TREG,
//     clean full-line write miss (allocate, no refill) = 2 + TREG;
//     a dirty victim adds WPL + 3 to a miss.
//     Write-through: write hit = WPL + 5 + TREG, write miss = RBEATS + 5 + TREG.
//   L1 (WPL = RBEATS = 8): hit 1, clean miss 12, dirty miss 23.
//   L2 (WPL = 1, RBEATS = 4, 2-way, TREG): read hit 3, write hit 2, clean
//   miss 10, dirty miss 14, full-line write miss 3 (dirty 7).
// - WRITE_BACK = 1: write-back, write-allocate, dirty bit per line. When a
//   CPU word is a whole line (CPU_W = LINE_BYTES*8) and all byte strobes are
//   set, a write miss allocates without fetching the line.
// - WRITE_BACK = 0: write-through, no write-allocate (requires MEM_W =
//   CPU_W). The memory side writes whole lines only, so a write sends the
//   merged line: on a hit the word is written into the cache and the updated
//   line is read out and written to memory; on a miss the line is fetched,
//   the word merged into it and the line written back without allocating.
//   resp_valid follows the memory write completion.
// - READ_ONLY = 1: writes are not supported. A write request is accepted and
//   acknowledged with resp_valid (1 cycle after acceptance) and otherwise
//   ignored: no state, data or counter changes.
// - CPU_HOLD = 1: the requester keeps req_wstrb and req_wdata stable from
//   acceptance through the resp_valid cycle (a write hit is written in that
//   cycle). The line protocol below guarantees this, so set it for the L2
//   behind npu_mem_arbiter; the scalar core's data port also does it. The
//   cache then uses them directly instead of registering a copy (saves
//   CPU_W + CPU_W/8 flops).
//   CPU_HOLD = 0 (default) registers them at acceptance.
// - hits / misses count accepted, non-ignored CPU requests (32-bit,
//   saturating). Replays after a refill are not counted.
//
// flush (one-cycle pulse, may arrive at any time; it is latched and served
// after the current request, ahead of new requests; a request accepted in
// the cycle of the pulse is the current request): set by set, the dirty
// ways of the set are written back and the set is invalidated; then
// flush_done pulses for one cycle (req_ready may be high in that cycle). A
// flush pulse arriving while a flush is pending or running, up to and
// including its last sweep cycle, is merged into it (one flush_done).
// Duration from an idle cache, pulse to flush_done: 2*SETS + 2 cycles plus
// WPL + 5 per dirty line (zero-wait memory side); without dirty bits
// (READ_ONLY or write-through) only the invalidate sweep, SETS + 2 cycles.
//
// Reset: the tag RAM holds {valid, dirty, tag} per way (plus the LRU bit); a
// SETS-cycle sweep after reset clears it, with req_ready low meanwhile. The
// tag and data RAMs have no reset and are simple dual port (one synchronous
// read port with a read enable, one write port) so Vivado infers BRAM or
// LUTRAM. A tag RAM of at most 64 sets is forced into LUTRAM (the L1s), so
// the 1-cycle hit compare starts at a flop, not at a BRAM output. The tag
// RAM output register is the working copy of the set's entry: it is only
// reloaded by a lookup, a replay or a flush step, and holds through a miss.
// The write data and write addresses of both RAMs never depend on
// mem_resp_valid: the memory response timing reaches the RAMs only through
// their write enables (the refill beat itself is the data RAM write data).
//
// Combinational port-to-port paths: none. Every output is a function of
// registers (state, tag RAM output register, data RAM output, counters), so
// the cache can sit between a requester that accepts same-cycle responses
// (the scalar core) and npu_mem_arbiter without forming a loop.
// Event-driven simulation: the control always_comb builds its output values
// in block-local variables (default, then overrides) and writes each output
// port exactly once, at the end. Writing a port more than once per run would
// glitch it in Icarus whenever the block re-runs (any input event, even one
// that does not change the result). A neighbour whose combinational logic
// reads the glitching port and feeds back into an input of this block then
// wakes it again, and the simulation loops forever at one time step: seen
// with resp_valid -> a combinational requester (the scalar core) -> req_*,
// and with mem_req_addr / req_ready through npu_mem_arbiter between L1 D$
// and the L2 (review round 3; the hardware was unaffected; regression in
// tb_npu_cache_hierarchy). Keep this rule when editing the block. The
// cross-module paths are register -> output -> logic in the neighbour: for
// the L1 hit, tag RAM output register -> tag compare -> resp_valid (about
// four LUT levels); for the L2, the data RAM output -> RESP_W beat mux ->
// mem_resp_rdata of an L1 (data RAM write data) and the tag RAM output ->
// way select -> mem_req_addr.
//
// Memory side, line protocol shared with npu_mem_arbiter and
// npu_axi4_line_bridge:
// - A request transfers on mem_req_valid && mem_req_ready; mem_req_addr is
//   line aligned. One request is outstanding at a time, and the requester
//   keeps mem_req_addr, mem_req_we and mem_req_wdata (the whole line, for a
//   write) stable from the request through the cycle of its last response
//   pulse.
// - A write completes with exactly one mem_resp_valid pulse (mem_resp_rdata
//   don't-care); the pulse follows the downstream write completion, e.g. the
//   AXI B response.
// - A read completes with exactly RBEATS = LINE_BYTES*8 / MEM_W mem_resp_valid
//   pulses (beats), in address order (beat 0 = the lowest line bytes), each
//   carrying MEM_W bits on mem_resp_rdata in its cycle. Beats may be
//   separated by idle cycles; the requester accepts a beat in every cycle
//   (no ready). The L1s use MEM_W = 32, the L2 MEM_W = 64 (the AXI beat).
module npu_cache #(
    parameter integer SIZE_BYTES = 1024,
    parameter integer LINE_BYTES = 32,
    parameter integer WAYS = 1,
    parameter integer CPU_W = 32,
    parameter integer WRITE_BACK = 1,
    parameter integer READ_ONLY = 0,
    parameter integer CPU_HOLD = 0,
    parameter integer MEM_W = CPU_W,      // memory-side read beat width
    parameter integer RESP_W = CPU_W      // CPU-side read response beat width
) (
    input  logic                      clk,
    input  logic                      rst_n,
    // CPU side
    input  logic                      req_valid,
    output logic                      req_ready,
    input  logic [31:0]               req_addr,
    input  logic                      req_we,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [CPU_W/8-1:0]        req_wstrb,   // unused when READ_ONLY
    input  logic [CPU_W-1:0]          req_wdata,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic                      resp_valid,
    output logic [RESP_W-1:0]         resp_rdata,
    input  logic                      flush,
    output logic                      flush_done,
    // Memory side: whole-line requests, read data in MEM_W-bit beats
    output logic                      mem_req_valid,
    input  logic                      mem_req_ready,
    output logic [31:0]               mem_req_addr,
    output logic                      mem_req_we,
    output logic [LINE_BYTES*8-1:0]   mem_req_wdata,
    input  logic                      mem_resp_valid,
    input  logic [MEM_W-1:0]          mem_resp_rdata,
    // Counters
    output logic [31:0]               hits,
    output logic [31:0]               misses
);
    localparam integer LINE_W = LINE_BYTES * 8;
    localparam integer SB = CPU_W / 8;                    // strobe bits
    localparam integer SETS = SIZE_BYTES / (LINE_BYTES * WAYS);
    localparam integer WPL = LINE_W / CPU_W;              // CPU words per line
    localparam integer OFF_W = $clog2(LINE_BYTES);
    localparam integer BYTE_W = $clog2(SB);
    localparam integer WSEL_W = OFF_W - BYTE_W;           // word-in-line bits
    localparam integer WS_W = (WSEL_W > 0) ? WSEL_W : 1;
    localparam integer IDX_W = $clog2(SETS);
    localparam integer TAG_W = 32 - OFF_W - IDX_W;
    localparam integer EW = TAG_W + 2;                    // {valid, dirty, tag}
    localparam integer TW = WAYS * EW + ((WAYS == 2) ? 1 : 0);
    localparam integer DDEPTH = WAYS * SETS * WPL;
    localparam integer DA_W = $clog2(DDEPTH);
    localparam integer MB = MEM_W / 8;                    // bytes per memory beat
    localparam integer RBEATS = LINE_W / MEM_W;           // read beats per line
    localparam integer BPW = CPU_W / MEM_W;               // read beats per data word
    localparam integer BPW_W = $clog2(BPW);
    localparam integer RRB = CPU_W / RESP_W;              // CPU read response beats
    localparam integer CMAX = (WPL > RBEATS) ? ((WPL > RRB) ? WPL : RRB)
                                             : ((RBEATS > RRB) ? RBEATS : RRB);
    localparam integer CNT_W = $clog2(CMAX + 1);          // counts 0..CMAX
    localparam integer DIRTY_EN = (WRITE_BACK != 0 && READ_ONLY == 0) ? 1 : 0;
    localparam integer WT_EN = (WRITE_BACK == 0 && READ_ONLY == 0) ? 1 : 0;
    localparam integer LBUF_EN = ((DIRTY_EN != 0 && WPL > 1) || WT_EN != 0) ? 1 : 0;
    // Tag RAM in BRAM (not forced to LUTRAM below): registered tag compare
    localparam integer TREG = (SETS > 64) ? 1 : 0;

    localparam logic [3:0] S_INIT      = 4'd0;   // tag clear sweep (reset, flush)
    localparam logic [3:0] S_IDLE      = 4'd1;
    localparam logic [3:0] S_LOOKUP    = 4'd2;   // tag compare
    localparam logic [3:0] S_RESP      = 4'd3;   // registered response (beats)
    localparam logic [3:0] S_EVICT     = 4'd4;   // read a line out of the data RAM
    localparam logic [3:0] S_WB_REQ    = 4'd5;
    localparam logic [3:0] S_WB_WAIT   = 4'd6;
    localparam logic [3:0] S_FILL_REQ  = 4'd7;
    localparam logic [3:0] S_FILL_WAIT = 4'd8;   // write read beats as they arrive
    localparam logic [3:0] S_REPLAY    = 4'd9;   // re-read tags after a refill
    localparam logic [3:0] S_ALLOC     = 4'd10;  // full-line write allocate
    localparam logic [3:0] S_FLUSH_RD  = 4'd11;
    localparam logic [3:0] S_FLUSH_CHK = 4'd12;
    localparam logic [3:0] S_TAGCMP    = 4'd13;  // register the BRAM tag compare

    initial begin
        if (WAYS != 1 && WAYS != 2)
            $fatal(1, "npu_cache WAYS must be 1 or 2");
        if (CPU_W < 8 || (CPU_W % 8) != 0 || CPU_W > LINE_W || (LINE_W % CPU_W) != 0)
            $fatal(1, "npu_cache CPU_W must be a multiple of 8 dividing the line");
        if ((1 << OFF_W) != LINE_BYTES || (1 << IDX_W) != SETS || SETS < 2 ||
            (1 << BYTE_W) != SB)
            $fatal(1, "npu_cache sizes must be powers of two with SETS >= 2");
        if (MEM_W < 8 || MEM_W > CPU_W || (1 << BPW_W) != BPW || BPW * MEM_W != CPU_W)
            $fatal(1, "npu_cache MEM_W must be CPU_W / 2^k, at least 8");
        if (RESP_W < 8 || RESP_W > CPU_W || RRB * RESP_W != CPU_W ||
            (1 << $clog2(RRB)) != RRB)
            $fatal(1, "npu_cache RESP_W must be CPU_W / 2^k, at least 8");
        if (WT_EN != 0 && MEM_W != CPU_W)
            $fatal(1, "npu_cache write-through requires MEM_W = CPU_W");
    end

    // ------------------------------------------------------------------
    // Registers
    // ------------------------------------------------------------------
    logic [3:0]        state_q, state_d;
    logic [TAG_W-1:0]  tag_q, tag_d;
    logic [IDX_W-1:0]  idx_q, idx_d;       // request set, or sweep index
    logic [WS_W-1:0]   word_q, word_d;
    logic              we_q, we_d;
    logic [SB-1:0]     cpu_wstrb;          // write strobes / data of the
    logic [CPU_W-1:0]  cpu_wdata;          // accepted request
    logic              way_q, way_d;       // hit / victim / flush way
    logic [CNT_W-1:0]  cnt_q, cnt_d;       // word / beat counter
    logic              replay_q, replay_d;
    /* verilator lint_off UNUSEDSIGNAL */
    logic              hit_q, hit_d;       // registered compare (TREG only)
    /* verilator lint_on UNUSEDSIGNAL */
    logic              flushing_q, flushing_d;
    logic              flush_pend_q;
    logic              flush_done_d;
    logic              hit_inc, miss_inc;
    logic [31:0]       hits_q, misses_q;

    // RAM ports
    logic              t_re, t_we;
    logic [IDX_W-1:0]  t_raddr;
    logic [TW-1:0]     t_wdata, t_rdata;
    logic              d_re;
    logic [SB-1:0]     d_we;
    logic [DA_W-1:0]   d_raddr, d_waddr;
    logic [CPU_W-1:0]  d_wdata, d_rdata;
    logic              d_wfill;            // write refill data, else CPU data
    /* verilator lint_off UNUSEDSIGNAL */
    // unused in some configurations (no line buffer, CPU_HOLD = 1, READ_ONLY)
    logic              lb_shift;
    logic [LINE_W-1:0] line_buf;
    logic              cpu_accept;
    /* verilator lint_on UNUSEDSIGNAL */

    // ------------------------------------------------------------------
    // Tag RAM: SETS x TW, simple dual port, written only at idx_q. t_rdata
    // is only updated by a read (t_re), so it holds the entry of the current
    // set from the lookup until the next lookup, replay or flush step.
    // ------------------------------------------------------------------
    generate
        if (SETS <= 64) begin : g_tag_lutram
            (* ram_style = "distributed" *) logic [TW-1:0] tag_mem [0:SETS-1];
            always_ff @(posedge clk) begin
                if (t_we)
                    tag_mem[idx_q] <= t_wdata;
                if (t_re)
                    t_rdata <= tag_mem[t_raddr];
            end
        end else begin : g_tag_ram
            logic [TW-1:0] tag_mem [0:SETS-1];
            always_ff @(posedge clk) begin
                if (t_we)
                    tag_mem[idx_q] <= t_wdata;
                if (t_re)
                    t_rdata <= tag_mem[t_raddr];
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Data RAM: (WAYS * SETS * WPL) x CPU_W, byte write enables
    // ------------------------------------------------------------------
    logic [CPU_W-1:0] data_mem [0:DDEPTH-1];
    always_ff @(posedge clk) begin
        for (int b = 0; b < SB; b++) begin
            if (d_we[b])
                data_mem[d_waddr][b*8 +: 8] <= d_wdata[b*8 +: 8];
        end
        if (d_re)
            d_rdata <= data_mem[d_raddr];
    end

    // Data RAM word address = {way, set, word} (bit fields, no adders)
    function automatic [DA_W-1:0] daddr(input logic w, input logic [IDX_W-1:0] i,
                                        input logic [WS_W-1:0] wd);
        logic [31:0] a;
        a = 32'(i) << WSEL_W;
        if (WSEL_W > 0)
            a = a | 32'(wd);
        if (WAYS == 2 && w)
            a = a | (32'd1 << (IDX_W + WSEL_W));
        daddr = DA_W'(a);
    endfunction

    // Tag entry layout {lru, way1, way0}, each way {valid, dirty, tag}. A
    // 1-way entry is zero-extended to this layout so every field has a
    // constant position and a way select is a 2:1 mux (no variable shifts).
    localparam integer TWP = 2 * EW + 1;
    // New entry: way w replaced by f, the other way kept, LRU bit = ~w
    /* verilator lint_off UNUSEDSIGNAL */
    function automatic [TW-1:0] ent_set(input logic [EW-1:0] e0, input logic [EW-1:0] e1,
                                        input logic w, input logic [EW-1:0] f);
        logic [TWP-1:0] r;
        r = {~w, w ? f : e1, w ? e0 : f};
        ent_set = TW'(r);
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    // ------------------------------------------------------------------
    // Lookup (all fields straight from the tag RAM output register)
    // ------------------------------------------------------------------
    logic             hit0, hit1, hit, hit_way;
    logic             vic_way, vic_dirty;
    logic             alloc_nf;
    logic             wt_write;
    logic             fill_last;
    logic [TAG_W-1:0] wb_tag;
    logic [IDX_W-1:0] req_idx;
    logic [WS_W-1:0]  req_word;
    logic [TAG_W-1:0] req_tag;
    logic             ent_lru;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [TWP-1:0]   cur_p;
    logic [EW-1:0]    cur_w0, cur_w1;
    logic [CNT_W-1:0] fill_slot;
    /* verilator lint_on UNUSEDSIGNAL */
    logic             cur_v0, cur_v1, cur_d0, cur_d1, hit_dirty;
    logic             hit_l, hway_l;      // hit / hit way as used by LOOKUP
    logic [TAG_W-1:0] cur_t0, cur_t1;
    logic [WS_W-1:0]  fill_word;
    logic [SB-1:0]    fill_strb;
    logic [CPU_W-1:0] fill_data;

    // Constant part selects kept out of the always_comb blocks (Icarus)
    assign cur_p = TWP'(t_rdata);
    assign cur_w0 = cur_p[0 +: EW];
    assign cur_w1 = cur_p[EW +: EW];
    assign cur_v0 = cur_w0[EW-1];
    assign cur_v1 = cur_w1[EW-1];
    assign cur_d0 = cur_w0[TAG_W];
    assign cur_d1 = cur_w1[TAG_W];
    assign cur_t0 = cur_w0[TAG_W-1:0];
    assign cur_t1 = cur_w1[TAG_W-1:0];
    assign ent_lru = cur_p[TWP-1];
    assign req_tag = req_addr[31 -: TAG_W];
    assign req_idx = req_addr[OFF_W +: IDX_W];
    assign req_word = (WSEL_W > 0) ? WS_W'(req_addr[OFF_W-1:0] >> BYTE_W) : '0;

    // Refill beat cnt_q goes to data word cnt_q / BPW, byte lanes of slot
    // cnt_q % BPW (the beat is replicated across the word)
    assign fill_word = WS_W'(cnt_q >> BPW_W);
    assign fill_slot = cnt_q & CNT_W'(BPW - 1);
    assign fill_data = {BPW{mem_resp_rdata}};
    assign fill_last = (cnt_q == CNT_W'(RBEATS - 1));
    always_comb begin
        for (int b = 0; b < SB; b++)
            fill_strb[b] = (BPW == 1) || (CNT_W'(b / MB) == fill_slot);
    end

    // Data RAM write data: one 2:1 mux (refill beat or CPU write data); a
    // read-only cache only ever writes refill beats
    assign d_wdata = (READ_ONLY != 0 || d_wfill) ? fill_data : cpu_wdata;

    always_comb begin
        hit0 = cur_v0 && (cur_t0 == tag_q);
        hit1 = (WAYS == 2) && cur_v1 && (cur_t1 == tag_q);
        hit = hit0 || hit1;
        hit_way = hit1;
        // TREG: LOOKUP uses the compare registered in TAGCMP (way_q holds the
        // hit way); otherwise the compare itself
        hit_l = (TREG != 0) ? hit_q : hit;
        hway_l = (TREG != 0) ? way_q : hit_way;
        hit_dirty = hway_l ? cur_d1 : cur_d0;
        // The LRU bit alone picks the victim: lines are only invalidated all
        // together (reset, flush), which also clears the LRU bit to way 0, and
        // the first fill of way 0 makes way 1 the LRU, so an invalid way is
        // always replaced before a valid one.
        vic_way = (WAYS == 2) && ent_lru;
        vic_dirty = (DIRTY_EN != 0) && (vic_way ? (cur_v1 && cur_d1) : (cur_v0 && cur_d0));
        // Whole-line write with all strobes: allocate without a refill
        alloc_nf = (DIRTY_EN != 0) && (WPL == 1) && we_q && (&cpu_wstrb);
        wt_write = (WT_EN != 0) && we_q;
        // Write-through writes the request's own line; write-back the victim
        wb_tag = (WT_EN != 0) ? tag_q : (way_q ? cur_t1 : cur_t0);
    end

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    // State TAGCMP exists only with TREG (a constant 0 otherwise)
    logic in_tagcmp;
    generate
        if (TREG != 0) begin : g_tagcmp
            assign in_tagcmp = (state_q == S_TAGCMP);
        end else begin : g_no_tagcmp
            assign in_tagcmp = 1'b0;
        end
    endgenerate

    always_comb begin : comb_main
        // Output port values under construction (default-then-override),
        // local to this block; each port is written once, at the end
        logic              req_ready_c, resp_valid_c;
        logic              mem_req_valid_c, mem_req_we_c;
        logic [31:0]       mem_req_addr_c;
        state_d = state_q;
        tag_d = tag_q;
        idx_d = idx_q;
        word_d = word_q;
        we_d = we_q;
        way_d = way_q;
        cnt_d = cnt_q;
        replay_d = replay_q;
        hit_d = hit_q;
        flushing_d = flushing_q;
        flush_done_d = 1'b0;
        hit_inc = 1'b0;
        miss_inc = 1'b0;
        req_ready_c = 1'b0;
        resp_valid_c = 1'b0;
        mem_req_valid_c = 1'b0;
        mem_req_we_c = 1'b0;
        mem_req_addr_c = {tag_q, idx_q, {OFF_W{1'b0}}};
        t_re = 1'b0;
        t_raddr = idx_q;
        t_we = 1'b0;
        t_wdata = '0;
        d_re = 1'b0;
        d_raddr = daddr(way_q, idx_q, word_q);
        d_we = '0;
        d_waddr = daddr(way_q, idx_q, word_q);
        d_wfill = 1'b0;
        lb_shift = 1'b0;

        case (state_q)
            S_INIT: begin
                t_we = 1'b1;
                idx_d = idx_q + 1'b1;
                if (idx_q == IDX_W'(SETS - 1)) begin
                    state_d = S_IDLE;
                    flush_done_d = flushing_q;
                    flushing_d = 1'b0;
                end
            end

            S_IDLE: begin
                req_ready_c = !flush_pend_q;
                cnt_d = '0;
                if (flush_pend_q) begin
                    idx_d = '0;
                    flushing_d = 1'b1;
                    state_d = (DIRTY_EN != 0) ? S_FLUSH_RD : S_INIT;
                end else if (req_valid) begin
                    tag_d = req_tag;
                    idx_d = req_idx;
                    word_d = req_word;
                    we_d = req_we;
                    replay_d = 1'b0;
                    if (READ_ONLY != 0 && req_we) begin
                        state_d = S_RESP;
                    end else begin
                        t_re = 1'b1;
                        t_raddr = req_idx;
                        if (WAYS == 1) begin
                            // only a read hit uses this word (a write hit
                            // writes, a miss re-reads in REPLAY): no BRAM
                            // read for writes
                            d_re = !req_we;
                            d_raddr = daddr(1'b0, req_idx, req_word);
                        end
                        state_d = (TREG != 0) ? S_TAGCMP : S_LOOKUP;
                    end
                end
            end

            S_LOOKUP: begin
                hit_inc = !replay_q && hit_l;
                miss_inc = !replay_q && !hit_l;
                cnt_d = '0;
                if (hit_l) begin
                    way_d = hway_l;
                    if (we_q) begin
                        d_we = cpu_wstrb;
                        d_waddr = daddr(hway_l, idx_q, word_q);
                        if (WAYS == 2 || DIRTY_EN != 0) begin
                            t_we = 1'b1;
                            t_wdata = ent_set(cur_w0, cur_w1, hway_l,
                                              {1'b1, (DIRTY_EN != 0) || hit_dirty, tag_q});
                        end
                        if (WT_EN != 0) begin
                            state_d = S_EVICT;             // write the line through
                        end else begin
                            resp_valid_c = 1'b1;
                            state_d = S_IDLE;
                        end
                    end else if (WAYS == 2) begin
                        t_we = 1'b1;                       // LRU update
                        t_wdata = ent_set(cur_w0, cur_w1, hway_l,
                                          {1'b1, hit_dirty, tag_q});
                        d_re = 1'b1;
                        d_raddr = daddr(hway_l, idx_q, word_q);
                        state_d = S_RESP;
                    end else if (RRB > 1) begin
                        state_d = S_RESP;                  // data read at accept
                    end else begin
                        resp_valid_c = 1'b1;                 // data read at accept
                        state_d = S_IDLE;
                    end
                end else begin
                    way_d = vic_way;
                    if (wt_write)
                        state_d = S_FILL_REQ;              // fetch, merge, write
                    else if (vic_dirty)
                        state_d = S_EVICT;
                    else if (alloc_nf)
                        state_d = S_ALLOC;
                    else
                        state_d = S_FILL_REQ;
                end
            end

            S_RESP: begin
                // One pulse for a write, RRB beats of the held d_rdata for a read
                resp_valid_c = 1'b1;
                cnt_d = cnt_q + 1'b1;
                if (we_q || RRB == 1 || cnt_q == CNT_W'(RRB - 1))
                    state_d = S_IDLE;
            end

            S_EVICT: begin
                // Read words 0..WPL-1; each arrives one cycle later
                if (cnt_q != CNT_W'(WPL)) begin
                    d_re = 1'b1;
                    d_raddr = daddr(way_q, idx_q, WS_W'(cnt_q));
                end
                if (cnt_q != '0)
                    lb_shift = 1'b1;
                cnt_d = cnt_q + 1'b1;
                if (cnt_q == CNT_W'(WPL))
                    state_d = S_WB_REQ;
            end

            S_WB_REQ: begin
                mem_req_valid_c = 1'b1;
                mem_req_we_c = 1'b1;
                mem_req_addr_c = {wb_tag, idx_q, {OFF_W{1'b0}}};
                if (mem_req_ready)
                    state_d = S_WB_WAIT;
            end

            S_WB_WAIT: begin
                mem_req_we_c = 1'b1;
                mem_req_addr_c = {wb_tag, idx_q, {OFF_W{1'b0}}};
                cnt_d = '0;
                if (mem_resp_valid) begin
                    if (flushing_q)
                        state_d = S_FLUSH_RD;              // re-read: next dirty way
                    else if (wt_write)
                        state_d = S_RESP;
                    else if (alloc_nf)
                        state_d = S_ALLOC;
                    else
                        state_d = S_FILL_REQ;
                end
            end

            S_FILL_REQ: begin
                mem_req_valid_c = 1'b1;
                if (mem_req_ready)
                    state_d = S_FILL_WAIT;
            end

            S_FILL_WAIT: begin
                // RAM write data / addresses from registers; mem_resp_valid
                // only gates the write enables and the counter
                d_waddr = daddr(way_q, idx_q, fill_word);
                d_wfill = 1'b1;
                t_wdata = ent_set(cur_w0, cur_w1, way_q, {1'b1, 1'b0, tag_q});
                if (mem_resp_valid) begin
                    cnt_d = cnt_q + 1'b1;
                    if (wt_write) begin
                        lb_shift = 1'b1;                   // merged line
                        if (fill_last)
                            state_d = S_WB_REQ;
                    end else begin
                        d_we = fill_strb;
                        if (fill_last) begin
                            t_we = 1'b1;
                            state_d = S_REPLAY;
                        end
                    end
                end
            end

            S_REPLAY: begin
                t_re = 1'b1;
                if (WAYS == 1) begin
                    d_re = 1'b1;
                    d_raddr = daddr(1'b0, idx_q, word_q);
                end
                replay_d = 1'b1;
                hit_d = 1'b1;          // TREG: the refilled line hits in way_q
                state_d = S_LOOKUP;
            end

            S_ALLOC: begin
                d_we = '1;
                d_waddr = daddr(way_q, idx_q, '0);
                t_we = 1'b1;
                t_wdata = ent_set(cur_w0, cur_w1, way_q, {1'b1, 1'b1, tag_q});
                resp_valid_c = 1'b1;
                state_d = S_IDLE;
            end

            S_FLUSH_RD: begin
                t_re = 1'b1;
                state_d = S_FLUSH_CHK;
            end

            S_FLUSH_CHK: begin
                // Write back one dirty way at a time: its dirty bit is cleared
                // in the tag RAM now (t_rdata keeps the old entry for wb_tag),
                // and the set is re-read after the write-back
                cnt_d = '0;
                if (cur_v0 && cur_d0) begin
                    way_d = 1'b0;
                    t_we = 1'b1;
                    t_wdata = ent_set(cur_w0, cur_w1, 1'b0, {1'b1, 1'b0, cur_t0});
                    state_d = S_EVICT;
                end else if (WAYS == 2 && cur_v1 && cur_d1) begin
                    way_d = 1'b1;
                    t_we = 1'b1;
                    t_wdata = ent_set(cur_w0, cur_w1, 1'b1, {1'b1, 1'b0, cur_t1});
                    state_d = S_EVICT;
                end else begin
                    t_we = 1'b1;                           // invalidate the set
                    idx_d = idx_q + 1'b1;
                    if (idx_q == IDX_W'(SETS - 1)) begin
                        flush_done_d = 1'b1;
                        flushing_d = 1'b0;
                        state_d = S_IDLE;
                    end else begin
                        state_d = S_FLUSH_RD;
                    end
                end
            end

            default: state_d = S_INIT;
        endcase

        // TAGCMP (TREG only, in_tagcmp is a constant 0 otherwise): register
        // the compare; the hit way, or else the victim way, into way_q
        if (in_tagcmp) begin
            hit_d = hit;
            way_d = hit ? hit_way : vic_way;
            state_d = S_LOOKUP;
        end

        // Each output port is written exactly once per evaluation, after the
        // default-then-override code above has settled (header: event-driven
        // simulation of the port-to-port chain)
        req_ready = req_ready_c;
        resp_valid = resp_valid_c;
        mem_req_valid = mem_req_valid_c;
        mem_req_we = mem_req_we_c;
        mem_req_addr = mem_req_addr_c;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= S_INIT;
            tag_q <= '0;
            idx_q <= '0;
            word_q <= '0;
            we_q <= 1'b0;
            way_q <= 1'b0;
            cnt_q <= '0;
            replay_q <= 1'b0;
            hit_q <= 1'b0;
            flushing_q <= 1'b0;
            flush_pend_q <= 1'b0;
            flush_done <= 1'b0;
            hits_q <= '0;
            misses_q <= '0;
        end else begin
            state_q <= state_d;
            tag_q <= tag_d;
            idx_q <= idx_d;
            word_q <= word_d;
            we_q <= we_d;
            way_q <= way_d;
            cnt_q <= cnt_d;
            replay_q <= replay_d;
            hit_q <= hit_d;
            flushing_q <= flushing_d;
            if (state_q == S_IDLE && flush_pend_q)
                flush_pend_q <= 1'b0;
            else if (flush && !flushing_q)
                flush_pend_q <= 1'b1;
            flush_done <= flush_done_d;
            if (hit_inc && hits_q != 32'hFFFF_FFFF)
                hits_q <= hits_q + 32'd1;
            if (miss_inc && misses_q != 32'hFFFF_FFFF)
                misses_q <= misses_q + 32'd1;
        end
    end

    // ------------------------------------------------------------------
    // Line buffer (write-back with WPL > 1, or write-through): a shift
    // register, word 0 at the bottom, that collects the line for a memory
    // write. Eviction shifts data RAM words in from the top; a write-through
    // write miss shifts in the refill beats with the CPU word merged. A
    // refill of a write-back or read-only cache does not use it (beats go
    // straight into the data RAM), so the read-only L1 has none.
    // ------------------------------------------------------------------
    generate
        if (LBUF_EN != 0) begin : g_lbuf
            logic [CPU_W-1:0] lb_in;
            if (WT_EN != 0) begin : g_merge
                // MEM_W = CPU_W: refill beat cnt_q is data word cnt_q
                logic [CPU_W-1:0] merged;
                always_comb begin
                    merged = CPU_W'(mem_resp_rdata);
                    if (WS_W'(cnt_q) == word_q) begin
                        for (int b = 0; b < SB; b++)
                            if (cpu_wstrb[b])
                                merged[b*8 +: 8] = cpu_wdata[b*8 +: 8];
                    end
                end
                assign lb_in = (state_q == S_FILL_WAIT) ? merged : d_rdata;
            end else begin : g_evict_only
                assign lb_in = d_rdata;
            end
            always_ff @(posedge clk) begin
                if (lb_shift) begin
                    if (WPL > 1)
                        line_buf <= {lb_in, line_buf[LINE_W-1:CPU_W]};
                    else
                        line_buf <= LINE_W'(lb_in);
                end
            end
            assign mem_req_wdata = line_buf;
        end else begin : g_nolbuf
            // WPL = 1 write-back: the victim line is the held data RAM output
            assign line_buf = '0;
            assign mem_req_wdata = LINE_W'(d_rdata);
        end
    endgenerate

    // ------------------------------------------------------------------
    // Write strobes and data of the accepted request. CPU_HOLD = 1 uses the
    // inputs directly (the requester holds them until resp_valid). A
    // read-only cache never writes CPU data.
    // ------------------------------------------------------------------
    assign cpu_accept = req_valid && req_ready;
    generate
        if (READ_ONLY != 0) begin : g_ro
            assign cpu_wstrb = '0;
            assign cpu_wdata = '0;
        end else if (CPU_HOLD != 0) begin : g_hold
            assign cpu_wstrb = req_wstrb;
            assign cpu_wdata = req_wdata;
        end else begin : g_copy
            logic [SB-1:0]    wstrb_q;
            logic [CPU_W-1:0] wdata_q;
            always_ff @(posedge clk) begin
                if (cpu_accept) begin
                    wstrb_q <= req_wstrb;
                    wdata_q <= req_wdata;
                end
            end
            assign cpu_wstrb = wstrb_q;
            assign cpu_wdata = wdata_q;
        end
    endgenerate

    // Read response: the data RAM output word, or its beat cnt_q (RRB > 1)
    generate
        if (RRB == 1) begin : g_resp_word
            assign resp_rdata = d_rdata;
        end else begin : g_resp_beat
            assign resp_rdata = d_rdata[cnt_q[$clog2(RRB)-1:0]*RESP_W +: RESP_W];
        end
    endgenerate
    assign hits = hits_q;
    assign misses = misses_q;
endmodule
