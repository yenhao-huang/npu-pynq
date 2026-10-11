`timescale 1ns/1ps

// Self-checking testbench for npu_scalar_core.
//
// Programs, their start PCs and the expected architectural results come from
// src/test/vectors/scalar_core_programs.txt, produced by the reference ISS
// (src/test/vectors/generate_scalar_vectors.py). For every run the testbench
// pulses start, waits for halted or fault, and compares the status, cause,
// fault_pc, instret, all 31 registers (x1..x31), and at the end of each case
// the vector-command log and every initialised or written memory word.
//
// Environment models:
// - instruction and data memories (64 KB, flat, shared array): random request
//   ready, response 0..4 cycles after the handshake (0 = combinational
//   response in the handshake cycle), bus error at or above 0x10000;
// - vector unit: random vcmd_ready, vec_busy held for a random time after each
//   accepted command (so VFENCE has to wait);
// - fixed-latency cases (latency L >= 0 in the vector file): always ready,
//   responses exactly L cycles after the handshake, vec_busy low; the cycle
//   counter must then match the ISS cycle model exactly (CPI measurement).
//   These restart as soon as possible after a stop, so with a slow
//   instruction port a fetch abandoned by a data bus error is still in flight
//   at the next start.
// Protocol checks: request fields stable while stalled, data request fields
// also stable from acceptance until the response (L1 D$ CPU_HOLD = 1), at
// most one outstanding request per port, aligned data addresses, no wstrb on loads,
// VFENCE (op 31) never sent and the next fetch after a VFENCE only after
// vec_busy was low, no output change between clock edges (the inputs that
// the environment changes mid-cycle -- the readies, same-cycle responses,
// start / start_pc -- must not reach any output combinationally), status
// and counters frozen while stopped, halt / fault causes clear while running.
// Also covered: start ignored while running, an asynchronous reset in every
// state (IDLE .. FAULT, white box), same-cycle responses on both ports in
// every state that issues requests, and the early fetch from EXEC (accepted
// or stalled, its response arriving before, at or after the retire state,
// held through a vcmd_ready stall with a bus error, and drained after a
// fault, with and without a bus error).
module tb_npu_scalar_core;
    localparam integer MEM_WORDS   = 16384;
    localparam integer MAX_VCMDS   = 512;
    localparam integer RUN_TIMEOUT = 400000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic        start = 1'b0;
    logic [31:0] start_pc = '0;
    logic        running, halted, halt_cause, fault;
    logic [2:0]  fault_cause;
    logic [31:0] fault_pc;
    logic        i_req_valid, i_req_ready;
    logic [31:0] i_req_addr;
    logic        i_resp_valid, i_resp_err;
    logic [31:0] i_resp_rdata;
    logic        d_req_valid, d_req_ready, d_req_we;
    logic [31:0] d_req_addr, d_req_wdata;
    logic [3:0]  d_req_wstrb;
    logic        d_resp_valid, d_resp_err;
    logic [31:0] d_resp_rdata;
    logic        vcmd_valid, vcmd_ready;
    logic [4:0]  vcmd_op;
    logic [1:0]  vcmd_type;
    logic [2:0]  vcmd_vd, vcmd_vs1, vcmd_vs2;
    logic [31:0] vcmd_scalar;
    logic        vec_busy;
    logic [31:0] cycles, instret;

    npu_scalar_core dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .start_pc(start_pc), .running(running), .halted(halted),
        .halt_cause(halt_cause), .fault(fault), .fault_cause(fault_cause),
        .fault_pc(fault_pc),
        .i_req_valid(i_req_valid), .i_req_ready(i_req_ready), .i_req_addr(i_req_addr),
        .i_resp_valid(i_resp_valid), .i_resp_rdata(i_resp_rdata), .i_resp_err(i_resp_err),
        .d_req_valid(d_req_valid), .d_req_ready(d_req_ready), .d_req_addr(d_req_addr),
        .d_req_we(d_req_we), .d_req_wstrb(d_req_wstrb), .d_req_wdata(d_req_wdata),
        .d_resp_valid(d_resp_valid), .d_resp_rdata(d_resp_rdata), .d_resp_err(d_resp_err),
        .vcmd_valid(vcmd_valid), .vcmd_ready(vcmd_ready), .vcmd_op(vcmd_op),
        .vcmd_type(vcmd_type), .vcmd_vd(vcmd_vd), .vcmd_vs1(vcmd_vs1), .vcmd_vs2(vcmd_vs2),
        .vcmd_scalar(vcmd_scalar), .vec_busy(vec_busy),
        .cycles(cycles), .instret(instret)
    );

    logic [31:0] mem [0:MEM_WORDS-1];
    logic        written [0:MEM_WORDS-1];
    logic        expected_word [0:MEM_WORDS-1];

    integer seed = 136;
    logic   ideal = 1'b0;     // fixed latency, always ready, ideal vector unit
    integer fix_lat = 1;      // instruction / data response latency of the
    integer fix_dlat = 1;     // fixed-latency cases

    task automatic bad(input string msg);
        $display("MISMATCH tb_npu_scalar_core: %s (t=%0t)", msg, $time);
        $fatal(1);
    endtask

    function automatic logic in_range(input logic [31:0] addr);
        in_range = addr < 32'h0001_0000;
    endfunction

    // ------------------------------------------------------------------
    // Random ready / stall generation (inputs change on the falling edge)
    // ------------------------------------------------------------------
    // i_lat_n / d_lat_n: response latency of a handshake in the next cycle.
    integer i_lat_n = 1, d_lat_n = 1;
    always @(negedge clk) begin
        if (ideal) begin
            i_req_ready <= 1'b1;
            d_req_ready <= 1'b1;
            vcmd_ready  <= 1'b1;
            i_lat_n     <= fix_lat;
            d_lat_n     <= fix_dlat;
        end else begin
            i_req_ready <= ($random(seed) & 3) != 0;
            d_req_ready <= ($random(seed) & 3) != 0;
            vcmd_ready  <= ($random(seed) & 3) == 0;
            i_lat_n     <= ($random(seed) & 32'h7fff_ffff) % 5;
            d_lat_n     <= ($random(seed) & 32'h7fff_ffff) % 5;
        end
    end

    // ------------------------------------------------------------------
    // Instruction memory port: registered response (latency >= 1) or a
    // combinational one in the handshake cycle (latency 0).
    // ------------------------------------------------------------------
    logic        i_pend, i_resp_q, i_err_r, i_zl;
    integer      i_cnt;
    logic [31:0] i_addr_q, i_rdata_r;
    integer      i_stalls = 0, d_stalls = 0, v_stalls = 0;

    assign i_zl         = i_req_valid && i_req_ready && i_lat_n == 0;
    assign i_resp_valid = i_zl || i_resp_q;
    assign i_resp_err   = i_zl ? !in_range(i_req_addr) : i_err_r;
    assign i_resp_rdata = !i_zl ? i_rdata_r :
                          in_range(i_req_addr) ? mem[i_req_addr[15:2]] : 32'hdead_beef;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i_pend    <= 1'b0;
            i_cnt     <= 0;
            i_addr_q  <= '0;
            i_resp_q  <= 1'b0;
            i_err_r   <= 1'b0;
            i_rdata_r <= '0;
        end else begin
            i_resp_q <= 1'b0;
            i_err_r  <= 1'b0;
            if (i_pend) begin
                if (i_cnt <= 1) begin
                    i_pend    <= 1'b0;
                    i_resp_q  <= 1'b1;
                    i_err_r   <= !in_range(i_addr_q);
                    i_rdata_r <= in_range(i_addr_q) ? mem[i_addr_q[15:2]] : 32'hdead_beef;
                end else begin
                    i_cnt <= i_cnt - 1;
                end
            end else if (i_req_valid && i_req_ready && i_lat_n != 0) begin
                if (i_lat_n == 1) begin
                    i_resp_q  <= 1'b1;
                    i_err_r   <= !in_range(i_req_addr);
                    i_rdata_r <= in_range(i_req_addr) ? mem[i_req_addr[15:2]] : 32'hdead_beef;
                end else begin
                    i_pend   <= 1'b1;
                    i_cnt    <= i_lat_n - 1;
                    i_addr_q <= i_req_addr;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Data memory port (writes take effect at the handshake)
    // ------------------------------------------------------------------
    logic        d_pend, d_resp_q, d_err_r, d_err_q, d_zl;
    integer      d_cnt;
    logic [31:0] d_rdata_q, d_rdata_r;

    assign d_zl         = d_req_valid && d_req_ready && d_lat_n == 0;
    assign d_resp_valid = d_zl || d_resp_q;
    assign d_resp_err   = d_zl ? !in_range(d_req_addr) : d_err_r;
    assign d_resp_rdata = !d_zl ? d_rdata_r :
                          in_range(d_req_addr) ? mem[d_req_addr[15:2]] : 32'hdead_beef;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d_pend    <= 1'b0;
            d_cnt     <= 0;
            d_err_q   <= 1'b0;
            d_rdata_q <= '0;
            d_resp_q  <= 1'b0;
            d_err_r   <= 1'b0;
            d_rdata_r <= '0;
        end else begin
            d_resp_q <= 1'b0;
            d_err_r  <= 1'b0;
            if (d_pend) begin
                if (d_cnt <= 1) begin
                    d_pend    <= 1'b0;
                    d_resp_q  <= 1'b1;
                    d_err_r   <= d_err_q;
                    d_rdata_r <= d_rdata_q;
                end else begin
                    d_cnt <= d_cnt - 1;
                end
            end else if (d_req_valid && d_req_ready) begin
                if (in_range(d_req_addr) && d_req_we) begin
                    for (int k = 0; k < 4; k++) begin
                        if (d_req_wstrb[k]) mem[d_req_addr[15:2]][8*k +: 8] <= d_req_wdata[8*k +: 8];
                    end
                    written[d_req_addr[15:2]] <= 1'b1;
                end
                if (d_lat_n == 1) begin
                    d_resp_q  <= 1'b1;
                    d_err_r   <= !in_range(d_req_addr);
                    d_rdata_r <= in_range(d_req_addr) ? mem[d_req_addr[15:2]] : 32'hdead_beef;
                end else if (d_lat_n > 1) begin
                    d_pend    <= 1'b1;
                    d_cnt     <= d_lat_n - 1;
                    d_err_q   <= !in_range(d_req_addr);
                    d_rdata_q <= in_range(d_req_addr) ? mem[d_req_addr[15:2]] : 32'hdead_beef;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Vector unit model: command log and busy counter
    // ------------------------------------------------------------------
    logic [4:0]  log_op     [0:MAX_VCMDS-1];
    logic [1:0]  log_type   [0:MAX_VCMDS-1];
    logic [2:0]  log_vd     [0:MAX_VCMDS-1];
    logic [2:0]  log_vs1    [0:MAX_VCMDS-1];
    logic [2:0]  log_vs2    [0:MAX_VCMDS-1];
    logic [31:0] log_scalar [0:MAX_VCMDS-1];
    integer      log_n = 0;
    integer      busy_cnt;
    integer      total_vcmds = 0;

    assign vec_busy = busy_cnt != 0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_cnt <= 0;
        end else begin
            if (vcmd_valid && vcmd_ready) begin
                if (log_n >= MAX_VCMDS) bad("vector command log overflow");
                log_op[log_n]     <= vcmd_op;
                log_type[log_n]   <= vcmd_type;
                log_vd[log_n]     <= vcmd_vd;
                log_vs1[log_n]    <= vcmd_vs1;
                log_vs2[log_n]    <= vcmd_vs2;
                log_scalar[log_n] <= vcmd_scalar;
                log_n             <= log_n + 1;
                total_vcmds       <= total_vcmds + 1;
                busy_cnt <= ideal ? 0 : busy_cnt + 1 + (($random(seed) & 32'h7fff_ffff) % 24);
            end else if (busy_cnt != 0) begin
                busy_cnt <= busy_cnt - 1;
            end
        end
    end

    // ------------------------------------------------------------------
    // Protocol monitors
    // ------------------------------------------------------------------
    logic        i_stall_q, d_stall_q, v_stall_q;
    logic [31:0] i_addr_hold, d_addr_hold, d_wdata_hold, v_scalar_hold;
    logic [3:0]  d_wstrb_hold;
    logic        d_we_hold;
    logic [15:0] v_fields_hold;
    logic [68:0] d_acc_fields;      // {addr, we, wstrb, wdata} at acceptance
    logic        fence_pend, fence_ok;
    integer      fence_wait_cycles = 0;
    // Early-fetch coverage (white box: S_FETCH = 1, S_IWAIT = 2, S_EXEC = 4,
    // S_WB = 7, S_VCMD = 8): EXEC fetch accepted / stalled; next-instruction
    // response before the retire state, in it, or after it (IWAIT); a
    // response drained after a fault while stopped / while FETCH waits.
    integer      pf_accepts = 0, pf_stalls = 0, pf_resp_early = 0, pf_resp_ret = 0;
    integer      pf_resp_late = 0, drain_stopped = 0, drain_fetch = 0, pf_nt_stall = 0;
    integer      mem_wait_fetch = 0;   // MEM holds the data request for the fetch
    integer      drain_err = 0;        // a drained response carrying a bus error
    integer      vcmd_err_hold = 0;    // a fetch error response waiting in VCMD
    // Same-cycle responses per state: FETCH, EXEC, retry after EXEC, data.
    integer      zl_fetch = 0, zl_exec = 0, zl_retry = 0, zl_data = 0;
    logic        nxt_fetch_q;   // a next-instruction fetch was issued after DECODE

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i_stall_q  <= 1'b0;
            d_stall_q  <= 1'b0;
            nxt_fetch_q <= 1'b0;
            v_stall_q  <= 1'b0;
            fence_pend <= 1'b0;
            fence_ok   <= 1'b0;
        end else begin
            // Stability under backpressure.
            if (i_stall_q && (!i_req_valid || i_req_addr !== i_addr_hold))
                bad("instruction request changed while stalled");
            if (d_stall_q && (!d_req_valid || d_req_addr !== d_addr_hold ||
                              d_req_we !== d_we_hold || d_req_wstrb !== d_wstrb_hold ||
                              d_req_wdata !== d_wdata_hold))
                bad("data request changed while stalled");
            if (v_stall_q && (!vcmd_valid || vcmd_scalar !== v_scalar_hold ||
                              {vcmd_op, vcmd_type, vcmd_vd, vcmd_vs1, vcmd_vs2} !== v_fields_hold))
                bad("vector command changed while stalled");
            i_stall_q     <= i_req_valid && !i_req_ready;
            d_stall_q     <= d_req_valid && !d_req_ready;
            v_stall_q     <= vcmd_valid && !vcmd_ready;
            i_addr_hold   <= i_req_addr;
            d_addr_hold   <= d_req_addr;
            d_we_hold     <= d_req_we;
            d_wstrb_hold  <= d_req_wstrb;
            d_wdata_hold  <= d_req_wdata;
            v_scalar_hold <= vcmd_scalar;
            v_fields_hold <= {vcmd_op, vcmd_type, vcmd_vd, vcmd_vs1, vcmd_vs2};
            if (i_req_valid && !i_req_ready) i_stalls <= i_stalls + 1;
            if (d_req_valid && !d_req_ready) d_stalls <= d_stalls + 1;
            if (vcmd_valid && !vcmd_ready) v_stalls <= v_stalls + 1;
            if (dut.state == 4'd4 && i_req_valid) begin
                if (i_req_ready) pf_accepts <= pf_accepts + 1;
                else pf_stalls <= pf_stalls + 1;
                // not-taken branch: PC + 4 from op_b, held into the next state
                if (!i_req_ready && dut.nt_q) pf_nt_stall <= pf_nt_stall + 1;
            end
            if (i_zl) begin
                if (dut.state == 4'd1) zl_fetch <= zl_fetch + 1;
                else if (dut.state == 4'd4) zl_exec <= zl_exec + 1;
                else zl_retry <= zl_retry + 1;
            end
            if (d_zl) zl_data <= zl_data + 1;
            if (dut.state == 4'd5 && !d_req_valid) mem_wait_fetch <= mem_wait_fetch + 1;
            if (dut.state == 4'd4 || dut.state == 4'd3) nxt_fetch_q <= 1'b0;
            if (i_req_valid && i_req_ready && dut.state != 4'd1) nxt_fetch_q <= 1'b1;
            if (i_resp_valid && i_resp_err && (!running || (dut.state == 4'd1 && !i_zl)))
                drain_err <= drain_err + 1;
            if (i_resp_valid && i_resp_err && dut.state == 4'd8 && !vcmd_ready)
                vcmd_err_hold <= vcmd_err_hold + 1;
            // start clears the causes; they only change when the core stops.
            if (running && (halt_cause !== 1'b0 || fault_cause !== 3'd0))
                bad("halt / fault cause set while running");
            if (i_resp_valid) begin
                if (!running) drain_stopped <= drain_stopped + 1;
                else if (dut.state == 4'd1 && !i_zl) drain_fetch <= drain_fetch + 1;
                else if (dut.state == 4'd2 && nxt_fetch_q) pf_resp_late <= pf_resp_late + 1;
                else if (dut.state == 4'd7 || (dut.state == 4'd8 && vcmd_ready))
                    pf_resp_ret <= pf_resp_ret + 1;
                else if (dut.state != 4'd1 && dut.state != 4'd2)
                    pf_resp_early <= pf_resp_early + 1;
            end

            // Data request held from acceptance until the response.
            if (d_req_valid && d_req_ready)
                d_acc_fields <= {d_req_addr, d_req_we, d_req_wstrb, d_req_wdata};
            else if ((d_pend || d_resp_q) &&
                     {d_req_addr, d_req_we, d_req_wstrb, d_req_wdata} !== d_acc_fields)
                bad("data request fields changed before the response");

            // One outstanding request per port; address/strobe shape.
            if (i_req_valid && (i_pend || i_resp_q))
                bad("second outstanding instruction request");
            if (d_req_valid && (d_pend || d_resp_q))
                bad("second outstanding data request");
            if (i_req_valid && i_req_addr[1:0] != 2'b00)
                bad("misaligned instruction request");
            if (d_req_valid && d_req_addr[1:0] != 2'b00)
                bad("data request address not word aligned");
            if (d_req_valid && !d_req_we && d_req_wstrb != 4'b0000)
                bad("load request with byte strobes");
            if (d_req_valid && d_req_we && d_req_wstrb == 4'b0000)
                bad("store request without byte strobes");
            if (vcmd_valid && vcmd_op == 5'd31)
                bad("VFENCE sent on the vector command port");
            if ((i_req_valid || d_req_valid || vcmd_valid) && !running)
                bad("request while not running");

            // VFENCE: the next fetch may only follow a cycle with vec_busy low.
            // (A response drained after a fault is not an instruction.)
            if (i_resp_valid && running && !(dut.state == 4'd1 && !i_zl) &&
                !i_resp_err && i_resp_rdata[6:0] == 7'b0001011 &&
                i_resp_rdata[29:25] == 5'd31) begin
                fence_pend <= 1'b1;
                fence_ok   <= 1'b0;
            end else if (fence_pend) begin
                if (!vec_busy) fence_ok <= 1'b1;
                else fence_wait_cycles <= fence_wait_cycles + 1;
                if (i_req_valid && i_req_ready) begin
                    if (!fence_ok) bad("fetch after VFENCE while the vector unit was busy");
                    fence_pend <= 1'b0;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // No combinational input -> output path: every output is sampled just
    // after the rising edge and must be unchanged just after the falling
    // edge, where the environment changes the readies, the response latencies
    // (and with them the same-cycle responses), start and start_pc.
    // ------------------------------------------------------------------
    logic [254:0] outs_now, outs_pos;
    logic         outs_pos_ok = 1'b0;
    assign outs_now = {running, halted, halt_cause, fault, fault_cause, fault_pc,
                       i_req_valid, i_req_addr,
                       d_req_valid, d_req_addr, d_req_we, d_req_wstrb, d_req_wdata,
                       vcmd_valid, vcmd_op, vcmd_type, vcmd_vd, vcmd_vs1, vcmd_vs2, vcmd_scalar,
                       cycles, instret};
    always @(posedge clk) begin
        #1;
        outs_pos    = outs_now;
        outs_pos_ok = rst_n;
    end
    always @(negedge rst_n) outs_pos_ok = 1'b0;
    always @(negedge clk) begin
        #1;
        if (outs_pos_ok && rst_n && outs_now !== outs_pos)
            bad("output changed between clock edges (combinational path from an input)");
    end

    // ------------------------------------------------------------------
    // Test sequence
    // ------------------------------------------------------------------
    integer fd, code, i, n_init, n_runs, n_vcmds, n_mem, r;
    integer case_idx = 0, total_runs = 0, total_instret = 0, wait_cycles;
    integer exp_status, exp_cause, exp_instret, exp_cycles;
    integer exp_op, exp_type, exp_vd, exp_vs1, exp_vs2;
    integer case_lat, case_dlat, spurious_at;
    logic [31:0] addr, data, exp_pc, exp_fault_pc, exp_scalar;
    logic [31:0] exp_regs [1:31];
    logic [31:0] init_addr [0:MEM_WORDS-1];
    logic [31:0] init_data [0:MEM_WORDS-1];
    reg [8*32-1:0] kw, name;
    // Status and counters at the last stop; frozen until the next start.
    logic          snap_v = 1'b0;
    logic [102:0]  snap;
    integer        n_resets = 0;
    reg [8*256-1:0] header;

    task automatic load_memory;
        for (int k = 0; k < MEM_WORDS; k++) begin
            mem[k] = 32'd0;
            written[k] = 1'b0;
            expected_word[k] = 1'b0;
        end
        for (int k = 0; k < n_init; k++) mem[init_addr[k][15:2]] = init_data[k];
        log_n = 0;
    endtask

    task automatic check_hold(input string where);
        if (snap_v && {running, halted, fault, halt_cause, fault_cause, fault_pc, cycles,
                       instret} !== snap)
            bad($sformatf("status or counters changed while stopped (%0s)", where));
    endtask

    task automatic pulse_start(input logic [31:0] pc);
        @(negedge clk);
        check_hold("before start");
        snap_v = 1'b0;
        start = 1'b1;
        start_pc = pc;
        @(negedge clk);
        start = 1'b0;
        start_pc = 32'hffff_fff0;
    endtask

    task automatic check_idle_outputs(input string where);
        if (running !== 1'b0 || halted !== 1'b0 || fault !== 1'b0 ||
            i_req_valid !== 1'b0 || d_req_valid !== 1'b0 || vcmd_valid !== 1'b0 ||
            cycles !== 32'd0 || instret !== 32'd0 || fault_cause !== 3'd0 ||
            fault_pc !== 32'd0 || halt_cause !== 1'b0)
            bad($sformatf("outputs not idle %0s", where));
    endtask

    // Asynchronous reset while the core is in state s (white box: 0 IDLE,
    // 1 FETCH, 2 IWAIT, 3 DECODE, 4 EXEC, 5 MEM, 6 MWAIT, 7 WB, 8 VCMD,
    // 9 VFENCE, 10 HALT, 11 FAULT), during a run of the loaded program in the
    // random environment (restarted if it stops first; FAULT from a
    // misaligned start PC; IDLE right after another reset), not always in
    // the first cycle of that state.
    // Everything returns to idle, and the next run must reach its target
    // state again (a reset that leaves stale fetch tracking hangs there).
    task automatic reset_in_state(input logic [3:0] s);
        integer guard, skip;
        snap_v = 1'b0;
        log_n  = 0;
        skip = ($random(seed) & 32'h7fff_ffff) % 3;
        guard = 0;
        if (s == 4'd11) pulse_start(32'h0000_0002);
        else if (s != 4'd0) pulse_start(32'h0);
        while (!(dut.state == s && skip == 0)) begin
            if (dut.state == s) skip = skip - 1;
            if ((halted || fault) && s != 4'd10 && s != 4'd11) begin
                log_n = 0;
                pulse_start(32'h0);
            end else @(negedge clk);
            guard = guard + 1;
            if (guard > 100000) bad($sformatf("state %0d not reached for the reset sweep", s));
        end
        n_resets++;
        rst_n = 1'b0;
        #1;
        check_idle_outputs($sformatf("during a reset in state %0d", s));
        repeat (2) @(negedge clk);
        check_idle_outputs("while held in reset");
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        check_idle_outputs("after reset");
    endtask

    initial begin
        i_req_ready = 1'b0;
        d_req_ready = 1'b0;
        vcmd_ready  = 1'b0;
        fd = $fopen("vectors/scalar_core_programs.txt", "r");
        if (fd == 0) fd = $fopen("src/test/vectors/scalar_core_programs.txt", "r");
        if (fd == 0) bad("cannot open scalar_core_programs.txt");
        code = $fgets(header, fd);

        @(negedge clk);   // the first clock edge applies the reset
        check_idle_outputs("in reset");
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);
        check_idle_outputs("after reset");

        code = $fscanf(fd, "%s", kw);
        while (kw != "DONE") begin
            if (code != 1) bad("unexpected end of vector file");
            if (kw != "CASE") bad("vector file out of sync at CASE");
            code = $fscanf(fd, "%s %d %d %d", name, case_lat, case_dlat, n_init);
            for (i = 0; i < n_init; i++) begin
                code = $fscanf(fd, "%h %h", addr, data);
                init_addr[i] = addr;
                init_data[i] = data;
            end
            if (name == "vector") begin
                // Reset sweep on the vector program (every state is reachable).
                ideal = 1'b0;
                load_memory();
                // IDLE last in each round: the reset before it leaves the core idle.
                for (int round = 0; round < 3; round++)
                    for (int s = 1; s <= 12; s++) reset_in_state(s % 12);
            end
            ideal = case_lat >= 0;
            fix_lat = (case_lat >= 0) ? case_lat : 1;
            fix_dlat = (case_dlat >= 0) ? case_dlat : 1;
            load_memory();
            repeat (2) @(negedge clk);

            code = $fscanf(fd, "%s %d", kw, n_runs);
            if (kw != "RUNS") bad("vector file out of sync at RUNS");
            for (r = 0; r < n_runs; r++) begin
                code = $fscanf(fd, "%s %h %d %d %h %d %d", kw, exp_pc, exp_status, exp_cause,
                               exp_fault_pc, exp_instret, exp_cycles);
                if (kw != "RUN") bad("vector file out of sync at RUN");
                for (i = 1; i < 32; i++) begin
                    code = $fscanf(fd, "%h", data);
                    exp_regs[i] = data;
                end

                pulse_start(exp_pc);
                spurious_at = ideal ? -1 : 5 + (($random(seed) & 32'h7fff_ffff) % 200);
                wait_cycles = 0;
                while (!(halted || fault)) begin
                    if (wait_cycles == spurious_at && running) begin
                        // start while running must be ignored
                        start = 1'b1;
                        start_pc = 32'h0000_0f00;
                    end
                    @(negedge clk);
                    start = 1'b0;
                    wait_cycles++;
                    if (wait_cycles > RUN_TIMEOUT) bad($sformatf("%0s run %0d timed out", name, r));
                end
                if (running) bad("running together with halted/fault");
                if (halted && fault) bad("halted and fault together");
                if (exp_status == 1) begin
                    if (!halted || halt_cause !== exp_cause[0] || fault_cause !== 3'd0)
                        bad($sformatf("%0s run %0d: want halt cause %0d, got halted=%0b fault=%0b halt_cause=%0d fault_cause=%0d fault_pc=%h instret=%0d",
                                      name, r, exp_cause, halted, fault, halt_cause, fault_cause, fault_pc, instret));
                end else begin
                    if (!fault || fault_cause !== exp_cause[2:0] || halt_cause !== 1'b0)
                        bad($sformatf("%0s run %0d: want fault cause %0d, got halted=%0b fault=%0b fault_cause=%0d fault_pc=%h instret=%0d",
                                      name, r, exp_cause, halted, fault, fault_cause, fault_pc, instret));
                end
                if (fault_pc !== exp_fault_pc)
                    bad($sformatf("%0s run %0d: fault_pc %h, want %h", name, r, fault_pc, exp_fault_pc));
                if (instret !== exp_instret)
                    bad($sformatf("%0s run %0d: instret %0d, want %0d", name, r, instret, exp_instret));
                if (exp_cycles >= 0 && cycles !== exp_cycles)
                    bad($sformatf("%0s run %0d: cycles %0d, want %0d", name, r, cycles, exp_cycles));
                for (i = 1; i < 32; i++) begin
                    if (dut.rf[i] !== exp_regs[i])
                        bad($sformatf("%0s run %0d: x%0d = %h, want %h", name, r, i, dut.rf[i], exp_regs[i]));
                end
                if (exp_cycles >= 0)
                    $display("tb_npu_scalar_core cpi %0s cycles=%0d instret=%0d", name, cycles, instret);
                snap   = {running, halted, fault, halt_cause, fault_cause, fault_pc, cycles, instret};
                snap_v = 1'b1;
                total_runs++;
                total_instret += instret;
                // Fixed-latency cases restart at once (abandoned fetch in flight).
                if (!ideal)
                    repeat ((($random(seed) & 32'h7fff_ffff) % 3)) @(negedge clk);
                check_hold("after the stop");
            end

            code = $fscanf(fd, "%s %d", kw, n_vcmds);
            if (kw != "VCMDS") bad("vector file out of sync at VCMDS");
            if (n_vcmds != log_n)
                bad($sformatf("%0s: %0d vector commands sent, want %0d", name, log_n, n_vcmds));
            for (i = 0; i < n_vcmds; i++) begin
                code = $fscanf(fd, "%d %d %d %d %d %h", exp_op, exp_type, exp_vd, exp_vs1, exp_vs2,
                               exp_scalar);
                if (log_op[i] != exp_op[4:0] || log_type[i] != exp_type[1:0] ||
                    log_vd[i] != exp_vd[2:0] || log_vs1[i] != exp_vs1[2:0] ||
                    log_vs2[i] != exp_vs2[2:0] || log_scalar[i] !== exp_scalar)
                    bad($sformatf("%0s: vector command %0d = op %0d type %0d vd %0d vs1 %0d vs2 %0d scalar %h",
                                  name, i, log_op[i], log_type[i], log_vd[i], log_vs1[i], log_vs2[i],
                                  log_scalar[i]));
            end

            code = $fscanf(fd, "%s %d", kw, n_mem);
            if (kw != "MEM") bad("vector file out of sync at MEM");
            for (i = 0; i < n_mem; i++) begin
                code = $fscanf(fd, "%h %h", addr, data);
                expected_word[addr[15:2]] = 1'b1;
                if (mem[addr[15:2]] !== data)
                    bad($sformatf("%0s: mem[%h] = %h, want %h", name, addr, mem[addr[15:2]], data));
            end
            for (i = 0; i < MEM_WORDS; i++) begin
                if (written[i] && !expected_word[i])
                    bad($sformatf("%0s: unexpected store to %h", name, i * 4));
            end
            case_idx++;
            code = $fscanf(fd, "%s", kw);
        end

        if (fence_wait_cycles == 0 || i_stalls == 0 || d_stalls == 0 || v_stalls == 0)
            bad("stall coverage hole");
        if (pf_accepts == 0 || pf_stalls == 0 || pf_resp_early == 0 || pf_resp_ret == 0 ||
            pf_resp_late == 0 || drain_stopped == 0 || drain_fetch == 0 || pf_nt_stall == 0 ||
            mem_wait_fetch == 0 || drain_err == 0 || vcmd_err_hold == 0)
            bad("early fetch coverage hole");
        if (zl_fetch == 0 || zl_exec == 0 || zl_retry == 0 || zl_data == 0)
            bad("same-cycle response coverage hole");
        $display("PASS tb_npu_scalar_core cases=%0d runs=%0d instret=%0d vcmds=%0d fence_wait_cycles=%0d stalls(i/d/v)=%0d/%0d/%0d early_fetch(acc/stall/nt_stall/early/ret/late)=%0d/%0d/%0d/%0d/%0d/%0d drain(stop/fetch/err)=%0d/%0d/%0d vcmd_err_hold=%0d mem_wait_fetch=%0d same_cycle(fetch/exec/retry/data)=%0d/%0d/%0d/%0d resets=%0d",
                 case_idx, total_runs, total_instret, total_vcmds, fence_wait_cycles,
                 i_stalls, d_stalls, v_stalls, pf_accepts, pf_stalls, pf_nt_stall, pf_resp_early, pf_resp_ret,
                 pf_resp_late, drain_stopped, drain_fetch, drain_err, vcmd_err_hold, mem_wait_fetch,
                 zl_fetch, zl_exec, zl_retry, zl_data, n_resets);
        $finish;
    end
endmodule
