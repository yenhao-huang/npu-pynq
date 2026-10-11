`timescale 1ns/1ps

// Self-checking test of npu_axi4_line_bridge against an AXI4 slave model
// with random AR/AW/W ready, random R beat gaps and random B delay.
// Checks: burst fields (len = 3, size = 3, INCR, cache 4'b0011, prot 0,
// line-aligned address even for unaligned request addresses), AXI valid
// stability rules on AR/AW/W, wstrb all ones and wlast on the last beat only,
// read data and write data integrity against a reference memory, exactly
// one mem_resp_valid per write and exactly 4 read beats (64-bit, in address
// order) per read with the line protocol (payload held until the response is
// complete), no response pulse without an outstanding request, RRESP/BRESP
// error injection: exactly one error pulse per erroneous R beat / B
// response, in the cycle of its mem_resp_valid, and never otherwise;
// zero-wait latency. A phase with a
// slave that raises AR/AW/W ready only after seeing valid catches a valid
// that waits for ready (AXI forbids it; it would deadlock).

module tb_npu_axi4_line_bridge;
    localparam integer LW = 256;
    localparam integer BEATS = 4;
    localparam logic [31:0] BASE = 32'h2000_0000;
    localparam integer REGION = 16384;

    logic clk = 1'b0;
    always #5 clk = ~clk;

    logic          rst_n;
    logic          mem_req_valid, mem_req_ready, mem_req_we, mem_resp_valid, error;
    logic [31:0]   mem_req_addr;
    logic [LW-1:0] mem_req_wdata;
    logic [63:0]   mem_resp_rdata;
    logic [31:0]   araddr, awaddr;
    logic [7:0]    arlen, awlen;
    logic [2:0]    arsize, awsize, arprot, awprot;
    logic [1:0]    arburst, awburst, rresp, bresp;
    logic [3:0]    arcache, awcache;
    logic          arvalid, arready, rlast, rvalid, rready;
    logic          awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [63:0]   rdata, wdata;
    logic [7:0]    wstrb;

    npu_axi4_line_bridge #(.LINE_BYTES(32)) dut (
        .clk(clk), .rst_n(rst_n),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_we(mem_req_we),
        .mem_req_wdata(mem_req_wdata), .mem_resp_valid(mem_resp_valid),
        .mem_resp_rdata(mem_resp_rdata), .error(error),
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

    integer seed = 77;
    logic [7:0] smem [0:REGION-1];    // slave memory
    logic [7:0] refm [0:REGION-1];    // reference

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_axi4_line_bridge: %s (t=%0t)", msg, $time);
        $fatal(1, "tb_npu_axi4_line_bridge");
    endtask

    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    // Slave timing knobs
    logic   zero_wait;
    logic   ready_after_valid;        // ready only in response to valid
    integer ready_pct, r_inject_pct, b_inject_pct;
    integer n_inject;                 // error responses injected so far

    // ------------------------------------------------------------------
    // AXI4 slave model
    // ------------------------------------------------------------------
    logic        r_active;
    integer      r_beat, r_err_beat;
    logic [31:0] r_addr;
    logic        aw_got;
    logic [31:0] aw_addr;
    integer      w_cnt, b_delay, b_err;
    logic [63:0] w_buf [0:BEATS-1];
    logic        b_pend;
    // previous-cycle values for the valid stability rules
    logic        ar_wait, aw_wait, w_wait;
    logic [31:0] ar_prev, aw_prev;
    logic [63:0] w_prev;
    logic        w_last_prev;

    task automatic check_burst(input logic [31:0] a, input logic [7:0] len,
                               input logic [2:0] size, input logic [1:0] burst,
                               input logic [3:0] cache, input logic [2:0] prot);
        if (a[4:0] != 5'd0) fail("burst address not line aligned");
        if (a < BASE || a >= BASE + REGION) fail("burst address outside the region");
        if (len != 8'd3 || size != 3'd3 || burst != 2'b01 || cache != 4'b0011 || prot != 3'b000)
            fail("burst fields");
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            rvalid <= 1'b0;
            bvalid <= 1'b0;
            r_active = 1'b0;
            aw_got = 1'b0;
            w_cnt = 0;
            b_pend = 1'b0;
            ar_wait <= 1'b0;
            aw_wait <= 1'b0;
            w_wait <= 1'b0;
        end else begin
            // valid must stay high with a stable payload until ready
            if (ar_wait && (!arvalid || araddr !== ar_prev)) fail("AR changed before arready");
            if (aw_wait && (!awvalid || awaddr !== aw_prev)) fail("AW changed before awready");
            if (w_wait && (!wvalid || wdata !== w_prev || wlast !== w_last_prev))
                fail("W changed before wready");
            ar_wait <= arvalid && !arready;
            aw_wait <= awvalid && !awready;
            w_wait <= wvalid && !wready;
            ar_prev <= araddr;
            aw_prev <= awaddr;
            w_prev <= wdata;
            w_last_prev <= wlast;

            // AR
            if (arvalid && arready) begin
                check_burst(araddr, arlen, arsize, arburst, arcache, arprot);
                if (r_active) fail("second AR before the read burst finished");
                r_active = 1'b1;
                r_addr = araddr;
                r_beat = 0;
                r_err_beat = (rnd(100) < r_inject_pct) ? rnd(BEATS) : -1;
            end
            // R
            if (rvalid && rready) begin
                r_beat = r_beat + 1;
                if (rlast) r_active = 1'b0;
            end
            if (r_active && (!rvalid || rready) && (zero_wait || rnd(100) < ready_pct)) begin
                rvalid <= 1'b1;
                for (int b = 0; b < 8; b++)
                    rdata[b*8 +: 8] <= smem[r_addr - BASE + 8 * r_beat + b];
                rlast <= (r_beat == BEATS - 1);
                if (r_beat == r_err_beat) begin
                    rresp <= rnd(2) ? 2'b10 : 2'b11;
                    n_inject = n_inject + 1;
                end else begin
                    rresp <= 2'b00;
                end
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
                rdata <= {$random(seed), $random(seed)};
                rlast <= rnd(2);
                rresp <= rnd(4);
            end

            // AW
            if (awvalid && awready) begin
                check_burst(awaddr, awlen, awsize, awburst, awcache, awprot);
                if (aw_got || b_pend) fail("second AW before B");
                aw_got = 1'b1;
                aw_addr = awaddr;
            end
            // W (may lead AW)
            if (wvalid && wready) begin
                if (w_cnt >= BEATS || b_pend) fail("too many W beats");
                if (wstrb != 8'hFF) fail("wstrb");
                if (wlast != (w_cnt == BEATS - 1)) fail("wlast");
                w_buf[w_cnt] = wdata;
                w_cnt = w_cnt + 1;
            end
            if (!b_pend && aw_got && w_cnt == BEATS) begin
                b_pend = 1'b1;
                b_delay = zero_wait ? 0 : rnd(5);
                b_err = rnd(100) < b_inject_pct;
            end
            // B
            if (bvalid && bready) begin
                bvalid <= 1'b0;
                bresp <= rnd(4);
                for (int i = 0; i < BEATS; i++)
                    for (int b = 0; b < 8; b++)
                        smem[aw_addr - BASE + 8 * i + b] = w_buf[i][b*8 +: 8];
                b_pend = 1'b0;
                aw_got = 1'b0;
                w_cnt = 0;
            end else if (b_pend && !bvalid) begin
                if (b_delay == 0) begin
                    bvalid <= 1'b1;
                    if (b_err) begin
                        bresp <= rnd(2) ? 2'b10 : 2'b11;
                        n_inject = n_inject + 1;
                    end else begin
                        bresp <= 2'b00;
                    end
                end else begin
                    b_delay = b_delay - 1;
                end
            end
        end
        if (ready_after_valid) begin
            arready <= arvalid && !arready && rnd(100) < ready_pct;
            awready <= awvalid && !awready && rnd(100) < ready_pct;
            wready <= wvalid && !wready && rnd(100) < ready_pct;
        end else begin
            arready <= zero_wait || rnd(100) < ready_pct;
            awready <= zero_wait || rnd(100) < ready_pct;
            wready <= zero_wait || rnd(100) < ready_pct;
        end
    end

    // ------------------------------------------------------------------
    // Line-side requester
    // ------------------------------------------------------------------
    integer n_acc, n_resp, pulses_left;
    integer n_err;                    // error pulses seen
    logic   outstanding;
    always @(posedge clk) begin
        if (!rst_n) begin
            pulses_left = 0;
        end else begin
            if (mem_resp_valid) begin
                if (!outstanding || pulses_left == 0) fail("mem_resp_valid without a request");
                pulses_left = pulses_left - 1;
                if (pulses_left == 0)
                    n_resp <= n_resp + 1;
            end
            if (mem_req_valid && mem_req_ready) begin
                if (pulses_left != 0) fail("request accepted while a response is incomplete");
                n_acc <= n_acc + 1;
                pulses_left = mem_req_we ? 1 : BEATS;
            end
            if (error) begin
                if (!mem_resp_valid) fail("error pulse without a response pulse");
                n_err = n_err + 1;
                if (n_err > n_inject) fail("error pulse without an error response");
            end
        end
    end

    integer last_lat;

    task automatic line_op(input logic we, input integer off);
        logic [LW-1:0] d, exp;
        integer lat;
        for (int i = 0; i < LW / 32; i++)
            d[i*32 +: 32] = $random(seed);
        mem_req_valid = 1'b1;
        mem_req_addr = BASE + off;            // low bits may be unaligned
        mem_req_we = we;
        mem_req_wdata = d;
        while (!mem_req_ready) @(negedge clk);
        @(negedge clk);
        mem_req_valid = 1'b0;
        outstanding = 1'b1;
        off = off & ~31;
        for (int b = 0; b < 32; b++)
            exp[b*8 +: 8] = refm[off + b];
        // one pulse for a write, BEATS beats (gaps allowed) for a read
        lat = 1;
        for (int bt = 0; bt < (we ? 1 : BEATS); bt++) begin
            if (bt > 0) begin
                @(negedge clk);
                lat = lat + 1;
            end
            while (!mem_resp_valid) begin
                @(negedge clk);
                lat = lat + 1;
                if (lat > 500) fail("no response");
            end
            if (!we && mem_resp_rdata !== exp[bt*64 +: 64]) fail("read beat data");
        end
        if (we) begin
            for (int b = 0; b < 32; b++)
                refm[off + b] = d[b*8 +: 8];
        end
        last_lat = lat;
        @(negedge clk);
        outstanding = 1'b0;
        if (n_err != n_inject) fail("error pulse count differs from the injected error responses");
        if (mem_resp_valid) fail("extra mem_resp_valid");
    endtask

    task automatic run_random(input integer n);
        for (int i = 0; i < n; i++) begin
            repeat (rnd(3)) @(negedge clk);
            line_op(rnd(2), rnd(REGION));
        end
    endtask

    integer lat_rd, lat_wr;

    task automatic do_reset;
        @(negedge clk);
        rst_n = 1'b0;
        n_inject = 0;
        n_err = 0;
        @(negedge clk);
        if (error || mem_resp_valid) fail("outputs active during reset");
        rst_n = 1'b1;
        n_acc = 0;
        n_resp = 0;
    endtask

    initial begin
        rst_n = 1'b0;
        mem_req_valid = 1'b0;
        mem_req_addr = '0;
        mem_req_we = 1'b0;
        mem_req_wdata = '0;
        zero_wait = 1'b1;
        ready_after_valid = 1'b0;
        ready_pct = 100;
        r_inject_pct = 0;
        b_inject_pct = 0;
        n_inject = 0;
        n_err = 0;
        n_acc = 0;
        n_resp = 0;
        outstanding = 1'b0;
        for (int i = 0; i < REGION; i++) begin
            smem[i] = $random(seed);
            refm[i] = smem[i];
        end
        repeat (3) @(negedge clk);
        if (arvalid || awvalid || wvalid || mem_resp_valid || error)
            fail("outputs active during reset");
        rst_n = 1'b1;
        @(negedge clk);

        // zero-wait latency
        line_op(1'b0, 64);
        lat_rd = last_lat;
        line_op(1'b1, 96);
        lat_wr = last_lat;
        line_op(1'b0, 96);
        $display("INFO tb_npu_axi4_line_bridge: zero-wait latency read=%0d write=%0d cycles (acceptance to the last read beat / write response)",
                 lat_rd, lat_wr);
        if (lat_rd != BEATS + 2 || lat_wr != BEATS + 2) fail("zero-wait latency");

        // random timing, no error responses: no error pulse
        zero_wait = 1'b0;
        ready_pct = 40;
        run_random(1500);
        if (n_err != 0) fail("error pulse without injection");

        // BRESP faults only, then RRESP faults only, then both, with a reset
        // in between: one pulse per erroneous response (checked per line)
        b_inject_pct = 10;
        run_random(400);
        if (n_inject == 0) fail("no BRESP fault was injected");
        do_reset();
        b_inject_pct = 0;
        r_inject_pct = 10;
        run_random(400);
        if (n_inject == 0) fail("no RRESP fault was injected");
        do_reset();
        b_inject_pct = 10;
        run_random(300);
        do_reset();
        r_inject_pct = 0;
        ready_pct = 80;
        run_random(1000);
        ready_after_valid = 1'b1;
        ready_pct = 70;
        run_random(400);
        ready_after_valid = 1'b0;
        zero_wait = 1'b1;
        run_random(500);
        if (n_acc != n_resp) fail("request and response counts differ");
        $display("PASS tb_npu_axi4_line_bridge 4500 random line transfers incl. SLVERR/DECERR injection");
        $finish;
    end

    initial begin
        #10_000_000;
        $display("MISMATCH tb_npu_axi4_line_bridge: timeout");
        $fatal(1, "timeout");
    end
endmodule
