`timescale 1ns/1ps

// Self-checking test of npu_mem_arbiter (P0_READ_ONLY = 0 and 1), each in a
// harness with two random line-protocol masters and a random downstream
// slave. Master k only uses addresses with bit 20 = k, and every line's data
// encodes its address, so a response routed to the wrong master or a
// corrupted payload is detected. Reads return 8 x 32-bit beats with random
// gaps. Checks: exactly one response pulse per write and exactly 8 beats per
// read, read data integrity per beat, write data reaching the slave, downstream payload
// stable while waiting for ready and until the response (also with a slave
// that raises ready only after seeing valid), one downstream
// transaction at a time, round-robin fairness (a waiting master is passed
// over at most once), no starvation, idle outputs during reset.

module tb_npu_mem_arbiter_harness #(
    parameter NAME = "rw",
    parameter integer P0_READ_ONLY = 0,
    parameter integer NOPS = 1500,       // requests per master
    parameter integer SEED = 1
) (
    input  logic clk,
    output logic done
);
    localparam integer LW = 256;
    localparam integer BW = 32;                // read beat width
    localparam integer RBEATS = LW / BW;

    logic          rst_n;
    logic          s_valid [0:1];
    logic          s_ready [0:1];
    logic [31:0]   s_addr  [0:1];
    logic          s_we    [0:1];
    logic [LW-1:0] s_wdata [0:1];
    logic          s_rvalid[0:1];
    logic [BW-1:0] s_rdata [0:1];
    logic          m_valid, m_ready, m_we, m_rvalid;
    logic [31:0]   m_addr;
    logic [LW-1:0] m_wdata;
    logic [BW-1:0] m_rdata;

    npu_mem_arbiter #(.LINE_BYTES(32), .BEAT_W(BW), .P0_READ_ONLY(P0_READ_ONLY)) dut (
        .clk(clk), .rst_n(rst_n),
        .s0_mem_req_valid(s_valid[0]), .s0_mem_req_ready(s_ready[0]),
        .s0_mem_req_addr(s_addr[0]), .s0_mem_req_we(s_we[0]),
        .s0_mem_req_wdata(s_wdata[0]), .s0_mem_resp_valid(s_rvalid[0]),
        .s0_mem_resp_rdata(s_rdata[0]),
        .s1_mem_req_valid(s_valid[1]), .s1_mem_req_ready(s_ready[1]),
        .s1_mem_req_addr(s_addr[1]), .s1_mem_req_we(s_we[1]),
        .s1_mem_req_wdata(s_wdata[1]), .s1_mem_resp_valid(s_rvalid[1]),
        .s1_mem_resp_rdata(s_rdata[1]),
        .m_mem_req_valid(m_valid), .m_mem_req_ready(m_ready),
        .m_mem_req_addr(m_addr), .m_mem_req_we(m_we), .m_mem_req_wdata(m_wdata),
        .m_mem_resp_valid(m_rvalid), .m_mem_resp_rdata(m_rdata)
    );

    integer seed = SEED;
    integer mode;            // 0: random gaps, 1: both masters back to back
    integer slave_ready_pct, slave_max_delay;
    logic   ready_after_valid;       // slave ready only in response to valid

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_mem_arbiter %s: %s (t=%0t)", NAME, msg, $time);
        $fatal(1, "tb_npu_mem_arbiter %s", NAME);
    endtask

    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    // Line content pattern: read data and write data both encode the address
    function automatic logic [LW-1:0] pattern(input logic [31:0] a, input logic w);
        logic [LW-1:0] p;
        for (int i = 0; i < LW / 32; i++)
            p[i*32 +: 32] = (a ^ (32'h9E37_79B9 * (i + 1))) ^ (w ? 32'hFFFF_0000 : 32'h0);
        pattern = p;
    endfunction

    // ------------------------------------------------------------------
    // Downstream slave
    // ------------------------------------------------------------------
    logic          sl_busy;
    integer        sl_delay;
    logic [31:0]   sl_addr;
    logic          sl_we;
    logic [LW-1:0] sl_wdata;
    logic          m_wait;          // valid without ready last cycle
    logic [31:0]   w_addr;
    logic          w_we;
    logic [LW-1:0] w_wdata;
    integer        n_down [0:1];
    integer        n_wr_down;

    integer        sl_beat;
    logic [LW-1:0] sl_line;

    // One slave step: the write response, or the next read beat
    task automatic sl_step;
        m_rvalid <= 1'b1;
        if (sl_we) begin
            m_rdata <= $random(seed);
            sl_busy <= 1'b0;
        end else begin
            m_rdata <= sl_line[sl_beat*BW +: BW];
            sl_beat = sl_beat + 1;
            if (sl_beat == RBEATS)
                sl_busy <= 1'b0;
            else
                sl_delay = (rnd(3) == 0) ? rnd(slave_max_delay + 2) : 0;
        end
    endtask

    always @(posedge clk) begin
        m_rvalid <= 1'b0;
        m_rdata <= $random(seed);
        if (!rst_n) begin
            sl_busy <= 1'b0;
            m_wait <= 1'b0;
        end else begin
            // payload stable while valid waits for ready
            if (m_wait && (!m_valid || m_addr !== w_addr || m_we !== w_we ||
                           (w_we && m_wdata !== w_wdata)))
                fail("downstream request changed while waiting for ready");
            m_wait <= m_valid && !m_ready;
            w_addr <= m_addr;
            w_we <= m_we;
            w_wdata <= m_wdata;
            if (m_valid && m_ready) begin
                if (sl_busy) fail("second downstream request while one is outstanding");
                sl_addr = m_addr;
                sl_we = m_we;
                sl_wdata = m_wdata;
                sl_line = pattern(m_addr, 1'b0);
                sl_beat = 0;
                if (m_we) begin
                    if (m_wdata !== pattern(m_addr, 1'b1)) fail("downstream write data");
                    if (P0_READ_ONLY != 0 && m_addr[20] == 1'b0)
                        fail("master 0 write forwarded in read-only mode");
                    n_wr_down = n_wr_down + 1;
                end
                n_down[m_addr[20]] = n_down[m_addr[20]] + 1;
                sl_busy <= 1'b1;
                sl_delay = rnd(slave_max_delay + 1);
                if (sl_delay == 0)
                    sl_step();
            end else if (sl_busy) begin
                if (m_valid) fail("downstream valid while a request is outstanding");
                if (m_addr !== sl_addr || m_we !== sl_we || (sl_we && m_wdata !== sl_wdata))
                    fail("downstream payload changed before the response");
                sl_delay = sl_delay - 1;
                if (sl_delay <= 0)
                    sl_step();
            end
        end
        if (ready_after_valid)
            m_ready <= m_valid && !m_ready && rnd(100) < slave_ready_pct;
        else
            m_ready <= rnd(100) < slave_ready_pct;
    end

    // ------------------------------------------------------------------
    // Fairness monitor: count grants to the other master while one waits
    // ------------------------------------------------------------------
    integer passed_over [0:1];
    integer max_wait [0:1];
    integer wait_cyc [0:1];
    always @(posedge clk) begin
        if (!rst_n) begin
            for (int k = 0; k < 2; k++) begin
                passed_over[k] <= 0;
                wait_cyc[k] <= 0;
            end
        end else begin
            for (int k = 0; k < 2; k++) begin
                if (s_valid[k] && s_ready[k]) begin
                    passed_over[k] <= 0;
                    wait_cyc[k] <= 0;
                end else if (s_valid[k]) begin
                    wait_cyc[k] <= wait_cyc[k] + 1;
                    if (wait_cyc[k] > max_wait[k]) max_wait[k] = wait_cyc[k];
                    if (s_valid[1-k] && s_ready[1-k]) begin
                        if (passed_over[k] >= 1)
                            fail("round robin: a waiting master was passed over twice");
                        passed_over[k] <= passed_over[k] + 1;
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Masters
    // ------------------------------------------------------------------
    integer n_resp [0:1];
    logic   outstanding [0:1];
    integer pulses_left [0:1];        // response pulses still expected
    integer n_wr [0:1];

    always @(posedge clk) begin
        for (int k = 0; k < 2; k++)
            if (!rst_n) begin
                pulses_left[k] = 0;
            end else begin
                if (s_rvalid[k]) begin
                    if (!outstanding[k] || pulses_left[k] == 0)
                        fail("response to a master without an outstanding request");
                    pulses_left[k] = pulses_left[k] - 1;
                end
                if (s_valid[k] && s_ready[k])
                    pulses_left[k] = s_we[k] ? 1 : RBEATS;
            end
    end

    task automatic master(input integer k);
        logic [LW-1:0] rd_line;
        logic [31:0] a;
        logic we;
        integer lat;
        for (int n = 0; n < NOPS; n++) begin
            if (mode == 0)
                repeat (rnd(4)) @(negedge clk);
            a = (rnd(65536) & 32'h0000_FFE0) | (k << 20);
            we = (P0_READ_ONLY != 0 && k == 0) ? 1'b0 : (rnd(3) == 0);
            s_valid[k] = 1'b1;
            s_addr[k] = a;
            s_we[k] = we;
            s_wdata[k] = we ? pattern(a, 1'b1) : {LW/32{$random(seed)}};
            rd_line = pattern(a, 1'b0);
            while (!s_ready[k]) @(negedge clk);
            @(negedge clk);
            s_valid[k] = 1'b0;
            outstanding[k] = 1'b1;
            // payload held until the response (line protocol); a read
            // collects RBEATS beats, which may have gaps
            for (int bt = 0; bt < (we ? 1 : RBEATS); bt++) begin
                lat = 0;
                while (!s_rvalid[k]) begin
                    @(negedge clk);
                    lat = lat + 1;
                    if (lat > 1000) fail("no response (starvation)");
                end
                if (!we && s_rdata[k] !== rd_line[bt*BW +: BW])
                    fail("read data routed or corrupted");
                @(negedge clk);
            end
            if (we) n_wr[k] = n_wr[k] + 1;
            n_resp[k] = n_resp[k] + 1;
            outstanding[k] = 1'b0;
            if (s_rvalid[k]) fail("extra response pulse");
        end
    endtask

    initial begin
        rst_n = 1'b0;
        done = 1'b0;
        for (int k = 0; k < 2; k++) begin
            s_valid[k] = 1'b0;
            s_addr[k] = '0;
            s_we[k] = 1'b0;
            s_wdata[k] = '0;
            n_resp[k] = 0;
            n_down[k] = 0;
            n_wr[k] = 0;
            outstanding[k] = 1'b0;
            max_wait[k] = 0;
        end
        n_wr_down = 0;
        mode = 0;
        ready_after_valid = 1'b0;
        slave_ready_pct = 60;
        slave_max_delay = 5;
        repeat (3) @(negedge clk);
        if (m_valid || s_ready[0] || s_ready[1] || s_rvalid[0] || s_rvalid[1])
            fail("outputs active during reset");
        rst_n = 1'b1;
        @(negedge clk);

        // Phase 1: both masters, random gaps, random slave timing
        fork
            master(0);
            master(1);
        join
        if (n_resp[0] != NOPS || n_resp[1] != NOPS) fail("response count");
        // Phase 2: both masters back to back, ready-always slave
        mode = 1;
        slave_ready_pct = 100;
        slave_max_delay = 0;
        fork
            master(0);
            master(1);
        join
        // Phase 3: master 1 alone, then master 0 alone, with a slave whose
        // ready waits for valid
        mode = 0;
        ready_after_valid = 1'b1;
        slave_ready_pct = 50;
        slave_max_delay = 3;
        master(1);
        master(0);
        if (n_resp[0] != 3 * NOPS || n_resp[1] != 3 * NOPS) fail("response count");
        if (n_down[0] != 3 * NOPS || n_down[1] != 3 * NOPS) fail("downstream count");
        if (n_wr_down != n_wr[0] + n_wr[1]) fail("write count");
        $display("INFO tb_npu_mem_arbiter %s: %0d+%0d transactions (%0d writes), longest request wait %0d/%0d cycles",
                 NAME, n_resp[0], n_resp[1], n_wr_down, max_wait[0], max_wait[1]);
        done = 1'b1;
    end
endmodule

module tb_npu_mem_arbiter;
    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic d_rw, d_ro;

    tb_npu_mem_arbiter_harness #(.NAME("rw"), .P0_READ_ONLY(0), .NOPS(1500), .SEED(5))
        h_rw (.clk(clk), .done(d_rw));
    tb_npu_mem_arbiter_harness #(.NAME("p0ro"), .P0_READ_ONLY(1), .NOPS(1500), .SEED(6))
        h_ro (.clk(clk), .done(d_ro));

    initial begin
        wait (d_rw && d_ro);
        $display("PASS tb_npu_mem_arbiter rw p0ro");
        $finish;
    end
    initial begin
        #20_000_000;
        $display("MISMATCH tb_npu_mem_arbiter: timeout");
        $fatal(1, "timeout");
    end
endmodule
