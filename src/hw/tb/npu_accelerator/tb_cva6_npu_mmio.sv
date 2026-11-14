`timescale 1ns/1ps

// MMIO job registers of the CVA6 scalar core (cva6_npu_mmio), driven through
// its AXI4 slave port as CVA6's uncached path would: single-beat 32-bit
// accesses on the 64-bit bus, the lane chosen by address bit 2. The matrix core
// and the tensor DMA are replaced by their handshake signals (ctl_accept,
// ld_ready / st_ready, jobs_written, error strobes), which the testbench drives.
//
// Covers: register write / read back (both lanes, byte strobes), DOORBELL issue
// (ctl_start and shape registered one cycle after the DMA hand-off, with the
// right addresses and counts), the write response held while the pending job
// cannot be accepted, the cooldown, FENCE held until every issued job is
// written back, DONE pulse and code, SLVERR for unknown offsets, read-only
// writes and bursts, bad shape and misalignment as latched faults, a matrix
// core fault dropping later jobs, and start clearing the fault.
//
// Inputs change on the falling edge; the DUT samples them on the rising edge.
module tb_cva6_npu_mmio;
    localparam integer ROWS = 16;
    localparam integer COLUMNS = 16;
    localparam integer MAX_K = 256;
    localparam integer IDW = 4;

    localparam logic [31:0] SHAPE_MN = 32'h00, SHAPE_K = 32'h04, ADDR_A = 32'h08,
        ADDR_B = 32'h0C, ADDR_C = 32'h10, DOORBELL = 32'h14, STATUS = 32'h18,
        FENCE = 32'h1C, DONE = 32'h20, DATA_BASE = 32'h24, PROG_LEN = 32'h28;
    localparam logic [1:0] OKAY = 2'b00, SLVERR = 2'b10;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic running = 1'b0;
    logic [31:0] data_base = 32'h1234_5000;
    logic [31:0] prog_len = 32'd77;

    logic [IDW-1:0] s_awid = '0;
    logic [31:0] s_awaddr = '0;
    logic [7:0]  s_awlen = '0;
    logic        s_awvalid = 1'b0;
    wire         s_awready;
    logic [63:0] s_wdata = '0;
    logic [7:0]  s_wstrb = '0;
    logic        s_wlast = 1'b0;
    logic        s_wvalid = 1'b0;
    wire         s_wready;
    wire [IDW-1:0] s_bid;
    wire [1:0]   s_bresp;
    wire         s_bvalid;
    logic        s_bready = 1'b0;
    logic [IDW-1:0] s_arid = '0;
    logic [31:0] s_araddr = '0;
    logic [7:0]  s_arlen = '0;
    logic        s_arvalid = 1'b0;
    wire         s_arready;
    wire [IDW-1:0] s_rid;
    wire [63:0]  s_rdata;
    wire [1:0]   s_rresp;
    wire         s_rlast;
    wire         s_rvalid;
    logic        s_rready = 1'b0;

    wire         ctl_start;
    wire [15:0]  ctl_m, ctl_n, ctl_k;
    logic        ctl_accept = 1'b1;
    logic        ctl_error = 1'b0;
    logic [7:0]  ctl_error_code = 8'd0;

    wire         ld_valid;
    logic        ld_ready = 1'b1;
    wire [31:0]  ld_a_addr, ld_b_addr;
    wire [12:0]  ld_a_bytes, ld_b_bytes;
    wire         st_valid;
    logic        st_ready = 1'b1;
    wire [31:0]  st_c_addr;
    wire [8:0]   st_words;
    logic        jobs_written = 1'b0;
    logic        read_error = 1'b0;
    logic        write_error = 1'b0;

    wire         done;
    wire [7:0]   done_code;
    wire         error;
    wire [7:0]   error_code;
    wire         busy;
    wire         drained;
    wire [31:0]  jobs_issued;
    wire [31:0]  jobs_done;

    integer checks = 0;
    integer cycle = 0;

    always #5 clk = ~clk;
    always @(posedge clk) cycle <= cycle + 1;

    cva6_npu_mmio #(.ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K), .ID_WIDTH(IDW)) dut (.*);

    task automatic check(input logic condition, input string message);
        begin
            checks = checks + 1;
            if (condition !== 1'b1) begin
                $display("FAIL tb_cva6_npu_mmio: %s (t=%0t)", message, $time);
                $fatal(1);
            end
        end
    endtask

    task automatic tick;
        begin
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    // ------------------------------------------------------------ monitors
    // Every issue (DMA hand-off) and every ctl_start, with what they carried.
    integer      issues = 0, starts = 0, dones = 0;
    integer      issue_cycle, start_cycle;
    logic [31:0] seen_a, seen_b, seen_c;
    logic [12:0] seen_a_bytes, seen_b_bytes;
    logic [8:0]  seen_c_words;
    logic [15:0] seen_m, seen_n, seen_k;
    logic [7:0]  seen_done_code;
    integer      issue_log [0:255];

    always @(posedge clk) begin
        if (rst_n && ld_valid) begin
            if (!st_valid || !ld_ready || !st_ready || !ctl_accept) begin
                $display("FAIL tb_cva6_npu_mmio: issue without ready (t=%0t)", $time);
                $fatal(1);
            end
            issues <= issues + 1;
            issue_cycle <= cycle;
            issue_log[issues] <= cycle;
            seen_a <= ld_a_addr; seen_b <= ld_b_addr; seen_c <= st_c_addr;
            seen_a_bytes <= ld_a_bytes; seen_b_bytes <= ld_b_bytes;
            seen_c_words <= st_words;
        end
        if (rst_n && ctl_start) begin
            starts <= starts + 1;
            start_cycle <= cycle;
            seen_m <= ctl_m; seen_n <= ctl_n; seen_k <= ctl_k;
        end
        if (rst_n && done) begin
            dones <= dones + 1;
            seen_done_code <= done_code;
        end
    end

    // ------------------------------------------------------- AXI master BFM
    // Presents AW and W together, returns the B response. The cycles from the
    // last W handshake to BVALID are reported so tests can check holding.
    task automatic axi_write_raw(input logic [31:0] addr, input logic [7:0] len,
                                 input logic [31:0] data, input logic [3:0] strb,
                                 input logic [IDW-1:0] id, output logic [1:0] resp,
                                 output integer wait_cycles);
        integer beats, guard;
        begin
            s_awid = id; s_awaddr = addr; s_awlen = len; s_awvalid = 1'b1;
            while (!s_awready) tick();
            tick();
            s_awvalid = 1'b0;
            for (beats = 0; beats <= len; beats = beats + 1) begin
                s_wdata = addr[2] ? {data, 32'hDEAD_BEEF} : {32'hDEAD_BEEF, data};
                s_wstrb = addr[2] ? {strb, 4'h0} : {4'h0, strb};
                s_wlast = (beats == len);
                s_wvalid = 1'b1;
                #1;
                while (!s_wready) begin tick(); #1; end
                tick();
            end
            s_wvalid = 1'b0;
            s_wlast = 1'b0;
            s_bready = 1'b1;
            wait_cycles = 0;
            guard = 0;
            #1;
            while (!s_bvalid) begin
                tick(); #1;
                wait_cycles = wait_cycles + 1;
                guard = guard + 1;
                if (guard > 2000) begin
                    $display("FAIL tb_cva6_npu_mmio: no B response (t=%0t)", $time);
                    $fatal(1);
                end
            end
            check(s_bid === id, "BID echoes AWID");
            resp = s_bresp;
            tick();
            s_bready = 1'b0;
        end
    endtask

    task automatic axi_write(input logic [31:0] addr, input logic [31:0] data,
                             input logic [1:0] expect_resp);
        logic [1:0] resp;
        integer w;
        begin
            axi_write_raw(addr, 8'd0, data, 4'hF, 4'd3, resp, w);
            check(resp === expect_resp, $sformatf("write 0x%02x response", addr));
        end
    endtask

    // Starts a read and waits for each beat; returns the first beat's data and
    // response and the cycles RVALID took to appear for the first beat.
    task automatic axi_read_raw(input logic [31:0] addr, input logic [7:0] len,
                                input logic [IDW-1:0] id, output logic [31:0] data,
                                output logic [1:0] resp, output integer wait_cycles);
        integer beats, guard;
        begin
            s_arid = id; s_araddr = addr; s_arlen = len; s_arvalid = 1'b1;
            while (!s_arready) tick();
            tick();
            s_arvalid = 1'b0;
            s_rready = 1'b1;
            wait_cycles = 0;
            for (beats = 0; beats <= len; beats = beats + 1) begin
                guard = 0;
                #1;
                while (!s_rvalid) begin
                    tick(); #1;
                    if (beats == 0) wait_cycles = wait_cycles + 1;
                    guard = guard + 1;
                    if (guard > 2000) begin
                        $display("FAIL tb_cva6_npu_mmio: no R beat (t=%0t)", $time);
                        $fatal(1);
                    end
                end
                check(s_rid === id, "RID echoes ARID");
                check(s_rlast === (beats == len), "RLAST on the final beat only");
                if (beats == 0) begin
                    data = addr[2] ? s_rdata[63:32] : s_rdata[31:0];
                    resp = s_rresp;
                end else begin
                    check(s_rresp === resp, "every burst beat has the same response");
                end
                tick();
            end
            s_rready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [31:0] addr, output logic [31:0] data,
                            input logic [1:0] expect_resp);
        logic [1:0] resp;
        integer w;
        begin
            axi_read_raw(addr, 8'd0, 4'd5, data, resp, w);
            check(resp === expect_resp, $sformatf("read 0x%02x response", addr));
        end
    endtask

    task automatic pulse_written;
        begin
            jobs_written = 1'b1;
            tick();
            jobs_written = 1'b0;
        end
    endtask

    task automatic do_start;
        begin
            start = 1'b1;
            tick();
            start = 1'b0;
        end
    endtask

    task automatic set_job(input [15:0] m, input [15:0] n, input [15:0] k,
                           input [31:0] a, input [31:0] b, input [31:0] c);
        begin
            axi_write(SHAPE_MN, {n, m}, OKAY);
            axi_write(SHAPE_K, {16'd0, k}, OKAY);
            axi_write(ADDR_A, a, OKAY);
            axi_write(ADDR_B, b, OKAY);
            axi_write(ADDR_C, c, OKAY);
        end
    endtask

    logic [31:0] rd;
    logic [1:0]  resp;
    integer      waited, base_issues;

    initial begin
        repeat (3) tick();
        rst_n = 1'b1;
        tick();
        do_start();
        running = 1'b1;

        // --------------------------------------------- register read back
        axi_write(SHAPE_MN, 32'h0007_0005, OKAY);
        axi_read(SHAPE_MN, rd, OKAY);
        check(rd === 32'h0007_0005, "SHAPE_MN reads back");
        axi_write(SHAPE_K, 32'hFFFF_0123, OKAY);
        axi_read(SHAPE_K, rd, OKAY);
        check(rd === 32'h0000_0123, "SHAPE_K keeps only K[15:0]");
        axi_write(ADDR_A, 32'h0010_0000, OKAY);
        axi_write(ADDR_B, 32'h0020_0008, OKAY);
        axi_write(ADDR_C, 32'h0030_0010, OKAY);
        axi_read(ADDR_A, rd, OKAY);
        check(rd === 32'h0010_0000, "ADDR_A reads back (low lane)");
        axi_read(ADDR_B, rd, OKAY);
        check(rd === 32'h0020_0008, "ADDR_B reads back (high lane)");
        axi_read(ADDR_C, rd, OKAY);
        check(rd === 32'h0030_0010, "ADDR_C reads back");
        begin : byte_strobe
            logic [1:0] r;
            integer w;
            axi_write_raw(ADDR_C, 8'd0, 32'hAABB_CC44, 4'b0001, 4'd1, r, w);
            check(r === OKAY, "byte write OKAY");
        end
        axi_read(ADDR_C, rd, OKAY);
        check(rd === 32'h0030_0044, "byte strobe writes one byte only");
        axi_read(DATA_BASE, rd, OKAY);
        check(rd === 32'h1234_5000, "DATA_BASE reflects the CSR");
        axi_read(PROG_LEN, rd, OKAY);
        check(rd === 32'd77, "PROG_LEN reflects the CSR");
        axi_read(STATUS, rd, OKAY);
        check(rd === 32'd0, "STATUS idle");
        check(drained === 1'b1 && busy === 1'b0, "idle is drained");

        // ------------------------------------------------- one GEMM job
        set_job(16'd5, 16'd7, 16'd300 - 16'd44, 32'h0010_0000, 32'h0020_0008, 32'h0030_0010);
        axi_write(DOORBELL, 32'd1, OKAY);
        repeat (2) tick();
        check(issues == 1 && starts == 1, "DOORBELL issues one job");
        check(start_cycle == issue_cycle + 1, "ctl_start is registered one cycle after the DMA hand-off");
        check(seen_m == 16'd5 && seen_n == 16'd7 && seen_k == 16'd256, "ctl_m/n/k carry the shape");
        check(ctl_m == 16'd5 && ctl_n == 16'd7 && ctl_k == 16'd256, "shape held after ctl_start");
        check(seen_a == 32'h0010_0000 && seen_b == 32'h0020_0008 && seen_c == 32'h0030_0010,
              "DMA gets the A/B/C addresses");
        check(seen_a_bytes == 13'd1280 && seen_b_bytes == 13'd1792 && seen_c_words == 9'd35,
              "DMA gets m*k, k*n bytes and m*n words");
        check(jobs_issued == 32'd1 && busy === 1'b1 && drained === 1'b0, "one job outstanding");
        axi_read(STATUS, rd, OKAY);
        check(rd[0] === 1'b1 && rd[23:16] === 8'd1 && rd[1] === 1'b0, "STATUS busy, one outstanding");

        // ------------------------------------------- FENCE waits for drain
        fork
            axi_read_raw(FENCE, 8'd0, 4'd2, rd, resp, waited);
            begin
                repeat (20) tick();
                check(s_rvalid === 1'b0, "FENCE not answered while a job is outstanding");
                pulse_written();
            end
        join
        check(resp === OKAY && waited >= 20, "FENCE answered only after the write-back");
        check(rd[0] === 1'b0 && rd[23:16] === 8'd0, "FENCE returns an idle STATUS");
        check(jobs_done == 32'd1 && drained === 1'b1, "job written back");

        // ------------------------------- B held while the job queue is full
        ctl_accept = 1'b0;
        set_job(16'd16, 16'd16, 16'd1, 32'h100, 32'h200, 32'h300);
        base_issues = issues;
        begin : held
            logic [1:0] r;
            integer w;
            // First job goes to the pending entry: answered at once.
            axi_write_raw(DOORBELL, 8'd0, 32'd1, 4'hF, 4'd7, r, w);
            check(r === OKAY && w <= 1, "first DOORBELL answered while the entry is free");
            check(issues == base_issues, "no issue while ctl_accept is low");
            // Second job: the entry is occupied, so BVALID is held.
            fork
                axi_write_raw(DOORBELL, 8'd0, 32'd1, 4'hF, 4'd7, r, w);
                begin
                    repeat (15) tick();
                    check(s_bvalid === 1'b0, "B held while the job cannot be accepted");
                    axi_read(STATUS, rd, OKAY);
                    check(rd[23:16] === 8'd1, "STATUS shows the pending job");
                    ctl_accept = 1'b1;
                end
            join
            check(r === OKAY && w >= 15, "second DOORBELL answered after the queue frees");
        end
        repeat (6) tick();
        check(issues == base_issues + 2 && starts == base_issues + 2, "both held jobs issue");
        check(start_cycle - issue_cycle == 1, "registered START for the second job too");
        pulse_written();
        pulse_written();
        check(drained === 1'b1, "drained after two write-backs");

        // ------------------------------------------ cooldown between issues
        begin : cooldown_check
            base_issues = issues;
            ctl_accept = 1'b0;
            axi_write(DOORBELL, 32'd1, OKAY);
            fork
                axi_write(DOORBELL, 32'd1, OKAY);  // latched when the first issues
                begin
                    repeat (4) tick();
                    ctl_accept = 1'b1;
                    #1;
                    check(ld_valid === 1'b1, "pending job issues as soon as ctl_accept rises");
                end
            join
            repeat (6) tick();
            check(issues == base_issues + 2, "both jobs issue");
            check(issue_log[base_issues + 1] - issue_log[base_issues] == 3,
                  "next issue waits out the two-cycle cooldown");
            pulse_written();
            pulse_written();
        end

        // ---------------------------------------------- DMA back-pressure
        ld_ready = 1'b0;
        base_issues = issues;
        axi_write(DOORBELL, 32'd1, OKAY);
        repeat (5) tick();
        check(issues == base_issues, "no issue while the DMA load side is busy");
        ld_ready = 1'b1;
        st_ready = 1'b0;
        repeat (5) tick();
        check(issues == base_issues, "no issue while the DMA store side is busy");
        st_ready = 1'b1;
        repeat (2) tick();
        check(issues == base_issues + 1, "issues once the DMA is ready");
        pulse_written();

        // ------------------------------------------------ SLVERR responses
        axi_write(32'h40, 32'd0, SLVERR);
        axi_read(32'h40, rd, SLVERR);
        axi_write(STATUS, 32'd0, SLVERR);
        axi_write(FENCE, 32'd0, SLVERR);
        begin : bursts
            logic [1:0] r;
            integer w;
            axi_read_raw(SHAPE_MN, 8'd3, 4'd9, rd, r, w);
            check(r === SLVERR, "read burst answered SLVERR on every beat");
            axi_write_raw(SHAPE_MN, 8'd2, 32'hFFFF_FFFF, 4'hF, 4'd9, r, w);
            check(r === SLVERR, "write burst answered SLVERR");
        end
        axi_read(SHAPE_MN, rd, OKAY);
        check(rd === 32'h0010_0010, "a write burst changes no register");
        check(error === 1'b0, "SLVERR accesses latch no fault");

        // ---------------------------------------------------------- DONE
        axi_write(DONE, 32'h0000_005A, OKAY);
        tick();
        check(dones == 1 && seen_done_code == 8'h5A, "DONE pulses once with its code");

        // ------------------------------------------------------ bad shape
        base_issues = issues;
        axi_write(SHAPE_MN, {16'd7, 16'd17}, OKAY);  // M = 17 > ROWS
        axi_write(DOORBELL, 32'd1, OKAY);
        repeat (3) tick();
        check(issues == base_issues, "bad shape does not issue");
        check(error === 1'b1 && error_code === 8'h11, "bad shape latches ERR_BAD_SHAPE");
        axi_read(STATUS, rd, OKAY);
        check(rd[1] === 1'b1 && rd[15:8] === 8'h11, "STATUS reports the latched fault");
        axi_write(SHAPE_MN, {16'd7, 16'd5}, OKAY);
        axi_write(DOORBELL, 32'd1, OKAY);
        repeat (3) tick();
        check(issues == base_issues, "a latched fault drops later DOORBELLs");
        check(error_code === 8'h11, "first fault is the one reported");
        axi_read_raw(FENCE, 8'd0, 4'd2, rd, resp, waited);
        check(resp === OKAY, "FENCE answers with a fault latched");

        do_start();
        check(error === 1'b0 && jobs_issued == 32'd0 && jobs_done == 32'd0,
              "start clears the fault and the per-run counters");

        axi_write(SHAPE_K, 32'd0, OKAY);                 // K = 0
        axi_write(DOORBELL, 32'd1, OKAY);
        tick();
        check(error === 1'b1 && error_code === 8'h11, "K = 0 is ERR_BAD_SHAPE");
        do_start();
        axi_write(SHAPE_K, 32'd257, OKAY);               // K > MAX_K
        axi_write(DOORBELL, 32'd1, OKAY);
        tick();
        check(error === 1'b1 && error_code === 8'h11, "K > MAX_K is ERR_BAD_SHAPE");
        do_start();
        axi_write(SHAPE_K, 32'd16, OKAY);
        axi_write(ADDR_B, 32'h0000_0204, OKAY);          // not 8-byte aligned
        axi_write(DOORBELL, 32'd1, OKAY);
        tick();
        check(error === 1'b1 && error_code === 8'h12, "misaligned address is ERR_MISALIGNED");
        do_start();
        axi_write(ADDR_B, 32'h0000_0200, OKAY);

        // ---------------------------------------------- matrix core fault
        base_issues = issues;
        axi_write(DOORBELL, 32'd1, OKAY);
        repeat (2) tick();
        check(issues == base_issues + 1, "job issues before the fault");
        ctl_error = 1'b1;
        ctl_error_code = 8'h04;
        tick();
        ctl_error = 1'b0;
        check(error === 1'b1 && error_code === 8'h04, "ctl_error latches the matrix core code");
        axi_write(DOORBELL, 32'd1, OKAY);
        repeat (3) tick();
        check(issues == base_issues + 1, "no issue after a matrix core fault");
        pulse_written();

        // ------------------------------------------------- DMA faults
        do_start();
        read_error = 1'b1;
        tick();
        read_error = 1'b0;
        check(error_code === 8'h13, "DMA read fault is ERR_AXI_READ");
        do_start();
        write_error = 1'b1;
        tick();
        write_error = 1'b0;
        check(error_code === 8'h14, "DMA write fault is ERR_AXI_WRITE");
        do_start();
        running = 1'b0;
        write_error = 1'b1;
        tick();
        write_error = 1'b0;
        check(error === 1'b0, "faults are latched only while the program runs");

        $display("PASS tb_cva6_npu_mmio: %0d checks", checks);
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL tb_cva6_npu_mmio: timeout");
        $fatal(1);
    end
endmodule
