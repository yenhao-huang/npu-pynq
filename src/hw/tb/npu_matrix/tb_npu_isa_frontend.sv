`timescale 1ns/1ps

// Runs encoder-generated NPU programs on the full 16 x 16 npu_accelerator
// against a behavioural AXI4 memory with pseudo-random ready/valid gaps, and
// compares every memory word with the golden ISA simulator's final image
// (src/test/vectors/generate_isa_vectors.py). Protocol checks: bursts of at
// most 16 beats that never cross a 128-byte line, 8-byte beats, WLAST on the
// last beat of every write burst.
module tb_npu_isa_frontend;
    localparam integer MEM_WORDS = 1 << 14;
    localparam integer QDEPTH = 64;

    logic s_axi_aclk = 0, s_axi_aresetn = 0;
    logic [7:0] s_axi_awaddr = 0, s_axi_araddr = 0;
    logic [2:0] s_axi_awprot = 0, s_axi_arprot = 0;
    logic s_axi_awvalid = 0, s_axi_wvalid = 0, s_axi_bready = 0;
    logic s_axi_arvalid = 0, s_axi_rready = 0;
    logic [31:0] s_axi_wdata = 0;
    logic [3:0] s_axi_wstrb = 0;
    logic s_axi_awready, s_axi_wready, s_axi_bvalid, s_axi_arready, s_axi_rvalid;
    logic [1:0] s_axi_bresp, s_axi_rresp;
    logic [31:0] s_axi_rdata;
    logic [63:0] s_axis_tdata = 0;
    logic [7:0] s_axis_tkeep = 0;
    logic s_axis_tvalid = 0, s_axis_tready, s_axis_tlast = 0;
    logic [31:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready = 0, m_axis_tlast;
    logic irq;

    logic [31:0] m_axi_araddr, m_axi_awaddr;
    logic [7:0] m_axi_arlen, m_axi_awlen;
    logic [2:0] m_axi_arsize, m_axi_awsize, m_axi_arprot, m_axi_awprot;
    logic [1:0] m_axi_arburst, m_axi_awburst;
    logic [3:0] m_axi_arcache, m_axi_awcache;
    logic m_axi_arvalid, m_axi_arready, m_axi_rlast, m_axi_rvalid, m_axi_rready;
    logic [63:0] m_axi_rdata, m_axi_wdata;
    logic [1:0] m_axi_rresp, m_axi_bresp;
    logic m_axi_awvalid, m_axi_awready, m_axi_wlast, m_axi_wvalid, m_axi_wready;
    logic [7:0] m_axi_wstrb;
    logic m_axi_bvalid, m_axi_bready;

    npu_accelerator #(
        .ROWS(16), .COLUMNS(16), .MAX_K(256), .IN_BYTES(8)
    ) dut (.*);
    always #5 s_axi_aclk = ~s_axi_aclk;

    logic [63:0] mem [0:MEM_WORDS-1];
    logic [63:0] expected [0:MEM_WORDS-1];

    // ------------------------------------------------- AXI memory model
    logic [31:0] lfsr = 32'h1;
    always @(posedge s_axi_aclk) lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

    logic [31:0] ar_q_addr [0:QDEPTH-1];
    logic [7:0] ar_q_len [0:QDEPTH-1];
    integer ar_head = 0, ar_tail = 0, r_beat = 0;
    logic [31:0] aw_q_addr [0:QDEPTH-1];
    logic [7:0] aw_q_len [0:QDEPTH-1];
    integer aw_head = 0, aw_tail = 0, w_beat = 0, b_pending = 0;
    integer reads = 0, writes = 0;
    integer index, lane;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_isa_frontend: %s", message);
            $fatal(1);
        end
    endtask

    task automatic check_burst(input [31:0] addr, input [7:0] len, input [2:0] size, input [1:0] burst);
        begin
            if (len > 8'd15) fail("burst longer than 16 beats");
            if (size != 3'd3 || burst != 2'b01) fail("burst is not 8-byte INCR");
            if (addr[2:0] != 3'd0) fail("unaligned burst address");
            if ({1'b0, addr[6:3]} + {1'b0, len[3:0]} > 5'd15) fail("burst crosses a 128-byte line");
            if ((addr >> 3) + len >= MEM_WORDS) fail("burst outside the memory model");
        end
    endtask

    assign m_axi_arready = lfsr[3] || lfsr[9];
    assign m_axi_awready = lfsr[5] || lfsr[11];
    assign m_axi_wready = lfsr[7] || lfsr[13] || lfsr[2];
    assign m_axi_rvalid = (ar_head != ar_tail) && (lfsr[4] || lfsr[12] || lfsr[17]);
    assign m_axi_rdata = mem[(ar_q_addr[ar_head % QDEPTH] >> 3) + r_beat];
    assign m_axi_rlast = (r_beat == ar_q_len[ar_head % QDEPTH]);
    assign m_axi_rresp = 2'b00;
    assign m_axi_bvalid = (b_pending > 0) && (lfsr[6] || lfsr[15]);
    assign m_axi_bresp = 2'b00;

    always @(posedge s_axi_aclk) begin
        if (m_axi_arvalid && m_axi_arready) begin
            check_burst(m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst);
            if (ar_tail - ar_head >= QDEPTH) fail("read queue overflow");
            ar_q_addr[ar_tail % QDEPTH] <= m_axi_araddr;
            ar_q_len[ar_tail % QDEPTH] <= m_axi_arlen;
            ar_tail <= ar_tail + 1;
            reads <= reads + 1;
        end
        if (m_axi_rvalid && m_axi_rready) begin
            if (m_axi_rlast) begin
                r_beat <= 0;
                ar_head <= ar_head + 1;
            end else begin
                r_beat <= r_beat + 1;
            end
        end
        if (m_axi_awvalid && m_axi_awready) begin
            check_burst(m_axi_awaddr, m_axi_awlen, m_axi_awsize, m_axi_awburst);
            aw_q_addr[aw_tail % QDEPTH] <= m_axi_awaddr;
            aw_q_len[aw_tail % QDEPTH] <= m_axi_awlen;
            aw_tail <= aw_tail + 1;
            writes <= writes + 1;
        end
        if (m_axi_wvalid && m_axi_wready) begin
            if (aw_head == aw_tail) fail("write data before its address");
            index = (aw_q_addr[aw_head % QDEPTH] >> 3) + w_beat;
            for (lane = 0; lane < 8; lane = lane + 1)
                if (m_axi_wstrb[lane]) mem[index][lane*8 +: 8] <= m_axi_wdata[lane*8 +: 8];
            if (m_axi_wlast != (w_beat == aw_q_len[aw_head % QDEPTH])) fail("WLAST misplaced");
            if (m_axi_wlast) begin
                w_beat <= 0;
                aw_head <= aw_head + 1;
            end else begin
                w_beat <= w_beat + 1;
            end
        end
        b_pending <= b_pending + ((m_axi_wvalid && m_axi_wready && m_axi_wlast) ? 1 : 0)
            - ((m_axi_bvalid && m_axi_bready) ? 1 : 0);
    end

    // ------------------------------------------------------ control path
    task automatic axi_write(input [7:0] address, input [31:0] value);
        begin
            @(negedge s_axi_aclk);
            s_axi_awaddr = address; s_axi_awvalid = 1;
            s_axi_wdata = value; s_axi_wstrb = 4'hf; s_axi_wvalid = 1;
            while (!(s_axi_awready && s_axi_wready)) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axi_awvalid = 0; s_axi_wvalid = 0;
            wait (s_axi_bvalid);
            s_axi_bready = 1;
            @(posedge s_axi_aclk); #1;
            s_axi_bready = 0;
        end
    endtask

    task automatic axi_read(input [7:0] address, output [31:0] value);
        begin
            @(negedge s_axi_aclk);
            s_axi_araddr = address; s_axi_arvalid = 1;
            while (!s_axi_arready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk); s_axi_arvalid = 0;
            wait (s_axi_rvalid);
            value = s_axi_rdata;
            s_axi_rready = 1;
            @(posedge s_axi_aclk); #1;
            s_axi_rready = 0;
        end
    endtask

    integer file, words, prog_len, data_base, want_error, want_jobs, i, cases, status_code;
    integer polls;
    string name;
    logic [31:0] value;
    logic [63:0] word;

    initial begin
        file = $fopen("vectors/isa_frontend_16x16.txt", "r");
        if (file == 0)
            file = $fopen("src/test/vectors/isa_frontend_16x16.txt", "r");
        if (file == 0) fail("cannot open isa_frontend_16x16.txt");
        status_code = $fgets(name, file);  // header comment

        repeat (5) @(posedge s_axi_aclk);
        s_axi_aresetn = 1;
        axi_read(8'h04, value);
        if (value != 32'h00010001) fail("VERSION is not 1.1");
        axi_read(8'h08, value);
        if (value[6] != 1'b1) fail("CAPABILITIES lacks ISA_FRONTEND");
        axi_read(8'h68, value);
        if (value != 32'd1) fail("ISA_VERSION is not 1");

        cases = 0;
        status_code = $fscanf(file, "%s %d %d %d %d %d %s\n", name, words, prog_len,
                              data_base, want_error, want_jobs, name);
        while (words != 0) begin
            for (i = 0; i < MEM_WORDS; i = i + 1) begin
                mem[i] = 64'd0;
                expected[i] = 64'd0;
            end
            for (i = 0; i < words; i = i + 1) begin
                status_code = $fscanf(file, "%h\n", word);
                mem[i] = word;
            end
            for (i = 0; i < words; i = i + 1) begin
                status_code = $fscanf(file, "%h\n", word);
                expected[i] = word;
            end

            axi_write(8'h48, 32'd0);           // PROG_ADDR
            axi_write(8'h4c, prog_len);        // PROG_LEN
            axi_write(8'h50, data_base);       // DATA_BASE
            axi_write(8'h40, 32'd1);           // ISA_CONTROL.START
            polls = 0;
            do begin
                axi_read(8'h44, value);
                polls = polls + 1;
                if (polls > 200000) fail({name, ": program did not finish"});
            end while (value[3] || value[0]); // BUSY or RUNNING

            axi_read(8'h54, value);
            if (want_error == 0) begin
                axi_read(8'h44, value);
                if (value[1] !== 1'b1 || value[2] !== 1'b0) fail({name, ": not DONE without error"});
            end else begin
                axi_read(8'h44, value);
                if (value[2] !== 1'b1) fail({name, ": expected an error"});
                axi_read(8'h54, value);
                if (value[7:0] != want_error[7:0]) begin
                    $display("ISA_ERROR=0x%02x expected 0x%02x", value[7:0], want_error[7:0]);
                    fail({name, ": wrong error code"});
                end
            end
            axi_read(8'h5c, value);
            if (value != want_jobs) begin
                $display("ISA_JOBS=%0d expected %0d", value, want_jobs);
                fail({name, ": wrong job count"});
            end
            for (i = 0; i < MEM_WORDS; i = i + 1)
                if (mem[i] !== expected[i]) begin
                    $display("word %0d (byte 0x%0x): got %016x expected %016x", i, i * 8, mem[i], expected[i]);
                    fail({name, ": memory mismatch"});
                end
            axi_read(8'h60, value);
            $display("case %0s: %0d jobs, %0d cycles, %0d reads, %0d writes", name, want_jobs, value, reads, writes);
            cases = cases + 1;
            status_code = $fscanf(file, "%s %d %d %d %d %d %s\n", name, words, prog_len,
                                  data_base, want_error, want_jobs, name);
        end
        if (cases != 7) fail("unexpected case count");
        $display("PASS tb_npu_isa_frontend: %0d programs", cases);
        $finish;
    end
endmodule
