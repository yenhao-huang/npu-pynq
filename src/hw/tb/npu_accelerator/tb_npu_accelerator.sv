`timescale 1ns/1ps

// Runs a hand-assembled two-job program on the 2 x 2 npu_accelerator through
// its public interfaces only: AXI-Lite registers and the AXI4 master. The
// memory model inserts pseudo-random ready/valid gaps, so both skid buffers
// between the front end and the controller see back-pressure. The program
// runs twice to show the accelerator is reusable after DONE.
module tb_npu_accelerator;
    localparam integer MEM_WORDS = 256;
    localparam integer QDEPTH = 16;
    localparam integer PROG_WORDS = 11;
    localparam integer DATA_BASE = 32'h400;        // byte address of the tiles
    localparam integer DATA_WORD = DATA_BASE / 8;

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
    logic irq;

    logic [31:0] m_axi_araddr, m_axi_awaddr;
    logic [7:0] m_axi_arlen, m_axi_awlen, m_axi_wstrb;
    logic [2:0] m_axi_arsize, m_axi_awsize, m_axi_arprot, m_axi_awprot;
    logic [1:0] m_axi_arburst, m_axi_awburst;
    logic [3:0] m_axi_arcache, m_axi_awcache;
    logic m_axi_arvalid, m_axi_arready, m_axi_rlast, m_axi_rvalid, m_axi_rready;
    logic m_axi_awvalid, m_axi_awready, m_axi_wlast, m_axi_wvalid, m_axi_wready;
    logic m_axi_bvalid, m_axi_bready;
    logic [63:0] m_axi_rdata, m_axi_wdata;
    logic [1:0] m_axi_rresp, m_axi_bresp;

    npu_accelerator #(
        .ROWS(2), .COLUMNS(2), .MAX_K(256), .IN_BYTES(8)
    ) dut (.*);
    always #5 s_axi_aclk = ~s_axi_aclk;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_accelerator: %s", message);
            $fatal(1);
        end
    endtask

    // ------------------------------------------------- AXI memory model
    logic [63:0] mem [0:MEM_WORDS-1];
    logic [31:0] lfsr = 32'h1;
    always @(posedge s_axi_aclk) lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

    logic [31:0] ar_q_addr [0:QDEPTH-1];
    logic [7:0] ar_q_len [0:QDEPTH-1];
    logic [31:0] aw_q_addr [0:QDEPTH-1];
    logic [7:0] aw_q_len [0:QDEPTH-1];
    integer ar_head = 0, ar_tail = 0, r_beat = 0;
    integer aw_head = 0, aw_tail = 0, w_beat = 0, b_pending = 0;
    integer word_index, lane;

    assign m_axi_arready = lfsr[3] || lfsr[9];
    assign m_axi_awready = lfsr[5] || lfsr[11];
    assign m_axi_wready = lfsr[7] || lfsr[13];
    assign m_axi_rvalid = (ar_head != ar_tail) && (lfsr[4] || lfsr[12]);
    assign m_axi_rdata = mem[(ar_q_addr[ar_head % QDEPTH] >> 3) + r_beat];
    assign m_axi_rlast = (r_beat == ar_q_len[ar_head % QDEPTH]);
    assign m_axi_rresp = 2'b00;
    assign m_axi_bvalid = (b_pending > 0) && (lfsr[6] || lfsr[15]);
    assign m_axi_bresp = 2'b00;

    always @(posedge s_axi_aclk) begin
        if (m_axi_arvalid && m_axi_arready) begin
            if (m_axi_arsize != 3'd3 || m_axi_arburst != 2'b01) fail("read burst is not 8-byte INCR");
            if ((m_axi_araddr >> 3) + m_axi_arlen >= MEM_WORDS) fail("read outside the memory model");
            ar_q_addr[ar_tail % QDEPTH] <= m_axi_araddr;
            ar_q_len[ar_tail % QDEPTH] <= m_axi_arlen;
            ar_tail <= ar_tail + 1;
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
            if (m_axi_awsize != 3'd3 || m_axi_awburst != 2'b01) fail("write burst is not 8-byte INCR");
            if ((m_axi_awaddr >> 3) + m_axi_awlen >= MEM_WORDS) fail("write outside the memory model");
            aw_q_addr[aw_tail % QDEPTH] <= m_axi_awaddr;
            aw_q_len[aw_tail % QDEPTH] <= m_axi_awlen;
            aw_tail <= aw_tail + 1;
        end
        if (m_axi_wvalid && m_axi_wready) begin
            if (aw_head == aw_tail) fail("write data before its address");
            word_index = (aw_q_addr[aw_head % QDEPTH] >> 3) + w_beat;
            for (lane = 0; lane < 8; lane = lane + 1)
                if (m_axi_wstrb[lane]) mem[word_index][lane*8 +: 8] <= m_axi_wdata[lane*8 +: 8];
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
            if (s_axi_bresp != 0) fail("AXI write response");
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
            if (s_axi_rresp != 0) fail("AXI read response");
            value = s_axi_rdata;
            s_axi_rready = 1;
            @(posedge s_axi_aclk); #1;
            s_axi_rready = 0;
        end
    endtask

    // Job 1: 2x2x2, A = [[-128 127] [7 -3]], B = [[-1 2] [4 -5]]
    //        C = [[636 -891] [-19 29]]
    // Job 2: 1x2x2, A = [2 -3], B = [[4 6] [5 -7]], C = [-7 33]
    // Tile offsets from DATA_BASE: A1 0x00, B1 0x08, A2 0x10, B2 0x18,
    // C1 0x20 (two words), C2 0x30.
    task automatic load_memory;
        integer i;
        begin
            for (i = 0; i < MEM_WORDS; i = i + 1) mem[i] = 64'hdead_beef_dead_beef;
            mem[0]  = 64'h1000_0000_0202_0002;     // SHAPE m=2 n=2 k=2
            mem[1]  = 64'h1100_0000_0000_0000;     // ADDR_A 0x00
            mem[2]  = 64'h1200_0000_0000_0008;     // ADDR_B 0x08
            mem[3]  = 64'h1300_0000_0000_0020;     // ADDR_C 0x20
            mem[4]  = 64'h2000_0000_0000_0000;     // GEMM
            mem[5]  = 64'h1000_0000_0102_0002;     // SHAPE m=1 n=2 k=2
            mem[6]  = 64'h1100_0000_0000_0010;     // ADDR_A 0x10
            mem[7]  = 64'h1200_0000_0000_0018;     // ADDR_B 0x18
            mem[8]  = 64'h1300_0000_0000_0030;     // ADDR_C 0x30
            mem[9]  = 64'h2000_0000_0000_0000;     // GEMM
            mem[10] = 64'h0100_0000_0000_0000;     // END
            mem[DATA_WORD + 0] = 64'h0000_0000_fd07_7f80;   // A1
            mem[DATA_WORD + 1] = 64'h0000_0000_fb04_02ff;   // B1
            mem[DATA_WORD + 2] = 64'h0000_0000_0000_fd02;   // A2
            mem[DATA_WORD + 3] = 64'h0000_0000_f905_0604;   // B2
        end
    endtask

    task automatic expect_word(input integer index, input [63:0] expected);
        begin
            if (mem[index] !== expected) begin
                $display("word %0d: got %016x expected %016x", index, mem[index], expected);
                fail("memory mismatch");
            end
        end
    endtask

    logic [31:0] value;
    integer run, polls, i;

    initial begin
        repeat (4) @(posedge s_axi_aclk);
        @(negedge s_axi_aclk); s_axi_aresetn = 1;

        axi_read(8'h00, value);
        if (value != 32'h3155504e) fail("MAGIC through public AXI");
        axi_read(8'h04, value);
        if (value != 32'h00010001) fail("VERSION is not 1.1");
        axi_read(8'h08, value);
        if (value[6] != 1'b1) fail("CAPABILITIES lacks ISA_FRONTEND");
        axi_read(8'h68, value);
        if (value != 32'd1) fail("ISA_VERSION is not 1");
        if (irq !== 1'b0) fail("IRQ asserted before any program ran");

        for (run = 0; run < 2; run = run + 1) begin
            load_memory();
            axi_write(8'h48, 32'd0);           // PROG_ADDR
            axi_write(8'h4c, PROG_WORDS);      // PROG_LEN
            axi_write(8'h50, DATA_BASE);       // DATA_BASE
            axi_write(8'h40, 32'd1);           // ISA_CONTROL.START
            polls = 0;
            do begin
                axi_read(8'h44, value);
                polls = polls + 1;
                if (polls > 5000) fail("program did not finish");
            end while (value[3] || value[0]);  // BUSY or RUNNING

            if (value[1] !== 1'b1 || value[2] !== 1'b0) fail("ISA_STATUS is not DONE without ERROR");
            if (irq !== 1'b1) fail("IRQ not asserted after DONE");
            axi_read(8'h54, value);
            if (value != 0) fail("ISA_ERROR is set");
            axi_read(8'h5c, value);
            if (value != 2) fail("ISA_JOBS is not 2");
            axi_read(8'h64, value);
            if (value != PROG_WORDS) fail("ISA_INSTRUCTIONS is not the program length");
            axi_read(8'h60, value);
            if (value == 0) fail("ISA_CYCLES is zero");
            // The controller's job registers mirror the last job the front end issued.
            axi_read(8'h18, value);
            if (value != 1) fail("M does not mirror the last job");
            axi_read(8'h1c, value);
            if (value != 2) fail("N does not mirror the last job");
            axi_read(8'h20, value);
            if (value != 2) fail("K does not mirror the last job");
            axi_read(8'h14, value);
            if (value != 0) fail("controller ERROR is set");

            expect_word(DATA_WORD + 4, {32'hffff_fc85, 32'h0000_027c});   // -891, 636
            expect_word(DATA_WORD + 5, {32'h0000_001d, 32'hffff_ffed});   // 29, -19
            expect_word(DATA_WORD + 6, {32'h0000_0021, 32'hffff_fff9});   // 33, -7
            for (i = DATA_WORD + 7; i < MEM_WORDS; i = i + 1)
                expect_word(i, 64'hdead_beef_dead_beef);
            for (i = PROG_WORDS; i < DATA_WORD; i = i + 1)
                expect_word(i, 64'hdead_beef_dead_beef);
        end

        $display("PASS tb_npu_accelerator");
        $finish;
    end
endmodule
