`timescale 1ns/1ps

module tb_csr;
    localparam integer ADDR_WIDTH = 8;
    localparam integer DATA_WIDTH = 32;

    logic                     clk = 1'b0;
    logic                     aresetn = 1'b0;
    logic [ADDR_WIDTH-1:0]    awaddr = '0;
    logic [2:0]               awprot = '0;
    logic                     awvalid = 1'b0;
    wire                      awready;
    logic [DATA_WIDTH-1:0]    wdata = '0;
    logic [DATA_WIDTH/8-1:0]  wstrb = '0;
    logic                     wvalid = 1'b0;
    wire                      wready;
    wire [1:0]                bresp;
    wire                      bvalid;
    logic                     bready = 1'b0;
    logic [ADDR_WIDTH-1:0]    araddr = '0;
    logic [2:0]               arprot = '0;
    logic                     arvalid = 1'b0;
    wire                      arready;
    wire [DATA_WIDTH-1:0]     rdata;
    wire [1:0]                rresp;
    wire                      rvalid;
    logic                     rready = 1'b0;

    logic                     status_busy = 1'b0;
    logic                     status_accept = 1'b1;
    logic                     status_done = 1'b0;
    logic                     status_error = 1'b0;
    logic [7:0]               error_code = 8'd0;
    logic [63:0]              cycles = 64'd0;
    wire                      soft_reset_pulse;
    logic                     isa_running = 1'b0;
    logic                     isa_busy = 1'b0;
    logic                     isa_done = 1'b0;
    logic                     isa_error = 1'b0;
    logic [7:0]               isa_error_code = 8'd0;
    logic [31:0]              isa_pc = 32'd0;
    logic [31:0]              isa_jobs = 32'd0;
    logic [31:0]              isa_cycles = 32'd0;
    logic [31:0]              isa_instructions = 32'd0;
    wire                      isa_start_pulse;
    wire [31:0]               isa_prog_addr;
    wire [31:0]               isa_prog_len;
    wire [31:0]               isa_data_base;
    integer isa_start_count = 0;

    integer reset_count = 0;

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (soft_reset_pulse)
            reset_count <= reset_count + 1;
        if (isa_start_pulse)
            isa_start_count <= isa_start_count + 1;
    end

    csr #(
        .C_S_AXI_ADDR_WIDTH(ADDR_WIDTH),
        .C_S_AXI_DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .s_axi_aclk(clk),
        .s_axi_aresetn(aresetn),
        .s_axi_awaddr(awaddr),
        .s_axi_awprot(awprot),
        .s_axi_awvalid(awvalid),
        .s_axi_awready(awready),
        .s_axi_wdata(wdata),
        .s_axi_wstrb(wstrb),
        .s_axi_wvalid(wvalid),
        .s_axi_wready(wready),
        .s_axi_bresp(bresp),
        .s_axi_bvalid(bvalid),
        .s_axi_bready(bready),
        .s_axi_araddr(araddr),
        .s_axi_arprot(arprot),
        .s_axi_arvalid(arvalid),
        .s_axi_arready(arready),
        .s_axi_rdata(rdata),
        .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid),
        .s_axi_rready(rready),
        .status_busy(status_busy),
        .status_accept(status_accept),
        .status_done(status_done),
        .status_error(status_error),
        .error_code(error_code),
        .cycles(cycles),
        .soft_reset_pulse(soft_reset_pulse),
        .isa_running(isa_running),
        .isa_busy(isa_busy),
        .isa_done(isa_done),
        .isa_error(isa_error),
        .isa_error_code(isa_error_code),
        .isa_pc(isa_pc),
        .isa_jobs(isa_jobs),
        .isa_cycles(isa_cycles),
        .isa_instructions(isa_instructions),
        .isa_start_pulse(isa_start_pulse),
        .isa_prog_addr(isa_prog_addr),
        .isa_prog_len(isa_prog_len),
        .isa_data_base(isa_data_base)
    );

    task automatic expect_word;
        input [31:0] actual;
        input [31:0] expected;
        input [8*64-1:0] label;
        begin
            if (actual !== expected) begin
                $display("MISMATCH %0s expected=%08x actual=%08x", label, expected, actual);
                $fatal(1);
            end
        end
    endtask

    task automatic axi_write;
        input [ADDR_WIDTH-1:0] address;
        input [31:0] value;
        input [3:0] strobes;
        input integer aw_delay;
        input integer w_delay;
        input integer response_stall;
        integer i;
        reg [1:0] held_resp;
        begin
            fork
                begin
                    repeat (aw_delay) @(negedge clk);
                    awaddr = address;
                    awvalid = 1'b1;
                    do @(posedge clk); while (!awready);
                    @(negedge clk);
                    awvalid = 1'b0;
                end
                begin
                    repeat (w_delay) @(negedge clk);
                    wdata = value;
                    wstrb = strobes;
                    wvalid = 1'b1;
                    do @(posedge clk); while (!wready);
                    @(negedge clk);
                    wvalid = 1'b0;
                end
            join
            while (!bvalid) @(posedge clk);
            held_resp = bresp;
            for (i = 0; i < response_stall; i = i + 1) begin
                @(posedge clk);
                if (!bvalid || bresp !== held_resp) begin
                    $display("MISMATCH write response changed under backpressure");
                    $fatal(1);
                end
            end
            if (bresp !== 2'b00) begin
                $display("MISMATCH expected AXI OKAY write response");
                $fatal(1);
            end
            bready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    task automatic axi_read;
        input [ADDR_WIDTH-1:0] address;
        input integer response_stall;
        output [31:0] value;
        integer i;
        reg [31:0] held_data;
        begin
            araddr = address;
            arvalid = 1'b1;
            do @(posedge clk); while (!arready);
            @(negedge clk);
            arvalid = 1'b0;
            while (!rvalid) @(posedge clk);
            held_data = rdata;
            for (i = 0; i < response_stall; i = i + 1) begin
                @(posedge clk);
                if (!rvalid || rdata !== held_data || rresp !== 2'b00) begin
                    $display("MISMATCH read response changed under backpressure");
                    $fatal(1);
                end
            end
            value = held_data;
            rready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            rready = 1'b0;
        end
    endtask

    reg [31:0] read_value;
    integer prior_start_count;
    integer prior_reset_count;
    integer offset;

    initial begin
        repeat (4) @(posedge clk);
        @(negedge clk);
        aresetn = 1'b1;

        axi_read(8'h00, 2, read_value);
        expect_word(read_value, 32'h3155504e, "MAGIC");
        axi_read(8'h04, 0, read_value);
        expect_word(read_value, 32'h00020000, "VERSION 2.0");
        axi_read(8'h08, 0, read_value);
        expect_word(read_value, 32'h0000007b, "CAPABILITIES");
        axi_read(8'h3c, 0, read_value);
        expect_word(read_value, 32'h00000000, "RESERVED");

        // ABI 2.0 retired the direct-job registers: 0x18-0x30 (M, N, K,
        // A/B/C_STRIDE, TIMEOUT_CYCLES) are reserved, ignore writes and read
        // as zero.
        for (offset = 8'h18; offset <= 8'h30; offset = offset + 4) begin
            axi_write(offset[7:0], 32'hffffffff, 4'b1111, offset % 3, (offset / 4) % 2, 0);
            axi_read(offset[7:0], 0, read_value);
            expect_word(read_value, 32'h00000000, "retired job register reads zero");
        end

        status_busy = 1'b1;
        status_done = 1'b1;
        status_error = 1'b1;
        error_code = 8'd3;
        cycles = 64'h1122334455667788;
        axi_read(8'h10, 2, read_value);
        expect_word(read_value, 32'h0000000f, "STATUS with ACCEPT");
        axi_read(8'h14, 0, read_value);
        expect_word(read_value, 32'h00000003, "ERROR");
        axi_read(8'h34, 0, read_value);
        expect_word(read_value, 32'h55667788, "CYCLES_LO");
        axi_read(8'h38, 0, read_value);
        expect_word(read_value, 32'h11223344, "CYCLES_HI");

        status_accept = 1'b0;
        axi_read(8'h10, 0, read_value);
        expect_word(read_value, 32'h00000007, "STATUS without ACCEPT");
        status_accept = 1'b1;

        // CONTROL.START (bit 0) was retired in ABI 2.0: the csr has no matrix
        // command output, and the write starts nothing, neither a soft reset
        // nor a program.
        prior_start_count = isa_start_count;
        prior_reset_count = reset_count;
        axi_write(8'h0c, 32'h00000001, 4'b0001, 2, 0, 0);
        repeat (2) @(posedge clk);
        if (isa_start_count !== prior_start_count || reset_count !== prior_reset_count) begin
            $display("MISMATCH retired CONTROL.START had an effect");
            $fatal(1);
        end
        axi_read(8'h0c, 0, read_value);
        expect_word(read_value, 32'h00000000, "CONTROL reads zero");

        status_busy = 1'b0;
        status_done = 1'b0;
        status_error = 1'b0;
        error_code = 8'd0;
        prior_reset_count = reset_count;
        axi_write(8'h0c, 32'h00000002, 4'b0001, 0, 2, 0);
        @(posedge clk);
        if (reset_count !== prior_reset_count + 1) begin
            $display("MISMATCH SOFT_RESET pulse count");
            $fatal(1);
        end
        prior_reset_count = reset_count;
        axi_write(8'h0c, 32'h00000002, 4'b0010, 0, 0, 0);
        @(posedge clk);
        if (reset_count !== prior_reset_count) begin
            $display("MISMATCH SOFT_RESET without byte-0 strobe");
            $fatal(1);
        end

        axi_write(8'h3c, 32'hffffffff, 4'b1111, 0, 0, 0);
        axi_read(8'h3c, 0, read_value);
        expect_word(read_value, 32'h00000000, "reserved remains zero");

        // The instruction-stream front end's registers.
        axi_write(8'h48, 32'h1f000040, 4'b1111, 0, 0, 0);
        axi_write(8'h4c, 32'd1234, 4'b1111, 1, 0, 0);
        axi_write(8'h50, 32'h1e000000, 4'b1111, 0, 1, 0);
        axi_write(8'h50, 32'h00000100, 4'b0010, 0, 0, 0);
        axi_read(8'h48, 0, read_value);
        expect_word(read_value, 32'h1f000040, "PROG_ADDR readback");
        axi_read(8'h4c, 0, read_value);
        expect_word(read_value, 32'd1234, "PROG_LEN readback");
        axi_read(8'h50, 0, read_value);
        expect_word(read_value, 32'h1e000100, "DATA_BASE WSTRB merge");
        expect_word(isa_prog_addr, 32'h1f000040, "PROG_ADDR output");
        prior_start_count = isa_start_count;
        axi_write(8'h40, 32'h00000001, 4'b0001, 0, 0, 0);
        @(posedge clk);
        if (isa_start_count !== prior_start_count + 1) begin
            $display("MISMATCH ISA START pulse count");
            $fatal(1);
        end
        isa_running = 1'b1; isa_busy = 1'b1; isa_error = 1'b1; isa_error_code = 8'h12;
        isa_pc = 32'd77; isa_jobs = 32'd5; isa_cycles = 32'd999; isa_instructions = 32'd80;
        axi_read(8'h44, 0, read_value);
        expect_word(read_value, 32'h0000000d, "ISA_STATUS");
        axi_read(8'h54, 0, read_value);
        expect_word(read_value, 32'h00000012, "ISA_ERROR");
        axi_read(8'h58, 0, read_value);
        expect_word(read_value, 32'd77, "ISA_PC");
        axi_read(8'h5c, 0, read_value);
        expect_word(read_value, 32'd5, "ISA_JOBS");
        axi_read(8'h60, 0, read_value);
        expect_word(read_value, 32'd999, "ISA_CYCLES");
        axi_read(8'h64, 0, read_value);
        expect_word(read_value, 32'd80, "ISA_INSTRUCTIONS");
        axi_read(8'h68, 0, read_value);
        expect_word(read_value, 32'd1, "ISA_VERSION");

        $display("PASS tb_csr");
        $finish;
    end
endmodule
