`timescale 1ns/1ps

module tb_npu_axi_lite_regs;
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
    logic                     status_done = 1'b0;
    logic                     status_error = 1'b0;
    logic [7:0]               error_code = 8'd0;
    logic [63:0]              cycles = 64'd0;
    wire                      start_pulse;
    wire                      soft_reset_pulse;
    wire [15:0]               cfg_m;
    wire [15:0]               cfg_n;
    wire [15:0]               cfg_k;
    wire [31:0]               cfg_a_stride;
    wire [31:0]               cfg_b_stride;
    wire [31:0]               cfg_c_stride;
    wire [31:0]               cfg_timeout_cycles;

    integer start_count = 0;
    integer reset_count = 0;

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (start_pulse)
            start_count <= start_count + 1;
        if (soft_reset_pulse)
            reset_count <= reset_count + 1;
    end

    npu_axi_lite_regs #(
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
        .status_done(status_done),
        .status_error(status_error),
        .error_code(error_code),
        .cycles(cycles),
        .start_pulse(start_pulse),
        .soft_reset_pulse(soft_reset_pulse),
        .cfg_m(cfg_m),
        .cfg_n(cfg_n),
        .cfg_k(cfg_k),
        .cfg_a_stride(cfg_a_stride),
        .cfg_b_stride(cfg_b_stride),
        .cfg_c_stride(cfg_c_stride),
        .cfg_timeout_cycles(cfg_timeout_cycles)
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

    initial begin
        repeat (4) @(posedge clk);
        @(negedge clk);
        aresetn = 1'b1;

        axi_read(8'h00, 2, read_value);
        expect_word(read_value, 32'h3155504e, "MAGIC");
        axi_read(8'h04, 0, read_value);
        expect_word(read_value, 32'h00010000, "VERSION");
        axi_read(8'h08, 0, read_value);
        expect_word(read_value, 32'h0000001b, "CAPABILITIES");
        axi_read(8'h3c, 0, read_value);
        expect_word(read_value, 32'h00000000, "RESERVED");

        axi_write(8'h18, 32'h00000102, 4'b0011, 0, 3, 2);
        expect_word({16'd0, cfg_m}, 32'h00000102, "M independent W delay");
        axi_write(8'h18, 32'h0000aa00, 4'b0010, 3, 0, 0);
        expect_word({16'd0, cfg_m}, 32'h0000aa02, "M WSTRB merge");
        axi_write(8'h1c, 32'd2, 4'b1111, 0, 0, 0);
        axi_write(8'h20, 32'd8, 4'b1111, 0, 1, 0);
        axi_write(8'h24, 32'd8, 4'b1111, 1, 0, 0);
        axi_write(8'h28, 32'd2, 4'b1111, 0, 0, 0);
        axi_write(8'h2c, 32'd8, 4'b1111, 0, 0, 0);
        axi_write(8'h30, 32'd1000, 4'b1111, 0, 0, 0);

        axi_read(8'h18, 1, read_value);
        expect_word(read_value, 32'h0000aa02, "M readback");
        axi_read(8'h20, 0, read_value);
        expect_word(read_value, 32'd8, "K readback");

        status_busy = 1'b1;
        status_done = 1'b1;
        status_error = 1'b1;
        error_code = 8'd3;
        cycles = 64'h1122334455667788;
        axi_read(8'h10, 2, read_value);
        expect_word(read_value, 32'h00000007, "STATUS");
        axi_read(8'h14, 0, read_value);
        expect_word(read_value, 32'h00000003, "ERROR");
        axi_read(8'h34, 0, read_value);
        expect_word(read_value, 32'h55667788, "CYCLES_LO");
        axi_read(8'h38, 0, read_value);
        expect_word(read_value, 32'h11223344, "CYCLES_HI");

        axi_write(8'h1c, 32'd1, 4'b1111, 0, 0, 0);
        expect_word({16'd0, cfg_n}, 32'd2, "busy config ignored");

        prior_start_count = start_count;
        axi_write(8'h0c, 32'h00000001, 4'b0001, 2, 0, 0);
        @(posedge clk);
        if (start_count !== prior_start_count + 1) begin
            $display("MISMATCH START pulse count");
            $fatal(1);
        end

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
        expect_word({16'd0, cfg_m}, 32'd0, "soft reset M");
        expect_word({16'd0, cfg_n}, 32'd0, "soft reset N");
        expect_word({16'd0, cfg_k}, 32'd0, "soft reset K");
        expect_word(cfg_a_stride, 32'd0, "soft reset A_STRIDE");
        expect_word(cfg_b_stride, 32'd0, "soft reset B_STRIDE");
        expect_word(cfg_c_stride, 32'd0, "soft reset C_STRIDE");
        expect_word(cfg_timeout_cycles, 32'd0, "soft reset TIMEOUT");

        axi_write(8'h3c, 32'hffffffff, 4'b1111, 0, 0, 0);
        axi_read(8'h3c, 0, read_value);
        expect_word(read_value, 32'h00000000, "reserved remains zero");

        $display("PASS tb_npu_axi_lite_regs");
        $finish;
    end
endmodule
