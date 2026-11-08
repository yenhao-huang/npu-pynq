`timescale 1ns/1ps

module tb_npu_accelerator;
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
    // The instruction-stream master stays idle in this register-driven test.
    logic [31:0] m_axi_araddr, m_axi_awaddr;
    logic [7:0] m_axi_arlen, m_axi_awlen, m_axi_wstrb;
    logic [2:0] m_axi_arsize, m_axi_awsize, m_axi_arprot, m_axi_awprot;
    logic [1:0] m_axi_arburst, m_axi_awburst;
    logic [3:0] m_axi_arcache, m_axi_awcache;
    logic m_axi_arvalid, m_axi_rready, m_axi_awvalid, m_axi_wlast, m_axi_wvalid, m_axi_bready;
    logic [63:0] m_axi_wdata;
    logic m_axi_arready = 0, m_axi_rlast = 0, m_axi_rvalid = 0;
    logic m_axi_awready = 0, m_axi_wready = 0, m_axi_bvalid = 0;
    logic [63:0] m_axi_rdata = 0;
    logic [1:0] m_axi_rresp = 0, m_axi_bresp = 0;
    logic [31:0] read_value, held_data;
    logic held_last;

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

    // One 2x2 operand is four bytes: a single 64-bit beat whose TKEEP marks
    // the low four lanes, as AXI DMA MM2S sends a 4-byte transfer.
    task automatic stream_operand(
        input integer signed b0, input integer signed b1,
        input integer signed b2, input integer signed b3
    );
        begin
            @(negedge s_axi_aclk);
            s_axis_tdata = {32'd0, b3[7:0], b2[7:0], b1[7:0], b0[7:0]};
            s_axis_tkeep = 8'h0f; s_axis_tlast = 1; s_axis_tvalid = 1;
            while (!s_axis_tready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axis_tvalid = 0; s_axis_tlast = 0; s_axis_tkeep = 0;
        end
    endtask

    // A 1x2 operand is two bytes: one beat with the low two lanes kept.
    task automatic stream_pair(input integer signed b0, input integer signed b1);
        begin
            @(negedge s_axi_aclk);
            s_axis_tdata = {48'd0, b1[7:0], b0[7:0]};
            s_axis_tkeep = 8'h03; s_axis_tlast = 1; s_axis_tvalid = 1;
            while (!s_axis_tready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axis_tvalid = 0; s_axis_tlast = 0; s_axis_tkeep = 0;
        end
    endtask

    task automatic take_result(
        input integer signed expected, input logic expected_last, input integer stall_cycles
    );
        integer index;
        begin
            wait (m_axis_tvalid);
            held_data = m_axis_tdata; held_last = m_axis_tlast;
            for (index = 0; index < stall_cycles; index = index + 1) begin
                @(posedge s_axi_aclk); #1;
                if (!m_axis_tvalid || m_axis_tdata !== held_data || m_axis_tlast !== held_last)
                    fail("result changed under backpressure");
            end
            if ($signed(held_data) != expected || held_last != expected_last)
                fail("result data or TLAST mismatch");
            @(negedge s_axi_aclk); m_axis_tready = 1;
            @(posedge s_axi_aclk); #1; m_axis_tready = 0;
        end
    endtask

    initial begin
        repeat (4) @(posedge s_axi_aclk);
        @(negedge s_axi_aclk); s_axi_aresetn = 1;

        axi_read(8'h00, read_value);
        if (read_value != 32'h3155504e) fail("MAGIC through public AXI");
        axi_write(8'h18, 2); axi_write(8'h1c, 2); axi_write(8'h20, 2);
        axi_write(8'h24, 2); axi_write(8'h28, 2); axi_write(8'h2c, 8);
        axi_write(8'h30, 200); axi_write(8'h0c, 1);

        axi_read(8'h10, read_value);
        if (read_value != 32'h9) fail("BUSY|ACCEPT did not assert through public AXI");

        stream_operand(-128, 127, 7, -3);
        stream_operand(-1, 2, 4, -5);

        // The A/B banks leave one queue entry free, so the second job's
        // configuration and START are admitted through the same register
        // window while the first job still owns the array.
        axi_read(8'h10, read_value);
        if (!(read_value & 32'h8)) fail("ACCEPT cleared with one job in flight");
        axi_write(8'h18, 1); axi_write(8'h1c, 2); axi_write(8'h20, 2);
        axi_write(8'h24, 2); axi_write(8'h28, 2); axi_write(8'h2c, 8);
        axi_write(8'h30, 400); axi_write(8'h0c, 1);
        axi_read(8'h14, read_value);
        if (read_value != 0) fail("queued START raised an error");
        axi_read(8'h10, read_value);
        if (read_value & 32'h8) fail("ACCEPT still set with a full queue");

        take_result(636, 0, 3);
        take_result(-891, 0, 1);
        take_result(-19, 0, 2);
        take_result(29, 1, 2);

        axi_read(8'h10, read_value);
        if (read_value != 32'hb) fail("expected BUSY|DONE|ACCEPT after first job");
        axi_read(8'h14, read_value);
        if (read_value != 0) fail("unexpected ERROR code");
        axi_read(8'h34, read_value);
        if (read_value == 0) fail("cycle count is zero");

        // Second job: 1x2x2 with A = [2 -3], B = [[4 6], [5 -7]].
        stream_pair(2, -3);
        stream_operand(4, 6, 5, -7);
        take_result(-7, 0, 0);
        take_result(33, 1, 2);

        axi_read(8'h10, read_value);
        if (read_value != 32'ha) fail("expected DONE|ACCEPT after the queue drained");
        axi_read(8'h14, read_value);
        if (read_value != 0) fail("unexpected ERROR code after second job");
        axi_read(8'h34, read_value);
        if (read_value == 0) fail("cycle count is zero");
        held_data = read_value;
        repeat (3) @(posedge s_axi_aclk);
        axi_read(8'h34, read_value);
        if (read_value != held_data) fail("cycle count changed after completion");

        $display("PASS tb_npu_accelerator");
        $finish;
    end
endmodule
