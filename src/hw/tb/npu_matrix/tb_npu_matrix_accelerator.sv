`timescale 1ns/1ps

module tb_npu_matrix_accelerator;
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
    logic [7:0] s_axis_tdata = 0;
    logic s_axis_tvalid = 0, s_axis_tready, s_axis_tlast = 0;
    logic [7:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready = 0, m_axis_tlast;
    logic irq;
    logic [31:0] read_value;
    logic [7:0] held_data;
    logic held_last;
    integer channel;

    npu_matrix_accelerator #(.ROWS(2), .COLUMNS(2), .MAX_K(256)) dut (.*);
    always #5 s_axi_aclk = ~s_axi_aclk;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_matrix_accelerator: %s", message);
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

    task automatic stream_beat(input integer signed value, input logic last);
        begin
            @(negedge s_axi_aclk);
            s_axis_tdata = value[7:0]; s_axis_tlast = last; s_axis_tvalid = 1;
            while (!s_axis_tready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axis_tvalid = 0; s_axis_tlast = 0;
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
        axi_write(8'h24, 2); axi_write(8'h28, 2); axi_write(8'h2c, 2);
        axi_write(8'h30, 400); axi_write(8'h3c, 3); axi_write(8'h40, 0);
        axi_write(8'h0c, 1);

        axi_read(8'h10, read_value);
        if (read_value != 1) fail("BUSY did not assert through public AXI");

        stream_beat(-128, 0); stream_beat(127, 0);
        stream_beat(7, 0); stream_beat(-3, 1);
        stream_beat(-1, 0); stream_beat(2, 0);
        stream_beat(4, 0); stream_beat(-5, 1);
        for (channel = 0; channel < 2; channel = channel + 1) begin
            stream_beat(0, 0); stream_beat(0, 0);
            stream_beat(0, 0); stream_beat(0, channel == 1);
        end
        for (channel = 0; channel < 2; channel = channel + 1) begin
            stream_beat(0, 0); stream_beat(0, 0); stream_beat(0, 0);
            stream_beat(64, channel == 1);
        end
        stream_beat(0, 0); stream_beat(0, 1);

        take_result(127, 0, 3);
        take_result(-128, 0, 1);
        take_result(-10, 0, 2);
        take_result(15, 1, 2);

        axi_read(8'h10, read_value);
        if (read_value != 2) fail("expected DONE without ERROR");
        axi_read(8'h14, read_value);
        if (read_value != 0) fail("unexpected ERROR code");
        axi_read(8'h34, read_value);
        if (read_value == 0) fail("cycle count is zero");
        held_data = read_value;
        repeat (3) @(posedge s_axi_aclk);
        axi_read(8'h34, read_value);
        if (read_value != held_data) fail("cycle count changed after completion");

        $display("PASS tb_npu_matrix_accelerator");
        $finish;
    end
endmodule
