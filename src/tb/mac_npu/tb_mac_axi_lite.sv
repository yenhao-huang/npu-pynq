`timescale 1ns/1ps

module tb_mac_axi_lite;
    logic clk = 1'b0;
    logic resetn = 1'b0;
    logic [4:0] awaddr = '0;
    logic awvalid = 1'b0;
    logic awready;
    logic [31:0] wdata = '0;
    logic [3:0] wstrb = '0;
    logic wvalid = 1'b0;
    logic wready;
    logic [1:0] bresp;
    logic bvalid;
    logic bready = 1'b0;
    logic [4:0] araddr = '0;
    logic arvalid = 1'b0;
    logic arready;
    logic [31:0] rdata;
    logic [1:0] rresp;
    logic rvalid;
    logic rready = 1'b0;
    logic [31:0] value;

    always #5 clk = ~clk;

    mac_axi_lite dut (
        .s_axi_aclk(clk), .s_axi_aresetn(resetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready)
    );

    task automatic axi_write(input logic [4:0] address, input logic [31:0] data);
        begin
            @(negedge clk);
            awaddr = address; awvalid = 1'b1;
            wdata = data; wstrb = 4'hf; wvalid = 1'b1;
            while (!(awready && wready)) @(negedge clk);
            @(negedge clk);
            awvalid = 1'b0; wvalid = 1'b0; bready = 1'b1;
            while (!bvalid) @(negedge clk);
            if (bresp != 2'b00) $fatal(1, "AXI write error at 0x%0h", address);
            @(negedge clk); bready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [4:0] address, output logic [31:0] data);
        begin
            @(negedge clk);
            araddr = address; arvalid = 1'b1; rready = 1'b1;
            while (!arready) @(negedge clk);
            @(negedge clk); arvalid = 1'b0;
            while (!rvalid) @(negedge clk);
            data = rdata;
            if (rresp != 2'b00) $fatal(1, "AXI read error at 0x%0h", address);
            @(negedge clk); rready = 1'b0;
        end
    endtask

    task automatic wait_done;
        integer timeout;
        begin
            timeout = 20;
            value = 0;
            while (value[0] == 0 && timeout > 0) begin
                axi_read(5'h0c, value);
                timeout = timeout - 1;
            end
            if (timeout == 0) $fatal(1, "status done timeout");
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        resetn = 1'b1;

        axi_write(5'h00, 32'h1);
        axi_read(5'h10, value);
        if ($signed(value) != 0) $fatal(1, "clear failed: %0d", $signed(value));

        axi_write(5'h04, 32'd2);
        axi_write(5'h08, 32'd3);
        axi_write(5'h00, 32'h2);
        wait_done();
        axi_read(5'h10, value);
        if ($signed(value) != 6) $fatal(1, "2*3 failed: %0d", $signed(value));

        axi_write(5'h04, 32'hfffffff9);
        axi_write(5'h08, 32'd6);
        axi_write(5'h00, 32'h2);
        wait_done();
        axi_read(5'h10, value);
        if ($signed(value) != -36) $fatal(1, "accumulate -7*6 failed: %0d", $signed(value));

        axi_read(5'h04, value);
        if ($signed(value) != -7) $fatal(1, "signed operand A readback failed");

        axi_write(5'h00, 32'h1);
        axi_read(5'h10, value);
        if ($signed(value) != 0) $fatal(1, "final clear failed: %0d", $signed(value));

        $display("PASS: AXI4-Lite writes a/b/clear/start and reads accumulator");
        $finish;
    end
endmodule
