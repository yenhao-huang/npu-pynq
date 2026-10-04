`timescale 1ns/1ps

module tb_npu_operand_buffer;
    logic clk = 1'b0;
    logic a_write_enable = 1'b0, a_read_enable = 1'b0;
    logic [0:0] a_write_word_address = '0;
    logic [3:0] a_read_address = '0;
    logic [63:0] a_write_data = '0;
    logic [7:0] a_read_data;
    logic b_write_enable = 1'b0, b_read_enable = 1'b0;
    logic [3:0] b_write_word_address = '0, b_read_address = '0;
    logic [7:0] b_write_data = '0, b_read_data;

    npu_operand_buffer #(.DEPTH(16), .WRITE_BYTES(8)) a_bank (
        .clk(clk), .write_enable(a_write_enable),
        .write_word_address(a_write_word_address),
        .write_data(a_write_data), .read_enable(a_read_enable),
        .read_address(a_read_address), .read_data(a_read_data)
    );
    npu_operand_buffer #(.DEPTH(16), .WRITE_BYTES(1)) b_bank (
        .clk(clk), .write_enable(b_write_enable),
        .write_word_address(b_write_word_address),
        .write_data(b_write_data), .read_enable(b_read_enable),
        .read_address(b_read_address), .read_data(b_read_data)
    );
    always #5 clk = ~clk;

    initial begin
        @(negedge clk);
        a_write_enable = 1'b1;
        a_write_data = 64'h0807060504030280;
        b_write_enable = 1'b1;
        b_write_word_address = 4'd4;
        b_write_data = 8'hfd;
        @(posedge clk); #1;
        a_write_enable = 1'b0;
        b_write_enable = 1'b0;

        @(negedge clk);
        a_read_enable = 1'b1;
        a_read_address = 4'd0;
        b_read_enable = 1'b1;
        b_read_address = 4'd4;
        @(posedge clk); #1;
        if (a_read_data !== 8'h80 || b_read_data !== 8'hfd)
            $fatal(1, "FAIL tb_npu_operand_buffer: synchronous bank read");

        @(negedge clk);
        a_read_address = 4'd7;
        a_read_enable = 1'b0;
        b_read_enable = 1'b0;
        @(posedge clk); #1;
        if (a_read_data !== 8'h80 || b_read_data !== 8'hfd)
            $fatal(1, "FAIL tb_npu_operand_buffer: disabled read holds data");

        @(negedge clk);
        a_read_enable = 1'b1;
        @(posedge clk); #1;
        if (a_read_data !== 8'h08)
            $fatal(1, "FAIL tb_npu_operand_buffer: final wide-write lane");

        $display("PASS tb_npu_operand_buffer");
        $finish;
    end
endmodule
