`timescale 1ns/1ps

module tb_npu_systolic_array_rect;
    localparam integer ROWS = 2;
    localparam integer COLUMNS = 3;
    localparam integer K = 2;

    logic clk;
    logic rst_n;
    logic clear;
    logic enable;
    logic signed [ROWS*8-1:0] a_in;
    logic [ROWS-1:0] a_valid_in;
    logic signed [COLUMNS*8-1:0] b_in;
    logic [COLUMNS-1:0] b_valid_in;
    logic signed [ROWS*COLUMNS*32-1:0] accumulators;

    integer signed matrix_a [0:ROWS-1][0:K-1];
    integer signed matrix_b [0:K-1][0:COLUMNS-1];
    integer logical_step;

    npu_systolic_array #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .enable(enable),
        .a_in(a_in),
        .a_valid_in(a_valid_in),
        .b_in(b_in),
        .b_valid_in(b_valid_in),
        .accumulators(accumulators)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task automatic drive_step(input integer step_index);
        integer row;
        integer column;
        integer reduction;
        begin
            @(negedge clk);
            a_in = '0;
            b_in = '0;
            a_valid_in = '0;
            b_valid_in = '0;
            enable = 1'b1;
            for (row = 0; row < ROWS; row = row + 1) begin
                reduction = step_index - row;
                if (reduction >= 0 && reduction < K) begin
                    a_in[row*8 +: 8] = matrix_a[row][reduction];
                    a_valid_in[row] = 1'b1;
                end
            end
            for (column = 0; column < COLUMNS; column = column + 1) begin
                reduction = step_index - column;
                if (reduction >= 0 && reduction < K) begin
                    b_in[column*8 +: 8] = matrix_b[reduction][column];
                    b_valid_in[column] = 1'b1;
                end
            end
            @(posedge clk);
            #1;
        end
    endtask

    task automatic check_accumulator(
        input integer row,
        input integer column,
        input integer signed expected
    );
        logic signed [31:0] actual;
        begin
            actual = accumulators[(row*COLUMNS+column)*32 +: 32];
            if (actual !== expected) begin
                $display(
                    "FAIL tb_npu_systolic_array_rect: [%0d,%0d] expected %0d got %0d",
                    row, column, expected, actual
                );
                $fatal(1);
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        enable = 1'b0;
        a_in = '0;
        b_in = '0;
        a_valid_in = '0;
        b_valid_in = '0;
        matrix_a[0][0] = 1;
        matrix_a[0][1] = -2;
        matrix_a[1][0] = 3;
        matrix_a[1][1] = 4;
        matrix_b[0][0] = 5;
        matrix_b[0][1] = 6;
        matrix_b[0][2] = 7;
        matrix_b[1][0] = -8;
        matrix_b[1][1] = 9;
        matrix_b[1][2] = 10;

        @(negedge clk);
        rst_n = 1'b1;
        for (logical_step = 0; logical_step <= 5; logical_step = logical_step + 1) begin
            drive_step(logical_step);
        end

        check_accumulator(0, 0, 21);
        check_accumulator(0, 1, -12);
        check_accumulator(0, 2, -13);
        check_accumulator(1, 0, -17);
        check_accumulator(1, 1, 54);
        check_accumulator(1, 2, 61);

        $display("PASS tb_npu_systolic_array_rect");
        $finish;
    end
endmodule
