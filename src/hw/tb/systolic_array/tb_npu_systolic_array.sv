`timescale 1ns/1ps

module tb_npu_systolic_array;
    localparam integer ROWS = 2;
    localparam integer COLUMNS = 2;
    localparam integer K_MAX = 3;

    logic clk;
    logic rst_n;
    logic clear;
    logic enable;
    logic signed [ROWS*8-1:0] a_in;
    logic [ROWS-1:0] a_valid_in;
    logic signed [COLUMNS*8-1:0] b_in;
    logic [COLUMNS-1:0] b_valid_in;
    logic signed [ROWS*COLUMNS*32-1:0] accumulators;

    integer signed matrix_a [0:ROWS-1][0:K_MAX-1];
    integer signed matrix_b [0:K_MAX-1][0:COLUMNS-1];
    integer active_m;
    integer active_n;
    integer active_k;
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

    task automatic drive_active_step(
        input integer step_index,
        input logic step_enable
    );
        integer row;
        integer column;
        integer reduction;
        begin
            @(negedge clk);
            a_in = '0;
            b_in = '0;
            a_valid_in = '0;
            b_valid_in = '0;
            clear = 1'b0;
            enable = step_enable;

            for (row = 0; row < active_m; row = row + 1) begin
                reduction = step_index - row;
                if (reduction >= 0 && reduction < active_k) begin
                    a_in[row*8 +: 8] = matrix_a[row][reduction];
                    a_valid_in[row] = 1'b1;
                end
            end
            for (column = 0; column < active_n; column = column + 1) begin
                reduction = step_index - column;
                if (reduction >= 0 && reduction < active_k) begin
                    b_in[column*8 +: 8] = matrix_b[reduction][column];
                    b_valid_in[column] = 1'b1;
                end
            end

            @(posedge clk);
            #1;
        end
    endtask

    task automatic clear_array;
        begin
            @(negedge clk);
            clear = 1'b1;
            enable = 1'b0;
            a_in = '0;
            b_in = '0;
            a_valid_in = '0;
            b_valid_in = '0;
            @(posedge clk);
            #1;
            clear = 1'b0;
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
                    "FAIL tb_npu_systolic_array: [%0d,%0d] expected %0d got %0d",
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

        #1;
        if (accumulators !== '0) begin
            $display("FAIL tb_npu_systolic_array: reset did not clear array");
            $fatal(1);
        end
        @(negedge clk);
        rst_n = 1'b1;

        active_m = 2;
        active_n = 2;
        active_k = 3;
        matrix_a[0][0] = 1;
        matrix_a[0][1] = 2;
        matrix_a[0][2] = 3;
        matrix_a[1][0] = -4;
        matrix_a[1][1] = 5;
        matrix_a[1][2] = -6;
        matrix_b[0][0] = 7;
        matrix_b[0][1] = -8;
        matrix_b[1][0] = 9;
        matrix_b[1][1] = 10;
        matrix_b[2][0] = -11;
        matrix_b[2][1] = 12;

        for (logical_step = 0; logical_step <= 4; logical_step = logical_step + 1) begin
            if (logical_step == 2) begin
                drive_active_step(logical_step, 1'b0);
            end
            drive_active_step(logical_step, 1'b1);
        end
        check_accumulator(0, 0, -8);
        check_accumulator(0, 1, 48);
        check_accumulator(1, 0, 83);
        check_accumulator(1, 1, 10);

        clear_array();
        check_accumulator(0, 0, 0);
        check_accumulator(0, 1, 0);
        check_accumulator(1, 0, 0);
        check_accumulator(1, 1, 0);

        active_m = 1;
        active_n = 1;
        active_k = 2;
        matrix_a[0][0] = 2;
        matrix_a[0][1] = -3;
        matrix_b[0][0] = 4;
        matrix_b[1][0] = 5;
        for (logical_step = 0; logical_step <= 1; logical_step = logical_step + 1) begin
            drive_active_step(logical_step, 1'b1);
        end
        check_accumulator(0, 0, -7);
        check_accumulator(0, 1, 0);
        check_accumulator(1, 0, 0);
        check_accumulator(1, 1, 0);

        $display("PASS tb_npu_systolic_array");
        $finish;
    end
endmodule
