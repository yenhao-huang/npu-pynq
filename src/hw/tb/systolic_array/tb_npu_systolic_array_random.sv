`timescale 1ns/1ps

module tb_npu_systolic_array_random;
    localparam integer ROWS = 2;
    localparam integer COLUMNS = 2;
    localparam integer MAX_K = 8;

    logic clk;
    logic rst_n;
    logic clear;
    logic enable;
    logic signed [ROWS*8-1:0] a_in;
    logic [ROWS-1:0] a_valid_in;
    logic signed [COLUMNS*8-1:0] b_in;
    logic [COLUMNS-1:0] b_valid_in;
    logic signed [ROWS*COLUMNS*32-1:0] accumulators;

    integer signed a_values [0:ROWS*MAX_K-1];
    integer signed b_values [0:MAX_K*COLUMNS-1];
    integer signed expected_values [0:ROWS*COLUMNS-1];
    integer vector_file;
    integer scan_result;
    integer format_version;
    integer seed;
    integer case_count;
    integer file_rows;
    integer file_columns;
    integer file_max_k;
    integer case_index;
    integer active_m;
    integer active_n;
    integer active_k;
    integer stall_step;
    integer logical_step;
    integer value_index;

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
            enable = step_enable;
            clear = 1'b0;

            for (row = 0; row < active_m; row = row + 1) begin
                reduction = step_index - row;
                if (reduction >= 0 && reduction < active_k) begin
                    a_in[row*8 +: 8] = a_values[row*MAX_K+reduction];
                    a_valid_in[row] = 1'b1;
                end
            end
            for (column = 0; column < active_n; column = column + 1) begin
                reduction = step_index - column;
                if (reduction >= 0 && reduction < active_k) begin
                    b_in[column*8 +: 8] = b_values[reduction*COLUMNS+column];
                    b_valid_in[column] = 1'b1;
                end
            end

            @(posedge clk);
            #1;
        end
    endtask

    task automatic check_case;
        integer row;
        integer column;
        integer result_index;
        logic signed [31:0] actual;
        begin
            for (row = 0; row < ROWS; row = row + 1) begin
                for (column = 0; column < COLUMNS; column = column + 1) begin
                    result_index = row*COLUMNS + column;
                    actual = accumulators[result_index*32 +: 32];
                    if (actual !== expected_values[result_index]) begin
                        $display(
                            "FAIL random case=%0d coord=[%0d,%0d] expected=%0d got=%0d",
                            case_index, row, column,
                            expected_values[result_index], actual
                        );
                        $fatal(1);
                    end
                end
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

        vector_file = $fopen("src/test/vectors/systolic_2x2_k8.txt", "r");
        if (vector_file == 0) begin
            $display("FAIL random: unable to open vector fixture");
            $fatal(1);
        end
        scan_result = $fscanf(
            vector_file, "%d %d %d %d %d %d",
            format_version, seed, case_count,
            file_rows, file_columns, file_max_k
        );
        if (scan_result != 6 || format_version != 1 ||
            file_rows != ROWS || file_columns != COLUMNS ||
            file_max_k != MAX_K || case_count < 100) begin
            $display("FAIL random: invalid vector header");
            $fatal(1);
        end

        @(negedge clk);
        rst_n = 1'b1;

        for (case_index = 0; case_index < case_count; case_index = case_index + 1) begin
            scan_result = $fscanf(
                vector_file, "%d %d %d %d",
                active_m, active_n, active_k, stall_step
            );
            if (scan_result != 4) begin
                $display("FAIL random: truncated case header %0d", case_index);
                $fatal(1);
            end
            for (value_index = 0; value_index < ROWS*MAX_K; value_index = value_index + 1) begin
                scan_result = $fscanf(vector_file, "%d", a_values[value_index]);
                if (scan_result != 1) $fatal(1, "truncated A values");
            end
            for (value_index = 0; value_index < MAX_K*COLUMNS; value_index = value_index + 1) begin
                scan_result = $fscanf(vector_file, "%d", b_values[value_index]);
                if (scan_result != 1) $fatal(1, "truncated B values");
            end
            for (value_index = 0; value_index < ROWS*COLUMNS; value_index = value_index + 1) begin
                scan_result = $fscanf(vector_file, "%d", expected_values[value_index]);
                if (scan_result != 1) $fatal(1, "truncated C values");
            end

            clear_array();
            for (
                logical_step = 0;
                logical_step <= active_m + active_n + active_k - 2;
                logical_step = logical_step + 1
            ) begin
                if (logical_step == stall_step) begin
                    drive_active_step(logical_step, 1'b0);
                end
                drive_active_step(logical_step, 1'b1);
            end
            check_case();
        end

        $fclose(vector_file);
        $display("PASS tb_npu_systolic_array_random seed=%0d cases=%0d", seed, case_count);
        $finish;
    end
endmodule
