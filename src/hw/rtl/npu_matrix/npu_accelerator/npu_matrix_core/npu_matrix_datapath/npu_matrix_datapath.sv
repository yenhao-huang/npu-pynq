`timescale 1ns/1ps

module npu_matrix_datapath #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256,
    parameter integer IN_BYTES = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic state_load_a,
    input  logic state_load_b,
    input  logic align_emit,
    input  logic [15:0] align_row,
    input  logic [15:0] align_word,
    input  logic [IN_BYTES*8-1:0] write_data,
    input  logic [31:0] compute_step,
    input  logic array_clear,
    input  logic array_enable,
    input  logic [ROWS-1:0] array_a_valid,
    input  logic [COLUMNS-1:0] array_b_valid,
    output wire signed [ROWS*COLUMNS*32-1:0] array_accumulators
);
    localparam integer K_ADDR_WIDTH = (MAX_K > 1) ? $clog2(MAX_K) : 1;
    localparam integer LANE_BITS = (IN_BYTES > 1) ? $clog2(IN_BYTES) : 1;
    localparam integer WORD_ADDR_WIDTH =
        (K_ADDR_WIDTH > LANE_BITS) ? K_ADDR_WIDTH - LANE_BITS : 1;

    wire signed [ROWS*8-1:0] array_a;
    wire signed [COLUMNS*8-1:0] array_b;

    for (genvar bank = 0; bank < ROWS; bank++) begin : gen_a_banks
        wire [K_ADDR_WIDTH-1:0] read_index = K_ADDR_WIDTH'(compute_step - bank);
        npu_operand_buffer #(.DEPTH(MAX_K), .WRITE_BYTES(IN_BYTES)) buffer (
            .clk(clk),
            .write_enable(align_emit && state_load_a && (align_row == 16'(bank))),
            .write_word_address(align_word[WORD_ADDR_WIDTH-1:0]),
            .write_data(write_data),
            .read_enable(array_enable),
            .read_address(read_index),
            .read_data(array_a[bank*8 +: 8])
        );
    end

    for (genvar bank = 0; bank < COLUMNS; bank++) begin : gen_b_banks
        wire [K_ADDR_WIDTH-1:0] read_index = K_ADDR_WIDTH'(compute_step - bank);
        npu_operand_buffer #(.DEPTH(MAX_K), .WRITE_BYTES(1)) buffer (
            .clk(clk),
            .write_enable(align_emit && state_load_b &&
                          (align_word == 16'(bank >> LANE_BITS))),
            .write_word_address(align_row[K_ADDR_WIDTH-1:0]),
            .write_data(write_data[(bank % IN_BYTES)*8 +: 8]),
            .read_enable(array_enable),
            .read_address(read_index),
            .read_data(array_b[bank*8 +: 8])
        );
    end

    npu_systolic_array #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .DATA_WIDTH(8), .ACC_WIDTH(32)
    ) array (
        .clk(clk), .rst_n(rst_n), .clear(array_clear), .enable(array_enable),
        .a_in(array_a), .a_valid_in(array_a_valid),
        .b_in(array_b), .b_valid_in(array_b_valid),
        .accumulators(array_accumulators)
    );
endmodule
