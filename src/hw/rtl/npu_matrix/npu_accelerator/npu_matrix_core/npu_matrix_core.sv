`timescale 1ns/1ps

module npu_matrix_core #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256,
    parameter integer IN_BYTES = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start_pulse,
    input  logic soft_reset_pulse,
    input  logic [15:0] cfg_m,
    input  logic [15:0] cfg_n,
    input  logic [15:0] cfg_k,
    input  logic [31:0] cfg_a_stride,
    input  logic [31:0] cfg_b_stride,
    input  logic [31:0] cfg_c_stride,
    input  logic [31:0] cfg_timeout_cycles,
    input  logic [IN_BYTES*8-1:0] s_axis_tdata,
    input  logic [IN_BYTES-1:0] s_axis_tkeep,
    input  logic s_axis_tvalid,
    output logic s_axis_tready,
    input  logic s_axis_tlast,
    output logic [31:0] m_axis_tdata,
    output logic m_axis_tvalid,
    input  logic m_axis_tready,
    output logic m_axis_tlast,
    output logic status_busy,
    output logic status_accept,
    output logic status_done,
    output logic status_error,
    output logic [7:0] error_code,
    output logic [63:0] cycles
);
    wire state_load_a, state_load_b, load_half, exec_half, align_emit;
    wire [15:0] align_row, align_word;
    wire [IN_BYTES*8-1:0] write_data;
    wire [31:0] compute_step;
    wire array_clear, array_enable;
    wire [ROWS-1:0] array_a_valid;
    wire [COLUMNS-1:0] array_b_valid;
    wire signed [ROWS*COLUMNS*32-1:0] array_accumulators;

    npu_matrix_controller #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K), .IN_BYTES(IN_BYTES)
    ) controller (.*);

    npu_matrix_datapath #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K), .IN_BYTES(IN_BYTES)
    ) datapath (.*);
endmodule
