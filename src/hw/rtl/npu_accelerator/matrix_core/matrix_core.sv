`timescale 1ns/1ps

// Matrix core: runs INT8 GEMM jobs over AXI-Stream. It wires the controller
// (u_controller), which sequences each job cycle by cycle, to the operand
// buffers (one bank per array edge) and the systolic array.
module matrix_core #(
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

    matrix_core_controller #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K), .IN_BYTES(IN_BYTES)
    ) u_controller (.*);

    // Operand memory and the systolic array, driven by the controller.
    localparam integer K_ADDR_WIDTH = (MAX_K > 1) ? $clog2(MAX_K) : 1;
    localparam integer LANE_BITS = (IN_BYTES > 1) ? $clog2(IN_BYTES) : 1;
    // Each bank holds two halves of 2**K_ADDR_WIDTH bytes; the address MSB is
    // the ping-pong half, load_half for writes and exec_half for reads.
    localparam integer BANK_DEPTH = 2 << K_ADDR_WIDTH;
    localparam integer A_LANE_BITS = (IN_BYTES > 1) ? $clog2(IN_BYTES) : 0;
    localparam integer A_HALF_WORD_BITS = K_ADDR_WIDTH - A_LANE_BITS;

    wire signed [ROWS*8-1:0] array_a;
    wire signed [COLUMNS*8-1:0] array_b;

    for (genvar bank = 0; bank < ROWS; bank++) begin : gen_a_banks
        wire [K_ADDR_WIDTH-1:0] read_index = K_ADDR_WIDTH'(compute_step - bank);
        wire [A_HALF_WORD_BITS:0] write_word;
        if (A_HALF_WORD_BITS > 0) begin : gen_word
            assign write_word = {load_half, align_word[A_HALF_WORD_BITS-1:0]};
        end else begin : gen_half
            assign write_word = load_half;
        end
        operand_buffer #(.DEPTH(BANK_DEPTH), .WRITE_BYTES(IN_BYTES)) buffer (
            .clk(clk),
            .write_enable(align_emit && state_load_a && (align_row == 16'(bank))),
            .write_word_address(write_word),
            .write_data(write_data),
            .read_enable(array_enable),
            .read_address({exec_half, read_index}),
            .read_data(array_a[bank*8 +: 8])
        );
    end

    for (genvar bank = 0; bank < COLUMNS; bank++) begin : gen_b_banks
        wire [K_ADDR_WIDTH-1:0] read_index = K_ADDR_WIDTH'(compute_step - bank);
        operand_buffer #(.DEPTH(BANK_DEPTH), .WRITE_BYTES(1)) buffer (
            .clk(clk),
            .write_enable(align_emit && state_load_b &&
                          (align_word == 16'(bank >> LANE_BITS))),
            .write_word_address({load_half, align_row[K_ADDR_WIDTH-1:0]}),
            .write_data(write_data[(bank % IN_BYTES)*8 +: 8]),
            .read_enable(array_enable),
            .read_address({exec_half, read_index}),
            .read_data(array_b[bank*8 +: 8])
        );
    end

    systolic_array #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .DATA_WIDTH(8), .ACC_WIDTH(32)
    ) array (
        .clk(clk), .rst_n(rst_n), .clear(array_clear), .enable(array_enable),
        .a_in(array_a), .a_valid_in(array_a_valid),
        .b_in(array_b), .b_valid_in(array_b_valid),
        .accumulators(array_accumulators)
    );
endmodule
