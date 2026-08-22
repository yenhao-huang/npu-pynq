`timescale 1ns/1ps

module npu_matrix_controller #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start_pulse,
    input  logic        soft_reset_pulse,
    input  logic [15:0] cfg_m,
    input  logic [15:0] cfg_n,
    input  logic [15:0] cfg_k,
    input  logic [31:0] cfg_a_stride,
    input  logic [31:0] cfg_b_stride,
    input  logic [31:0] cfg_c_stride,
    input  logic [31:0] cfg_timeout_cycles,
    input  logic [7:0]  s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,
    output logic        status_busy,
    output logic        status_done,
    output logic        status_error,
    output logic [7:0]  error_code,
    output logic [63:0] cycles
);
    localparam logic [7:0] ERR_INVALID_DIMENSION = 8'd1;
    localparam logic [7:0] ERR_INVALID_STRIDE = 8'd2;
    localparam logic [7:0] ERR_BUSY_START = 8'd3;
    localparam logic [7:0] ERR_STREAM_LENGTH = 8'd4;
    localparam logic [7:0] ERR_TIMEOUT = 8'd5;
    localparam logic [7:0] ERR_INVALID_TIMEOUT = 8'd6;

    typedef enum logic [2:0] {
        STATE_IDLE,
        STATE_LOAD_A,
        STATE_LOAD_B,
        STATE_CLEAR,
        STATE_COMPUTE,
        STATE_OUTPUT
    } state_t;

    state_t state;
    logic [15:0] active_m, active_n, active_k;
    logic [31:0] active_timeout;
    logic [15:0] load_outer;
    logic [15:0] load_inner;
    logic [31:0] compute_step;
    logic [15:0] output_row_count;
    logic [15:0] output_column_count;
    logic signed [7:0] a_buffer [0:ROWS*MAX_K-1];
    logic signed [7:0] b_buffer [0:MAX_K*COLUMNS-1];

    logic array_clear, array_enable;
    logic signed [ROWS*8-1:0] array_a;
    logic [ROWS-1:0] array_a_valid;
    logic signed [COLUMNS*8-1:0] array_b;
    logic [COLUMNS-1:0] array_b_valid;
    logic signed [ROWS*8-1:0] scheduled_a;
    logic [ROWS-1:0] scheduled_a_valid;
    logic signed [COLUMNS*8-1:0] scheduled_b;
    logic [COLUMNS-1:0] scheduled_b_valid;
    wire signed [ROWS*COLUMNS*32-1:0] array_accumulators;

    integer row_index;
    integer column_index;
    integer reduction_index;

    npu_systolic_array #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS),
        .DATA_WIDTH(8),
        .ACC_WIDTH(32)
    ) array (
        .clk(clk),
        .rst_n(rst_n),
        .clear(array_clear),
        .enable(array_enable),
        .a_in(array_a),
        .a_valid_in(array_a_valid),
        .b_in(array_b),
        .b_valid_in(array_b_valid),
        .accumulators(array_accumulators)
    );

    initial begin
        if (ROWS <= 0 || COLUMNS <= 0 || MAX_K <= 0)
            $fatal(1, "npu_matrix_controller parameters must be positive");
    end

    always_comb begin
        s_axis_tready = status_busy &&
            ((state == STATE_LOAD_A) || (state == STATE_LOAD_B));
        m_axis_tvalid = status_busy && (state == STATE_OUTPUT);
        m_axis_tdata = 32'd0;
        m_axis_tlast = 1'b0;
        if (state == STATE_OUTPUT) begin
            m_axis_tdata = array_accumulators[
                (output_row_count*COLUMNS+output_column_count)*32 +: 32
            ];
            m_axis_tlast = (output_row_count == active_m - 1) &&
                (output_column_count == active_n - 1);
        end

        array_clear = (state == STATE_CLEAR);
        array_enable = (state == STATE_COMPUTE);
        scheduled_a = '0;
        scheduled_a_valid = '0;
        scheduled_b = '0;
        scheduled_b_valid = '0;
        if (state == STATE_COMPUTE) begin
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1) begin
                reduction_index = compute_step - row_index;
                if ((row_index < active_m) &&
                    (reduction_index >= 0) && (reduction_index < active_k)) begin
                    scheduled_a[row_index*8 +: 8] =
                        a_buffer[row_index*MAX_K + reduction_index];
                    scheduled_a_valid[row_index] = 1'b1;
                end
            end
            for (column_index = 0; column_index < COLUMNS;
                 column_index = column_index + 1) begin
                reduction_index = compute_step - column_index;
                if ((column_index < active_n) &&
                    (reduction_index >= 0) && (reduction_index < active_k)) begin
                    scheduled_b[column_index*8 +: 8] =
                        b_buffer[reduction_index*COLUMNS + column_index];
                    scheduled_b_valid[column_index] = 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_m <= 0;
            active_n <= 0;
            active_k <= 0;
            active_timeout <= 0;
            load_outer <= 0;
            load_inner <= 0;
            compute_step <= 0;
            output_row_count <= 0;
            output_column_count <= 0;
            status_busy <= 1'b0;
            status_done <= 1'b0;
            status_error <= 1'b0;
            error_code <= 0;
            cycles <= 0;
        end else if (soft_reset_pulse) begin
            state <= STATE_IDLE;
            active_m <= 0;
            active_n <= 0;
            active_k <= 0;
            active_timeout <= 0;
            load_outer <= 0;
            load_inner <= 0;
            compute_step <= 0;
            output_row_count <= 0;
            output_column_count <= 0;
            status_busy <= 1'b0;
            status_done <= 1'b0;
            status_error <= 1'b0;
            error_code <= 0;
            cycles <= 0;
        end else if (status_busy) begin
            if (start_pulse && !status_error) begin
                status_error <= 1'b1;
                error_code <= ERR_BUSY_START;
            end

            if (((cycles + 1) >= active_timeout) &&
                !((state == STATE_OUTPUT) && m_axis_tvalid &&
                  m_axis_tready && m_axis_tlast)) begin
                state <= STATE_IDLE;
                status_busy <= 1'b0;
                status_done <= 1'b0;
                if (!status_error) begin
                    status_error <= 1'b1;
                    error_code <= ERR_TIMEOUT;
                end
                cycles <= cycles + 1;
            end else begin
                cycles <= cycles + 1;
                case (state)
                    STATE_LOAD_A: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (s_axis_tlast !=
                                ((load_outer == active_m - 1) &&
                                 (load_inner == active_k - 1))) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                if (!status_error) begin
                                    status_error <= 1'b1;
                                    error_code <= ERR_STREAM_LENGTH;
                                end
                            end else begin
                                if ((load_outer == active_m - 1) &&
                                    (load_inner == active_k - 1)) begin
                                    load_outer <= 0;
                                    load_inner <= 0;
                                    state <= STATE_LOAD_B;
                                end else if (load_inner == active_k - 1) begin
                                    load_outer <= load_outer + 1;
                                    load_inner <= 0;
                                end else begin
                                    load_inner <= load_inner + 1;
                                end
                            end
                        end
                    end
                    STATE_LOAD_B: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (s_axis_tlast !=
                                ((load_outer == active_k - 1) &&
                                 (load_inner == active_n - 1))) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                if (!status_error) begin
                                    status_error <= 1'b1;
                                    error_code <= ERR_STREAM_LENGTH;
                                end
                            end else begin
                                if ((load_outer == active_k - 1) &&
                                    (load_inner == active_n - 1)) begin
                                    load_outer <= 0;
                                    load_inner <= 0;
                                    compute_step <= 0;
                                    state <= STATE_CLEAR;
                                end else if (load_inner == active_n - 1) begin
                                    load_outer <= load_outer + 1;
                                    load_inner <= 0;
                                end else begin
                                    load_inner <= load_inner + 1;
                                end
                            end
                        end
                    end
                    STATE_CLEAR: begin
                        compute_step <= 0;
                        state <= STATE_COMPUTE;
                    end
                    STATE_COMPUTE: begin
                        if (compute_step >=
                            (active_k + active_m + active_n - 1)) begin
                            output_row_count <= 0;
                            output_column_count <= 0;
                            state <= STATE_OUTPUT;
                        end else begin
                            compute_step <= compute_step + 1;
                        end
                    end
                    STATE_OUTPUT: begin
                        if (m_axis_tvalid && m_axis_tready) begin
                            if (m_axis_tlast) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b1;
                            end else begin
                                if (output_column_count == active_n - 1) begin
                                    output_column_count <= 0;
                                    output_row_count <= output_row_count + 1;
                                end else begin
                                    output_column_count <= output_column_count + 1;
                                end
                            end
                        end
                    end
                    default: begin
                        state <= STATE_IDLE;
                        status_busy <= 1'b0;
                    end
                endcase
            end
        end else if (start_pulse) begin
            status_done <= 1'b0;
            status_error <= 1'b0;
            error_code <= 0;
            cycles <= 0;
            load_outer <= 0;
            load_inner <= 0;
            compute_step <= 0;
            output_row_count <= 0;
            output_column_count <= 0;
            if ((cfg_m < 1) || (cfg_m > ROWS) ||
                (cfg_n < 1) || (cfg_n > COLUMNS) ||
                (cfg_k < 1) || (cfg_k > MAX_K)) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_DIMENSION;
            end else if ((cfg_a_stride != cfg_k) ||
                         (cfg_b_stride != cfg_n) ||
                         (cfg_c_stride != (4 * cfg_n))) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_STRIDE;
            end else if (cfg_timeout_cycles == 0) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_TIMEOUT;
            end else begin
                active_m <= cfg_m;
                active_n <= cfg_n;
                active_k <= cfg_k;
                active_timeout <= cfg_timeout_cycles;
                status_busy <= 1'b1;
                state <= STATE_LOAD_A;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (status_busy && (state == STATE_LOAD_A) &&
            s_axis_tvalid && s_axis_tready &&
            (s_axis_tlast == ((load_outer == active_m - 1) &&
                              (load_inner == active_k - 1)))) begin
            a_buffer[load_outer*MAX_K + load_inner] <= s_axis_tdata;
        end
        if (status_busy && (state == STATE_LOAD_B) &&
            s_axis_tvalid && s_axis_tready &&
            (s_axis_tlast == ((load_outer == active_k - 1) &&
                              (load_inner == active_n - 1)))) begin
            b_buffer[load_outer*COLUMNS + load_inner] <= s_axis_tdata;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            array_a <= '0;
            array_a_valid <= '0;
            array_b <= '0;
            array_b_valid <= '0;
        end else if (soft_reset_pulse || (state != STATE_COMPUTE)) begin
            array_a <= '0;
            array_a_valid <= '0;
            array_b <= '0;
            array_b_valid <= '0;
        end else begin
            array_a <= scheduled_a;
            array_a_valid <= scheduled_a_valid;
            array_b <= scheduled_b;
            array_b_valid <= scheduled_b_valid;
        end
    end
endmodule
