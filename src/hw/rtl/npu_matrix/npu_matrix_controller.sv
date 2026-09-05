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
    input  logic [1:0]  cfg_job_flags,
    input  logic signed [7:0] cfg_output_zero_point,
    input  logic [7:0]  s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,
    output logic [7:0]  m_axis_tdata,
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
    localparam logic [7:0] ERR_INVALID_REQUANTIZATION = 8'd7;
    localparam logic [31:0] ROWS_U32 = ROWS;
    localparam logic [31:0] COLUMNS_U32 = COLUMNS;
    localparam logic [31:0] MAX_K_U32 = MAX_K;
    localparam integer COLUMN_INDEX_WIDTH = COLUMNS <= 1 ? 1 : $clog2(COLUMNS);

    typedef enum logic [4:0] {
        STATE_IDLE,
        STATE_LOAD_A,
        STATE_LOAD_B,
        STATE_CLEAR,
        STATE_COMPUTE,
        STATE_ACCUMULATE,
        STATE_LOAD_BIAS,
        STATE_LOAD_MULTIPLIER,
        STATE_LOAD_SHIFT,
        STATE_QUANT_BIAS,
        STATE_QUANT_MULTIPLY_PARTS,
        STATE_QUANT_MULTIPLY_ALIGN,
        STATE_QUANT_MULTIPLY_ADD,
        STATE_QUANT_MAGNITUDE,
        STATE_QUANT_ADD,
        STATE_QUANT_SHIFT,
        STATE_QUANT_SIGN,
        STATE_QUANT_OFFSET,
        STATE_OUTPUT
    } state_t;

    state_t state;
    logic [15:0] active_m, active_n, active_k;
    logic [31:0] active_timeout;
    logic [1:0] active_job_flags;
    logic signed [7:0] active_output_zero_point;
    logic [15:0] load_outer;
    logic [15:0] load_inner;
    logic [31:0] compute_step;
    logic [15:0] output_row_count;
    logic [15:0] output_column_count;
    logic [15:0] accumulate_row_count;
    logic [15:0] accumulate_column_count;
    logic accumulator_valid;
    logic [15:0] accumulator_m;
    logic [15:0] accumulator_n;
    logic signed [7:0] a_buffer [0:ROWS*MAX_K-1];
    logic signed [7:0] b_buffer [0:MAX_K*COLUMNS-1];
    logic signed [31:0] accumulator_buffer [0:ROWS*COLUMNS-1];
    logic signed [31:0] bias_buffer [0:COLUMNS-1];
    logic signed [31:0] multiplier_buffer [0:COLUMNS-1];
    logic [5:0] shift_buffer [0:COLUMNS-1];
    logic signed [32:0] quant_biased;
    logic signed [48:0] quant_product_high;
    logic signed [49:0] quant_product_low;
    logic signed [64:0] quant_product_high_shifted;
    logic signed [64:0] quant_product;
    logic quant_negative;
    logic [64:0] quant_magnitude;
    logic [64:0] quant_adjusted;
    logic [64:0] quant_shifted;
    logic signed [64:0] quant_rounded;
    logic signed [7:0] quant_output;

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

    function automatic [31:0] widen_u16;
        input [15:0] value;
        begin
            widen_u16 = {16'd0, value};
        end
    endfunction

    function automatic signed [7:0] saturate_output;
        input logic signed [64:0] rounded;
        input logic signed [7:0] zero_point;
        logic signed [65:0] shifted;
        begin
            shifted = rounded + $signed({{58{zero_point[7]}}, zero_point});
            if (shifted > 127)
                saturate_output = 8'sd127;
            else if (shifted < -128)
                saturate_output = -8'sd128;
            else
                saturate_output = shifted[7:0];
        end
    endfunction

    function automatic signed [31:0] saturating_add;
        input logic signed [31:0] left;
        input logic signed [31:0] right;
        logic signed [32:0] sum;
        begin
            sum = $signed(left) + $signed(right);
            if (sum > 33'sh07fffffff)
                saturating_add = 32'sh7fffffff;
            else if (sum < -33'sh080000000)
                saturating_add = -32'sh80000000;
            else
                saturating_add = sum[31:0];
        end
    endfunction

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
            ((state == STATE_LOAD_A) || (state == STATE_LOAD_B) ||
             (state == STATE_LOAD_BIAS) ||
             (state == STATE_LOAD_MULTIPLIER) ||
             (state == STATE_LOAD_SHIFT));
        m_axis_tvalid = status_busy && (state == STATE_OUTPUT);
        m_axis_tdata = quant_output;
        m_axis_tlast = 1'b0;
        if (state == STATE_OUTPUT) begin
            m_axis_tlast = (output_row_count == active_m - 1) &&
                (output_column_count == active_n - 1);
        end

        array_clear = (state == STATE_CLEAR);
        array_enable = (state == STATE_COMPUTE);
        scheduled_a = '0;
        scheduled_a_valid = '0;
        scheduled_b = '0;
        scheduled_b_valid = '0;
        reduction_index = 0;
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
            active_job_flags <= 0;
            active_output_zero_point <= 0;
            load_outer <= 0;
            load_inner <= 0;
            compute_step <= 0;
            output_row_count <= 0;
            output_column_count <= 0;
            accumulate_row_count <= 0;
            accumulate_column_count <= 0;
            accumulator_valid <= 1'b0;
            accumulator_m <= 0;
            accumulator_n <= 0;
            quant_biased <= 0;
            quant_product_high <= 0;
            quant_product_low <= 0;
            quant_product_high_shifted <= 0;
            quant_product <= 0;
            quant_negative <= 0;
            quant_magnitude <= 0;
            quant_adjusted <= 0;
            quant_shifted <= 0;
            quant_rounded <= 0;
            quant_output <= 0;
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
            active_job_flags <= 0;
            active_output_zero_point <= 0;
            load_outer <= 0;
            load_inner <= 0;
            compute_step <= 0;
            output_row_count <= 0;
            output_column_count <= 0;
            accumulate_row_count <= 0;
            accumulate_column_count <= 0;
            accumulator_valid <= 1'b0;
            accumulator_m <= 0;
            accumulator_n <= 0;
            quant_biased <= 0;
            quant_product_high <= 0;
            quant_product_low <= 0;
            quant_product_high_shifted <= 0;
            quant_product <= 0;
            quant_negative <= 0;
            quant_magnitude <= 0;
            quant_adjusted <= 0;
            quant_shifted <= 0;
            quant_rounded <= 0;
            quant_output <= 0;
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

            if (((cycles + 1) >= {32'd0, active_timeout}) &&
                !((state == STATE_OUTPUT) && m_axis_tvalid &&
                  m_axis_tready && m_axis_tlast)) begin
                state <= STATE_IDLE;
                status_busy <= 1'b0;
                status_done <= 1'b0;
                accumulator_valid <= 1'b0;
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
                            (widen_u16(active_k) + widen_u16(active_m) +
                             widen_u16(active_n) - 32'd1)) begin
                            accumulate_row_count <= 0;
                            accumulate_column_count <= 0;
                            state <= STATE_ACCUMULATE;
                        end else begin
                            compute_step <= compute_step + 1;
                        end
                    end
                    STATE_ACCUMULATE: begin
                        if (active_job_flags[0]) begin
                            accumulator_buffer[
                                widen_u16(accumulate_row_count) * COLUMNS +
                                widen_u16(accumulate_column_count)] <=
                                array_accumulators[
                                    (widen_u16(accumulate_row_count) * COLUMNS +
                                     widen_u16(accumulate_column_count)) * 32 +: 32];
                        end else begin
                            accumulator_buffer[
                                widen_u16(accumulate_row_count) * COLUMNS +
                                widen_u16(accumulate_column_count)] <=
                                saturating_add(
                                    accumulator_buffer[
                                        widen_u16(accumulate_row_count) * COLUMNS +
                                        widen_u16(accumulate_column_count)],
                                    array_accumulators[
                                        (widen_u16(accumulate_row_count) * COLUMNS +
                                         widen_u16(accumulate_column_count)) * 32 +: 32]);
                        end
                        if ((accumulate_row_count == active_m - 1) &&
                            (accumulate_column_count == active_n - 1)) begin
                            accumulator_valid <= 1'b1;
                            accumulator_m <= active_m;
                            accumulator_n <= active_n;
                            load_outer <= 0;
                            load_inner <= 0;
                            if (active_job_flags[1])
                                state <= STATE_LOAD_BIAS;
                            else begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b1;
                            end
                        end else if (accumulate_column_count == active_n - 1) begin
                            accumulate_column_count <= 0;
                            accumulate_row_count <= accumulate_row_count + 1;
                        end else begin
                            accumulate_column_count <= accumulate_column_count + 1;
                        end
                    end
                    STATE_LOAD_BIAS: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (s_axis_tlast != ((load_outer == active_n - 1) &&
                                                (load_inner == 3))) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                accumulator_valid <= 1'b0;
                                status_error <= 1'b1;
                                error_code <= ERR_STREAM_LENGTH;
                            end else if ((load_outer == active_n - 1) &&
                                         (load_inner == 3)) begin
                                load_outer <= 0;
                                load_inner <= 0;
                                state <= STATE_LOAD_MULTIPLIER;
                            end else if (load_inner == 3) begin
                                load_outer <= load_outer + 1;
                                load_inner <= 0;
                            end else begin
                                load_inner <= load_inner + 1;
                            end
                        end
                    end
                    STATE_LOAD_MULTIPLIER: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (s_axis_tlast != ((load_outer == active_n - 1) &&
                                                (load_inner == 3))) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                accumulator_valid <= 1'b0;
                                status_error <= 1'b1;
                                error_code <= ERR_STREAM_LENGTH;
                            end else if ((load_outer == active_n - 1) &&
                                         (load_inner == 3)) begin
                                load_outer <= 0;
                                load_inner <= 0;
                                state <= STATE_LOAD_SHIFT;
                            end else if (load_inner == 3) begin
                                load_outer <= load_outer + 1;
                                load_inner <= 0;
                            end else begin
                                load_inner <= load_inner + 1;
                            end
                        end
                    end
                    STATE_LOAD_SHIFT: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (s_axis_tdata > 31) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                accumulator_valid <= 1'b0;
                                status_error <= 1'b1;
                                error_code <= ERR_INVALID_REQUANTIZATION;
                            end else if (s_axis_tlast != (load_outer == active_n - 1)) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                accumulator_valid <= 1'b0;
                                status_error <= 1'b1;
                                error_code <= ERR_STREAM_LENGTH;
                            end else if (load_outer == active_n - 1) begin
                                output_row_count <= 0;
                                output_column_count <= 0;
                                state <= STATE_QUANT_BIAS;
                            end else begin
                                load_outer <= load_outer + 1;
                            end
                        end
                    end
                    STATE_QUANT_BIAS: begin
                        quant_biased <= $signed(accumulator_buffer[
                            widen_u16(output_row_count) * COLUMNS +
                            widen_u16(output_column_count)]) +
                            $signed(bias_buffer[
                                output_column_count[COLUMN_INDEX_WIDTH-1:0]]);
                        state <= STATE_QUANT_MULTIPLY_PARTS;
                    end
                    STATE_QUANT_MULTIPLY_PARTS: begin
                        quant_product_high <= quant_biased * $signed(
                            multiplier_buffer[
                                output_column_count[COLUMN_INDEX_WIDTH-1:0]][31:16]);
                        quant_product_low <= quant_biased * $signed({
                            1'b0,
                            multiplier_buffer[
                                output_column_count[COLUMN_INDEX_WIDTH-1:0]][15:0]
                        });
                        state <= STATE_QUANT_MULTIPLY_ALIGN;
                    end
                    STATE_QUANT_MULTIPLY_ALIGN: begin
                        quant_product_high_shifted <= $signed({
                            {16{quant_product_high[48]}}, quant_product_high
                        }) <<< 16;
                        state <= STATE_QUANT_MULTIPLY_ADD;
                    end
                    STATE_QUANT_MULTIPLY_ADD: begin
                        quant_product <= quant_product_high_shifted + $signed({
                            {15{quant_product_low[49]}}, quant_product_low
                        });
                        state <= STATE_QUANT_MAGNITUDE;
                    end
                    STATE_QUANT_MAGNITUDE: begin
                        quant_negative <= quant_product < 0;
                        quant_magnitude <= quant_product < 0 ?
                            -quant_product : quant_product;
                        state <= STATE_QUANT_ADD;
                    end
                    STATE_QUANT_ADD: begin
                        quant_adjusted <= quant_magnitude +
                            (65'd1 << (30 + shift_buffer[
                                output_column_count[COLUMN_INDEX_WIDTH-1:0]]));
                        state <= STATE_QUANT_SHIFT;
                    end
                    STATE_QUANT_SHIFT: begin
                        quant_shifted <= quant_adjusted >>
                            (31 + shift_buffer[
                                output_column_count[COLUMN_INDEX_WIDTH-1:0]]);
                        state <= STATE_QUANT_SIGN;
                    end
                    STATE_QUANT_SIGN: begin
                        quant_rounded <= quant_negative ?
                            -$signed(quant_shifted) : $signed(quant_shifted);
                        state <= STATE_QUANT_OFFSET;
                    end
                    STATE_QUANT_OFFSET: begin
                        quant_output <= saturate_output(
                            quant_rounded, active_output_zero_point
                        );
                        state <= STATE_OUTPUT;
                    end
                    STATE_OUTPUT: begin
                        if (m_axis_tvalid && m_axis_tready) begin
                            if (m_axis_tlast) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b1;
                                accumulator_valid <= 1'b0;
                            end else begin
                                if (output_column_count == active_n - 1) begin
                                    output_column_count <= 0;
                                    output_row_count <= output_row_count + 1;
                                end else begin
                                    output_column_count <= output_column_count + 1;
                                end
                                state <= STATE_QUANT_BIAS;
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
            accumulate_row_count <= 0;
            accumulate_column_count <= 0;
            if ((cfg_m < 1) || (widen_u16(cfg_m) > ROWS_U32) ||
                (cfg_n < 1) || (widen_u16(cfg_n) > COLUMNS_U32) ||
                (cfg_k < 1) || (widen_u16(cfg_k) > MAX_K_U32)) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_DIMENSION;
            end else if ((cfg_a_stride != widen_u16(cfg_k)) ||
                         (cfg_b_stride != widen_u16(cfg_n)) ||
                         (cfg_c_stride != widen_u16(cfg_n))) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_STRIDE;
            end else if (cfg_timeout_cycles == 0) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_TIMEOUT;
            end else if ((!cfg_job_flags[0] && !accumulator_valid) ||
                         (!cfg_job_flags[0] &&
                          ((cfg_m != accumulator_m) ||
                           (cfg_n != accumulator_n)))) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_REQUANTIZATION;
            end else begin
                active_m <= cfg_m;
                active_n <= cfg_n;
                active_k <= cfg_k;
                active_timeout <= cfg_timeout_cycles;
                active_job_flags <= cfg_job_flags;
                active_output_zero_point <= cfg_output_zero_point;
                if (cfg_job_flags[0])
                    accumulator_valid <= 1'b0;
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
            a_buffer[widen_u16(load_outer) * MAX_K +
                     widen_u16(load_inner)] <= s_axis_tdata;
        end
        if (status_busy && (state == STATE_LOAD_B) &&
            s_axis_tvalid && s_axis_tready &&
            (s_axis_tlast == ((load_outer == active_k - 1) &&
                              (load_inner == active_n - 1)))) begin
            b_buffer[widen_u16(load_outer) * COLUMNS +
                     widen_u16(load_inner)] <= s_axis_tdata;
        end
        if (status_busy && (state == STATE_LOAD_BIAS) &&
            s_axis_tvalid && s_axis_tready &&
            (s_axis_tlast == ((load_outer == active_n - 1) &&
                              (load_inner == 3)))) begin
            bias_buffer[load_outer[COLUMN_INDEX_WIDTH-1:0]][load_inner*8 +: 8] <= s_axis_tdata;
        end
        if (status_busy && (state == STATE_LOAD_MULTIPLIER) &&
            s_axis_tvalid && s_axis_tready &&
            (s_axis_tlast == ((load_outer == active_n - 1) &&
                              (load_inner == 3)))) begin
            multiplier_buffer[load_outer[COLUMN_INDEX_WIDTH-1:0]][load_inner*8 +: 8] <= s_axis_tdata;
        end
        if (status_busy && (state == STATE_LOAD_SHIFT) &&
            s_axis_tvalid && s_axis_tready && (s_axis_tdata <= 31) &&
            (s_axis_tlast == (load_outer == active_n - 1))) begin
            shift_buffer[load_outer[COLUMN_INDEX_WIDTH-1:0]] <= s_axis_tdata[5:0];
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
