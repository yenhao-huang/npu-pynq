`timescale 1ns/1ps

module npu_matrix_controller #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256,
    // Bytes per input AXI-Stream beat. Operands keep their row-major byte
    // order; each beat carries IN_BYTES consecutive bytes, lowest byte in lane 0.
    parameter integer IN_BYTES = 8
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
    input  logic [IN_BYTES*8-1:0] s_axis_tdata,
    input  logic [IN_BYTES-1:0]   s_axis_tkeep,
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
    localparam logic [31:0] ROWS_U32 = ROWS;
    localparam logic [31:0] COLUMNS_U32 = COLUMNS;
    localparam logic [31:0] MAX_K_U32 = MAX_K;
    localparam logic [31:0] IN_BYTES_U32 = IN_BYTES;
    localparam integer LANE_BITS = (IN_BYTES > 1) ? $clog2(IN_BYTES) : 1;
    localparam integer A_WORDS = (ROWS*MAX_K + IN_BYTES - 1) / IN_BYTES;
    localparam integer B_WORDS = (MAX_K*COLUMNS + IN_BYTES - 1) / IN_BYTES;
    localparam integer A_WORD_BITS = (A_WORDS > 1) ? $clog2(A_WORDS) : 1;
    localparam integer B_WORD_BITS = (B_WORDS > 1) ? $clog2(B_WORDS) : 1;

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
    // Operand stream position: the next buffer word to write and the bytes of
    // the current operand not yet received.
    logic [15:0] load_beat;
    logic [31:0] load_remaining;
    logic [31:0] active_b_bytes;
    logic [31:0] compute_step;
    // compute_step * active_n, kept incrementally so that B addressing needs no
    // variable multiplier on the compute read path.
    logic [31:0] compute_step_n;
    logic [15:0] output_row_count;
    logic [15:0] output_column_count;
    // A and B are stored in stream order, IN_BYTES bytes per word: A element
    // (r, k) at byte r*active_k + k, B element (k, c) at byte k*active_n + c.
    logic [IN_BYTES*8-1:0] a_buffer [0:A_WORDS-1];
    logic [IN_BYTES*8-1:0] b_buffer [0:B_WORDS-1];
    // Per-row and per-column address offsets, fixed for a job:
    // A byte = a_row_base[r] + compute_step, B byte = compute_step_n - b_column_offset[c].
    logic [31:0] a_row_base [0:ROWS-1];
    logic [31:0] b_column_offset [0:COLUMNS-1];
    logic beat_is_last;
    logic [IN_BYTES-1:0] beat_expected_keep;
    logic beat_ok;
    logic [31:0] a_address;
    logic [31:0] b_address;

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
    integer lane_index;

    function automatic [31:0] widen_u16;
        input [15:0] value;
        begin
            widen_u16 = {16'd0, value};
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
        if (ROWS <= 0 || COLUMNS <= 0 || MAX_K <= 0 || IN_BYTES <= 0)
            $fatal(1, "npu_matrix_controller parameters must be positive");
        if ((IN_BYTES & (IN_BYTES - 1)) != 0)
            $fatal(1, "npu_matrix_controller IN_BYTES must be a power of two");
    end

    // A beat is accepted only when TLAST and TKEEP match the bytes the current
    // operand still owes: full beats before the end, then one final beat whose
    // low lanes hold the remaining bytes.
    always_comb begin
        beat_is_last = (load_remaining <= IN_BYTES_U32);
        for (lane_index = 0; lane_index < IN_BYTES; lane_index = lane_index + 1)
            beat_expected_keep[lane_index] = (lane_index < load_remaining);
        beat_ok = (s_axis_tlast == beat_is_last) &&
            (s_axis_tkeep == beat_expected_keep);
    end

    always_comb begin
        s_axis_tready = status_busy &&
            ((state == STATE_LOAD_A) || (state == STATE_LOAD_B));
        m_axis_tvalid = status_busy && (state == STATE_OUTPUT);
        m_axis_tdata = 32'd0;
        m_axis_tlast = 1'b0;
        if (state == STATE_OUTPUT) begin
            m_axis_tdata = array_accumulators[
                (widen_u16(output_row_count) * COLUMNS +
                 widen_u16(output_column_count)) * 32 +: 32
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
        reduction_index = 0;
        a_address = 0;
        b_address = 0;
        if (state == STATE_COMPUTE) begin
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1) begin
                reduction_index = compute_step - row_index;
                if ((row_index < active_m) &&
                    (reduction_index >= 0) && (reduction_index < active_k)) begin
                    a_address = a_row_base[row_index] + compute_step;
                    scheduled_a[row_index*8 +: 8] =
                        a_buffer[a_address >> LANE_BITS][
                            a_address[LANE_BITS-1:0]*8 +: 8];
                    scheduled_a_valid[row_index] = 1'b1;
                end
            end
            for (column_index = 0; column_index < COLUMNS;
                 column_index = column_index + 1) begin
                reduction_index = compute_step - column_index;
                if ((column_index < active_n) &&
                    (reduction_index >= 0) && (reduction_index < active_k)) begin
                    b_address = compute_step_n - b_column_offset[column_index];
                    scheduled_b[column_index*8 +: 8] =
                        b_buffer[b_address >> LANE_BITS][
                            b_address[LANE_BITS-1:0]*8 +: 8];
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
            load_beat <= 0;
            load_remaining <= 0;
            active_b_bytes <= 0;
            compute_step <= 0;
            compute_step_n <= 0;
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
            load_beat <= 0;
            load_remaining <= 0;
            active_b_bytes <= 0;
            compute_step <= 0;
            compute_step_n <= 0;
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

            if (((cycles + 1) >= {32'd0, active_timeout}) &&
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
                            if (!beat_ok) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                if (!status_error) begin
                                    status_error <= 1'b1;
                                    error_code <= ERR_STREAM_LENGTH;
                                end
                            end else if (beat_is_last) begin
                                load_beat <= 0;
                                load_remaining <= active_b_bytes;
                                state <= STATE_LOAD_B;
                            end else begin
                                load_beat <= load_beat + 1;
                                load_remaining <= load_remaining - IN_BYTES_U32;
                            end
                        end
                    end
                    STATE_LOAD_B: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (!beat_ok) begin
                                state <= STATE_IDLE;
                                status_busy <= 1'b0;
                                status_done <= 1'b0;
                                if (!status_error) begin
                                    status_error <= 1'b1;
                                    error_code <= ERR_STREAM_LENGTH;
                                end
                            end else if (beat_is_last) begin
                                load_beat <= 0;
                                load_remaining <= 0;
                                compute_step <= 0;
                                compute_step_n <= 0;
                                state <= STATE_CLEAR;
                            end else begin
                                load_beat <= load_beat + 1;
                                load_remaining <= load_remaining - IN_BYTES_U32;
                            end
                        end
                    end
                    STATE_CLEAR: begin
                        compute_step <= 0;
                        compute_step_n <= 0;
                        state <= STATE_COMPUTE;
                    end
                    STATE_COMPUTE: begin
                        if (compute_step >=
                            (widen_u16(active_k) + widen_u16(active_m) +
                             widen_u16(active_n) - 32'd1)) begin
                            output_row_count <= 0;
                            output_column_count <= 0;
                            state <= STATE_OUTPUT;
                        end else begin
                            compute_step <= compute_step + 1;
                            compute_step_n <= compute_step_n + widen_u16(active_n);
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
            load_beat <= 0;
            load_remaining <= widen_u16(cfg_m) * widen_u16(cfg_k);
            active_b_bytes <= widen_u16(cfg_k) * widen_u16(cfg_n);
            compute_step <= 0;
            compute_step_n <= 0;
            output_row_count <= 0;
            output_column_count <= 0;
            if ((cfg_m < 1) || (widen_u16(cfg_m) > ROWS_U32) ||
                (cfg_n < 1) || (widen_u16(cfg_n) > COLUMNS_U32) ||
                (cfg_k < 1) || (widen_u16(cfg_k) > MAX_K_U32)) begin
                status_error <= 1'b1;
                error_code <= ERR_INVALID_DIMENSION;
            end else if ((cfg_a_stride != widen_u16(cfg_k)) ||
                         (cfg_b_stride != widen_u16(cfg_n)) ||
                         (cfg_c_stride != (32'd4 * widen_u16(cfg_n)))) begin
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

    // Whole words are written; lanes past the end of an operand hold
    // don't-care bytes that the compute schedule never addresses.
    always_ff @(posedge clk) begin
        if (status_busy && (state == STATE_LOAD_A) &&
            s_axis_tvalid && s_axis_tready && beat_ok) begin
            a_buffer[load_beat[A_WORD_BITS-1:0]] <= s_axis_tdata;
        end
        if (status_busy && (state == STATE_LOAD_B) &&
            s_axis_tvalid && s_axis_tready && beat_ok) begin
            b_buffer[load_beat[B_WORD_BITS-1:0]] <= s_axis_tdata;
        end
    end

    // Address offsets are fixed when a job starts, off the compute read path.
    always_ff @(posedge clk) begin
        if (!status_busy && start_pulse) begin
            for (int row = 0; row < ROWS; row++)
                a_row_base[row] <= row * (widen_u16(cfg_k) - 32'd1);
            for (int column = 0; column < COLUMNS; column++)
                b_column_offset[column] <=
                    column * (widen_u16(cfg_n) - 32'd1);
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
