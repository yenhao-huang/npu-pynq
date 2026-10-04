`timescale 1ns/1ps

module npu_matrix_controller #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256,
    // Bytes per input AXI-Stream beat. Operands keep their dense row-major
    // byte order; each beat carries IN_BYTES consecutive bytes, the earliest
    // in lane 0, and a frame's final beat marks its remaining bytes in TKEEP.
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
    output logic [63:0] cycles,
    output logic state_load_a,
    output logic state_load_b,
    output logic align_emit,
    output logic [15:0] align_row,
    output logic [15:0] align_word,
    output logic [IN_BYTES*8-1:0] write_data,
    output logic [31:0] compute_step,
    output logic array_clear,
    output logic array_enable,
    output logic [ROWS-1:0] array_a_valid,
    output logic [COLUMNS-1:0] array_b_valid,
    input  logic signed [ROWS*COLUMNS*32-1:0] array_accumulators
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
    localparam integer ALIGN_BYTES = 2 * IN_BYTES;
    localparam integer COUNT_BITS = $clog2(ALIGN_BYTES + 1);

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
    // Bytes of the current frame not yet accepted from the stream.
    logic [31:0] load_remaining;
    logic [31:0] active_b_bytes;
    logic [15:0] output_row_count;
    logic [15:0] output_column_count;

    // Row aligner. The stream is dense, so one beat can span a row boundary,
    // while each A bank holds one row and each B bank one column. The aligner
    // buffers up to ALIGN_BYTES stream bytes and emits one row-aligned word
    // per cycle: the next IN_BYTES bytes of the current row, or the rest of
    // the row. A frame of R rows of L bytes takes R*ceil(L/IN_BYTES) words.
    // Lanes past the end of a row carry the next row's bytes; they land in
    // bank locations the compute schedule never marks valid.
    logic [ALIGN_BYTES*8-1:0] align_buffer;
    logic [COUNT_BITS-1:0] align_count;
    logic [15:0] frame_rows, row_bytes;
    logic [31:0] row_left;
    logic [COUNT_BITS-1:0] align_need;
    logic [COUNT_BITS-1:0] count_after_emit;
    logic loading, row_done, frame_done;
    logic beat_is_last, beat_ok, beat_take;
    logic [IN_BYTES-1:0] beat_expected_keep;
    logic [IN_BYTES*8-1:0] beat_bytes;
    logic [COUNT_BITS-1:0] beat_count;
    logic [ALIGN_BYTES*8-1:0] align_buffer_next;

    logic [ROWS-1:0] scheduled_a_valid;
    logic [COLUMNS-1:0] scheduled_b_valid;

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

    initial begin
        if (ROWS <= 0 || COLUMNS <= 0 || MAX_K <= 0 || IN_BYTES <= 0)
            $fatal(1, "npu_matrix_controller parameters must be positive");
        if ((IN_BYTES & (IN_BYTES - 1)) != 0 || (MAX_K % IN_BYTES) != 0)
            $fatal(1, "npu_matrix_controller IN_BYTES must be a power of two dividing MAX_K");
    end

    assign write_data = align_buffer[IN_BYTES*8-1:0];
    assign state_load_a = (state == STATE_LOAD_A);
    assign state_load_b = (state == STATE_LOAD_B);

    always_comb begin
        loading = status_busy &&
            ((state == STATE_LOAD_A) || (state == STATE_LOAD_B));
        frame_rows = (state == STATE_LOAD_B) ? active_k : active_m;
        row_bytes = (state == STATE_LOAD_B) ? active_n : active_k;
        row_left = widen_u16(row_bytes) - (widen_u16(align_word) << LANE_BITS);
        align_need = (row_left >= IN_BYTES_U32) ?
            COUNT_BITS'(IN_BYTES) : COUNT_BITS'(row_left);
        align_emit = loading && (align_row < frame_rows) &&
            (align_count >= align_need);
        row_done = align_emit && (widen_u16({{(16-COUNT_BITS){1'b0}}, align_need}) == row_left);
        frame_done = row_done && (align_row == frame_rows - 1);
        count_after_emit = align_count - (align_emit ? align_need : '0);

        // A beat is accepted only when TLAST and TKEEP match the bytes the
        // frame still owes: full beats before the end, then one final beat
        // whose low lanes hold the remaining bytes. Input waits while the
        // aligner lacks room for a whole beat.
        s_axis_tready = loading && (load_remaining != 0) &&
            (count_after_emit <= COUNT_BITS'(IN_BYTES));
        beat_is_last = (load_remaining <= IN_BYTES_U32);
        for (lane_index = 0; lane_index < IN_BYTES; lane_index = lane_index + 1) begin
            beat_expected_keep[lane_index] = (lane_index < load_remaining);
            beat_bytes[lane_index*8 +: 8] = s_axis_tkeep[lane_index] ?
                s_axis_tdata[lane_index*8 +: 8] : 8'd0;
        end
        beat_ok = (s_axis_tlast == beat_is_last) &&
            (s_axis_tkeep == beat_expected_keep);
        beat_take = s_axis_tvalid && s_axis_tready && beat_ok;
        beat_count = beat_is_last ? COUNT_BITS'(load_remaining) : COUNT_BITS'(IN_BYTES);

        // Bytes above align_count are always zero, so a beat is ORed in.
        align_buffer_next = align_emit ?
            (align_buffer >> {align_need, 3'b000}) : align_buffer;
        if (beat_take)
            align_buffer_next = align_buffer_next |
                ({{(ALIGN_BYTES-IN_BYTES)*8{1'b0}}, beat_bytes} <<
                 {count_after_emit, 3'b000});
    end

    always_comb begin
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

        // A is complete before B loading begins. Advance the entire array only
        // when the next complete B row is resident; holding enable also holds
        // every PE and the registered boundary inputs across stream stalls.
        array_clear = (state == STATE_LOAD_A);
        array_enable = (state == STATE_COMPUTE) ||
            ((state == STATE_LOAD_B) && (compute_step < widen_u16(align_row)));
        scheduled_a_valid = '0;
        scheduled_b_valid = '0;
        reduction_index = 0;
        if ((state == STATE_LOAD_B) || (state == STATE_COMPUTE)) begin
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1) begin
                reduction_index = compute_step - row_index;
                if ((row_index < active_m) &&
                    (reduction_index >= 0) && (reduction_index < active_k)) begin
                    scheduled_a_valid[row_index] = 1'b1;
                end
            end
            for (column_index = 0; column_index < COLUMNS;
                 column_index = column_index + 1) begin
                reduction_index = compute_step - column_index;
                if ((column_index < active_n) &&
                    (reduction_index >= 0) && (reduction_index < active_k)) begin
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
            load_remaining <= 0;
            active_b_bytes <= 0;
            align_buffer <= '0;
            align_count <= 0;
            align_row <= 0;
            align_word <= 0;
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
            load_remaining <= 0;
            active_b_bytes <= 0;
            align_buffer <= '0;
            align_count <= 0;
            align_row <= 0;
            align_word <= 0;
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
                    STATE_LOAD_A, STATE_LOAD_B: begin
                        if ((state == STATE_LOAD_B) && array_enable)
                            compute_step <= compute_step + 1;
                        if (s_axis_tvalid && s_axis_tready && !beat_ok) begin
                            state <= STATE_IDLE;
                            status_busy <= 1'b0;
                            status_done <= 1'b0;
                            if (!status_error) begin
                                status_error <= 1'b1;
                                error_code <= ERR_STREAM_LENGTH;
                            end
                        end else begin
                            align_buffer <= align_buffer_next;
                            align_count <= count_after_emit +
                                (beat_take ? beat_count : '0);
                            if (beat_take)
                                load_remaining <= beat_is_last ? 32'd0 :
                                    load_remaining - IN_BYTES_U32;
                            if (row_done) begin
                                align_word <= 0;
                                align_row <= align_row + 1;
                            end else if (align_emit) begin
                                align_word <= align_word + 1;
                            end
                            // The whole frame was accepted before its last
                            // word, so nothing else is in flight here.
                            if (frame_done) begin
                                align_row <= 0;
                                align_word <= 0;
                                if (state == STATE_LOAD_A) begin
                                    load_remaining <= active_b_bytes;
                                    state <= STATE_LOAD_B;
                                end else begin
                                    state <= STATE_CLEAR;
                                end
                            end
                        end
                    end
                    STATE_CLEAR: begin
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
            load_remaining <= widen_u16(cfg_m) * widen_u16(cfg_k);
            active_b_bytes <= widen_u16(cfg_k) * widen_u16(cfg_n);
            align_buffer <= '0;
            align_count <= 0;
            align_row <= 0;
            align_word <= 0;
            compute_step <= 0;
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            array_a_valid <= '0;
            array_b_valid <= '0;
        end else if (soft_reset_pulse || (state == STATE_IDLE) ||
                     (state == STATE_LOAD_A)) begin
            array_a_valid <= '0;
            array_b_valid <= '0;
        end else if (array_enable) begin
            array_a_valid <= scheduled_a_valid;
            array_b_valid <= scheduled_b_valid;
        end
    end
endmodule
