`timescale 1ns/1ps

// Matrix job controller with an A/B ping-pong buffer.
//
// Two state machines share a two-entry job queue:
//
//   load engine : drains s_axis through the row aligner into the operand-bank
//                 half named by load_ptr
//   exec engine : clears, walks the systolic wavefront over the half named by
//                 exec_ptr, and drains m_axis
//
// The datapath's operand banks are two halves deep; load_half and exec_half
// select the half each engine touches, so the next job's operands stream in
// while the current job computes and outputs. The exec engine starts as soon
// as a job's A frame has landed and advances the wavefront while that job's B
// rows still arrive. A job is accepted whenever a queue entry is free; START
// with both entries occupied still raises ERR_BUSY_START.
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
    output logic        status_accept,
    output logic        status_done,
    output logic        status_error,
    output logic [7:0]  error_code,
    output logic [63:0] cycles,
    output logic state_load_a,
    output logic state_load_b,
    output logic load_half,
    output logic exec_half,
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

    typedef enum logic [1:0] {
        LOAD_IDLE,
        LOAD_A,
        LOAD_B
    } load_state_t;

    typedef enum logic [1:0] {
        EXEC_IDLE,
        EXEC_CLEAR,
        EXEC_COMPUTE,
        EXEC_OUTPUT
    } exec_state_t;

    load_state_t load_state;
    exec_state_t exec_state;

    // Two-entry job queue. alloc_ptr takes accepted jobs, load_ptr owns the
    // half being written, exec_ptr owns the half being read. slot_a_loaded
    // lets the exec engine start; slot_loaded marks the whole B frame landed.
    logic        slot_valid    [0:1];
    logic        slot_a_loaded [0:1];
    logic        slot_loaded   [0:1];
    logic [15:0] slot_m [0:1];
    logic [15:0] slot_n [0:1];
    logic [15:0] slot_k [0:1];
    logic [31:0] slot_timeout [0:1];
    logic [63:0] slot_cycles [0:1];
    logic        alloc_ptr, load_ptr, exec_ptr;
    logic [1:0]  occupancy;

    logic [15:0] active_m, active_n, active_k;
    logic [15:0] load_m, load_n, load_k;
    // Bytes of the current frame not yet accepted from the stream.
    logic [31:0] load_remaining;
    logic [15:0] output_row_count;
    logic [15:0] output_column_count;
    logic [63:0] span_cycles;

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

    logic retire_now, timeout_hit, start_accepted, start_invalid;
    logic exec_b_ready;

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

    assign load_m = slot_m[load_ptr];
    assign load_n = slot_n[load_ptr];
    assign load_k = slot_k[load_ptr];
    assign status_busy = (occupancy != 2'd0);
    assign status_accept = (occupancy != 2'd2);
    assign write_data = align_buffer[IN_BYTES*8-1:0];
    assign state_load_a = (load_state == LOAD_A);
    assign state_load_b = (load_state == LOAD_B);
    assign load_half = load_ptr;
    assign exec_half = exec_ptr;

    always_comb begin
        loading = (load_state == LOAD_A) || (load_state == LOAD_B);
        frame_rows = (load_state == LOAD_B) ? load_k : load_m;
        row_bytes = (load_state == LOAD_B) ? load_n : load_k;
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
        m_axis_tvalid = (exec_state == EXEC_OUTPUT);
        m_axis_tdata = 32'd0;
        m_axis_tlast = 1'b0;
        if (exec_state == EXEC_OUTPUT) begin
            m_axis_tdata = array_accumulators[
                (widen_u16(output_row_count) * COLUMNS +
                 widen_u16(output_column_count)) * 32 +: 32
            ];
            m_axis_tlast = (output_row_count == active_m - 1) &&
                (output_column_count == active_n - 1);
        end

        retire_now = (exec_state == EXEC_OUTPUT) && m_axis_tvalid &&
            m_axis_tready && m_axis_tlast;

        // Every outstanding job runs its own timeout, including one that is
        // still queued behind the array. The job retiring this cycle is exempt,
        // matching the serialized behaviour on the final output beat.
        timeout_hit =
            (slot_valid[0] &&
             ((slot_cycles[0] + 64'd1) >= {32'd0, slot_timeout[0]}) &&
             !(retire_now && (exec_ptr == 1'b0))) ||
            (slot_valid[1] &&
             ((slot_cycles[1] + 64'd1) >= {32'd0, slot_timeout[1]}) &&
             !(retire_now && (exec_ptr == 1'b1)));

        start_invalid = (cfg_m < 1) || (widen_u16(cfg_m) > ROWS_U32) ||
            (cfg_n < 1) || (widen_u16(cfg_n) > COLUMNS_U32) ||
            (cfg_k < 1) || (widen_u16(cfg_k) > MAX_K_U32) ||
            (cfg_a_stride != widen_u16(cfg_k)) ||
            (cfg_b_stride != widen_u16(cfg_n)) ||
            (cfg_c_stride != (32'd4 * widen_u16(cfg_n))) ||
            (cfg_timeout_cycles == 32'd0);

        start_accepted = start_pulse && status_accept && !start_invalid &&
            !(status_busy && status_error);

        // A is complete before the exec engine starts. While the same job's
        // B frame is still loading, advance the whole array only once the next
        // complete B row is resident; holding enable holds every PE and the
        // registered boundary inputs across stream stalls.
        exec_b_ready = slot_loaded[exec_ptr] ||
            ((load_state == LOAD_B) && (load_ptr == exec_ptr) &&
             (compute_step < widen_u16(align_row)));
        array_clear = (exec_state == EXEC_CLEAR);
        array_enable = (exec_state == EXEC_COMPUTE) && exec_b_ready;
        scheduled_a_valid = '0;
        scheduled_b_valid = '0;
        reduction_index = 0;
        if (exec_state == EXEC_COMPUTE) begin
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
            load_state <= LOAD_IDLE;
            exec_state <= EXEC_IDLE;
            slot_valid[0] <= 1'b0;
            slot_valid[1] <= 1'b0;
            slot_a_loaded[0] <= 1'b0;
            slot_a_loaded[1] <= 1'b0;
            slot_loaded[0] <= 1'b0;
            slot_loaded[1] <= 1'b0;
            slot_m[0] <= 16'd0; slot_m[1] <= 16'd0;
            slot_n[0] <= 16'd0; slot_n[1] <= 16'd0;
            slot_k[0] <= 16'd0; slot_k[1] <= 16'd0;
            slot_timeout[0] <= 32'd0; slot_timeout[1] <= 32'd0;
            slot_cycles[0] <= 64'd0; slot_cycles[1] <= 64'd0;
            alloc_ptr <= 1'b0;
            load_ptr <= 1'b0;
            exec_ptr <= 1'b0;
            occupancy <= 2'd0;
            active_m <= 16'd0;
            active_n <= 16'd0;
            active_k <= 16'd0;
            load_remaining <= 32'd0;
            align_buffer <= '0;
            align_count <= '0;
            align_row <= 16'd0;
            align_word <= 16'd0;
            compute_step <= 32'd0;
            output_row_count <= 16'd0;
            output_column_count <= 16'd0;
            span_cycles <= 64'd0;
            status_done <= 1'b0;
            status_error <= 1'b0;
            error_code <= 8'd0;
            cycles <= 64'd0;
        end else if (soft_reset_pulse) begin
            // SOFT_RESET discards both halves, the in-flight load, and the
            // queue. Bank contents are left as-is; they are only read for a
            // job whose A frame completed.
            load_state <= LOAD_IDLE;
            exec_state <= EXEC_IDLE;
            slot_valid[0] <= 1'b0;
            slot_valid[1] <= 1'b0;
            slot_a_loaded[0] <= 1'b0;
            slot_a_loaded[1] <= 1'b0;
            slot_loaded[0] <= 1'b0;
            slot_loaded[1] <= 1'b0;
            slot_cycles[0] <= 64'd0;
            slot_cycles[1] <= 64'd0;
            alloc_ptr <= 1'b0;
            load_ptr <= 1'b0;
            exec_ptr <= 1'b0;
            occupancy <= 2'd0;
            active_m <= 16'd0;
            active_n <= 16'd0;
            active_k <= 16'd0;
            load_remaining <= 32'd0;
            align_buffer <= '0;
            align_count <= '0;
            align_row <= 16'd0;
            align_word <= 16'd0;
            compute_step <= 32'd0;
            output_row_count <= 16'd0;
            output_column_count <= 16'd0;
            span_cycles <= 64'd0;
            status_done <= 1'b0;
            status_error <= 1'b0;
            error_code <= 8'd0;
            cycles <= 64'd0;
        end else begin
            // ---------------------------------------------------------------
            // Per-job and span counters
            // ---------------------------------------------------------------
            if (status_busy) begin
                span_cycles <= span_cycles + 64'd1;
                if (slot_valid[0]) slot_cycles[0] <= slot_cycles[0] + 64'd1;
                if (slot_valid[1]) slot_cycles[1] <= slot_cycles[1] + 64'd1;
            end

            // ---------------------------------------------------------------
            // START handling
            // ---------------------------------------------------------------
            if (start_pulse) begin
                if (!status_accept) begin
                    if (!status_error) begin
                        status_error <= 1'b1;
                        error_code <= ERR_BUSY_START;
                    end
                end else if (status_busy && status_error) begin
                    // An error is latched and the pipeline is draining; ignore.
                end else if (start_invalid) begin
                    status_done <= 1'b0;
                    status_error <= 1'b1;
                    error_code <= (((cfg_m < 1) || (widen_u16(cfg_m) > ROWS_U32) ||
                                    (cfg_n < 1) || (widen_u16(cfg_n) > COLUMNS_U32) ||
                                    (cfg_k < 1) || (widen_u16(cfg_k) > MAX_K_U32))
                                   ? ERR_INVALID_DIMENSION
                                   : (((cfg_a_stride != widen_u16(cfg_k)) ||
                                       (cfg_b_stride != widen_u16(cfg_n)) ||
                                       (cfg_c_stride != (32'd4 * widen_u16(cfg_n))))
                                      ? ERR_INVALID_STRIDE
                                      : ERR_INVALID_TIMEOUT));
                    // A rejected job quiesces whatever is in flight.
                    load_state <= LOAD_IDLE;
                    exec_state <= EXEC_IDLE;
                    slot_valid[0] <= 1'b0;
                    slot_valid[1] <= 1'b0;
                    slot_a_loaded[0] <= 1'b0;
                    slot_a_loaded[1] <= 1'b0;
                    slot_loaded[0] <= 1'b0;
                    slot_loaded[1] <= 1'b0;
                    slot_cycles[0] <= 64'd0;
                    slot_cycles[1] <= 64'd0;
                    alloc_ptr <= 1'b0;
                    load_ptr <= 1'b0;
                    exec_ptr <= 1'b0;
                    occupancy <= 2'd0;
                    span_cycles <= 64'd0;
                    cycles <= 64'd0;
                end else begin
                    if (!status_busy) begin
                        // A fresh pipeline clears the previous outcome.
                        status_done <= 1'b0;
                        status_error <= 1'b0;
                        error_code <= 8'd0;
                        cycles <= 64'd0;
                        span_cycles <= 64'd0;
                    end
                    slot_valid[alloc_ptr] <= 1'b1;
                    slot_a_loaded[alloc_ptr] <= 1'b0;
                    slot_loaded[alloc_ptr] <= 1'b0;
                    slot_m[alloc_ptr] <= cfg_m;
                    slot_n[alloc_ptr] <= cfg_n;
                    slot_k[alloc_ptr] <= cfg_k;
                    slot_timeout[alloc_ptr] <= cfg_timeout_cycles;
                    slot_cycles[alloc_ptr] <= 64'd0;
                    alloc_ptr <= ~alloc_ptr;
                    occupancy <= occupancy + 2'd1;
                    // Hand straight to an idle load engine so the A frame opens
                    // on the cycle after START.
                    if ((load_state == LOAD_IDLE) && (load_ptr == alloc_ptr)) begin
                        load_state <= LOAD_A;
                        load_remaining <= widen_u16(cfg_m) * widen_u16(cfg_k);
                        align_buffer <= '0;
                        align_count <= '0;
                        align_row <= 16'd0;
                        align_word <= 16'd0;
                    end
                end
            end

            // ---------------------------------------------------------------
            // Load engine
            // ---------------------------------------------------------------
            case (load_state)
                LOAD_IDLE: begin
                    if (slot_valid[load_ptr] && !slot_a_loaded[load_ptr]) begin
                        load_state <= LOAD_A;
                        load_remaining <= widen_u16(load_m) * widen_u16(load_k);
                        align_buffer <= '0;
                        align_count <= '0;
                        align_row <= 16'd0;
                        align_word <= 16'd0;
                    end
                end
                LOAD_A, LOAD_B: begin
                    if (s_axis_tvalid && s_axis_tready && !beat_ok) begin
                        status_done <= 1'b0;
                        if (!status_error) begin
                            status_error <= 1'b1;
                            error_code <= ERR_STREAM_LENGTH;
                        end
                        load_state <= LOAD_IDLE;
                        exec_state <= EXEC_IDLE;
                        slot_valid[0] <= 1'b0;
                        slot_valid[1] <= 1'b0;
                        slot_a_loaded[0] <= 1'b0;
                        slot_a_loaded[1] <= 1'b0;
                        slot_loaded[0] <= 1'b0;
                        slot_loaded[1] <= 1'b0;
                        slot_cycles[0] <= 64'd0;
                        slot_cycles[1] <= 64'd0;
                        alloc_ptr <= 1'b0;
                        load_ptr <= 1'b0;
                        exec_ptr <= 1'b0;
                        occupancy <= 2'd0;
                        cycles <= span_cycles + 64'd1;
                        span_cycles <= 64'd0;
                    end else begin
                        align_buffer <= align_buffer_next;
                        align_count <= count_after_emit +
                            (beat_take ? beat_count : '0);
                        if (beat_take)
                            load_remaining <= beat_is_last ? 32'd0 :
                                load_remaining - IN_BYTES_U32;
                        if (row_done) begin
                            align_word <= 16'd0;
                            align_row <= align_row + 16'd1;
                        end else if (align_emit) begin
                            align_word <= align_word + 16'd1;
                        end
                        // The whole frame was accepted before its last word,
                        // so nothing else is in flight here.
                        if (frame_done) begin
                            align_row <= 16'd0;
                            align_word <= 16'd0;
                            if (load_state == LOAD_A) begin
                                slot_a_loaded[load_ptr] <= 1'b1;
                                load_remaining <= widen_u16(load_k) * widen_u16(load_n);
                                load_state <= LOAD_B;
                                // Hand to an idle exec engine on the same cycle,
                                // so compute overlaps this job's own B frame.
                                if ((exec_state == EXEC_IDLE) && (exec_ptr == load_ptr)) begin
                                    exec_state <= EXEC_CLEAR;
                                    active_m <= load_m;
                                    active_n <= load_n;
                                    active_k <= load_k;
                                    compute_step <= 32'd0;
                                end
                            end else begin
                                slot_loaded[load_ptr] <= 1'b1;
                                load_ptr <= ~load_ptr;
                                // Open the queued job's A frame with no bubble.
                                if (slot_valid[~load_ptr] && !slot_a_loaded[~load_ptr]) begin
                                    load_state <= LOAD_A;
                                    load_remaining <= widen_u16(slot_m[~load_ptr]) *
                                        widen_u16(slot_k[~load_ptr]);
                                end else begin
                                    load_state <= LOAD_IDLE;
                                end
                            end
                        end
                    end
                end
                default: load_state <= LOAD_IDLE;
            endcase

            // ---------------------------------------------------------------
            // Exec engine
            // ---------------------------------------------------------------
            case (exec_state)
                EXEC_IDLE: begin
                    if (slot_valid[exec_ptr] && slot_a_loaded[exec_ptr]) begin
                        exec_state <= EXEC_CLEAR;
                        active_m <= slot_m[exec_ptr];
                        active_n <= slot_n[exec_ptr];
                        active_k <= slot_k[exec_ptr];
                        compute_step <= 32'd0;
                    end
                end
                EXEC_CLEAR: begin
                    compute_step <= 32'd0;
                    exec_state <= EXEC_COMPUTE;
                end
                EXEC_COMPUTE: begin
                    if (compute_step >=
                        (widen_u16(active_k) + widen_u16(active_m) +
                         widen_u16(active_n) - 32'd1)) begin
                        output_row_count <= 16'd0;
                        output_column_count <= 16'd0;
                        exec_state <= EXEC_OUTPUT;
                    end else if (array_enable) begin
                        compute_step <= compute_step + 32'd1;
                    end
                end
                EXEC_OUTPUT: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        if (m_axis_tlast) begin
                            slot_valid[exec_ptr] <= 1'b0;
                            slot_a_loaded[exec_ptr] <= 1'b0;
                            slot_loaded[exec_ptr] <= 1'b0;
                            slot_cycles[exec_ptr] <= 64'd0;
                            exec_ptr <= ~exec_ptr;
                            occupancy <= start_accepted ? occupancy : (occupancy - 2'd1);
                            status_done <= 1'b1;
                            // `cycles` reports this job's issue interval: the
                            // cycles since the previous retirement, or since the
                            // pipeline went busy for the first job.
                            cycles <= span_cycles + 64'd1;
                            span_cycles <= 64'd0;
                            // Chain straight into a job whose A frame already
                            // landed, so back-to-back jobs cost no bubble.
                            if (slot_valid[~exec_ptr] && slot_a_loaded[~exec_ptr]) begin
                                exec_state <= EXEC_CLEAR;
                                active_m <= slot_m[~exec_ptr];
                                active_n <= slot_n[~exec_ptr];
                                active_k <= slot_k[~exec_ptr];
                                compute_step <= 32'd0;
                            end else begin
                                exec_state <= EXEC_IDLE;
                            end
                        end else if (output_column_count == active_n - 1) begin
                            output_column_count <= 16'd0;
                            output_row_count <= output_row_count + 16'd1;
                        end else begin
                            output_column_count <= output_column_count + 16'd1;
                        end
                    end
                end
                default: exec_state <= EXEC_IDLE;
            endcase

            // ---------------------------------------------------------------
            // Timeout has the last word and drops every outstanding job.
            // ---------------------------------------------------------------
            if (status_busy && timeout_hit) begin
                load_state <= LOAD_IDLE;
                exec_state <= EXEC_IDLE;
                slot_valid[0] <= 1'b0;
                slot_valid[1] <= 1'b0;
                slot_a_loaded[0] <= 1'b0;
                slot_a_loaded[1] <= 1'b0;
                slot_loaded[0] <= 1'b0;
                slot_loaded[1] <= 1'b0;
                slot_cycles[0] <= 64'd0;
                slot_cycles[1] <= 64'd0;
                alloc_ptr <= 1'b0;
                load_ptr <= 1'b0;
                exec_ptr <= 1'b0;
                occupancy <= 2'd0;
                status_done <= 1'b0;
                if (!status_error) begin
                    status_error <= 1'b1;
                    error_code <= ERR_TIMEOUT;
                end
                cycles <= span_cycles + 64'd1;
                span_cycles <= 64'd0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            array_a_valid <= '0;
            array_b_valid <= '0;
        end else if (soft_reset_pulse || (exec_state == EXEC_IDLE) ||
                     (exec_state == EXEC_CLEAR)) begin
            array_a_valid <= '0;
            array_b_valid <= '0;
        end else if (array_enable) begin
            array_a_valid <= scheduled_a_valid;
            array_b_valid <= scheduled_b_valid;
        end
    end
endmodule
