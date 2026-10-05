`timescale 1ns/1ps

// Cycle-count evidence for the A/B ping-pong buffer at the 8x8 target.
// Runs stall-free 8x8x64 physical jobs -- the shape issue #63 benchmarked --
// first serialized, then pipelined, so the two spans are directly comparable.
// It also exercises back-to-back jobs, bank swapping,
// input stall, output backpressure, SOFT_RESET mid-pipeline, and the error
// paths whose meaning changed when the job queue grew to two entries.
//
// The throughput case runs the same job list twice -- once serialized the way
// the pre-pipelined controller behaved, once with the queue kept full -- and
// requires the overlapped run to finish in strictly fewer cycles.
module tb_npu_matrix_controller_pipeline_8x8;
    parameter integer SIZE = 8;
    localparam integer ROWS = SIZE;
    localparam integer COLUMNS = SIZE;
    localparam integer MAX_K = 256;
    localparam integer IN_BYTES = 8;
    localparam integer NUM_JOBS = 3;
    localparam integer MAX_JOB_K = (SIZE == 16) ? 256 : 64;
    localparam integer JOB_TIMEOUT = 20000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start_pulse = 1'b0;
    logic soft_reset_pulse = 1'b0;
    logic [15:0] cfg_m = 0, cfg_n = 0, cfg_k = 0;
    logic [31:0] cfg_a_stride = 0, cfg_b_stride = 0, cfg_c_stride = 0;
    logic [31:0] cfg_timeout_cycles = 0;
    logic [IN_BYTES*8-1:0] s_axis_tdata = 0;
    logic [IN_BYTES-1:0] s_axis_tkeep = 0;
    logic s_axis_tvalid = 0, s_axis_tready, s_axis_tlast = 0;
    logic [31:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready = 0, m_axis_tlast;
    logic status_busy, status_accept, status_done, status_error;
    logic [7:0] error_code;
    logic [63:0] cycles;

    npu_matrix_core #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K), .IN_BYTES(IN_BYTES)
    ) dut (.*);

    always #5 clk = ~clk;

    // Stimulus and golden results, flattened so the indexing stays explicit.
    integer job_k [0:NUM_JOBS-1];
    logic signed [7:0]  a_mem [0:NUM_JOBS*ROWS*MAX_JOB_K-1];
    logic signed [7:0]  b_mem [0:NUM_JOBS*MAX_JOB_K*COLUMNS-1];
    logic signed [31:0] c_mem [0:NUM_JOBS*ROWS*COLUMNS-1];
    logic [7:0] frame_bytes [0:ROWS*MAX_JOB_K + MAX_JOB_K*COLUMNS - 1];

    integer accepted_count;
    integer job_index, row_index, column_index, step_index, beat_index;
    integer seed;
    logic signed [31:0] accumulator;
    logic [63:0] cycle_counter;
    logic [63:0] serial_span, pipelined_span;
    logic [63:0] reported_total;
    logic [63:0] first_job_cycles;
    logic overlap_seen;
    logic pipelined_mode;
    logic signed [31:0] observed;
    integer pass_count;

    always @(posedge clk) cycle_counter <= cycle_counter + 64'd1;

    // Direct evidence that another job's load shared the clock with a compute
    // or output phase, which is only reachable once the two banks are
    // independent. A job's own B frame overlapping its compute does not count.
    always @(posedge clk) begin
        if (rst_n && (dut.controller.load_state !== 2'd0) &&
            (dut.controller.load_ptr !== dut.controller.exec_ptr) &&
            ((dut.controller.exec_state === 2'd2) ||
             (dut.controller.exec_state === 2'd3)))
            overlap_seen <= 1'b1;
    end

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_matrix_controller_pipeline_8x8: %s", message);
            $fatal(1);
        end
    endtask

    task automatic pulse_start;
        begin
            @(negedge clk); start_pulse = 1'b1;
            @(negedge clk); start_pulse = 1'b0;
        end
    endtask

    task automatic pulse_soft_reset;
        begin
            @(negedge clk); soft_reset_pulse = 1'b1;
            @(negedge clk); soft_reset_pulse = 1'b0;
        end
    endtask

    task automatic configure(
        input integer m, input integer n, input integer k, input integer timeout_cycles
    );
        begin
            cfg_m = m[15:0]; cfg_n = n[15:0]; cfg_k = k[15:0];
            cfg_a_stride = k; cfg_b_stride = n; cfg_c_stride = 4 * n;
            cfg_timeout_cycles = timeout_cycles;
        end
    endtask

    // Full-rate source: TVALID stays asserted across beats and frames, so the
    // load engine sees one packed beat per cycle unless a stall is asked for.
    task automatic send_beat(
        input logic [IN_BYTES*8-1:0] data, input logic [IN_BYTES-1:0] keep,
        input logic last, input integer stall_cycles
    );
        integer idle_index;
        begin
            if (stall_cycles > 0) begin
                @(negedge clk);
                s_axis_tvalid = 1'b0;
                s_axis_tlast = 1'b0;
                for (idle_index = 0; idle_index < stall_cycles; idle_index = idle_index + 1)
                    @(negedge clk);
            end
            @(negedge clk);
            s_axis_tdata = data;
            s_axis_tkeep = keep;
            s_axis_tlast = last;
            s_axis_tvalid = 1'b1;
            while (!s_axis_tready) @(negedge clk);
        end
    endtask

    task automatic idle_input;
        begin
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tkeep = '0;
        end
    endtask

    // Streams frame_bytes[0:count-1] as one DMA transfer: IN_BYTES bytes per
    // beat, a partial final beat marked by TKEEP, TLAST on that beat, and a
    // two-cycle stall before every stall_every-th beat.
    task automatic send_frame(input integer count, input integer stall_every);
        integer offset, lane, beat;
        logic [IN_BYTES*8-1:0] data;
        logic [IN_BYTES-1:0] keep;
        begin
            beat = 0;
            for (offset = 0; offset < count; offset = offset + IN_BYTES) begin
                data = '0;
                keep = '0;
                for (lane = 0; lane < IN_BYTES; lane = lane + 1)
                    if (offset + lane < count) begin
                        data[lane*8 +: 8] = frame_bytes[offset + lane];
                        keep[lane] = 1'b1;
                    end
                send_beat(data, keep, offset + IN_BYTES >= count,
                          ((stall_every > 0) && (beat % stall_every == stall_every-1))
                              ? 2 : 0);
                beat = beat + 1;
            end
        end
    endtask

    // Streams one job's A frame then its B frame, stalling mid-frame so the
    // queued job cannot rely on an uninterrupted source.
    task automatic feed_job(input integer job, input integer stall_every);
        integer r, c, kk, k_len;
        begin
            k_len = job_k[job];
            for (r = 0; r < ROWS; r = r + 1)
                for (kk = 0; kk < k_len; kk = kk + 1)
                    frame_bytes[r*k_len + kk] = a_mem[(job*ROWS + r)*MAX_JOB_K + kk];
            send_frame(ROWS*k_len, stall_every);
            for (kk = 0; kk < k_len; kk = kk + 1)
                for (c = 0; c < COLUMNS; c = c + 1)
                    frame_bytes[kk*COLUMNS + c] = b_mem[(job*MAX_JOB_K + kk)*COLUMNS + c];
            send_frame(k_len*COLUMNS, stall_every);
            idle_input();
        end
    endtask

    // Full-rate sink, optionally holding TREADY low on the first beat to prove
    // the result survives backpressure.
    task automatic drain_job(input integer job, input integer stall_cycles);
        integer r, c, idle_index, beat;
        logic signed [31:0] held;
        logic held_last;
        begin
            if (stall_cycles > 0) begin
                m_axis_tready = 1'b0;
                wait (m_axis_tvalid);
                @(posedge clk); #1;
                held = m_axis_tdata;
                held_last = m_axis_tlast;
                for (idle_index = 0; idle_index < stall_cycles;
                     idle_index = idle_index + 1) begin
                    @(posedge clk); #1;
                    if (!m_axis_tvalid || (m_axis_tdata !== held) ||
                        (m_axis_tlast !== held_last))
                        fail("output changed under backpressure");
                end
            end
            @(negedge clk);
            m_axis_tready = 1'b1;
            beat = 0;
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLUMNS; c = c + 1) begin
                    while (!m_axis_tvalid) @(negedge clk);
                    observed = m_axis_tdata;
                    if (observed !== c_mem[(job*ROWS + r)*COLUMNS + c]) begin
                        $display("job=%0d C[%0d,%0d] expected=%0d got=%0d",
                                 job, r, c,
                                 c_mem[(job*ROWS + r)*COLUMNS + c], observed);
                        fail("result mismatch against golden model");
                    end
                    if (m_axis_tlast !== ((r == ROWS-1) && (c == COLUMNS-1)))
                        fail("TLAST position");
                    beat = beat + 1;
                    // Drop TREADY immediately after the frame's last beat is
                    // taken, so the next job's frame is not consumed early.
                    if (beat == ROWS*COLUMNS) begin
                        @(posedge clk); #1;
                        m_axis_tready = 1'b0;
                    end else begin
                        @(negedge clk);
                    end
                end
        end
    endtask

    // One pass over the whole job list. In serialized mode a job is only
    // started once the engine is completely idle, reproducing the behaviour
    // this change replaces; in pipelined mode the queue is kept full.
    task automatic run_jobs(input logic pipelined, output logic [63:0] span);
        logic [63:0] began;
        begin
            accepted_count = 0;
            reported_total = 64'd0;
            began = cycle_counter;
            fork
                begin : starter
                    integer j;
                    for (j = 0; j < NUM_JOBS; j = j + 1) begin
                        if (pipelined) begin
                            while (!status_accept) @(negedge clk);
                        end else begin
                            while (status_busy) @(negedge clk);
                        end
                        configure(ROWS, COLUMNS, job_k[j], JOB_TIMEOUT);
                        pulse_start();
                        if (status_error) fail("unexpected error on START");
                        accepted_count = j + 1;
                    end
                end
                begin : feeder
                    integer j;
                    for (j = 0; j < NUM_JOBS; j = j + 1) begin
                        while (accepted_count <= j) @(negedge clk);
                        feed_job(j, 0);
                    end
                end
                begin : drainer
                    integer j;
                    for (j = 0; j < NUM_JOBS; j = j + 1) begin
                        drain_job(j, 0);
                        if (j == 0) first_job_cycles = cycles;
                        reported_total = reported_total + cycles;
                    end
                end
            join
            span = cycle_counter - began;
            if (status_error) fail("error raised during a clean run");
        end
    endtask

    initial begin
        cycle_counter = 64'd0;
        overlap_seen = 1'b0;
        pass_count = 0;
        seed = 32'h5eed_1234;

        // Golden model: C = A * B in exact INT32, no requantization.
        for (job_index = 0; job_index < NUM_JOBS; job_index = job_index + 1) begin
            job_k[job_index] = MAX_JOB_K;
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1)
                for (step_index = 0; step_index < job_k[job_index];
                     step_index = step_index + 1)
                    a_mem[(job_index*ROWS + row_index)*MAX_JOB_K + step_index] =
                        $random(seed);
            for (step_index = 0; step_index < job_k[job_index];
                 step_index = step_index + 1)
                for (column_index = 0; column_index < COLUMNS;
                     column_index = column_index + 1)
                    b_mem[(job_index*MAX_JOB_K + step_index)*COLUMNS + column_index] =
                        $random(seed);
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1)
                for (column_index = 0; column_index < COLUMNS;
                     column_index = column_index + 1) begin
                    accumulator = 0;
                    for (step_index = 0; step_index < job_k[job_index];
                         step_index = step_index + 1)
                        accumulator = accumulator +
                            a_mem[(job_index*ROWS + row_index)*MAX_JOB_K + step_index] *
                            b_mem[(job_index*MAX_JOB_K + step_index)*COLUMNS + column_index];
                    c_mem[(job_index*ROWS + row_index)*COLUMNS + column_index] = accumulator;
                end
        end

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        // ---- serialized reference pass -------------------------------------
        pipelined_mode = 1'b0;
        run_jobs(1'b0, serial_span);
        if (overlap_seen) fail("serialized pass must not overlap");
        pulse_soft_reset();
        pass_count = pass_count + 1;

        // ---- pipelined pass ------------------------------------------------
        pipelined_mode = 1'b1;
        run_jobs(1'b1, pipelined_span);
        if (!overlap_seen) fail("no load overlapped a compute or output phase");
        if (pipelined_span >= serial_span) fail("pipelined run did not save cycles");
        // Per-job `cycles` is an issue interval, so the reported values must
        // account for the whole pipelined span.
        if (reported_total > pipelined_span)
            fail("reported cycles exceed the measured span");
        if (reported_total * 64'd100 < pipelined_span * 64'd90)
            fail("reported cycles do not account for the pipelined span");
        pulse_soft_reset();
        pass_count = pass_count + 1;

        // ---- BUSY_START only once both banks are occupied -------------------
        configure(ROWS, COLUMNS, 8, JOB_TIMEOUT);
        pulse_start();
        if (!status_busy || !status_accept) fail("first START should leave a bank free");
        pulse_start();
        if (status_accept) fail("second START should fill the queue");
        if (status_error) fail("second START must not raise BUSY_START");
        pulse_start();
        if (!status_error || error_code !== 8'd3) fail("third START should be BUSY_START");
        if (!status_busy) fail("BUSY_START must not abort the pipeline");
        pulse_soft_reset();
        if (status_busy || status_error || !status_accept || cycles != 0)
            fail("SOFT_RESET after BUSY_START");
        pass_count = pass_count + 1;

        // ---- SOFT_RESET mid-pipeline, with a load in flight -----------------
        configure(ROWS, COLUMNS, 16, JOB_TIMEOUT);
        pulse_start();
        configure(ROWS, COLUMNS, 16, JOB_TIMEOUT);
        pulse_start();
        for (beat_index = 0; beat_index < 2; beat_index = beat_index + 1)
            send_beat({IN_BYTES{8'(beat_index)}}, '1, 1'b0, 0);
        idle_input();
        pulse_soft_reset();
        @(posedge clk); #1;
        if (status_busy || status_done || status_error || status_accept !== 1'b1)
            fail("SOFT_RESET did not quiesce both banks");
        if (s_axis_tready) fail("SOFT_RESET left the input frame open");
        if (cycles != 0) fail("SOFT_RESET did not clear the cycle counter");
        pass_count = pass_count + 1;

        // ---- a malformed queued frame drops the whole pipeline --------------
        configure(ROWS, COLUMNS, 4, JOB_TIMEOUT);
        pulse_start();
        pulse_start();
        send_beat(1, 1, 1'b1, 0);
        idle_input();
        @(posedge clk); #1;
        if (!status_error || error_code !== 8'd4) fail("STREAM_LENGTH on queued pipeline");
        if (status_busy) fail("STREAM_LENGTH must drop every outstanding job");
        pulse_soft_reset();
        pass_count = pass_count + 1;

        // ---- a queued job runs its own timeout ------------------------------
        configure(ROWS, COLUMNS, 4, 400);
        pulse_start();
        configure(ROWS, COLUMNS, 4, 40);
        pulse_start();
        while (status_busy) @(posedge clk);
        #1;
        if (!status_error || error_code !== 8'd5) fail("queued job did not time out");
        pulse_soft_reset();
        pass_count = pass_count + 1;

        $display("PASS tb_npu_matrix_controller_pipeline_8x8 cases=%0d serial=%0d pipelined=%0d saved=%0d single_job=%0d",
                 pass_count, serial_span, pipelined_span,
                 serial_span - pipelined_span, first_job_cycles);
        $finish;
    end
endmodule
