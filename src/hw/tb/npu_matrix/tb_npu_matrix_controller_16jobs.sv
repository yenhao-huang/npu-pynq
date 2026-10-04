`timescale 1ns/1ps

// Sixteen-job throughput experiment for the A/B ping-pong buffer.
// Runs the same NUM_JOBS stall-free ROWS x COLUMNS x JOB_K matrix jobs twice:
// once serialized (each START waits for !BUSY, the pre-ping-pong behaviour)
// and once with the two-entry queue kept full. Every result is checked against
// an exact INT32 golden model, and the pass prints both wall-clock spans plus
// the per-job CYCLES the controller reported.
//
// ROWS, COLUMNS, JOB_K and NUM_JOBS are parameters so the experiment can be
// swept with `iverilog -P tb_npu_matrix_controller_16jobs.<name>=<value>`.
// Defining NPU_SERIAL_BASELINE drops the status_accept port and the pipelined
// pass, so the serialized pass can also be run against the controller that
// predates this change.
module tb_npu_matrix_controller_16jobs #(
    parameter integer ROWS = 8,
    parameter integer COLUMNS = 8,
    parameter integer JOB_K = 64,
    parameter integer NUM_JOBS = 16
);
    localparam integer MAX_K = 256;
    localparam integer JOB_TIMEOUT = 100000;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start_pulse = 1'b0;
    logic soft_reset_pulse = 1'b0;
    logic [15:0] cfg_m = 0, cfg_n = 0, cfg_k = 0;
    logic [31:0] cfg_a_stride = 0, cfg_b_stride = 0, cfg_c_stride = 0;
    logic [31:0] cfg_timeout_cycles = 0;
    logic [7:0] s_axis_tdata = 0;
    logic s_axis_tvalid = 0, s_axis_tready, s_axis_tlast = 0;
    logic [31:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready = 0, m_axis_tlast;
    logic status_busy, status_done, status_error;
`ifndef NPU_SERIAL_BASELINE
    logic status_accept;
`endif
    logic [7:0] error_code;
    logic [63:0] cycles;

    npu_matrix_controller #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K)
    ) dut (.*);

    always #5 clk = ~clk;

    logic signed [7:0]  a_mem [0:NUM_JOBS*ROWS*JOB_K-1];
    logic signed [7:0]  b_mem [0:NUM_JOBS*JOB_K*COLUMNS-1];
    logic signed [31:0] c_mem [0:NUM_JOBS*ROWS*COLUMNS-1];
    logic [63:0] job_cycles [0:NUM_JOBS-1];

    integer accepted_count;
    integer job_index, row_index, column_index, step_index;
    integer seed;
    logic signed [31:0] accumulator;
    logic [63:0] cycle_counter;
    logic [63:0] serial_span, pipelined_span;
    logic [63:0] serial_first, serial_steady;
    logic [63:0] pipelined_first, pipelined_steady;

    always @(posedge clk) cycle_counter <= cycle_counter + 64'd1;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_matrix_controller_16jobs: %s", message);
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

    task automatic configure;
        begin
            cfg_m = ROWS; cfg_n = COLUMNS; cfg_k = JOB_K;
            cfg_a_stride = JOB_K; cfg_b_stride = COLUMNS; cfg_c_stride = 4 * COLUMNS;
            cfg_timeout_cycles = JOB_TIMEOUT;
        end
    endtask

    task automatic send_beat(input integer signed value, input logic last);
        begin
            @(negedge clk);
            s_axis_tdata = value[7:0];
            s_axis_tlast = last;
            s_axis_tvalid = 1'b1;
            while (!s_axis_tready) @(negedge clk);
        end
    endtask

    // A frame (row-major) then B frame (row-major), one byte per cycle.
    task automatic feed_job(input integer job);
        integer r, c, kk;
        begin
            for (r = 0; r < ROWS; r = r + 1)
                for (kk = 0; kk < JOB_K; kk = kk + 1)
                    send_beat(a_mem[(job*ROWS + r)*JOB_K + kk],
                              (r == ROWS-1) && (kk == JOB_K-1));
            for (kk = 0; kk < JOB_K; kk = kk + 1)
                for (c = 0; c < COLUMNS; c = c + 1)
                    send_beat(b_mem[(job*JOB_K + kk)*COLUMNS + c],
                              (kk == JOB_K-1) && (c == COLUMNS-1));
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
        end
    endtask

    task automatic drain_job(input integer job);
        integer r, c, beat;
        begin
            @(negedge clk);
            m_axis_tready = 1'b1;
            beat = 0;
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLUMNS; c = c + 1) begin
                    while (!m_axis_tvalid) @(negedge clk);
                    if ($signed(m_axis_tdata) !== c_mem[(job*ROWS + r)*COLUMNS + c]) begin
                        $display("job=%0d C[%0d,%0d] expected=%0d got=%0d", job, r, c,
                                 c_mem[(job*ROWS + r)*COLUMNS + c], $signed(m_axis_tdata));
                        fail("result mismatch against golden model");
                    end
                    if (m_axis_tlast !== ((r == ROWS-1) && (c == COLUMNS-1)))
                        fail("TLAST position");
                    beat = beat + 1;
                    // Drop TREADY right after the last beat so the next job's
                    // frame is not consumed early.
                    if (beat == ROWS*COLUMNS) begin
                        @(posedge clk); #1;
                        m_axis_tready = 1'b0;
                    end else begin
                        @(negedge clk);
                    end
                end
        end
    endtask

    // first: CYCLES of job 0; steady: mean CYCLES of jobs 1..NUM_JOBS-1.
    task automatic run_jobs(
        input logic pipelined, output logic [63:0] span,
        output logic [63:0] first, output logic [63:0] steady
    );
        logic [63:0] began, rest;
        integer j;
        begin
            accepted_count = 0;
            began = cycle_counter;
            fork
                begin : starter
                    integer sj;
                    for (sj = 0; sj < NUM_JOBS; sj = sj + 1) begin
`ifndef NPU_SERIAL_BASELINE
                        if (pipelined) begin
                            while (!status_accept) @(negedge clk);
                        end else
`endif
                        begin
                            while (status_busy) @(negedge clk);
                        end
                        configure();
                        pulse_start();
                        if (status_error) fail("unexpected error on START");
                        accepted_count = sj + 1;
                    end
                end
                begin : feeder
                    integer fj;
                    for (fj = 0; fj < NUM_JOBS; fj = fj + 1) begin
                        while (accepted_count <= fj) @(negedge clk);
                        feed_job(fj);
                    end
                end
                begin : drainer
                    integer dj;
                    for (dj = 0; dj < NUM_JOBS; dj = dj + 1) begin
                        drain_job(dj);
                        job_cycles[dj] = cycles;
                    end
                end
            join
            span = cycle_counter - began;
            if (status_error) fail("error raised during a clean run");
            first = job_cycles[0];
            rest = 64'd0;
            for (j = 1; j < NUM_JOBS; j = j + 1) rest = rest + job_cycles[j];
            steady = rest / (NUM_JOBS - 1);
            for (j = 0; j < NUM_JOBS; j = j + 1)
                $display("  %s job=%0d cycles=%0d",
                         pipelined ? "pipelined" : "serial", j, job_cycles[j]);
        end
    endtask

    initial begin
        cycle_counter = 64'd0;
        seed = 32'h16_0b5;

        for (job_index = 0; job_index < NUM_JOBS; job_index = job_index + 1) begin
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1)
                for (step_index = 0; step_index < JOB_K; step_index = step_index + 1)
                    a_mem[(job_index*ROWS + row_index)*JOB_K + step_index] = $random(seed);
            for (step_index = 0; step_index < JOB_K; step_index = step_index + 1)
                for (column_index = 0; column_index < COLUMNS; column_index = column_index + 1)
                    b_mem[(job_index*JOB_K + step_index)*COLUMNS + column_index] = $random(seed);
            for (row_index = 0; row_index < ROWS; row_index = row_index + 1)
                for (column_index = 0; column_index < COLUMNS; column_index = column_index + 1) begin
                    accumulator = 0;
                    for (step_index = 0; step_index < JOB_K; step_index = step_index + 1)
                        accumulator = accumulator +
                            a_mem[(job_index*ROWS + row_index)*JOB_K + step_index] *
                            b_mem[(job_index*JOB_K + step_index)*COLUMNS + column_index];
                    c_mem[(job_index*ROWS + row_index)*COLUMNS + column_index] = accumulator;
                end
        end

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        run_jobs(1'b0, serial_span, serial_first, serial_steady);
        pulse_soft_reset();

`ifdef NPU_SERIAL_BASELINE
        $display("PASS tb_npu_matrix_controller_16jobs baseline %0dx%0dx%0d jobs=%0d serial=%0d first=%0d steady=%0d",
                 ROWS, COLUMNS, JOB_K, NUM_JOBS, serial_span, serial_first, serial_steady);
`else
        run_jobs(1'b1, pipelined_span, pipelined_first, pipelined_steady);
        if (pipelined_span >= serial_span) fail("pipelined run did not save cycles");
        pulse_soft_reset();

        $display("PASS tb_npu_matrix_controller_16jobs %0dx%0dx%0d jobs=%0d serial=%0d pipelined=%0d saved=%0d serial_steady=%0d pipelined_steady=%0d first=%0d",
                 ROWS, COLUMNS, JOB_K, NUM_JOBS, serial_span, pipelined_span,
                 serial_span - pipelined_span, serial_steady, pipelined_steady,
                 pipelined_first);
`endif
        $finish;
    end
endmodule
