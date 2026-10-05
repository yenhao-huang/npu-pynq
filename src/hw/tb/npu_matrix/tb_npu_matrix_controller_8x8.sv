`timescale 1ns/1ps

module tb_npu_matrix_controller_8x8;
    localparam integer ROWS = 8;
    localparam integer COLUMNS = 8;
    localparam integer MAX_K = 16;
    localparam integer IN_BYTES = 8;

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

    integer signed matrix_a [0:ROWS-1][0:MAX_K-1];
    integer signed matrix_b [0:MAX_K-1][0:COLUMNS-1];
    integer signed expected [0:ROWS-1][0:COLUMNS-1];
    logic [7:0] stream_bytes [0:ROWS*MAX_K + MAX_K*COLUMNS - 1];
    integer input_beats;

    npu_matrix_core #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS),
        .MAX_K(MAX_K),
        .IN_BYTES(IN_BYTES)
    ) dut (.*);

    always #5 clk = ~clk;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_matrix_controller_8x8: %s", message);
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

    always @(posedge clk)
        if (s_axis_tvalid && s_axis_tready) input_beats <= input_beats + 1;

    // Stream stream_bytes[0:count-1] as one DMA transfer: IN_BYTES bytes per
    // beat with TVALID held high back to back, a partial final beat marked by
    // TKEEP, and TLAST on that final beat.
    task automatic send_bytes(input integer count);
        integer offset, lane;
        begin
            for (offset = 0; offset < count; offset = offset + IN_BYTES) begin
                @(negedge clk);
                s_axis_tdata = 0;
                s_axis_tkeep = 0;
                for (lane = 0; lane < IN_BYTES; lane = lane + 1) begin
                    if (offset + lane < count) begin
                        s_axis_tdata[lane*8 +: 8] = stream_bytes[offset + lane];
                        s_axis_tkeep[lane] = 1'b1;
                    end
                end
                s_axis_tlast = (offset + IN_BYTES >= count);
                s_axis_tvalid = 1'b1;
                #1;
                while (!s_axis_tready) begin
                    @(negedge clk); #1;
                end
            end
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tkeep = 0;
        end
    endtask

    task automatic prepare_case(
        input integer active_m,
        input integer active_n,
        input integer active_k,
        input integer seed
    );
        integer row, column, reduction;
        begin
            for (row = 0; row < ROWS; row = row + 1) begin
                for (reduction = 0; reduction < MAX_K; reduction = reduction + 1)
                    matrix_a[row][reduction] =
                        ((row * 29 + reduction * 17 + seed) % 255) - 127;
            end
            for (reduction = 0; reduction < MAX_K; reduction = reduction + 1) begin
                for (column = 0; column < COLUMNS; column = column + 1)
                    matrix_b[reduction][column] =
                        ((reduction * 31 + column * 13 + seed * 3) % 255) - 127;
            end
            for (row = 0; row < active_m; row = row + 1) begin
                for (column = 0; column < active_n; column = column + 1) begin
                    expected[row][column] = 0;
                    for (reduction = 0; reduction < active_k; reduction = reduction + 1)
                        expected[row][column] = expected[row][column] +
                            matrix_a[row][reduction] * matrix_b[reduction][column];
                end
            end
        end
    endtask

    task automatic run_case(
        input integer active_m,
        input integer active_n,
        input integer active_k,
        input integer seed
    );
        integer row, column, reduction;
        integer a_bytes, b_bytes, expected_beats;
        begin
            prepare_case(active_m, active_n, active_k, seed);
            cfg_m = active_m;
            cfg_n = active_n;
            cfg_k = active_k;
            cfg_a_stride = active_k;
            cfg_b_stride = active_n;
            cfg_c_stride = 4 * active_n;
            cfg_timeout_cycles = 10000;
            pulse_start();
            if (!status_busy || !s_axis_tready)
                fail("valid job did not enter the input state");

            a_bytes = active_m * active_k;
            b_bytes = active_k * active_n;
            expected_beats = (a_bytes + IN_BYTES - 1) / IN_BYTES +
                (b_bytes + IN_BYTES - 1) / IN_BYTES;
            input_beats = 0;
            for (row = 0; row < active_m; row = row + 1)
                for (reduction = 0; reduction < active_k; reduction = reduction + 1)
                    stream_bytes[row * active_k + reduction] = matrix_a[row][reduction];
            send_bytes(a_bytes);
            for (reduction = 0; reduction < active_k; reduction = reduction + 1)
                for (column = 0; column < active_n; column = column + 1)
                    stream_bytes[reduction * active_n + column] = matrix_b[reduction][column];
            send_bytes(b_bytes);
            if (input_beats != expected_beats)
                fail("input beat count is not ceil(bytes / IN_BYTES) per operand");

            m_axis_tready = 1'b1;
            wait (m_axis_tvalid);
            for (row = 0; row < active_m; row = row + 1) begin
                for (column = 0; column < active_n; column = column + 1) begin
                    @(negedge clk);
                    if (!m_axis_tvalid)
                        fail("output stream ended early");
                    if ($signed(m_axis_tdata) !== expected[row][column]) begin
                        $display(
                            "FAIL tb_npu_matrix_controller_8x8: [%0d,%0d] expected %0d got %0d",
                            row, column, expected[row][column], $signed(m_axis_tdata)
                        );
                        $fatal(1);
                    end
                    if (m_axis_tlast !== ((row == active_m - 1) &&
                                          (column == active_n - 1)))
                        fail("TLAST position mismatch");
                end
            end
            @(negedge clk);
            m_axis_tready = 1'b0;
            if (status_busy || !status_done || status_error || error_code != 0)
                fail("successful job status mismatch");
            // Stall-free job length: one cycle per row-aligned word of A and
            // B, then drain and output, plus one aligner fill cycle per frame.
            // The extra cycle is the idle edge this testbench leaves after
            // START. The ping-pong exec engine has no idle CLEAR cycle between
            // the B frame and the wavefront tail, so the constant is 3, not 4.
            if (cycles != active_m * ((active_k + IN_BYTES - 1) / IN_BYTES) +
                          active_k * ((active_n + IN_BYTES - 1) / IN_BYTES) +
                          active_m * active_n + active_m + active_n + 3 + 1) begin
                $display("FAIL tb_npu_matrix_controller_8x8: cycles %0d for %0dx%0dx%0d",
                         cycles, active_m, active_n, active_k);
                $fatal(1);
            end
            pulse_soft_reset();
        end
    endtask

    initial begin
        integer case_index;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        run_case(8, 8, 11, 7);
        run_case(7, 5, 13, 41);
        for (case_index = 0; case_index < 32; case_index = case_index + 1) begin
            run_case(
                1 + ((case_index * 5 + 3) % ROWS),
                1 + ((case_index * 7 + 1) % COLUMNS),
                1 + ((case_index * 11 + 5) % MAX_K),
                97 + case_index * 37
            );
        end

        $display("PASS tb_npu_matrix_controller_8x8 cases=34");
        $finish;
    end
endmodule
