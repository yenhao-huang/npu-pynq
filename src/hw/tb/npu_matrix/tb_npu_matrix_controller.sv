`timescale 1ns/1ps

module tb_npu_matrix_controller;
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
    logic [7:0] error_code;
    logic [63:0] cycles;
    logic signed [31:0] held_data;
    logic held_last;
    logic [63:0] held_cycles;

    npu_matrix_controller #(.ROWS(2), .COLUMNS(2), .MAX_K(256)) dut (.*);

    always #5 clk = ~clk;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_matrix_controller: %s", message);
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
        input integer m, input integer n, input integer k,
        input integer a_stride, input integer b_stride,
        input integer c_stride, input integer timeout_cycles
    );
        begin
            cfg_m = m; cfg_n = n; cfg_k = k;
            cfg_a_stride = a_stride; cfg_b_stride = b_stride;
            cfg_c_stride = c_stride; cfg_timeout_cycles = timeout_cycles;
        end
    endtask

    task automatic send_beat(input integer signed value, input logic last);
        begin
            @(negedge clk);
            s_axis_tdata = value[7:0];
            s_axis_tlast = last;
            s_axis_tvalid = 1'b1;
            while (!s_axis_tready) @(negedge clk);
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
        end
    endtask

    task automatic expect_error(input [7:0] expected, input string label_text);
        integer wait_count;
        begin
            wait_count = 0;
            while (status_busy && wait_count < 20) begin
                @(posedge clk);
                #1;
                wait_count = wait_count + 1;
            end
            @(posedge clk); #1;
            if (status_busy || status_done || !status_error || error_code != expected)
                fail(label_text);
            held_cycles = cycles;
            repeat (3) @(posedge clk);
            if (cycles != held_cycles) fail("cycle counter changed after error");
        end
    endtask

    task automatic run_board_matrix(input logic inject_busy_start);
        begin
            configure(2, 2, 2, 2, 2, 8, 200);
            pulse_start();
            if (!status_busy || !s_axis_tready) fail("valid start did not enter A frame");
            if (inject_busy_start) pulse_start();

            send_beat(-128, 0); send_beat(127, 0);
            repeat (2) @(negedge clk);
            send_beat(7, 0); send_beat(-3, 1);
            send_beat(-1, 0); send_beat(2, 0);
            repeat (2) @(negedge clk);
            send_beat(4, 0); send_beat(-5, 1);

            m_axis_tready = 1'b0;
            wait (m_axis_tvalid);
            held_data = m_axis_tdata;
            held_last = m_axis_tlast;
            repeat (3) begin
                @(posedge clk); #1;
                if (m_axis_tdata !== held_data || m_axis_tlast !== held_last || !m_axis_tvalid)
                    fail("output changed under backpressure");
            end
            m_axis_tready = 1'b1;
            @(posedge clk); #1;
            if ($signed(held_data) != 636 || held_last) fail("C[0,0] mismatch");
            if ($signed(m_axis_tdata) != -891 || m_axis_tlast) fail("C[0,1] mismatch");
            @(posedge clk); #1;
            if ($signed(m_axis_tdata) != -19 || m_axis_tlast) fail("C[1,0] mismatch");
            @(posedge clk); #1;
            if ($signed(m_axis_tdata) != 29 || !m_axis_tlast) begin
                $display("final got data=%0d last=%0b valid=%0b", $signed(m_axis_tdata), m_axis_tlast, m_axis_tvalid);
                fail("C[1,1]/TLAST mismatch");
            end
            @(posedge clk); #1;
            m_axis_tready = 1'b0;
            if (status_busy || !status_done || cycles == 0) fail("successful completion status");
            if (inject_busy_start) begin
                if (!status_error || error_code != 8'd3) fail("BUSY_START not sticky");
            end else if (status_error || error_code != 0) begin
                fail("unexpected successful-job error");
            end
            held_cycles = cycles;
            repeat (3) @(posedge clk);
            if (cycles != held_cycles) fail("cycles not stable after DONE");
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        configure(3, 2, 1, 1, 2, 8, 50); pulse_start();
        expect_error(8'd1, "INVALID_DIMENSION"); pulse_soft_reset();

        configure(2, 2, 2, 3, 2, 8, 50); pulse_start();
        expect_error(8'd2, "INVALID_STRIDE"); pulse_soft_reset();

        configure(2, 2, 2, 2, 2, 8, 0); pulse_start();
        expect_error(8'd6, "INVALID_TIMEOUT"); pulse_soft_reset();

        configure(1, 1, 2, 2, 1, 4, 50); pulse_start();
        send_beat(2, 1); expect_error(8'd4, "early A TLAST"); pulse_soft_reset();

        configure(1, 1, 1, 1, 1, 4, 50); pulse_start();
        send_beat(2, 0); expect_error(8'd4, "missing A TLAST"); pulse_soft_reset();

        configure(1, 1, 1, 1, 1, 4, 3); pulse_start();
        expect_error(8'd5, "TIMEOUT"); pulse_soft_reset();

        run_board_matrix(1'b0); pulse_soft_reset();

        configure(1, 2, 2, 2, 2, 8, 100); pulse_start();
        send_beat(2, 0); send_beat(-3, 1);
        send_beat(4, 0); send_beat(6, 0); send_beat(5, 0); send_beat(-7, 1);
        m_axis_tready = 1'b1;
        wait (m_axis_tvalid);
        if ($signed(m_axis_tdata) != -7 || m_axis_tlast) fail("masked C[0,0]");
        @(posedge clk); #1;
        if ($signed(m_axis_tdata) != 33 || !m_axis_tlast) fail("masked C[0,1]/TLAST");
        @(posedge clk); #1; m_axis_tready = 1'b0;
        if (!status_done || status_error) fail("masked job status");
        pulse_soft_reset();

        run_board_matrix(1'b1); pulse_soft_reset();
        if (status_busy || status_done || status_error || error_code != 0 || cycles != 0)
            fail("SOFT_RESET did not restore lifecycle state");

        $display("PASS tb_npu_matrix_controller");
        $finish;
    end
endmodule
