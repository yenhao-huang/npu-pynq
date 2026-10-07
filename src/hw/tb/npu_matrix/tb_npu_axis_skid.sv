`timescale 1ns/1ps

// npu_axis_skid under random valid/ready: every beat arrives once, in order,
// and back-to-back transfers run at one beat per cycle.
module tb_npu_axis_skid;
    logic clk = 0, rst_n = 0;
    logic s_valid = 0, s_ready, m_valid, m_ready = 0;
    logic [15:0] s_data = 0, m_data;
    integer sent = 0, received = 0, cycles = 0, burst = 0;
    logic [31:0] lfsr = 32'hace1;

    npu_axis_skid #(.WIDTH(16)) dut (.*);
    always #5 clk = ~clk;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_npu_axis_skid: %s", message);
            $fatal(1);
        end
    endtask

    always @(posedge clk) begin
        lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        if (rst_n) begin
            cycles <= cycles + 1;
            if (s_valid && s_ready) sent <= sent + 1;
            if (m_valid && m_ready) begin
                if (m_data !== received[15:0]) fail("beat out of order or lost");
                received <= received + 1;
            end
        end
    end

    always @(negedge clk) begin
        if (rst_n) begin
            // Phase 1: random; phase 2: both sides always ready (throughput).
            if (sent < 2000) begin
                s_valid = lfsr[3] | lfsr[7];
                m_ready = lfsr[5] | lfsr[11];
            end else begin
                s_valid = (sent < 3000);
                m_ready = 1'b1;
            end
            s_data = sent[15:0];
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1;
        wait (sent == 2000);
        burst = cycles;
        wait (sent == 3000);
        if (cycles - burst > 1003) fail("not one beat per cycle when unstalled");
        repeat (5) @(posedge clk);
        if (received != 3000) fail("beats lost");
        $display("PASS tb_npu_axis_skid: %0d beats", received);
        $finish;
    end
endmodule
