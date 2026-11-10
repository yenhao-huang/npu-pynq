`timescale 1ns/1ps

// Run-control state transitions of scalar_core_controller, with the pipeline
// stages replaced by the events they report. Inputs change on the falling
// edge and are sampled by the controller on the next rising edge, so every
// check states the cycle a transition takes.
module tb_scalar_core_controller;
    logic        clk = 1'b0;
    logic        rst_n = 1'b0;
    logic        isa_start = 1'b0;
    logic        decode_error = 1'b0;
    logic [7:0]  decode_error_code = 8'd0;
    logic        program_end = 1'b0;
    logic        job_valid = 1'b0;
    logic        issue = 1'b0;
    logic        job_written = 1'b0;
    logic        fetch_error = 1'b0;
    logic        read_error = 1'b0;
    logic        write_error = 1'b0;
    logic        ctl_error = 1'b0;
    logic [7:0]  ctl_error_code = 8'd0;
    wire         start;
    wire         running;
    wire         stop;
    wire         drained;
    wire         fault;
    wire         fault_latched;
    wire         busy;
    wire         done;
    wire         error;
    wire [7:0]   error_code;
    wire [31:0]  jobs;
    wire [31:0]  cycles;

    integer checks = 0;

    always #5 clk = ~clk;

    scalar_core_controller dut (.*);

    task automatic check(input logic condition, input string message);
        begin
            checks = checks + 1;
            if (condition !== 1'b1) begin
                $display("FAIL tb_scalar_core_controller: %s (t=%0t)", message, $time);
                $display("  running=%b busy=%b done=%b reported=%b code=0x%02x drained=%b stop=%b fault=%b latched=%b jobs=%0d cycles=%0d",
                    running, busy, done, error, error_code, drained, stop, fault,
                    fault_latched, jobs, cycles);
                $fatal(1);
            end
        end
    endtask

    // One clock; leaves the testbench on the falling edge after it.
    task automatic tick;
        begin
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    // Requests a start for one cycle and reports whether it was admitted.
    task automatic request_start(output logic admitted);
        begin
            isa_start = 1'b1;
            #1 admitted = start;
            tick();
            isa_start = 1'b0;
        end
    endtask

    // ID hands one job to EX: job_valid for `wait_cycles` cycles, then issue.
    task automatic issue_job(input integer wait_cycles);
        begin
            job_valid = 1'b1;
            repeat (wait_cycles) tick();
            issue = 1'b1;
            tick();
            issue = 1'b0;
            job_valid = 1'b0;
        end
    endtask

    task automatic write_back;
        begin
            job_written = 1'b1;
            tick();
            job_written = 1'b0;
        end
    endtask

    task automatic expect_idle(input string where);
        begin
            check(!running && !busy && stop && drained, {where, ": idle"});
        end
    endtask

    logic admitted;
    logic [31:0] cycles_seen;

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        tick();

        // ------------------------------------------------ reset state
        expect_idle("after reset");
        check(!done && !error && error_code == 8'd0, "after reset: no status");
        check(jobs == 32'd0 && cycles == 32'd0, "after reset: counters zero");
        check(!start && !fault && !fault_latched, "after reset: no start, no fault");

        // ------------------------------------- start while idle is admitted
        request_start(admitted);
        check(admitted, "start while idle is admitted");
        check(running && busy && !stop, "running one cycle after start");
        check(!done && !error && cycles == 32'd0, "start clears status and cycles");
        check(!start, "start is a pulse");
        tick();
        check(cycles == 32'd1, "cycles count while running");

        // ----------------------------------- start while running is ignored
        request_start(admitted);
        check(!admitted, "start while running is ignored");
        check(running && cycles == 32'd2, "an ignored start does not restart the program");

        // -------------------------- jobs outstanding: not drained, busy held
        check(drained, "drained before the first job");
        job_valid = 1'b1;
        #1 check(!drained, "a decoded job waiting to issue is not drained");
        issue_job(2);
        check(!drained && busy, "an issued job is outstanding");
        issue_job(0);
        check(!drained, "two jobs outstanding");
        check(jobs == 32'd0, "jobs counts written-back jobs only");
        write_back();
        check(!drained && jobs == 32'd1, "one of two jobs written back");
        write_back();
        check(drained && jobs == 32'd2, "drained once every job is written back");
        check(running && busy, "still running until END");

        // ------------------------------------------------ normal completion
        cycles_seen = cycles;
        program_end = 1'b1;
        tick();
        program_end = 1'b0;
        check(!running && done && !error, "END ends the run with DONE");
        check(!busy && stop && drained, "idle after END");
        check(cycles == cycles_seen + 32'd1, "the END cycle is the last one counted");
        tick();
        check(cycles == cycles_seen + 32'd1 && done, "cycles and DONE hold after the run");
        check(jobs == 32'd2, "jobs holds after the run");

        // ------------------- decode error with a job in flight: busy drains
        request_start(admitted);
        check(admitted, "start after DONE is admitted");
        check(running && !done && jobs == 32'd0 && cycles == 32'd0,
            "a new program clears DONE, jobs and cycles");
        issue_job(1);
        decode_error = 1'b1;
        decode_error_code = 8'h11;
        tick();
        decode_error = 1'b0;
        decode_error_code = 8'd0;
        check(!running && stop, "a decode error stops the run");
        check(error && error_code == 8'h11 && !done, "the decode error code is reported");
        check(!fault_latched, "a decode error is not a fault");
        check(busy && !drained, "busy holds while the issued job drains");
        request_start(admitted);
        check(!admitted, "start while draining is ignored");
        check(!running && error && error_code == 8'h11, "an ignored start keeps the status");
        write_back();
        check(!busy && drained && jobs == 32'd1, "busy falls when the job is written back");
        check(error && error_code == 8'h11, "the code is sticky until the next start");

        // -------- a decoded job that never issued also holds busy (job_valid)
        job_valid = 1'b1;
        #1 check(busy && !running, "a pending job holds busy");
        job_valid = 1'b0;
        #1 check(!busy, "busy follows the pending job");

        // ------------------ matrix-core fault latched in the middle of a run
        request_start(admitted);
        check(admitted && running && !error && error_code == 8'd0,
            "start clears the previous code");
        issue_job(0);
        ctl_error = 1'b1;
        ctl_error_code = 8'd3;
        #1 check(fault && !fault_latched && running, "the fault is seen in its cycle");
        tick();
        check(!running && stop, "a fault stops the run");
        check(fault_latched && error && error_code == 8'd3, "the matrix-core code is latched");
        check(busy, "busy holds while the issued job drains after a fault");
        // The matrix core's flag is sticky and its code may change; a second
        // source appears. The first latched code stays.
        ctl_error_code = 8'd5;
        read_error = 1'b1;
        tick();
        check(error_code == 8'd3, "the first fault's code is kept");
        // A decode error reported now does not displace the fault.
        decode_error = 1'b1;
        decode_error_code = 8'h10;
        tick();
        decode_error = 1'b0;
        decode_error_code = 8'd0;
        check(error && error_code == 8'd3, "a latched fault takes precedence over a decode error");
        write_back();
        check(!busy && drained, "busy falls after the fault once the job is written back");
        check(!done, "no DONE after a fault");
        ctl_error = 1'b0;
        ctl_error_code = 8'd0;
        read_error = 1'b0;
        tick();
        check(fault_latched && error_code == 8'd3, "the latch outlives the fault source");

        // ------------------------------ fault sources and their code priority
        request_start(admitted);
        check(admitted && !error, "start clears the latched fault");
        write_error = 1'b1;
        read_error = 1'b1;
        fetch_error = 1'b1;
        tick();
        check(error && error_code == 8'h15, "fetch outranks load and store");
        fetch_error = 1'b0; read_error = 1'b0; write_error = 1'b0;
        tick();

        request_start(admitted);
        write_error = 1'b1;
        read_error = 1'b1;
        tick();
        check(error && error_code == 8'h13, "load outranks store");
        read_error = 1'b0; write_error = 1'b0;
        tick();

        request_start(admitted);
        write_error = 1'b1;
        tick();
        check(error && error_code == 8'h14 && !running, "a store fault is reported");
        write_error = 1'b0;
        tick();

        request_start(admitted);
        fetch_error = 1'b1;
        ctl_error = 1'b1;
        ctl_error_code = 8'd4;
        tick();
        check(error && error_code == 8'd4, "the matrix core outranks the AXI sources");
        fetch_error = 1'b0; ctl_error = 1'b0; ctl_error_code = 8'd0;
        tick();

        // -------------------------- a fault while idle is not latched
        request_start(admitted);
        program_end = 1'b1;
        tick();
        program_end = 1'b0;
        check(done && !error, "an empty program completes");
        ctl_error = 1'b1;
        ctl_error_code = 8'd1;
        tick();
        tick();
        check(fault && !fault_latched && !error && done, "a fault outside a run is not latched");
        // It is latched by the first running cycle of the next program.
        request_start(admitted);
        check(admitted && running && !error, "start is admitted with a fault pending");
        tick();
        check(!running && error && error_code == 8'd1, "a pending fault stops the next program");
        ctl_error = 1'b0;
        ctl_error_code = 8'd0;
        tick();

        // ------------------------------------ END and a fault in one cycle
        request_start(admitted);
        program_end = 1'b1;
        read_error = 1'b1;
        tick();
        program_end = 1'b0;
        read_error = 1'b0;
        check(!running && done && error && error_code == 8'h13,
            "END with a fault in the same cycle reports both");

        // ------------------------------------------------------- reset
        rst_n = 1'b0;
        tick();
        rst_n = 1'b1;
        tick();
        expect_idle("after a second reset");
        check(!done && !error && jobs == 32'd0 && cycles == 32'd0, "reset clears status");

        $display("PASS tb_scalar_core_controller: %0d checks", checks);
        $finish;
    end
endmodule
