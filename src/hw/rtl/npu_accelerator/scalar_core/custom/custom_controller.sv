`timescale 1ns/1ps

// Controller of the CUSTOM scalar core: program-level run control over the pipeline
// stages. (matrix_core_controller, its counterpart in the matrix core,
// sequences one job over the operand buffers and the systolic array cycle by
// cycle; this module never looks inside a job.) The stages report events; the
// run state lives here.
//
// It owns:
//   - start admission: ISA START is taken only while the core is not busy;
//   - the run state. The flops are running, done, the decode error and the
//     latched fault; with the outstanding-job count they give
//       idle      !running, nothing outstanding      (START is admitted)
//       running   decode executes the program
//       draining  !running after a decode error or a fault, while the jobs
//                 already issued are written back    (busy stays high)
//       done / error  sticky until the next admitted START;
//   - stop to IF (no further fetch once the program is not running);
//   - the outstanding-job count, and from it the "drained" condition that
//     FENCE and END wait for in ID, and the busy composition;
//   - fault collection (fetch, load, store, matrix core), the first-fault
//     latch, and merging it with the decode error into the reported error
//     and code. A latched fault takes precedence over a decode error;
//   - the program counters that follow the run state: jobs written back and
//     cycles run. (PC and the instruction count belong to ID.)
//
// It does not own the EX issue handshake: job_ready, the issue strobe, the
// cooldown and the registered START and shape that drive the matrix core stay
// in custom_scalar_core, beside the tensor DMA they hand each job to. The
// controller only gates that handshake through fault_latched and counts its
// result.
//
// Error codes match src/vendors/npu_pynq/spec/abi.py.
module custom_controller (
    input  logic        clk,
    input  logic        rst_n,

    // Program start request from the CSR, and the admitted start pulse that
    // clears every stage.
    input  logic        isa_start,
    output logic        start,

    // ID: events of the instruction being executed.
    input  logic        decode_error,
    input  logic [7:0]  decode_error_code,
    input  logic        program_end,
    // A decoded job is waiting to issue.
    input  logic        job_valid,
    // EX: a job was issued / a job's result was written back.
    input  logic        issue,
    input  logic        job_written,

    // Fault sources: IF, the tensor DMA, the matrix core.
    input  logic        fetch_error,
    input  logic        read_error,
    input  logic        write_error,
    input  logic        ctl_error,
    input  logic [7:0]  ctl_error_code,

    // Control to the stages.
    output logic        running,
    output logic        stop,
    output logic        drained,
    output logic        fault,
    output logic        fault_latched,

    // Status to the CSR.
    output logic        busy,
    output logic        done,
    output logic        error,
    output logic [7:0]  error_code,
    output logic [31:0] jobs,
    output logic [31:0] cycles
);
    localparam logic [7:0] ERR_AXI_READ = 8'h13;
    localparam logic [7:0] ERR_AXI_WRITE = 8'h14;
    localparam logic [7:0] ERR_FETCH = 8'h15;

    logic        decode_error_latched;
    logic [7:0]  decode_error_code_latched;
    logic [7:0]  fault_code, fault_code_latched;
    // Free-running across programs; only their difference is used.
    logic [31:0] jobs_issued, jobs_written, jobs_at_start;
    logic        ex_idle;

    always_comb begin
        ex_idle = (jobs_issued == jobs_written);
        // Every job decoded so far has been issued and written back.
        drained = ex_idle && !job_valid;
        busy = running || job_valid || !ex_idle;
        start = isa_start && !busy;
        stop = !running;

        fault = fetch_error || read_error || write_error || ctl_error;
        fault_code = ctl_error ? ctl_error_code :
            fetch_error ? ERR_FETCH :
            read_error ? ERR_AXI_READ : ERR_AXI_WRITE;

        error = decode_error_latched || fault_latched;
        error_code = fault_latched ? fault_code_latched : decode_error_code_latched;
        jobs = jobs_written - jobs_at_start;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            decode_error_latched <= 1'b0;
            decode_error_code_latched <= 8'd0;
            fault_latched <= 1'b0;
            fault_code_latched <= 8'd0;
            jobs_issued <= 32'd0;
            jobs_written <= 32'd0;
            jobs_at_start <= 32'd0;
            cycles <= 32'd0;
        end else if (start) begin
            running <= 1'b1;
            done <= 1'b0;
            decode_error_latched <= 1'b0;
            decode_error_code_latched <= 8'd0;
            fault_latched <= 1'b0;
            fault_code_latched <= 8'd0;
            jobs_at_start <= jobs_written;
            cycles <= 32'd0;
        end else begin
            // ID stops the program: an instruction it cannot execute, or END
            // once everything before it has drained.
            if (decode_error) begin
                decode_error_latched <= 1'b1;
                decode_error_code_latched <= decode_error_code;
                running <= 1'b0;
            end
            if (program_end) begin
                running <= 1'b0;
                done <= 1'b1;
            end

            // A fault elsewhere (fetch, load/store, matrix core) stops the
            // program too. The first one while running is the one reported.
            if (fault)
                running <= 1'b0;
            if (running && fault && !fault_latched) begin
                fault_latched <= 1'b1;
                fault_code_latched <= fault_code;
            end

            if (issue)
                jobs_issued <= jobs_issued + 32'd1;
            if (job_written)
                jobs_written <= jobs_written + 32'd1;
            if (running)
                cycles <= cycles + 32'd1;
        end
    end
endmodule
