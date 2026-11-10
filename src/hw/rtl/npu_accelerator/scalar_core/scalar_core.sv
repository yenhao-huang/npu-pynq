`timescale 1ns/1ps

// Scalar core of the NPU: runs the instruction stream and drives the matrix
// core. (The software ABI still calls it the ISA front end.)
//
//   IF   scalar_core_fetch       PC and prefetch queue, reads
//                                PROG_ADDR..+8*PROG_LEN
//   ID   scalar_core_decode      instruction decode, architectural state,
//                                REPEAT, one job per GEMM
//   EX   this module             issues each job: starts the matrix controller
//                                and hands the A/B/C addresses to
//        scalar_core_lsu         which moves the operands from DDR into the
//                                controller's input stream and its results
//                                back to DDR
//   ctl  scalar_core_controller  program-level run control across the stages:
//                                start admission, run state, stop, drained,
//                                busy, faults and the reported error
//
// Besides the instances, this module holds only what joins them: the EX issue
// handshake with its registered START and shape, and the arbitration of the
// AXI read channel between IF and the LSU. The issue logic is here and not in
// the controller because it is one job's handshake between ID, the LSU and
// the matrix core, not program state; the controller gates it (fault_latched)
// and counts its result (issue).
//
// The scalar core is the matrix core's only command source: ctl_start and the
// job shape go straight to it, and both come from flip-flops here, because the
// 100 MHz margin of the 16 x 16 array leaves no room for logic in front of the
// controller.
//
// IF and the LSU share one AXI4 master. Read bursts carry a one-bit tag in an
// in-order FIFO (one AXI ID, so R returns in AR order) that routes R beats
// back to their requester.
module scalar_core #(
    parameter integer ROWS = 16,
    parameter integer COLUMNS = 16,
    parameter integer MAX_K = 256
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        isa_start,
    input  logic [31:0] prog_addr,
    input  logic [31:0] prog_len,
    input  logic [31:0] data_base,
    output logic        isa_running,
    output logic        isa_busy,
    output logic        isa_done,
    output logic        isa_error,
    output logic [7:0]  isa_error_code,
    output logic [31:0] isa_pc,
    output logic [31:0] isa_instructions,
    output logic [31:0] isa_jobs,
    output logic [31:0] isa_cycles,

    output logic        ctl_start,
    output logic [15:0] ctl_m,
    output logic [15:0] ctl_n,
    output logic [15:0] ctl_k,
    input  logic        ctl_accept,
    input  logic        ctl_error,
    input  logic [7:0]  ctl_error_code,

    output logic [63:0] s_axis_tdata,
    output logic [7:0]  s_axis_tkeep,
    output logic        s_axis_tvalid,
    input  logic        s_axis_tready,
    output logic        s_axis_tlast,
    input  logic [31:0] m_axis_tdata,
    input  logic        m_axis_tvalid,
    output logic        m_axis_tready,
    input  logic        m_axis_tlast,

    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic [3:0]  m_axi_arcache,
    output logic [2:0]  m_axi_arprot,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic [3:0]  m_axi_awcache,
    output logic [2:0]  m_axi_awprot,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready
);
    logic start, stop, drained;
    logic dec_error, dec_end;
    logic [7:0] dec_error_code;
    logic instr_valid, instr_ready;
    logic [63:0] instr_data;
    logic job_valid, job_ready;
    logic [7:0] job_m, job_n;
    logic [15:0] job_k;
    logic [31:0] job_a, job_b, job_c;
    logic fetch_error, read_error, write_error, job_written;
    logic fault, fault_latched;

    logic f_ar_valid, f_ar_ready, f_r_valid, f_r_ready;
    logic [31:0] f_ar_addr;
    logic [7:0] f_ar_len;
    logic l_ar_valid, l_ar_ready, l_r_valid, l_r_ready;
    logic [31:0] l_ar_addr;
    logic [7:0] l_ar_len;
    logic ld_ready, st_ready;
    logic [12:0] a_bytes, b_bytes;
    logic [8:0] c_words;
    // A job issues in EX (issue); START reaches the controller one register
    // later (ctl_start), so its ACCEPT is stale for two cycles after an issue.
    logic issue;
    logic [1:0] cooldown;

    scalar_core_controller u_controller (
        .clk(clk), .rst_n(rst_n),
        .isa_start(isa_start), .start(start),
        .decode_error(dec_error), .decode_error_code(dec_error_code),
        .program_end(dec_end),
        .job_valid(job_valid), .issue(issue), .job_written(job_written),
        .fetch_error(fetch_error), .read_error(read_error),
        .write_error(write_error),
        .ctl_error(ctl_error), .ctl_error_code(ctl_error_code),
        .running(isa_running), .stop(stop), .drained(drained),
        .fault(fault), .fault_latched(fault_latched),
        .busy(isa_busy), .done(isa_done),
        .error(isa_error), .error_code(isa_error_code),
        .jobs(isa_jobs), .cycles(isa_cycles)
    );

    scalar_core_fetch fetch (
        .clk(clk), .rst_n(rst_n),
        .start(start), .stop(stop),
        .prog_addr(prog_addr), .prog_len(prog_len),
        .ar_valid(f_ar_valid), .ar_ready(f_ar_ready),
        .ar_addr(f_ar_addr), .ar_len(f_ar_len),
        .r_valid(f_r_valid), .r_ready(f_r_ready),
        .r_data(m_axi_rdata), .r_resp(m_axi_rresp), .r_last(m_axi_rlast),
        .instr_valid(instr_valid), .instr_data(instr_data),
        .instr_ready(instr_ready), .fetch_error(fetch_error)
    );

    scalar_core_decode #(.ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K)) decode (
        .clk(clk), .rst_n(rst_n),
        .start(start), .running(isa_running), .drained(drained),
        .fault(fault), .data_base(data_base),
        .instr_valid(instr_valid), .instr_data(instr_data),
        .instr_ready(instr_ready),
        .job_valid(job_valid), .job_ready(job_ready),
        .job_m(job_m), .job_n(job_n), .job_k(job_k),
        .job_a(job_a), .job_b(job_b), .job_c(job_c),
        .job_a_bytes(a_bytes), .job_b_bytes(b_bytes), .job_c_words(c_words),
        .decode_error(dec_error), .decode_error_code(dec_error_code),
        .program_end(dec_end),
        .pc(isa_pc), .instructions(isa_instructions)
    );

    // ------------------------------------------------------------- execute
    // A job issues when the controller has a free queue entry and the LSU can
    // take both its load and its store. START and the shape are registered
    // and reach the controller together on the next cycle; its ACCEPT reflects
    // the new occupancy one cycle after that, which the two-cycle cooldown
    // covers. An admitted program start leaves the cooldown as it is.
    always_comb begin
        job_ready = ctl_accept && ld_ready && st_ready && !fault_latched && (cooldown == 2'd0);
        issue = job_valid && job_ready;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctl_start <= 1'b0;
            ctl_m <= 16'd0;
            ctl_n <= 16'd0;
            ctl_k <= 16'd0;
            cooldown <= 2'd0;
        end else begin
            if (!start)
                cooldown <= issue ? 2'd2 : (cooldown != 2'd0 ? cooldown - 2'd1 : 2'd0);
            ctl_start <= issue;
            if (issue) begin
                ctl_m <= {8'd0, job_m};
                ctl_n <= {8'd0, job_n};
                ctl_k <= job_k;
            end
        end
    end

    scalar_core_lsu lsu (
        .clk(clk), .rst_n(rst_n), .clear_errors(start),
        .ld_valid(issue), .ld_ready(ld_ready),
        .ld_a_addr(job_a), .ld_a_bytes(a_bytes),
        .ld_b_addr(job_b), .ld_b_bytes(b_bytes),
        .st_valid(issue), .st_ready(st_ready),
        .st_c_addr(job_c), .st_words(c_words),
        .ar_valid(l_ar_valid), .ar_ready(l_ar_ready),
        .ar_addr(l_ar_addr), .ar_len(l_ar_len),
        .r_valid(l_r_valid), .r_ready(l_r_ready),
        .r_data(m_axi_rdata), .r_resp(m_axi_rresp),
        .aw_valid(m_axi_awvalid), .aw_ready(m_axi_awready),
        .aw_addr(m_axi_awaddr), .aw_len(m_axi_awlen),
        .w_valid(m_axi_wvalid), .w_ready(m_axi_wready),
        .w_data(m_axi_wdata), .w_strb(m_axi_wstrb), .w_last(m_axi_wlast),
        .b_valid(m_axi_bvalid), .b_ready(m_axi_bready), .b_resp(m_axi_bresp),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast),
        .jobs_written(job_written),
        .read_error(read_error), .write_error(write_error)
    );

    // ------------------------------------------------------- read arbiter
    logic tag_full, tag_empty, tag_head, grant_fetch;

    sync_fifo #(.WIDTH(1), .DEPTH(16)) read_tags (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(m_axi_arvalid && m_axi_arready), .push_data(grant_fetch),
        .full(tag_full),
        .pop(m_axi_rvalid && m_axi_rready && m_axi_rlast),
        .head(tag_head), .empty(tag_empty)
    );

    always_comb begin
        grant_fetch = f_ar_valid;
        m_axi_arvalid = (f_ar_valid || l_ar_valid) && !tag_full;
        m_axi_araddr = grant_fetch ? f_ar_addr : l_ar_addr;
        m_axi_arlen = grant_fetch ? f_ar_len : l_ar_len;
        f_ar_ready = m_axi_arready && !tag_full;
        l_ar_ready = m_axi_arready && !tag_full && !grant_fetch;
        f_r_valid = m_axi_rvalid && !tag_empty && tag_head;
        l_r_valid = m_axi_rvalid && !tag_empty && !tag_head;
        m_axi_rready = !tag_empty && (tag_head ? f_r_ready : l_r_ready);

        m_axi_arsize = 3'd3;    // 8-byte beats
        m_axi_arburst = 2'b01;  // INCR
        m_axi_arcache = 4'b0011;
        m_axi_arprot = 3'b000;
        m_axi_awsize = 3'd3;
        m_axi_awburst = 2'b01;
        m_axi_awcache = 4'b0011;
        m_axi_awprot = 3'b000;
    end
endmodule
