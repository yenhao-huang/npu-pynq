`timescale 1ns/1ps

// Instruction-stream front end of the NPU.
//
//   IF  npu_isa_fetch   PC and prefetch queue, reads PROG_ADDR..+8*PROG_LEN
//   ID  npu_isa_decode  architectural state, REPEAT, one job per GEMM
//   EX  this module     starts the matrix controller for each job and hands
//                       its A/B/C addresses to npu_isa_lsu, which moves the
//                       operands from DDR into the controller's input stream
//                       and its results back to DDR
//
// IF and the LSU share one AXI4 master. Read bursts carry a one-bit tag in an
// in-order FIFO (one AXI ID, so R returns in AR order) that routes R beats
// back to their requester.
module npu_isa_frontend #(
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
    localparam logic [7:0] ERR_AXI_READ = 8'h13;
    localparam logic [7:0] ERR_AXI_WRITE = 8'h14;
    localparam logic [7:0] ERR_FETCH = 8'h15;

    logic start;
    logic dec_running, dec_done, dec_error;
    logic [7:0] dec_error_code;
    logic instr_valid, instr_ready;
    logic [63:0] instr_data;
    logic job_valid, job_ready;
    logic [7:0] job_m, job_n;
    logic [15:0] job_k;
    logic [31:0] job_a, job_b, job_c;
    logic fetch_error, read_error, write_error, job_written;
    logic fault, fault_latched;
    logic [7:0] fault_code, isa_fault_code;
    logic [31:0] jobs_issued, jobs_written;
    logic ex_idle;

    logic f_ar_valid, f_ar_ready, f_r_valid, f_r_ready;
    logic [31:0] f_ar_addr;
    logic [7:0] f_ar_len;
    logic l_ar_valid, l_ar_ready, l_r_valid, l_r_ready;
    logic [31:0] l_ar_addr;
    logic [7:0] l_ar_len;
    logic ld_ready, st_ready;
    logic [12:0] a_bytes, b_bytes;
    logic [8:0] c_words;
    // START reaches the controller one register later (npu_accelerator), so its
    // ACCEPT is stale for two cycles after an issue.
    logic [1:0] cooldown;

    assign start = isa_start && !isa_busy;
    assign ex_idle = (jobs_issued == jobs_written);

    npu_isa_fetch fetch (
        .clk(clk), .rst_n(rst_n),
        .start(start), .stop(!dec_running),
        .prog_addr(prog_addr), .prog_len(prog_len),
        .ar_valid(f_ar_valid), .ar_ready(f_ar_ready),
        .ar_addr(f_ar_addr), .ar_len(f_ar_len),
        .r_valid(f_r_valid), .r_ready(f_r_ready),
        .r_data(m_axi_rdata), .r_resp(m_axi_rresp), .r_last(m_axi_rlast),
        .instr_valid(instr_valid), .instr_data(instr_data),
        .instr_ready(instr_ready), .fetch_error(fetch_error)
    );

    npu_isa_decode #(.ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K)) decode (
        .clk(clk), .rst_n(rst_n),
        .start(start), .fault(fault), .data_base(data_base),
        .instr_valid(instr_valid), .instr_data(instr_data),
        .instr_ready(instr_ready),
        .ex_idle(ex_idle),
        .job_valid(job_valid), .job_ready(job_ready),
        .job_m(job_m), .job_n(job_n), .job_k(job_k),
        .job_a(job_a), .job_b(job_b), .job_c(job_c),
        .job_a_bytes(a_bytes), .job_b_bytes(b_bytes), .job_c_words(c_words),
        .running(dec_running), .done(dec_done),
        .error(dec_error), .error_code(dec_error_code),
        .pc(isa_pc), .instructions(isa_instructions)
    );

    // ------------------------------------------------------------- execute
    // A job issues when the controller has a free queue entry and the LSU can
    // take both its load and its store; START and the shape reach the
    // controller in the same cycle, and the controller's ACCEPT reflects the
    // new occupancy one cycle later, so back-to-back issue is safe.
    always_comb begin
        job_ready = ctl_accept && ld_ready && st_ready && !fault_latched && (cooldown == 2'd0);
        ctl_start = job_valid && job_ready;
        ctl_m = {8'd0, job_m};
        ctl_n = {8'd0, job_n};
        ctl_k = job_k;
    end

    npu_isa_lsu lsu (
        .clk(clk), .rst_n(rst_n), .clear_errors(start),
        .ld_valid(ctl_start), .ld_ready(ld_ready),
        .ld_a_addr(job_a), .ld_a_bytes(a_bytes),
        .ld_b_addr(job_b), .ld_b_bytes(b_bytes),
        .st_valid(ctl_start), .st_ready(st_ready),
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

    npu_isa_fifo #(.WIDTH(1), .DEPTH(16)) read_tags (
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

    // -------------------------------------------------------------- status
    always_comb begin
        fault = fetch_error || read_error || write_error || ctl_error;
        fault_code = ctl_error ? ctl_error_code :
            fetch_error ? ERR_FETCH :
            read_error ? ERR_AXI_READ : ERR_AXI_WRITE;
        isa_running = dec_running;
        isa_busy = dec_running || job_valid || !ex_idle;
        isa_done = dec_done;
        isa_error = dec_error || fault_latched;
        isa_error_code = fault_latched ? isa_fault_code : dec_error_code;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            jobs_issued <= 32'd0;
            jobs_written <= 32'd0;
            cooldown <= 2'd0;
            fault_latched <= 1'b0;
            isa_fault_code <= 8'd0;
            isa_cycles <= 32'd0;
        end else if (start) begin
            fault_latched <= 1'b0;
            isa_fault_code <= 8'd0;
            isa_cycles <= 32'd0;
        end else begin
            if (ctl_start)
                jobs_issued <= jobs_issued + 32'd1;
            cooldown <= ctl_start ? 2'd2 : (cooldown != 2'd0 ? cooldown - 2'd1 : 2'd0);
            if (job_written)
                jobs_written <= jobs_written + 32'd1;
            if (dec_running && fault && !fault_latched) begin
                fault_latched <= 1'b1;
                isa_fault_code <= fault_code;
            end
            if (dec_running)
                isa_cycles <= isa_cycles + 32'd1;
        end
    end

    // Jobs written back since the last start.
    logic [31:0] jobs_at_start;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            jobs_at_start <= 32'd0;
        else if (start)
            jobs_at_start <= jobs_written;
    end
    assign isa_jobs = jobs_written - jobs_at_start;
endmodule
