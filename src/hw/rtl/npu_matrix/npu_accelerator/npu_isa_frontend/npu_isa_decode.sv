`timescale 1ns/1ps

// ID stage: decodes instruction words from the IF queue against the
// architectural state (shape, A/B/C offsets, post-increments) and emits one
// physical matrix job per GEMM execution. REPEAT n makes the following word
// execute n times; for GEMM that is n jobs, walking tiles by post-increment.
// FENCE and END wait until the execute stage has written every job back.
// Opcodes and error codes match src/isa/isa.py.
module npu_isa_decode #(
    parameter integer ROWS = 16,
    parameter integer COLUMNS = 16,
    parameter integer MAX_K = 256
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        fault,
    input  logic [31:0] data_base,

    input  logic        instr_valid,
    input  logic [63:0] instr_data,
    output logic        instr_ready,

    input  logic        ex_idle,
    output logic        job_valid,
    input  logic        job_ready,
    output logic [7:0]  job_m,
    output logic [7:0]  job_n,
    output logic [15:0] job_k,
    output logic [31:0] job_a,
    output logic [31:0] job_b,
    output logic [31:0] job_c,
    // Byte counts of the job, computed here so no multiplier sits on the issue path.
    output logic [12:0] job_a_bytes,
    output logic [12:0] job_b_bytes,
    output logic [8:0]  job_c_words,

    output logic        running,
    output logic        done,
    output logic        error,
    output logic [7:0]  error_code,
    output logic [31:0] pc,
    output logic [31:0] instructions
);
    localparam logic [7:0] OP_NOP = 8'h00;
    localparam logic [7:0] OP_END = 8'h01;
    localparam logic [7:0] OP_FENCE = 8'h02;
    localparam logic [7:0] OP_SHAPE = 8'h10;
    localparam logic [7:0] OP_ADDR_A = 8'h11;
    localparam logic [7:0] OP_ADDR_B = 8'h12;
    localparam logic [7:0] OP_ADDR_C = 8'h13;
    localparam logic [7:0] OP_INCR = 8'h14;
    localparam logic [7:0] OP_REPEAT = 8'h15;
    localparam logic [7:0] OP_GEMM = 8'h20;

    localparam logic [7:0] ERR_ILLEGAL_OPCODE = 8'h10;
    localparam logic [7:0] ERR_BAD_SHAPE = 8'h11;
    localparam logic [7:0] ERR_MISALIGNED = 8'h12;

    // The word being executed and how many more times it runs.
    logic        cur_valid;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [63:0] cur_word;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [31:0] cur_count;
    logic        rep_valid;
    logic [31:0] rep_count;

    logic [7:0]  m, n;
    logic [15:0] k;
    logic [31:0] a_off, b_off, c_off;
    logic [18:0] a_inc, b_inc, c_inc;

    logic [7:0]  op;
    logic        shape_ok, drained, emit;

    always_comb begin
        op = cur_word[63:56];
        shape_ok = (m != 8'd0) && ({24'd0, m} <= 32'(ROWS)) &&
            (n != 8'd0) && ({24'd0, n} <= 32'(COLUMNS)) &&
            (k != 16'd0) && ({16'd0, k} <= 32'(MAX_K));
        drained = ex_idle && !job_valid;
        emit = running && cur_valid && (op == OP_GEMM) && shape_ok &&
            (!job_valid || job_ready);
        instr_ready = running && !cur_valid;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cur_valid <= 1'b0;
            cur_word <= 64'd0;
            cur_count <= 32'd0;
            rep_valid <= 1'b0;
            rep_count <= 32'd0;
            m <= 8'd0; n <= 8'd0; k <= 16'd0;
            a_off <= 32'd0; b_off <= 32'd0; c_off <= 32'd0;
            a_inc <= '0; b_inc <= '0; c_inc <= '0;
            job_valid <= 1'b0;
            job_m <= 8'd0; job_n <= 8'd0; job_k <= 16'd0;
            job_a <= 32'd0; job_b <= 32'd0; job_c <= 32'd0;
            job_a_bytes <= 13'd0; job_b_bytes <= 13'd0; job_c_words <= 9'd0;
            running <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            error_code <= 8'd0;
            pc <= 32'd0;
            instructions <= 32'd0;
        end else if (start) begin
            cur_valid <= 1'b0;
            rep_valid <= 1'b0;
            m <= 8'd0; n <= 8'd0; k <= 16'd0;
            a_off <= 32'd0; b_off <= 32'd0; c_off <= 32'd0;
            a_inc <= '0; b_inc <= '0; c_inc <= '0;
            job_valid <= 1'b0;
            running <= 1'b1;
            done <= 1'b0;
            error <= 1'b0;
            error_code <= 8'd0;
            pc <= 32'd0;
            instructions <= 32'd0;
        end else begin
            if (job_valid && job_ready)
                job_valid <= 1'b0;

            if (instr_valid && instr_ready) begin
                pc <= pc + 32'd1;
                instructions <= instructions + 32'd1;
                if (instr_data[63:56] == OP_REPEAT) begin
                    rep_valid <= 1'b1;
                    rep_count <= (instr_data[31:0] == 32'd0) ? 32'd1 : instr_data[31:0];
                end else begin
                    cur_valid <= 1'b1;
                    cur_word <= instr_data;
                    cur_count <= rep_valid ? rep_count : 32'd1;
                    rep_valid <= 1'b0;
                end
            end

            if (running && cur_valid) begin
                case (op)
                    OP_NOP: cur_valid <= 1'b0;
                    OP_SHAPE: begin
                        m <= cur_word[31:24];
                        n <= cur_word[23:16];
                        k <= cur_word[15:0];
                        cur_valid <= 1'b0;
                    end
                    OP_ADDR_A, OP_ADDR_B, OP_ADDR_C: begin
                        if (cur_word[2:0] != 3'd0) begin
                            error <= 1'b1;
                            error_code <= ERR_MISALIGNED;
                            running <= 1'b0;
                        end
                        if (op == OP_ADDR_A) a_off <= cur_word[31:0];
                        if (op == OP_ADDR_B) b_off <= cur_word[31:0];
                        if (op == OP_ADDR_C) c_off <= cur_word[31:0];
                        cur_valid <= 1'b0;
                    end
                    OP_INCR: begin
                        a_inc <= {cur_word[15:0], 3'b000};
                        b_inc <= {cur_word[31:16], 3'b000};
                        c_inc <= {cur_word[47:32], 3'b000};
                        cur_valid <= 1'b0;
                    end
                    OP_FENCE: begin
                        if (drained)
                            cur_valid <= 1'b0;
                    end
                    OP_END: begin
                        if (drained) begin
                            cur_valid <= 1'b0;
                            running <= 1'b0;
                            done <= 1'b1;
                        end
                    end
                    OP_GEMM: begin
                        if (!shape_ok) begin
                            error <= 1'b1;
                            error_code <= ERR_BAD_SHAPE;
                            running <= 1'b0;
                        end else if (emit) begin
                            job_valid <= 1'b1;
                            job_m <= m;
                            job_n <= n;
                            job_k <= k;
                            job_a <= data_base + a_off;
                            job_b <= data_base + b_off;
                            job_c <= data_base + c_off;
                            job_a_bytes <= 13'(m * k);
                            job_b_bytes <= 13'(k * n);
                            job_c_words <= 9'(m * n);
                            if (cur_word[0]) a_off <= a_off + {13'd0, a_inc};
                            if (cur_word[1]) b_off <= b_off + {13'd0, b_inc};
                            if (cur_word[2]) c_off <= c_off + {13'd0, c_inc};
                            cur_count <= cur_count - 32'd1;
                            if (cur_count == 32'd1)
                                cur_valid <= 1'b0;
                        end
                    end
                    default: begin
                        error <= 1'b1;
                        error_code <= ERR_ILLEGAL_OPCODE;
                        running <= 1'b0;
                    end
                endcase
            end

            // A fault elsewhere (fetch, load/store, controller) stops decode;
            // the front end reports its code.
            if (fault) begin
                running <= 1'b0;
                job_valid <= 1'b0;
            end
        end
    end
endmodule
