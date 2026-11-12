`timescale 1ns/1ps

// MMIO job registers of the CVA6 scalar core (SCALAR_IMPL = "CVA6"). Firmware
// running on CVA6 describes one GEMM job in the shape and address registers
// and writes DOORBELL; this module issues the job to the matrix core and the
// tensor DMA (cva6_dma) with the same handshake as the EX stage of the CUSTOM
// core (custom_scalar_core):
//
//   - ctl_start and ctl_m/n/k come straight from flip-flops (the 100 MHz
//     margin of the 16 x 16 array leaves no room for logic in front of the
//     matrix controller);
//   - a job issues when the matrix controller accepts (ctl_accept), the DMA can
//     take both its load and its store, no fault is latched, and the two-cycle
//     cooldown after the previous issue has expired (ACCEPT is stale for two
//     cycles after an issue);
//   - the A/B byte counts and the C word count are computed when the job is
//     latched, so no multiplier sits on the issue path.
//
// DOORBELL latches the job into a one-entry pending register; its AXI write
// response is held while that register is still occupied by the previous job
// (the job queue is full). A shape outside 1..ROWS x 1..COLUMNS x 1..MAX_K is
// ERR_BAD_SHAPE and an A/B/C address that is not 8-byte aligned is
// ERR_MISALIGNED, as in custom_decode; such a job is not issued.
//
// Faults (matrix core, DMA read, DMA write, bad job) latch the first error code
// while the program runs. A latched fault drops the pending job and every later
// DOORBELL (answered OKAY so the firmware does not trap), and is reported to the
// top through `error`, which ends the program.
//
// AXI4 slave, plain signals (no pulp structs) so this module compiles without
// the CVA6 packages. 32-bit registers on the 64-bit data bus: the lane is
// selected by address bit 2, and reads return the register in both lanes.
// Only single-beat accesses are supported; a burst, an unknown offset or a
// write to a read-only register is answered SLVERR (every beat of a burst).
// One write and one read are handled at a time.
//
// Register map (offsets in the 4 KiB window; docs/plans/2026-1112-003 section 5):
//   0x00 SHAPE_MN   RW  M [15:0], N [31:16]
//   0x04 SHAPE_K    RW  K [15:0]
//   0x08 ADDR_A     RW  A operand address
//   0x0C ADDR_B     RW  B operand address
//   0x10 ADDR_C     RW  C result address
//   0x14 DOORBELL   W   issue one job from the registers above
//   0x18 STATUS     R   [0] busy, [1] error (sticky), [15:8] error code,
//                       [23:16] jobs outstanding (pending + issued, not
//                       written back)
//   0x1C FENCE      R   as STATUS, returned only once every issued job has been
//                       written back (or a fault is latched)
//   0x20 DONE       W   [7:0] error code, 0 = success; ends the program
//   0x24 DATA_BASE  R   data_base from the CSR (job descriptor table)
//   0x28 PROG_LEN   R   prog_len from the CSR
// Error codes match src/vendors/npu_pynq/spec/abi.py and custom_decode /
// custom_controller.
// Not instantiated by the default (CUSTOM) flow, where it is linted on its own.
/* verilator lint_off MULTITOP */
module cva6_npu_mmio #(
    parameter integer ROWS = 16,
    parameter integer COLUMNS = 16,
    parameter integer MAX_K = 256,
    parameter integer ID_WIDTH = 4
) (
/* verilator lint_on MULTITOP */
    input  logic                clk,
    input  logic                rst_n,
    // Program start (clears the fault, the counters and the pending job) and
    // run state, from scalar_core_cva6.
    input  logic                start,
    input  logic                running,
    input  logic [31:0]         data_base,
    input  logic [31:0]         prog_len,

    // AXI4 slave.
    input  logic [ID_WIDTH-1:0] s_awid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0]         s_awaddr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [7:0]          s_awlen,
    input  logic                s_awvalid,
    output logic                s_awready,
    input  logic [63:0]         s_wdata,
    input  logic [7:0]          s_wstrb,
    input  logic                s_wlast,
    input  logic                s_wvalid,
    output logic                s_wready,
    output logic [ID_WIDTH-1:0] s_bid,
    output logic [1:0]          s_bresp,
    output logic                s_bvalid,
    input  logic                s_bready,
    input  logic [ID_WIDTH-1:0] s_arid,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0]         s_araddr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [7:0]          s_arlen,
    input  logic                s_arvalid,
    output logic                s_arready,
    output logic [ID_WIDTH-1:0] s_rid,
    output logic [63:0]         s_rdata,
    output logic [1:0]          s_rresp,
    output logic                s_rlast,
    output logic                s_rvalid,
    input  logic                s_rready,

    // Matrix core dispatch.
    output logic                ctl_start,
    output logic [15:0]         ctl_m,
    output logic [15:0]         ctl_n,
    output logic [15:0]         ctl_k,
    input  logic                ctl_accept,
    input  logic                ctl_error,
    input  logic [7:0]          ctl_error_code,

    // Tensor DMA (cva6_dma).
    output logic                ld_valid,
    input  logic                ld_ready,
    output logic [31:0]         ld_a_addr,
    output logic [12:0]         ld_a_bytes,
    output logic [31:0]         ld_b_addr,
    output logic [12:0]         ld_b_bytes,
    output logic                st_valid,
    input  logic                st_ready,
    output logic [31:0]         st_c_addr,
    output logic [8:0]          st_words,
    input  logic                jobs_written,
    input  logic                read_error,
    input  logic                write_error,

    // To scalar_core_cva6.
    output logic                done,        // one-cycle pulse on a DONE write
    output logic [7:0]          done_code,
    output logic                error,       // sticky fault
    output logic [7:0]          error_code,
    output logic                busy,
    output logic                drained,
    output logic [31:0]         jobs_issued, // since start
    output logic [31:0]         jobs_done    // written back since start
);
    localparam logic [9:0] REG_SHAPE_MN  = 10'h000;
    localparam logic [9:0] REG_SHAPE_K   = 10'h001;
    localparam logic [9:0] REG_ADDR_A    = 10'h002;
    localparam logic [9:0] REG_ADDR_B    = 10'h003;
    localparam logic [9:0] REG_ADDR_C    = 10'h004;
    localparam logic [9:0] REG_DOORBELL  = 10'h005;
    localparam logic [9:0] REG_STATUS    = 10'h006;
    localparam logic [9:0] REG_FENCE     = 10'h007;
    localparam logic [9:0] REG_DONE      = 10'h008;
    localparam logic [9:0] REG_DATA_BASE = 10'h009;
    localparam logic [9:0] REG_PROG_LEN  = 10'h00A;

    localparam logic [7:0] ERR_BAD_SHAPE  = 8'h11;
    localparam logic [7:0] ERR_MISALIGNED = 8'h12;
    localparam logic [7:0] ERR_AXI_READ   = 8'h13;
    localparam logic [7:0] ERR_AXI_WRITE  = 8'h14;

    localparam logic [1:0] RESP_OKAY   = 2'b00;
    localparam logic [1:0] RESP_SLVERR = 2'b10;

    // ------------------------------------------------------------ registers
    logic [15:0] shape_m, shape_n, shape_k;
    logic [31:0] addr_a, addr_b, addr_c;

    // ---------------------------------------------------------- job state
    logic        job_valid;
    logic [15:0] job_m, job_n, job_k;
    logic [31:0] job_a, job_b, job_c;
    logic [12:0] job_a_bytes, job_b_bytes;
    logic [8:0]  job_c_words;
    logic        job_ready, issue;
    logic [1:0]  cooldown;
    logic [31:0] issued_total, written_total, written_at_start, issued_at_start;
    logic [7:0]  outstanding;

    logic        shape_ok, aligned_ok;
    logic        fault, error_latched;
    logic [7:0]  fault_code, error_code_latched;
    logic        job_fault;              // a DOORBELL with a bad job, this cycle
    logic [7:0]  job_fault_code;

    always_comb begin
        shape_ok = (shape_m != 16'd0) && ({16'd0, shape_m} <= 32'(ROWS)) &&
            (shape_n != 16'd0) && ({16'd0, shape_n} <= 32'(COLUMNS)) &&
            (shape_k != 16'd0) && ({16'd0, shape_k} <= 32'(MAX_K));
        aligned_ok = (addr_a[2:0] == 3'd0) && (addr_b[2:0] == 3'd0) &&
            (addr_c[2:0] == 3'd0);
        job_fault_code = !shape_ok ? ERR_BAD_SHAPE : ERR_MISALIGNED;

        job_ready = ctl_accept && ld_ready && st_ready && !error_latched &&
            (cooldown == 2'd0);
        issue = job_valid && job_ready;

        outstanding = 8'(issued_total - written_total) + {7'd0, job_valid};
        drained = !job_valid && (issued_total == written_total);
        busy = !drained;
        jobs_issued = issued_total - issued_at_start;
        jobs_done = written_total - written_at_start;

        fault = ctl_error || read_error || write_error || job_fault;
        fault_code = ctl_error ? ctl_error_code :
            read_error ? ERR_AXI_READ :
            write_error ? ERR_AXI_WRITE : job_fault_code;
        error = error_latched;
        error_code = error_code_latched;

        ld_valid = issue;
        st_valid = issue;
        ld_a_addr = job_a;
        ld_a_bytes = job_a_bytes;
        ld_b_addr = job_b;
        ld_b_bytes = job_b_bytes;
        st_c_addr = job_c;
        st_words = job_c_words;
    end

    // -------------------------------------------------------- write channel
    typedef enum logic [1:0] {W_ADDR, W_DATA, W_EXEC, W_RESP} wstate_t;
    wstate_t     wstate;
    logic [ID_WIDTH-1:0] w_id;
    logic [9:0]  w_reg;
    logic        w_lane_hi, w_burst;
    logic [31:0] w_data;
    logic [3:0]  w_strb;
    logic [1:0]  w_resp;
    logic [31:0] w_mask;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] w_merged_k;
    /* verilator lint_on UNUSEDSIGNAL */

    assign s_awready = (wstate == W_ADDR);
    assign s_wready = (wstate == W_DATA);
    assign s_bvalid = (wstate == W_RESP);
    assign s_bid = w_id;
    assign s_bresp = w_resp;

    always_comb begin
        w_mask = {{8{w_strb[3]}}, {8{w_strb[2]}}, {8{w_strb[1]}}, {8{w_strb[0]}}};
        w_merged_k = (w_data & w_mask) | ({16'd0, shape_k} & ~w_mask);
        job_fault = (wstate == W_EXEC) && (w_reg == REG_DOORBELL) &&
            !error_latched && !(shape_ok && aligned_ok);
    end

    function automatic [31:0] merge(input [31:0] old, input [31:0] data,
                                    input [31:0] mask);
        merge = (old & ~mask) | (data & mask);
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wstate <= W_ADDR;
            w_id <= '0;
            w_reg <= 10'd0;
            w_lane_hi <= 1'b0;
            w_burst <= 1'b0;
            w_data <= 32'd0;
            w_strb <= 4'd0;
            w_resp <= RESP_OKAY;
            shape_m <= 16'd0; shape_n <= 16'd0; shape_k <= 16'd0;
            addr_a <= 32'd0; addr_b <= 32'd0; addr_c <= 32'd0;
            job_valid <= 1'b0;
            job_m <= 16'd0; job_n <= 16'd0; job_k <= 16'd0;
            job_a <= 32'd0; job_b <= 32'd0; job_c <= 32'd0;
            job_a_bytes <= 13'd0; job_b_bytes <= 13'd0; job_c_words <= 9'd0;
            done <= 1'b0;
            done_code <= 8'd0;
        end else begin
            done <= 1'b0;
            if (issue)
                job_valid <= 1'b0;

            case (wstate)
                W_ADDR: if (s_awvalid) begin
                    w_id <= s_awid;
                    w_reg <= s_awaddr[11:2];
                    w_lane_hi <= s_awaddr[2];
                    w_burst <= (s_awlen != 8'd0);
                    wstate <= W_DATA;
                end
                W_DATA: if (s_wvalid) begin
                    w_data <= w_lane_hi ? s_wdata[63:32] : s_wdata[31:0];
                    w_strb <= w_lane_hi ? s_wstrb[7:4] : s_wstrb[3:0];
                    if (w_burst) begin
                        // Sink the whole burst, then answer SLVERR.
                        if (s_wlast) begin
                            w_resp <= RESP_SLVERR;
                            wstate <= W_RESP;
                        end
                    end else begin
                        wstate <= W_EXEC;
                    end
                end
                W_EXEC: begin
                    w_resp <= RESP_OKAY;
                    wstate <= W_RESP;
                    case (w_reg)
                        REG_SHAPE_MN: {shape_n, shape_m} <= merge({shape_n, shape_m}, w_data, w_mask);
                        REG_SHAPE_K: shape_k <= w_merged_k[15:0];
                        REG_ADDR_A: addr_a <= merge(addr_a, w_data, w_mask);
                        REG_ADDR_B: addr_b <= merge(addr_b, w_data, w_mask);
                        REG_ADDR_C: addr_c <= merge(addr_c, w_data, w_mask);
                        REG_DOORBELL: begin
                            // A latched fault drops the job; a bad job is a
                            // fault (job_fault). Otherwise hold the response
                            // until the pending entry is free.
                            if (!error_latched && shape_ok && aligned_ok) begin
                                if (job_valid && !issue) begin
                                    wstate <= W_EXEC;
                                end else begin
                                    job_valid <= 1'b1;
                                    job_m <= shape_m;
                                    job_n <= shape_n;
                                    job_k <= shape_k;
                                    job_a <= addr_a;
                                    job_b <= addr_b;
                                    job_c <= addr_c;
                                    job_a_bytes <= 13'(shape_m * shape_k);
                                    job_b_bytes <= 13'(shape_k * shape_n);
                                    job_c_words <= 9'(shape_m * shape_n);
                                end
                            end
                        end
                        REG_DONE: begin
                            done <= 1'b1;
                            done_code <= w_data[7:0];
                        end
                        default: w_resp <= RESP_SLVERR;
                    endcase
                end
                W_RESP: if (s_bready)
                    wstate <= W_ADDR;
                default: wstate <= W_ADDR;
            endcase

            // A fault drops the job that has not issued yet; a new start
            // clears it.
            if (fault || start)
                job_valid <= 1'b0;
        end
    end

    // --------------------------------------------------------- read channel
    typedef enum logic [0:0] {R_ADDR, R_DATA} rstate_t;
    rstate_t     rstate;
    logic [ID_WIDTH-1:0] r_id;
    logic [9:0]  r_reg;
    logic        r_burst;
    logic [7:0]  r_len, r_beat;
    logic        r_valid_q;
    logic [31:0] r_word;
    logic        r_known, r_ready_to_answer;
    logic [31:0] status_word;

    always_comb begin
        status_word = {8'd0, outstanding, error_code_latched, 6'd0,
                       error_latched, busy};
        r_known = 1'b1;
        case (r_reg)
            REG_SHAPE_MN:  r_word = {shape_n, shape_m};
            REG_SHAPE_K:   r_word = {16'd0, shape_k};
            REG_ADDR_A:    r_word = addr_a;
            REG_ADDR_B:    r_word = addr_b;
            REG_ADDR_C:    r_word = addr_c;
            REG_STATUS,
            REG_FENCE:     r_word = status_word;
            REG_DATA_BASE: r_word = data_base;
            REG_PROG_LEN:  r_word = prog_len;
            default: begin
                r_word = 32'd0;
                r_known = 1'b0;
            end
        endcase
        // FENCE answers once everything issued has drained, or when a fault
        // makes waiting pointless.
        r_ready_to_answer = r_burst || (r_reg != REG_FENCE) || drained ||
            error_latched;
    end

    assign s_arready = (rstate == R_ADDR);
    assign s_rvalid = r_valid_q;
    assign s_rid = r_id;
    assign s_rlast = (r_beat == r_len);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rstate <= R_ADDR;
            r_id <= '0;
            r_reg <= 10'd0;
            r_burst <= 1'b0;
            r_len <= 8'd0;
            r_beat <= 8'd0;
            r_valid_q <= 1'b0;
            s_rdata <= 64'd0;
            s_rresp <= RESP_OKAY;
        end else begin
            case (rstate)
                R_ADDR: if (s_arvalid) begin
                    r_id <= s_arid;
                    r_reg <= s_araddr[11:2];
                    r_burst <= (s_arlen != 8'd0);
                    r_len <= s_arlen;
                    r_beat <= 8'd0;
                    rstate <= R_DATA;
                end
                R_DATA: begin
                    // Data and response are captured when RVALID rises and
                    // held until the beat is taken.
                    if (!r_valid_q && r_ready_to_answer) begin
                        r_valid_q <= 1'b1;
                        s_rdata <= r_burst ? 64'd0 : {r_word, r_word};
                        s_rresp <= (r_burst || !r_known) ? RESP_SLVERR : RESP_OKAY;
                    end
                    if (r_valid_q && s_rready) begin
                        r_valid_q <= 1'b0;
                        if (r_beat == r_len)
                            rstate <= R_ADDR;
                        else
                            r_beat <= r_beat + 8'd1;
                    end
                end
                default: rstate <= R_ADDR;
            endcase
        end
    end

    // ---------------------------------------------------------------- issue
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctl_start <= 1'b0;
            ctl_m <= 16'd0;
            ctl_n <= 16'd0;
            ctl_k <= 16'd0;
            cooldown <= 2'd0;
            issued_total <= 32'd0;
            written_total <= 32'd0;
            issued_at_start <= 32'd0;
            written_at_start <= 32'd0;
            error_latched <= 1'b0;
            error_code_latched <= 8'd0;
        end else begin
            if (!start)
                cooldown <= issue ? 2'd2 : (cooldown != 2'd0 ? cooldown - 2'd1 : 2'd0);
            ctl_start <= issue;
            if (issue) begin
                ctl_m <= job_m;
                ctl_n <= job_n;
                ctl_k <= job_k;
            end
            if (issue)
                issued_total <= issued_total + 32'd1;
            if (jobs_written)
                written_total <= written_total + 32'd1;

            if (start) begin
                error_latched <= 1'b0;
                error_code_latched <= 8'd0;
                issued_at_start <= issued_total;
                written_at_start <= written_total;
            end else if (running && fault && !error_latched) begin
                error_latched <= 1'b1;
                error_code_latched <= fault_code;
            end
        end
    end
endmodule
