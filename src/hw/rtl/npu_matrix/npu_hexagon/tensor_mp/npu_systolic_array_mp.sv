`timescale 1ns/1ps

// ROWS x COLUMNS output-stationary systolic array of npu_pe_mp, with the same
// edge wiring as npu_systolic_array: row r's A stream enters PE (r, 0) and
// flows right, column c's B stream enters PE (0, c) and flows down, and PE
// (r, c) accumulates C[r][c]. Operand lanes are 16 bits; prec (broadcast to
// every PE) selects INT8 / INT16 / FP16 as documented in npu_pe_mp.
// accumulators packs the 48-bit accumulator of PE (r, c) at index
// r * COLUMNS + c.
//
// Driving the array (controller contract), for C = A (ROWS x K) * B (K x
// COLUMNS):
// 1. Job start: set prec and assert clear for at least one clock. clear zeroes
//    every accumulator, operand register and operation in flight and wins
//    over enable; edge inputs are ignored meanwhile. Keep prec constant from
//    the clear until the results have been read.
// 2. Beats: a beat is a clock with enable = 1; with enable = 0 the edge inputs
//    are ignored and the wavefront holds (an FP32 addition already issued
//    still completes). Pair k is assigned beat T_k, with T_0 any beat after
//    the clear and T_k+1 - T_k = S_k. In beat t, drive
//    a_in[r] = A[r][k] with a_valid_in[r] = 1 if t = T_k + r, and
//    b_in[c] = B[k][c] with b_valid_in[c] = 1 if t = T_k + c; otherwise
//    drive valid = 0 (data are don't care). PE (r, c) then sees pair k in
//    beat T_k + r + c, and only a pair with both valids set is accumulated.
// 3. Spacing: S_k >= 1 in the INT modes; in FP16 mode S_k >= 3 (npu_pe_mp's
//    FP16 MAC interval: two valid = 0 bubble beats after every valid beat).
//    S_k may vary with k but must be the same on every edge stream. FP16
//    spacing is not checked in hardware: closer pairs give wrong sums (and,
//    with a shared adder, collide with a neighbour's issue).
// 4. Drain: the last pair reaches PE (ROWS-1, COLUMNS-1) in beat
//    tL = T_K-1 + ROWS + COLUMNS - 2. All accumulators are final after beat
//    tL + 1 (INT) or after beat tL + 2 plus three further clocks, with
//    enable either way (FP16); the beats after T_K-1 carry valid = 0.
// 5. The results then stay constant (enable high or low, valid = 0 beats)
//    until the next clear or reset. INT8 sums are sign-extended INT32 values
//    in accumulator[47:0], INT16 sums INT48 values, FP16 sums fp32 bits in
//    accumulator[31:0] with [47:32] = 0.
//
// FP32 adder sharing: N = FP_ADDER_SHARE (1..3) neighbouring PEs of a row,
// columns [g*N, g*N+N-1], share one FP32 accumulation adder (npu_fp32_add with
// SUBNORMALS = 0, see npu_pe_mp). PE (r, c + i) issues its addition for pair k
// in enabled cycle T0' + r + c + i + sum(S over earlier k); with every S >= 3
// and i < 3, two issues in a group never fall in the same cycle, so the adder
// takes the one requesting PE's operands (one-hot AND-OR) and every PE picks
// up its own result three clocks later. Results are bit-identical to one
// adder per PE; the FP16 spacing rule above is the only requirement (not
// checked in hardware, as in npu_pe_mp). FP_ADDER_SHARE = 1 gives every PE
// its own adder. The number of adders is ROWS * ceil(COLUMNS / FP_ADDER_SHARE)
// (for COLUMNS = 4, shares 2 and 3 both give 2 per row).
//
// PPA (Yosys synth_xilinx estimate, 4 x 4, DSP INT accumulation): share 3
// ~12.2-12.7k LUT / 3.3k FF / 16 DSP (8 adders), share 1 ~16-17k LUT / 4.4k
// FF (16 adders); Yosys LUT counts vary by ~5% with the netlist order.
// One SUBNORMALS = 0 FP32 adder ~0.48k LUT and 147 FF stand-alone (~0.6k
// with the operand select in front); a PE without adder ~0.36-0.43k LUT.
//
// FP16 throughput option evaluated and not implemented: three interleaved
// partial accumulators per PE (one FP16 MAC per cycle instead of 1/3; the
// adder is busy every cycle, so no sharing). Estimate per PE: 0.43k + 0.6k
// adder + ~0.08k LUT and ~100 FF for the partial sums, their select and the
// final fold, i.e. ~1.1k LUT per FP16 MAC/cycle, against ~1.9k for share 3
// (0.43k + 0.6k / 3 + mux, at 1/3 MAC/cycle) and ~3.1k for share 1. It is
// 1.7x better per area for FP16 only; INT8/INT16 (one MAC per cycle in both)
// would cost ~1.1k instead of ~0.65k LUT per PE (+70%). It also needs an
// end-of-job drain (fold (p0 + p1) + p2: two dependent additions, ~6 more
// cycles) that npu_pe's interface has no signal for, partial sums indexed by
// pair number (k mod 3, with parking registers for bubbles and stalls), and a
// new golden accumulation order. Since INT8 is the main tensor workload and
// the device is area-bound, the sequential share-3 design stays; a
// dedicated FP16 array variant would be the place for interleaving.
module npu_systolic_array_mp #(
    parameter integer ROWS = 4,
    parameter integer COLUMNS = 4,
    // PEs beyond the budget use LUT multipliers with identical behaviour.
    parameter integer DSP_BUDGET = 192,
    parameter integer FP_ADDER_SHARE = 3
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         clear,
    input  logic                         enable,
    input  logic [1:0]                   prec,
    input  logic [ROWS*16-1:0]           a_in,
    input  logic [ROWS-1:0]              a_valid_in,
    input  logic [COLUMNS*16-1:0]        b_in,
    input  logic [COLUMNS-1:0]           b_valid_in,
    output wire  [ROWS*COLUMNS*48-1:0]   accumulators
);
    wire [15:0] a_data [0:ROWS-1][0:COLUMNS];
    wire        a_valid [0:ROWS-1][0:COLUMNS];
    wire [15:0] b_data [0:ROWS][0:COLUMNS-1];
    wire        b_valid [0:ROWS][0:COLUMNS-1];

    initial begin
        if (ROWS <= 0) begin
            $fatal(1, "npu_systolic_array_mp requires ROWS > 0");
        end
        if (COLUMNS <= 0) begin
            $fatal(1, "npu_systolic_array_mp requires COLUMNS > 0");
        end
        if (FP_ADDER_SHARE < 1 || FP_ADDER_SHARE > 3) begin
            $fatal(1, "npu_systolic_array_mp FP_ADDER_SHARE must be 1, 2 or 3");
        end
    end

    localparam integer GROUPS = (COLUMNS + FP_ADDER_SHARE - 1) / FP_ADDER_SHARE;

    wire        fp_req [0:ROWS-1][0:COLUMNS-1];
    wire [31:0] fp_a [0:ROWS-1][0:COLUMNS-1];
    wire [31:0] fp_b [0:ROWS-1][0:COLUMNS-1];
    wire [31:0] fp_y [0:ROWS-1][0:GROUPS-1];

    genvar row;
    genvar column;
    generate
        for (row = 0; row < ROWS; row = row + 1) begin : gen_a_edges
            assign a_data[row][0] = a_in[row*16 +: 16];
            assign a_valid[row][0] = a_valid_in[row];
        end

        for (column = 0; column < COLUMNS; column = column + 1) begin : gen_b_edges
            assign b_data[0][column] = b_in[column*16 +: 16];
            assign b_valid[0][column] = b_valid_in[column];
        end

        for (row = 0; row < ROWS; row = row + 1) begin : gen_rows
            for (column = 0; column < COLUMNS; column = column + 1) begin : gen_columns
                npu_pe_mp #(
                    .DSP_STYLE((row*COLUMNS+column < DSP_BUDGET) ? "yes" : "no"),
                    .SHARED_FP_ADDER(1)
                ) pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .clear(clear),
                    .enable(enable),
                    .prec(prec),
                    .a_in(a_data[row][column]),
                    .b_in(b_data[row][column]),
                    .a_valid_in(a_valid[row][column]),
                    .b_valid_in(b_valid[row][column]),
                    .a_out(a_data[row][column+1]),
                    .b_out(b_data[row+1][column]),
                    .a_valid_out(a_valid[row][column+1]),
                    .b_valid_out(b_valid[row+1][column]),
                    .accumulator(accumulators[(row*COLUMNS+column)*48 +: 48]),
                    .fp_add_req(fp_req[row][column]),
                    .fp_add_a(fp_a[row][column]),
                    .fp_add_b(fp_b[row][column]),
                    .fp_add_y(fp_y[row][column / FP_ADDER_SHARE])
                );
            end
        end

        for (row = 0; row < ROWS; row = row + 1) begin : gen_adder_rows
            for (genvar group = 0; group < GROUPS; group = group + 1) begin : gen_adders
                logic        issue;
                logic [31:0] add_a, add_b;
                /* verilator lint_off UNUSEDSIGNAL */
                logic        add_out_valid_unused;  // each PE tracks its issues
                /* verilator lint_on UNUSEDSIGNAL */

                // At most one PE of the group requests per cycle (see header).
                always_comb begin
                    issue = 1'b0;
                    add_a = '0;
                    add_b = '0;
                    for (int i = 0; i < FP_ADDER_SHARE; i = i + 1) begin
                        if (group * FP_ADDER_SHARE + i < COLUMNS) begin
                            if (fp_req[row][group * FP_ADDER_SHARE + i]) begin
                                issue = 1'b1;
                                add_a = add_a | fp_a[row][group * FP_ADDER_SHARE + i];
                                add_b = add_b | fp_b[row][group * FP_ADDER_SHARE + i];
                            end
                        end
                    end
                end

                npu_fp32_add #(
                    .SUBNORMALS(0)
                ) fp32_accumulate (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(enable && !clear && issue),
                    .a(add_a),
                    .b(add_b),
                    .sub(1'b0),
                    .out_valid(add_out_valid_unused),
                    .y(fp_y[row][group])
                );
            end
        end
    endgenerate
endmodule
