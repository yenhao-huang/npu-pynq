`timescale 1ns/1ps

// Generic IEEE 754 adder core shared by npu_fp16_add (EW=5, MW=10) and
// npu_fp32_add (EW=8, MW=23).
//
// Contract
// - y = a + b (a - b when sub=1), round-to-nearest-even.
// - Subnormal inputs and outputs are supported (no flush to zero).
// - Overflow rounds to +-Inf. Every NaN result (NaN operand, Inf - Inf) is the
//   canonical quiet NaN {0, all-ones exponent, 1, 0...}.
// - Exact-zero results are +0, except (-0) + (-0) = -0 (IEEE RNE rule).
// - Fixed latency 3: out_valid is in_valid delayed by three clocks. Fully
//   pipelined, one operation per clock, no stall input.
//
// Pipeline
//   S1: unpack, special cases, magnitude compare and swap so |x| >= |y|,
//       align y by the exponent difference into MW+4 bits (guard, round,
//       sticky). The clamped shift amount and the sticky bit are formed for
//       both operand orders in parallel with the compare, which only
//       selects them. With SUBNORMALS = 0 only the coarse part of the shift
//       (multiples of 4) is applied here.
//   S2: (SUBNORMALS = 0: the remaining 0..3 alignment places, sticky
//       preserving) significand add/subtract, leading-zero count,
//       normalisation shift amount limited so the exponent never drops
//       below the subnormal one.
//   S3: normalise, round (RNE), pack, select special results.
// The split is for timing only: results are bit-identical to a single
// alignment shift in S1 (checked formally against the previous pipeline).
//
// SUBNORMALS = 0 drops the gradual-underflow logic (subnormal exponent
// adjustment, normalisation limit, subnormal packing) for callers that can
// prove no operand is subnormal and no result underflows: npu_pe_mp's FP32
// accumulator, whose operands are all multiples of 2^-48 (see npu_pe_mp).
// Subnormal operands are then outside the contract (simulation stops on one).
//
// Only the valid bits are reset; datapath registers are not, which keeps the
// reset fan-out small. y is undefined until the first out_valid.
module npu_fp_add_core #(
    parameter integer EW = 8,
    parameter integer MW = 23,
    parameter integer SUBNORMALS = 1
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             in_valid,
    input  logic [EW+MW:0]   a,
    input  logic [EW+MW:0]   b,
    input  logic             sub,
    output logic             out_valid,
    output logic [EW+MW:0]   y
);
    localparam integer W = 1 + EW + MW;
    localparam integer SW = MW + 1;   // significand incl. hidden bit
    localparam integer AW = SW + 3;   // aligned significand with G, R, S
    localparam integer NW = AW + 1;   // sum incl. carry out
    localparam integer SHW = $clog2(AW + 1);
    localparam [EW-1:0] EXP_ONES = {EW{1'b1}};
    localparam [EW-1:0] EXP_ONE = {{(EW-1){1'b0}}, 1'b1};
    localparam [EW-1:0] EXP_MAX_FINITE = {{(EW-1){1'b1}}, 1'b0};
    localparam [W-1:0] CANONICAL_NAN = {1'b0, EXP_ONES, 1'b1, {(MW-1){1'b0}}};
    localparam [SHW-1:0] SHIFT_ALL = SHW'(AW);
    localparam integer CW = (EW > SHW) ? EW : SHW;  // compare width
    // The low FINE bits of the alignment shift are applied at the start of
    // S2, moving one mux level out of S1, whose operands are muxed in front
    // of the core in npu_systolic_array_mp's shared adder. Only without
    // subnormals, whose S2 has no normalisation limit and so has the slack;
    // with FINE = 0 (full IEEE, npu_fp16_add) S2 is unchanged.
    localparam integer FINE = (SUBNORMALS == 0) ? 2 : 0;
    localparam integer FW = (FINE > 0) ? FINE : 1;

    // S1 sticky bit for exponent difference d: OR of the significand bits
    // that the coarse alignment shift moves below the sticky position. Bit j
    // sits at aligned position j+3 and is lost when the coarse shift
    // ((min(d, AW) >> FINE) << FINE) exceeds j+3, i.e. when d >= TH with
    // TH = (((j+3) >> FINE) + 1) << FINE (never when TH > AW). Testing d
    // directly keeps the clamp off this path.
    function automatic logic sticky_of(input logic [SW-1:0] sig, input logic [EW-1:0] d);
        begin
            sticky_of = 1'b0;
            for (int j = 0; j < SW; j = j + 1) begin
                if (((((j + 3) >> FINE) + 1) << FINE) <= AW) begin
                    if (32'(d) >= ((((j + 3) >> FINE) + 1) << FINE)) begin
                        sticky_of = sticky_of | sig[j];
                    end
                end
            end
        end
    endfunction

    // ---------------------------------------------------------------- S1
    logic          sa, sb;
    logic [EW-1:0] ea, eb, ea_eff, eb_eff;
    logic [MW-1:0] fa, fb;
    logic          a_nan, b_nan, a_inf, b_inf;
    logic          a_ge;
    logic          s1_nan, s1_inf, s1_inf_sign;
    logic          sx, sy;
    logic [EW-1:0] ex_eff;
    logic [SW-1:0] sigx, sigy;
    logic [SW-1:0] sig_a, sig_b;
    logic [EW-1:0] d_ab, d_ba;
    logic [SHW-1:0] shift_ab, shift_ba, shift, coarse_shift;
    logic          sticky_a, sticky_b;
    logic [AW-1:0] y_shifted;
    logic          y_sticky;

    always_comb begin
        sa = a[W-1];
        sb = b[W-1] ^ sub;
        ea = a[W-2:MW];
        eb = b[W-2:MW];
        fa = a[MW-1:0];
        fb = b[MW-1:0];
        a_nan = (ea == EXP_ONES) && (fa != '0);
        b_nan = (eb == EXP_ONES) && (fb != '0);
        a_inf = (ea == EXP_ONES) && (fa == '0);
        b_inf = (eb == EXP_ONES) && (fb == '0);
        s1_nan = a_nan || b_nan || (a_inf && b_inf && (sa != sb));
        s1_inf = a_inf || b_inf;
        s1_inf_sign = a_inf ? sa : sb;

        // Subnormals use exponent 1 with a zero hidden bit.
        if (SUBNORMALS != 0) begin
            ea_eff = (ea == '0) ? EXP_ONE : ea;
            eb_eff = (eb == '0) ? EXP_ONE : eb;
        end else begin
            ea_eff = ea;
            eb_eff = eb;
        end
        d_ab = ea_eff - eb_eff;
        d_ba = eb_eff - ea_eff;
        sig_a = {(ea != '0), fa};
        sig_b = {(eb != '0), fb};
        // Shift amount and sticky bit for both orders, formed in parallel
        // with the magnitude compare (a_ge only selects them). Shifts of AW
        // or more move every bit into the sticky position. The low FINE
        // shift bits are applied in S2 (see FINE), so the sticky bit here
        // covers the coarse shift only.
        shift_ab = (d_ab >= EW'(AW)) ? SHIFT_ALL : d_ab[SHW-1:0];
        shift_ba = (d_ba >= EW'(AW)) ? SHIFT_ALL : d_ba[SHW-1:0];
        sticky_b = sticky_of(sig_b, d_ab);
        sticky_a = sticky_of(sig_a, d_ba);

        // Order by magnitude so the significand difference is never negative.
        a_ge = (a[W-2:0] >= b[W-2:0]);
        sx = a_ge ? sa : sb;
        sy = a_ge ? sb : sa;
        ex_eff = a_ge ? ea_eff : eb_eff;
        sigx = a_ge ? sig_a : sig_b;
        sigy = a_ge ? sig_b : sig_a;
        shift = a_ge ? shift_ab : shift_ba;
        y_sticky = a_ge ? sticky_b : sticky_a;
        coarse_shift = (shift >> FINE) << FINE;
        y_shifted = {sigy, 3'b000} >> coarse_shift;
    end

    logic          s1_valid;
    logic          s1_sx, s1_sub, s1_nan_q, s1_inf_q, s1_inf_sign_q;
    logic [EW-1:0] s1_ex;
    logic [AW-1:0] s1_x, s1_y;
    logic [FW-1:0] s1_fine;

    always_ff @(posedge clk) begin
        s1_sx <= sx;
        s1_sub <= sx ^ sy;
        s1_nan_q <= s1_nan;
        s1_inf_q <= s1_inf;
        s1_inf_sign_q <= s1_inf_sign;
        s1_ex <= ex_eff;
        s1_x <= {sigx, 3'b000};
        s1_y <= {y_shifted[AW-1:1], y_shifted[0] | y_sticky};
        s1_fine <= shift[FW-1:0];
    end

    // ---------------------------------------------------------------- S2
    logic [AW-1:0]  y_aligned;
    logic [NW-1:0]  sum;
    logic [SHW-1:0] lz;
    logic [SHW-1:0] norm_shift;
    logic           lz_found;

    always_comb begin
        // Remaining FINE shift bits; the bits shifted out are ORed into the
        // sticky position (bit 0, which already holds the S1 sticky).
        y_aligned = s1_y;
        if (FINE != 0) begin
            y_aligned = s1_y >> s1_fine;
            for (int k = 0; k < (1 << FINE) - 1; k = k + 1) begin
                if (32'(s1_fine) > k) begin
                    y_aligned[0] = y_aligned[0] | s1_y[k];
                end
            end
        end
        if (s1_sub) begin
            sum = {1'b0, s1_x} - {1'b0, y_aligned};
        end else begin
            sum = {1'b0, s1_x} + {1'b0, y_aligned};
        end

        lz = SHIFT_ALL;
        lz_found = 1'b0;
        for (int i = AW - 1; i >= 0; i = i - 1) begin
            if (!lz_found && sum[i]) begin
                lz = SHW'(AW - 1 - i);
                lz_found = 1'b1;
            end
        end

        // Never normalise below the minimum exponent (gradual underflow).
        if (SUBNORMALS == 0) begin
            norm_shift = lz;
        end else if (CW'(lz) > CW'(s1_ex - EXP_ONE)) begin
            norm_shift = SHW'(s1_ex - EXP_ONE);
        end else begin
            norm_shift = lz;
        end
    end

    logic           s2_valid;
    logic           s2_sx, s2_sub, s2_nan, s2_inf, s2_inf_sign;
    logic [EW-1:0]  s2_ex;
    logic [NW-1:0]  s2_sum;
    logic [SHW-1:0] s2_shift;

    always_ff @(posedge clk) begin
        s2_sx <= s1_sx;
        s2_sub <= s1_sub;
        s2_nan <= s1_nan_q;
        s2_inf <= s1_inf_q;
        s2_inf_sign <= s1_inf_sign_q;
        s2_ex <= s1_ex;
        s2_sum <= sum;
        s2_shift <= norm_shift;
    end

    // ---------------------------------------------------------------- S3
    logic          carry;
    logic [AW-1:0] norm;
    logic          extra_sticky;
    logic [EW-1:0] e_after;
    logic          overflow;
    logic [W-2:0]  packed_mag;
    logic          guard, sticky, round_up;
    logic [W-2:0]  rounded;
    logic          sum_zero;
    logic [W-1:0]  result;

    always_comb begin
        carry = s2_sum[NW-1];
        if (carry) begin
            norm = s2_sum[NW-1:1];
            extra_sticky = s2_sum[0];
            e_after = s2_ex + EXP_ONE;
        end else begin
            norm = s2_sum[AW-1:0] << s2_shift;
            extra_sticky = 1'b0;
            e_after = s2_ex - EW'(s2_shift);
        end
        // A carry out of the largest finite exponent overflows.
        overflow = carry && (s2_ex == EXP_MAX_FINITE);

        // Exponent field and fraction. A zero hidden bit only occurs with
        // e_after = 1 (subnormal), whose field is 0. Rounding is one increment
        // of {exponent, fraction}: a carry ripples into the exponent
        // (subnormal -> normal, largest finite -> Inf) for free.
        packed_mag = {(norm[AW-1] || (SUBNORMALS == 0)) ? e_after : {EW{1'b0}},
                      norm[AW-2:3]};
        guard = norm[2];
        sticky = norm[1] | norm[0] | extra_sticky;
        round_up = guard && (sticky || norm[3]);
        rounded = packed_mag + {{(W-2){1'b0}}, round_up};

        // Exact zero: +0 unless both operands were zeros of the same sign.
        sum_zero = (s2_sum == '0);

        if (s2_nan) begin
            result = CANONICAL_NAN;
        end else if (s2_inf) begin
            result = {s2_inf_sign, EXP_ONES, {MW{1'b0}}};
        end else if (overflow) begin
            result = {s2_sx, EXP_ONES, {MW{1'b0}}};
        end else if (sum_zero) begin
            result = {s2_sx && !s2_sub, {(W-1){1'b0}}};
        end else begin
            result = {s2_sx, rounded};
        end
    end

    always_ff @(posedge clk) begin
        y <= result;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            s1_valid <= in_valid;
            s2_valid <= s1_valid;
            out_valid <= s2_valid;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (SUBNORMALS == 0 && in_valid &&
            (((ea == '0) && (fa != '0)) || ((eb == '0) && (fb != '0)))) begin
            $display("MISMATCH npu_fp_add_core: subnormal operand with SUBNORMALS=0 (a=%h b=%h)",
                     a, b);
            $fatal(1);
        end
        if (SUBNORMALS == 0 && s2_valid && !s2_nan && !s2_inf && !sum_zero &&
            !carry && (EW'(s2_shift) >= s2_ex)) begin
            $display("MISMATCH npu_fp_add_core: result underflow with SUBNORMALS=0");
            $fatal(1);
        end
    end
`endif
endmodule
