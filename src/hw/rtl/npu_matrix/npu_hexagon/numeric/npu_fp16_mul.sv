`timescale 1ns/1ps

// IEEE 754 binary16 multiplier for the vector unit.
//
// Contract
// - y = a * b, round-to-nearest-even, subnormal inputs and outputs (no flush
//   to zero), overflow to +-Inf, canonical NaN 0x7E00 for every NaN result
//   (NaN operand or 0 * Inf), zero results carry sign(a) ^ sign(b).
// - Fixed latency LAT_MUL = 2: out_valid is in_valid delayed by two clocks.
//   Fully pipelined, one operation per clock, no stall input.
//
// Pipeline
//   S1: unpack, classify, normalise subnormal significands (leading-zero
//       count) so both significands are in [1, 2), 11 x 11 product in one
//       DSP48 (registered product), exponent sum.
//       The exponent-dependent S2 controls (shift amount, exponent field,
//       overflow) are precomputed for both product ranges.
//   S2: product in [1, 4): one right shift (one-bit normalise or subnormal
//       denormalise) with sticky, RNE rounding, pack, special results.
//
// Only the valid bits are reset; the datapath registers are not, so the
// product register can be packed into the DSP48. y is undefined until the
// first out_valid.
module npu_fp16_mul (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic        out_valid,
    output logic [15:0] y
);
    localparam [15:0] CANONICAL_NAN = 16'h7E00;

    function automatic [3:0] lzc11(input logic [10:0] value);
        logic found;
        begin
            lzc11 = 4'd11;
            found = 1'b0;
            for (int i = 10; i >= 0; i = i - 1) begin
                if (!found && value[i]) begin
                    lzc11 = 4'(10 - i);
                    found = 1'b1;
                end
            end
        end
    endfunction

    // ---------------------------------------------------------------- S1
    logic [4:0]  ea, eb;
    logic [9:0]  fa, fb;
    logic        a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    logic [10:0] sig_a, sig_b, norm_a, norm_b;
    logic [3:0]  lz_a, lz_b;
    logic signed [7:0] exp_a, exp_b, exp_sum;

    always_comb begin
        ea = a[14:10];
        eb = b[14:10];
        fa = a[9:0];
        fb = b[9:0];
        a_zero = (ea == 5'd0) && (fa == 10'd0);
        b_zero = (eb == 5'd0) && (fb == 10'd0);
        a_inf = (ea == 5'h1f) && (fa == 10'd0);
        b_inf = (eb == 5'h1f) && (fb == 10'd0);
        a_nan = (ea == 5'h1f) && (fa != 10'd0);
        b_nan = (eb == 5'h1f) && (fb != 10'd0);

        sig_a = {(ea != 5'd0), fa};
        sig_b = {(eb != 5'd0), fb};
        lz_a = lzc11(sig_a);
        lz_b = lzc11(sig_b);
        norm_a = sig_a << lz_a;
        norm_b = sig_b << lz_b;
        // Unbiased-plus-15 exponent of each normalised operand (subnormals
        // use exponent 1 before normalisation).
        exp_a = $signed({3'b000, (ea == 5'd0) ? 5'd1 : ea}) - $signed({4'b0000, lz_a});
        exp_b = $signed({3'b000, (eb == 5'd0) ? 5'd1 : eb}) - $signed({4'b0000, lz_b});
        exp_sum = exp_a + exp_b - 8'sd15;
    end

    // Exponent-dependent S2 controls are precomputed here, off the DSP path,
    // for both values of the product's top bit (high = product >= 2):
    //   exp_sum >= 1: normal result, the product is shifted right by high;
    //   exp_sum <= 0: subnormal (or, for exp_sum = 0 and high, the minimum
    //                 normal) result, shifted right by 1 - exp_sum whatever
    //                 high is; 23 or more places leave only sticky bits.
    // The shift applies to {product, 0}, whose bit 22 or 21 is the leading
    // one, so the fraction always sits at bits [20:11] after the shift.
    logic              lo_range;
    logic [4:0]        lo_shift;
    logic [4:0]        field0, field1;
    logic              ovf0, ovf1;

    always_comb begin
        lo_range = (exp_sum < 8'sd1);
        lo_shift = (exp_sum < -8'sd21) ? 5'd23 : 5'(8'sd1 - exp_sum);
        field0 = (exp_sum < 8'sd0) ? 5'd0 : exp_sum[4:0];
        field1 = (exp_sum < 8'sd0) ? 5'd0 : 5'(exp_sum + 8'sd1);
        ovf0 = (exp_sum > 8'sd30);
        ovf1 = (exp_sum > 8'sd29);
    end

    (* use_dsp = "yes" *) logic [21:0] s1_product;
    logic              s1_lo_range;
    logic [4:0]        s1_lo_shift, s1_field0, s1_field1;
    logic              s1_ovf0, s1_ovf1;
    logic              s1_sign, s1_nan, s1_inf, s1_zero;
    logic              s1_valid;

    always_ff @(posedge clk) begin
        s1_product <= norm_a * norm_b;
        s1_lo_range <= lo_range;
        s1_lo_shift <= lo_shift;
        s1_field0 <= field0;
        s1_field1 <= field1;
        s1_ovf0 <= ovf0;
        s1_ovf1 <= ovf1;
        s1_sign <= a[15] ^ b[15];
        s1_nan <= a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero);
        s1_inf <= a_inf || b_inf;
        s1_zero <= a_zero || b_zero;
    end

    // ---------------------------------------------------------------- S2
    logic              high;
    logic [22:0]       mant;           // {product, 0}
    logic [4:0]        rshift;
    logic              overflow;
    logic [4:0]        exp_field;
    logic [20:0]       mant_shifted;   // below the leading-one position
    logic              shift_sticky;
    logic [14:0]       packed_mag;
    logic              guard, sticky, round_up;
    logic [14:0]       rounded;
    logic [15:0]       result;

    always_comb begin
        // Both significands are in [1, 2): the product is in [1, 4).
        high = s1_product[21];
        mant = {s1_product, 1'b0};
        rshift = s1_lo_range ? s1_lo_shift : {4'd0, high};
        exp_field = high ? s1_field1 : s1_field0;
        overflow = high ? s1_ovf1 : s1_ovf0;

        mant_shifted = 21'(mant >> rshift);
        shift_sticky = 1'b0;
        for (int j = 0; j < 23; j = j + 1) begin
            if (rshift > 5'(j)) begin
                shift_sticky = shift_sticky | mant[j];
            end
        end

        // Rounding is one increment of {exponent, fraction}: a carry ripples
        // into the exponent (subnormal -> normal, largest finite -> Inf).
        packed_mag = {exp_field, mant_shifted[20:11]};
        guard = mant_shifted[10];
        sticky = (mant_shifted[9:0] != 10'd0) || shift_sticky;
        round_up = guard && (sticky || mant_shifted[11]);
        rounded = packed_mag + {14'd0, round_up};

        if (s1_nan) begin
            result = CANONICAL_NAN;
        end else if (s1_inf || (!s1_zero && overflow)) begin
            result = {s1_sign, 15'h7c00};
        end else if (s1_zero) begin
            result = {s1_sign, 15'd0};
        end else begin
            result = {s1_sign, rounded};
        end
    end

    always_ff @(posedge clk) begin
        y <= result;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            s1_valid <= in_valid;
            out_valid <= s1_valid;
        end
    end
endmodule
