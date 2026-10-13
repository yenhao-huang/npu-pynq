`timescale 1ns/1ps

// Multi-precision systolic processing element.
//
// Same systolic dataflow and handshake as npu_pe: on every enabled clock the
// operands and their valids are forwarded (a_out/b_out, one register stage),
// and a valid pair (a_valid_in && b_valid_in) contributes one MAC. clear
// zeroes the forwarded operands, the valids, the accumulator and every MAC in
// flight; it wins over enable. Operand lanes are 16 bits wide and the runtime
// precision prec selects the arithmetic:
//
//   prec = 0 INT8   operands are the low bytes (sign-extended, upper bytes
//                   ignored); 32-bit saturating accumulator with exactly
//                   npu_pe's semantics, sign-extended onto accumulator[47:0].
//   prec = 1 INT16  16 x 16 signed product, 48-bit saturating accumulator.
//   prec = 2 FP16   fp16 x fp16 product formed exactly as fp32 (the 22-bit
//                   significand product fits, no rounding), accumulated with
//                   npu_fp32_add (RNE, canonical NaN 0x7FC00000) starting from
//                   +0 in arrival order; fp32 bits in accumulator[31:0],
//                   accumulator[47:32] = 0.
//   prec = 3        reserved: no MAC is performed (operands still forward).
//
// prec must stay constant while a job is in flight; change it together with
// clear.
//
// Timing and throughput
// - INT8/INT16: one MAC per enabled clock, as npu_pe. A pair presented in
//   cycle t (enable high) is multiplied and added in the DSP48 in cycle t
//   (P = C + A*B into PREG; C is the saturated running sum) and the clamped
//   sum is registered into accumulator in the next enabled cycle, so the
//   accumulator includes it from cycle t+2, exactly as in npu_pe (which
//   registers the product first and adds in the fabric). See g_dsp_mac.
// - FP16: the FP32 adder has latency 3 and the accumulator is a loop through
//   it, so valid pairs at a PE must be at least 3 enabled cycles apart (the
//   FP16 MAC interval: two enabled non-valid cycles between valid pairs). In a
//   systolic array the controller meets this by inserting two bubble beats
//   (valid = 0) after every valid beat on every edge stream; the bubbles flow
//   through the array with the wavefront, so every PE sees the same spacing.
//   Throughput is 1/3 MAC per cycle per PE. A pair presented in cycle t is
//   turned into an fp32 product in the next enabled cycle, issued to the adder
//   in the enabled cycle after that, and is in the accumulator 4 cycles after
//   the issue (the adder result is bypassed to the adder input so the next
//   pair can issue 3 cycles later). With enable held high the accumulator
//   includes the pair from cycle t+6. An FP32 addition already issued
//   completes even if enable drops (enable stalls the wavefront, not the
//   adder); clear discards it.
// - Hazards are not checked in hardware: closer FP16 spacing gives wrong sums.
//
// FP32 adder: local or shared
// - The accumulator only ever holds +0 (after clear) or sums of exact
//   fp16 x fp16 products, all multiples of 2^-48, so no adder operand is
//   subnormal and no sum underflows (fp32 normals start at 2^-126). The adder
//   is therefore npu_fp32_add with SUBNORMALS = 0, bit-exact with the
//   full IEEE adder for every reachable input. This relies on clear (or reset)
//   before an FP16 job, as required above.
// - SHARED_FP_ADDER = 0: the PE has its own adder; fp_add_y is ignored.
// - SHARED_FP_ADDER = 1: the PE has no adder. It presents fp_add_req (a
//   product is waiting; it issues in the first cycle with fp_add_req, enable
//   and !clear), the operands fp_add_a (accumulator, or the result being
//   written back) and fp_add_b (product), and takes the sum on fp_add_y three
//   clocks after the issue. npu_systolic_array_mp shares one adder between up
//   to three neighbouring PEs of a row: with the FP16 spacing above, their
//   issues always fall in different cycles (see npu_systolic_array_mp).
//   Deviation from architecture.md (npu_pe ports only): these four ports.
//
// Resources: one DSP48E1 (shared 16 x 16 signed multiplier: sign-extended
// INT8, INT16, or zero-extended 11-bit FP16 significands, plus the 48-bit INT
// accumulation adder in its post-adder), a 48-bit clamp mux, the fp32 product
// normaliser and (local mode) one FP32 adder. DSP_STYLE = "no" (PEs beyond
// the array's DSP budget) uses a LUT multiplier and a fabric 48-bit
// saturating add instead (g_lut_mac), with identical results and cycle
// timing.
//
// Vivado mapping of g_dsp_mac (one DSP48E1 per PE): the expression
// dsp_p <= C + A*B with a combinational C is the UG901 multiply-add form;
// Vivado packs multiplier, post-adder and the dsp_p register into one DSP48E1
// with AREG = BREG = 0 (the operands come from the neighbour's asynchronously
// reset a_out/b_out through the precision mux, so they cannot be absorbed),
// MREG = 0, CREG = 0, PREG = 1 with CEP = pair_ce and RSTP unused, and a
// static OPMODE (Z = C, X/Y = M). The clamp is ordinary fabric logic from the
// P output back to the C input; it is not an accumulator feedback (Z = P)
// pattern, so it does not affect inference. dsp_p must stay without reset:
// DSP48E1 registers only have a synchronous reset, and an asynchronous one
// would move the register into the fabric. clear acts through dsp_live
// (C = 0 for the first MAC after clear), not through RSTP. Expected DRC
// advisories: DPIP-1 (no input register) and DPOP-2 (no MREG).
//
// Timing (Yosys netlist, rough 7-series -1 delay model): the DSP A/B input
// path (operand select, 3 LUT levels, into A*B + C -> PREG without M
// register) is ~6.8 ns from the neighbour's a_out register, the P -> clamp ->
// C port loop ~5.4 ns. With the local adder the worst PE path is the FP32
// adder's stage 2 (~7.1 ns).
module npu_pe_mp #(
    parameter DSP_STYLE = "yes",
    parameter integer SHARED_FP_ADDER = 0
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        clear,
    input  logic        enable,
    input  logic [1:0]  prec,
    input  logic [15:0] a_in,
    input  logic [15:0] b_in,
    input  logic        a_valid_in,
    input  logic        b_valid_in,
    output logic [15:0] a_out,
    output logic [15:0] b_out,
    output logic        a_valid_out,
    output logic        b_valid_out,
    output logic [47:0] accumulator,
    // FP32 adder port, used when SHARED_FP_ADDER = 1
    output logic        fp_add_req,
    output logic [31:0] fp_add_a,
    output logic [31:0] fp_add_b,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0] fp_add_y        // ignored when SHARED_FP_ADDER = 0
    /* verilator lint_on UNUSEDSIGNAL */
);
    localparam [1:0] PREC_INT8 = 2'd0;
    localparam [1:0] PREC_INT16 = 2'd1;
    localparam [1:0] PREC_FP16 = 2'd2;
    localparam [31:0] FP32_CANONICAL_NAN = 32'h7fc00000;
    localparam [47:0] INT32_MAX_48 = 48'h0000_7fff_ffff;
    localparam [47:0] INT32_MIN_48 = 48'hffff_8000_0000;
    localparam [47:0] INT48_MAX = 48'h7fff_ffff_ffff;
    localparam [47:0] INT48_MIN = 48'h8000_0000_0000;

    initial begin
        if (DSP_STYLE != "yes" && DSP_STYLE != "no") begin
            $fatal(1, "npu_pe_mp DSP_STYLE must be yes or no");
        end
    end

    function automatic [4:0] lzc22(input logic [21:0] value);
        logic found;
        begin
            lzc22 = 5'd22;
            found = 1'b0;
            for (int i = 21; i >= 0; i = i - 1) begin
                if (!found && value[i]) begin
                    lzc22 = 5'(21 - i);
                    found = 1'b1;
                end
            end
        end
    endfunction

    // ------------------------------------------------ operand decode (cycle t)
    logic        is_int, is_fp;
    logic [4:0]  ea, eb;
    logic [9:0]  fa, fb;
    logic        a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    logic signed [15:0] mul_a, mul_b;
    logic [5:0]  exp_sum;

    always_comb begin
        is_int = (prec == PREC_INT8) || (prec == PREC_INT16);
        is_fp = (prec == PREC_FP16);
        ea = a_in[14:10];
        eb = b_in[14:10];
        fa = a_in[9:0];
        fb = b_in[9:0];
        a_zero = (ea == 5'd0) && (fa == 10'd0);
        b_zero = (eb == 5'd0) && (fb == 10'd0);
        a_inf = (ea == 5'h1f) && (fa == 10'd0);
        b_inf = (eb == 5'h1f) && (fb == 10'd0);
        a_nan = (ea == 5'h1f) && (fa != 10'd0);
        b_nan = (eb == 5'h1f) && (fb != 10'd0);
        // Subnormals use exponent 1 with a zero hidden bit.
        exp_sum = {1'b0, (ea == 5'd0) ? 5'd1 : ea} + {1'b0, (eb == 5'd0) ? 5'd1 : eb};

        case (prec)
            PREC_INT8: begin
                mul_a = {{8{a_in[7]}}, a_in[7:0]};
                mul_b = {{8{b_in[7]}}, b_in[7:0]};
            end
            PREC_INT16: begin
                mul_a = a_in;
                mul_b = b_in;
            end
            default: begin
                mul_a = {5'd0, (ea != 5'd0), fa};
                mul_b = {5'd0, (eb != 5'd0), fb};
            end
        endcase
    end

    // ------------------------------ stage 1: shared multiplier, flags, INT
    // The multiplier registers (product, FP flags) are not reset (the valid
    // bits are) and load only on an enabled valid pair (no toggling on
    // bubbles).
    logic               pair_ce;
    logic [21:0]        fp_sig_product;  // registered significand product
    logic [47:0]        int_result;      // saturated sum for int_valid
    logic       fp_sign, fp_nan, fp_inf, fp_zero;
    logic [5:0] fp_exp_sum;
    logic       int_valid, fp_valid;

    assign pair_ce = enable && a_valid_in && b_valid_in;

    always_ff @(posedge clk) begin
        if (pair_ce) begin
            fp_sign <= a_in[15] ^ b_in[15];
            fp_nan <= a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero);
            fp_inf <= a_inf || b_inf;
            fp_zero <= a_zero || b_zero;
            fp_exp_sum <= exp_sum;
        end
    end

    generate
        if (DSP_STYLE == "yes") begin : g_dsp_mac
            // INT accumulation inside the DSP48E1 (A*B + C -> PREG, no
            // M register): dsp_p <= int_base + mul_a * mul_b for a valid
            // pair in cycle t. dsp_p is the not yet saturated sum: exact in
            // INT8 mode (INT32 base plus one INT8 product), 48-bit wrapped in
            // INT16 mode. Saturation is applied one cycle late, outside
            // the DSP: int_base clamps dsp_p; it is the visible accumulator
            // (registered at the next enabled cycle, i.e. t+2 as in npu_pe)
            // and, through the C port, the base of the next MAC. Overflow
            // detection without a carry chain:
            //   INT8 : dsp_p[47:31] not all equal (the sum of an INT32 base
            //          and an INT8 product always fits 48 bits);
            //   INT16: base and product signs equal and dsp_p[47] different
            //          (operand signs kept in base_sign/product_sign; a zero
            //          product leaves dsp_p = base and cannot trigger it).
            // In FP16 mode C = 0 and dsp_p[21:0] is the significand product.
            (* use_dsp = "yes" *) logic signed [47:0] dsp_p;
            logic        dsp_live;   // dsp_p written since clear/reset
            logic        base_sign, product_sign;
            logic        ovf_pos, ovf_neg;
            logic [47:0] int_base;   // saturated dsp_p, or 0 (see below)

            always_comb begin
                if (prec == PREC_INT8) begin
                    ovf_pos = !dsp_p[47] && (dsp_p[46:31] != 16'h0000);
                    ovf_neg = dsp_p[47] && (dsp_p[46:31] != 16'hffff);
                end else begin
                    ovf_pos = !base_sign && !product_sign && dsp_p[47];
                    ovf_neg = base_sign && product_sign && !dsp_p[47];
                end
                // clear/reset (dsp_live = 0) restarts from 0; FP16 adds 0.
                if (!dsp_live || !is_int) begin
                    int_base = 48'd0;
                end else if (ovf_pos) begin
                    int_base = (prec == PREC_INT8) ? INT32_MAX_48 : INT48_MAX;
                end else if (ovf_neg) begin
                    int_base = (prec == PREC_INT8) ? INT32_MIN_48 : INT48_MIN;
                end else begin
                    int_base = dsp_p;
                end
            end

            always_ff @(posedge clk) begin
                if (pair_ce) begin
                    dsp_p <= $signed(int_base) + mul_a * mul_b;
                    base_sign <= int_base[47];
                    product_sign <= mul_a[15] ^ mul_b[15];
                end
            end

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    dsp_live <= 1'b0;
                end else if (clear) begin
                    dsp_live <= 1'b0;
                end else if (pair_ce) begin
                    dsp_live <= 1'b1;
                end
            end

            assign fp_sig_product = dsp_p[21:0];
            // int_valid implies dsp_live and an INT mode, so int_base is the
            // saturated sum there.
            assign int_result = int_base;
        end else begin : g_lut_mac
            // LUT multiplier (DSP budget exhausted): registered product and
            // a fabric 48-bit saturating add, same results and cycle timing.
            // A fused multiply-add would not fit one 100 MHz cycle in LUTs.
            (* use_dsp = "no" *) logic signed [31:0] lut_product;
            logic [47:0] product_extended;
            logic [47:0] wrapped_sum;
            logic [47:0] saturated_sum;

            always_ff @(posedge clk) begin
                if (pair_ce) begin
                    lut_product <= mul_a * mul_b;
                end
            end

            always_comb begin
                product_extended = {{16{lut_product[31]}}, lut_product};
                wrapped_sum = accumulator + product_extended;
                if (prec == PREC_INT8) begin
                    // The 32-bit value cannot overflow 48 bits; clamp to INT32.
                    if (!wrapped_sum[47] && (wrapped_sum[46:31] != 16'h0000)) begin
                        saturated_sum = INT32_MAX_48;
                    end else if (wrapped_sum[47] && (wrapped_sum[46:31] != 16'hffff)) begin
                        saturated_sum = INT32_MIN_48;
                    end else begin
                        saturated_sum = wrapped_sum;
                    end
                end else begin
                    if (!accumulator[47] && !product_extended[47] && wrapped_sum[47]) begin
                        saturated_sum = INT48_MAX;
                    end else if (accumulator[47] && product_extended[47] && !wrapped_sum[47]) begin
                        saturated_sum = INT48_MIN;
                    end else begin
                        saturated_sum = wrapped_sum;
                    end
                end
            end

            assign fp_sig_product = lut_product[21:0];
            assign int_result = saturated_sum;
        end
    endgenerate

    // --------------------------- stage 2 FP: exact fp32 product normaliser
    // Significand product P = sig_a * sig_b (22 bits, may have leading zeros
    // for subnormal inputs) has value P * 2^(ea + eb - 50); with L leading
    // zeros the biased fp32 exponent is ea + eb + 98 - L (always normal).
    logic [21:0] fp_mant_raw;
    logic [4:0]  fp_lz;
    logic [20:0] fp_mant_norm;   // fraction below the leading one
    logic [7:0]  fp_exp32;
    logic [31:0] fp_product32;

    always_comb begin
        fp_mant_raw = fp_sig_product;
        fp_lz = lzc22(fp_mant_raw);
        fp_mant_norm = 21'(fp_mant_raw << fp_lz);
        fp_exp32 = {2'b00, fp_exp_sum} + 8'd98 - {3'b000, fp_lz};
        if (fp_nan) begin
            fp_product32 = FP32_CANONICAL_NAN;
        end else if (fp_inf) begin
            fp_product32 = {fp_sign, 8'hff, 23'd0};
        end else if (fp_zero) begin
            fp_product32 = {fp_sign, 31'd0};
        end else begin
            fp_product32 = {fp_sign, fp_exp32, fp_mant_norm, 2'b00};
        end
    end

    logic [31:0] fp_product_q;
    logic        fp_product_valid;   // product waiting to issue to the adder

    always_ff @(posedge clk) begin
        if (fp_valid) begin   // product of the pair waiting in stage 1
            fp_product_q <= fp_product32;
        end
    end

    // -------------------------------------- stage 3+ FP: FP32 accumulation
    logic        add_in_valid;
    logic [31:0] add_y;
    logic [2:0]  fp_pending;
    logic        fp_done;

    // fp_pending tracks this PE's issued additions: the result of an issue
    // in cycle t is on add_y in cycle t+3. clear (or reset) empties it, which
    // discards results still inside the adder.
    assign fp_done = fp_pending[2];
    assign add_in_valid = enable && !clear && fp_product_valid;
    assign fp_add_req = fp_product_valid;
    // Bypass the result that is being written back this cycle.
    assign fp_add_a = fp_done ? add_y : accumulator[31:0];
    assign fp_add_b = fp_product_q;

    generate
        if (SHARED_FP_ADDER != 0) begin : g_shared_adder
            assign add_y = fp_add_y;
        end else begin : g_local_adder
            /* verilator lint_off UNUSEDSIGNAL */
            logic add_out_valid_unused;  // fp_pending tracks the results
            /* verilator lint_on UNUSEDSIGNAL */

            npu_fp32_add #(
                .SUBNORMALS(0)
            ) fp32_accumulate (
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(add_in_valid),
                .a(fp_add_a),
                .b(fp_add_b),
                .sub(1'b0),
                .out_valid(add_out_valid_unused),
                .y(add_y)
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
            int_valid <= 1'b0;
            fp_valid <= 1'b0;
            fp_product_valid <= 1'b0;
            fp_pending <= '0;
        end else if (clear) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
            int_valid <= 1'b0;
            fp_valid <= 1'b0;
            fp_product_valid <= 1'b0;
            fp_pending <= '0;
        end else begin
            fp_pending <= {fp_pending[1:0], add_in_valid};
            if (fp_done) begin
                accumulator <= {16'd0, add_y};
            end
            if (enable) begin
                a_out <= a_in;
                b_out <= b_in;
                a_valid_out <= a_valid_in;
                b_valid_out <= b_valid_in;
                int_valid <= a_valid_in && b_valid_in && is_int;
                fp_valid <= a_valid_in && b_valid_in && is_fp;
                fp_product_valid <= fp_valid;
                if (int_valid) begin
                    accumulator <= int_result;
                end
            end
        end
    end
endmodule
