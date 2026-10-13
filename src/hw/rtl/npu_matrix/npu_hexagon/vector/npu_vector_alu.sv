`timescale 1ns/1ps

// npu_vector_alu: integer / FP16-compare ALU for one 32-bit slice of a vector
// register (issue #136). The vector unit processes a VLEN-bit register as
// VLEN/32 slices, one slice per clock; a slice holds 4 INT8, 2 INT16, 1 INT32
// or 2 FP16 lanes, so no lane ever crosses a slice boundary.
//
// Functions (fn), lane type typ (0 INT8, 1 INT16, 2 INT32, 3 FP16):
//   ALU_PASS : y = p                       (VMOV, VSPLAT, VLD data)
//   ALU_ADD  : y = sat(a + b)              integer lanes, saturating
//   ALU_SUB  : y = sat(a - b)              integer lanes, saturating
//   ALU_MAX  : y = max(a, b)               signed integer, or FP16 (VRELU is
//   ALU_MIN  : y = min(a, b)               MAX with b = 0 from the caller)
//   ALU_SRA  : y = (a + (1 << (s-1))) >>> s, integer lanes; the caller
//              passes the clamped shift already decremented (see below)
// FP16 MAX/MIN follow the numeric contract: a NaN operand gives 0x7E00,
// max(-0, +0) = +0, min(-0, +0) = -0. The FP16 order shares the INT16 signed
// compare of the subtractor: for raw bit patterns, signed a < b already orders
// FP16 values when the signs differ (including -0 < +0) and when both are
// positive; when both are negative the magnitude order is reversed, so the
// FP16 less-than is the INT16 less-than XOR (a_sign & b_sign). Equal bit
// patterns may resolve either way, which selects the same value.
// SRA: s = 0 returns a unchanged (sh_zero); s >= lane width W is clamped to
// W, which gives 0 for every input, exactly what the golden sra_round()
// returns (the rounded quotient of a W-bit value by 2^s, s >= W, is 0). The
// rounded result always fits the lane, so SRA never saturates. The caller
// computes shm1 = min(s, W) - 1 (don't care when sh_zero) from registered
// command fields one stage earlier, so X1 only shifts.
//
// Pipeline (latency 2, fully pipelined, no valid/stall: the caller tracks
// which slice is where):
//   X1: segmented 32-bit adder/subtractor with per-lane overflow and signed
//       less-than, segmented arithmetic right shift by shm1 (sign fill at the
//       lane top); all registered.
//   X2: saturation / max-min select / SRA rounding (segmented increment, then
//       one more shift: (t + 1) >>> 1 with t = a >>> (s-1)); registered y.
// Both segmented adders are a single carry chain with a spacer bit between
// bytes (36 / 37 bits): the spacer pair (1, 0) propagates the byte carry
// inside a lane, the pair (c, c) starts a new lane with carry-in c. No LUT
// sits between bytes of the chain, which keeps X1 and X2 to one carry chain
// plus a few LUT levels (X1 worst path, Yosys synth_xilinx estimate: fn
// decode and operand LUTs, the 36-bit chain, three or four LUTs of
// overflow / FP16 order / select into selb2, about 6.5-7.4 ns).
// Pure datapath: no reset (y is undefined until two clocks after the first
// input).
module npu_vector_alu (
    input  logic        clk,
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [31:0] p,
    input  logic [1:0]  typ,
    input  logic [2:0]  fn,
    input  logic [4:0]  shm1,      // SRA: min(s, lane width) - 1
    input  logic        sh_zero,   // SRA: s = 0 (copy)
    output logic [31:0] y
);
    localparam [2:0] ALU_PASS = 3'd0;
    localparam [2:0] ALU_ADD  = 3'd1;
    localparam [2:0] ALU_SUB  = 3'd2;
    localparam [2:0] ALU_MAX  = 3'd3;
    localparam [2:0] ALU_MIN  = 3'd4;
    localparam [2:0] ALU_SRA  = 3'd5;

    localparam [1:0] T_INT8  = 2'd0;
    localparam [1:0] T_INT32 = 2'd2;
    localparam [1:0] T_FP16  = 2'd3;

    // Lane boundaries per byte: start = lowest byte of a lane, top = byte
    // holding the lane's sign bit. FP16 lanes use the INT16 boundaries.
    function automatic logic lane_start(input logic [1:0] t, input integer i);
        case (t)
            T_INT8:  lane_start = 1'b1;
            T_INT32: lane_start = (i == 0);
            default: lane_start = (i == 0) || (i == 2);
        endcase
    endfunction

    function automatic logic lane_top(input logic [1:0] t, input integer i);
        case (t)
            T_INT8:  lane_top = 1'b1;
            T_INT32: lane_top = (i == 3);
            default: lane_top = (i == 1) || (i == 3);
        endcase
    endfunction

    // Index of the top byte of the lane that contains byte i.
    function automatic integer top_byte(input logic [1:0] t, input integer i);
        case (t)
            T_INT8:  top_byte = i;
            T_INT32: top_byte = 3;
            default: top_byte = i | 1;
        endcase
    endfunction

    function automatic logic fp_nan(input logic [14:0] x);
        fp_nan = (x[14:10] == 5'h1f) && (x[9:0] != 10'd0);
    endfunction

    // ---------------------------------------------------------------- X1
    // The combinational stages are written as automatic functions: the
    // intermediate vectors are function locals, so an event-driven simulator
    // evaluates each stage once per input change (an always_comb that reads
    // back its own part-selected intermediates re-triggers itself in Icarus;
    // this halved the testbench run time). The shifter keeps the per-bit
    // form: a whole-vector (>>> per lane) form simulates faster but
    // synthesised about 50 LUT larger in Yosys.

    // Segmented add/sub and compare: {sel_b, neg_lane, ovf_lane, sum}.
    function automatic logic [43:0] x1_arith(input logic [31:0] va, input logic [31:0] vb,
                                             input logic [1:0] t, input logic [2:0] f);
        logic        sub;
        logic [31:0] bx, sm;
        logic [35:0] add_a, add_b, add_s;
        logic [3:0]  ovf_byte, ovf_l, neg_l, negb_lane, lt_lane, lt, selb;
        sub = (f == ALU_SUB) || (f == ALU_MAX) || (f == ALU_MIN);
        bx = sub ? ~vb : vb;
        // byte i at [9i+1 +: 8], spacer between byte i and i+1 at 9i+9,
        // bit 0 injects the carry-in of the lowest lane
        add_a[0] = sub;
        add_b[0] = sub;
        for (int i = 0; i < 4; i++) begin
            add_a[9*i+1 +: 8] = va[i*8 +: 8];
            add_b[9*i+1 +: 8] = bx[i*8 +: 8];
        end
        for (int i = 0; i < 3; i++) begin
            add_a[9*i+9] = lane_start(t, i + 1) ? sub : 1'b1;
            add_b[9*i+9] = lane_start(t, i + 1) ? sub : 1'b0;
        end
        add_s = add_a + add_b;
        for (int i = 0; i < 4; i++)
            sm[i*8 +: 8] = add_s[9*i+1 +: 8];
        for (int i = 0; i < 4; i++)
            ovf_byte[i] = (va[i*8+7] == bx[i*8+7]) && (sm[i*8+7] != va[i*8+7]);
        for (int i = 0; i < 4; i++) begin
            ovf_l[i] = ovf_byte[top_byte(t, i)];
            neg_l[i] = va[top_byte(t, i)*8 + 7];
            negb_lane[i] = vb[top_byte(t, i)*8 + 7];
            lt_lane[i] = sm[top_byte(t, i)*8 + 7] ^ ovf_byte[top_byte(t, i)];
        end
        // max: take vb when va < vb; min: take vb when !(va < vb) (equal values
        // have identical bits, so either choice gives the same result).
        for (int i = 0; i < 4; i++) begin
            lt[i] = (t == T_FP16) ? (lt_lane[i] ^ (neg_l[i] & negb_lane[i])) : lt_lane[i];
            selb[i] = (f == ALU_MAX) ? lt[i] : !lt[i];
        end
        x1_arith = {selb, neg_l, ovf_l, sm};
    endfunction

    // SRA by s = shm1 + 1: first the segmented arithmetic shift by shm1
    // (always < lane width, so the 8- and 16-bit shift steps never apply to
    // INT8 lanes and the 16-bit step never to INT16 lanes), the last step is
    // in X2.
    function automatic logic [31:0] x1_shift(input logic [31:0] va, input logic [1:0] t,
                                             input logic [4:0] sa);
        logic [31:0] sh_v, stage;
        logic        in_lane, sign;
        sh_v = va;
        for (int st = 0; st < 5; st++) begin
            stage = sh_v;
            for (int i = 0; i < 32; i++) begin
                case (t)
                    T_INT8: begin
                        in_lane = ((i % 8) + (1 << st)) < 8;
                        sign = stage[i | 7];
                    end
                    T_INT32: begin
                        in_lane = (i + (1 << st)) < 32;
                        sign = stage[31];
                    end
                    default: begin
                        in_lane = ((i % 16) + (1 << st)) < 16;
                        sign = stage[i | 15];
                    end
                endcase
                if (sa[st] && (t == T_INT32 || st < 4) && (t != T_INT8 || st < 3))
                    sh_v[i] = in_lane ? stage[(i + (1 << st)) & 31] : sign;
            end
        end
        x1_shift = sh_v;
    endfunction

    logic [31:0] sum, shifted;
    logic [3:0]  ovf_lane, neg_lane, sel_b;
    logic [1:0]  fp_nan_lane;

    assign {sel_b, neg_lane, ovf_lane, sum} = x1_arith(a, b, typ, fn);
    assign shifted = x1_shift(a, typ, shm1);
    assign fp_nan_lane[0] = fp_nan(a[14:0]) || fp_nan(b[14:0]);
    assign fp_nan_lane[1] = fp_nan(a[30:16]) || fp_nan(b[30:16]);

    logic [31:0] a2, b2, s2;
    logic [3:0]  ovf2, neg2, selb2;
    logic [1:0]  nan2;
    logic [1:0]  typ2;
    logic [2:0]  fn2;
    logic        sh_zero2;

    always_ff @(posedge clk) begin
        a2 <= (fn == ALU_PASS) ? p : a;
        b2 <= b;
        s2 <= (fn == ALU_SRA) ? shifted : sum;
        ovf2 <= ovf_lane;
        neg2 <= neg_lane;
        selb2 <= sel_b;
        nan2 <= fp_nan_lane;
        typ2 <= typ;
        fn2 <= fn;
        sh_zero2 <= sh_zero;
    end

    // ---------------------------------------------------------------- X2
    function automatic logic [31:0] x2_result(
        input logic [31:0] sv, input logic [31:0] av, input logic [31:0] bv,
        input logic [3:0] ovf, input logic [3:0] neg, input logic [3:0] selb,
        input logic [1:0] nan, input logic [1:0] t, input logic [2:0] f,
        input logic shz);
        logic [31:0] inc, sra_y, r;
        logic [4:0]  icarry;
        logic [36:0] inc_a, inc_b, inc_s;
        // Segmented increment of t = sv, then drop one bit: (t + 1) >>> 1 in
        // W + 1 bits. The new lane top bit is t_top ^ carry-out of the lane.
        // Same spacer layout as X1: a lane-start spacer (1, 1) injects the
        // next lane's +1 and its sum bit is the carry out of the byte below;
        // bit 36 (0, 0) is the carry out of byte 3.
        inc_a[0] = 1'b1;
        inc_b[0] = 1'b1;
        for (int i = 0; i < 4; i++) begin
            inc_a[9*i+1 +: 8] = sv[i*8 +: 8];
            inc_b[9*i+1 +: 8] = 8'd0;
        end
        for (int i = 0; i < 3; i++) begin
            inc_a[9*i+9] = 1'b1;
            inc_b[9*i+9] = lane_start(t, i + 1);
        end
        inc_a[36] = 1'b0;
        inc_b[36] = 1'b0;
        inc_s = inc_a + inc_b;
        icarry[0] = 1'b1;
        for (int i = 0; i < 4; i++) begin
            inc[i*8 +: 8] = inc_s[9*i+1 +: 8];
            icarry[i+1] = inc_s[9*i+9];   // valid at lane-top bytes only
        end
        for (int i = 0; i < 4; i++) begin
            sra_y[i*8 +: 7] = inc[i*8+1 +: 7];
            if (lane_top(t, i))
                sra_y[i*8+7] = sv[i*8+7] ^ icarry[i+1];
            else
                sra_y[i*8+7] = inc[(i*8+8) & 31];  // bit 0 of the next byte, same lane
        end

        r = av;
        case (f)
            ALU_ADD, ALU_SUB: begin
                for (int i = 0; i < 4; i++) begin
                    if (ovf[i])
                        r[i*8 +: 8] = lane_top(t, i) ? (neg[i] ? 8'h80 : 8'h7f)
                                                             : (neg[i] ? 8'h00 : 8'hff);
                    else
                        r[i*8 +: 8] = sv[i*8 +: 8];
                end
            end
            ALU_MAX, ALU_MIN: begin
                for (int i = 0; i < 4; i++)
                    r[i*8 +: 8] = selb[i] ? bv[i*8 +: 8] : av[i*8 +: 8];
                if (t == T_FP16) begin
                    for (int j = 0; j < 2; j++)
                        if (nan[j]) r[j*16 +: 16] = 16'h7e00;
                end
            end
            ALU_SRA: r = shz ? av : sra_y;
            default: r = av;
        endcase
        x2_result = r;
    endfunction

    logic [31:0] result;
    assign result = x2_result(s2, a2, b2, ovf2, neg2, selb2, nan2, typ2, fn2, sh_zero2);

    always_ff @(posedge clk) begin
        y <= result;
    end
endmodule
