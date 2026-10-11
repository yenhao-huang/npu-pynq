`timescale 1ns/1ps

// Self-checking testbench for npu_pe_mp (SHARED_FP_ADDER = 0, local adder).
// - Two instances on the same stimulus: DSP_STYLE "yes" (INT accumulation in
//   the DSP48 with the one-cycle-late clamp) and a lockstep twin with
//   DSP_STYLE "no" (LUT multiplier, fabric saturating add); every output of
//   the twin must equal the DSP version on every clock.
// - Reset (initial and mid-job) and clear behaviour, operand/valid forwarding
//   and stall (enable low) checked on every clock.
// - Exact accumulator timing: INT pairs land 2 cycles after they are
//   presented, FP16 pairs 6 cycles after; minimum FP16 spacing (3 cycles) uses
//   the adder bypass; an FP32 addition in flight completes during a stall and
//   is discarded by clear; a stall in the issue cycle neither issues nor
//   repeats the issue.
// - FP16 tiny products (subnormal x subnormal) and near-total cancellation
//   (the local adder runs with SUBNORMALS = 0 and checks that assumption).
// - Reserved precision 3 performs no MAC.
// - Saturation corners: INT8 32-bit clamp and INT16 48-bit clamp, both signs,
//   exact boundary, repeated saturation (back to back, zero products of
//   either sign) and not sticky. INT16 to INT48_MAX is reached by 2^17 real
//   MACs; the other corners preload the running sum (white box, see
//   preload).
// - INT8 is also compared on every clock with the existing npu_pe (cycle
//   exact, including stalls and saturation).
// - 120 golden cases from src/test/vectors/hx_pe_mp.txt (generated with the
//   hx_numeric model, accumulation in arrival order): INT8 with garbage upper
//   bytes, INT16 endpoints, FP16 random values plus subnormal/Inf/NaN cases.
//   Pairs are fed after 0..3 leading bubbles, with random bubbles (FP16: at
//   least two) and random stalls in between.
module tb_npu_pe_mp;
    logic clk = 1'b0;
    logic rst_n;
    logic clear;
    logic enable;
    logic [1:0] prec;
    logic [15:0] a_in, b_in;
    logic a_valid_in, b_valid_in;
    logic [15:0] a_out, b_out;
    logic a_valid_out, b_valid_out;
    logic [47:0] accumulator;

    npu_pe_mp dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable), .prec(prec),
        .a_in(a_in), .b_in(b_in), .a_valid_in(a_valid_in), .b_valid_in(b_valid_in),
        .a_out(a_out), .b_out(b_out), .a_valid_out(a_valid_out),
        .b_valid_out(b_valid_out), .accumulator(accumulator)
    );
    // Lockstep twin with the LUT multiplier and fabric accumulator
    // (DSP_STYLE = "no", used beyond the array's DSP budget): every output
    // must match the DSP version on every clock.
    logic [15:0] lut_a_out, lut_b_out;
    logic lut_a_valid_out, lut_b_valid_out;
    logic [47:0] lut_accumulator;
    /* verilator lint_off PINCONNECTEMPTY */
    npu_pe_mp #(.DSP_STYLE("no")) dut_lut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable), .prec(prec),
        .a_in(a_in), .b_in(b_in), .a_valid_in(a_valid_in), .b_valid_in(b_valid_in),
        .a_out(lut_a_out), .b_out(lut_b_out), .a_valid_out(lut_a_valid_out),
        .b_valid_out(lut_b_valid_out), .accumulator(lut_accumulator),
        .fp_add_req(), .fp_add_a(), .fp_add_b(), .fp_add_y(32'h0)
    );
    /* verilator lint_on PINCONNECTEMPTY */
    // The existing INT8 npu_pe as a cycle-exact reference for prec = 0: its
    // 32-bit accumulator, sign-extended, must equal accumulator every clock.
    logic [7:0] ref_a_out, ref_b_out;
    logic ref_a_valid_out, ref_b_valid_out;
    logic [31:0] ref_accumulator;
    npu_pe ref_int8 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable),
        .a_in(a_in[7:0]), .b_in(b_in[7:0]), .a_valid_in(a_valid_in),
        .b_valid_in(b_valid_in), .a_out(ref_a_out), .b_out(ref_b_out),
        .a_valid_out(ref_a_valid_out), .b_valid_out(ref_b_valid_out),
        .accumulator(ref_accumulator)
    );
    always #5 clk = ~clk;

    logic [15:0] exp_a_out, exp_b_out;
    logic exp_a_valid_out, exp_b_valid_out;
    integer seed, case_index, cases, file, status, k, i, gap, stalls;
    integer case_prec;
    logic [15:0] pair_a [0:63];
    logic [15:0] pair_b [0:63];
    logic [47:0] expected;
    logic [47:0] golden;
    reg [8*256-1:0] header;

    task automatic mismatch(input [8*96-1:0] what);
        begin
            $display("MISMATCH tb_npu_pe_mp: %0s (case %0d prec %0d acc=%h expected %h)",
                     what, case_index, prec, accumulator, expected);
            $fatal(1);
        end
    endtask

    // One clock with the given inputs; checks forwarding after the edge.
    task automatic step(input logic [15:0] next_a, input logic [15:0] next_b,
                        input logic next_av, input logic next_bv,
                        input logic next_enable, input logic next_clear);
        begin
            @(negedge clk);
            a_in = next_a;
            b_in = next_b;
            a_valid_in = next_av;
            b_valid_in = next_bv;
            enable = next_enable;
            clear = next_clear;
            @(posedge clk);
            #1;
            if (next_clear) begin
                exp_a_out = '0;
                exp_b_out = '0;
                exp_a_valid_out = 1'b0;
                exp_b_valid_out = 1'b0;
            end else if (next_enable) begin
                exp_a_out = next_a;
                exp_b_out = next_b;
                exp_a_valid_out = next_av;
                exp_b_valid_out = next_bv;
            end
            if (a_out !== exp_a_out || b_out !== exp_b_out)
                mismatch("forwarded operands");
            if (a_valid_out !== exp_a_valid_out || b_valid_out !== exp_b_valid_out)
                mismatch("forwarded valids");
            if (lut_accumulator !== accumulator || lut_a_out !== a_out ||
                lut_b_out !== b_out || lut_a_valid_out !== a_valid_out ||
                lut_b_valid_out !== b_valid_out) begin
                expected = lut_accumulator;
                mismatch("DSP_STYLE no twin differs (expected = twin accumulator)");
            end
            if (prec == 2'd0 && accumulator !== {{16{ref_accumulator[31]}}, ref_accumulator}) begin
                expected = {{16{ref_accumulator[31]}}, ref_accumulator};
                mismatch("INT8 differs from npu_pe (expected = npu_pe accumulator)");
            end
        end
    endtask

    task automatic idle(input integer cycles);
        begin
            repeat (cycles) step($random(seed), $random(seed), 1'b0, 1'b0, 1'b1, 1'b0);
        end
    endtask

    task automatic mac(input logic [15:0] x, input logic [15:0] y);
        begin
            step(x, y, 1'b1, 1'b1, 1'b1, 1'b0);
        end
    endtask

    task automatic expect_acc(input logic [47:0] value, input [8*96-1:0] what);
        begin
            expected = value;
            if (accumulator !== value) mismatch(what);
        end
    endtask

    task automatic start_job(input logic [1:0] p);
        begin
            prec = p;
            step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b1);
            expect_acc(48'h0, "clear zeroes the accumulator");
        end
    endtask

    // Preload an INT accumulator between clocks (saturation corners without
    // 2^17 MACs each). White box: the DSP version keeps the running sum in
    // the DSP P register (with its live flag and overflow sign flags), the
    // LUT twin in the accumulator register.
    task automatic preload(input logic [47:0] value);
        begin
            @(negedge clk);
            clear = 1'b0;
            enable = 1'b0;
            a_valid_in = 1'b0;
            b_valid_in = 1'b0;
            dut.accumulator = value;
            dut.g_dsp_mac.dsp_p = value;
            dut.g_dsp_mac.dsp_live = 1'b1;
            dut.g_dsp_mac.base_sign = value[47];
            dut.g_dsp_mac.product_sign = value[47];
            dut_lut.accumulator = value;
            ref_int8.accumulator = value[31:0];
        end
    endtask

    // n back-to-back MACs of the same pair (inputs held, n clocks).
    task automatic mac_n(input logic [15:0] x, input logic [15:0] y, input integer n);
        begin
            step(x, y, 1'b1, 1'b1, 1'b1, 1'b0);
            repeat (n - 1) @(posedge clk);
            #1;
            if (lut_accumulator !== accumulator) begin
                expected = lut_accumulator;
                mismatch("DSP_STYLE no twin differs after a MAC run");
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;  // X -> 0 at time 0 triggers the asynchronous reset
        clear = 1'b0;
        enable = 1'b0;
        prec = 2'd0;
        a_in = '0;
        b_in = '0;
        a_valid_in = 1'b0;
        b_valid_in = 1'b0;
        seed = 32'h136_0004;
        case_index = -1;
        expected = '0;
        exp_a_out = '0;
        exp_b_out = '0;
        exp_a_valid_out = 1'b0;
        exp_b_valid_out = 1'b0;
        #1;
        if (accumulator !== 48'h0 || a_out !== 16'h0 || b_out !== 16'h0 ||
            a_valid_out !== 1'b0 || b_valid_out !== 1'b0)
            mismatch("asynchronous reset");
        @(negedge clk);
        rst_n = 1'b1;

        // ---------------------------------------------- INT8 timing, stall
        start_job(2'd0);
        mac(16'h5580, 16'hAA7F);                        // -128 * 127, upper bytes ignored
        expect_acc(48'h0, "INT8 product is pipelined");
        step(16'h0005, 16'h0006, 1'b1, 1'b0, 1'b1, 1'b0);  // b invalid: no MAC
        expect_acc(48'hffff_ffff_c080, "INT8 MAC lands after two cycles");
        step(16'h0063, 16'h00b3, 1'b1, 1'b1, 1'b0, 1'b0);  // stall: nothing moves
        expect_acc(48'hffff_ffff_c080, "single-valid pair is not accumulated");
        step(16'h0002, 16'h0003, 1'b1, 1'b1, 1'b1, 1'b0);
        idle(1);
        expect_acc(48'hffff_ffff_c086, "INT8 MAC after stall");
        step(16'h0001, 16'h0001, 1'b1, 1'b1, 1'b0, 1'b1);
        expect_acc(48'h0, "clear wins over disabled enable");

        // ------------------------------------------- INT8 saturation corners
        start_job(2'd0);
        preload(48'h0000_7fff_bfff);                    // INT32_MAX - 16384
        mac(16'h0080, 16'h0080);                        // +16384 -> exactly INT32_MAX
        idle(1);
        expect_acc(48'h0000_7fff_ffff, "INT8 exact INT32_MAX is not saturated");
        mac(16'h0001, 16'h0001);
        idle(1);
        expect_acc(48'h0000_7fff_ffff, "INT8 positive saturation");
        mac(16'h0080, 16'h0080);                        // saturated again, back to back
        mac(16'h0000, 16'h00ff);                        // zero product while saturated
        idle(1);
        expect_acc(48'h0000_7fff_ffff, "INT8 repeated positive saturation");
        mac(16'h0080, 16'h007f);                        // -16256
        idle(1);
        expect_acc(48'h0000_7fff_c07f, "INT8 saturation is not sticky");
        start_job(2'd0);
        preload(48'hffff_8000_3f80);                    // INT32_MIN + 16256
        mac(16'h0080, 16'h007f);
        idle(1);
        expect_acc(48'hffff_8000_0000, "INT8 exact INT32_MIN is not saturated");
        mac(16'h00ff, 16'h0001);
        mac(16'h0080, 16'h007f);
        idle(1);
        expect_acc(48'hffff_8000_0000, "INT8 negative saturation");
        mac(16'h0001, 16'h0005);
        idle(1);
        expect_acc(48'hffff_8000_0005, "INT8 negative saturation is not sticky");

        // ------------------------------------------ INT16 saturation corners
        start_job(2'd1);
        mac(16'h8000, 16'h7fff);
        idle(1);
        expect_acc(48'hffff_c000_8000, "INT16 product -32768 * 32767");
        // One corner by real accumulation only (2^17 MACs, no preloading),
        // so the DSP accumulator and its late clamp are run end to end.
        start_job(2'd1);
        mac_n(16'h8000, 16'h8000, 131071);              // 131071 * 2^30
        mac(16'h7fff, 16'h7fff);
        mac(16'h7fff, 16'h0002);
        idle(1);
        expect_acc(48'h7fff_ffff_ffff, "INT16 exact INT48_MAX is not saturated");
        mac(16'h0001, 16'h0001);
        idle(1);
        expect_acc(48'h7fff_ffff_ffff, "INT16 positive saturation");
        mac(16'h8000, 16'h8000);                        // wraps the 48-bit sum again
        mac(16'h0000, 16'h8000);                        // zero products with either sign
        mac(16'h8000, 16'h0000);
        idle(1);
        expect_acc(48'h7fff_ffff_ffff, "INT16 repeated positive saturation");
        mac(16'h8000, 16'h7fff);
        idle(1);
        expect_acc(48'h7fff_c000_7fff, "INT16 saturation is not sticky");
        start_job(2'd1);
        preload(48'h8000_3fff_8000);                    // INT48_MIN + 1073709056
        mac(16'h8000, 16'h7fff);
        idle(1);
        expect_acc(48'h8000_0000_0000, "INT16 exact INT48_MIN is not saturated");
        mac(16'hffff, 16'h0001);
        mac(16'h7fff, 16'h8000);
        mac(16'h0000, 16'h8000);
        mac(16'hffff, 16'h0000);
        idle(1);
        expect_acc(48'h8000_0000_0000, "INT16 negative saturation");
        mac(16'hffff, 16'hfffd);
        idle(1);
        expect_acc(48'h8000_0000_0003, "INT16 negative saturation is not sticky");
        start_job(2'd1);
        preload(48'h7fff_ffff_fff0);
        mac(16'h8000, 16'h8000);                        // +2^30
        mac(16'h0000, 16'h0000);
        idle(1);
        expect_acc(48'h7fff_ffff_ffff, "INT16 positive saturation from preload");
        start_job(2'd1);
        preload(48'h8000_0000_0003);
        mac(16'h8000, 16'h7fff);
        idle(1);
        expect_acc(48'h8000_0000_0000, "INT16 negative saturation from preload");

        // ------------------------------------------ FP16 timing and hazards
        start_job(2'd2);
        mac(16'h3c00, 16'h4000);                        // 1 * 2
        for (i = 1; i <= 5; i = i + 1) begin
            idle(1);
            if (i < 5) expect_acc(48'h0, "FP16 result not early");
        end
        expect_acc(48'h0000_4000_0000, "FP16 MAC lands after six cycles");
        // Minimum spacing (two bubbles): exercises the adder bypass.
        mac(16'h3c00, 16'h3c00);
        idle(2);
        mac(16'h3800, 16'h3c00);
        idle(2);
        mac(16'hbc00, 16'h4200);                        // -3
        idle(6);
        expect_acc(48'h0000_3f00_0000, "FP16 back-to-back at minimum spacing");
        // An issued addition completes while enable is low.
        mac(16'h4000, 16'h4000);                        // +4
        idle(2);                                        // product issued to the adder
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        expect_acc(48'h0000_4090_0000, "FP16 addition completes during stall");
        // A stall before the issue holds the product until enable returns.
        mac(16'h3c00, 16'h3c00);
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        idle(6);
        expect_acc(48'h0000_40b0_0000, "FP16 product held across stall");
        // clear discards an addition in flight.
        mac(16'h4000, 16'h4000);
        idle(2);
        step(16'h0, 16'h0, 1'b0, 1'b0, 1'b1, 1'b1);
        idle(6);
        expect_acc(48'h0, "clear discards FP16 addition in flight");
        // Inf - Inf in the accumulation gives the canonical NaN.
        mac(16'h7c00, 16'h3c00);
        idle(2);
        mac(16'hfc00, 16'h3c00);
        idle(6);
        expect_acc(48'h0000_7fc0_0000, "FP16 Inf - Inf accumulation is NaN");
        // A stall while a product waits to issue must not issue it (nor
        // issue it more than once): stall right at the issue cycle.
        start_job(2'd2);
        mac(16'h4000, 16'h4000);                        // +4
        idle(1);                                        // product waits to issue
        repeat (5) step(16'h0, 16'h0, 1'b1, 1'b1, 1'b0, 1'b0);
        idle(6);
        expect_acc(48'h0000_4080_0000, "FP16 stall at the issue cycle");
        mac(16'h3c00, 16'h3c00);                        // +1
        idle(1);
        repeat (2) step(16'h0, 16'h0, 1'b0, 1'b0, 1'b0, 1'b0);
        idle(6);
        expect_acc(48'h0000_40a0_0000, "FP16 second stall at the issue cycle");
        // Tiny products (subnormal x subnormal, multiples of 2^-48) and a
        // cancellation that needs a long normalisation shift: exact, and the
        // adder (SUBNORMALS = 0) never sees a subnormal or underflows.
        start_job(2'd2);
        mac(16'h0001, 16'h0001);                        // 2^-48
        idle(2);
        mac(16'h0003, 16'h0001);
        idle(2);
        mac(16'h8001, 16'h0001);
        idle(6);
        expect_acc(48'h0000_2840_0000, "FP16 subnormal products, 3 * 2^-48");
        start_job(2'd2);
        mac(16'h03ff, 16'h03ff);
        idle(2);
        mac(16'h83ff, 16'h03fe);                        // cancels all but 1023 * 2^-48
        idle(2);
        mac(16'h8001, 16'h0001);
        idle(6);
        expect_acc(48'h0000_2c7f_8000, "FP16 near-total cancellation of tiny products");

        // ------------------------------- reserved precision 3: no MAC at all
        start_job(2'd3);
        mac(16'h3c00, 16'h3c00);
        mac(16'h0101, 16'h0101);
        idle(1);
        mac(16'h7fff, 16'h7fff);
        idle(7);
        expect_acc(48'h0, "reserved precision performs no MAC");

        // ------------------------------------------- golden vector cases
        file = $fopen("vectors/hx_pe_mp.txt", "r");
        if (file == 0) file = $fopen("src/test/vectors/hx_pe_mp.txt", "r");
        if (file == 0) begin
            $display("MISMATCH tb_npu_pe_mp: cannot open hx_pe_mp.txt");
            $fatal(1);
        end
        status = $fgets(header, file);
        status = $fscanf(file, "%d\n", cases);
        stalls = 0;
        for (case_index = 0; case_index < cases; case_index = case_index + 1) begin
            status = $fscanf(file, "%d %d\n", case_prec, k);
            for (i = 0; i < k; i = i + 1) begin
                status = $fscanf(file, "%h %h\n", pair_a[i], pair_b[i]);
            end
            status = $fscanf(file, "%h\n", golden);
            if (status != 1) begin
                $display("MISMATCH tb_npu_pe_mp: short vector file at case %0d", case_index);
                $fatal(1);
            end
            start_job(case_prec[1:0]);
            // Leading bubbles (a PE inside an array sees r + c of them
            // between clear and its first pair): the first pair must still
            // start from zero, not from the previous job's running sum.
            gap = $random(seed) & 3;
            while (gap > 0) begin
                step($random(seed), $random(seed), $random(seed) & 1, 1'b0, 1'b1, 1'b0);
                gap = gap - 1;
            end
            for (i = 0; i < k; i = i + 1) begin
                mac(pair_a[i], pair_b[i]);
                gap = (case_prec == 2) ? 2 + (($random(seed) & 3) == 0) : ($random(seed) & 1);
                while (gap > 0) begin
                    if (($random(seed) & 7) == 0) begin
                        // Stall with junk valid operands: must be ignored.
                        step($random(seed), $random(seed), 1'b1, 1'b1, 1'b0, 1'b0);
                        stalls = stalls + 1;
                    end else begin
                        // Bubble, sometimes with a single-sided valid.
                        step($random(seed), $random(seed), $random(seed) & 1, 1'b0,
                             1'b1, 1'b0);
                        gap = gap - 1;
                    end
                end
            end
            idle(7);
            expect_acc(golden, "golden accumulation");
        end
        $fclose(file);

        // ------------------------------------------- mid-job reset
        start_job(2'd1);
        mac(16'h1234, 16'h4321);
        idle(1);
        #1;
        rst_n = 1'b0;
        #1;
        if (accumulator !== 48'h0 || a_out !== 16'h0 || b_out !== 16'h0 ||
            a_valid_out !== 1'b0 || b_valid_out !== 1'b0)
            mismatch("mid-job asynchronous reset");

        $display("PASS tb_npu_pe_mp cases=%0d stalls=%0d", cases, stalls);
        $finish;
    end
endmodule
