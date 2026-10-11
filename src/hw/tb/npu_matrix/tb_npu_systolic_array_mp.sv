`timescale 1ns/1ps

// Self-checking testbench for npu_systolic_array_mp (4 x 4), three instances
// on the same stimulus: FP_ADDER_SHARE = 3, 2 and 1 (the last one also with
// a DSP budget of 8, so half its PEs use the LUT multiplier style).
// 60 GEMM cases from src/test/vectors/hx_systolic_mp_4x4.txt (generated with
// the hx_numeric model, C[r][c] accumulated in k order exactly like the PE):
// 20 per precision (INT8 with garbage upper bytes and endpoints, INT16 with
// endpoints, FP16 random values, half of them with subnormal/Inf/NaN
// operands), K = 1..12. Operands enter with the usual skew; consecutive k
// are S_k beats apart, the same on every edge stream: S_k = Smin, Smin + 1,
// or random per k in Smin..Smin+2 (INT: Smin = 1, FP16: Smin = 3, the
// minimum FP16 spacing, which the shared FP32 adders rely on). Bubble beats
// carry random data with valid = 0,
// and random stall cycles (enable low) carry random data and random valids,
// which must be ignored. The results are checked at the earliest cycle the
// array's controller contract allows (minimum drain). Reset and clear (with
// junk valid edges) are checked.
module tb_npu_systolic_array_mp;
    localparam integer ROWS = 4;
    localparam integer COLUMNS = 4;
    localparam integer MAX_K = 12;

    logic clk = 1'b0;
    logic rst_n;
    logic clear, enable;
    logic [1:0] prec;
    logic [ROWS*16-1:0] a_in;
    logic [ROWS-1:0] a_valid_in;
    logic [COLUMNS*16-1:0] b_in;
    logic [COLUMNS-1:0] b_valid_in;
    localparam integer ACC_BITS = ROWS * COLUMNS * 48;
    // Three arrays on the same stimulus, one per FP32 adder sharing factor:
    // variant 0 = FP_ADDER_SHARE 3 (default; groups of 3 + 1 on 4 columns),
    // variant 1 = 2 (two full groups), variant 2 = 1 (an adder per PE, and
    // DSP_BUDGET = 8 so PEs 8..15 use the LUT multiplier style).
    wire [3*ACC_BITS-1:0] acc_all;

    npu_systolic_array_mp #(.ROWS(ROWS), .COLUMNS(COLUMNS)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable), .prec(prec),
        .a_in(a_in), .a_valid_in(a_valid_in), .b_in(b_in), .b_valid_in(b_valid_in),
        .accumulators(acc_all[0 +: ACC_BITS])
    );
    npu_systolic_array_mp #(.ROWS(ROWS), .COLUMNS(COLUMNS), .FP_ADDER_SHARE(2)) dut_share2 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable), .prec(prec),
        .a_in(a_in), .a_valid_in(a_valid_in), .b_in(b_in), .b_valid_in(b_valid_in),
        .accumulators(acc_all[ACC_BITS +: ACC_BITS])
    );
    npu_systolic_array_mp #(.ROWS(ROWS), .COLUMNS(COLUMNS), .FP_ADDER_SHARE(1),
                            .DSP_BUDGET(8)) dut_share1 (
        .clk(clk), .rst_n(rst_n), .clear(clear), .enable(enable), .prec(prec),
        .a_in(a_in), .a_valid_in(a_valid_in), .b_in(b_in), .b_valid_in(b_valid_in),
        .accumulators(acc_all[2*ACC_BITS +: ACC_BITS])
    );
    always #5 clk = ~clk;

    logic [15:0] mat_a [0:ROWS-1][0:MAX_K-1];
    logic [15:0] mat_b [0:MAX_K-1][0:COLUMNS-1];
    logic [47:0] mat_c [0:ROWS-1][0:COLUMNS-1];
    integer file, status, cases, case_index, case_prec, k, r, c, i, v;
    integer spacing, beat, last_beat, index, stalls, stall_rate, prec_cases [0:2];
    integer kstart [0:MAX_K-1];
    reg [8*256-1:0] header;
    logic [47:0] word;

    task automatic mismatch(input [8*64-1:0] what, input integer variant,
                            input integer row, input integer column);
        begin
            $display("MISMATCH tb_npu_systolic_array_mp: %0s (variant %0d case %0d prec %0d C[%0d][%0d] got %h expected %h)",
                     what, variant, case_index, case_prec, row, column,
                     acc_all[variant*ACC_BITS + (row*COLUMNS+column)*48 +: 48],
                     mat_c[row][column]);
            $fatal(1);
        end
    endtask

    // Index of the k whose beat is t on a stream, or -1.
    function automatic integer k_at(input integer t);
        integer kk;
        begin
            k_at = -1;
            for (kk = 0; kk < k; kk = kk + 1)
                if (kstart[kk] == t) k_at = kk;
        end
    endfunction

    // Drive the edge streams for one enabled beat of the skewed schedule.
    task automatic drive_beat(input integer t);
        integer rr, cc, kk;
        begin
            for (rr = 0; rr < ROWS; rr = rr + 1) begin
                kk = k_at(t - rr);
                if (kk >= 0) begin
                    a_in[rr*16 +: 16] = mat_a[rr][kk];
                    a_valid_in[rr] = 1'b1;
                end else begin
                    a_in[rr*16 +: 16] = $random;
                    a_valid_in[rr] = 1'b0;
                end
            end
            for (cc = 0; cc < COLUMNS; cc = cc + 1) begin
                kk = k_at(t - cc);
                if (kk >= 0) begin
                    b_in[cc*16 +: 16] = mat_b[kk][cc];
                    b_valid_in[cc] = 1'b1;
                end else begin
                    b_in[cc*16 +: 16] = $random;
                    b_valid_in[cc] = 1'b0;
                end
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
        a_valid_in = '0;
        b_valid_in = '0;
        stalls = 0;
        prec_cases[0] = 0;
        prec_cases[1] = 0;
        prec_cases[2] = 0;
        case_index = -1;
        case_prec = 0;
        #1;
        if (acc_all !== '0) begin
            $display("MISMATCH tb_npu_systolic_array_mp: reset leaves accumulators nonzero");
            $fatal(1);
        end
        @(negedge clk);
        rst_n = 1'b1;

        file = $fopen("vectors/hx_systolic_mp_4x4.txt", "r");
        if (file == 0) file = $fopen("src/test/vectors/hx_systolic_mp_4x4.txt", "r");
        if (file == 0) begin
            $display("MISMATCH tb_npu_systolic_array_mp: cannot open vector file");
            $fatal(1);
        end
        status = $fgets(header, file);
        status = $fscanf(file, "%d\n", cases);

        for (case_index = 0; case_index < cases; case_index = case_index + 1) begin
            status = $fscanf(file, "%d %d\n", case_prec, k);
            for (r = 0; r < ROWS; r = r + 1)
                for (i = 0; i < k; i = i + 1)
                    begin
                        status = $fscanf(file, "%h", word);
                        mat_a[r][i] = word[15:0];
                    end
            for (i = 0; i < k; i = i + 1)
                for (c = 0; c < COLUMNS; c = c + 1)
                    begin
                        status = $fscanf(file, "%h", word);
                        mat_b[i][c] = word[15:0];
                    end
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLUMNS; c = c + 1)
                    begin
                        status = $fscanf(file, "%h", word);
                        mat_c[r][c] = word;
                    end
            if (status != 1) begin
                $display("MISMATCH tb_npu_systolic_array_mp: short vector file at case %0d",
                         case_index);
                $fatal(1);
            end
            prec_cases[case_prec] = prec_cases[case_prec] + 1;
            // spacing mode: minimum, minimum + 1, or random per k
            spacing = (case_prec == 2) ? 3 : 1;
            kstart[0] = 0;
            for (i = 1; i < k; i = i + 1) begin
                case ((case_index / 3) % 3)
                    0: kstart[i] = kstart[i-1] + spacing;
                    1: kstart[i] = kstart[i-1] + spacing + 1;
                    default: kstart[i] = kstart[i-1] + spacing + ($unsigned($random) % 3);
                endcase
            end
            stall_rate = (case_index % 4 == 0) ? 0 : 6;

            // clear (with junk on the edges) starts the job
            @(negedge clk);
            prec = case_prec[1:0];
            clear = 1'b1;
            enable = 1'b1;
            a_valid_in = '1;
            b_valid_in = '1;
            @(negedge clk);
            clear = 1'b0;
            if (acc_all !== '0) begin
                $display("MISMATCH tb_npu_systolic_array_mp: clear leaves accumulators nonzero");
                $fatal(1);
            end

            // Skewed feed, then the minimum drain documented in
            // npu_systolic_array_mp: the last pair reaches PE (ROWS-1,
            // COLUMNS-1) in beat tL = kstart[k-1] + ROWS + COLUMNS - 2; the
            // results are final after beat tL + 1 (INT) or after beat tL + 2
            // and three more clocks, here with enable low (FP16).
            last_beat = kstart[k-1] + ROWS + COLUMNS - 2 + ((case_prec == 2) ? 2 : 1);
            beat = 0;
            while (beat <= last_beat) begin
                if (stall_rate != 0 && ($random % stall_rate) == 0) begin
                    enable = 1'b0;
                    a_in = {$random, $random};
                    b_in = {$random, $random};
                    a_valid_in = $random;
                    b_valid_in = $random;
                    stalls = stalls + 1;
                end else begin
                    enable = 1'b1;
                    drive_beat(beat);
                    beat = beat + 1;
                end
                @(negedge clk);
            end
            enable = 1'b0;
            a_in = {$random, $random};
            b_in = {$random, $random};
            a_valid_in = $random;
            b_valid_in = $random;
            if (case_prec == 2) repeat (3) @(negedge clk);

            for (v = 0; v < 3; v = v + 1)
                for (r = 0; r < ROWS; r = r + 1)
                    for (c = 0; c < COLUMNS; c = c + 1)
                        if (acc_all[v*ACC_BITS + (r*COLUMNS+c)*48 +: 48] !== mat_c[r][c])
                            mismatch("GEMM result", v, r, c);
        end
        $fclose(file);

        // mid-job reset clears everything
        @(negedge clk);
        enable = 1'b1;
        drive_beat(0);
        #1;
        rst_n = 1'b0;
        #1;
        if (acc_all !== '0) begin
            $display("MISMATCH tb_npu_systolic_array_mp: mid-job reset");
            $fatal(1);
        end

        $display("PASS tb_npu_systolic_array_mp cases=%0d int8=%0d int16=%0d fp16=%0d stalls=%0d fp_adder_share=3,2,1",
                 cases, prec_cases[0], prec_cases[1], prec_cases[2], stalls);
        $finish;
    end
endmodule
