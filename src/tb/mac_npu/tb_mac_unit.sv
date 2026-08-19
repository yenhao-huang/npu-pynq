`timescale 1ns/1ps

module tb_mac_unit;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear = 1'b0;
    logic valid = 1'b0;
    logic signed [7:0] a = '0;
    logic signed [7:0] b = '0;
    logic signed [31:0] result;
    logic result_valid;

    logic signed [31:0] expected = '0;
    integer random_a;
    integer random_b;
    integer vector_count = 0;
    integer seed = 32'h004d4143;

    always #5 clk = ~clk;

    mac_unit dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .valid(valid),
        .a(a),
        .b(b),
        .result(result),
        .result_valid(result_valid)
    );

    task automatic apply_mac(input integer operand_a, input integer operand_b);
        begin
            @(negedge clk);
            a = operand_a;
            b = operand_b;
            valid = 1'b1;
            @(posedge clk);
            #1;
            expected = expected + operand_a * operand_b;
            vector_count = vector_count + 1;
            if (!result_valid) begin
                $fatal(1, "result_valid was low for vector %0d", vector_count);
            end
            if (result !== expected) begin
                $fatal(1, "vector %0d failed: a=%0d b=%0d expected=%0d actual=%0d",
                       vector_count, operand_a, operand_b, expected, result);
            end
        end
    endtask

    task automatic apply_idle;
        logic signed [31:0] held_result;
        begin
            @(negedge clk);
            held_result = result;
            valid = 1'b0;
            a = '0;
            b = '0;
            @(posedge clk);
            #1;
            if (result_valid || result !== held_result) begin
                $fatal(1, "idle cycle changed output state");
            end
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        #1;
        if (result !== 0 || result_valid !== 0) begin
            $fatal(1, "reset did not clear outputs");
        end

        @(negedge clk);
        rst_n = 1'b1;

        apply_mac(2, 3);
        apply_mac(4, 5);
        apply_mac(-7, 6);
        apply_mac(-8, -9);
        apply_mac(-128, -128);
        apply_mac(127, -128);
        apply_mac(0, 127);
        apply_idle();

        @(negedge clk);
        clear = 1'b1;
        valid = 1'b1;
        a = 8'sd12;
        b = 8'sd12;
        @(posedge clk);
        #1;
        expected = '0;
        if (result !== 0 || result_valid !== 0) begin
            $fatal(1, "clear did not take priority over valid");
        end
        @(negedge clk);
        clear = 1'b0;
        valid = 1'b0;

        repeat (256) begin
            random_a = ($random(seed) & 8'hff) - 128;
            random_b = ($random(seed) & 8'hff) - 128;
            apply_mac(random_a, random_b);
        end

        apply_idle();
        $display("PASS: %0d MAC vectors matched", vector_count);
        $finish;
    end

endmodule
