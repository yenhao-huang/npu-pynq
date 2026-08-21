`timescale 1ns/1ps

module npu_pe #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         clear,
    input  logic                         enable,
    input  logic signed [DATA_WIDTH-1:0] a_in,
    input  logic signed [DATA_WIDTH-1:0] b_in,
    input  logic                         a_valid_in,
    input  logic                         b_valid_in,
    output logic signed [DATA_WIDTH-1:0] a_out,
    output logic signed [DATA_WIDTH-1:0] b_out,
    output logic                         a_valid_out,
    output logic                         b_valid_out,
    output logic signed [ACC_WIDTH-1:0]  accumulator
);
    localparam integer PRODUCT_WIDTH = 2 * DATA_WIDTH;
    localparam integer SUM_WIDTH = ACC_WIDTH + 1;
    localparam integer PRODUCT_EXTENSION = SUM_WIDTH - PRODUCT_WIDTH;

    localparam logic signed [ACC_WIDTH:0] ACC_MAX_EXT = {
        2'b00, {(ACC_WIDTH-1){1'b1}}
    };
    localparam logic signed [ACC_WIDTH:0] ACC_MIN_EXT = {
        2'b11, {(ACC_WIDTH-1){1'b0}}
    };

    logic signed [PRODUCT_WIDTH-1:0] product;
    logic signed [ACC_WIDTH:0] product_extended;
    logic signed [ACC_WIDTH:0] accumulator_extended;
    logic signed [ACC_WIDTH:0] sum_extended;
    logic signed [ACC_WIDTH-1:0] saturated_sum;

    initial begin
        if (DATA_WIDTH != 8) begin
            $fatal(1, "npu_pe ABI v1 requires DATA_WIDTH=8");
        end
        if (ACC_WIDTH != 32) begin
            $fatal(1, "npu_pe ABI v1 requires ACC_WIDTH=32");
        end
        if (PRODUCT_EXTENSION < 1) begin
            $fatal(1, "npu_pe accumulator must be wider than product");
        end
    end

    always @* begin
        product = $signed(a_in) * $signed(b_in);
        product_extended = {
            {PRODUCT_EXTENSION{product[PRODUCT_WIDTH-1]}}, product
        };
        accumulator_extended = {accumulator[ACC_WIDTH-1], accumulator};
        sum_extended = accumulator_extended + product_extended;

        if (sum_extended > ACC_MAX_EXT) begin
            saturated_sum = {1'b0, {(ACC_WIDTH-1){1'b1}}};
        end else if (sum_extended < ACC_MIN_EXT) begin
            saturated_sum = {1'b1, {(ACC_WIDTH-1){1'b0}}};
        end else begin
            saturated_sum = sum_extended[ACC_WIDTH-1:0];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
        end else if (clear) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
        end else if (enable) begin
            a_out <= a_in;
            b_out <= b_in;
            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;
            if (a_valid_in && b_valid_in) begin
                accumulator <= saturated_sum;
            end
        end
    end
endmodule
