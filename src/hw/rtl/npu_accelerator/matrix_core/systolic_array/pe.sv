`timescale 1ns/1ps

// Processing element (PE): one cell of the systolic array. It multiplies the
// A and B bytes on its inputs, registers the product, and one cycle later adds
// it to a saturating INT32 accumulator if both bytes were valid. A and B, with
// their valid bits, are registered and forwarded one cycle later: A to the
// next column of the same row, B to the next row of the same column
// (systolic_array wires a_out and b_out that way). clear zeroes every
// register; all of them hold while enable is low.
module pe #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter DSP_STYLE = "yes"
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
    localparam integer PRODUCT_EXTENSION = ACC_WIDTH - PRODUCT_WIDTH;

    logic signed [PRODUCT_WIDTH-1:0] product;
    logic signed [ACC_WIDTH-1:0] product_extended;
    (* use_dsp = DSP_STYLE *) logic signed [ACC_WIDTH-1:0] product_pipeline;
    logic                         product_valid_pipeline;
    logic signed [ACC_WIDTH-1:0] wrapped_sum;
    logic signed [ACC_WIDTH-1:0] saturated_sum;

    initial begin
        if (DSP_STYLE != "yes" && DSP_STYLE != "no") begin
            $fatal(1, "pe DSP_STYLE must be yes or no");
        end
        if (DATA_WIDTH != 8) begin
            $fatal(1, "pe requires DATA_WIDTH=8 (INT8 operands)");
        end
        if (ACC_WIDTH != 32) begin
            $fatal(1, "pe requires ACC_WIDTH=32 (INT32 accumulator)");
        end
        if (PRODUCT_EXTENSION < 1) begin
            $fatal(1, "pe accumulator must be wider than product");
        end
    end

    always @* begin
        product = $signed(a_in) * $signed(b_in);
        product_extended = {
            {PRODUCT_EXTENSION{product[PRODUCT_WIDTH-1]}}, product
        };
        wrapped_sum = accumulator + product_pipeline;

        if (!accumulator[ACC_WIDTH-1] && !product_pipeline[ACC_WIDTH-1] &&
            wrapped_sum[ACC_WIDTH-1]) begin
            saturated_sum = {1'b0, {(ACC_WIDTH-1){1'b1}}};
        end else if (accumulator[ACC_WIDTH-1] && product_pipeline[ACC_WIDTH-1] &&
                     !wrapped_sum[ACC_WIDTH-1]) begin
            saturated_sum = {1'b1, {(ACC_WIDTH-1){1'b0}}};
        end else begin
            saturated_sum = wrapped_sum;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
            product_pipeline <= '0;
            product_valid_pipeline <= 1'b0;
        end else if (clear) begin
            a_out <= '0;
            b_out <= '0;
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
            accumulator <= '0;
            product_pipeline <= '0;
            product_valid_pipeline <= 1'b0;
        end else if (enable) begin
            a_out <= a_in;
            b_out <= b_in;
            a_valid_out <= a_valid_in;
            b_valid_out <= b_valid_in;
            product_pipeline <= product_extended;
            product_valid_pipeline <= a_valid_in && b_valid_in;
            if (product_valid_pipeline) begin
                accumulator <= saturated_sum;
            end
        end
    end
endmodule
