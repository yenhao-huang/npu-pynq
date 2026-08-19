`timescale 1ns/1ps

module mac_unit #(
    parameter integer A_WIDTH   = 8,
    parameter integer B_WIDTH   = 8,
    parameter integer ACC_WIDTH = 32
) (
    input  logic                            clk,
    input  logic                            rst_n,
    input  logic                            clear,
    input  logic                            valid,
    input  logic signed [A_WIDTH-1:0]        a,
    input  logic signed [B_WIDTH-1:0]        b,
    output logic signed [ACC_WIDTH-1:0]      result,
    output logic                            result_valid
);

    localparam integer PRODUCT_WIDTH = A_WIDTH + B_WIDTH;

    logic signed [PRODUCT_WIDTH-1:0] product;
    logic signed [ACC_WIDTH-1:0]     extended_product;

    assign product = a * b;
    // Sign-extend to match the accumulator width 32 bits
    assign extended_product = {{(ACC_WIDTH-PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}}, product};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            result       <= '0;
            result_valid <= 1'b0;
        end else if (clear) begin
            result       <= '0;
            result_valid <= 1'b0;
        end else begin
            result_valid <= valid;
            if (valid) begin
                result <= result + extended_product;
            end
        end
    end

endmodule
