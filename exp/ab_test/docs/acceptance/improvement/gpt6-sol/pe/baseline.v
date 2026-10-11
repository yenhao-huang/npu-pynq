module pe (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] c
);
    wire [31:0] product = a * b;

    always @(posedge clk or posedge rst) begin
        if (rst)
            c <= 32'b0;
        else
            c <= c + product;
    end
endmodule
