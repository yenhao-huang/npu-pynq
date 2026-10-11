module multi_8bit (
    input  wire [7:0]  A,
    input  wire [7:0]  B,
    output reg  [15:0] product
);

    reg [15:0] multiplicand;
    integer i;

    // Shift-and-add: for each set bit of B, add the shifted multiplicand.
    always @(*) begin
        product      = 16'd0;
        multiplicand = {8'd0, A};
        for (i = 0; i < 8; i = i + 1) begin
            if (B[i])
                product = product + multiplicand;
            multiplicand = multiplicand << 1;
        end
    end

endmodule
