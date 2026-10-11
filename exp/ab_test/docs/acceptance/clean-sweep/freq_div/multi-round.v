module freq_div (
    input CLK_in,
    input RST,
    output CLK_50,
    output CLK_10,
    output CLK_1
);
    // a counts 3..12 (cycle mod 10 plus 3), b counts 3..12 (tens plus 3)
    reg [3:0] a, b;
    wire a_wrap = a[3] & a[2];
    wire b_wrap = b[3] & b[2];

    assign CLK_50 = ~a[0];
    assign CLK_10 = a[3];
    assign CLK_1  = b[3];

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            a <= 4'd3;
            b <= 4'd3;
        end else begin
            a <= a_wrap ? 4'd3 : a + 4'd1;
            if (a_wrap) b <= b_wrap ? 4'd3 : b + 4'd1;
        end
    end
endmodule
