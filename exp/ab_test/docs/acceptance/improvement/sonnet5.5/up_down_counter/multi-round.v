module up_down_counter (
    input clk,
    input reset,
    input up_down,
    output reg [15:0] count
);
    wire [15:0] x = count ^ {16{~up_down}};
    wire [15:0] t;
    assign t[0] = 1'b1;
    genvar i;
    generate
        for (i = 1; i < 16; i = i + 1) begin : g
            assign t[i] = &x[i-1:0];
        end
    endgenerate
    always @(posedge clk) begin
        if (reset)
            count <= 16'd0;
        else
            count <= count ^ t;
    end
endmodule
