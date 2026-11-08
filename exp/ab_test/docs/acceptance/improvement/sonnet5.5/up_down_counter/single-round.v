module up_down_counter (
    input clk,
    input reset,
    input up_down,
    output reg [15:0] count
);
    always @(posedge clk) begin
        if (reset)
            count <= 16'd0;
        else
            count <= count + {{15{~up_down}}, 1'b1};
    end
endmodule
