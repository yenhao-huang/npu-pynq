module ring_counter (
    input wire clk,
    input wire reset,
    output wire [7:0] out
);
    reg [2:0] state;

    always @(posedge clk or posedge reset) begin
        if (reset)
            state <= 3'b111;
        else
            state <= state - 3'd1;
    end

    assign out = 8'b0000_0001 << (~state);
endmodule
