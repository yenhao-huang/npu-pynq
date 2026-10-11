module edge_detect(
    input clk,
    input rst_n,
    input a,
    output reg rise,
    output reg down
);
    reg a0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a0   <= 1'b0;
            rise <= 1'b0;
            down <= 1'b0;
        end else begin
            a0   <= a;
            rise <= a & ~a0;
            down <= ~a & a0;
        end
    end
endmodule
