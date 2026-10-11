module edge_detect(
    input  clk,
    input  rst_n,
    input  a,
    output rise,
    output down
);
    reg r1, r2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r1 <= 1'b0;
            r2 <= 1'b0;
        end else begin
            r1 <= a;
            r2 <= r1;
        end
    end
    assign rise = r1 & ~r2;
    assign down = ~r1 & r2;
endmodule
