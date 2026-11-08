module edge_detect (
    input  wire clk,
    input  wire rst_n,
    input  wire a,
    output wire rise,
    output wire down
);

    // a_d1 = a delayed 1 clk; a_d2n = ~a delayed 2 clks (set flop).
    // Lets rise be a plain AND2 and down a plain NOR2.
    reg a_d1, a_d2n;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_d1  <= 1'b0;
            a_d2n <= 1'b1;
        end else begin
            a_d1  <= a;
            a_d2n <= ~a_d1;
        end
    end

    assign rise =  a_d1 & a_d2n;
    assign down = ~(a_d1 | a_d2n);

endmodule
