module freq_divbyodd #(
    parameter NUM_DIV = 5
)(
    input  wire clk,
    input  wire rst_n,
    output wire clk_div
);
    generate
    if (NUM_DIV == 5) begin : g_div5
        // Per edge, a 3-bit shift register (p,q,r) holding the divider value
        // two, one and zero steps ahead; r is the divider. Sequence from
        // reset: 111 -> 011 -> 001 -> 000 -> 100 -> 110 -> 011 ...
        // so r = 1,1,1,0,0,0,1,1,0,0,0,... and p' = ~(q | r).
        // Bits are stored inverted (P=~p, ...) so all flops reset to 0:
        //   P' = ~(Q & R), Q' = P, R' = Q, clk_div = ~(R1 & R2).
        reg P1, Q1, R1, P2, Q2, R2;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                P1 <= 1'b0; Q1 <= 1'b0; R1 <= 1'b0;
            end else begin
                P1 <= ~(Q1 & R1);
                Q1 <= P1;
                R1 <= Q1;
            end
        end

        always @(negedge clk or negedge rst_n) begin
            if (!rst_n) begin
                P2 <= 1'b0; Q2 <= 1'b0; R2 <= 1'b0;
            end else begin
                P2 <= ~(Q2 & R2);
                Q2 <= P2;
                R2 <= Q2;
            end
        end

        assign clk_div = ~(R1 & R2);
    end else begin : g_gen
        // Generic: per-phase down counter, divider bit is a state bit.
        localparam HI = NUM_DIV / 2;
        localparam LO = NUM_DIV - HI;
        localparam MX = (HI > LO - 1) ? HI : LO - 1;
        localparam W  = (MX > 1) ? $clog2(MX + 1) : 1;

        reg [W-1:0] cnt1, cnt2;
        reg         clk_div1, clk_div2;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                clk_div1 <= 1'b1;
                cnt1     <= HI[W-1:0];
            end else if (cnt1 == {W{1'b0}}) begin
                clk_div1 <= ~clk_div1;
                cnt1     <= clk_div1 ? LO[W-1:0] - 1'b1 : HI[W-1:0] - 1'b1;
            end else begin
                cnt1     <= cnt1 - 1'b1;
            end
        end

        always @(negedge clk or negedge rst_n) begin
            if (!rst_n) begin
                clk_div2 <= 1'b1;
                cnt2     <= HI[W-1:0];
            end else if (cnt2 == {W{1'b0}}) begin
                clk_div2 <= ~clk_div2;
                cnt2     <= clk_div2 ? LO[W-1:0] - 1'b1 : HI[W-1:0] - 1'b1;
            end else begin
                cnt2     <= cnt2 - 1'b1;
            end
        end

        assign clk_div = clk_div1 | clk_div2;
    end
    endgenerate
endmodule
