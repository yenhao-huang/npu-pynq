module freq_div (
    input  wire CLK_in,
    input  wire RST,
    output reg  CLK_50,
    output wire CLK_10,
    output reg  CLK_1
);

    // Only CLK_50 runs at the input rate.
    // c is a mod-5 shift sequence 000 -> 001 -> 011 -> 110 -> 100 -> 000
    // clocked by CLK_50 rising (odd input cycles), so c[2] is high for
    // input cycles [5,9) of every 10. z re-times c[2] on CLK_50 falling,
    // high for [6,10). CLK_10 = c[2] | z is high for [5,10); the two never
    // switch on the same edge, so the OR is glitch-free.
    // z falling marks CLK_10 falling, which advances the mod-5 cnt_j;
    // cnt_j[2] falling marks cnt_100 == 49, where CLK_1 toggles.
    reg [2:0] c;
    reg       z;
    reg [2:0] cnt_j;

    assign CLK_10 = c[2] | z;

    always @(posedge CLK_in or posedge RST) begin
        if (RST)
            CLK_50 <= 1'b0;
        else
            CLK_50 <= ~CLK_50;
    end

    always @(posedge CLK_50 or posedge RST) begin
        if (RST)
            c <= 3'd0;
        else
            c <= {c[1:0], ~(c[2] | c[1])};
    end

    always @(negedge CLK_50 or posedge RST) begin
        if (RST)
            z <= 1'b0;
        else
            z <= c[2];
    end

    always @(negedge z or posedge RST) begin
        if (RST)
            cnt_j <= 3'd0;
        else
            cnt_j <= {cnt_j[1:0], ~(cnt_j[2] | cnt_j[1])};
    end

    always @(negedge cnt_j[2] or posedge RST) begin
        if (RST)
            CLK_1 <= 1'b0;
        else
            CLK_1 <= ~CLK_1;
    end

endmodule
