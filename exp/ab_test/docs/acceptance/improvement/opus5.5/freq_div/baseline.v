module freq_div (
    input  wire CLK_in,
    input  wire RST,
    output reg  CLK_50,
    output reg  CLK_10,
    output reg  CLK_1
);

    // cnt_10 counts 0..4; CLK_10 toggles on cnt_10 == 4.
    // The original cnt_100 (0..49) equals 10*cnt_hi + 5*CLK_10 + cnt_10,
    // so only a mod-5 counter (cnt_hi) is needed on top of cnt_10/CLK_10.
    reg [2:0] cnt_10;
    reg [2:0] cnt_hi;

    wire wrap5  = (cnt_10 == 3'd4);
    wire wrap10 = wrap5 & CLK_10;
    wire wrap50 = wrap10 & (cnt_hi == 3'd4);

    always @(posedge CLK_in or posedge RST) begin
        if (RST)
            CLK_50 <= 1'b0;
        else
            CLK_50 <= ~CLK_50;
    end

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            CLK_10 <= 1'b0;
            cnt_10 <= 3'd0;
        end else if (wrap5) begin
            CLK_10 <= ~CLK_10;
            cnt_10 <= 3'd0;
        end else begin
            cnt_10 <= cnt_10 + 3'd1;
        end
    end

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            CLK_1  <= 1'b0;
            cnt_hi <= 3'd0;
        end else if (wrap50) begin
            CLK_1  <= ~CLK_1;
            cnt_hi <= 3'd0;
        end else if (wrap10) begin
            cnt_hi <= cnt_hi + 3'd1;
        end
    end

endmodule
