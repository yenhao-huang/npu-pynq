module freq_div (
    input  wire CLK_in,
    input  wire RST,
    output reg  CLK_50,
    output reg  CLK_10,
    output reg  CLK_1
);

    reg [1:0] phase_hi;
    reg [2:0] cnt_50;
    wire phase_lo = CLK_50 ^ CLK_10;
    wire tick5 = (phase_hi == 2'd2);
    wire tick10 = tick5 && CLK_10;
    wire tick50 = tick10 && (cnt_50 == 3'd4);

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            CLK_50 <= 1'b0;
            CLK_10 <= 1'b0;
            CLK_1 <= 1'b0;
            phase_hi <= 2'd0;
            cnt_50 <= 3'd0;
        end else begin
            CLK_50 <= ~CLK_50;
            CLK_10 <= CLK_10 ^ tick5;
            CLK_1 <= CLK_1 ^ tick50;
            if (tick5)
                phase_hi <= 2'd0;
            else if (phase_lo)
                phase_hi <= phase_hi + 2'd1;
            if (tick10)
                cnt_50 <= (cnt_50 == 3'd4) ? 3'd0 : cnt_50 + 3'd1;
        end
    end

endmodule
