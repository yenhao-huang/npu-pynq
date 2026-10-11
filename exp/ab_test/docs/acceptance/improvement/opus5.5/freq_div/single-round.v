// Frequency divider: 100MHz -> 50MHz, 10MHz, 1MHz.
// CLK_1 reuses the divide-by-10 chain (cnt_10 + CLK_10) followed by a
// mod-5 counter, giving the same output timing as a mod-50 counter with
// fewer flops.
module freq_div (
    input      CLK_in,
    input      RST,
    output reg CLK_50,
    output reg CLK_10,
    output reg CLK_1
);
    reg [2:0] cnt_10;
    reg [2:0] cnt_5;

    wire tc10 = cnt_10[2];             // cnt_10 == 4 (only value with bit2 set)
    wire tc50 = tc10 & CLK_10;         // every 10th cycle
    wire tc5  = cnt_5[2];              // cnt_5 == 4

    always @(posedge CLK_in or posedge RST)
        if (RST) CLK_50 <= 1'b0;
        else     CLK_50 <= ~CLK_50;

    always @(posedge CLK_in or posedge RST)
        if (RST) begin
            CLK_10 <= 1'b0;
            cnt_10 <= 3'd0;
        end else if (tc10) begin
            CLK_10 <= ~CLK_10;
            cnt_10 <= 3'd0;
        end else begin
            cnt_10 <= cnt_10 + 3'd1;
        end

    always @(posedge CLK_in or posedge RST)
        if (RST) begin
            CLK_1 <= 1'b0;
            cnt_5 <= 3'd0;
        end else if (tc50) begin
            if (tc5) begin
                CLK_1 <= ~CLK_1;
                cnt_5 <= 3'd0;
            end else begin
                cnt_5 <= cnt_5 + 3'd1;
            end
        end
endmodule
