module freq_div (
    input  wire CLK_in,
    input  wire RST,
    output reg  CLK_50,
    output reg  CLK_10,
    output reg  CLK_1
);

    reg [2:0] cnt_10;
    reg [3:0] cnt_50;

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            CLK_50  <= 1'b0;
            CLK_10  <= 1'b0;
            CLK_1   <= 1'b0;
            cnt_10  <= 3'd0;
            cnt_50  <= 4'd0;
        end else begin
            CLK_50 <= ~CLK_50;

            if (cnt_10 == 3'd4) begin
                CLK_10 <= ~CLK_10;
                cnt_10 <= 3'd0;
                if (cnt_50 == 4'd9) begin
                    CLK_1  <= ~CLK_1;
                    cnt_50 <= 4'd0;
                end else begin
                    cnt_50 <= cnt_50 + 4'd1;
                end
            end else begin
                cnt_10 <= cnt_10 + 3'd1;
            end

        end
    end

endmodule
