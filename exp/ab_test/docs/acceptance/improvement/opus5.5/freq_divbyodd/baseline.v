module freq_divbyodd (
    input  clk,
    input  rst_n,
    output clk_div
);
    parameter NUM_DIV = 5;
    localparam W = (NUM_DIV > 2) ? $clog2(NUM_DIV) : 1;

    reg [W-1:0] cnt1;
    reg [W-1:0] cnt2;
    reg         clk_div1;
    reg         clk_div2;

    // Rising-edge counter and divider
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt1 <= {W{1'b0}};
        else if (cnt1 < NUM_DIV - 1)
            cnt1 <= cnt1 + 1'b1;
        else
            cnt1 <= {W{1'b0}};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            clk_div1 <= 1'b1;
        else if (cnt1 < NUM_DIV / 2)
            clk_div1 <= 1'b1;
        else
            clk_div1 <= 1'b0;
    end

    // Falling-edge counter and divider
    always @(negedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt2 <= {W{1'b0}};
        else if (cnt2 < NUM_DIV - 1)
            cnt2 <= cnt2 + 1'b1;
        else
            cnt2 <= {W{1'b0}};
    end

    always @(negedge clk or negedge rst_n) begin
        if (!rst_n)
            clk_div2 <= 1'b1;
        else if (cnt2 < NUM_DIV / 2)
            clk_div2 <= 1'b1;
        else
            clk_div2 <= 1'b0;
    end

    assign clk_div = clk_div1 | clk_div2;

endmodule
