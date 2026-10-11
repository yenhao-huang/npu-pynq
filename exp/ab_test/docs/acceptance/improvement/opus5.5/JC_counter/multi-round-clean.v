module JC_counter (
    input         clk,
    input         rst_n,
    output [63:0] Q
);

    // The 128-state Johnson sequence is held as a 7-bit binary count
    // k = 64*hi + 8*m + s, and Q[63-(8*a+b)] = hi ^ (8*a+b < 8*m+s).
    // Within group a the bits are a constant unless a == m, so each output
    // is one AOI21 over 8 group signals (E, C) and 8 column signals (Y).
    reg [6:0] k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            k <= 7'd0;
        else
            k <= k + 7'd1;
    end

    wire       hi = k[6];
    wire [2:0] m  = k[5:3];
    wire [2:0] s  = k[2:0];

    wire [7:0] E;   // a == m: this group holds the moving edge
    wire [7:0] C;   // a != m and the group is all zeros
    wire [7:0] Y;   // inverted output pattern of the moving group

    genvar a, b;
    generate
        for (a = 0; a < 8; a = a + 1) begin : g_grp
            assign E[a] = (m == a);
            assign C[a] = (m != a) & (hi == (a < m));
            assign Y[a] = ~(hi ^ (a < s));
        end
        for (a = 0; a < 8; a = a + 1) begin : g_a
            for (b = 0; b < 8; b = b + 1) begin : g_b
                assign Q[63-(8*a+b)] = ~((E[a] & Y[b]) | C[a]);
            end
        end
    endgenerate

endmodule
