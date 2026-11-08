module JC_counter (
    input         clk,
    input         rst_n,
    output [63:0] Q
);

    // 7-bit ripple counter: only k[0] is on clk; each higher bit toggles on
    // the falling edge of the bit below it.
    wire [6:0] k;
    reg        k0;

    always @(posedge clk or negedge rst_n)
        if (!rst_n) k0 <= 1'b0;
        else        k0 <= ~k0;

    assign k[0] = k0;

    genvar i;
    generate
        for (i = 1; i < 7; i = i + 1) begin : g_rip
            reg t;
            always @(negedge k[i-1] or negedge rst_n)
                if (!rst_n) t <= 1'b0;
                else        t <= ~t;
            assign k[i] = t;
        end
    endgenerate

    // k = 64*hi + 8*m + s; Q[63-(8*a+b)] = hi ^ (8*a+b < 8*m+s).
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
