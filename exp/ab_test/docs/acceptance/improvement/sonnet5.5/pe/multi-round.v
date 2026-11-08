module pe (
    input clk,
    input rst,
    input [31:0] a,
    input [31:0] b,
    output [31:0] c
);
    // accumulator kept in carry-save form (cs + cc); c = cs + cc
    reg [31:0] cs, cc;

    function integer nxt(input integer n);
        nxt = (n / 3) * 2 + (n % 3);
    endfunction
    function integer cnt(input integer s);
        integer j;
        begin
            cnt = 34;
            for (j = 0; j < s; j = j + 1) cnt = nxt(cnt);
        end
    endfunction

    // stage s holds cnt(s) rows of 32 bits, flattened
    wire [34*32-1:0] st [0:8];
    genvar i, s, g;
    generate
        for (i = 0; i < 32; i = i + 1) begin : pp
            assign st[0][i*32 +: 32] = b[i] ? (a << i) : 32'd0;
        end
        assign st[0][32*32 +: 32] = cs;
        assign st[0][33*32 +: 32] = cc;
        for (s = 0; s < 8; s = s + 1) begin : stg
            localparam N = cnt(s);
            localparam G = N / 3;
            localparam R = N % 3;
            for (g = 0; g < G; g = g + 1) begin : fa
                wire [31:0] x = st[s][(3*g)*32 +: 32];
                wire [31:0] y = st[s][(3*g+1)*32 +: 32];
                wire [31:0] z = st[s][(3*g+2)*32 +: 32];
                assign st[s+1][(2*g)*32 +: 32]   = x ^ y ^ z;
                assign st[s+1][(2*g+1)*32 +: 32] = ((x & y) | (x & z) | (y & z)) << 1;
            end
            for (g = 0; g < R; g = g + 1) begin : rest
                assign st[s+1][(2*G+g)*32 +: 32] = st[s][(3*G+g)*32 +: 32];
            end
            if (N + 0 < 34) begin : pad
                assign st[s+1][34*32-1 : nxt(N)*32] = 'd0;
            end
        end
    endgenerate

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cs <= 32'd0;
            cc <= 32'd0;
        end else begin
            cs <= st[8][0 +: 32];
            cc <= st[8][32 +: 32];
        end
    end
    assign c = cs + cc;
endmodule
