module JC_counter (
    input         clk,
    input         rst_n,
    output [63:0] Q
);

    // The 128-state Johnson sequence is held as a 7-bit binary count k.
    // State k has its top (k mod 64) bits set to ~k[6] and the rest set to k[6].
    reg [6:0] k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            k <= 7'd0;
        else
            k <= k + 7'd1;
    end

    genvar j;
    generate
        for (j = 0; j < 64; j = j + 1) begin : g_dec
            assign Q[63-j] = k[6] ^ (j < k[5:0]);
        end
    endgenerate

endmodule
