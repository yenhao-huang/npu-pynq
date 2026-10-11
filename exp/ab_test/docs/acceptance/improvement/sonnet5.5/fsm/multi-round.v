module fsm(
    input IN,
    input CLK,
    input RST,
    output MATCH
);
    // S0=000 S1=001 S2=010 S3=100 S4=101
    reg a, b, c;

    assign MATCH = a & c & IN;

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            a <= 1'b0;
            b <= 1'b0;
            c <= 1'b0;
        end else begin
            a <= IN ? (a & ~c) : b;
            b <= ~IN & c;
            c <= IN;
        end
    end
endmodule
