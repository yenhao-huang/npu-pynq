module fsm(
    input IN,
    input CLK,
    input RST,
    output MATCH
);
    reg [3:0] h;

    assign MATCH = (h == 4'b1001) & IN & ~RST;

    always @(posedge CLK or posedge RST) begin
        if (RST) h <= 4'b0000;
        else h <= {h[2:0], IN};
    end
endmodule
