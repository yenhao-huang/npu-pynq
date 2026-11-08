module fsm (
    input  wire IN,
    input  wire CLK,
    input  wire RST,
    output wire MATCH
);
    reg [3:0] history;

    always @(posedge CLK or posedge RST) begin
        if (RST)
            history <= 4'b0000;
        else
            history <= {history[2:0], IN};
    end

    assign MATCH = !RST && (history == 4'b1001) && IN;
endmodule
