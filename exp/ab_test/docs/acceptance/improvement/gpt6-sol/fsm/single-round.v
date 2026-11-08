module fsm (
    input  wire IN,
    input  wire CLK,
    input  wire RST,
    output wire MATCH
);
    reg one;
    reg ten;
    reg hundred;

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            one <= 1'b0;
            ten <= 1'b0;
            hundred <= 1'b0;
        end else begin
            one <= IN;
            ten <= !IN && one;
            hundred <= (!IN && ten) || (IN && hundred && !one);
        end
    end

    assign MATCH = !RST && hundred && one && IN;
endmodule
