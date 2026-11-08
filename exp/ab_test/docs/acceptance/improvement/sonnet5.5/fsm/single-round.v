module fsm(
    input IN,
    input CLK,
    input RST,
    output MATCH
);
    localparam S0 = 3'd0, S1 = 3'd1, S2 = 3'd2, S3 = 3'd3, S4 = 3'd4;
    reg [2:0] st;

    assign MATCH = (st == S4) & IN;

    always @(posedge CLK or posedge RST) begin
        if (RST) st <= S0;
        else begin
            case (st)
                S0: st <= IN ? S1 : S0;
                S1: st <= IN ? S1 : S2;
                S2: st <= IN ? S1 : S3;
                S3: st <= IN ? S4 : S0;
                S4: st <= IN ? S1 : S2;
                default: st <= S0;
            endcase
        end
    end
endmodule
