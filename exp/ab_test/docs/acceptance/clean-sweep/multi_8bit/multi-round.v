module multi_8bit (
    input  wire [7:0]  A,
    input  wire [7:0]  B,
    output wire [15:0] product
);
    // Shift-and-add: adjacent partial products are summed in narrow pairs,
    // then the four pair sums are merged with a 4:2 carry-save stage.
    wire [9:0]  r01 = {2'b0, A & {8{B[0]}}} + {1'b0, A & {8{B[1]}}, 1'b0};
    wire [9:0]  r23 = {2'b0, A & {8{B[2]}}} + {1'b0, A & {8{B[3]}}, 1'b0};
    wire [9:0]  r45 = {2'b0, A & {8{B[4]}}} + {1'b0, A & {8{B[5]}}, 1'b0};
    wire [9:0]  r67 = {2'b0, A & {8{B[6]}}} + {1'b0, A & {8{B[7]}}, 1'b0};

    wire [15:0] x0 = {6'd0, r01};
    wire [15:0] x1 = {4'd0, r23, 2'd0};
    wire [15:0] x2 = {2'd0, r45, 4'd0};
    wire [15:0] x3 = {r67, 6'd0};

    wire [15:0] t  = x0 ^ x1 ^ x2;
    wire [15:0] k  = ((x0 & x1) | (x0 & x2) | (x1 & x2)) << 1;
    wire [15:0] s  = t ^ x3 ^ k;
    wire [15:0] c  = ((t & x3) | (t & k) | (x3 & k)) << 1;

    assign product = s + c;
endmodule
