module pe (
    input wire clk,
    input wire rst,
    input wire [31:0] a,
    input wire [31:0] b,
    output reg [31:0] c
);
    wire [31:0] three_a = a + (a << 1);
    wire [31:0] row [0:11];
    wire [9:0] negative;
    genvar i;
    generate
        for (i = 0; i < 10; i = i + 1) begin : booth
            wire [3:0] code;
            wire one, two, three, four;
            wire [31:0] magnitude;
            if (i == 0) begin : first
                assign code = {b[2:0], 1'b0};
            end else begin : later
                assign code = b[3*i+2:3*i-1];
            end
            assign one = (code == 4'd1) || (code == 4'd2) ||
                         (code == 4'd13) || (code == 4'd14);
            assign two = (code == 4'd3) || (code == 4'd4) ||
                         (code == 4'd11) || (code == 4'd12);
            assign three = (code == 4'd5) || (code == 4'd6) ||
                           (code == 4'd9) || (code == 4'd10);
            assign four = (code == 4'd7) || (code == 4'd8);
            assign negative[i] = (code >= 4'd8) && (code <= 4'd14);
            assign magnitude = ({32{one}} & a) |
                               ({32{two}} & (a << 1)) |
                               ({32{three}} & three_a) |
                               ({32{four}} & (a << 2));
            if (i == 0) begin : packed_first
                assign row[i] = magnitude ^ {32{negative[i]}};
            end else begin : packed_other
                assign row[i] = ((magnitude ^ {32{negative[i]}}) << (3*i)) |
                                ({31'b0, negative[i-1]} << (3*i-3));
            end
        end
    endgenerate
    wire [1:0] high_digit = {b[31], 1'b0} + {1'b0, b[30]} +
                            {1'b0, b[29]};
    wire [1:0] high_product = a[1:0] * high_digit;
    assign row[10] = {high_product, 30'b0} |
                     ({31'b0, negative[9]} << 27);
    assign row[11] = c;

    wire [31:0] s1 [0:8];
    wire [31:0] s2 [0:5];
    wire [31:0] s3 [0:3];
    wire [31:0] s4 [0:2];
    wire [31:0] s5 [0:1];
    genvar j;
    generate
        for (j = 0; j < 4; j = j + 1) begin : level1
            assign s1[2*j] = row[3*j] ^ row[3*j+1] ^ row[3*j+2];
            assign s1[2*j+1] = ((row[3*j] & row[3*j+1]) |
                                  (row[3*j] & row[3*j+2]) |
                                  (row[3*j+1] & row[3*j+2])) << 1;
        end
        assign s1[8] = 32'b0;
        for (j = 0; j < 3; j = j + 1) begin : level2
            assign s2[2*j] = s1[3*j] ^ s1[3*j+1] ^ s1[3*j+2];
            assign s2[2*j+1] = ((s1[3*j] & s1[3*j+1]) |
                                  (s1[3*j] & s1[3*j+2]) |
                                  (s1[3*j+1] & s1[3*j+2])) << 1;
        end
        for (j = 0; j < 2; j = j + 1) begin : level3
            assign s3[2*j] = s2[3*j] ^ s2[3*j+1] ^ s2[3*j+2];
            assign s3[2*j+1] = ((s2[3*j] & s2[3*j+1]) |
                                  (s2[3*j] & s2[3*j+2]) |
                                  (s2[3*j+1] & s2[3*j+2])) << 1;
        end
    endgenerate
    assign s4[0] = s3[0] ^ s3[1] ^ s3[2];
    assign s4[1] = ((s3[0] & s3[1]) | (s3[0] & s3[2]) |
                    (s3[1] & s3[2])) << 1;
    assign s4[2] = s3[3];
    assign s5[0] = s4[0] ^ s4[1] ^ s4[2];
    assign s5[1] = ((s4[0] & s4[1]) | (s4[0] & s4[2]) |
                    (s4[1] & s4[2])) << 1;
    wire [31:0] result = s5[0] + s5[1];

    always @(posedge clk or posedge rst) begin
        if (rst)
            c <= 32'b0;
        else
            c <= result;
    end
endmodule
