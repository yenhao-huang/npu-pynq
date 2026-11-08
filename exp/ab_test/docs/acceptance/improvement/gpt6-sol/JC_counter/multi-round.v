module JC_counter (
    input wire clk,
    input wire rst_n,
    output wire [63:0] Q
);
    reg [6:0] position;
    wire [7:0] lower_pattern;
    wire [15:0] local_pattern;
    wire [3:0] group_one;
    wire [3:0] group_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) position <= 7'b0;
        else position <= position + 7'd1;
    end

    genvar group_index, local_index;
    generate
        for (local_index = 0; local_index < 8; local_index = local_index + 1) begin : lower_decode
            assign lower_pattern[local_index] = position[6] ^
                (position[2:0] > (7 - local_index));
            assign local_pattern[local_index] =
                (position[6] & ~position[3]) |
                (position[3] & lower_pattern[local_index]);
            assign local_pattern[8 + local_index] =
                ((~position[6]) & position[3]) |
                ((~position[3]) & lower_pattern[local_index]);
        end
        for (group_index = 0; group_index < 4; group_index = group_index + 1) begin : group_decode
            assign group_one[group_index] =
                ((~position[6]) & (position[5:4] > (3 - group_index))) |
                (position[6] & (position[5:4] < (3 - group_index)));
            assign group_active[group_index] = position[5:4] == (3 - group_index);
            for (local_index = 0; local_index < 16; local_index = local_index + 1) begin : bits
                assign Q[16*group_index + local_index] = group_one[group_index] |
                    (group_active[group_index] & local_pattern[local_index]);
            end
        end
    endgenerate
endmodule
