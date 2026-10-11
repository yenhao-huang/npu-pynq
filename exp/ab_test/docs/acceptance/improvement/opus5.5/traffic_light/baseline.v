module traffic_light (
    input  wire       rst_n,
    input  wire       clk,
    input  wire       pass_request,
    output wire [7:0] clock,
    output reg        red,
    output reg        yellow,
    output reg        green
);

    parameter idle      = 2'd0,
              s1_red    = 2'd1,
              s2_yellow = 2'd2,
              s3_green  = 2'd3;

    reg [7:0] cnt;
    reg [1:0] state;
    reg       p_red, p_yellow, p_green;

    // State transition logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= idle;
            p_red    <= 1'b0;
            p_green  <= 1'b0;
            p_yellow <= 1'b0;
        end else begin
            case (state)
                idle: begin
                    p_red    <= 1'b0;
                    p_green  <= 1'b0;
                    p_yellow <= 1'b0;
                    state    <= s1_red;
                end
                s1_red: begin
                    p_red    <= 1'b1;
                    p_green  <= 1'b0;
                    p_yellow <= 1'b0;
                    if (cnt == 8'd3) state <= s3_green;
                    else             state <= s1_red;
                end
                s2_yellow: begin
                    p_red    <= 1'b0;
                    p_green  <= 1'b0;
                    p_yellow <= 1'b1;
                    if (cnt == 8'd3) state <= s1_red;
                    else             state <= s2_yellow;
                end
                s3_green: begin
                    p_red    <= 1'b0;
                    p_green  <= 1'b1;
                    p_yellow <= 1'b0;
                    if (cnt == 8'd3) state <= s2_yellow;
                    else             state <= s3_green;
                end
            endcase
        end
    end

    // Counter logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt <= 8'd10;
        else if (pass_request && green && (cnt > 8'd10))
            cnt <= 8'd10;
        else if (!green && p_green)
            cnt <= 8'd60;
        else if (!yellow && p_yellow)
            cnt <= 8'd5;
        else if (!red && p_red)
            cnt <= 8'd10;
        else
            cnt <= cnt - 8'd1;
    end

    assign clock = cnt;

    // Output registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            red    <= 1'b0;
            yellow <= 1'b0;
            green  <= 1'b0;
        end else begin
            red    <= p_red;
            yellow <= p_yellow;
            green  <= p_green;
        end
    end

endmodule
