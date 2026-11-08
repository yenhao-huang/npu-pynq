// Minimal-area traffic_light: cycle-equivalent to the registered-pipeline
// version, using only the 2-bit phase and a 6-bit counter (8 flops).
// Invariants of the reachable state space:
//   - cnt stays in 1..60, so cnt[7:6] are always 0
//   - the lights lag the phase by two cycles (cnt == 2 and cnt == 1),
//     and the counter reloads when cnt == 1
//   - the idle code covers startup (cnt = 10, 9, 8, lights off); at cnt == 8
//     it reloads 10 and enters s1_red
module traffic_light (
    input        rst_n,
    input        clk,
    input        pass_request,
    output [7:0] clock,
    output       red,
    output       yellow,
    output       green
);
    parameter idle      = 2'd0,
              s1_red    = 2'd1,
              s2_yellow = 2'd2,
              s3_green  = 2'd3;

    reg [5:0] cnt;
    reg [1:0] state;

    wire hi0     = ~|cnt[5:2];               // cnt in {1,2,3}
    wire cnt_is3 = hi0 & cnt[1] & cnt[0];
    wire lag     = hi0 & ~(cnt[1] & cnt[0]); // cnt in {1,2}
    wire end_ph  = hi0 & ~cnt[1];            // cnt == 1

    wire is_i = (state == idle);
    wire is_r = (state == s1_red);
    wire is_y = (state == s2_yellow);
    wire is_g = (state == s3_green);
    wire start_end = is_i & ~cnt[1] & ~cnt[0];  // cnt == 8 during startup

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= idle;
        else if (start_end)
            state <= s1_red;
        else if (cnt_is3)
            state <= is_r ? s3_green : (is_g ? s2_yellow : s1_red);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt <= 6'd10;
        else if (pass_request && is_g && (cnt > 6'd10))
            cnt <= 6'd10;
        else if (end_ph | start_end)
            cnt <= is_g ? 6'd60 : (is_y ? 6'd5 : 6'd10);
        else
            cnt <= cnt - 6'd1;
    end

    assign clock = {2'b00, cnt};

    // Light currently shown: previous phase colour while lagging.
    assign red    = lag ? is_g : is_r;
    assign green  = lag ? is_y : is_g;
    assign yellow = lag ? is_r : is_y;
endmodule
