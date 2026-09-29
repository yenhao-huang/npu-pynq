// A reference counter and a deliberately broken copy of it, with a testbench
// that runs both side by side. The bug is narrow on purpose: `broken` skips
// one increment at count 7, so the two agree for seven cycles and then
// diverge by one and stay wrong.
//
// Finding that by eye means scrolling a waveform. `first_mismatch` reports it
// as one line, which is the behaviour this fixture exists to prove.
module counter_ref (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       enable,
    output logic [7:0] count
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      count <= 8'd0;
        else if (enable) count <= count + 8'd1;
    end
endmodule

module counter_broken (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       enable,
    output logic [7:0] count
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)              count <= 8'd0;
        else if (enable) begin
            if (count != 8'd7)   count <= count + 8'd1;  // BUG: stalls once at 7
        end
    end
endmodule
