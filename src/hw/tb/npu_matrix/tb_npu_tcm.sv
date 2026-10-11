`timescale 1ns/1ps

// npu_tcm (1024 x 64) against a reference array. Checks gnt = req, rdata one
// cycle after every request (and held while req is low), byte strobes, and
// read-during-write: same port READ_FIRST (old word), cross-port write/read
// on one address returns the old word. Same-address write/write on both
// ports is undefined and never generated. Phases: initialisation through
// both ports, directed latency/hold/strobe cases, random traffic over the
// whole array, random traffic on 8 addresses (dense collisions), and a
// reset pulse that must not disturb the contents.
module tb_npu_tcm;
    localparam integer W = 64;
    localparam integer D = 1024;
    localparam integer AW = 10;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic a_req = 1'b0, a_we = 1'b0, b_req = 1'b0, b_we = 1'b0;
    logic [AW-1:0] a_addr = '0, b_addr = '0;
    logic [W-1:0] a_wdata = '0, b_wdata = '0;
    logic [7:0] a_wstrb = '0, b_wstrb = '0;
    logic a_gnt, b_gnt;
    logic [W-1:0] a_rdata, b_rdata;

    npu_tcm #(.WIDTH(W), .DEPTH(D)) dut (
        .clk(clk), .rst_n(rst_n),
        .a_req(a_req), .a_we(a_we), .a_addr(a_addr), .a_wdata(a_wdata),
        .a_wstrb(a_wstrb), .a_gnt(a_gnt), .a_rdata(a_rdata),
        .b_req(b_req), .b_we(b_we), .b_addr(b_addr), .b_wdata(b_wdata),
        .b_wstrb(b_wstrb), .b_gnt(b_gnt), .b_rdata(b_rdata)
    );

    logic [W-1:0] ref_mem [0:D-1];
    logic [W-1:0] exp_a, exp_b;
    integer seed = 136;
    integer n_rdw_same = 0, n_rdw_cross = 0, n_reads = 0, n_writes = 0;

    function automatic integer rnd(input integer n);
        rnd = ($random(seed) & 32'h7fff_ffff) % n;
    endfunction

    function automatic logic [W-1:0] merge(input logic [W-1:0] old,
                                           input logic [W-1:0] data,
                                           input logic [7:0] strb);
        merge = old;
        for (int i = 0; i < 8; i++)
            if (strb[i]) merge[i*8 +: 8] = data[i*8 +: 8];
    endfunction

    task automatic fail(input string msg);
        $display("MISMATCH tb_npu_tcm: %s", msg);
        $fatal(1);
    endtask

    // Apply the signals set up since the last negedge for one clock and
    // check the result against the reference. Called at a negedge.
    task automatic step;
        #1;
        if (a_gnt !== a_req || b_gnt !== b_req) fail("gnt != req");
        if (a_req && b_req && a_we && b_we && a_addr == b_addr)
            fail("testbench generated an undefined write/write collision");
        // Expected read data: the word before this cycle's writes.
        if (a_req) exp_a = ref_mem[a_addr];
        if (b_req) exp_b = ref_mem[b_addr];
        if (a_req && a_we && b_req && b_addr == a_addr) n_rdw_cross = n_rdw_cross + 1;
        if (b_req && b_we && a_req && a_addr == b_addr) n_rdw_cross = n_rdw_cross + 1;
        if (a_req && a_we) n_rdw_same = n_rdw_same + 1;
        if (b_req && b_we) n_rdw_same = n_rdw_same + 1;
        n_reads = n_reads + a_req + b_req;
        n_writes = n_writes + (a_req && a_we) + (b_req && b_we);
        if (a_req && a_we) ref_mem[a_addr] = merge(ref_mem[a_addr], a_wdata, a_wstrb);
        if (b_req && b_we) ref_mem[b_addr] = merge(ref_mem[b_addr], b_wdata, b_wstrb);
        @(posedge clk);
        #1;
        if (a_rdata !== exp_a) begin
            $display("MISMATCH tb_npu_tcm: port A rdata %h, expected %h", a_rdata, exp_a);
            $fatal(1);
        end
        if (b_rdata !== exp_b) begin
            $display("MISMATCH tb_npu_tcm: port B rdata %h, expected %h", b_rdata, exp_b);
            $fatal(1);
        end
        @(negedge clk);
    endtask

    task automatic idle;
        a_req = 1'b0; a_we = 1'b0; b_req = 1'b0; b_we = 1'b0;
    endtask

    task automatic random_cycle(input integer addr_span);
        a_req = rnd(100) < 80;
        b_req = rnd(100) < 80;
        a_we = rnd(2);
        b_we = rnd(2);
        a_addr = rnd(addr_span);
        b_addr = rnd(addr_span);
        a_wdata = {$random(seed), $random(seed)};
        b_wdata = {$random(seed), $random(seed)};
        a_wstrb = (rnd(4) == 0) ? 8'hFF : $random(seed);
        b_wstrb = (rnd(4) == 0) ? 8'hFF : $random(seed);
        if (a_req && b_req && a_we && b_we && a_addr == b_addr)
            b_we = 1'b0;   // avoid the undefined case
    endtask

    initial begin
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        // Initialise all words through both ports, full strobes.
        for (int i = 0; i < D / 2; i++) begin
            a_req = 1'b1; a_we = 1'b1; a_addr = 2 * i; a_wstrb = 8'hFF;
            a_wdata = {$random(seed), $random(seed)};
            b_req = 1'b1; b_we = 1'b1; b_addr = 2 * i + 1; b_wstrb = 8'hFF;
            b_wdata = {$random(seed), $random(seed)};
            // rdata of an initial write is the uninitialised word: skip it.
            #1;
            ref_mem[a_addr] = merge('0, a_wdata, a_wstrb);
            ref_mem[b_addr] = merge('0, b_wdata, b_wstrb);
            @(posedge clk);
            @(negedge clk);
        end
        idle();

        // Directed: read latency is exactly one cycle, rdata holds while idle.
        a_req = 1'b1; a_addr = 10'd5;
        b_req = 1'b1; b_addr = 10'd6;
        step();
        idle();
        exp_a = ref_mem[5]; exp_b = ref_mem[6];
        repeat (3) step();                                  // held
        // Byte strobes: write one byte, then check via a read.
        a_req = 1'b1; a_we = 1'b1; a_addr = 10'd5; a_wstrb = 8'b0000_0100;
        a_wdata = 64'h1111_2222_33CC_4444;
        step();                                             // read-first: old word
        a_we = 1'b0;
        step();                                             // new word, byte 2 only
        if (a_rdata[23:16] !== 8'hCC) fail("byte strobe");
        // Zero strobe writes nothing.
        a_we = 1'b1; a_wstrb = 8'h00; a_wdata = '1;
        step();
        a_we = 1'b0;
        step();
        // Cross-port write/read on one address: reader sees the old word.
        a_req = 1'b1; a_we = 1'b1; a_addr = 10'd9; a_wstrb = 8'hFF; a_wdata = 64'hDEAD_BEEF_0123_4567;
        b_req = 1'b1; b_we = 1'b0; b_addr = 10'd9;
        step();
        idle();
        b_req = 1'b1; b_addr = 10'd9;
        step();
        if (b_rdata !== 64'hDEAD_BEEF_0123_4567) fail("cross-port write then read");
        // Both ports read the same word.
        a_req = 1'b1; a_addr = 10'd9; b_req = 1'b1; b_addr = 10'd9;
        step();
        idle();
        // Back-to-back reads on consecutive addresses (one per cycle).
        for (int i = 0; i < 16; i++) begin
            a_req = 1'b1; a_addr = 10'd100 + i;
            b_req = 1'b1; b_addr = 10'd1023 - i;
            step();
        end
        idle();
        step();

        // Random traffic over the whole array.
        for (int i = 0; i < 6000; i++) begin
            random_cycle(D);
            step();
        end
        // Dense collisions on 8 addresses.
        for (int i = 0; i < 4000; i++) begin
            random_cycle(8);
            step();
        end
        idle();

        // A reset pulse leaves the contents intact.
        rst_n = 1'b0;
        step();
        rst_n = 1'b1;
        step();
        for (int i = 0; i < D; i++) begin
            a_req = 1'b1; a_addr = i;
            b_req = (i % 3) == 0; b_addr = D - 1 - i;
            step();
        end
        idle();
        step();

        if (n_rdw_cross < 100) fail("too few cross-port collisions exercised");
        $display("PASS tb_npu_tcm reads=%0d writes=%0d same_port_rdw=%0d cross_port_rdw=%0d",
                 n_reads, n_writes, n_rdw_same, n_rdw_cross);
        $finish;
    end
endmodule
