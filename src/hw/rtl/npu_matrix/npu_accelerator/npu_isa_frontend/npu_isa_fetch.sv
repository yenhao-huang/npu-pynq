`timescale 1ns/1ps

// IF stage: fetches PROG_LEN 64-bit instruction words starting at PROG_ADDR
// into a prefetch queue. One read burst of up to 16 beats is outstanding at a
// time, aligned so it never crosses a 128-byte (and so never a 4 KiB) line,
// and only when the queue has room for the whole burst, so R never stalls.
module npu_isa_fetch #(
    parameter integer DEPTH = 32
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        stop,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [31:0] prog_addr,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [31:0] prog_len,

    output logic        ar_valid,
    input  logic        ar_ready,
    output logic [31:0] ar_addr,
    output logic [7:0]  ar_len,
    input  logic        r_valid,
    output logic        r_ready,
    input  logic [63:0] r_data,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]  r_resp,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic        r_last,

    output logic        instr_valid,
    output logic [63:0] instr_data,
    input  logic        instr_ready,
    output logic        fetch_error
);
    localparam integer PTR_BITS = $clog2(DEPTH);

    logic [63:0] queue [0:DEPTH-1];
    logic [PTR_BITS-1:0] head, tail;
    logic [PTR_BITS:0] count;
    // Words reserved by the outstanding burst but not yet returned.
    logic [4:0] in_flight;
    logic [31:0] fetch_addr;
    logic [31:0] words_left;
    logic burst_open;
    // A burst still open when a new program starts belongs to the old one.
    logic discard;
    logic [4:0] burst_beats;
    logic push, pop;

    always_comb begin
        // Beats to the end of the 128-byte line, capped by the program end.
        burst_beats = 5'd16 - {1'b0, fetch_addr[6:3]};
        if ({27'd0, burst_beats} > words_left)
            burst_beats = words_left[4:0];
        ar_addr = fetch_addr;
        ar_len = {3'd0, burst_beats - 5'd1};
        ar_valid = !burst_open && !stop && (words_left != 0) &&
            ((count + {{(PTR_BITS-4){1'b0}}, in_flight}) <= (PTR_BITS+1)'(DEPTH - 16));
        r_ready = burst_open;
        push = r_valid && r_ready && !discard;
        instr_valid = (count != 0);
        instr_data = queue[head];
        pop = instr_valid && instr_ready;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= '0;
            tail <= '0;
            count <= '0;
            in_flight <= '0;
            fetch_addr <= 32'd0;
            words_left <= 32'd0;
            burst_open <= 1'b0;
            discard <= 1'b0;
            fetch_error <= 1'b0;
        end else if (start) begin
            discard <= burst_open && !(r_valid && r_last);
            if (r_valid && r_ready && r_last)
                burst_open <= 1'b0;
            head <= '0;
            tail <= '0;
            count <= '0;
            in_flight <= '0;
            fetch_addr <= {prog_addr[31:3], 3'b000};
            words_left <= prog_len;
            fetch_error <= 1'b0;
        end else begin
            if (ar_valid && ar_ready) begin
                burst_open <= 1'b1;
                in_flight <= burst_beats;
                fetch_addr <= fetch_addr + {24'd0, burst_beats, 3'b000};
                words_left <= words_left - {27'd0, burst_beats};
            end
            if (r_valid && r_ready && discard && r_last) begin
                discard <= 1'b0;
                burst_open <= 1'b0;
            end
            if (push) begin
                queue[tail] <= r_data;
                tail <= tail + 1'b1;
                in_flight <= in_flight - 5'd1;
                if (r_resp[1])
                    fetch_error <= 1'b1;
                if (r_last)
                    burst_open <= 1'b0;
            end
            case ({push, pop})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: ;
            endcase
            if (pop)
                head <= head + 1'b1;
            if (stop && !burst_open)
                words_left <= 32'd0;
        end
    end
endmodule
