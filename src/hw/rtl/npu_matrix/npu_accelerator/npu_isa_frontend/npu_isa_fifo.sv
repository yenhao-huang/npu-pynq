`timescale 1ns/1ps

// Small synchronous FIFO with a combinational head, used for job and burst
// bookkeeping in the ISA front end. DEPTH must be a power of two.
module npu_isa_fifo #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 4
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             clear,
    input  logic             push,
    input  logic [WIDTH-1:0] push_data,
    output logic             full,
    input  logic             pop,
    output logic [WIDTH-1:0] head,
    output logic             empty
);
    localparam integer PTR_BITS = (DEPTH > 1) ? $clog2(DEPTH) : 1;

    logic [WIDTH-1:0] slots [0:DEPTH-1];
    logic [PTR_BITS-1:0] rd_ptr, wr_ptr;
    logic [PTR_BITS:0] count;

    assign full = (count == (PTR_BITS+1)'(DEPTH));
    assign empty = (count == '0);
    assign head = slots[rd_ptr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
            wr_ptr <= '0;
            count <= '0;
        end else if (clear) begin
            rd_ptr <= '0;
            wr_ptr <= '0;
            count <= '0;
        end else begin
            if (push && !full) begin
                slots[wr_ptr] <= push_data;
                wr_ptr <= wr_ptr + 1'b1;
            end
            if (pop && !empty)
                rd_ptr <= rd_ptr + 1'b1;
            case ({push && !full, pop && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: ;
            endcase
        end
    end
endmodule
