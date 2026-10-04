`timescale 1ns/1ps

// One operand bank: word writes for A, byte writes for B, and a clocked read.
// No reset on storage or read data, so Vivado can infer block RAM.
module npu_operand_buffer #(
    parameter integer DEPTH = 256,
    parameter integer WRITE_BYTES = 8,
    parameter integer ADDR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1,
    parameter integer LANE_BITS = (WRITE_BYTES > 1) ? $clog2(WRITE_BYTES) : 0,
    parameter integer WORD_ADDR_WIDTH =
        (ADDR_WIDTH > LANE_BITS) ? ADDR_WIDTH - LANE_BITS : 1
) (
    input  logic clk,
    input  logic write_enable,
    input  logic [WORD_ADDR_WIDTH-1:0] write_word_address,
    input  logic [WRITE_BYTES*8-1:0] write_data,
    input  logic read_enable,
    input  logic [ADDR_WIDTH-1:0] read_address,
    output logic [7:0] read_data
);
    (* ram_style = "block" *) logic [7:0] memory [0:DEPTH-1];

    initial begin
        if (DEPTH <= 0 || WRITE_BYTES <= 0 || WRITE_BYTES > DEPTH)
            $fatal(1, "npu_operand_buffer invalid depth or write width");
    end

    if (WRITE_BYTES == 1) begin : byte_writes
        always_ff @(posedge clk) begin
            if (write_enable)
                memory[write_word_address] <= write_data[7:0];
            if (read_enable)
                read_data <= memory[read_address];
        end
    end else begin : word_writes
        always_ff @(posedge clk) begin : write_word
            logic [LANE_BITS-1:0] lane;
            for (int index = 0; index < WRITE_BYTES; index++) begin
                lane = LANE_BITS'(index);
                if (write_enable)
                    memory[{write_word_address, lane}] <= write_data[index*8 +: 8];
            end
            if (read_enable)
                read_data <= memory[read_address];
        end
    end
endmodule
