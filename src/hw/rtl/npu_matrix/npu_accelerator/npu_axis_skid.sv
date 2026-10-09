`timescale 1ns/1ps

// Two-entry AXI-Stream register slice: full throughput, and every output
// (m_valid, m_data, s_ready) comes straight from a flip-flop, so neither the
// data nor the ready path combines across it.
module npu_axis_skid #(
    parameter integer WIDTH = 8
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             s_valid,
    output logic             s_ready,
    input  logic [WIDTH-1:0] s_data,
    output logic             m_valid,
    input  logic             m_ready,
    output logic [WIDTH-1:0] m_data
);
    logic             skid_valid;
    logic [WIDTH-1:0] skid_data;

    assign s_ready = !skid_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_data <= '0;
            skid_valid <= 1'b0;
            skid_data <= '0;
        end else if (m_ready || !m_valid) begin
            // The output register is free this cycle.
            if (skid_valid) begin
                m_valid <= 1'b1;
                m_data <= skid_data;
                skid_valid <= 1'b0;
            end else begin
                m_valid <= s_valid;
                m_data <= s_data;
            end
        end else if (s_valid && s_ready) begin
            // Output stalled: park the accepted beat.
            skid_valid <= 1'b1;
            skid_data <= s_data;
        end
    end
endmodule
