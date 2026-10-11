`timescale 1ns/1ps

// npu_vector_imul: integer lane multiplier for one 32-bit slice of a vector
// register (issue #136). Every lane keeps the low half of the product (two's
// complement wrap), so signed and unsigned products are identical and all
// partial products are unsigned.
//
// Four multipliers per slice, shared by the lane types (typ: 0 INT8, 1 INT16,
// 2 INT32; 3 is not used by the vector unit):
//   INT8 : m_i = a_i * b_i for the four byte lanes, y_i = m_i[7:0]
//   INT16: m0 = a_lo * b_lo, m1 = a_hi * b_hi, y = {m1[15:0], m0[15:0]}
//   INT32: m0 = a_lo * b_lo (full 32 bits), m1 = a_hi * b_lo, m2 = a_lo * b_hi
//          (low 16 bits each); a * b mod 2^32 = m0 + ((m1 + m2) << 16)
// m0..m2 are 16 x 16 and map to one DSP48E1 each; m3 (8 x 8, low 8 bits) is
// small enough for the synthesis tool to place in LUTs or a DSP.
//
// Latency 2, fully pipelined: X1 selects the operand halves and registers the
// products (the DSP M register); X2 combines them (one 3-input 16-bit add for
// INT32) into the registered y. Pure datapath, no reset.
module npu_vector_imul (
    input  logic        clk,
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [1:0]  typ,
    output logic [31:0] y
);
    localparam [1:0] T_INT8  = 2'd0;
    localparam [1:0] T_INT16 = 2'd1;

    logic [15:0] ma0, mb0, ma1, mb1, ma2, mb2;
    logic [7:0]  ma3, mb3;

    always_comb begin
        if (typ == T_INT8) begin
            ma0 = {8'd0, a[7:0]};   mb0 = {8'd0, b[7:0]};
            ma1 = {8'd0, a[15:8]};  mb1 = {8'd0, b[15:8]};
            ma2 = {8'd0, a[23:16]}; mb2 = {8'd0, b[23:16]};
        end else begin
            ma0 = a[15:0];  mb0 = b[15:0];
            ma1 = a[31:16]; mb1 = (typ == T_INT16) ? b[31:16] : b[15:0];
            ma2 = a[15:0];  mb2 = b[31:16];
        end
        ma3 = a[31:24];
        mb3 = b[31:24];
    end

    (* use_dsp = "yes" *) logic [31:0] p0;
    (* use_dsp = "yes" *) logic [15:0] p1;
    (* use_dsp = "yes" *) logic [15:0] p2;
    logic [7:0]  p3;
    logic [1:0]  typ2;

    always_ff @(posedge clk) begin
        p0 <= ma0 * mb0;
        p1 <= 16'(ma1 * mb1);
        p2 <= 16'(ma2 * mb2);
        p3 <= 8'(ma3 * mb3);
        typ2 <= typ;
    end

    always_ff @(posedge clk) begin
        case (typ2)
            T_INT8:  y <= {p3, p2[7:0], p1[7:0], p0[7:0]};
            T_INT16: y <= {p1, p0[15:0]};
            default: y <= {p0[31:16] + p1 + p2, p0[15:0]};
        endcase
    end
endmodule
