`timescale 1ns/1ps

// Scalar core of the NPU: runs the program and drives the matrix core. This
// module is a thin wrapper; SCALAR_IMPL selects the implementation, and every
// implementation has exactly this port list:
//
//   "CUSTOM"  custom/custom_scalar_core.sv  the 64-bit NPU instruction
//                                           sequencer (default)
//   "CVA6"    cva6/scalar_core_cva6.sv      a CVA6 RISC-V core that drives the
//                                           matrix core through MMIO registers
//
// The CVA6 branch is compiled only when NPU_SCALAR_CVA6 is defined. cva6.f
// defines it; the default flows (src/test/Makefile lint and sim,
// build_overlay.tcl) do not read the CVA6 sources and so do not define it. Without the guard, every tool would have to resolve
// scalar_core_cva6 even in the unselected branch, which needs the whole CVA6
// upstream tree. Selecting "CVA6" without the define stops elaboration.
//
// Both implementations keep the property the matrix core depends on: ctl_start
// and the job shape come straight from flip-flops, because the 100 MHz margin
// of the 16 x 16 array leaves no room for logic in front of its controller.
module scalar_core #(
    parameter integer ROWS = 16,
    parameter integer COLUMNS = 16,
    parameter integer MAX_K = 256,
    // Untyped on purpose: iverilog 13 cannot use a `parameter string` in a
    // parameter override or another parameter's value ("Unable to bind").
    parameter         SCALAR_IMPL = "CUSTOM"
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        isa_start,
    input  logic [31:0] prog_addr,
    input  logic [31:0] prog_len,
    input  logic [31:0] data_base,
    output logic        isa_running,
    output logic        isa_busy,
    output logic        isa_done,
    output logic        isa_error,
    output logic [7:0]  isa_error_code,
    output logic [31:0] isa_pc,
    output logic [31:0] isa_instructions,
    output logic [31:0] isa_jobs,
    output logic [31:0] isa_cycles,

    output logic        ctl_start,
    output logic [15:0] ctl_m,
    output logic [15:0] ctl_n,
    output logic [15:0] ctl_k,
    input  logic        ctl_accept,
    input  logic        ctl_error,
    input  logic [7:0]  ctl_error_code,

    output logic [63:0] s_axis_tdata,
    output logic [7:0]  s_axis_tkeep,
    output logic        s_axis_tvalid,
    input  logic        s_axis_tready,
    output logic        s_axis_tlast,
    input  logic [31:0] m_axis_tdata,
    input  logic        m_axis_tvalid,
    output logic        m_axis_tready,
    input  logic        m_axis_tlast,

    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic [2:0]  m_axi_arsize,
    output logic [1:0]  m_axi_arburst,
    output logic [3:0]  m_axi_arcache,
    output logic [2:0]  m_axi_arprot,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic [2:0]  m_axi_awsize,
    output logic [1:0]  m_axi_awburst,
    output logic [3:0]  m_axi_awcache,
    output logic [2:0]  m_axi_awprot,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready
);
    generate
        if (SCALAR_IMPL == "CVA6") begin : g_cva6
`ifdef NPU_SCALAR_CVA6
            scalar_core_cva6 #(
                .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K)
            ) u_core (.*);
`else
            $fatal(1, "scalar_core: SCALAR_IMPL=\"CVA6\" needs the CVA6 sources (cva6/cva6.f, NPU_SCALAR_CVA6)");
`endif
        end else if (SCALAR_IMPL == "CUSTOM") begin : g_custom
            custom_scalar_core #(
                .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K)
            ) u_core (.*);
        end else begin : g_unknown
            $fatal(1, "scalar_core: unknown SCALAR_IMPL \"%s\"", SCALAR_IMPL);
        end
    endgenerate
endmodule
