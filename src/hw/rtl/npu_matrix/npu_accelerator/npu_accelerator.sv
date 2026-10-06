`timescale 1ns/1ps

module npu_accelerator #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256,
    parameter integer IN_BYTES = 8,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 8
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF s_axi:s_axis:m_axis:m_axi, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
    input  logic                              s_axi_aclk,
    /* verilator lint_off SYNCASYNCNET */
    input  logic                              s_axi_aresetn,
    /* verilator lint_on SYNCASYNCNET */
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  logic [2:0]                        s_axi_awprot,
    input  logic                              s_axi_awvalid,
    output logic                              s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  logic                              s_axi_wvalid,
    output logic                              s_axi_wready,
    output logic [1:0]                        s_axi_bresp,
    output logic                              s_axi_bvalid,
    input  logic                              s_axi_bready,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  logic [2:0]                        s_axi_arprot,
    input  logic                              s_axi_arvalid,
    output logic                              s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output logic [1:0]                        s_axi_rresp,
    output logic                              s_axi_rvalid,
    input  logic                              s_axi_rready,

    input  logic [IN_BYTES*8-1:0]             s_axis_tdata,
    input  logic [IN_BYTES-1:0]               s_axis_tkeep,
    input  logic                              s_axis_tvalid,
    output logic                              s_axis_tready,
    input  logic                              s_axis_tlast,
    output logic [31:0]                       m_axis_tdata,
    output logic                              m_axis_tvalid,
    input  logic                              m_axis_tready,
    output logic                              m_axis_tlast,

    // AXI4 master of the instruction-stream front end: instruction fetch and
    // tile loads/stores, 32-bit addresses, 64-bit data.
    output logic [31:0]                       m_axi_araddr,
    output logic [7:0]                        m_axi_arlen,
    output logic [2:0]                        m_axi_arsize,
    output logic [1:0]                        m_axi_arburst,
    output logic [3:0]                        m_axi_arcache,
    output logic [2:0]                        m_axi_arprot,
    output logic                              m_axi_arvalid,
    input  logic                              m_axi_arready,
    input  logic [63:0]                       m_axi_rdata,
    input  logic [1:0]                        m_axi_rresp,
    input  logic                              m_axi_rlast,
    input  logic                              m_axi_rvalid,
    output logic                              m_axi_rready,
    output logic [31:0]                       m_axi_awaddr,
    output logic [7:0]                        m_axi_awlen,
    output logic [2:0]                        m_axi_awsize,
    output logic [1:0]                        m_axi_awburst,
    output logic [3:0]                        m_axi_awcache,
    output logic [2:0]                        m_axi_awprot,
    output logic                              m_axi_awvalid,
    input  logic                              m_axi_awready,
    output logic [63:0]                       m_axi_wdata,
    output logic [7:0]                        m_axi_wstrb,
    output logic                              m_axi_wlast,
    output logic                              m_axi_wvalid,
    input  logic                              m_axi_wready,
    input  logic [1:0]                        m_axi_bresp,
    input  logic                              m_axi_bvalid,
    output logic                              m_axi_bready,
    output logic                              irq
);
    logic start_pulse, soft_reset_pulse;
    logic [15:0] cfg_m, cfg_n, cfg_k;
    logic [31:0] cfg_a_stride, cfg_b_stride, cfg_c_stride;
    logic [31:0] cfg_timeout_cycles;
    logic status_busy, status_accept, status_done, status_error;
    logic [7:0] error_code;
    logic [63:0] cycles;

    logic isa_running, isa_busy, isa_done, isa_error, isa_start_pulse;
    logic [7:0] isa_error_code;
    logic [31:0] isa_pc, isa_jobs, isa_cycles, isa_instructions;
    logic [31:0] isa_prog_addr, isa_prog_len, isa_data_base;

    // The controller is driven either by the registers and the AXI DMA
    // streams (ABI 1.0 path) or, while a program is in flight, by the front end.
    logic ctl_start;
    logic [15:0] ctl_m, ctl_n, ctl_k;
    logic [31:0] ctl_a_stride, ctl_b_stride, ctl_c_stride, ctl_timeout;
    logic fe_start;
    logic [15:0] fe_m, fe_n, fe_k;
    logic [IN_BYTES*8-1:0] core_s_tdata;
    logic [IN_BYTES-1:0] core_s_tkeep;
    logic core_s_tvalid, core_s_tready, core_s_tlast;
    logic [31:0] core_m_tdata;
    logic core_m_tvalid, core_m_tready, core_m_tlast;
    logic [63:0] fe_s_tdata;
    logic [7:0] fe_s_tkeep;
    logic fe_s_tvalid, fe_s_tlast, fe_m_tready;

    assign irq = status_done | status_error | isa_done | isa_error;

    // One assign per signal: the stream readies loop back through the
    // controller, and a shared always_comb would look circular to Verilator.
    assign ctl_start = isa_busy ? fe_start : start_pulse;
    assign ctl_m = isa_busy ? fe_m : cfg_m;
    assign ctl_n = isa_busy ? fe_n : cfg_n;
    assign ctl_k = isa_busy ? fe_k : cfg_k;
    assign ctl_a_stride = isa_busy ? {16'd0, fe_k} : cfg_a_stride;
    assign ctl_b_stride = isa_busy ? {16'd0, fe_n} : cfg_b_stride;
    assign ctl_c_stride = isa_busy ? {14'd0, fe_n, 2'b00} : cfg_c_stride;
    assign ctl_timeout = isa_busy ? 32'hffff_ffff : cfg_timeout_cycles;
    assign core_s_tdata = isa_busy ? fe_s_tdata[IN_BYTES*8-1:0] : s_axis_tdata;
    assign core_s_tkeep = isa_busy ? fe_s_tkeep[IN_BYTES-1:0] : s_axis_tkeep;
    assign core_s_tvalid = isa_busy ? fe_s_tvalid : s_axis_tvalid;
    assign core_s_tlast = isa_busy ? fe_s_tlast : s_axis_tlast;
    assign core_m_tready = isa_busy ? fe_m_tready : m_axis_tready;
    assign s_axis_tready = isa_busy ? 1'b0 : core_s_tready;
    assign m_axis_tvalid = isa_busy ? 1'b0 : core_m_tvalid;
    assign m_axis_tdata = core_m_tdata;
    assign m_axis_tlast = core_m_tlast;

    npu_isa_frontend #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS),
        .MAX_K(MAX_K)
    ) frontend (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .isa_start(isa_start_pulse),
        .prog_addr(isa_prog_addr),
        .prog_len(isa_prog_len),
        .data_base(isa_data_base),
        .isa_running(isa_running),
        .isa_busy(isa_busy),
        .isa_done(isa_done),
        .isa_error(isa_error),
        .isa_error_code(isa_error_code),
        .isa_pc(isa_pc),
        .isa_instructions(isa_instructions),
        .isa_jobs(isa_jobs),
        .isa_cycles(isa_cycles),
        .ctl_start(fe_start),
        .ctl_m(fe_m),
        .ctl_n(fe_n),
        .ctl_k(fe_k),
        .ctl_accept(status_accept),
        .ctl_error(status_error),
        .ctl_error_code(error_code),
        .s_axis_tdata(fe_s_tdata),
        .s_axis_tkeep(fe_s_tkeep),
        .s_axis_tvalid(fe_s_tvalid),
        .s_axis_tready(core_s_tready),
        .s_axis_tlast(fe_s_tlast),
        .m_axis_tdata(core_m_tdata),
        .m_axis_tvalid(core_m_tvalid),
        .m_axis_tready(fe_m_tready),
        .m_axis_tlast(core_m_tlast),
        .*
    );

    npu_axi_lite_regs #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
    ) control_regs (
        .*,
        .status_busy(status_busy),
        .status_accept(status_accept),
        .status_done(status_done),
        .status_error(status_error),
        .error_code(error_code),
        .cycles(cycles),
        .start_pulse(start_pulse),
        .soft_reset_pulse(soft_reset_pulse),
        .cfg_m(cfg_m),
        .cfg_n(cfg_n),
        .cfg_k(cfg_k),
        .cfg_a_stride(cfg_a_stride),
        .cfg_b_stride(cfg_b_stride),
        .cfg_c_stride(cfg_c_stride),
        .cfg_timeout_cycles(cfg_timeout_cycles)
    );

    npu_matrix_core #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS),
        .MAX_K(MAX_K),
        .IN_BYTES(IN_BYTES)
    ) controller (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .start_pulse(ctl_start),
        .soft_reset_pulse(soft_reset_pulse),
        .cfg_m(ctl_m),
        .cfg_n(ctl_n),
        .cfg_k(ctl_k),
        .cfg_a_stride(ctl_a_stride),
        .cfg_b_stride(ctl_b_stride),
        .cfg_c_stride(ctl_c_stride),
        .cfg_timeout_cycles(ctl_timeout),
        .s_axis_tdata(core_s_tdata),
        .s_axis_tkeep(core_s_tkeep),
        .s_axis_tvalid(core_s_tvalid),
        .s_axis_tready(core_s_tready),
        .s_axis_tlast(core_s_tlast),
        .m_axis_tdata(core_m_tdata),
        .m_axis_tvalid(core_m_tvalid),
        .m_axis_tready(core_m_tready),
        .m_axis_tlast(core_m_tlast),
        .status_busy(status_busy),
        .status_accept(status_accept),
        .status_done(status_done),
        .status_error(status_error),
        .error_code(error_code),
        .cycles(cycles)
    );
endmodule
