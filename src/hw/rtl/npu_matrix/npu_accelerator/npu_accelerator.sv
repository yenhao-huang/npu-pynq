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
    // Every input of the controller still comes from a flip-flop: the front end
    // loads START and the job shape into the register file's own flops, the
    // owner flag is a register, and both streams cross a two-entry skid buffer.
    // The 100 MHz margin of the 16 x 16 array leaves no room for muxes there.
    logic own_isa;
    logic fe_start;
    logic [15:0] fe_m, fe_n, fe_k;
    // Stream into the controller: mux -> skid -> core.
    logic [IN_BYTES*8-1:0] in_tdata, core_s_tdata;
    logic [IN_BYTES-1:0] in_tkeep, core_s_tkeep;
    logic in_tvalid, in_tready, in_tlast;
    logic core_s_tvalid, core_s_tready, core_s_tlast;
    // Stream out of the controller: core -> skid -> mux.
    logic [31:0] core_m_tdata, out_tdata;
    logic core_m_tvalid, core_m_tready, core_m_tlast;
    logic out_tvalid, out_tready, out_tlast;
    logic [63:0] fe_s_tdata;
    logic [7:0] fe_s_tkeep;
    logic fe_s_tvalid, fe_s_tlast, fe_m_tready;

    assign irq = status_done | status_error | isa_done | isa_error;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn)
            own_isa <= 1'b0;
        else
            own_isa <= isa_busy;
    end

    assign in_tdata = own_isa ? fe_s_tdata[IN_BYTES*8-1:0] : s_axis_tdata;
    assign in_tkeep = own_isa ? fe_s_tkeep[IN_BYTES-1:0] : s_axis_tkeep;
    assign in_tvalid = own_isa ? fe_s_tvalid : s_axis_tvalid;
    assign in_tlast = own_isa ? fe_s_tlast : s_axis_tlast;
    assign s_axis_tready = own_isa ? 1'b0 : in_tready;

    npu_axis_skid #(.WIDTH(IN_BYTES * 9 + 1)) input_slice (
        .clk(s_axi_aclk), .rst_n(s_axi_aresetn),
        .s_valid(in_tvalid), .s_ready(in_tready), .s_data({in_tlast, in_tkeep, in_tdata}),
        .m_valid(core_s_tvalid), .m_ready(core_s_tready),
        .m_data({core_s_tlast, core_s_tkeep, core_s_tdata})
    );

    npu_axis_skid #(.WIDTH(33)) output_slice (
        .clk(s_axi_aclk), .rst_n(s_axi_aresetn),
        .s_valid(core_m_tvalid), .s_ready(core_m_tready), .s_data({core_m_tlast, core_m_tdata}),
        .m_valid(out_tvalid), .m_ready(out_tready), .m_data({out_tlast, out_tdata})
    );

    assign out_tready = own_isa ? fe_m_tready : m_axis_tready;
    assign m_axis_tvalid = own_isa ? 1'b0 : out_tvalid;
    assign m_axis_tdata = out_tdata;
    assign m_axis_tlast = out_tlast;

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
        .s_axis_tready(in_tready && own_isa),
        .s_axis_tlast(fe_s_tlast),
        .m_axis_tdata(out_tdata),
        .m_axis_tvalid(out_tvalid && own_isa),
        .m_axis_tready(fe_m_tready),
        .m_axis_tlast(out_tlast),
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
        .cfg_timeout_cycles(cfg_timeout_cycles),
        .ext_own(own_isa),
        .ext_start(fe_start),
        .ext_m(fe_m),
        .ext_n(fe_n),
        .ext_k(fe_k)
    );

    npu_matrix_core #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS),
        .MAX_K(MAX_K),
        .IN_BYTES(IN_BYTES)
    ) controller (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .start_pulse(start_pulse),
        .soft_reset_pulse(soft_reset_pulse),
        .cfg_m(cfg_m),
        .cfg_n(cfg_n),
        .cfg_k(cfg_k),
        .cfg_a_stride(cfg_a_stride),
        .cfg_b_stride(cfg_b_stride),
        .cfg_c_stride(cfg_c_stride),
        .cfg_timeout_cycles(cfg_timeout_cycles),
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
