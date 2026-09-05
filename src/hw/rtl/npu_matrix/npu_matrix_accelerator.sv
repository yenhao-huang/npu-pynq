`timescale 1ns/1ps

module npu_matrix_accelerator #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer MAX_K = 256,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 8
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF s_axi:s_axis:m_axis, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
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

    input  logic [7:0]                        s_axis_tdata,
    input  logic                              s_axis_tvalid,
    output logic                              s_axis_tready,
    input  logic                              s_axis_tlast,
    output logic [7:0]                        m_axis_tdata,
    output logic                              m_axis_tvalid,
    input  logic                              m_axis_tready,
    output logic                              m_axis_tlast,
    output logic                              irq
);
    logic start_pulse, soft_reset_pulse;
    logic [15:0] cfg_m, cfg_n, cfg_k;
    logic [31:0] cfg_a_stride, cfg_b_stride, cfg_c_stride;
    logic [31:0] cfg_timeout_cycles;
    logic [1:0] cfg_job_flags;
    logic signed [7:0] cfg_output_zero_point;
    logic status_busy, status_done, status_error;
    logic [7:0] error_code;
    logic [63:0] cycles;

    assign irq = status_done | status_error;

    npu_axi_lite_regs #(
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH)
    ) control_regs (
        .*,
        .status_busy(status_busy),
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
        .cfg_job_flags(cfg_job_flags),
        .cfg_output_zero_point(cfg_output_zero_point)
    );

    npu_matrix_controller #(
        .ROWS(ROWS),
        .COLUMNS(COLUMNS),
        .MAX_K(MAX_K)
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
        .cfg_job_flags(cfg_job_flags),
        .cfg_output_zero_point(cfg_output_zero_point),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .status_busy(status_busy),
        .status_done(status_done),
        .status_error(status_error),
        .error_code(error_code),
        .cycles(cycles)
    );
endmodule
