`timescale 1ns/1ps

module npu_axi_lite_regs #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 8
) (
    input  logic                              s_axi_aclk,
    input  logic                              s_axi_aresetn,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [2:0]                        s_axi_awprot,
    /* verilator lint_on UNUSEDSIGNAL */
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
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [2:0]                        s_axi_arprot,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                              s_axi_arvalid,
    output logic                              s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output logic [1:0]                        s_axi_rresp,
    output logic                              s_axi_rvalid,
    input  logic                              s_axi_rready,

    input  logic                              status_busy,
    input  logic                              status_done,
    input  logic                              status_error,
    input  logic [7:0]                        error_code,
    input  logic [63:0]                       cycles,
    output logic                              start_pulse,
    output logic                              soft_reset_pulse,
    output logic [15:0]                       cfg_m,
    output logic [15:0]                       cfg_n,
    output logic [15:0]                       cfg_k,
    output logic [31:0]                       cfg_a_stride,
    output logic [31:0]                       cfg_b_stride,
    output logic [31:0]                       cfg_c_stride,
    output logic [31:0]                       cfg_timeout_cycles,
    output logic [1:0]                        cfg_job_flags,
    output logic signed [7:0]                 cfg_output_zero_point
);
    localparam logic [31:0] ABI_MAGIC = 32'h3155504e;
    localparam logic [31:0] ABI_VERSION = 32'h00020000;
    localparam logic [31:0] ABI_CAPABILITIES = 32'h0000001f;

    localparam logic [7:0] REG_MAGIC = 8'h00;
    localparam logic [7:0] REG_VERSION = 8'h04;
    localparam logic [7:0] REG_CAPABILITIES = 8'h08;
    localparam logic [7:0] REG_CONTROL = 8'h0c;
    localparam logic [7:0] REG_STATUS = 8'h10;
    localparam logic [7:0] REG_ERROR = 8'h14;
    localparam logic [7:0] REG_M = 8'h18;
    localparam logic [7:0] REG_N = 8'h1c;
    localparam logic [7:0] REG_K = 8'h20;
    localparam logic [7:0] REG_A_STRIDE = 8'h24;
    localparam logic [7:0] REG_B_STRIDE = 8'h28;
    localparam logic [7:0] REG_C_STRIDE = 8'h2c;
    localparam logic [7:0] REG_TIMEOUT_CYCLES = 8'h30;
    localparam logic [7:0] REG_CYCLES_LO = 8'h34;
    localparam logic [7:0] REG_CYCLES_HI = 8'h38;
    localparam logic [7:0] REG_JOB_FLAGS = 8'h3c;
    localparam logic [7:0] REG_OUTPUT_ZERO_POINT = 8'h40;

    logic [C_S_AXI_ADDR_WIDTH-1:0] aw_hold_addr;
    logic                          aw_hold_valid;
    logic [C_S_AXI_DATA_WIDTH-1:0] w_hold_data;
    logic [(C_S_AXI_DATA_WIDTH/8)-1:0] w_hold_strb;
    logic                          w_hold_valid;

    function automatic [31:0] merge_wstrb;
        input [31:0] current_value;
        input [31:0] write_value;
        input [3:0] write_strobes;
        integer byte_index;
        begin
            merge_wstrb = current_value;
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1) begin
                if (write_strobes[byte_index])
                    merge_wstrb[byte_index*8 +: 8] = write_value[byte_index*8 +: 8];
            end
        end
    endfunction

    function automatic [15:0] merge_wstrb16;
        input [15:0] current_value;
        input [31:0] write_value;
        input [3:0] write_strobes;
        integer byte_index;
        begin
            merge_wstrb16 = current_value;
            for (byte_index = 0; byte_index < 2; byte_index = byte_index + 1) begin
                if (write_strobes[byte_index])
                    merge_wstrb16[byte_index*8 +: 8] = write_value[byte_index*8 +: 8];
            end
        end
    endfunction

    function automatic [31:0] read_word;
        input [7:0] address;
        begin
            case (address)
                REG_MAGIC:          read_word = ABI_MAGIC;
                REG_VERSION:        read_word = ABI_VERSION;
                REG_CAPABILITIES:   read_word = ABI_CAPABILITIES;
                REG_CONTROL:        read_word = 32'd0;
                REG_STATUS:         read_word = {29'd0, status_error, status_done, status_busy};
                REG_ERROR:          read_word = {24'd0, error_code};
                REG_M:              read_word = {16'd0, cfg_m};
                REG_N:              read_word = {16'd0, cfg_n};
                REG_K:              read_word = {16'd0, cfg_k};
                REG_A_STRIDE:       read_word = cfg_a_stride;
                REG_B_STRIDE:       read_word = cfg_b_stride;
                REG_C_STRIDE:       read_word = cfg_c_stride;
                REG_TIMEOUT_CYCLES: read_word = cfg_timeout_cycles;
                REG_CYCLES_LO:      read_word = cycles[31:0];
                REG_CYCLES_HI:      read_word = cycles[63:32];
                REG_JOB_FLAGS:      read_word = {30'd0, cfg_job_flags};
                REG_OUTPUT_ZERO_POINT:
                    read_word = {{24{cfg_output_zero_point[7]}}, cfg_output_zero_point};
                default:            read_word = 32'd0;
            endcase
        end
    endfunction

    initial begin
        if (C_S_AXI_DATA_WIDTH != 32)
            $fatal(1, "npu_axi_lite_regs ABI v2 requires 32-bit AXI data");
        if (C_S_AXI_ADDR_WIDTH < 8)
            $fatal(1, "npu_axi_lite_regs ABI v2 requires at least 8 address bits");
    end

    always_comb begin
        s_axi_awready = !aw_hold_valid && !s_axi_bvalid;
        s_axi_wready = !w_hold_valid && !s_axi_bvalid;
        s_axi_bresp = 2'b00;
        s_axi_arready = !s_axi_rvalid;
        s_axi_rresp = 2'b00;
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_hold_addr <= '0;
            aw_hold_valid <= 1'b0;
            w_hold_data <= '0;
            w_hold_strb <= '0;
            w_hold_valid <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_rdata <= '0;
            s_axi_rvalid <= 1'b0;
            start_pulse <= 1'b0;
            soft_reset_pulse <= 1'b0;
            cfg_m <= 16'd0;
            cfg_n <= 16'd0;
            cfg_k <= 16'd0;
            cfg_a_stride <= 32'd0;
            cfg_b_stride <= 32'd0;
            cfg_c_stride <= 32'd0;
            cfg_timeout_cycles <= 32'd0;
            cfg_job_flags <= 2'd0;
            cfg_output_zero_point <= 8'sd0;
        end else begin
            start_pulse <= 1'b0;
            soft_reset_pulse <= 1'b0;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_hold_addr <= s_axi_awaddr;
                aw_hold_valid <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_hold_data <= s_axi_wdata;
                w_hold_strb <= s_axi_wstrb;
                w_hold_valid <= 1'b1;
            end

            if (aw_hold_valid && w_hold_valid && !s_axi_bvalid) begin
                case (aw_hold_addr[7:0])
                    REG_CONTROL: begin
                        if (w_hold_strb[0]) begin
                            if (w_hold_data[1]) begin
                                soft_reset_pulse <= 1'b1;
                                cfg_m <= 16'd0;
                                cfg_n <= 16'd0;
                                cfg_k <= 16'd0;
                                cfg_a_stride <= 32'd0;
                                cfg_b_stride <= 32'd0;
                                cfg_c_stride <= 32'd0;
                                cfg_timeout_cycles <= 32'd0;
                                cfg_job_flags <= 2'd0;
                                cfg_output_zero_point <= 8'sd0;
                            end else if (w_hold_data[0]) begin
                                start_pulse <= 1'b1;
                            end
                        end
                    end
                    REG_M: begin
                        if (!status_busy)
                            cfg_m <= merge_wstrb16(cfg_m, w_hold_data, w_hold_strb);
                    end
                    REG_N: begin
                        if (!status_busy)
                            cfg_n <= merge_wstrb16(cfg_n, w_hold_data, w_hold_strb);
                    end
                    REG_K: begin
                        if (!status_busy)
                            cfg_k <= merge_wstrb16(cfg_k, w_hold_data, w_hold_strb);
                    end
                    REG_A_STRIDE: begin
                        if (!status_busy)
                            cfg_a_stride <= merge_wstrb(cfg_a_stride, w_hold_data, w_hold_strb);
                    end
                    REG_B_STRIDE: begin
                        if (!status_busy)
                            cfg_b_stride <= merge_wstrb(cfg_b_stride, w_hold_data, w_hold_strb);
                    end
                    REG_C_STRIDE: begin
                        if (!status_busy)
                            cfg_c_stride <= merge_wstrb(cfg_c_stride, w_hold_data, w_hold_strb);
                    end
                    REG_TIMEOUT_CYCLES: begin
                        if (!status_busy)
                            cfg_timeout_cycles <= merge_wstrb(
                                cfg_timeout_cycles, w_hold_data, w_hold_strb
                            );
                    end
                    REG_JOB_FLAGS: begin
                        if (!status_busy && w_hold_strb[0])
                            cfg_job_flags <= w_hold_data[1:0];
                    end
                    REG_OUTPUT_ZERO_POINT: begin
                        if (!status_busy && w_hold_strb[0])
                            cfg_output_zero_point <= w_hold_data[7:0];
                    end
                    default: begin
                    end
                endcase
                aw_hold_valid <= 1'b0;
                w_hold_valid <= 1'b0;
                s_axi_bvalid <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rdata <= read_word(s_axi_araddr[7:0]);
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
endmodule
