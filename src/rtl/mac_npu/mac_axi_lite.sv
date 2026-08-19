`timescale 1ns/1ps

module mac_axi_lite #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 5
) (
    input  logic                              s_axi_aclk,
    input  logic                              s_axi_aresetn,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]      s_axi_awaddr,
    input  logic                              s_axi_awvalid,
    output logic                              s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0]      s_axi_wdata,
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0]  s_axi_wstrb,
    input  logic                              s_axi_wvalid,
    output logic                              s_axi_wready,
    output logic [1:0]                        s_axi_bresp,
    output logic                              s_axi_bvalid,
    input  logic                              s_axi_bready,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]      s_axi_araddr,
    input  logic                              s_axi_arvalid,
    output logic                              s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0]      s_axi_rdata,
    output logic [1:0]                        s_axi_rresp,
    output logic                              s_axi_rvalid,
    input  logic                              s_axi_rready
);

    localparam logic [C_S_AXI_ADDR_WIDTH-1:0] ADDR_CONTROL = 5'h00;
    localparam logic [C_S_AXI_ADDR_WIDTH-1:0] ADDR_A       = 5'h04;
    localparam logic [C_S_AXI_ADDR_WIDTH-1:0] ADDR_B       = 5'h08;
    localparam logic [C_S_AXI_ADDR_WIDTH-1:0] ADDR_STATUS  = 5'h0c;
    localparam logic [C_S_AXI_ADDR_WIDTH-1:0] ADDR_RESULT  = 5'h10;

    logic [C_S_AXI_ADDR_WIDTH-1:0] awaddr_hold;
    logic                          awaddr_pending;
    logic [C_S_AXI_DATA_WIDTH-1:0] wdata_hold;
    logic [(C_S_AXI_DATA_WIDTH/8)-1:0] wstrb_hold;
    logic                          wdata_pending;

    logic signed [7:0] operand_a;
    logic signed [7:0] operand_b;
    logic              clear_pulse;
    logic              start_pulse;
    logic              done_sticky;
    logic signed [31:0] accumulator;
    logic              mac_result_valid;
    integer             byte_index;

    wire write_commit = awaddr_pending && wdata_pending && !s_axi_bvalid;

    assign s_axi_awready = !awaddr_pending && !s_axi_bvalid;
    assign s_axi_wready  = !wdata_pending && !s_axi_bvalid;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    mac_unit mac_core (
        .clk          (s_axi_aclk),
        .rst_n        (s_axi_aresetn),
        .clear        (clear_pulse),
        .valid        (start_pulse),
        .a            (operand_a),
        .b            (operand_b),
        .result       (accumulator),
        .result_valid (mac_result_valid)
    );

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            awaddr_hold   <= '0;
            awaddr_pending <= 1'b0;
            wdata_hold    <= '0;
            wstrb_hold    <= '0;
            wdata_pending <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            operand_a     <= '0;
            operand_b     <= '0;
            clear_pulse   <= 1'b0;
            start_pulse   <= 1'b0;
            done_sticky   <= 1'b0;
        end else begin
            clear_pulse <= 1'b0;
            start_pulse <= 1'b0;

            if (s_axi_awready && s_axi_awvalid) begin
                awaddr_hold    <= s_axi_awaddr;
                awaddr_pending <= 1'b1;
            end
            if (s_axi_wready && s_axi_wvalid) begin
                wdata_hold    <= s_axi_wdata;
                wstrb_hold    <= s_axi_wstrb;
                wdata_pending <= 1'b1;
            end

            if (write_commit) begin
                case (awaddr_hold)
                    ADDR_CONTROL: begin
                        if (wstrb_hold[0]) begin
                            clear_pulse <= wdata_hold[0];
                            start_pulse <= wdata_hold[1];
                            if (wdata_hold[0] || wdata_hold[1]) begin
                                done_sticky <= 1'b0;
                            end
                        end
                    end
                    ADDR_A: begin
                        if (wstrb_hold[0]) operand_a <= wdata_hold[7:0];
                    end
                    ADDR_B: begin
                        if (wstrb_hold[0]) operand_b <= wdata_hold[7:0];
                    end
                    default: begin end
                endcase
                awaddr_pending <= 1'b0;
                wdata_pending  <= 1'b0;
                s_axi_bvalid   <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (mac_result_valid) begin
                done_sticky <= 1'b1;
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= '0;
        end else begin
            if (s_axi_arready && s_axi_arvalid) begin
                case (s_axi_araddr)
                    ADDR_CONTROL: s_axi_rdata <= '0;
                    ADDR_A:       s_axi_rdata <= {{24{operand_a[7]}}, operand_a};
                    ADDR_B:       s_axi_rdata <= {{24{operand_b[7]}}, operand_b};
                    ADDR_STATUS:  s_axi_rdata <= {{31{1'b0}}, done_sticky};
                    ADDR_RESULT:  s_axi_rdata <= accumulator;
                    default:      s_axi_rdata <= '0;
                endcase
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
