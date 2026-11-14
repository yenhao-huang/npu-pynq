`timescale 1ns/1ps

`include "axi/typedef.svh"

// CVA6 implementation of the scalar core (SCALAR_IMPL = "CVA6", selected by
// the scalar_core wrapper; same port list). A CVA6 RISC-V core runs firmware
// from DDR and drives the matrix core through the MMIO job registers of
// cva6_npu_mmio; cva6_dma streams each job's operands and result.
//
//                 +------------+      +-------------+  0 DDR
//   cva6  ------> | stop gate  | ---> | axi_demux   | ------------+
//   (4-bit IDs)   +------------+      | (address)   |  1 MMIO     |
//                                     |             | ---> cva6_npu_mmio --> ctl_*, cva6_dma
//                                     |             |  2 unmapped |
//                                     +-------------+ ---> axi_err_slv (DECERR)
//                                                                 v
//   cva6_dma (ID 0) ---------------------------------------> axi_mux (5-bit IDs)
//                                                                 v
//                                                   axi_id_serialize (one ID)
//                                                                 v
//                                                   m_axi_* (32-bit address, no ID)
//
// The pulp axi library (axi_demux, axi_mux, axi_id_serialize, axi_err_slv)
// comes from the vendored copy inside the CVA6 submodule. The mux sits before
// the ID serializer: serializing last leaves the master port with a constant
// ID, so the ID-less m_axi needs no response-ID bookkeeping of its own (the
// serializer keeps the slave-side IDs and the in-order slave returns responses
// in request order).
//
// Run control:
//   IDLE   CVA6 held in reset (rst_ni low). An isa_start while not busy loads
//          boot_addr_i = prog_addr and releases the reset.
//   RUN    firmware runs. A DONE write or a fault latched by cva6_npu_mmio
//          ends the program.
//   STOP   new CVA6 AR / AW requests are held back; an AR / AW already
//          presented completes, and every outstanding CVA6 transaction is
//          allowed to finish so no response arrives after the reset.
//   DRAIN  CVA6 back in reset (its caches are clean again for the next run);
//          wait until every issued job has been written back by the DMA, then
//          isa_done (code 0) or isa_error with the code.
// The D$ is write-through, so DDR is current when the firmware writes DONE.
//
// ctl_start and ctl_m/n/k come straight from flip-flops in cva6_npu_mmio.
//
// Needs the CVA6 packages: read only through cva6.f, which also defines
// NPU_SCALAR_CVA6 so the wrapper instantiates this module.
module scalar_core_cva6 #(
    parameter integer ROWS = 16,
    parameter integer COLUMNS = 16,
    parameter integer MAX_K = 256
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
    localparam config_pkg::cva6_cfg_t CVA6Cfg =
        build_config_pkg::build_config(cva6_npu_config_pkg::cva6_cfg);

    localparam int unsigned AddrWidth = CVA6Cfg.AxiAddrWidth;
    localparam int unsigned DataWidth = CVA6Cfg.AxiDataWidth;
    localparam int unsigned IdWidth = CVA6Cfg.AxiIdWidth;
    localparam int unsigned UserWidth = CVA6Cfg.AxiUserWidth;
    localparam int unsigned MuxIdWidth = IdWidth + 1;  // + one bit for 2 mux ports
    localparam int unsigned MaxTxns = 8;

    typedef logic [AddrWidth-1:0]   addr_t;
    typedef logic [DataWidth-1:0]   data_t;
    typedef logic [DataWidth/8-1:0] strb_t;
    typedef logic [UserWidth-1:0]   user_t;
    typedef logic [IdWidth-1:0]     id_t;
    typedef logic [MuxIdWidth-1:0]  mux_id_t;
    typedef logic [0:0]             ser_id_t;

    // cpu_*: CVA6 side and the slave ports of the mux; the field order matches
    // the default noc_req_t / noc_resp_t of cva6.sv.
    `AXI_TYPEDEF_ALL(cpu, addr_t, id_t, data_t, strb_t, user_t)
    `AXI_TYPEDEF_ALL(mux, addr_t, mux_id_t, data_t, strb_t, user_t)
    `AXI_TYPEDEF_ALL(ser, addr_t, ser_id_t, data_t, strb_t, user_t)

    localparam logic [1:0] PORT_DDR = 2'd0;
    localparam logic [1:0] PORT_MMIO = 2'd1;
    localparam logic [1:0] PORT_ERR = 2'd2;

    // ------------------------------------------------------------ run control
    typedef enum logic [1:0] {IDLE, RUN, STOP, DRAIN} state_t;
    state_t      state;
    logic        start;
    logic        cpu_run;          // CVA6 out of reset
    logic        cpu_rst_n;
    logic [31:0] boot_addr;
    logic        end_error;
    logic [7:0]  end_code;
    logic        quiesced;

    logic        mmio_done, mmio_error, mmio_drained;
    /* verilator lint_off UNUSEDSIGNAL */
    logic        mmio_busy;     // STATUS[0]; the top uses mmio_drained
    /* verilator lint_on UNUSEDSIGNAL */
    logic [7:0]  mmio_done_code, mmio_error_code;
    logic [31:0] mmio_jobs_issued, mmio_jobs_done;

    always_comb begin
        isa_busy = (state != IDLE) || !mmio_drained;
        start = isa_start && !isa_busy;
        isa_running = (state == RUN);
        isa_jobs = mmio_jobs_done;
        // Open decision (plan 2026-1112-003 section 9.2): on CVA6,
        // isa_instructions counts NPU jobs issued (DOORBELLs accepted), not
        // retired CPU instructions; isa_pc reports the boot address, as CVA6
        // exposes no PC outside its RVFI trace.
        isa_instructions = mmio_jobs_issued;
        isa_pc = boot_addr;
    end

    // The reset comes from a flip-flop on the same clock; asserted
    // asynchronously with rst_n.
    assign cpu_rst_n = rst_n && cpu_run;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cpu_run <= 1'b0;
            boot_addr <= 32'd0;
            end_error <= 1'b0;
            end_code <= 8'd0;
            isa_done <= 1'b0;
            isa_error <= 1'b0;
            isa_error_code <= 8'd0;
            isa_cycles <= 32'd0;
        end else begin
            if (state != IDLE)
                isa_cycles <= isa_cycles + 32'd1;
            case (state)
                IDLE: if (start) begin
                    state <= RUN;
                    cpu_run <= 1'b1;
                    boot_addr <= prog_addr;
                    end_error <= 1'b0;
                    end_code <= 8'd0;
                    isa_done <= 1'b0;
                    isa_error <= 1'b0;
                    isa_error_code <= 8'd0;
                    isa_cycles <= 32'd0;
                end
                RUN: begin
                    // A latched fault takes precedence over the firmware's code.
                    if (mmio_error) begin
                        state <= STOP;
                        end_error <= 1'b1;
                        end_code <= mmio_error_code;
                        isa_error <= 1'b1;
                        isa_error_code <= mmio_error_code;
                    end else if (mmio_done) begin
                        state <= STOP;
                        end_error <= (mmio_done_code != 8'd0);
                        end_code <= mmio_done_code;
                    end
                end
                STOP: if (quiesced) begin
                    state <= DRAIN;
                    cpu_run <= 1'b0;
                end
                DRAIN: if (mmio_drained) begin
                    state <= IDLE;
                    // A fault while draining replaces a clean DONE.
                    if (mmio_error && !end_error) begin
                        isa_error <= 1'b1;
                        isa_error_code <= mmio_error_code;
                    end else if (end_error) begin
                        isa_error <= 1'b1;
                        isa_error_code <= end_code;
                    end else begin
                        isa_done <= 1'b1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------ CVA6
    cpu_req_t  cpu_req, gate_req;
    cpu_resp_t cpu_resp, gate_resp;

    // The RVFI trace and the CV-X-IF request are not used (CvxifEn = 0).
    /* verilator lint_off PINCONNECTEMPTY */
    cva6 #(
        .CVA6Cfg       (CVA6Cfg),
        .axi_ar_chan_t (cpu_ar_chan_t),
        .axi_aw_chan_t (cpu_aw_chan_t),
        .axi_w_chan_t  (cpu_w_chan_t),
        .b_chan_t      (cpu_b_chan_t),
        .r_chan_t      (cpu_r_chan_t),
        .noc_req_t     (cpu_req_t),
        .noc_resp_t    (cpu_resp_t)
    ) u_cva6 (
        .clk_i         (clk),
        .rst_ni        (cpu_rst_n),
        .boot_addr_i   (boot_addr),
        .hart_id_i     ('0),
        .irq_i         (2'b00),
        .ipi_i         (1'b0),
        .time_irq_i    (1'b0),
        .debug_req_i   (1'b0),
        .rvfi_probes_o (),
        .cvxif_req_o   (),
        .cvxif_resp_i  ('0),
        .noc_req_o     (cpu_req),
        .noc_resp_i    (cpu_resp)
    );
    /* verilator lint_on PINCONNECTEMPTY */

    // ------------------------------------------------------------ stop gate
    // In STOP, an AR / AW already presented (held) completes; no new one is
    // let through. quiesced once nothing is held and every CVA6 read and write
    // has had its last response. While CVA6 is in reset its requests are
    // masked as well.
    logic       ar_held, aw_held, ar_allow, aw_allow;
    logic [7:0] rd_outstanding, wr_outstanding;

    always_comb begin
        ar_allow = cpu_run && ((state == RUN) || ar_held);
        aw_allow = cpu_run && ((state == RUN) || aw_held);

        gate_req = cpu_req;
        gate_req.ar_valid = cpu_req.ar_valid && ar_allow;
        gate_req.aw_valid = cpu_req.aw_valid && aw_allow;
        gate_req.w_valid = cpu_req.w_valid && cpu_run;

        cpu_resp = gate_resp;
        cpu_resp.ar_ready = gate_resp.ar_ready && ar_allow;
        cpu_resp.aw_ready = gate_resp.aw_ready && aw_allow;
        cpu_resp.w_ready = gate_resp.w_ready && cpu_run;

        quiesced = !ar_held && !aw_held && (rd_outstanding == 8'd0) &&
            (wr_outstanding == 8'd0);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_held <= 1'b0;
            aw_held <= 1'b0;
            rd_outstanding <= 8'd0;
            wr_outstanding <= 8'd0;
        end else begin
            ar_held <= gate_req.ar_valid && !gate_resp.ar_ready;
            aw_held <= gate_req.aw_valid && !gate_resp.aw_ready;
            case ({gate_req.ar_valid && gate_resp.ar_ready,
                   gate_resp.r_valid && gate_req.r_ready && gate_resp.r.last})
                2'b10: rd_outstanding <= rd_outstanding + 8'd1;
                2'b01: rd_outstanding <= rd_outstanding - 8'd1;
                default: ;
            endcase
            case ({gate_req.aw_valid && gate_resp.aw_ready,
                   gate_resp.b_valid && gate_req.b_ready})
                2'b10: wr_outstanding <= wr_outstanding + 8'd1;
                2'b01: wr_outstanding <= wr_outstanding - 8'd1;
                default: ;
            endcase
        end
    end

    // --------------------------------------------------------- address split
    // The DDR base is 0, so its lower bound is constant; kept for a different
    // map.
    /* verilator lint_off UNSIGNED */
    function automatic logic [1:0] port_of(input addr_t addr);
        if ((64'(addr) >= cva6_npu_config_pkg::NpuDdrBase) &&
            (65'(addr) < 65'(cva6_npu_config_pkg::NpuDdrBase) +
                         65'(cva6_npu_config_pkg::NpuDdrLength)))
            port_of = PORT_DDR;
        else if ((64'(addr) >= cva6_npu_config_pkg::NpuMmioBase) &&
                 (65'(addr) < 65'(cva6_npu_config_pkg::NpuMmioBase) +
                              65'(cva6_npu_config_pkg::NpuMmioLength)))
            port_of = PORT_MMIO;
        else
            port_of = PORT_ERR;
    endfunction
    /* verilator lint_on UNSIGNED */

    cpu_req_t  [2:0] demux_req;
    cpu_resp_t [2:0] demux_resp;
    // One variable per demux port, each driven in one place.
    cpu_resp_t ddr_resp, mmio_resp, err_resp;

    assign demux_resp[PORT_DDR] = ddr_resp;
    assign demux_resp[PORT_MMIO] = mmio_resp;
    assign demux_resp[PORT_ERR] = err_resp;

    axi_demux #(
        .AxiIdWidth  (IdWidth),
        .aw_chan_t   (cpu_aw_chan_t),
        .w_chan_t    (cpu_w_chan_t),
        .b_chan_t    (cpu_b_chan_t),
        .ar_chan_t   (cpu_ar_chan_t),
        .r_chan_t    (cpu_r_chan_t),
        .req_t       (cpu_req_t),
        .resp_t      (cpu_resp_t),
        .NoMstPorts  (3),
        .MaxTrans    (MaxTxns),
        .AxiLookBits (IdWidth)
    ) u_demux (
        .clk_i           (clk),
        .rst_ni          (rst_n),
        .test_i          (1'b0),
        .slv_req_i       (gate_req),
        .slv_aw_select_i (port_of(gate_req.aw.addr)),
        .slv_ar_select_i (port_of(gate_req.ar.addr)),
        .slv_resp_o      (gate_resp),
        .mst_reqs_o      (demux_req),
        .mst_resps_i     (demux_resp)
    );

    axi_err_slv #(
        .AxiIdWidth (IdWidth),
        .req_t      (cpu_req_t),
        .resp_t     (cpu_resp_t),
        .Resp       (axi_pkg::RESP_DECERR),
        .ATOPs      (1'b1),
        .MaxTrans   (1)
    ) u_unmapped (
        .clk_i      (clk),
        .rst_ni     (rst_n),
        .test_i     (1'b0),
        .slv_req_i  (demux_req[PORT_ERR]),
        .slv_resp_o (err_resp)
    );

    // ------------------------------------------------------------------ MMIO
    logic        dma_ld_valid, dma_ld_ready, dma_st_valid, dma_st_ready;
    logic [31:0] dma_ld_a_addr, dma_ld_b_addr, dma_st_c_addr;
    logic [12:0] dma_ld_a_bytes, dma_ld_b_bytes;
    logic [8:0]  dma_st_words;
    logic        dma_jobs_written, dma_read_error, dma_write_error;

    // Flat response signals of the MMIO slave, packed into the struct in one
    // place.
    logic [IdWidth-1:0] mmio_bid, mmio_rid;
    logic [1:0]  mmio_bresp, mmio_rresp;
    logic        mmio_awready, mmio_wready, mmio_bvalid, mmio_arready;
    logic        mmio_rvalid, mmio_rlast;
    logic [63:0] mmio_rdata;

    always_comb begin
        mmio_resp = '0;
        mmio_resp.aw_ready = mmio_awready;
        mmio_resp.w_ready = mmio_wready;
        mmio_resp.b_valid = mmio_bvalid;
        mmio_resp.b.id = mmio_bid;
        mmio_resp.b.resp = mmio_bresp;
        mmio_resp.ar_ready = mmio_arready;
        mmio_resp.r_valid = mmio_rvalid;
        mmio_resp.r.id = mmio_rid;
        mmio_resp.r.data = mmio_rdata;
        mmio_resp.r.resp = mmio_rresp;
        mmio_resp.r.last = mmio_rlast;
    end

    cva6_npu_mmio #(
        .ROWS(ROWS), .COLUMNS(COLUMNS), .MAX_K(MAX_K), .ID_WIDTH(IdWidth)
    ) u_mmio (
        .clk(clk), .rst_n(rst_n),
        .start(start), .running(state != IDLE),
        .data_base(data_base), .prog_len(prog_len),
        .s_awid(demux_req[PORT_MMIO].aw.id),
        .s_awaddr(demux_req[PORT_MMIO].aw.addr[31:0]),
        .s_awlen(demux_req[PORT_MMIO].aw.len),
        .s_awvalid(demux_req[PORT_MMIO].aw_valid),
        .s_awready(mmio_awready),
        .s_wdata(demux_req[PORT_MMIO].w.data),
        .s_wstrb(demux_req[PORT_MMIO].w.strb),
        .s_wlast(demux_req[PORT_MMIO].w.last),
        .s_wvalid(demux_req[PORT_MMIO].w_valid),
        .s_wready(mmio_wready),
        .s_bid(mmio_bid),
        .s_bresp(mmio_bresp),
        .s_bvalid(mmio_bvalid),
        .s_bready(demux_req[PORT_MMIO].b_ready),
        .s_arid(demux_req[PORT_MMIO].ar.id),
        .s_araddr(demux_req[PORT_MMIO].ar.addr[31:0]),
        .s_arlen(demux_req[PORT_MMIO].ar.len),
        .s_arvalid(demux_req[PORT_MMIO].ar_valid),
        .s_arready(mmio_arready),
        .s_rid(mmio_rid),
        .s_rdata(mmio_rdata),
        .s_rresp(mmio_rresp),
        .s_rlast(mmio_rlast),
        .s_rvalid(mmio_rvalid),
        .s_rready(demux_req[PORT_MMIO].r_ready),
        .ctl_start(ctl_start), .ctl_m(ctl_m), .ctl_n(ctl_n), .ctl_k(ctl_k),
        .ctl_accept(ctl_accept), .ctl_error(ctl_error),
        .ctl_error_code(ctl_error_code),
        .ld_valid(dma_ld_valid), .ld_ready(dma_ld_ready),
        .ld_a_addr(dma_ld_a_addr), .ld_a_bytes(dma_ld_a_bytes),
        .ld_b_addr(dma_ld_b_addr), .ld_b_bytes(dma_ld_b_bytes),
        .st_valid(dma_st_valid), .st_ready(dma_st_ready),
        .st_c_addr(dma_st_c_addr), .st_words(dma_st_words),
        .jobs_written(dma_jobs_written),
        .read_error(dma_read_error), .write_error(dma_write_error),
        .done(mmio_done), .done_code(mmio_done_code),
        .error(mmio_error), .error_code(mmio_error_code),
        .busy(mmio_busy), .drained(mmio_drained),
        .jobs_issued(mmio_jobs_issued), .jobs_done(mmio_jobs_done)
    );

    // ------------------------------------------------------------ tensor DMA
    // Flat DMA signals on mux port 1 (ID 0, 8-byte INCR bursts, the CUSTOM
    // core's cache / prot attributes).
    cpu_req_t  dma_req;
    cpu_resp_t dma_resp;
    logic        d_ar_valid, d_ar_ready, d_r_ready, d_aw_valid, d_aw_ready;
    logic        d_w_valid, d_w_ready, d_w_last, d_b_ready;
    logic [31:0] d_ar_addr, d_aw_addr;
    logic [7:0]  d_ar_len, d_aw_len;
    logic [63:0] d_w_data;
    logic [7:0]  d_w_strb;

    cva6_dma dma (
        .clk(clk), .rst_n(rst_n), .clear_errors(start),
        .ld_valid(dma_ld_valid), .ld_ready(dma_ld_ready),
        .ld_a_addr(dma_ld_a_addr), .ld_a_bytes(dma_ld_a_bytes),
        .ld_b_addr(dma_ld_b_addr), .ld_b_bytes(dma_ld_b_bytes),
        .st_valid(dma_st_valid), .st_ready(dma_st_ready),
        .st_c_addr(dma_st_c_addr), .st_words(dma_st_words),
        .ar_valid(d_ar_valid), .ar_ready(d_ar_ready),
        .ar_addr(d_ar_addr), .ar_len(d_ar_len),
        .r_valid(dma_resp.r_valid), .r_ready(d_r_ready),
        .r_data(dma_resp.r.data), .r_resp(dma_resp.r.resp),
        .aw_valid(d_aw_valid), .aw_ready(d_aw_ready),
        .aw_addr(d_aw_addr), .aw_len(d_aw_len),
        .w_valid(d_w_valid), .w_ready(d_w_ready),
        .w_data(d_w_data), .w_strb(d_w_strb), .w_last(d_w_last),
        .b_valid(dma_resp.b_valid), .b_ready(d_b_ready), .b_resp(dma_resp.b.resp),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast),
        .jobs_written(dma_jobs_written),
        .read_error(dma_read_error), .write_error(dma_write_error)
    );

    always_comb begin
        dma_req = '0;
        dma_req.ar.addr = addr_t'(d_ar_addr);
        dma_req.ar.len = d_ar_len;
        dma_req.ar.size = 3'd3;           // 8-byte beats
        dma_req.ar.burst = axi_pkg::BURST_INCR;
        dma_req.ar.cache = 4'b0011;
        dma_req.ar_valid = d_ar_valid;
        dma_req.r_ready = d_r_ready;
        dma_req.aw.addr = addr_t'(d_aw_addr);
        dma_req.aw.len = d_aw_len;
        dma_req.aw.size = 3'd3;
        dma_req.aw.burst = axi_pkg::BURST_INCR;
        dma_req.aw.cache = 4'b0011;
        dma_req.aw_valid = d_aw_valid;
        dma_req.w.data = d_w_data;
        dma_req.w.strb = d_w_strb;
        dma_req.w.last = d_w_last;
        dma_req.w_valid = d_w_valid;
        dma_req.b_ready = d_b_ready;
        d_ar_ready = dma_resp.ar_ready;
        d_aw_ready = dma_resp.aw_ready;
        d_w_ready = dma_resp.w_ready;
    end

    // ------------------------------------------------- shared AXI4 master
    cpu_req_t  [1:0] mux_slv_req;
    cpu_resp_t [1:0] mux_slv_resp;
    mux_req_t  mux_req;
    mux_resp_t mux_resp;
    ser_req_t  ser_req;
    ser_resp_t ser_resp;

    assign mux_slv_req[0] = demux_req[PORT_DDR];
    assign ddr_resp = mux_slv_resp[0];
    assign mux_slv_req[1] = dma_req;
    assign dma_resp = mux_slv_resp[1];

    axi_mux #(
        .SlvAxiIDWidth (IdWidth),
        .slv_aw_chan_t (cpu_aw_chan_t),
        .mst_aw_chan_t (mux_aw_chan_t),
        .w_chan_t      (cpu_w_chan_t),
        .slv_b_chan_t  (cpu_b_chan_t),
        .mst_b_chan_t  (mux_b_chan_t),
        .slv_ar_chan_t (cpu_ar_chan_t),
        .mst_ar_chan_t (mux_ar_chan_t),
        .slv_r_chan_t  (cpu_r_chan_t),
        .mst_r_chan_t  (mux_r_chan_t),
        .slv_req_t     (cpu_req_t),
        .slv_resp_t    (cpu_resp_t),
        .mst_req_t     (mux_req_t),
        .mst_resp_t    (mux_resp_t),
        .NoSlvPorts    (2),
        .MaxWTrans     (MaxTxns)
    ) u_mux (
        .clk_i       (clk),
        .rst_ni      (rst_n),
        .test_i      (1'b0),
        .slv_reqs_i  (mux_slv_req),
        .slv_resps_o (mux_slv_resp),
        .mst_req_o   (mux_req),
        .mst_resp_i  (mux_resp)
    );

    axi_id_serialize #(
        .AxiSlvPortIdWidth      (MuxIdWidth),
        .AxiSlvPortMaxTxns      (MaxTxns),
        .AxiMstPortIdWidth      (1),
        .AxiMstPortMaxUniqIds   (1),
        .AxiMstPortMaxTxnsPerId (MaxTxns),
        .AxiAddrWidth           (AddrWidth),
        .AxiDataWidth           (DataWidth),
        .AxiUserWidth           (UserWidth),
        .slv_req_t              (mux_req_t),
        .slv_resp_t             (mux_resp_t),
        .mst_req_t              (ser_req_t),
        .mst_resp_t             (ser_resp_t)
    ) u_id_serialize (
        .clk_i      (clk),
        .rst_ni     (rst_n),
        .slv_req_i  (mux_req),
        .slv_resp_o (mux_resp),
        .mst_req_o  (ser_req),
        .mst_resp_i (ser_resp)
    );

    // Pulp structs <-> flat m_axi. The address decode only routes addresses
    // below NpuDdrLength here, so dropping bits [63:32] loses nothing; the
    // constant ID, QoS, region, lock, atop and user fields are not carried.
    always_comb begin
        m_axi_araddr = ser_req.ar.addr[31:0];
        m_axi_arlen = ser_req.ar.len;
        m_axi_arsize = ser_req.ar.size;
        m_axi_arburst = ser_req.ar.burst;
        m_axi_arcache = ser_req.ar.cache;
        m_axi_arprot = ser_req.ar.prot;
        m_axi_arvalid = ser_req.ar_valid;
        m_axi_rready = ser_req.r_ready;
        m_axi_awaddr = ser_req.aw.addr[31:0];
        m_axi_awlen = ser_req.aw.len;
        m_axi_awsize = ser_req.aw.size;
        m_axi_awburst = ser_req.aw.burst;
        m_axi_awcache = ser_req.aw.cache;
        m_axi_awprot = ser_req.aw.prot;
        m_axi_awvalid = ser_req.aw_valid;
        m_axi_wdata = ser_req.w.data;
        m_axi_wstrb = ser_req.w.strb;
        m_axi_wlast = ser_req.w.last;
        m_axi_wvalid = ser_req.w_valid;
        m_axi_bready = ser_req.b_ready;

        ser_resp = '0;
        ser_resp.ar_ready = m_axi_arready;
        ser_resp.aw_ready = m_axi_awready;
        ser_resp.w_ready = m_axi_wready;
        ser_resp.b_valid = m_axi_bvalid;
        ser_resp.b.resp = m_axi_bresp;
        ser_resp.r_valid = m_axi_rvalid;
        ser_resp.r.data = m_axi_rdata;
        ser_resp.r.resp = m_axi_rresp;
        ser_resp.r.last = m_axi_rlast;
    end
endmodule
