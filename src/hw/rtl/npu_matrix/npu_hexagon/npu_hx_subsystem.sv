`timescale 1ns/1ps

// npu_hx_subsystem: the Hexagon-style heterogeneous subsystem (issue #136),
// the only place where its units are wired together
// (docs/goals/hexagon_npu/architecture.md, "Subsystem integration").
//
//   host AXI-lite --> npu_hx_regs <-- scalar MMIO (0x8000_0xxx)
//                          |  \--> TCM port B (host window, lowest priority)
//   npu_scalar_core -- i --> L1 I$ --\
//        |          -- d --> decode --> L1 D$ --> npu_mem_arbiter --> L2 --> line bridge --\
//        |                     |--> TCM port B (32-bit lane)                              |
//        |                     `--> npu_hx_regs MMIO                                      v
//        `-- vcmd --> npu_vector_unit --> TCM port A      npu_tcm_dma --> npu_axi4_arbiter --> m_axi
//                                         npu_tcm port B <-- DMA (highest priority)
//
// Ports: s_axi_* AXI4-Lite slave (32-bit data, 16-bit address), m_axi_* AXI4
// master (32-bit address, 64-bit data, no ID; names as npu_accelerator's
// m_axi_*), irq (level). Clock s_axi_aclk, active-low reset s_axi_aresetn
// (synchronous to s_axi_aclk, from proc_sys_reset; used as the units'
// asynchronous rst_n).
//
// Scalar data-port address map (the core's word-aligned d_req_addr):
//   0x0000_0000-0x3FFF_FFFF  DDR through L1 D$ (and L2)
//   0x4000_0000-0x4000_FFFF  TCM port B, uncached: byte address a selects TCM
//                            word a[15:3] and the 32-bit lane a[2]; stores
//                            map d_req_wstrb onto that lane's byte strobes.
//                            Words at or past TCM_DEPTH (a[15:3] >= depth,
//                            i.e. a >= 0x4000_2000 for 1024 words) are
//                            unmapped.
//   0x8000_0000-0x8000_0FFF  npu_hx_regs registers (offset = a[11:0])
//   anything else            unmapped: the access is answered with
//                            d_resp_err the cycle after the handshake, so the
//                            core stops with fault_cause 6 (data bus error).
// Instruction fetches always go to the L1 I$ (the TCM and MMIO are not
// executable; an instruction fetch outside DDR reads whatever the AXI slave
// returns). AXI response errors of cache traffic do not reach the core (the
// caches have no error path): the line bridge's error pulse sets the sticky
// STATUS.bus_error instead, so i_resp_err is tied to 0 and d_resp_err only
// flags unmapped data addresses.
//
// TCM port B arbiter: fixed priority DMA > scalar > host, combinational
// grant from the clients' requests. The scalar and host requests come from
// registers (the scalar window adapter registers the handshake and drives
// the core's request fields, which the core holds until its response; the
// host window request is a register in npu_hx_regs); the DMA's request
// never depends on gnt and the DMA tolerates gnt low. Scalar window latency:
// handshake at t, port-B request at t+1, response (read data lane) at t+2
// when the DMA is not using the port. MMIO latency: response at t+2.
// Port A belongs to the vector unit.
//
// CONTROL.FLUSH sequencing (flush_busy = STATUS.flush_busy): pulse the L1 D$
// flush (write back dirty lines into the L2, invalidate) and the L1 I$ flush
// (a READ_ONLY cache only invalidates), then, after the L1 D$ flush_done,
// pulse the L2 flush (write back to DDR, invalidate); flush_busy falls once
// the L2 and the L1 I$ have both reported flush_done. A FLUSH while busy is
// ignored. The caches serve a flush between CPU requests, so it is safe with
// the core running, but software normally flushes after the program halted.
//
// Parameters for PPA deltas (each disabled unit vanishes from synthesis and
// its CAPS bit reads 0):
//   HX_ENABLE_VECTOR = 0  no npu_vector_unit; vector commands are consumed
//                         (vcmd_ready = 1, vec_busy = 0) and set the sticky
//                         STATUS.vec_illegal; TCM port A is idle.
//   HX_ENABLE_L2 = 0      no L2: npu_mem_arbiter's line port goes to the line
//                         bridge through a 64 -> 32-bit read-beat serializer
//                         (4 x 64-bit distributed-RAM FIFO; the line widths
//                         match, only the beat width differs). The L2
//                         counters read 0 and FLUSH skips the L2 step.
//   HX_ENABLE_DMA = 0     no npu_tcm_dma and no npu_axi4_arbiter: the line
//                         bridge drives m_axi_* directly; DMA_* registers read
//                         0, DMA_START is ignored, STATUS dma bits read 0.
// CAPS = {TCM words / 64, VLEN/8 (0 without the vector unit), 0, DMA, TCM=1,
// L2, L1D=1, L1I=1, VECTOR, SCALAR=1}.
//
// IRQ (deviation from architecture.md's "STATUS & IRQ_ENABLE"): irq is
// (halt event & IRQ_ENABLE[0]) | (fault event & [1]) | (dma_done & [2]), where
// the halt / fault events are sticky copies of the rising edges of
// STATUS.halted / fault (STATUS bits 10 / 11, additions). CONTROL.CLEAR (or
// the next START) clears them, so the host can acknowledge a level interrupt
// without restarting the core. See npu_hx_regs.
//
// Glue timing: every inter-unit path below starts at a unit's registered
// output or at a register of this module (port-B requests, response
// pulses, flush pulses, the serializer output); the remaining combinational
// glue is the data-port decode (address bits -> valid/ready select), the
// response mux and the port-B priority mux. Cross-module chains (LUT levels,
// review round 1 estimate):
//   core alu_q -> d_is_* decode (3) -> d_req_ready / l1d_req_valid (1) ->
//     core state, L1 D$ accept, ts_pend_q / mm_pend_q / err_resp_q (~2): ~6;
//   L1 D$ tag register -> tag compare (resp_valid, 3) -> d_resp_valid (1) ->
//     core state / load enable (~2): ~6 (the load data mux does not wait
//     for the compare, see d_resp_rdata);
//   DMA rd_issue (3) -> port-B mux (2) -> TCM BRAM address / WE setup, and
//     -> h_gnt / ts_gnt -> npu_hx_regs window FSM, ts_pend_q (~2): ~5;
//   TCM BRAM clock-to-out -> window lane mux -> core load_q / s_axi_rdata
//     (2) (BRAM without output register: ~2 ns clock-to-out).
// npu_hx_regs: AXI hold registers / core address -> register index mux ->
// 32:1 read mux -> s_axi_rdata / mmio_resp_rdata, ~7 levels. All are well
// inside 10 ns; the scalar core's own DECODE paths (~7 ns) stay the worst.
module npu_hx_subsystem #(
    parameter integer HX_ENABLE_VECTOR = 1,
    parameter integer HX_ENABLE_L2 = 1,
    parameter integer HX_ENABLE_DMA = 1,
    parameter integer HX_TCM_DEPTH = 1024
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF s_axi:m_axi, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
    input  logic        s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aresetn, POLARITY ACTIVE_LOW" *)
    input  logic        s_axi_aresetn,
    input  logic [15:0] s_axi_awaddr,
    input  logic [2:0]  s_axi_awprot,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [15:0] s_axi_araddr,
    input  logic [2:0]  s_axi_arprot,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

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
    output logic        m_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output logic        irq
);
    localparam integer VLEN = 64;
    localparam integer LINE_BYTES = 32;
    localparam integer LINE_W = LINE_BYTES * 8;
    localparam integer TCM_DEPTH = HX_TCM_DEPTH;
    localparam integer TCM_AW = $clog2(TCM_DEPTH);
    localparam logic EN_VEC = (HX_ENABLE_VECTOR != 0);
    localparam logic EN_L2 = (HX_ENABLE_L2 != 0);
    localparam logic EN_DMA = (HX_ENABLE_DMA != 0);
    localparam logic [31:0] CAPS = {16'(TCM_DEPTH / 64), (EN_VEC ? 8'(VLEN / 8) : 8'd0),
                                    1'b0, EN_DMA, 1'b1, EN_L2, 1'b1, 1'b1, EN_VEC, 1'b1};

    initial begin
        if (TCM_DEPTH < 64 || TCM_DEPTH > 4096 || (1 << TCM_AW) != TCM_DEPTH)
            $fatal(1, "npu_hx_subsystem HX_TCM_DEPTH must be a power of two, 64..4096");
    end

    logic clk, rst_n;
    assign clk = s_axi_aclk;
    assign rst_n = s_axi_aresetn;

    // ------------------------------------------------------------ signals
    // scalar core
    logic        core_start, running, halted, halt_cause, fault;
    logic [2:0]  fault_cause;
    logic [31:0] start_pc, fault_pc, core_cycles, core_instret;
    logic        i_req_valid, i_req_ready, i_resp_valid;
    logic [31:0] i_req_addr, i_resp_rdata;
    logic        d_req_valid, d_req_ready, d_req_we, d_resp_valid, d_resp_err;
    logic [31:0] d_req_addr, d_req_wdata, d_resp_rdata;
    logic [3:0]  d_req_wstrb;
    logic        vcmd_valid, vcmd_ready, vec_busy, vec_illegal;
    logic [4:0]  vcmd_op;
    logic [1:0]  vcmd_type;
    logic [2:0]  vcmd_vd, vcmd_vs1, vcmd_vs2;
    logic [31:0] vcmd_scalar;
    // L1 D$ CPU side
    logic        l1d_req_valid, l1d_req_ready, l1d_resp_valid;
    logic [31:0] l1d_resp_rdata;
    // caches: flush and counters
    logic        l1i_flush, l1d_flush, l2_flush;
    logic        l1i_flush_done, l1d_flush_done, l2_flush_done;
    logic [31:0] l1i_hits, l1i_misses, l1d_hits, l1d_misses, l2_hits, l2_misses;
    // line ports: L1I -> arb s0, L1D -> arb s1, arb m -> L2 (or serializer)
    logic                 l1i_mreq_valid, l1i_mreq_ready, l1i_mreq_we, l1i_mresp_valid;
    logic [31:0]          l1i_mreq_addr, l1i_mresp_rdata;
    logic [LINE_W-1:0]    l1i_mreq_wdata;
    logic                 l1d_mreq_valid, l1d_mreq_ready, l1d_mreq_we, l1d_mresp_valid;
    logic [31:0]          l1d_mreq_addr, l1d_mresp_rdata;
    logic [LINE_W-1:0]    l1d_mreq_wdata;
    logic                 arb_req_valid, arb_req_ready, arb_req_we, arb_resp_valid;
    logic [31:0]          arb_req_addr, arb_resp_rdata;
    logic [LINE_W-1:0]    arb_req_wdata;
    // line bridge slave side
    logic                 br_req_valid, br_req_ready, br_req_we, br_resp_valid, br_error;
    logic [31:0]          br_req_addr;
    logic [LINE_W-1:0]    br_req_wdata;
    logic [63:0]          br_resp_rdata;
    // line bridge AXI4 master
    logic [31:0] br_araddr, br_awaddr;
    logic [7:0]  br_arlen, br_awlen;
    logic [2:0]  br_arsize, br_awsize, br_arprot, br_awprot;
    logic [1:0]  br_arburst, br_awburst, br_rresp, br_bresp;
    logic [3:0]  br_arcache, br_awcache;
    logic        br_arvalid, br_arready, br_rlast, br_rvalid, br_rready;
    logic        br_awvalid, br_awready, br_wlast, br_wvalid, br_wready;
    logic        br_bvalid, br_bready;
    logic [63:0] br_rdata, br_wdata;
    logic [7:0]  br_wstrb;
    // DMA
    logic        dma_start, dma_dir, dma_busy, dma_done, dma_error;
    logic [31:0] dma_ddr_addr;
    logic [TCM_AW-1:0] dma_tcm_addr;
    logic [15:0] dma_words;
    logic        dma_b_req, dma_b_we;
    logic [TCM_AW-1:0] dma_b_addr;
    logic [63:0] dma_b_wdata;
    logic [7:0]  dma_b_wstrb;
    // TCM
    logic              a_req, a_we, a_gnt;
    logic [TCM_AW-1:0] a_addr;
    logic [63:0]       a_wdata, a_rdata;
    logic [7:0]        a_wstrb;
    logic              b_req, b_we, b_gnt;
    logic [TCM_AW-1:0] b_addr;
    logic [63:0]       b_wdata, b_rdata;
    logic [7:0]        b_wstrb;
    // regs
    logic        mmio_req_valid, mmio_req_ready, mmio_resp_valid;
    logic [31:0] mmio_resp_rdata;
    logic              h_req, h_we, h_gnt;
    logic [TCM_AW-1:0] h_addr;
    logic [63:0]       h_wdata;
    logic [7:0]        h_wstrb;
    logic        flush_start, flush_busy, clear_pulse;

    // ------------------------------------------------------------ scalar core
    npu_scalar_core u_core (
        .clk(clk), .rst_n(rst_n),
        .start(core_start), .start_pc(start_pc),
        .running(running), .halted(halted), .halt_cause(halt_cause),
        .fault(fault), .fault_cause(fault_cause), .fault_pc(fault_pc),
        .i_req_valid(i_req_valid), .i_req_ready(i_req_ready), .i_req_addr(i_req_addr),
        .i_resp_valid(i_resp_valid), .i_resp_rdata(i_resp_rdata), .i_resp_err(1'b0),
        .d_req_valid(d_req_valid), .d_req_ready(d_req_ready), .d_req_addr(d_req_addr),
        .d_req_we(d_req_we), .d_req_wstrb(d_req_wstrb), .d_req_wdata(d_req_wdata),
        .d_resp_valid(d_resp_valid), .d_resp_rdata(d_resp_rdata), .d_resp_err(d_resp_err),
        .vcmd_valid(vcmd_valid), .vcmd_ready(vcmd_ready), .vcmd_op(vcmd_op),
        .vcmd_type(vcmd_type), .vcmd_vd(vcmd_vd), .vcmd_vs1(vcmd_vs1), .vcmd_vs2(vcmd_vs2),
        .vcmd_scalar(vcmd_scalar), .vec_busy(vec_busy),
        .cycles(core_cycles), .instret(core_instret)
    );

    // ------------------------------------------------------------ data-port decode
    logic d_is_ddr, d_is_tcm, d_is_mmio, d_is_err;
    logic ts_pend_q, ts_resp_q, ts_lane_q;
    logic [7:0]  ts_wstrb;
    logic        err_resp_q;

    // Continuous assignments (not one always_comb): each output is written
    // once, so simulators see no intermediate values on the valid/ready nets.
    assign d_is_ddr = (d_req_addr[31:30] == 2'b00);
    assign d_is_tcm = (d_req_addr[31:16] == 16'h4000) && (32'(d_req_addr[15:3]) < TCM_DEPTH);
    assign d_is_mmio = (d_req_addr[31:12] == 20'h80000);
    assign d_is_err = !d_is_ddr && !d_is_tcm && !d_is_mmio;
    assign l1d_req_valid = d_req_valid && d_is_ddr;
    assign mmio_req_valid = d_req_valid && d_is_mmio;
    assign d_req_ready = (d_is_ddr && l1d_req_ready) || (d_is_tcm && !ts_pend_q) ||
                         (d_is_mmio && mmio_req_ready) || d_is_err;
    // One response source at a time (one request outstanding). The read data
    // is selected by the two registered response flags (TCM window, MMIO) and
    // defaults to the L1 D$, whose resp_valid is its tag compare: that compare
    // only gates the core's load enable, not the 32-bit data mux. d_resp_rdata
    // is don't-care without d_resp_valid (and with d_resp_err).
    assign d_resp_valid = l1d_resp_valid || ts_resp_q || mmio_resp_valid || err_resp_q;
    assign d_resp_err = err_resp_q;
    assign d_resp_rdata = ts_resp_q ? (ts_lane_q ? b_rdata[63:32] : b_rdata[31:0]) :
                          mmio_resp_valid ? mmio_resp_rdata : l1d_resp_rdata;

    // Scalar TCM window adapter: registered request, held until granted.
    logic ts_gnt;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ts_pend_q <= 1'b0;
            ts_resp_q <= 1'b0;
            err_resp_q <= 1'b0;
        end else begin
            ts_resp_q <= ts_gnt;
            err_resp_q <= d_req_valid && d_is_err;
            if (d_req_valid && d_is_tcm && !ts_pend_q)
                ts_pend_q <= 1'b1;
            else if (ts_gnt)
                ts_pend_q <= 1'b0;
        end
    end

    // The core holds d_req_addr / we / wstrb / wdata until its response, so
    // the port-B request uses them directly; only the lane of a read is kept
    // for the response cycle.
    always_ff @(posedge clk) begin
        if (ts_gnt)
            ts_lane_q <= d_req_addr[2];
    end
    assign ts_wstrb = d_req_addr[2] ? {d_req_wstrb, 4'd0} : {4'd0, d_req_wstrb};

    // ------------------------------------------------------------ TCM port-B arbiter
    // Fixed priority DMA > scalar > host.
    assign ts_gnt = ts_pend_q && !dma_b_req && b_gnt;
    assign h_gnt = h_req && !dma_b_req && !ts_pend_q && b_gnt;
    assign b_req = dma_b_req || ts_pend_q || h_req;
    assign b_we = dma_b_req ? dma_b_we : (ts_pend_q ? d_req_we : h_we);
    assign b_addr = dma_b_req ? dma_b_addr : (ts_pend_q ? d_req_addr[TCM_AW+2:3] : h_addr);
    assign b_wdata = dma_b_req ? dma_b_wdata : (ts_pend_q ? {d_req_wdata, d_req_wdata} : h_wdata);
    assign b_wstrb = dma_b_req ? dma_b_wstrb : (ts_pend_q ? ts_wstrb : h_wstrb);

    npu_tcm #(.WIDTH(VLEN), .DEPTH(TCM_DEPTH), .AW(TCM_AW)) u_tcm (
        .clk(clk), .rst_n(rst_n),
        .a_req(a_req), .a_we(a_we), .a_addr(a_addr), .a_wdata(a_wdata), .a_wstrb(a_wstrb),
        .a_gnt(a_gnt), .a_rdata(a_rdata),
        .b_req(b_req), .b_we(b_we), .b_addr(b_addr), .b_wdata(b_wdata), .b_wstrb(b_wstrb),
        .b_gnt(b_gnt), .b_rdata(b_rdata)
    );

    // ------------------------------------------------------------ vector unit
    generate
        if (HX_ENABLE_VECTOR != 0) begin : g_vec
            npu_vector_unit #(.VLEN(VLEN), .NREGS(8), .TCM_AW(TCM_AW)) u_vec (
                .clk(clk), .rst_n(rst_n),
                .cmd_valid(vcmd_valid), .cmd_ready(vcmd_ready), .cmd_op(vcmd_op),
                .cmd_type(vcmd_type), .cmd_vd(vcmd_vd), .cmd_vs1(vcmd_vs1), .cmd_vs2(vcmd_vs2),
                .cmd_addr(vcmd_scalar[TCM_AW-1:0]), .cmd_scalar(vcmd_scalar),
                .busy(vec_busy), .illegal(vec_illegal), .illegal_clear(clear_pulse),
                .tcm_req(a_req), .tcm_we(a_we), .tcm_addr(a_addr), .tcm_wdata(a_wdata),
                .tcm_wstrb(a_wstrb), .tcm_gnt(a_gnt), .tcm_rdata(a_rdata)
            );
        end else begin : g_no_vec
            // Commands are consumed and flagged as illegal.
            logic illegal_q;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    illegal_q <= 1'b0;
                else if (vcmd_valid)
                    illegal_q <= 1'b1;
                else if (clear_pulse)
                    illegal_q <= 1'b0;
            end
            assign vcmd_ready = 1'b1;
            assign vec_busy = 1'b0;
            assign vec_illegal = illegal_q;
            assign a_req = 1'b0;
            assign a_we = 1'b0;
            assign a_addr = '0;
            assign a_wdata = '0;
            assign a_wstrb = '0;
            /* verilator lint_off UNUSEDSIGNAL */
            logic unused_vec;
            assign unused_vec = a_gnt ^ (^a_rdata) ^ (^vcmd_op) ^ (^vcmd_type) ^
                                (^vcmd_vd) ^ (^vcmd_vs1) ^ (^vcmd_vs2) ^ (^vcmd_scalar);
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate

    // ------------------------------------------------------------ caches
    npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(LINE_BYTES), .READ_ONLY(1)) u_l1i (
        .clk(clk), .rst_n(rst_n),
        .req_valid(i_req_valid), .req_ready(i_req_ready), .req_addr(i_req_addr),
        .req_we(1'b0), .req_wstrb(4'd0), .req_wdata(32'd0),
        .resp_valid(i_resp_valid), .resp_rdata(i_resp_rdata),
        .flush(l1i_flush), .flush_done(l1i_flush_done),
        .mem_req_valid(l1i_mreq_valid), .mem_req_ready(l1i_mreq_ready),
        .mem_req_addr(l1i_mreq_addr), .mem_req_we(l1i_mreq_we), .mem_req_wdata(l1i_mreq_wdata),
        .mem_resp_valid(l1i_mresp_valid), .mem_resp_rdata(l1i_mresp_rdata),
        .hits(l1i_hits), .misses(l1i_misses)
    );

    npu_cache #(.SIZE_BYTES(1024), .LINE_BYTES(LINE_BYTES), .WRITE_BACK(1), .CPU_HOLD(1)) u_l1d (
        .clk(clk), .rst_n(rst_n),
        .req_valid(l1d_req_valid), .req_ready(l1d_req_ready), .req_addr(d_req_addr),
        .req_we(d_req_we), .req_wstrb(d_req_wstrb), .req_wdata(d_req_wdata),
        .resp_valid(l1d_resp_valid), .resp_rdata(l1d_resp_rdata),
        .flush(l1d_flush), .flush_done(l1d_flush_done),
        .mem_req_valid(l1d_mreq_valid), .mem_req_ready(l1d_mreq_ready),
        .mem_req_addr(l1d_mreq_addr), .mem_req_we(l1d_mreq_we), .mem_req_wdata(l1d_mreq_wdata),
        .mem_resp_valid(l1d_mresp_valid), .mem_resp_rdata(l1d_mresp_rdata),
        .hits(l1d_hits), .misses(l1d_misses)
    );

    npu_mem_arbiter #(.LINE_BYTES(LINE_BYTES), .BEAT_W(32), .P0_READ_ONLY(1)) u_mem_arb (
        .clk(clk), .rst_n(rst_n),
        .s0_mem_req_valid(l1i_mreq_valid), .s0_mem_req_ready(l1i_mreq_ready),
        .s0_mem_req_addr(l1i_mreq_addr), .s0_mem_req_we(l1i_mreq_we),
        .s0_mem_req_wdata(l1i_mreq_wdata),
        .s0_mem_resp_valid(l1i_mresp_valid), .s0_mem_resp_rdata(l1i_mresp_rdata),
        .s1_mem_req_valid(l1d_mreq_valid), .s1_mem_req_ready(l1d_mreq_ready),
        .s1_mem_req_addr(l1d_mreq_addr), .s1_mem_req_we(l1d_mreq_we),
        .s1_mem_req_wdata(l1d_mreq_wdata),
        .s1_mem_resp_valid(l1d_mresp_valid), .s1_mem_resp_rdata(l1d_mresp_rdata),
        .m_mem_req_valid(arb_req_valid), .m_mem_req_ready(arb_req_ready),
        .m_mem_req_addr(arb_req_addr), .m_mem_req_we(arb_req_we),
        .m_mem_req_wdata(arb_req_wdata),
        .m_mem_resp_valid(arb_resp_valid), .m_mem_resp_rdata(arb_resp_rdata)
    );

    generate
        if (HX_ENABLE_L2 != 0) begin : g_l2
            npu_cache #(.SIZE_BYTES(8192), .LINE_BYTES(LINE_BYTES), .WAYS(2), .CPU_W(LINE_W),
                        .WRITE_BACK(1), .CPU_HOLD(1), .MEM_W(64), .RESP_W(32)) u_l2 (
                .clk(clk), .rst_n(rst_n),
                .req_valid(arb_req_valid), .req_ready(arb_req_ready), .req_addr(arb_req_addr),
                .req_we(arb_req_we), .req_wstrb({(LINE_W/8){1'b1}}), .req_wdata(arb_req_wdata),
                .resp_valid(arb_resp_valid), .resp_rdata(arb_resp_rdata),
                .flush(l2_flush), .flush_done(l2_flush_done),
                .mem_req_valid(br_req_valid), .mem_req_ready(br_req_ready),
                .mem_req_addr(br_req_addr), .mem_req_we(br_req_we), .mem_req_wdata(br_req_wdata),
                .mem_resp_valid(br_resp_valid), .mem_resp_rdata(br_resp_rdata),
                .hits(l2_hits), .misses(l2_misses)
            );
        end else begin : g_no_l2
            // The arbiter's line port goes straight to the bridge; 64-bit read
            // beats are split into the 32-bit beats of the L1 line protocol.
            // At most LINE_BYTES/8 = 4 beats per transaction, one transaction
            // at a time (the arbiter holds the request until its last 32-bit
            // beat), so a 4-entry FIFO never overflows.
            logic [63:0] sfifo [0:3];
            logic [2:0]  s_wptr_q, s_rptr_q;
            logic        s_half_q, s_valid_q;
            logic [31:0] s_data_q;
            logic        s_empty;
            logic [63:0] s_head;
            assign br_req_valid = arb_req_valid;
            assign arb_req_ready = br_req_ready;
            assign br_req_addr = arb_req_addr;
            assign br_req_we = arb_req_we;
            assign br_req_wdata = arb_req_wdata;
            assign s_empty = (s_wptr_q == s_rptr_q);
            assign s_head = sfifo[s_rptr_q[1:0]];
            always_ff @(posedge clk) begin
                if (br_resp_valid && !arb_req_we)
                    sfifo[s_wptr_q[1:0]] <= br_resp_rdata;
            end
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    s_wptr_q <= '0;
                    s_rptr_q <= '0;
                    s_half_q <= 1'b0;
                    s_valid_q <= 1'b0;
                    s_data_q <= '0;
                end else begin
                    if (br_resp_valid && !arb_req_we)
                        s_wptr_q <= s_wptr_q + 3'd1;
                    if (!s_empty) begin
                        s_valid_q <= 1'b1;
                        s_data_q <= s_half_q ? s_head[63:32] : s_head[31:0];
                        s_half_q <= !s_half_q;
                        if (s_half_q)
                            s_rptr_q <= s_rptr_q + 3'd1;
                    end else begin
                        // write response: one pulse, passed through registered
                        s_valid_q <= br_resp_valid && arb_req_we;
                    end
                end
            end
            assign arb_resp_valid = s_valid_q;
            assign arb_resp_rdata = s_data_q;
            assign l2_flush_done = 1'b0;
            assign l2_hits = 32'd0;
            assign l2_misses = 32'd0;
            /* verilator lint_off UNUSEDSIGNAL */
            logic unused_l2;
            assign unused_l2 = l2_flush;
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate

    // ------------------------------------------------------------ flush sequencing
    localparam logic [1:0] F_IDLE = 2'd0;
    localparam logic [1:0] F_L1   = 2'd1;   // L1 D$ (and L1 I$) flushing
    localparam logic [1:0] F_L2   = 2'd2;   // L2 flushing
    localparam logic [1:0] F_I    = 2'd3;   // waiting for the L1 I$ only
    logic [1:0] fstate_q;
    logic       l1i_done_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fstate_q <= F_IDLE;
            l1i_done_q <= 1'b0;
            l1i_flush <= 1'b0;
            l1d_flush <= 1'b0;
            l2_flush <= 1'b0;
        end else begin
            l1i_flush <= 1'b0;
            l1d_flush <= 1'b0;
            l2_flush <= 1'b0;
            if (l1i_flush_done)
                l1i_done_q <= 1'b1;
            case (fstate_q)
                F_IDLE: begin
                    if (flush_start) begin
                        l1i_flush <= 1'b1;
                        l1d_flush <= 1'b1;
                        l1i_done_q <= 1'b0;
                        fstate_q <= F_L1;
                    end
                end
                F_L1: begin
                    if (l1d_flush_done) begin
                        if (EN_L2) begin
                            l2_flush <= 1'b1;
                            fstate_q <= F_L2;
                        end else begin
                            fstate_q <= F_I;
                        end
                    end
                end
                F_L2: begin
                    if (l2_flush_done)
                        fstate_q <= F_I;
                end
                default: begin   // F_I
                    if (l1i_done_q || l1i_flush_done)
                        fstate_q <= F_IDLE;
                end
            endcase
        end
    end
    assign flush_busy = (fstate_q != F_IDLE);

    // ------------------------------------------------------------ line bridge
    npu_axi4_line_bridge #(.LINE_BYTES(LINE_BYTES)) u_bridge (
        .clk(clk), .rst_n(rst_n),
        .mem_req_valid(br_req_valid), .mem_req_ready(br_req_ready),
        .mem_req_addr(br_req_addr), .mem_req_we(br_req_we), .mem_req_wdata(br_req_wdata),
        .mem_resp_valid(br_resp_valid), .mem_resp_rdata(br_resp_rdata), .error(br_error),
        .m_axi_araddr(br_araddr), .m_axi_arlen(br_arlen), .m_axi_arsize(br_arsize),
        .m_axi_arburst(br_arburst), .m_axi_arcache(br_arcache), .m_axi_arprot(br_arprot),
        .m_axi_arvalid(br_arvalid), .m_axi_arready(br_arready),
        .m_axi_rdata(br_rdata), .m_axi_rresp(br_rresp), .m_axi_rlast(br_rlast),
        .m_axi_rvalid(br_rvalid), .m_axi_rready(br_rready),
        .m_axi_awaddr(br_awaddr), .m_axi_awlen(br_awlen), .m_axi_awsize(br_awsize),
        .m_axi_awburst(br_awburst), .m_axi_awcache(br_awcache), .m_axi_awprot(br_awprot),
        .m_axi_awvalid(br_awvalid), .m_axi_awready(br_awready),
        .m_axi_wdata(br_wdata), .m_axi_wstrb(br_wstrb), .m_axi_wlast(br_wlast),
        .m_axi_wvalid(br_wvalid), .m_axi_wready(br_wready),
        .m_axi_bresp(br_bresp), .m_axi_bvalid(br_bvalid), .m_axi_bready(br_bready)
    );

    // ------------------------------------------------------------ DMA + AXI4 arbiter
    generate
        if (HX_ENABLE_DMA != 0) begin : g_dma
            logic [31:0] dm_araddr, dm_awaddr;
            logic [7:0]  dm_arlen, dm_awlen;
            logic [2:0]  dm_arsize, dm_awsize, dm_arprot, dm_awprot;
            logic [1:0]  dm_arburst, dm_awburst, dm_rresp, dm_bresp;
            logic [3:0]  dm_arcache, dm_awcache;
            logic        dm_arvalid, dm_arready, dm_rlast, dm_rvalid, dm_rready;
            logic        dm_awvalid, dm_awready, dm_wlast, dm_wvalid, dm_wready;
            logic        dm_bvalid, dm_bready;
            logic [63:0] dm_rdata, dm_wdata;
            logic [7:0]  dm_wstrb;
            logic        dma_b_gnt;
            assign dma_b_gnt = dma_b_req && b_gnt;

            npu_tcm_dma #(.TCM_DEPTH(TCM_DEPTH), .TCM_AW(TCM_AW)) u_dma (
                .clk(clk), .rst_n(rst_n),
                .start(dma_start), .dir(dma_dir), .ddr_addr(dma_ddr_addr),
                .tcm_addr(dma_tcm_addr), .words(dma_words),
                .busy(dma_busy), .done(dma_done), .error(dma_error),
                .tcm_port_req(dma_b_req), .tcm_port_we(dma_b_we), .tcm_port_addr(dma_b_addr),
                .tcm_port_wdata(dma_b_wdata), .tcm_port_wstrb(dma_b_wstrb),
                .tcm_port_gnt(dma_b_gnt), .tcm_port_rdata(b_rdata),
                .m_axi_araddr(dm_araddr), .m_axi_arlen(dm_arlen), .m_axi_arsize(dm_arsize),
                .m_axi_arburst(dm_arburst), .m_axi_arcache(dm_arcache), .m_axi_arprot(dm_arprot),
                .m_axi_arvalid(dm_arvalid), .m_axi_arready(dm_arready),
                .m_axi_rdata(dm_rdata), .m_axi_rresp(dm_rresp), .m_axi_rlast(dm_rlast),
                .m_axi_rvalid(dm_rvalid), .m_axi_rready(dm_rready),
                .m_axi_awaddr(dm_awaddr), .m_axi_awlen(dm_awlen), .m_axi_awsize(dm_awsize),
                .m_axi_awburst(dm_awburst), .m_axi_awcache(dm_awcache), .m_axi_awprot(dm_awprot),
                .m_axi_awvalid(dm_awvalid), .m_axi_awready(dm_awready),
                .m_axi_wdata(dm_wdata), .m_axi_wstrb(dm_wstrb), .m_axi_wlast(dm_wlast),
                .m_axi_wvalid(dm_wvalid), .m_axi_wready(dm_wready),
                .m_axi_bresp(dm_bresp), .m_axi_bvalid(dm_bvalid), .m_axi_bready(dm_bready)
            );

            npu_axi4_arbiter u_axi_arb (
                .clk(clk), .rst_n(rst_n),
                .s0_axi_araddr(br_araddr), .s0_axi_arlen(br_arlen), .s0_axi_arsize(br_arsize),
                .s0_axi_arburst(br_arburst), .s0_axi_arcache(br_arcache),
                .s0_axi_arprot(br_arprot), .s0_axi_arvalid(br_arvalid),
                .s0_axi_arready(br_arready), .s0_axi_rdata(br_rdata), .s0_axi_rresp(br_rresp),
                .s0_axi_rlast(br_rlast), .s0_axi_rvalid(br_rvalid), .s0_axi_rready(br_rready),
                .s0_axi_awaddr(br_awaddr), .s0_axi_awlen(br_awlen), .s0_axi_awsize(br_awsize),
                .s0_axi_awburst(br_awburst), .s0_axi_awcache(br_awcache),
                .s0_axi_awprot(br_awprot), .s0_axi_awvalid(br_awvalid),
                .s0_axi_awready(br_awready), .s0_axi_wdata(br_wdata), .s0_axi_wstrb(br_wstrb),
                .s0_axi_wlast(br_wlast), .s0_axi_wvalid(br_wvalid), .s0_axi_wready(br_wready),
                .s0_axi_bresp(br_bresp), .s0_axi_bvalid(br_bvalid), .s0_axi_bready(br_bready),
                .s1_axi_araddr(dm_araddr), .s1_axi_arlen(dm_arlen), .s1_axi_arsize(dm_arsize),
                .s1_axi_arburst(dm_arburst), .s1_axi_arcache(dm_arcache),
                .s1_axi_arprot(dm_arprot), .s1_axi_arvalid(dm_arvalid),
                .s1_axi_arready(dm_arready), .s1_axi_rdata(dm_rdata), .s1_axi_rresp(dm_rresp),
                .s1_axi_rlast(dm_rlast), .s1_axi_rvalid(dm_rvalid), .s1_axi_rready(dm_rready),
                .s1_axi_awaddr(dm_awaddr), .s1_axi_awlen(dm_awlen), .s1_axi_awsize(dm_awsize),
                .s1_axi_awburst(dm_awburst), .s1_axi_awcache(dm_awcache),
                .s1_axi_awprot(dm_awprot), .s1_axi_awvalid(dm_awvalid),
                .s1_axi_awready(dm_awready), .s1_axi_wdata(dm_wdata), .s1_axi_wstrb(dm_wstrb),
                .s1_axi_wlast(dm_wlast), .s1_axi_wvalid(dm_wvalid), .s1_axi_wready(dm_wready),
                .s1_axi_bresp(dm_bresp), .s1_axi_bvalid(dm_bvalid), .s1_axi_bready(dm_bready),
                .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
                .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
                .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
                .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
                .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
                .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
                .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
                .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
                .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot),
                .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
                .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
                .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
                .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
                .m_axi_bready(m_axi_bready)
            );
        end else begin : g_no_dma
            assign dma_busy = 1'b0;
            assign dma_done = 1'b0;
            assign dma_error = 1'b0;
            assign dma_b_req = 1'b0;
            assign dma_b_we = 1'b0;
            assign dma_b_addr = '0;
            assign dma_b_wdata = '0;
            assign dma_b_wstrb = '0;
            assign m_axi_araddr = br_araddr;
            assign m_axi_arlen = br_arlen;
            assign m_axi_arsize = br_arsize;
            assign m_axi_arburst = br_arburst;
            assign m_axi_arcache = br_arcache;
            assign m_axi_arprot = br_arprot;
            assign m_axi_arvalid = br_arvalid;
            assign br_arready = m_axi_arready;
            assign br_rdata = m_axi_rdata;
            assign br_rresp = m_axi_rresp;
            assign br_rlast = m_axi_rlast;
            assign br_rvalid = m_axi_rvalid;
            assign m_axi_rready = br_rready;
            assign m_axi_awaddr = br_awaddr;
            assign m_axi_awlen = br_awlen;
            assign m_axi_awsize = br_awsize;
            assign m_axi_awburst = br_awburst;
            assign m_axi_awcache = br_awcache;
            assign m_axi_awprot = br_awprot;
            assign m_axi_awvalid = br_awvalid;
            assign br_awready = m_axi_awready;
            assign m_axi_wdata = br_wdata;
            assign m_axi_wstrb = br_wstrb;
            assign m_axi_wlast = br_wlast;
            assign m_axi_wvalid = br_wvalid;
            assign br_wready = m_axi_wready;
            assign br_bresp = m_axi_bresp;
            assign br_bvalid = m_axi_bvalid;
            assign m_axi_bready = br_bready;
            /* verilator lint_off UNUSEDSIGNAL */
            logic unused_dma;
            assign unused_dma = dma_start ^ dma_dir ^ (^dma_ddr_addr) ^ (^dma_tcm_addr) ^
                                (^dma_words);
            /* verilator lint_on UNUSEDSIGNAL */
        end
    endgenerate

    // ------------------------------------------------------------ registers
    npu_hx_regs #(.TCM_DEPTH(TCM_DEPTH), .TCM_AW(TCM_AW), .CAPS(CAPS),
                  .HAS_DMA(HX_ENABLE_DMA)) u_regs (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready), .s_axi_araddr(s_axi_araddr), .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .mmio_req_valid(mmio_req_valid), .mmio_req_ready(mmio_req_ready),
        .mmio_req_addr(d_req_addr[11:0]), .mmio_req_we(d_req_we),
        .mmio_req_wstrb(d_req_wstrb), .mmio_req_wdata(d_req_wdata),
        .mmio_resp_valid(mmio_resp_valid), .mmio_resp_rdata(mmio_resp_rdata),
        .tcm_req(h_req), .tcm_we(h_we), .tcm_addr(h_addr), .tcm_wdata(h_wdata),
        .tcm_wstrb(h_wstrb), .tcm_gnt(h_gnt), .tcm_rdata(b_rdata),
        .core_start(core_start), .start_pc(start_pc), .flush_start(flush_start),
        .dma_start(dma_start), .dma_dir(dma_dir), .dma_ddr_addr(dma_ddr_addr),
        .dma_tcm_addr(dma_tcm_addr), .dma_words(dma_words), .clear(clear_pulse),
        .running(running), .halted(halted), .halt_cause(halt_cause), .fault(fault),
        .fault_cause(fault_cause), .fault_pc(fault_pc),
        .vec_busy(vec_busy), .vec_illegal(vec_illegal),
        .dma_busy(dma_busy), .dma_done(dma_done), .dma_error(dma_error),
        .flush_busy(flush_busy), .bus_error(br_error),
        .cycles(core_cycles), .instret(core_instret),
        .l1i_hits(l1i_hits), .l1i_misses(l1i_misses),
        .l1d_hits(l1d_hits), .l1d_misses(l1d_misses),
        .l2_hits(l2_hits), .l2_misses(l2_misses),
        .irq(irq)
    );

    // Unused bits of the core's word-aligned data address and of the L1 I$
    // write side.
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_top;
    assign unused_top = (^d_req_addr[1:0]) ^ l1i_mreq_we ^ (^l1i_mreq_wdata);
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
