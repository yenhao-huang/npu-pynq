`timescale 1ns/1ps

// npu_hx_regs: control/status registers of the Hexagon-style subsystem
// (issue #136, docs/goals/hexagon_npu/architecture.md "Host register map").
//
// Two masters reach the same register file:
// - the host, through an AXI4-Lite slave (32-bit data, 16-bit address,
//   the handshake style of npu_axi_lite_regs: AW and W are held
//   independently, BRESP/RRESP always OKAY). Offsets 0x0000-0x0FFF are the
//   registers (0x0080-0x7FFF read 0, writes ignored), 0x8000-0xFFFF is a
//   32-bit window onto TCM port B (byte offset o: TCM word o[14:3], lane
//   o[2], wstrb maps to the lane's four byte strobes). A window access past
//   the TCM (word >= TCM_DEPTH) reads 0 and ignores writes (still OKAY, so a
//   stray host access never raises a bus fault on the ARM).
// - the scalar core, through a word-oriented MMIO port (the cache CPU-side
//   protocol, offset = 12 address bits): the handshake is registered, the
//   access executes the next cycle and is answered with one mmio_resp_valid
//   the cycle after (2 cycles after the handshake). mmio_req_ready is low
//   only while a request is pending. The requester must hold mmio_req_addr /
//   we / wstrb / wdata from the handshake until mmio_resp_valid (the scalar
//   core does: CPU_HOLD contract), so they are not copied here.
//
// Ordering: one register access executes per cycle. A pending scalar access
// takes precedence; the host's register access waits one cycle (the core
// issues at most one MMIO access every few cycles, so the host never starves).
// Host reads and writes alternate when both are pending. A host TCM window
// access holds its request (tcm_req) until tcm_gnt (port B is shared with the
// DMA and the scalar window, which have priority), reads take the extra
// cycle for the TCM read latency; scalar register accesses proceed meanwhile.
//
// Register map (offsets in bytes, see architecture.md; additions marked +):
//   0x000 MAGIC        R   0x314E5848
//   0x004 VERSION      R   0x00010000
//   0x008 CAPS         R   CAPS parameter (built by the subsystem)
//   0x00C CONTROL      W   bit0 START, bit1 FLUSH, bit2 DMA_START, bit3 CLEAR
//                          (reads 0; acts when wstrb[0] is set)
//   0x010 STATUS       R   bit0 running, 1 halted, 2 fault, 3 vec_busy,
//                          4 dma_busy, 5 flush_busy, 6 vec_illegal,
//                          7 dma_error, 8 dma_done (sticky), 9 bus_error
//                          (sticky), +10 halt event (sticky), +11 fault event
//                          (sticky)
//   0x014 START_PC     RW
//   0x018 CAUSE        R   [2:0] fault_cause, [8] halt_cause
//   0x01C FAULT_PC     R
//   0x020 CYCLES       R   scalar core cycles since the last accepted START
//   0x024 INSTRET      R
//   0x028/0x02C L1I_HITS / L1I_MISSES, 0x030/0x034 L1D, 0x038/0x03C L2   R
//                          (the caches' saturating counters: cleared only by
//                          reset, not by START; read 0 without the L2)
//   0x040 DMA_DDR_ADDR RW  (reads 0 when HAS_DMA = 0, likewise 0x044/0x048)
//   0x044 DMA_TCM_ADDR RW  TCM word address (32 bits stored)
//   0x048 DMA_LEN      RW  [15:0] words, [31] dir (1 = TCM -> DDR)
//   0x04C IRQ_ENABLE   RW  bit0 halt, bit1 fault, bit2 DMA done
//   0x050-0x05C MAILBOX0-3 RW
//   0x060 VEC_CYCLES   R   cycles with vec_busy high since the last START
//
// CONTROL semantics:
//   START     pulses core_start (the core ignores it while running). When the
//             core is not running, the halt/fault events and VEC_CYCLES clear.
//   FLUSH     pulses flush_start (the subsystem sequences L1D, L2, L1I).
//   DMA_START pulses dma_start; when the DMA is idle it also clears the sticky
//             dma_done (the DMA ignores a start while busy). A DMA_TCM_ADDR
//             with bits set above TCM_AW is flagged by forcing bit 0 of the
//             DDR address the DMA sees (dma_ddr_addr), so the DMA rejects the
//             descriptor as misaligned (error + done, nothing transferred).
//   CLEAR     clears dma_done, bus_error and the halt/fault events and pulses
//             clear (vector unit illegal_clear). An event in the same cycle
//             wins over CLEAR.
// IRQ (deviation, documented in the subsystem): irq = (halt event &
// IRQ_ENABLE[0]) | (fault event & [1]) | (dma_done & [2]), registered. The
// events are sticky copies of the rising edges of halted / fault, so the host
// acknowledges an interrupt with CLEAR (STATUS.halted / fault themselves are
// core state that only the next START clears).
//
// Reset: every register (control, status, configuration, mailboxes, AXI
// handshake state) is reset to 0. The host TCM window datapath registers are
// reset too (small).
module npu_hx_regs #(
    parameter integer TCM_DEPTH = 1024,
    parameter integer TCM_AW = 10,
    parameter logic [31:0] CAPS = 32'h0000_0000,
    parameter integer HAS_DMA = 1
) (
    input  logic               clk,
    input  logic               rst_n,

    // AXI4-Lite slave (host)
    input  logic [15:0]        s_axi_awaddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [2:0]         s_axi_awprot,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic               s_axi_awvalid,
    output logic               s_axi_awready,
    input  logic [31:0]        s_axi_wdata,
    input  logic [3:0]         s_axi_wstrb,
    input  logic               s_axi_wvalid,
    output logic               s_axi_wready,
    output logic [1:0]         s_axi_bresp,
    output logic               s_axi_bvalid,
    input  logic               s_axi_bready,
    input  logic [15:0]        s_axi_araddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [2:0]         s_axi_arprot,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic               s_axi_arvalid,
    output logic               s_axi_arready,
    output logic [31:0]        s_axi_rdata,
    output logic [1:0]         s_axi_rresp,
    output logic               s_axi_rvalid,
    input  logic               s_axi_rready,

    // Scalar MMIO port (offset = addr[11:0], word aligned)
    input  logic               mmio_req_valid,
    output logic               mmio_req_ready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [11:0]        mmio_req_addr,   // [1:0] unused (word aligned)
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic               mmio_req_we,
    input  logic [3:0]         mmio_req_wstrb,
    input  logic [31:0]        mmio_req_wdata,
    output logic               mmio_resp_valid,
    output logic [31:0]        mmio_resp_rdata,

    // Host TCM window (npu_tcm port protocol through the port-B arbiter)
    output logic               tcm_req,
    output logic               tcm_we,
    output logic [TCM_AW-1:0]  tcm_addr,
    output logic [63:0]        tcm_wdata,       // from the W hold registers
    output logic [7:0]         tcm_wstrb,
    input  logic               tcm_gnt,
    input  logic [63:0]        tcm_rdata,

    // Control
    output logic               core_start,
    output logic [31:0]        start_pc,
    output logic               flush_start,
    output logic               dma_start,
    output logic               dma_dir,
    output logic [31:0]        dma_ddr_addr,
    output logic [TCM_AW-1:0]  dma_tcm_addr,
    output logic [15:0]        dma_words,
    output logic               clear,

    // Status
    input  logic               running,
    input  logic               halted,
    input  logic               halt_cause,
    input  logic               fault,
    input  logic [2:0]         fault_cause,
    input  logic [31:0]        fault_pc,
    input  logic               vec_busy,
    input  logic               vec_illegal,
    input  logic               dma_busy,
    input  logic               dma_done,
    input  logic               dma_error,
    input  logic               flush_busy,
    input  logic               bus_error,
    input  logic [31:0]        cycles,
    input  logic [31:0]        instret,
    input  logic [31:0]        l1i_hits,
    input  logic [31:0]        l1i_misses,
    input  logic [31:0]        l1d_hits,
    input  logic [31:0]        l1d_misses,
    input  logic [31:0]        l2_hits,
    input  logic [31:0]        l2_misses,

    output logic               irq
);
    localparam logic [31:0] HX_MAGIC = 32'h314E_5848;
    localparam logic [31:0] HX_VERSION = 32'h0001_0000;

    localparam logic [4:0] R_MAGIC    = 5'h00;   // word index = offset[6:2]
    localparam logic [4:0] R_VERSION  = 5'h01;
    localparam logic [4:0] R_CAPS     = 5'h02;
    localparam logic [4:0] R_CONTROL  = 5'h03;
    localparam logic [4:0] R_STATUS   = 5'h04;
    localparam logic [4:0] R_START_PC = 5'h05;
    localparam logic [4:0] R_CAUSE    = 5'h06;
    localparam logic [4:0] R_FAULT_PC = 5'h07;
    localparam logic [4:0] R_CYCLES   = 5'h08;
    localparam logic [4:0] R_INSTRET  = 5'h09;
    localparam logic [4:0] R_L1I_HIT  = 5'h0A;
    localparam logic [4:0] R_L1I_MISS = 5'h0B;
    localparam logic [4:0] R_L1D_HIT  = 5'h0C;
    localparam logic [4:0] R_L1D_MISS = 5'h0D;
    localparam logic [4:0] R_L2_HIT   = 5'h0E;
    localparam logic [4:0] R_L2_MISS  = 5'h0F;
    localparam logic [4:0] R_DMA_DDR  = 5'h10;
    localparam logic [4:0] R_DMA_TCM  = 5'h11;
    localparam logic [4:0] R_DMA_LEN  = 5'h12;
    localparam logic [4:0] R_IRQ_EN   = 5'h13;
    localparam logic [4:0] R_MBOX0    = 5'h14;
    localparam logic [4:0] R_MBOX1    = 5'h15;
    localparam logic [4:0] R_MBOX2    = 5'h16;
    localparam logic [4:0] R_MBOX3    = 5'h17;
    localparam logic [4:0] R_VEC_CYC  = 5'h18;

    localparam logic [1:0] H_IDLE = 2'd0;
    localparam logic [1:0] H_TCM  = 2'd1;   // window request waiting for gnt
    localparam logic [1:0] H_TCMD = 2'd2;   // window read data cycle

    initial begin
        if (TCM_AW < 1 || TCM_AW > 12 || TCM_DEPTH > (1 << TCM_AW))
            $fatal(1, "npu_hx_regs TCM_AW must address TCM_DEPTH words (<= 12 bits, the 32 KB window)");
    end

    function automatic [31:0] merge_wstrb(input [31:0] cur, input [31:0] wd,
                                          input [3:0] strb);
        merge_wstrb = cur;
        for (int i = 0; i < 4; i++) begin
            if (strb[i])
                merge_wstrb[i*8 +: 8] = wd[i*8 +: 8];
        end
    endfunction

    // ------------------------------------------------------------ host AXI-lite holds
    logic [15:0] aw_addr_q, ar_addr_q;
    logic        aw_hold_q, w_hold_q, ar_hold_q;
    logic [31:0] w_data_q;
    logic [3:0]  w_strb_q;
    logic [1:0]  hstate_q;
    logic        prio_rd_q;       // serve a read first when both are pending
    logic        tcm_lane_q;      // lane of the window read in flight

    // ------------------------------------------------------------ scalar MMIO hold
    logic        mm_pend_q;

    // ------------------------------------------------------------ registers
    logic [31:0] start_pc_q, dma_ddr_q, dma_tcm_q, mbox_q [0:3], vec_cyc_q;
    logic [15:0] dma_words_q;
    logic        dma_dir_q;
    logic [2:0]  irq_en_q;
    logic        dma_done_q, bus_err_q, halt_evt_q, fault_evt_q;
    logic        halted_d_q, fault_d_q;

    // ------------------------------------------------------------ access port
    // One register access per cycle: the pending scalar access, else the host.
    logic        wr_pend, rd_pend, sel_wr, host_go;
    logic        h_tcm, h_tcm_ok;     // selected host access targets the window
    /* verilator lint_off UNUSEDSIGNAL */
    logic [15:0] h_addr;              // [1:0] unused (word accesses)
    /* verilator lint_on UNUSEDSIGNAL */
    logic        acc_valid, acc_we, acc_map;
    logic [4:0]  acc_idx;
    logic [31:0] acc_wdata;
    logic [3:0]  acc_strb;
    logic [31:0] rd_word;

    assign wr_pend = aw_hold_q && w_hold_q && !s_axi_bvalid;
    assign rd_pend = ar_hold_q && !s_axi_rvalid;
    assign sel_wr = wr_pend && (!rd_pend || !prio_rd_q);
    assign h_addr = sel_wr ? aw_addr_q : ar_addr_q;
    assign h_tcm = h_addr[15];
    // TCM word = h_addr[14:3]
    assign h_tcm_ok = (32'(h_addr[14:3]) < TCM_DEPTH);
    // The host engine acts in IDLE when a request is pending.
    assign host_go = (hstate_q == H_IDLE) && (wr_pend || rd_pend);

    always_comb begin
        if (mm_pend_q) begin
            acc_valid = 1'b1;
            // the requester holds the request fields until the response
            acc_we = mmio_req_we;
            acc_map = (mmio_req_addr[11:7] == 5'd0);
            acc_idx = mmio_req_addr[6:2];
            acc_wdata = mmio_req_wdata;
            acc_strb = mmio_req_wstrb;
        end else begin
            acc_valid = host_go && !h_tcm;
            acc_we = sel_wr;
            acc_map = (h_addr[14:7] == 8'd0);
            acc_idx = h_addr[6:2];
            acc_wdata = w_data_q;
            acc_strb = w_strb_q;
        end
    end

    logic [31:0] status_word;
    assign status_word = {20'd0, fault_evt_q, halt_evt_q, bus_err_q, dma_done_q,
                          dma_error, vec_illegal, flush_busy, dma_busy, vec_busy,
                          fault, halted, running};

    always_comb begin
        rd_word = 32'd0;
        if (acc_map) begin
            case (acc_idx)
                R_MAGIC:    rd_word = HX_MAGIC;
                R_VERSION:  rd_word = HX_VERSION;
                R_CAPS:     rd_word = CAPS;
                R_STATUS:   rd_word = status_word;
                R_START_PC: rd_word = start_pc_q;
                R_CAUSE:    rd_word = {23'd0, halt_cause, 5'd0, fault_cause};
                R_FAULT_PC: rd_word = fault_pc;
                R_CYCLES:   rd_word = cycles;
                R_INSTRET:  rd_word = instret;
                R_L1I_HIT:  rd_word = l1i_hits;
                R_L1I_MISS: rd_word = l1i_misses;
                R_L1D_HIT:  rd_word = l1d_hits;
                R_L1D_MISS: rd_word = l1d_misses;
                R_L2_HIT:   rd_word = l2_hits;
                R_L2_MISS:  rd_word = l2_misses;
                R_DMA_DDR:  rd_word = dma_ddr_q;
                R_DMA_TCM:  rd_word = dma_tcm_q;
                R_DMA_LEN:  rd_word = {dma_dir_q, 15'd0, dma_words_q};
                R_IRQ_EN:   rd_word = {29'd0, irq_en_q};
                R_MBOX0:    rd_word = mbox_q[0];
                R_MBOX1:    rd_word = mbox_q[1];
                R_MBOX2:    rd_word = mbox_q[2];
                R_MBOX3:    rd_word = mbox_q[3];
                R_VEC_CYC:  rd_word = vec_cyc_q;
                default:    rd_word = 32'd0;
            endcase
        end
    end

    // Write decode of the access port
    logic acc_wr, ctl_wr;
    assign acc_wr = acc_valid && acc_we && acc_map;
    assign ctl_wr = acc_wr && (acc_idx == R_CONTROL) && acc_strb[0];

    // ------------------------------------------------------------ handshakes
    always_comb begin
        s_axi_awready = !aw_hold_q && !s_axi_bvalid;
        s_axi_wready = !w_hold_q && !s_axi_bvalid;
        s_axi_bresp = 2'b00;
        s_axi_arready = !ar_hold_q && !s_axi_rvalid;
        s_axi_rresp = 2'b00;
        mmio_req_ready = !mm_pend_q;
    end

    // Host engine: holds, window accesses, responses.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_addr_q <= '0;
            aw_hold_q <= 1'b0;
            w_data_q <= '0;
            w_strb_q <= '0;
            w_hold_q <= 1'b0;
            ar_addr_q <= '0;
            ar_hold_q <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= '0;
            hstate_q <= H_IDLE;
            prio_rd_q <= 1'b0;
            tcm_req <= 1'b0;
            tcm_we <= 1'b0;
            tcm_addr <= '0;
            tcm_lane_q <= 1'b0;
        end else begin
            if (s_axi_awvalid && s_axi_awready) begin
                aw_addr_q <= s_axi_awaddr;
                aw_hold_q <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_data_q <= s_axi_wdata;
                w_strb_q <= s_axi_wstrb;
                w_hold_q <= 1'b1;
            end
            if (s_axi_arvalid && s_axi_arready) begin
                ar_addr_q <= s_axi_araddr;
                ar_hold_q <= 1'b1;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;

            case (hstate_q)
                H_IDLE: begin
                    // A pending scalar access owns the register port this cycle
                    if (host_go && !(mm_pend_q && !h_tcm)) begin
                        prio_rd_q <= sel_wr;
                        if (h_tcm && h_tcm_ok) begin
                            tcm_req <= 1'b1;
                            tcm_we <= sel_wr;
                            tcm_addr <= h_addr[TCM_AW+2:3];
                            tcm_lane_q <= h_addr[2];
                            hstate_q <= H_TCM;
                        end else if (sel_wr) begin
                            // register write (executed below) or ignored window write
                            aw_hold_q <= 1'b0;
                            w_hold_q <= 1'b0;
                            s_axi_bvalid <= 1'b1;
                        end else begin
                            ar_hold_q <= 1'b0;
                            s_axi_rvalid <= 1'b1;
                            s_axi_rdata <= h_tcm ? 32'd0 : rd_word;
                        end
                    end
                end
                H_TCM: begin
                    if (tcm_gnt) begin
                        tcm_req <= 1'b0;
                        if (tcm_we) begin
                            aw_hold_q <= 1'b0;
                            w_hold_q <= 1'b0;
                            s_axi_bvalid <= 1'b1;
                            hstate_q <= H_IDLE;
                        end else begin
                            hstate_q <= H_TCMD;
                        end
                    end
                end
                H_TCMD: begin
                    ar_hold_q <= 1'b0;
                    s_axi_rvalid <= 1'b1;
                    s_axi_rdata <= tcm_lane_q ? tcm_rdata[63:32] : tcm_rdata[31:0];
                    hstate_q <= H_IDLE;
                end
                default: hstate_q <= H_IDLE;
            endcase
        end
    end

    // Scalar MMIO: register at the handshake, execute next cycle, answer after.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mm_pend_q <= 1'b0;
            mmio_resp_valid <= 1'b0;
            mmio_resp_rdata <= '0;
        end else begin
            mmio_resp_valid <= mm_pend_q;
            if (mm_pend_q) begin
                mmio_resp_rdata <= rd_word;
                mm_pend_q <= 1'b0;
            end else if (mmio_req_valid) begin
                mm_pend_q <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------ register file
    logic start_ok;
    assign start_ok = ctl_wr && acc_wdata[0] && !running;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pc_q <= '0;
            dma_ddr_q <= '0;
            dma_tcm_q <= '0;
            dma_words_q <= '0;
            dma_dir_q <= 1'b0;
            irq_en_q <= '0;
            for (int i = 0; i < 4; i++)
                mbox_q[i] <= '0;
            core_start <= 1'b0;
            flush_start <= 1'b0;
            dma_start <= 1'b0;
            clear <= 1'b0;
        end else begin
            core_start <= ctl_wr && acc_wdata[0];
            flush_start <= ctl_wr && acc_wdata[1];
            dma_start <= ctl_wr && acc_wdata[2] && (HAS_DMA != 0);
            clear <= ctl_wr && acc_wdata[3];
            if (acc_wr) begin
                case (acc_idx)
                    R_START_PC: start_pc_q <= merge_wstrb(start_pc_q, acc_wdata, acc_strb);
                    R_DMA_DDR: if (HAS_DMA != 0)
                        dma_ddr_q <= merge_wstrb(dma_ddr_q, acc_wdata, acc_strb);
                    R_DMA_TCM: if (HAS_DMA != 0)
                        dma_tcm_q <= merge_wstrb(dma_tcm_q, acc_wdata, acc_strb);
                    R_DMA_LEN: if (HAS_DMA != 0) begin
                        if (acc_strb[0]) dma_words_q[7:0] <= acc_wdata[7:0];
                        if (acc_strb[1]) dma_words_q[15:8] <= acc_wdata[15:8];
                        if (acc_strb[3]) dma_dir_q <= acc_wdata[31];
                    end
                    R_IRQ_EN: if (acc_strb[0]) irq_en_q <= acc_wdata[2:0];
                    R_MBOX0: mbox_q[0] <= merge_wstrb(mbox_q[0], acc_wdata, acc_strb);
                    R_MBOX1: mbox_q[1] <= merge_wstrb(mbox_q[1], acc_wdata, acc_strb);
                    R_MBOX2: mbox_q[2] <= merge_wstrb(mbox_q[2], acc_wdata, acc_strb);
                    R_MBOX3: mbox_q[3] <= merge_wstrb(mbox_q[3], acc_wdata, acc_strb);
                    default: ;
                endcase
            end
        end
    end

    // Sticky status, events, VEC_CYCLES, irq
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_done_q <= 1'b0;
            bus_err_q <= 1'b0;
            halt_evt_q <= 1'b0;
            fault_evt_q <= 1'b0;
            halted_d_q <= 1'b0;
            fault_d_q <= 1'b0;
            vec_cyc_q <= '0;
            irq <= 1'b0;
        end else begin
            halted_d_q <= halted;
            fault_d_q <= fault;
            if (ctl_wr && acc_wdata[3]) begin
                dma_done_q <= 1'b0;
                bus_err_q <= 1'b0;
                halt_evt_q <= 1'b0;
                fault_evt_q <= 1'b0;
            end
            if (start_ok) begin
                halt_evt_q <= 1'b0;
                fault_evt_q <= 1'b0;
            end
            if (ctl_wr && acc_wdata[2] && !dma_busy)
                dma_done_q <= 1'b0;
            // events win over CLEAR in the same cycle
            if (dma_done && (HAS_DMA != 0))
                dma_done_q <= 1'b1;
            if (bus_error)
                bus_err_q <= 1'b1;
            if (halted && !halted_d_q)
                halt_evt_q <= 1'b1;
            if (fault && !fault_d_q)
                fault_evt_q <= 1'b1;
            if (start_ok)
                vec_cyc_q <= '0;
            else if (vec_busy)
                vec_cyc_q <= vec_cyc_q + 32'd1;
            irq <= (halt_evt_q && irq_en_q[0]) || (fault_evt_q && irq_en_q[1]) ||
                   (dma_done_q && irq_en_q[2]);
        end
    end

    // Window write data straight from the W hold registers (held until BVALID)
    assign tcm_wdata = {w_data_q, w_data_q};
    assign tcm_wstrb = tcm_lane_q ? {w_strb_q, 4'd0} : {4'd0, w_strb_q};
    assign start_pc = start_pc_q;
    assign dma_dir = dma_dir_q;
    assign dma_words = dma_words_q;
    assign dma_tcm_addr = dma_tcm_q[TCM_AW-1:0];
    // Upper DMA_TCM_ADDR bits set: make the DMA reject the descriptor.
    assign dma_ddr_addr = {dma_ddr_q[31:1], dma_ddr_q[0] | (|dma_tcm_q[31:TCM_AW])};
endmodule
