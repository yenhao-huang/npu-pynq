`timescale 1ns/1ps

// tb_npu_hx_subsystem: system test of npu_hx_subsystem (issue #136).
//
// Four subsystem configurations are instantiated (0 = all units,
// 1 = HX_ENABLE_VECTOR 0, 2 = HX_ENABLE_L2 0, 3 = HX_ENABLE_DMA 0); only the
// selected one is clocked (gated clock), the others hold still. Around the
// selected configuration:
// - an AXI4 slave memory model: 64 KB of DDR at address 0 (8192 x 64 bit),
//   single ID / in order, random AR/AW/W ready, random burst start latency
//   and random R/B valid gaps (CFG), SLVERR for any beat at or past 0x10000;
//   it checks the AXI rules the masters must follow (valid/payload stable
//   until ready, INCR, 8-byte beats, aligned, no 4 KB crossing, WLAST);
// - an AXI4-Lite host BFM with random AW/W ordering and delays and random
//   BREADY/RREADY delays, or (HFAST 1) back-to-back transactions with
//   BREADY/RREADY raised together with the address; RW issues a write and a
//   read concurrently (the Cortex-A9 may post a write and issue the next read
//   before BRESP). A monitor checks the slave side of the AXI4-Lite rules:
//   BVALID/RVALID and their payload held until BREADY/RREADY.
// The scenario comes from src/test/vectors/hx_subsystem_script.txt, written by
// src/test/vectors/generate_hx_subsystem_vectors.py (command list there):
// register map and reset values, scalar programs in DDR (sum / copy / sort),
// FLUSH, host DMA + vector kernels (INT8 saturating residual add, FP16
// a*x+y) checked against the Python model, scalar-driven DMA over MMIO with a
// mailbox handshake, the host and scalar TCM windows, faults / halts / IRQ /
// CLEAR, SLVERR -> STATUS.bus_error, I$ refill after FLUSH, CONTROL.STOP (also
// during a FLUSH; every core state must see a STOP arm, see "stop coverage") and
// resume, the board's vector kernels (src/runtime/hexagon.py). Cycle counts,
// cache counters and a per-instruction-class cycle profile of the scalar core
// (PROF, see "profiler") are printed for the report.
module tb_npu_hx_subsystem;
    localparam integer NCFG = 4;
    localparam integer DDR_WORDS = 8192;           // 64 KB
    localparam integer MAX_CYCLES = 4000000;

    logic clk = 1'b0;
    always #5 clk = ~clk;

    logic rst_n;
    integer sel;
    logic [NCFG-1:0] cen;
    integer cycle;

    // ------------------------------------------------------------ host AXI-lite (broadcast)
    logic [15:0] awaddr, araddr;
    logic        awvalid, wvalid, bready, arvalid, rready;
    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic        awready, wready, bvalid, arready, rvalid;
    logic [1:0]  bresp, rresp;
    logic [31:0] rdata;
    logic        irq;

    // ------------------------------------------------------------ memory side (muxed)
    logic [31:0] m_araddr, m_awaddr;
    logic [7:0]  m_arlen, m_awlen;
    logic [2:0]  m_arsize, m_awsize, m_arprot, m_awprot;
    logic [1:0]  m_arburst, m_awburst;
    logic [3:0]  m_arcache, m_awcache;
    logic        m_arvalid, m_awvalid, m_wvalid, m_wlast, m_rready, m_bready;
    logic [63:0] m_wdata;
    logic [7:0]  m_wstrb;
    logic        m_arready, m_awready, m_wready, m_rvalid, m_rlast, m_bvalid;
    logic [63:0] m_rdata;
    logic [1:0]  m_rresp, m_bresp;
    logic        host_fast;      // HFAST: back-to-back host transactions

    // Per-configuration DUT outputs, packed
    logic [NCFG-1:0]    d_awready, d_wready, d_bvalid, d_arready, d_rvalid, d_irq;
    logic [NCFG*2-1:0]  d_bresp, d_rresp;
    logic [NCFG*32-1:0] d_rdata;
    logic [NCFG*32-1:0] d_araddr, d_awaddr;
    logic [NCFG*8-1:0]  d_arlen, d_awlen, d_wstrb;
    logic [NCFG*3-1:0]  d_arsize, d_awsize, d_arprot, d_awprot;
    logic [NCFG*2-1:0]  d_arburst, d_awburst;
    logic [NCFG*4-1:0]  d_arcache, d_awcache;
    logic [NCFG-1:0]    d_arvalid, d_awvalid, d_wvalid, d_wlast, d_rready, d_bready;
    logic [NCFG*64-1:0] d_wdata;
    // Profiler taps per configuration
    logic [NCFG*4-1:0]  p_state;
    logic [NCFG*32-1:0] p_instr;
    logic [NCFG-1:0]    p_go, p_ivalid, p_iready, p_dvalid, p_dready, p_vready, p_stop;
    logic [NCFG*2-1:0]  p_dregion;

    genvar g;
    generate
        for (g = 0; g < NCFG; g = g + 1) begin : dut
            logic dclk;
            assign dclk = clk & cen[g];
            npu_hx_subsystem #(
                .HX_ENABLE_VECTOR((g == 1) ? 0 : 1),
                .HX_ENABLE_L2((g == 2) ? 0 : 1),
                .HX_ENABLE_DMA((g == 3) ? 0 : 1)
            ) u (
                .s_axi_aclk(dclk), .s_axi_aresetn(rst_n),
                .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid),
                .s_axi_awready(d_awready[g]),
                .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid),
                .s_axi_wready(d_wready[g]),
                .s_axi_bresp(d_bresp[g*2 +: 2]), .s_axi_bvalid(d_bvalid[g]), .s_axi_bready(bready),
                .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid),
                .s_axi_arready(d_arready[g]),
                .s_axi_rdata(d_rdata[g*32 +: 32]), .s_axi_rresp(d_rresp[g*2 +: 2]),
                .s_axi_rvalid(d_rvalid[g]), .s_axi_rready(rready),
                .m_axi_araddr(d_araddr[g*32 +: 32]), .m_axi_arlen(d_arlen[g*8 +: 8]),
                .m_axi_arsize(d_arsize[g*3 +: 3]), .m_axi_arburst(d_arburst[g*2 +: 2]),
                .m_axi_arcache(d_arcache[g*4 +: 4]), .m_axi_arprot(d_arprot[g*3 +: 3]),
                .m_axi_arvalid(d_arvalid[g]), .m_axi_arready(m_arready),
                .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp), .m_axi_rlast(m_rlast),
                .m_axi_rvalid(m_rvalid), .m_axi_rready(d_rready[g]),
                .m_axi_awaddr(d_awaddr[g*32 +: 32]), .m_axi_awlen(d_awlen[g*8 +: 8]),
                .m_axi_awsize(d_awsize[g*3 +: 3]), .m_axi_awburst(d_awburst[g*2 +: 2]),
                .m_axi_awcache(d_awcache[g*4 +: 4]), .m_axi_awprot(d_awprot[g*3 +: 3]),
                .m_axi_awvalid(d_awvalid[g]), .m_axi_awready(m_awready),
                .m_axi_wdata(d_wdata[g*64 +: 64]), .m_axi_wstrb(d_wstrb[g*8 +: 8]),
                .m_axi_wlast(d_wlast[g]), .m_axi_wvalid(d_wvalid[g]), .m_axi_wready(m_wready),
                .m_axi_bresp(m_bresp), .m_axi_bvalid(m_bvalid), .m_axi_bready(d_bready[g]),
                .irq(d_irq[g])
            );
            // Profiler taps (hierarchical, measurement only; see "profiler")
            assign p_state[g*4 +: 4] = u.u_core.state;
            assign p_instr[g*32 +: 32] = u.u_core.instr;
            assign p_go[g] = u.u_core.start && !u.u_core.running;
            assign p_ivalid[g] = u.u_core.i_req_valid;
            assign p_iready[g] = u.u_core.i_req_ready;
            assign p_dvalid[g] = u.u_core.d_req_valid;
            assign p_dready[g] = u.u_core.d_req_ready;
            assign p_dregion[g*2 +: 2] = u.d_is_ddr ? 2'd0 : u.d_is_tcm ? 2'd1 :
                                         u.d_is_mmio ? 2'd2 : 2'd3;
            assign p_vready[g] = u.u_core.vcmd_ready;
            assign p_stop[g] = u.stop_pend_q;
        end
    endgenerate

    always_comb begin
        awready = d_awready[sel];
        wready = d_wready[sel];
        bvalid = d_bvalid[sel];
        bresp = d_bresp[sel*2 +: 2];
        arready = d_arready[sel];
        rvalid = d_rvalid[sel];
        rresp = d_rresp[sel*2 +: 2];
        rdata = d_rdata[sel*32 +: 32];
        irq = d_irq[sel];
        m_araddr = d_araddr[sel*32 +: 32];
        m_arlen = d_arlen[sel*8 +: 8];
        m_arsize = d_arsize[sel*3 +: 3];
        m_arburst = d_arburst[sel*2 +: 2];
        m_arcache = d_arcache[sel*4 +: 4];
        m_arprot = d_arprot[sel*3 +: 3];
        m_arvalid = d_arvalid[sel];
        m_rready = d_rready[sel];
        m_awaddr = d_awaddr[sel*32 +: 32];
        m_awlen = d_awlen[sel*8 +: 8];
        m_awsize = d_awsize[sel*3 +: 3];
        m_awburst = d_awburst[sel*2 +: 2];
        m_awcache = d_awcache[sel*4 +: 4];
        m_awprot = d_awprot[sel*3 +: 3];
        m_awvalid = d_awvalid[sel];
        m_wdata = d_wdata[sel*64 +: 64];
        m_wstrb = d_wstrb[sel*8 +: 8];
        m_wlast = d_wlast[sel];
        m_wvalid = d_wvalid[sel];
        m_bready = d_bready[sel];
    end

    task automatic die(input [8*96-1:0] msg);
        begin
            $display("MISMATCH tb_npu_hx_subsystem cycle=%0d config=%0d: %0s", cycle, sel, msg);
            $fatal(1, "tb_npu_hx_subsystem stopped");
        end
    endtask

    // ------------------------------------------------------------ AXI4 memory model
    logic [63:0] ddr [0:DDR_WORDS-1];
    integer ready_pct, max_lat;
    // AR queue
    logic [31:0] arq_addr [0:7];
    logic [7:0]  arq_len [0:7];
    integer arq_w, arq_r, arq_n;
    // R burst in progress
    integer r_active, r_beat, r_len, r_wait;
    logic [31:0] r_addr;
    // AW queue, W beat
    logic [31:0] awq_addr [0:7];
    logic [7:0]  awq_len [0:7];
    integer awq_w, awq_r, awq_n, w_beat, w_err;
    // B queue
    logic [1:0]  bq_resp [0:7];
    integer bq_w, bq_r, bq_n;
    // protocol monitor (previous cycle)
    logic        p_arstall, p_awstall, p_wstall;
    logic [31:0] p_araddr, p_awaddr;
    logic [7:0]  p_arlen, p_awlen;
    logic [63:0] p_wdata;
    logic [7:0]  p_wstrb;
    logic        p_wlast;
    logic        p_bstall, p_rstall;
    logic [1:0]  p_bresp, p_rresp;
    logic [31:0] p_rdata;
    integer      beats_r, beats_w;

    function automatic logic chance(input integer pct);
        chance = (($urandom % 100) < pct);
    endfunction

    task automatic check_burst(input [31:0] addr, input [7:0] len, input [2:0] size,
                               input [1:0] burst);
        begin
            if (size != 3'd3) die("AXI burst size is not 8 bytes");
            if (burst != 2'b01) die("AXI burst is not INCR");
            if (addr[2:0] != 3'd0) die("AXI address is not 8-byte aligned");
            if (len > 8'd15) die("AXI burst longer than 16 beats");
            if ((addr[11:0] + ((len + 1) * 8)) > 4096) die("AXI burst crosses 4 KB");
        end
    endtask

    always @(posedge clk) begin : mem_model
        logic        rv, bv;
        logic [31:0] a;
        if (!rst_n) begin
            arq_w = 0; arq_r = 0; arq_n = 0;
            awq_w = 0; awq_r = 0; awq_n = 0;
            bq_w = 0; bq_r = 0; bq_n = 0;
            r_active = 0; r_beat = 0; r_len = 0; r_wait = 0;
            w_beat = 0; w_err = 0;
            m_arready <= 1'b0;
            m_awready <= 1'b0;
            m_wready <= 1'b0;
            m_rvalid <= 1'b0;
            m_rlast <= 1'b0;
            m_rdata <= '0;
            m_rresp <= 2'b00;
            m_bvalid <= 1'b0;
            m_bresp <= 2'b00;
            p_arstall <= 1'b0;
            p_awstall <= 1'b0;
            p_wstall <= 1'b0;
            p_bstall <= 1'b0;
            p_rstall <= 1'b0;
        end else begin
            if ((^{m_arvalid, m_awvalid, m_wvalid, m_rready, m_bready, irq,
                   bvalid, rvalid, awready, wready, arready}) === 1'bx)
                die("X on an AXI control output");
            // ---- protocol monitor: valid and payload stable until ready
            if (p_arstall && (!m_arvalid || m_araddr != p_araddr || m_arlen != p_arlen))
                die("ARVALID / AR payload changed while waiting for ARREADY");
            if (p_awstall && (!m_awvalid || m_awaddr != p_awaddr || m_awlen != p_awlen))
                die("AWVALID / AW payload changed while waiting for AWREADY");
            if (p_wstall && (!m_wvalid || m_wdata != p_wdata || m_wstrb != p_wstrb ||
                             m_wlast != p_wlast))
                die("WVALID / W payload changed while waiting for WREADY");
            // ---- AXI4-Lite slave: BVALID / RVALID and payload held until ready
            if (p_bstall && (!bvalid || bresp != p_bresp))
                die("BVALID / BRESP changed while waiting for BREADY");
            if (p_rstall && (!rvalid || rdata != p_rdata || rresp != p_rresp))
                die("RVALID / RDATA changed while waiting for RREADY");
            p_bstall <= bvalid && !bready;
            p_rstall <= rvalid && !rready;
            p_bresp <= bresp;
            p_rresp <= rresp;
            p_rdata <= rdata;
            p_arstall <= m_arvalid && !m_arready;
            p_awstall <= m_awvalid && !m_awready;
            p_wstall <= m_wvalid && !m_wready;
            p_araddr <= m_araddr;
            p_arlen <= m_arlen;
            p_awaddr <= m_awaddr;
            p_awlen <= m_awlen;
            p_wdata <= m_wdata;
            p_wstrb <= m_wstrb;
            p_wlast <= m_wlast;

            // ---- AR
            if (m_arvalid && m_arready) begin
                check_burst(m_araddr, m_arlen, m_arsize, m_arburst);
                arq_addr[arq_w] = m_araddr;
                arq_len[arq_w] = m_arlen;
                arq_w = (arq_w + 1) % 8;
                arq_n = arq_n + 1;
            end
            m_arready <= (arq_n < 7) && chance(ready_pct);

            // ---- R
            rv = m_rvalid;
            if (m_rvalid && m_rready) begin
                rv = 1'b0;
                beats_r = beats_r + 1;
                if (r_beat == r_len)
                    r_active = 0;
                else
                    r_beat = r_beat + 1;
            end
            if (!r_active && arq_n > 0) begin
                r_addr = arq_addr[arq_r];
                r_len = arq_len[arq_r];
                arq_r = (arq_r + 1) % 8;
                arq_n = arq_n - 1;
                r_beat = 0;
                r_wait = (max_lat > 0) ? ($urandom % (max_lat + 1)) : 0;
                r_active = 1;
            end
            if (r_active && !rv) begin
                if (r_wait > 0) begin
                    r_wait = r_wait - 1;
                end else if (chance(ready_pct)) begin
                    rv = 1'b1;
                    a = r_addr + r_beat * 8;
                    if (a < DDR_WORDS * 8) begin
                        m_rdata <= ddr[a[15:3]];
                        m_rresp <= 2'b00;
                    end else begin
                        m_rdata <= 64'hDEAD_BEEF_DEAD_BEEF;
                        m_rresp <= 2'b10;
                    end
                    m_rlast <= (r_beat == r_len);
                end
            end
            m_rvalid <= rv;

            // ---- W (a burst's beats follow its AW: wready only with an AW queued)
            if (m_wvalid && m_wready) begin
                beats_w = beats_w + 1;
                a = awq_addr[awq_r] + w_beat * 8;
                if (m_wlast != (w_beat == awq_len[awq_r])) die("WLAST on the wrong beat");
                if (a < DDR_WORDS * 8) begin
                    for (int i = 0; i < 8; i++)
                        if (m_wstrb[i]) ddr[a[15:3]][i*8 +: 8] = m_wdata[i*8 +: 8];
                end else begin
                    w_err = 1;
                end
                if (w_beat == awq_len[awq_r]) begin
                    bq_resp[bq_w] = w_err ? 2'b10 : 2'b00;
                    bq_w = (bq_w + 1) % 8;
                    bq_n = bq_n + 1;
                    awq_r = (awq_r + 1) % 8;
                    awq_n = awq_n - 1;
                    w_beat = 0;
                    w_err = 0;
                end else begin
                    w_beat = w_beat + 1;
                end
            end
            if (m_awvalid && m_awready) begin
                check_burst(m_awaddr, m_awlen, m_awsize, m_awburst);
                awq_addr[awq_w] = m_awaddr;
                awq_len[awq_w] = m_awlen;
                awq_w = (awq_w + 1) % 8;
                awq_n = awq_n + 1;
            end
            m_awready <= (awq_n < 7) && chance(ready_pct);
            m_wready <= (awq_n > 0) && chance(ready_pct);

            // ---- B
            bv = m_bvalid;
            if (m_bvalid && m_bready) begin
                bv = 1'b0;
                bq_r = (bq_r + 1) % 8;
                bq_n = bq_n - 1;
            end
            if (!bv && bq_n > 0 && chance(ready_pct)) begin
                bv = 1'b1;
                m_bresp <= bq_resp[bq_r];
            end
            m_bvalid <= bv;
        end
    end

    // ------------------------------------------------------------ profiler
    // Measurement only (no checks): from the scalar core's state, cycles are
    // charged to the class of the instruction in flight (latched in DECODE).
    // FETCH / IWAIT come after the retire, so they are charged to the
    // instruction whose successor was late (class "start" before the first
    // DECODE). Buckets: decode / exec / wb, mem_fetch (MEM waiting for the
    // overlapped fetch to be accepted), mem_ready (d_req_valid && !ready),
    // mem_hs (the handshake cycle), mwait by region (response latency after
    // the handshake: DDR = L1 D$, TCM window, MMIO, unmapped), vcmd_bp
    // (vcmd_valid && !vcmd_ready), vcmd (the accept cycle), vfence (waiting
    // for vec_busy), ifetch (FETCH + IWAIT). The counters restart with every
    // accepted START; PROF prints them. Cycle model reference (the core's
    // header, both ports at latency L): ALU / branch / vcmd 3 cycles, loads /
    // stores 4 + L, so mwait = L per access and every other bucket beyond
    // decode + exec + wb (+ mem_hs) is a stall.
    localparam integer NCLS = 9, NB = 16;
    localparam integer B_DEC = 0, B_EXEC = 1, B_WB = 2, B_IF = 3, B_MEMF = 4, B_MEMR = 5;
    localparam integer B_MEMH = 6, B_MWD = 7, B_MWT = 8, B_MWM = 9, B_MWU = 10, B_VBP = 11;
    localparam integer B_VCMD = 12, B_VF = 13, B_N = 14, B_CYC = 15;
    integer prof [0:NCLS*NB-1];
    integer prof_cls, prof_istall;
    logic [1:0] prof_region;
    // CONTROL.STOP coverage: core state in the cycle STOP arms
    integer stop_cov [0:15];
    logic   p_stop_d;
    initial for (int k = 0; k < 16; k++) stop_cov[k] = 0;
    always @(posedge clk) begin
        p_stop_d <= rst_n && p_stop[sel];
        if (rst_n && p_stop[sel] && !p_stop_d)
            stop_cov[p_state[sel*4 +: 4]] = stop_cov[p_state[sel*4 +: 4]] + 1;
    end

    function automatic integer instr_class(input [31:0] ins);
        case (ins[6:0])
            7'b1100011: instr_class = 1;                        // branch
            7'b1101111, 7'b1100111: instr_class = 2;            // JAL, JALR
            7'b0000011: instr_class = 3;                        // load
            7'b0100011: instr_class = 4;                        // store
            7'b0001011: instr_class = (ins[29:25] == 5'd31) ? 6 : 5;   // vfence / vcmd
            7'b1110011: instr_class = 7;                        // system
            default:    instr_class = 0;                        // ALU, LUI, AUIPC, FENCE
        endcase
    endfunction

    always @(posedge clk) begin : profiler
        integer st, b;
        st = p_state[sel*4 +: 4];
        if (!rst_n || p_go[sel]) begin
            for (int k = 0; k < NCLS * NB; k++)
                prof[k] = 0;
            prof_cls = 8;
            prof_istall = 0;
        end else if (st != 0 && st != 10 && st != 11) begin
            if (st == 3)
                prof_cls = instr_class(p_instr[sel*32 +: 32]);
            b = -1;
            case (st)
                1, 2: b = B_IF;
                3: begin b = B_DEC; prof[prof_cls*NB + B_N] = prof[prof_cls*NB + B_N] + 1; end
                4: b = B_EXEC;
                5: begin
                    if (!p_dvalid[sel]) b = B_MEMF;
                    else if (!p_dready[sel]) b = B_MEMR;
                    else begin b = B_MEMH; prof_region = p_dregion[sel*2 +: 2]; end
                end
                6: b = B_MWD + prof_region;
                7: b = B_WB;
                8: b = p_vready[sel] ? B_VCMD : B_VBP;
                9: b = B_VF;
                default: b = -1;
            endcase
            if (b >= 0) begin
                prof[prof_cls*NB + b] = prof[prof_cls*NB + b] + 1;
                prof[prof_cls*NB + B_CYC] = prof[prof_cls*NB + B_CYC] + 1;
            end
            if (p_ivalid[sel] && !p_iready[sel])
                prof_istall = prof_istall + 1;
        end
    end

    task automatic print_profile(input [8*64-1:0] tag);
        reg [8*8-1:0] cname;
        integer tot [0:NB-1];
        begin
            for (int b = 0; b < NB; b++)
                tot[b] = 0;
            for (int c = 0; c < NCLS; c++) begin
                for (int b = 0; b < NB; b++)
                    tot[b] = tot[b] + prof[c*NB + b];
                case (c)
                    0: cname = "alu"; 1: cname = "branch"; 2: cname = "jump";
                    3: cname = "load"; 4: cname = "store"; 5: cname = "vcmd";
                    6: cname = "vfence"; 7: cname = "system"; default: cname = "start";
                endcase
                if (prof[c*NB + B_CYC] != 0)
                    $display("  prof %0s %0s n=%0d cyc=%0d dec=%0d exec=%0d wb=%0d ifetch=%0d mem_fetch=%0d mem_ready=%0d mem_hs=%0d mwait_ddr=%0d mwait_tcm=%0d mwait_mmio=%0d mwait_unmapped=%0d vcmd=%0d vcmd_bp=%0d vfence=%0d",
                             tag, cname, prof[c*NB + B_N], prof[c*NB + B_CYC], prof[c*NB + B_DEC],
                             prof[c*NB + B_EXEC], prof[c*NB + B_WB], prof[c*NB + B_IF],
                             prof[c*NB + B_MEMF], prof[c*NB + B_MEMR], prof[c*NB + B_MEMH],
                             prof[c*NB + B_MWD], prof[c*NB + B_MWT], prof[c*NB + B_MWM],
                             prof[c*NB + B_MWU], prof[c*NB + B_VCMD], prof[c*NB + B_VBP],
                             prof[c*NB + B_VF]);
            end
            $display("  prof %0s total n=%0d cyc=%0d ifetch=%0d mem_fetch=%0d mem_ready=%0d mwait_ddr=%0d mwait_tcm=%0d mwait_mmio=%0d vcmd_bp=%0d vfence=%0d i_req_wait=%0d",
                     tag, tot[B_N], tot[B_CYC], tot[B_IF], tot[B_MEMF], tot[B_MEMR], tot[B_MWD],
                     tot[B_MWT], tot[B_MWM], tot[B_VBP], tot[B_VF], prof_istall);
        end
    endtask

    // ------------------------------------------------------------ host BFM
    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (cycle > MAX_CYCLES) die("watchdog: the script did not finish");
    end

    // Drive on the falling edge, sample on the rising edge. In HFAST mode a
    // transaction starts at the falling edge where the previous one ended
    // (no idle cycle) and BREADY / RREADY are high from the start.
    task automatic host_edge;
        begin
            if (!(host_fast && ($time % 10 == 0)))
                @(negedge clk);
        end
    endtask

    task automatic host_write(input [15:0] a, input [31:0] d, input [3:0] s);
        integer t0, dly_aw, dly_w, dly_b;
        begin
            t0 = cycle;
            dly_aw = host_fast ? 0 : $urandom % 3;
            dly_w = host_fast ? 0 : $urandom % 3;
            dly_b = host_fast ? 0 : $urandom % 3;
            host_edge();
            if (host_fast)
                bready = 1'b1;
            fork
                begin
                    repeat (dly_aw) @(negedge clk);
                    awaddr = a;
                    awvalid = 1'b1;
                    @(posedge clk);
                    while (!awready) @(posedge clk);
                    @(negedge clk);
                    awvalid = 1'b0;
                end
                begin
                    repeat (dly_w) @(negedge clk);
                    wdata = d;
                    wstrb = s;
                    wvalid = 1'b1;
                    @(posedge clk);
                    while (!wready) @(posedge clk);
                    @(negedge clk);
                    wvalid = 1'b0;
                end
            join
            repeat (dly_b) @(negedge clk);
            bready = 1'b1;
            @(posedge clk);
            while (!bvalid) @(posedge clk);
            if (bresp != 2'b00) die("host write BRESP is not OKAY");
            @(negedge clk);
            bready = 1'b0;
            if (cycle - t0 > 2000) die("host write took too long");
        end
    endtask

    task automatic host_read(input [15:0] a, output [31:0] d);
        integer t0, dly_ar, dly_r;
        begin
            t0 = cycle;
            dly_ar = host_fast ? 0 : $urandom % 3;
            dly_r = host_fast ? 0 : $urandom % 3;
            host_edge();
            if (host_fast)
                rready = 1'b1;
            repeat (dly_ar) @(negedge clk);
            araddr = a;
            arvalid = 1'b1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            @(negedge clk);
            arvalid = 1'b0;
            repeat (dly_r) @(negedge clk);
            rready = 1'b1;
            @(posedge clk);
            while (!rvalid) @(posedge clk);
            d = rdata;
            if (rresp != 2'b00) die("host read RRESP is not OKAY");
            @(negedge clk);
            rready = 1'b0;
            if (cycle - t0 > 2000) die("host read took too long");
        end
    endtask

    task automatic do_reset;
        begin
            @(negedge clk);
            rst_n = 1'b0;
            repeat (4) @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(negedge clk);
        end
    endtask

    // ------------------------------------------------------------ script interpreter
    integer fd, code, n, i, polls, ncmd, ncase, nt, seed;
    reg [8*16-1:0] kw;
    reg [8*64-1:0] label;
    reg [8*64-1:0] t_label [0:15];
    integer        t_start [0:15];
    reg [8*256-1:0] line;
    logic [31:0] addr, data, expv, mask, value, addr2;
    logic [3:0]  strb;
    logic        done_flag;

    initial begin
        rst_n = 1'b0;
        sel = 0;
        cen = '0;
        cycle = 0;
        awaddr = '0; araddr = '0; wdata = '0; wstrb = '0;
        awvalid = 1'b0; wvalid = 1'b0; bready = 1'b0; arvalid = 1'b0; rready = 1'b0;
        host_fast = 1'b0;
        ready_pct = 100;
        max_lat = 0;
        beats_r = 0;
        beats_w = 0;
        ncmd = 0;
        ncase = 0;
        nt = 0;
        for (i = 0; i < DDR_WORDS; i = i + 1)
            ddr[i] = 64'd0;
        seed = 32'h136;
        i = $urandom(seed);

        fd = $fopen("vectors/hx_subsystem_script.txt", "r");
        if (fd == 0)
            fd = $fopen("src/test/vectors/hx_subsystem_script.txt", "r");
        if (fd == 0) die("cannot open hx_subsystem_script.txt");

        done_flag = 1'b0;
        while (!done_flag) begin
            code = $fscanf(fd, "%s", kw);
            if (code != 1) die("unexpected end of the script");
            ncmd = ncmd + 1;
            if (kw == "#") begin
                code = $fgets(line, fd);
            end else if (kw == "CONFIG") begin
                code = $fscanf(fd, "%d", n);
                @(negedge clk);
                sel = n;
                cen = 4'b0001 << n;
                do_reset();
                $display("config %0d", n);
            end else if (kw == "CASE") begin
                code = $fscanf(fd, "%s", label);
                ncase = ncase + 1;
                $display("case %0s (cycle %0d)", label, cycle);
            end else if (kw == "CFG") begin
                code = $fscanf(fd, "%d %d", ready_pct, max_lat);
            end else if (kw == "RST") begin
                do_reset();
            end else if (kw == "MW") begin
                code = $fscanf(fd, "%h %h", addr, data);
                if (addr >= DDR_WORDS * 8) die("MW outside the DDR model");
                ddr[addr[15:3]][addr[2]*32 +: 32] = data;
            end else if (kw == "MC") begin
                code = $fscanf(fd, "%h %h", addr, data);
                value = ddr[addr[15:3]][addr[2]*32 +: 32];
                if (value !== data) begin
                    $display("MISMATCH DDR[%08h] = %08h, expected %08h", addr, value, data);
                    die("DDR content");
                end
            end else if (kw == "W") begin
                code = $fscanf(fd, "%h %h", addr, data);
                host_write(addr[15:0], data, 4'hF);
            end else if (kw == "WS") begin
                code = $fscanf(fd, "%h %h %h", addr, data, strb);
                host_write(addr[15:0], data, strb);
            end else if (kw == "HFAST") begin
                code = $fscanf(fd, "%d", n);
                host_fast = n[0];
            end else if (kw == "RW") begin
                // concurrent write and read (different registers)
                code = $fscanf(fd, "%h %h %h %h %h", addr, data, addr2, expv, mask);
                fork
                    host_write(addr[15:0], data, 4'hF);
                    host_read(addr2[15:0], value);
                join
                if ((value & mask) !== expv) begin
                    $display("MISMATCH concurrent host read [%04h] = %08h (mask %08h), expected %08h",
                             addr2[15:0], value, mask, expv);
                    die("concurrent host read");
                end
            end else if (kw == "R") begin
                code = $fscanf(fd, "%h %h %h", addr, expv, mask);
                host_read(addr[15:0], value);
                if ((value & mask) !== expv) begin
                    $display("MISMATCH host read [%04h] = %08h (mask %08h), expected %08h",
                             addr[15:0], value, mask, expv);
                    die("host register read");
                end
            end else if (kw == "P" || kw == "PN") begin
                if (kw == "P")
                    code = $fscanf(fd, "%h %h %h %d", addr, mask, expv, n);
                else
                    code = $fscanf(fd, "%h %h %d", addr, mask, n);
                polls = 0;
                host_read(addr[15:0], value);
                while ((kw == "P") ? ((value & mask) !== expv) : ((value & mask) == 0)) begin
                    polls = polls + 1;
                    if (polls > n) begin
                        $display("MISMATCH poll [%04h] = %08h (mask %08h) timed out",
                                 addr[15:0], value, mask);
                        die("poll timeout");
                    end
                    host_read(addr[15:0], value);
                end
            end else if (kw == "IRQ") begin
                code = $fscanf(fd, "%d", n);
                repeat (3) @(posedge clk);
                if (irq !== n[0]) begin
                    $display("MISMATCH irq = %b, expected %0d", irq, n);
                    die("irq level");
                end
            end else if (kw == "T0") begin
                code = $fscanf(fd, "%s", label);
                n = -1;
                for (i = 0; i < 16; i = i + 1)
                    if (i < nt && t_label[i] == label) n = i;
                if (n < 0) begin
                    if (nt >= 16) die("too many stopwatches");
                    n = nt;
                    nt = nt + 1;
                end
                t_label[n] = label;
                t_start[n] = cycle;
            end else if (kw == "T1") begin
                code = $fscanf(fd, "%s", label);
                n = -1;
                for (i = 0; i < 16; i = i + 1)
                    if (i < nt && t_label[i] == label) n = i;
                if (n < 0) die("T1 without T0");
                $display("cycles %0s = %0d", label, cycle - t_start[n]);
            end else if (kw == "SHOW") begin
                code = $fscanf(fd, "%h %s", addr, label);
                host_read(addr[15:0], value);
                $display("  %0s = %0d", label, value);
            end else if (kw == "CPY") begin
                code = $fscanf(fd, "%h %h", addr, addr2);
                host_read(addr[15:0], value);
                host_write(addr2[15:0], value, 4'hF);
            end else if (kw == "PROF") begin
                code = $fscanf(fd, "%s", label);
                print_profile(label);
            end else if (kw == "END") begin
                done_flag = 1'b1;
            end else begin
                $display("MISMATCH unknown script command %0s", kw);
                die("script syntax");
            end
        end
        $fclose(fd);
        repeat (10) @(posedge clk);
        $display("stop coverage (core state when STOP armed): fetch=%0d iwait=%0d decode=%0d exec=%0d mem=%0d mwait=%0d wb=%0d vcmd=%0d vfence=%0d",
                 stop_cov[1], stop_cov[2], stop_cov[3], stop_cov[4], stop_cov[5], stop_cov[6],
                 stop_cov[7], stop_cov[8], stop_cov[9]);
        // every state a running core can be in must have seen a STOP arm
        // (review round 3: FETCH was not reached before the STOP-during-FLUSH steps)
        for (int k = 1; k <= 9; k++)
            if (stop_cov[k] == 0) begin
                $display("MISMATCH stop coverage: no STOP armed in core state %0d", k);
                die("stop coverage");
            end
        $display("PASS tb_npu_hx_subsystem configs=%0d cases=%0d commands=%0d axi_r_beats=%0d axi_w_beats=%0d cycles=%0d",
                 NCFG, ncase, ncmd, beats_r, beats_w, cycle);
        $finish;
    end
endmodule
