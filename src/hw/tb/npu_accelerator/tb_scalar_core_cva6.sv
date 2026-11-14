`timescale 1ns/1ps

// System test of the CVA6 scalar core: the full 16 x 16 npu_accelerator with
// SCALAR_IMPL = "CVA6" runs the RISC-V translations of the ISA test programs
// (src/test/vectors/risc_v_npu/isa_to_rv.py) against the same behavioural AXI4
// memory as tb_scalar_core (pseudo-random ready/valid gaps, same LFSR), and
// compares every memory word with the golden image of isa_sim.py. Protocol
// checks as in tb_scalar_core: bursts of at most 16 beats that never cross a
// 128-byte line, 8-byte INCR beats, WLAST on the last beat of every write.
//
// Runs under Verilator only (CVA6 needs it): `make -C src/test sim-cva6`, with
// --timing.
// Not part of the default iverilog `make sim`.
//
// Defines / plusargs:
//   TB_IMPL_CUSTOM   build the same bench with SCALAR_IMPL = "CUSTOM" (the A
//                    side of exp/scalar_core_compare) instead of "CVA6"
//   +vectors=<file>  vector file (default build/cva6_frontend_16x16.txt, or
//                    vectors/isa_frontend_16x16.txt with TB_IMPL_CUSTOM)
//   +cases=<n>       expected case count (default 7)
//
// Per case it prints one line
//   case <name>: <jobs> jobs, <isa_cycles> cycles, first_job <c>, job_gap
//   <min>/<avg>/<max>, tail <c>, <reads> reads, <writes> writes PASS
// where first_job counts clock cycles from ISA_CONTROL.START to the first
// matrix-core START, job_gap the cycles between consecutive STARTs, and tail
// those from the last START to the end of the program (busy low).
module tb_scalar_core_cva6;
    localparam integer MEM_WORDS = 1 << 14;
    localparam integer QDEPTH = 64;
    localparam integer MAX_CYCLES = 2000000;
`ifdef TB_IMPL_CUSTOM
    localparam string IMPL = "CUSTOM";
`else
    localparam string IMPL = "CVA6";
`endif

    logic s_axi_aclk = 0, s_axi_aresetn = 0;
    logic [7:0] s_axi_awaddr = 0, s_axi_araddr = 0;
    logic [2:0] s_axi_awprot = 0, s_axi_arprot = 0;
    logic s_axi_awvalid = 0, s_axi_wvalid = 0, s_axi_bready = 0;
    logic s_axi_arvalid = 0, s_axi_rready = 0;
    logic [31:0] s_axi_wdata = 0;
    logic [3:0] s_axi_wstrb = 0;
    logic s_axi_awready, s_axi_wready, s_axi_bvalid, s_axi_arready, s_axi_rvalid;
    logic [1:0] s_axi_bresp, s_axi_rresp;
    logic [31:0] s_axi_rdata;
    logic irq;

    logic [31:0] m_axi_araddr, m_axi_awaddr;
    logic [7:0] m_axi_arlen, m_axi_awlen;
    logic [2:0] m_axi_arsize, m_axi_awsize, m_axi_arprot, m_axi_awprot;
    logic [1:0] m_axi_arburst, m_axi_awburst;
    logic [3:0] m_axi_arcache, m_axi_awcache;
    logic m_axi_arvalid, m_axi_arready, m_axi_rlast, m_axi_rvalid, m_axi_rready;
    logic [63:0] m_axi_rdata, m_axi_wdata;
    logic [1:0] m_axi_rresp, m_axi_bresp;
    logic m_axi_awvalid, m_axi_awready, m_axi_wlast, m_axi_wvalid, m_axi_wready;
    logic [7:0] m_axi_wstrb;
    logic m_axi_bvalid, m_axi_bready;

    npu_accelerator #(
        .ROWS(16), .COLUMNS(16), .MAX_K(256), .IN_BYTES(8), .SCALAR_IMPL(IMPL)
    ) dut (.*);
    always #5 s_axi_aclk = ~s_axi_aclk;

    logic [63:0] mem [0:MEM_WORDS-1];
    logic [63:0] expected [0:MEM_WORDS-1];

    // ------------------------------------------------- AXI memory model
    logic [31:0] lfsr = 32'h1;
    always @(posedge s_axi_aclk) lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

    logic [31:0] ar_q_addr [0:QDEPTH-1];
    logic [7:0] ar_q_len [0:QDEPTH-1];
    integer ar_head = 0, ar_tail = 0, r_beat = 0;
    logic [31:0] aw_q_addr [0:QDEPTH-1];
    logic [7:0] aw_q_len [0:QDEPTH-1];
    integer aw_head = 0, aw_tail = 0, w_beat = 0, b_pending = 0;
    integer reads = 0, writes = 0;
    integer index, lane;

    task automatic fail(input string message);
        begin
            $display("FAIL tb_scalar_core_cva6 (%0s): %s", IMPL, message);
            $fatal(1);
        end
    endtask

    // As in tb_scalar_core: no START into a full matrix-core queue, and only
    // legal shapes reach the controller.
    always @(posedge s_axi_aclk) begin
        if (s_axi_aresetn && dut.u_matrix_core.start_pulse) begin
            if (!dut.u_matrix_core.status_accept)
                fail("scalar core issued START to a full matrix-core queue");
            if (dut.u_matrix_core.cfg_m == 0 || dut.u_matrix_core.cfg_m > 16 ||
                dut.u_matrix_core.cfg_n == 0 || dut.u_matrix_core.cfg_n > 16 ||
                dut.u_matrix_core.cfg_k == 0 || dut.u_matrix_core.cfg_k > 256)
                fail("scalar core issued an out-of-range job shape");
        end
    end

    task automatic check_burst(input [31:0] addr, input [7:0] len, input [2:0] size, input [1:0] burst);
        begin
            if (len > 8'd15) fail("burst longer than 16 beats");
            if (size != 3'd3 || burst != 2'b01) begin
                $display("addr 0x%08x len %0d size %0d burst %0d", addr, len, size, burst);
                fail("burst is not 8-byte INCR");
            end
            if (addr[2:0] != 3'd0) fail("unaligned burst address");
            if ({1'b0, addr[6:3]} + {1'b0, len[3:0]} > 5'd15) fail("burst crosses a 128-byte line");
            if ((addr >> 3) + 32'(len) >= MEM_WORDS) begin
                $display("addr 0x%08x len %0d", addr, len);
                fail("burst outside the memory model");
            end
        end
    endtask

    assign m_axi_arready = lfsr[3] || lfsr[9];
    assign m_axi_awready = lfsr[5] || lfsr[11];
    assign m_axi_wready = lfsr[7] || lfsr[13] || lfsr[2];
    assign m_axi_rvalid = (ar_head != ar_tail) && (lfsr[4] || lfsr[12] || lfsr[17]);
    assign m_axi_rdata = mem[14'((ar_q_addr[ar_head % QDEPTH] >> 3) + r_beat)];
    assign m_axi_rlast = (r_beat == 32'(ar_q_len[ar_head % QDEPTH]));
    assign m_axi_rresp = 2'b00;
    assign m_axi_bvalid = (b_pending > 0) && (lfsr[6] || lfsr[15]);
    assign m_axi_bresp = 2'b00;

    always @(posedge s_axi_aclk) begin
        if (m_axi_arvalid && m_axi_arready) begin
            check_burst(m_axi_araddr, m_axi_arlen, m_axi_arsize, m_axi_arburst);
            if (ar_tail - ar_head >= QDEPTH) fail("read queue overflow");
            ar_q_addr[ar_tail % QDEPTH] <= m_axi_araddr;
            ar_q_len[ar_tail % QDEPTH] <= m_axi_arlen;
            ar_tail <= ar_tail + 1;
            reads <= reads + 1;
        end
        if (m_axi_rvalid && m_axi_rready) begin
            if (m_axi_rlast) begin
                r_beat <= 0;
                ar_head <= ar_head + 1;
            end else begin
                r_beat <= r_beat + 1;
            end
        end
        if (m_axi_awvalid && m_axi_awready) begin
            check_burst(m_axi_awaddr, m_axi_awlen, m_axi_awsize, m_axi_awburst);
            aw_q_addr[aw_tail % QDEPTH] <= m_axi_awaddr;
            aw_q_len[aw_tail % QDEPTH] <= m_axi_awlen;
            aw_tail <= aw_tail + 1;
            writes <= writes + 1;
        end
        if (m_axi_wvalid && m_axi_wready) begin
            if (aw_head == aw_tail) fail("write data before its address");
            index = 32'(aw_q_addr[aw_head % QDEPTH] >> 3) + w_beat;
            for (lane = 0; lane < 8; lane = lane + 1)
                if (m_axi_wstrb[lane]) mem[index][lane*8 +: 8] <= m_axi_wdata[lane*8 +: 8];
            if (m_axi_wlast != (w_beat == 32'(aw_q_len[aw_head % QDEPTH]))) fail("WLAST misplaced");
            if (m_axi_wlast) begin
                w_beat <= 0;
                aw_head <= aw_head + 1;
            end else begin
                w_beat <= w_beat + 1;
            end
        end
        b_pending <= b_pending + ((m_axi_wvalid && m_axi_wready && m_axi_wlast) ? 1 : 0)
            - ((m_axi_bvalid && m_axi_bready) ? 1 : 0);
    end

    // ------------------------------------------- cycle breakdown probes
    // Clock cycles from ISA_CONTROL.START to the first matrix-core START,
    // between STARTs, and from the last START to the end of the program.
    integer cyc = 0, t_start = 0, t_first = -1, t_last = -1, gap_min = 0, gap_max = 0, gap_sum = 0;
    integer job_starts = 0;
    always @(posedge s_axi_aclk) begin
        cyc <= cyc + 1;
        if (s_axi_aresetn && dut.isa_start_pulse) begin
            t_start <= cyc;
            t_first <= -1;
            t_last <= -1;
            job_starts <= 0;
            gap_min <= 0; gap_max <= 0; gap_sum <= 0;
        end else if (s_axi_aresetn && dut.job_start) begin
            if (t_first < 0) t_first <= cyc;
            else begin
                gap_sum <= gap_sum + (cyc - t_last);
                if (job_starts == 1 || cyc - t_last < gap_min) gap_min <= cyc - t_last;
                if (cyc - t_last > gap_max) gap_max <= cyc - t_last;
            end
            t_last <= cyc;
            job_starts <= job_starts + 1;
        end
    end
    integer t_end = 0;
    always @(negedge dut.isa_busy) t_end = cyc;

    // CVA6 only: where the firmware's time goes. first_fetch = cycles from
    // START to CVA6's first read on the DDR port (reset release + cache init);
    // fetches = CVA6 DDR reads (I$ line fills; the programs touch no DDR data);
    // mmio_wr / mmio_rd = accesses to cva6_npu_mmio; mmio_wr_lat = average
    // cycles from MMIO AW handshake to its B handshake.
    integer t_fetch = -1, fetches = 0, mmio_wr = 0, mmio_rd = 0, mmio_wr_cycles = 0;
`ifndef TB_IMPL_CUSTOM
    logic mmio_wr_open = 0;
    always @(posedge s_axi_aclk) begin
        if (s_axi_aresetn && dut.isa_start_pulse) begin
            t_fetch <= -1; fetches <= 0; mmio_wr <= 0; mmio_rd <= 0; mmio_wr_cycles <= 0;
        end else if (s_axi_aresetn) begin
            if (dut.u_scalar_core.g_cva6.u_core.mux_slv_req[0].ar_valid &&
                dut.u_scalar_core.g_cva6.u_core.mux_slv_resp[0].ar_ready) begin
                fetches <= fetches + 1;
                if (t_fetch < 0) t_fetch <= cyc - t_start;
            end
            if (dut.u_scalar_core.g_cva6.u_core.demux_req[1].aw_valid &&
                dut.u_scalar_core.g_cva6.u_core.mmio_awready) begin
                mmio_wr <= mmio_wr + 1;
                mmio_wr_open <= 1;
            end
            if (dut.u_scalar_core.g_cva6.u_core.mmio_bvalid &&
                dut.u_scalar_core.g_cva6.u_core.demux_req[1].b_ready)
                mmio_wr_open <= 0;
            if (mmio_wr_open) mmio_wr_cycles <= mmio_wr_cycles + 1;
            if (dut.u_scalar_core.g_cva6.u_core.demux_req[1].ar_valid &&
                dut.u_scalar_core.g_cva6.u_core.mmio_arready)
                mmio_rd <= mmio_rd + 1;
        end
    end
`endif

    // ------------------------------------------------------ control path
    task automatic axi_write(input [7:0] address, input [31:0] value);
        begin
            @(negedge s_axi_aclk);
            s_axi_awaddr = address; s_axi_awvalid = 1;
            s_axi_wdata = value; s_axi_wstrb = 4'hf; s_axi_wvalid = 1;
            while (!(s_axi_awready && s_axi_wready)) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axi_awvalid = 0; s_axi_wvalid = 0;
            wait (s_axi_bvalid);
            s_axi_bready = 1;
            @(posedge s_axi_aclk); #1;
            s_axi_bready = 0;
        end
    endtask

    task automatic axi_read(input [7:0] address, output [31:0] value);
        begin
            @(negedge s_axi_aclk);
            s_axi_araddr = address; s_axi_arvalid = 1;
            while (!s_axi_arready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk); s_axi_arvalid = 0;
            wait (s_axi_rvalid);
            value = s_axi_rdata;
            s_axi_rready = 1;
            @(posedge s_axi_aclk); #1;
            s_axi_rready = 0;
        end
    endtask

    integer file, words, prog_len, data_base, want_error, want_jobs, i, cases, status_code;
    integer want_cases, polls, start_cycle, failures, mismatches;
    string vectors, tag, name;
    logic [31:0] value, cycles_csr, jobs_csr, error_csr;
    logic [63:0] word;
    logic ok;

    initial begin
`ifdef TB_IMPL_CUSTOM
        vectors = "vectors/isa_frontend_16x16.txt";
`else
        vectors = "build/cva6_frontend_16x16.txt";
`endif
        if (!$value$plusargs("vectors=%s", vectors)) ;
        if (!$value$plusargs("cases=%d", want_cases)) want_cases = 7;
        file = $fopen(vectors, "r");
        if (file == 0) fail({"cannot open ", vectors});
        status_code = $fgets(tag, file);  // header comment

        repeat (5) @(posedge s_axi_aclk);
        s_axi_aresetn = 1;
        axi_read(8'h04, value);
        if (value != 32'h00020000) fail("VERSION is not 2.0");
        axi_read(8'h08, value);
        $display("impl %0s: CAPABILITIES 0x%08x", IMPL, value);
        if (value[6] != 1'b1) fail("CAPABILITIES lacks ISA_FRONTEND");
        if (value[7] != (IMPL == "CVA6")) fail("CAPABILITIES bit 7 (SCALAR_CVA6) wrong");

        cases = 0;
        failures = 0;
        status_code = $fscanf(file, "%s %d %d %d %d %d %s\n", tag, words, prog_len,
                              data_base, want_error, want_jobs, name);
        while (words != 0) begin
            for (i = 0; i < MEM_WORDS; i = i + 1) begin
                mem[i] = 64'd0;
                expected[i] = 64'd0;
            end
            for (i = 0; i < words; i = i + 1) begin
                status_code = $fscanf(file, "%h\n", word);
                mem[i] = word;
            end
            for (i = 0; i < words; i = i + 1) begin
                status_code = $fscanf(file, "%h\n", word);
                expected[i] = word;
            end

            axi_write(8'h48, 32'd0);           // PROG_ADDR
            axi_write(8'h4c, prog_len);        // PROG_LEN
            axi_write(8'h50, data_base);       // DATA_BASE
            axi_write(8'h40, 32'd1);           // ISA_CONTROL.START
            start_cycle = cyc;
            polls = 0;
            do begin
                axi_read(8'h44, value);
                polls = polls + 1;
                if (cyc - start_cycle > MAX_CYCLES) begin
                    $display("case %0s: STATUS 0x%08x after %0d cycles", name, value, cyc - start_cycle);
                    fail("program did not finish");
                end
            end while (value[3] || value[0]); // BUSY or RUNNING

            ok = 1;
            axi_read(8'h44, value);
            axi_read(8'h54, error_csr);
            if (want_error == 0) begin
                if (value[1] !== 1'b1 || value[2] !== 1'b0) begin
                    $display("case %0s: not DONE without error (STATUS 0x%08x, ISA_ERROR 0x%02x)",
                             name, value, error_csr[7:0]);
                    ok = 0;
                end
            end else begin
                if (value[2] !== 1'b1 || error_csr[7:0] != want_error[7:0]) begin
                    $display("case %0s: STATUS 0x%08x ISA_ERROR 0x%02x, expected error 0x%02x",
                             name, value, error_csr[7:0], want_error[7:0]);
                    ok = 0;
                end
            end
            axi_read(8'h5c, jobs_csr);
            if (jobs_csr != want_jobs) begin
                $display("case %0s: ISA_JOBS=%0d expected %0d", name, jobs_csr, want_jobs);
                ok = 0;
            end
            mismatches = 0;
            for (i = 0; i < MEM_WORDS; i = i + 1)
                if (mem[i] !== expected[i]) begin
                    if (mismatches < 4)
                        $display("case %0s: word %0d (byte 0x%0x): got %016x expected %016x",
                                 name, i, i * 8, mem[i], expected[i]);
                    mismatches = mismatches + 1;
                end
            if (mismatches != 0) begin
                $display("case %0s: %0d memory words differ", name, mismatches);
                ok = 0;
            end
            axi_read(8'h60, cycles_csr);
            $display("case %0s: %0d jobs, %0d cycles, first_job %0d, job_gap %0d/%0d/%0d, tail %0d, %0d reads, %0d writes, prog %0d words %0s",
                     name, jobs_csr, cycles_csr,
                     (t_first < 0) ? -1 : t_first - t_start,
                     gap_min, (job_starts > 1) ? gap_sum / (job_starts - 1) : 0, gap_max,
                     (t_last < 0) ? -1 : t_end - t_last,
                     reads, writes, prog_len, ok ? "PASS" : "MISMATCH");
            if (IMPL == "CVA6")
                $display("cva6 %0s: first_fetch %0d, fetches %0d, mmio_wr %0d, mmio_rd %0d, mmio_wr_lat %0d.%0d",
                         name, t_fetch, fetches, mmio_wr, mmio_rd,
                         (mmio_wr > 0) ? mmio_wr_cycles / mmio_wr : 0,
                         (mmio_wr > 0) ? (mmio_wr_cycles * 10 / mmio_wr) % 10 : 0);
            if (!ok) failures = failures + 1;
            cases = cases + 1;
            status_code = $fscanf(file, "%s %d %d %d %d %d %s\n", tag, words, prog_len,
                                  data_base, want_error, want_jobs, name);
        end
        if (cases != want_cases) fail("unexpected case count");
        if (failures != 0) begin
            $display("FAIL tb_scalar_core_cva6 (%0s): %0d of %0d programs mismatch", IMPL, failures, cases);
            $fatal(1);
        end
        $display("PASS tb_scalar_core_cva6 (%0s): %0d programs", IMPL, cases);
        $finish;
    end
endmodule
