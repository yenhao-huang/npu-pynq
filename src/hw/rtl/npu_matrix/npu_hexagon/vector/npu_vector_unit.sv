`timescale 1ns/1ps

// npu_vector_unit: SIMD vector unit of the Hexagon-style subsystem (issue #136).
//
// NREGS vector registers of VLEN bits, lanes INT8 / INT16 / INT32 / FP16
// (cmd_type 0..3; 8 / 4 / 2 / 4 lanes for VLEN = 64), load/store to the TCM.
// Commands arrive from the scalar core (custom-0) on a valid/ready port.
//
// Opcodes (vd, vs1, vs2 are register numbers, "lane op" applies per lane):
//   0 VLD    vd <- TCM[addr]                       (type ignored)
//   1 VST    TCM[addr] <- vs1                      (type ignored, all bytes)
//   2 VSPLAT every lane of vd <- scalar truncated to the lane width
//   3 VADD   vd <- vs1 + vs2    int: saturating; FP16: npu_fp16_add (RNE)
//   4 VSUB   vd <- vs1 - vs2    int: saturating; FP16: npu_fp16_add sub=1
//   5 VMUL   vd <- vs1 * vs2    int: low half (wraps); FP16: npu_fp16_mul
//   6 VMAX   vd <- max(vs1, vs2) signed int; FP16 per the numeric contract
//   7 VMIN   vd <- min(vs1, vs2) (NaN -> 0x7E00, max(-0,+0)=+0, min = -0)
//   8 VRELU  vd <- max(vs1, 0)  same rules as VMAX with b = +0, so
//                               relu(-0) = +0 and relu(NaN) = 0x7E00
//   9 VSRA   vd <- (vs1 + (1 << (s-1))) >>> s, s = scalar[4:0], int only:
//                               s = 0 is a copy, s >= lane width gives 0
//                               (golden sra_round), never saturates
//  10 VMOV   vd <- vs1                             (type ignored)
//  11..31, and VSRA with type FP16: illegal. The command is consumed and
//  dropped (no architectural effect) and the sticky `illegal` output is set.
//  `illegal` clears on reset or on an illegal_clear pulse; a new illegal
//  command in the same cycle as illegal_clear wins (the flag stays set).
//
// Organisation: each command is executed as VLEN/32 slice operations of 32
// bits (one slice per clock), so the datapath is one 32-bit slice wide: two
// npu_fp16_add, two npu_fp16_mul (2 DSP), npu_vector_imul (3-4 DSP) and
// npu_vector_alu. The register file is NREGS*VLEN/32 words of 32 bits with
// two asynchronous read ports and one write port (LUTRAM). VLEN = 64 (the
// subsystem configuration) and VLEN = 128 are verified (tb_npu_vector_unit
// runs one instance of each); any VLEN = 32 * 2^k >= 64 elaborates.
//
// Pipeline (every slice operation has the same latency, so register writes
// happen in issue order and WAW/WAR need no checks):
//   ISSUE  read vs1/vs2 slice from the register file (or the VSPLAT value),
//          register the operands. VLD issues its TCM read here. The VSRA
//          shift amount is clamped to the lane width and decremented here
//          (from the registered command), so the ALU's X1 stage only shifts.
//   X1     ALU / IMUL / FP16 units start; VLD data arrives from the TCM;
//          VST slices are collected into the store buffer.
//   X2, X3 ALU, IMUL (latency 2) and FP16 MUL (latency 2) finish at X3.
//   X4     FP16 ADD (latency 3) finishes; the slice is written to the
//          register file at the end of X4.
// RAW hazards: a slice operation issues only when no operation in X1..X4
// writes the same register slice (a dependent command waits for the write,
// no forwarding). TCM ordering: a VST writes the TCM from a one-entry store
// buffer (loaded when its last slice is in X1, written when tcm_gnt is
// high); a VLD or another VST issues only when no VST slice is in X1 and the
// buffer is empty, so a VLD after a VST to the same address returns the
// stored word.
// Port A of npu_tcm returns read data exactly one cycle after a granted
// request; tcm_req/tcm_addr/tcm_we/tcm_wdata are held until tcm_gnt and
// never depend combinationally on tcm_gnt, so tcm_gnt may be a combinational
// function of tcm_req (npu_tcm port A: gnt = req) without a loop.
//
// Command queue: one command executes (cur), one waits in a buffer; cmd_ready
// is high whenever the buffer is empty (registered, independent of
// cmd_valid). An accepted command goes straight to cur when cur is free.
//
// Throughput (VLEN = 64, two slices per command, measured in
// tb_npu_vector_unit): independent commands issue every 2 cycles; a command
// that reads the previous command's result issues 5 cycles after it; a VLD
// or VST after a VST issues 4 cycles after it (store buffer drain); a VST of
// a VLD's destination issues 5 cycles after the VLD. A slice issued in cycle
// t is written at the end of cycle t+4 (readable by an issue in t+5); a
// VST's TCM write happens in cycle t+3 after its first slice issue at t (if
// tcm_gnt is high). An illegal command is dropped in one cycle. The scalar
// core sends at most one vector command every 3 cycles (npu_scalar_core
// cycle model), so independent commands never stall it; a chain of dependent
// commands does once the two-entry queue is full.
// End-to-end (tb_npu_vector_unit paced kernels, commands at the scalar
// core's rate, cycles to the final busy fall): INT8 residual add over 8
// words (vld, vld, vadd, vst) 207 as a rolled loop, 130 unrolled with the
// addresses already in registers, 100 software-pipelined; FP16 axpy over 16
// words (vld, vld, vmul, vadd, vst) 429 / 309 / 247. Rolled loops are bound
// by the scalar core (pointer bumps and the branch hide the 5-cycle
// dependency), and the software-pipelined order reaches the scalar bound
// (3 cycles per command). Evaluated and not implemented: X4 -> issue
// forwarding (dependent 5 -> 4 cycles, about +80 LUT = +4.5%: loops 207 ->
// 205 and 429 -> 427, unrolled 130 -> 114 and 309 -> 262), and letting a
// VLD pass a VST in X1 to a different address (VST -> VLD 4 -> 2 cycles:
// loops unchanged, unrolled 130 -> 116 and 309 -> 279). Neither gains more
// than 1% on the loops; the unrolled gains need code that keeps every
// address in its own scalar register, and reordering gets there for free.
//
// busy is high while any accepted command has not finished its architectural
// effect (register write, TCM write, or for an illegal command the cycle in
// which it is dropped); it rises the cycle after the cmd_valid && cmd_ready
// handshake and falls the cycle after the last register file / TCM write
// (VST slices no longer count once they leave X1), so the scalar core's
// VFENCE is exact.
//
// Combinational paths: no input reaches an output combinationally. Every
// output is a register or a function of registers only (tcm_wstrb is
// constant; rst_n clears the registers asynchronously), checked by a
// structural dependency trace of the flattened Yosys netlist, so the unit
// chains with npu_tcm (gnt = req), the scalar core and the subsystem without
// loops. Inputs reach only registers: cmd_* -> buf / cur, tcm_gnt -> issue
// (slice counter, X1 enables, cur / buf loading), tcm_rdata -> ALU pass
// operand -> ALU X2 register and ld_buf, illegal_clear -> illegal.
//
// Power: buf loads only when a command arrives while cur is busy, cur only
// when a command moves in, the X1 fields and operands only on issue, the
// store / load buffers only for VST / VLD slices. The ALU, IMUL and FP16
// inputs therefore hold still while the unit is idle or stalled, and the
// registers behind them (which have no enable) reload unchanged values.
// Not done: operand isolation per unit (the FP16 units and the IMUL also
// see the operands of operations they do not execute; separate gated
// operand registers would cost about 64 FF per unit).
// Deviations from architecture.md (additions only): tcm_wstrb (all ones on
// every request, for a direct npu_tcm port A connection) and the
// illegal_clear input (subsystem CONTROL.CLEAR); cmd_addr is driven by the
// subsystem from the scalar operand (vcmd_scalar[TCM_AW-1:0]).
module npu_vector_unit #(
    parameter integer VLEN = 64,
    parameter integer NREGS = 8,
    parameter integer TCM_AW = 10
) (
    input  logic                clk,
    input  logic                rst_n,
    // Command port (one command per cmd_valid && cmd_ready)
    input  logic                cmd_valid,
    output logic                cmd_ready,
    input  logic [4:0]          cmd_op,
    input  logic [1:0]          cmd_type,
    input  logic [2:0]          cmd_vd,
    input  logic [2:0]          cmd_vs1,
    input  logic [2:0]          cmd_vs2,
    input  logic [TCM_AW-1:0]   cmd_addr,
    input  logic [31:0]         cmd_scalar,
    output logic                busy,
    output logic                illegal,
    input  logic                illegal_clear,
    // TCM port A
    output logic                tcm_req,
    output logic                tcm_we,
    output logic [TCM_AW-1:0]   tcm_addr,
    output logic [VLEN-1:0]     tcm_wdata,
    output logic [VLEN/8-1:0]   tcm_wstrb,
    input  logic                tcm_gnt,
    input  logic [VLEN-1:0]     tcm_rdata
);
    localparam integer NSLICE = VLEN / 32;
    localparam integer SW = (NSLICE > 1) ? $clog2(NSLICE) : 1;
    localparam integer RFD = NREGS * NSLICE;

    localparam [4:0] OP_VLD    = 5'd0;
    localparam [4:0] OP_VST    = 5'd1;
    localparam [4:0] OP_VSPLAT = 5'd2;
    localparam [4:0] OP_VADD   = 5'd3;
    localparam [4:0] OP_VSUB   = 5'd4;
    localparam [4:0] OP_VMUL   = 5'd5;
    localparam [4:0] OP_VMAX   = 5'd6;
    localparam [4:0] OP_VMIN   = 5'd7;
    localparam [4:0] OP_VRELU  = 5'd8;
    localparam [4:0] OP_VSRA   = 5'd9;
    localparam [4:0] OP_VMOV   = 5'd10;

    localparam [1:0] T_INT8  = 2'd0;
    localparam [1:0] T_INT32 = 2'd2;
    localparam [1:0] T_FP16  = 2'd3;

    localparam [2:0] ALU_PASS = 3'd0;
    localparam [2:0] ALU_ADD  = 3'd1;
    localparam [2:0] ALU_SUB  = 3'd2;
    localparam [2:0] ALU_MAX  = 3'd3;
    localparam [2:0] ALU_MIN  = 3'd4;
    localparam [2:0] ALU_SRA  = 3'd5;

    // Result source of a slice operation.
    localparam [1:0] R_ALU   = 2'd0;
    localparam [1:0] R_IMUL  = 2'd1;
    localparam [1:0] R_FPMUL = 2'd2;
    localparam [1:0] R_FPADD = 2'd3;

    initial begin
        if (VLEN < 64 || (VLEN % 32) != 0 || (NSLICE & (NSLICE - 1)) != 0)
            $fatal(1, "npu_vector_unit requires VLEN = 32 * 2^k, VLEN >= 64");
        if (NREGS != 8)
            $fatal(1, "npu_vector_unit requires NREGS = 8 (3-bit register fields)");
    end

    // ------------------------------------------------------------ command queue
    logic                buf_valid;
    logic [4:0]          buf_op;
    logic [1:0]          buf_type;
    logic [2:0]          buf_vd, buf_vs1, buf_vs2;
    logic [TCM_AW-1:0]   buf_addr;
    logic [31:0]         buf_scalar;

    logic                cur_valid;
    logic [4:0]          cur_op;
    logic [1:0]          cur_type;
    logic [2:0]          cur_vd, cur_vs1, cur_vs2;
    logic [TCM_AW-1:0]   cur_addr;
    logic [31:0]         cur_scalar;
    logic [SW-1:0]       cur_slice;

    logic                accept, cur_free, cur_done, issue;

    assign cmd_ready = !buf_valid;
    assign accept = cmd_valid && cmd_ready;

    // ------------------------------------------------------------ pipeline state
    logic                x1_v, x2_v, x3_v, x4_v;
    logic                x1_wen, x2_wen, x3_wen, x4_wen;
    logic [2:0]          x1_vd, x2_vd, x3_vd, x4_vd;
    logic [SW-1:0]       x1_sl, x2_sl, x3_sl, x4_sl;
    logic [1:0]          x1_rsel, x2_rsel, x3_rsel, x4_rsel;
    logic                x1_vld, x1_vst, x1_last, x1_sub;
    logic [1:0]          x1_type;
    logic [2:0]          x1_fn;
    logic [4:0]          x1_shm1;
    logic                x1_shz;
    logic [31:0]         x1_a, x1_b;

    logic                st_x1;        // a VST slice is in X1 (not yet in the buffer)
    logic [VLEN-33:0]    st_buf;       // VST slices 0..NSLICE-2
    logic                sb_valid;     // store buffer: one TCM write
    logic [TCM_AW-1:0]   sb_addr;
    logic [31:0]         sb_hi;        // last VST slice; slices below stay in st_buf
    logic [VLEN-33:0]    ld_buf;       // VLD slices 1..NSLICE-1

    // ------------------------------------------------------------ decode (cur)
    logic cur_ill, uses_vs1, uses_vs2, cur_vld, cur_vst, cur_first, cur_last;
    logic hazard, mem_ok, vld_req;
    logic [2:0] cur_fn;
    logic [1:0] cur_rsel;
    logic [31:0] splat;
    logic [4:0] sra_shm1;

    // TCM ordering: VLD and VST start only with no store in flight. A VST
    // whose last slice has issued is in X1 or in the store buffer (the slices
    // of the VST in cur itself never block it). vld_req drives tcm_req and
    // must not depend on tcm_gnt; it is kept out of the always_comb below
    // (which reads tcm_gnt) so that tools that analyse a block as one node
    // (Verilator UNOPTFLAT) see no loop when tcm_gnt is a combinational
    // function of tcm_req, as with npu_tcm's gnt = req.
    assign st_x1 = x1_v && x1_vst;
    assign vld_req = cur_valid && (cur_op == OP_VLD) && (cur_slice == '0) && !st_x1 && !sb_valid;

    always_comb begin
        cur_ill = (cur_op > OP_VMOV) || (cur_op == OP_VSRA && cur_type == T_FP16);
        cur_vld = (cur_op == OP_VLD);
        cur_vst = (cur_op == OP_VST);
        uses_vs1 = !(cur_vld || cur_op == OP_VSPLAT);
        uses_vs2 = (cur_op == OP_VADD) || (cur_op == OP_VSUB) || (cur_op == OP_VMUL) ||
                   (cur_op == OP_VMAX) || (cur_op == OP_VMIN);
        cur_first = (cur_slice == '0);
        cur_last = (cur_slice == SW'(NSLICE - 1));

        // RAW: same register slice written by an operation still in flight.
        hazard = 1'b0;
        if (x1_v && x1_wen && x1_sl == cur_slice &&
            ((uses_vs1 && x1_vd == cur_vs1) || (uses_vs2 && x1_vd == cur_vs2)))
            hazard = 1'b1;
        if (x2_v && x2_wen && x2_sl == cur_slice &&
            ((uses_vs1 && x2_vd == cur_vs1) || (uses_vs2 && x2_vd == cur_vs2)))
            hazard = 1'b1;
        if (x3_v && x3_wen && x3_sl == cur_slice &&
            ((uses_vs1 && x3_vd == cur_vs1) || (uses_vs2 && x3_vd == cur_vs2)))
            hazard = 1'b1;
        if (x4_v && x4_wen && x4_sl == cur_slice &&
            ((uses_vs1 && x4_vd == cur_vs1) || (uses_vs2 && x4_vd == cur_vs2)))
            hazard = 1'b1;

        if (cur_vld && cur_first)
            mem_ok = !st_x1 && !sb_valid && tcm_gnt;
        else if (cur_vst && cur_first)
            mem_ok = !st_x1 && !sb_valid;
        else
            mem_ok = 1'b1;

        issue = cur_valid && !cur_ill && !hazard && mem_ok;
        cur_done = cur_valid && (cur_ill || (issue && cur_last));
        cur_free = !cur_valid || cur_done;

        case (cur_op)
            OP_VADD:  begin cur_fn = ALU_ADD;  cur_rsel = (cur_type == T_FP16) ? R_FPADD : R_ALU;  end
            OP_VSUB:  begin cur_fn = ALU_SUB;  cur_rsel = (cur_type == T_FP16) ? R_FPADD : R_ALU;  end
            OP_VMUL:  begin cur_fn = ALU_PASS; cur_rsel = (cur_type == T_FP16) ? R_FPMUL : R_IMUL; end
            OP_VMAX, OP_VRELU: begin cur_fn = ALU_MAX; cur_rsel = R_ALU; end
            OP_VMIN:  begin cur_fn = ALU_MIN;  cur_rsel = R_ALU; end
            OP_VSRA:  begin cur_fn = ALU_SRA;  cur_rsel = R_ALU; end
            default:  begin cur_fn = ALU_PASS; cur_rsel = R_ALU; end
        endcase

        case (cur_type)
            T_INT8:  splat = {4{cur_scalar[7:0]}};
            T_INT32: splat = cur_scalar;
            default: splat = {2{cur_scalar[15:0]}};
        endcase

        // VSRA: shift clamped to the lane width, minus one (the ALU's X1
        // shift; the last step of 1 is its rounding stage). Computed here
        // from registered fields so the ALU's X1 stage only shifts.
        case (cur_type)
            T_INT8:  sra_shm1 = (cur_scalar[4:3] != 2'd0) ? 5'd7  : cur_scalar[4:0] - 5'd1;
            T_INT32: sra_shm1 = cur_scalar[4:0] - 5'd1;
            default: sra_shm1 = cur_scalar[4]             ? 5'd15 : cur_scalar[4:0] - 5'd1;
        endcase
    end

    // ------------------------------------------------------------ register file
    logic [31:0] rf [0:RFD-1];
    logic [31:0] rf_rd1, rf_rd2, rf_wdata;
    logic        rf_we;

    assign rf_rd1 = rf[{cur_vs1, cur_slice}];
    assign rf_rd2 = rf[{cur_vs2, cur_slice}];

    always_ff @(posedge clk) begin
        if (rf_we)
            rf[{x4_vd, x4_sl}] <= rf_wdata;
    end

    // ------------------------------------------------------------ queue control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_valid <= 1'b0;
            cur_valid <= 1'b0;
            cur_slice <= '0;
            illegal <= 1'b0;
        end else begin
            if (cur_free) begin
                cur_valid <= buf_valid || accept;
                cur_slice <= '0;
                buf_valid <= 1'b0;
            end else if (issue) begin
                cur_slice <= cur_slice + 1'b1;
            end
            if (accept && !(cur_free && !buf_valid))
                buf_valid <= 1'b1;
            if (cur_valid && cur_ill)
                illegal <= 1'b1;
            else if (illegal_clear)
                illegal <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        // buf only loads when the command may have to wait (cur busy); a
        // command accepted into an idle unit goes straight to cur.
        if (accept && cur_valid) begin
            buf_op <= cmd_op;
            buf_type <= cmd_type;
            buf_vd <= cmd_vd;
            buf_vs1 <= cmd_vs1;
            buf_vs2 <= cmd_vs2;
            buf_addr <= cmd_addr;
            buf_scalar <= cmd_scalar;
        end
        // cur only loads when a command moves in (no toggling while idle)
        if (cur_free && (buf_valid || accept)) begin
            if (buf_valid) begin
                cur_op <= buf_op;
                cur_type <= buf_type;
                cur_vd <= buf_vd;
                cur_vs1 <= buf_vs1;
                cur_vs2 <= buf_vs2;
                cur_addr <= buf_addr;
                cur_scalar <= buf_scalar;
            end else begin
                cur_op <= cmd_op;
                cur_type <= cmd_type;
                cur_vd <= cmd_vd;
                cur_vs1 <= cmd_vs1;
                cur_vs2 <= cmd_vs2;
                cur_addr <= cmd_addr;
                cur_scalar <= cmd_scalar;
            end
        end
    end

    // ------------------------------------------------------------ issue -> X4
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x1_v <= 1'b0;
            x2_v <= 1'b0;
            x3_v <= 1'b0;
            x4_v <= 1'b0;
            sb_valid <= 1'b0;
        end else begin
            x1_v <= issue;
            x2_v <= x1_v;
            x3_v <= x2_v;
            x4_v <= x3_v;
            if (x1_v && x1_vst && x1_last) begin
                sb_valid <= 1'b1;
            end else if (sb_valid && tcm_gnt) begin
                sb_valid <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk) begin
        // Clock enable: the X1 fields and operands load only on issue, so the
        // ALU, IMUL and FP16 inputs (and every register behind them) hold
        // still while the unit is idle or stalled. All X1 fields are read
        // only together with x1_v.
        if (issue) begin
            x1_wen <= !cur_vst;
            x1_vd <= cur_vd;
            x1_sl <= cur_slice;
            x1_rsel <= cur_rsel;
            x1_vld <= cur_vld;
            x1_vst <= cur_vst;
            x1_last <= cur_last;
            x1_sub <= (cur_op == OP_VSUB);
            x1_type <= cur_type;
            x1_fn <= cur_fn;
            x1_shm1 <= sra_shm1;
            x1_shz <= (cur_scalar[4:0] == 5'd0);
            x1_a <= (cur_op == OP_VSPLAT) ? splat : rf_rd1;
            x1_b <= (cur_op == OP_VRELU) ? 32'd0 : rf_rd2;
        end

        x2_wen <= x1_wen;   x3_wen <= x2_wen;   x4_wen <= x3_wen;
        x2_vd <= x1_vd;     x3_vd <= x2_vd;     x4_vd <= x3_vd;
        x2_sl <= x1_sl;     x3_sl <= x2_sl;     x4_sl <= x3_sl;
        x2_rsel <= x1_rsel; x3_rsel <= x2_rsel; x4_rsel <= x3_rsel;

        // The buffer is empty from a VST's first slice issue (which waited for
        // that) until its last slice reaches X1, so sb_addr can take the
        // address when the last slice issues.
        if (issue && cur_vst && cur_last)
            sb_addr <= cur_addr;
        // VLD: slice 0 comes straight from tcm_rdata, the rest is held here.
        if (x1_v && x1_vld && x1_sl == '0)
            ld_buf <= tcm_rdata[VLEN-1:32];
        // VST: collect slices; the last one goes with them into the buffer.
        if (x1_v && x1_vst) begin
            for (int k = 0; k < NSLICE - 1; k++)
                if (x1_sl == SW'(k)) st_buf[k*32 +: 32] <= x1_a;
            if (x1_last)
                sb_hi <= x1_a;
        end
    end

    // Pass value for the ALU: VLD data or the issued operand (VMOV, VSPLAT).
    logic [31:0] x1_pass;
    always_comb begin
        x1_pass = x1_a;
        if (x1_vld) begin
            x1_pass = tcm_rdata[31:0];
            for (int k = 1; k < NSLICE; k++)
                if (x1_sl == SW'(k)) x1_pass = ld_buf[(k-1)*32 +: 32];
        end
    end

    // ------------------------------------------------------------ datapath
    logic [31:0] alu_y, imul_y, fpmul_y, fpadd_y;
    logic [1:0]  fpmul_v, fpadd_v;
    logic [31:0] x4_r;

    npu_vector_alu alu (
        .clk(clk), .a(x1_a), .b(x1_b), .p(x1_pass), .typ(x1_type), .fn(x1_fn),
        .shm1(x1_shm1), .sh_zero(x1_shz), .y(alu_y)
    );

    npu_vector_imul imul (
        .clk(clk), .a(x1_a), .b(x1_b), .typ(x1_type), .y(imul_y)
    );

    // Two FP16 lanes per slice. (Explicit instances: a generate loop around
    // npu_fp16_add trips a Verilator 5.006 parameter-cloning bug in
    // npu_fp_add_core when linting together with npu_fp32_add.)
    logic fp_mul_go, fp_add_go;
    assign fp_mul_go = x1_v && x1_rsel == R_FPMUL;
    assign fp_add_go = x1_v && x1_rsel == R_FPADD;

    npu_fp16_mul fmul0 (
        .clk(clk), .rst_n(rst_n), .in_valid(fp_mul_go), .a(x1_a[15:0]), .b(x1_b[15:0]),
        .out_valid(fpmul_v[0]), .y(fpmul_y[15:0])
    );
    npu_fp16_mul fmul1 (
        .clk(clk), .rst_n(rst_n), .in_valid(fp_mul_go), .a(x1_a[31:16]), .b(x1_b[31:16]),
        .out_valid(fpmul_v[1]), .y(fpmul_y[31:16])
    );
    npu_fp16_add fadd0 (
        .clk(clk), .rst_n(rst_n), .in_valid(fp_add_go), .a(x1_a[15:0]), .b(x1_b[15:0]),
        .sub(x1_sub), .out_valid(fpadd_v[0]), .y(fpadd_y[15:0])
    );
    npu_fp16_add fadd1 (
        .clk(clk), .rst_n(rst_n), .in_valid(fp_add_go), .a(x1_a[31:16]), .b(x1_b[31:16]),
        .sub(x1_sub), .out_valid(fpadd_v[1]), .y(fpadd_y[31:16])
    );

    // X3 -> X4: ALU, IMUL and FP16 MUL results are ready at X3.
    always_ff @(posedge clk) begin
        case (x3_rsel)
            R_IMUL:  x4_r <= imul_y;
            R_FPMUL: x4_r <= fpmul_y;
            default: x4_r <= alu_y;
        endcase
    end

    assign rf_wdata = (x4_rsel == R_FPADD) ? fpadd_y : x4_r;
    assign rf_we = x4_v && x4_wen;

    // ------------------------------------------------------------ TCM port A
    assign tcm_req = sb_valid || vld_req;
    assign tcm_we = sb_valid;
    assign tcm_addr = sb_valid ? sb_addr : cur_addr;
    // st_buf is not overwritten while the buffer is full: the next VST issues
    // only after the buffer has drained.
    assign tcm_wdata = {sb_hi, st_buf};
    assign tcm_wstrb = {(VLEN/8){1'b1}};

    // A VST slice in X1 is still on its way to the store buffer; in X2..X4 it
    // has no effect left (the buffer covers it), so only writing slices count.
    assign busy = buf_valid || cur_valid || x1_v || (x2_v && x2_wen) || (x3_v && x3_wen) ||
                  (x4_v && x4_wen) || sb_valid;

    // The FP16 units' valid outputs only mirror the pipeline's own tracking.
    logic unused_ok;
    assign unused_ok = &{1'b0, fpmul_v, fpadd_v};
endmodule
