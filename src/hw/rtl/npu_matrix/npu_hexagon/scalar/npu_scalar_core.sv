`timescale 1ns/1ps

// npu_scalar_core: RV32I control core of the Hexagon-style subsystem (issue #136).
//
// Multi-cycle, unpipelined state machine, one instruction at a time:
//
//   FETCH -> IWAIT -> DECODE -> EXEC -> [MEM -> MWAIT | VCMD | VFENCE] -> WB
//
// DECODE reads the register file and computes everything EXEC needs into
// registers: the operands and ALU controls, the jump / branch target or
// PC + 4 (npc), the branch decision (a not-taken branch carries PC + 4 in
// op_b) and the load/store alignment check. EXEC then already issues the
// fetch of the next instruction for every instruction except VFENCE (whose
// successor is fetched in WB, after the vector unit went idle) and except
// those that fault in EXEC, so the fetch overlaps the ALU, memory or
// vector-command stages. The instruction retires in WB (in VCMD for a vector
// command, when it is accepted) and goes straight to DECODE if the fetch
// response has arrived; otherwise it waits in IWAIT (or re-requests from
// FETCH if the fetch has not been accepted yet). After DECODE,
// instr is not used beyond EXEC (later stages use registered copies of the
// fields they need), so the response may land in any state from EXEC on.
// Every stage boundary is registered: the EXEC ALU sees only registered
// operands and controls, the load lane extraction happens in WB from a
// registered response word, and every port output comes from registers
// (i_req_addr through one 3:1 PC select).
//
// Cycle model (exact; src/test/model/rv32_iss.py implements it and the
// testbench checks the cycles counter against it) with both ports always
// ready and a fixed response latency of L cycles after the handshake (L = 0:
// same cycle), vcmd_ready high and vec_busy low, counted from one DECODE to
// the next:
//
//   instruction class                                    L = 0  L = 1  any L
//   OP, OP-IMM, LUI, AUIPC, FENCE, JAL, JALR, branches,
//   vector command                                         3      3    max(3, L + 2)
//   loads, stores, VFENCE                                  4      5    4 + L
//   start to the first DECODE                              1      2    1 + L
//   until the stop: ECALL / EBREAK / illegal (DECODE)      1      1    1
//                   misaligned access or jump target       2      2    2
//                   data bus error                         3      4    3 + L
//                   instruction bus error (instead of
//                   the next DECODE)                       0      0    0
//
// Port stalls, vcmd_ready low and a VFENCE waiting for vec_busy lengthen an
// instruction beyond this (the model does not cover them).
//
// Timing at 100 MHz: the longest paths (Yosys synth_xilinx netlist, rough
// 7-series -1 delay estimate) are in DECODE, both from the instr register
// through the LUTRAM register-file read: -> 33-bit branch compare -> nt_q
// (7.15 ns) and -> target adder -> npc (7.1 ns), i.e. ~2.9 ns of slack, with
// the estimate charging 1.2 ns for the LUTRAM read. Every other path is
// below 6 ns, and every output comes straight from registers or through one
// 3:1 select (i_req_addr, 4.0 ns). Registering the register-file read would
// only bring the worst path down to the next one (5.9 ns, instruction
// decode -> control / status registers) at the cost of one more cycle on
// every instruction (CPI 3 -> 4, loads / stores 5 -> 6) and 64 flops, so it
// is not done. Should post-route timing ever fail on these two paths, the
// cheaper fallbacks are moving the branch compare (branches only) and the
// JALR target add (JALR only; JAL / branch targets do not read the
// register file) to EXEC, +1 cycle each.
//
// ISA: RV32I. FENCE and FENCE.I execute as NOPs. ECALL and EBREAK stop the core
// in the halted state (halt_cause = 0 for ECALL, 1 for EBREAK) and count as
// retired. Every other SYSTEM encoding (CSR*, MRET, WFI, ...) and every other
// undefined encoding is an illegal instruction. Custom-0 (opcode 7'b0001011,
// R-type fields) sends one vector command; the packing follows
// docs/goals/hexagon_npu/architecture.md (op = inst[29:25], type = inst[13:12],
// vd = inst[9:7], vs1 = inst[22:20], vs2 = {inst[31:30], inst[14]},
// scalar = x[rs1]). rd[4:3] and rs2[4:3] are ignored. op = 31 (VFENCE) is not
// sent; the core waits in VFENCE until vec_busy is low. The vector command
// port is not checked for illegal ops: the vector unit flags those itself.
//
// Control:
//   start (pulse) is honoured only when the core is not running (idle, halted
//   or faulted); it loads the architectural PC from start_pc, clears halt and
//   fault status and both performance counters, and begins fetching. A start
//   while running is ignored. The register file keeps its contents across
//   start (it has no reset at all: x1..x31 are undefined after power-up until
//   software writes them).
//   halted   : ECALL/EBREAK retired. fault_pc holds the PC of that instruction.
//              (fault_pc keeps the value of the last stop while running; only
//              rst_n clears it.)
//   fault    : the instruction at fault_pc did not complete (no register or
//              memory side effect). fault_cause:
//                1 illegal instruction
//                2 misaligned instruction fetch: a taken branch / JAL / JALR
//                  whose target is not 4-aligned (fault_pc = the jump), or a
//                  misaligned start_pc (fault_pc = start_pc)
//                3 misaligned load  (LW not 4-aligned, LH/LHU not 2-aligned)
//                4 misaligned store (SW not 4-aligned, SH not 2-aligned)
//                5 instruction bus error (i_resp_err with i_resp_valid;
//                  fault_pc = the address fetched, e.g. start_pc)
//                6 data bus error (d_resp_err with d_resp_valid)
//   cycles   : clock cycles spent running since the last accepted start.
//   instret  : instructions retired since the last accepted start (32-bit,
//              wrapping).
//
// Ports (cache CPU-side protocol of architecture.md): a request is
// transferred when req_valid && req_ready; the request fields hold steady
// while valid is high and ready is low. Every request, writes included, is
// completed by exactly one resp_valid, in the handshake cycle or any later
// cycle (so npu_cache, a registered TCM / MMIO adapter and a combinational
// one all work). The core keeps at most one request outstanding per port,
// and keeps every d_req_* field stable from the request until its resp_valid
// (so the L1 D$ may use CPU_HOLD = 1). ready may depend on valid; valid and
// the request fields never depend on ready or on a response in the same
// cycle (every *_valid and request field is a function of registers only).
// More strongly, no output depends combinationally on any input (no path
// from a response, a ready, start / start_pc or vec_busy to any output), so
// a cache, TCM or MMIO adapter that answers in the handshake cycle and / or
// derives req_ready from req_valid and the request fields cannot form a
// loop through the core. (Checked with Yosys `check -assert` on a wrapper
// with such responders on all three ports, and in the testbench, which
// requires every output to stay constant between clock edges while it
// changes the readies, the same-cycle responses and start mid-cycle.)
// The fetch issued in EXEC is speculative only in one case: when the load or
// store then gets a data bus error, the core stops with that fetch possibly
// still in flight; its response is ignored, and after the next start FETCH
// waits for it before requesting. (MEM raises d_req_valid only once that
// fetch has been accepted, so no request is ever withdrawn.)
//   i_*: read-only instruction port, word-aligned addresses.
//   d_*: word-oriented data port. d_req_addr is word aligned ({addr[31:2],
//        2'b00}); stores drive the byte lanes in d_req_wstrb with the store
//        data replicated across the word (SB: 4 x byte, SH: 2 x half); loads
//        drive d_req_wstrb = 0 and extract / sign- or zero-extend the lane
//        from d_resp_rdata. Address decode (DDR / TCM / MMIO) is outside.
//   i_resp_err / d_resp_err: bus error flag qualified by resp_valid (an
//        addition to the cache protocol; tie to 0 when unused).
//   vcmd_*: held stable while vcmd_valid is high; the instruction retires in
//        the vcmd_valid && vcmd_ready cycle. The core may sit in VCMD for any
//        number of cycles; the next fetch is issued meanwhile and its
//        response (including a bus error, reported after the command
//        retires, fault_pc = the next PC) is held until then.
//
// Assumptions on the environment:
//   - vec_busy covers an accepted vector command no later than the third
//     cycle after its handshake (the earliest cycle in which a following
//     VFENCE samples it; npu_vector_unit raises busy the next cycle) and
//     until its last register-file / TCM write.
//   - A response only answers a request of this core: the memory ports are
//     reset together with the core (rst_n), so no response of a request
//     issued before a reset arrives afterwards.
//   - The successor of every instruction except VFENCE is fetched while the
//     instruction executes, i.e. before a store writes memory: a store to the
//     next instruction's address is not seen by that fetch (no
//     self-modifying code; the I$ is not coherent with the D$ either).
//
// Reset: rst_n (asynchronous, allowed in any state) clears the control
// state, the status outputs and the counters; the core is then idle.
// Datapath registers (pc, npc, instr, operands, controls latched in DECODE,
// ALU / load results, store data and strobes) have no reset: each is written
// before it is used after start, and every output derived from them is
// qualified by a valid that is reset. This keeps the reset net small and lets
// Vivado use the flop set/reset pins for logic.
module npu_scalar_core (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        start,
    input  logic [31:0] start_pc,
    output logic        running,
    output logic        halted,
    output logic        halt_cause,
    output logic        fault,
    output logic [2:0]  fault_cause,
    output logic [31:0] fault_pc,

    output logic        i_req_valid,
    input  logic        i_req_ready,
    output logic [31:0] i_req_addr,
    input  logic        i_resp_valid,
    input  logic [31:0] i_resp_rdata,
    input  logic        i_resp_err,

    output logic        d_req_valid,
    input  logic        d_req_ready,
    output logic [31:0] d_req_addr,
    output logic        d_req_we,
    output logic [3:0]  d_req_wstrb,
    output logic [31:0] d_req_wdata,
    input  logic        d_resp_valid,
    input  logic [31:0] d_resp_rdata,
    input  logic        d_resp_err,

    output logic        vcmd_valid,
    input  logic        vcmd_ready,
    output logic [4:0]  vcmd_op,
    output logic [1:0]  vcmd_type,
    output logic [2:0]  vcmd_vd,
    output logic [2:0]  vcmd_vs1,
    output logic [2:0]  vcmd_vs2,
    output logic [31:0] vcmd_scalar,
    input  logic        vec_busy,

    output logic [31:0] cycles,
    output logic [31:0] instret
);
    localparam logic [3:0] S_IDLE   = 4'd0;
    localparam logic [3:0] S_FETCH  = 4'd1;
    localparam logic [3:0] S_IWAIT  = 4'd2;
    localparam logic [3:0] S_DECODE = 4'd3;
    localparam logic [3:0] S_EXEC   = 4'd4;
    localparam logic [3:0] S_MEM    = 4'd5;
    localparam logic [3:0] S_MWAIT  = 4'd6;
    localparam logic [3:0] S_WB     = 4'd7;
    localparam logic [3:0] S_VCMD   = 4'd8;
    localparam logic [3:0] S_VFENCE = 4'd9;
    localparam logic [3:0] S_HALT   = 4'd10;
    localparam logic [3:0] S_FAULT  = 4'd11;

    localparam logic [6:0] OPC_LUI     = 7'b0110111;
    localparam logic [6:0] OPC_AUIPC   = 7'b0010111;
    localparam logic [6:0] OPC_JAL     = 7'b1101111;
    localparam logic [6:0] OPC_JALR    = 7'b1100111;
    localparam logic [6:0] OPC_BRANCH  = 7'b1100011;
    localparam logic [6:0] OPC_LOAD    = 7'b0000011;
    localparam logic [6:0] OPC_STORE   = 7'b0100011;
    localparam logic [6:0] OPC_OPIMM   = 7'b0010011;
    localparam logic [6:0] OPC_OP      = 7'b0110011;
    localparam logic [6:0] OPC_FENCE   = 7'b0001111;
    localparam logic [6:0] OPC_SYSTEM  = 7'b1110011;
    localparam logic [6:0] OPC_CUSTOM0 = 7'b0001011;

    localparam logic [2:0] FC_ILLEGAL   = 3'd1;
    localparam logic [2:0] FC_FETCH_MIS = 3'd2;
    localparam logic [2:0] FC_LOAD_MIS  = 3'd3;
    localparam logic [2:0] FC_STORE_MIS = 3'd4;
    localparam logic [2:0] FC_IBUS      = 3'd5;
    localparam logic [2:0] FC_DBUS      = 3'd6;

    logic [3:0]  state;
    logic [31:0] pc;        // PC of the instruction in flight
    logic [31:0] npc;       // next PC: target (jump / branch) or PC + 4
    logic        nt_q;      // branch not taken: the next PC is op_b (PC + 4)
    logic [31:0] instr;     // used by DECODE and EXEC only
    logic [32:0] op_a;      // rs1 / pc / 0 (bit-reversed for SLL); [32] SLT/SRA sign
    logic [32:0] op_b;      // rs2 / imm / 4 / PC + 4 (branch), inverted for SUB/SLT;
                            // [32] SLT sign
    logic [2:0]  alu_f3_q;  // ALU result select (funct3 of OP/OP-IMM, else 0 = add)
    logic        sub_q;     // adder carry-in (op_b already inverted)
    logic [31:0] alu_q;     // ALU result / effective address / link value
    logic [31:0] load_q;    // raw load response word
    logic [31:0] wdata_q;   // store data, replicated across the word
    logic [3:0]  wstrb_q;
    logic        st_q, ld_q, wen_q, pf_q;
    logic [2:0]  lf3_q;     // load funct3 (lane size / sign) for WB
    logic [4:0]  rd_q;
    logic [5:0]  vfld_q;    // {type, vd, vs2[0]} of a vector command (rest in op_b)
    // Next-instruction fetch status after DECODE.
    logic        i_out_q;   // a fetch was accepted and its response is pending
    logic        i_have_q;  // its response arrived early (instr / i_err_q hold it)
    logic        i_err_q;

    // 32 x 32 register file: no reset so it maps to LUTRAM. Writes to x0 are
    // not filtered (rf[0] holds garbage); reads of x0 are forced to zero.
    logic [31:0] rf [0:31];

    // ------------------------------------------------------------------
    // Instruction fields and decode (from the registered instruction word)
    // ------------------------------------------------------------------
    logic [6:0] opcode, funct7;
    logic [4:0] rd, rs1, rs2;
    logic [2:0] funct3;
    assign opcode = instr[6:0];
    assign rd     = instr[11:7];
    assign funct3 = instr[14:12];
    assign rs1    = instr[19:15];
    assign rs2    = instr[24:20];
    assign funct7 = instr[31:25];

    logic is_lui, is_auipc, is_jal, is_jalr, is_branch, is_load, is_store;
    logic is_opimm, is_op, is_system, is_custom, is_vfence, writes_rd, legal;
    assign is_lui    = opcode == OPC_LUI;
    assign is_auipc  = opcode == OPC_AUIPC;
    assign is_jal    = opcode == OPC_JAL;
    assign is_jalr   = opcode == OPC_JALR;
    assign is_branch = opcode == OPC_BRANCH;
    assign is_load   = opcode == OPC_LOAD;
    assign is_store  = opcode == OPC_STORE;
    assign is_opimm  = opcode == OPC_OPIMM;
    assign is_op     = opcode == OPC_OP;
    assign is_system = opcode == OPC_SYSTEM;
    assign is_custom = opcode == OPC_CUSTOM0;
    assign is_vfence = is_custom && instr[29:25] == 5'd31;
    assign writes_rd = is_lui | is_auipc | is_jal | is_jalr | is_load | is_opimm | is_op;

    // Part selects stay in continuous assignments (Icarus does not support
    // constant selects inside always_* blocks).
    logic [1:0] f3_hi, f3_lo;
    assign f3_hi = funct3[2:1];
    assign f3_lo = funct3[1:0];
    always_comb begin
        case (opcode)
            OPC_LUI, OPC_AUIPC, OPC_JAL, OPC_CUSTOM0: legal = 1'b1;
            OPC_JALR:   legal = funct3 == 3'b000;
            OPC_BRANCH: legal = f3_hi != 2'b01;
            OPC_LOAD:   legal = f3_lo != 2'b11 && f3_hi != 2'b11;
            OPC_STORE:  legal = funct3 < 3'd3;                       // SB, SH, SW
            OPC_OPIMM:  legal = (funct3 == 3'b001) ? funct7 == 7'h00 :
                                (funct3 == 3'b101) ? (funct7 == 7'h00 || funct7 == 7'h20) :
                                1'b1;
            OPC_OP:     legal = funct7 == 7'h00 ||
                                (funct7 == 7'h20 && (funct3 == 3'b000 || funct3 == 3'b101));
            OPC_FENCE:  legal = f3_hi == 2'b00;                     // FENCE, FENCE.I
            OPC_SYSTEM: legal = instr == 32'h0000_0073 || instr == 32'h0010_0073;
            default:    legal = 1'b0;
        endcase
    end

    logic [31:0] imm, imm_i, imm_s, imm_b, imm_u, imm_j;
    assign imm_i = {{20{instr[31]}}, instr[31:20]};
    assign imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
    assign imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    assign imm_u = {instr[31:12], 12'b0};
    assign imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
    always_comb begin
        case (opcode)
            OPC_STORE:          imm = imm_s;
            OPC_BRANCH:         imm = imm_b;
            OPC_LUI, OPC_AUIPC: imm = imm_u;
            OPC_JAL:            imm = imm_j;
            default:            imm = imm_i;
        endcase
    end

    logic [31:0] rs1_val, rs2_val;
    assign rs1_val = (rs1 == 5'd0) ? 32'd0 : rf[rs1];
    assign rs2_val = (rs2 == 5'd0) ? 32'd0 : rf[rs2];

    // ------------------------------------------------------------------
    // DECODE: operands, ALU controls, next PC and branch decision
    // ------------------------------------------------------------------
    // The EXEC ALU sees only registered operands and controls: SUB/SLT/SLTU
    // get op_b inverted here (carry-in sub_q), SLT/SLTU/SRA get a sign bit in
    // op_a[32] / op_b[32] so the 33-bit sum's MSB is the compare result and
    // the shifter's fill bit, SLL gets op_a bit-reversed (one right shifter),
    // and JAL/JALR compute their link value as pc + 4 on the adder.
    logic        is_aluop, is_sll, is_sra, is_slt, alu_sub;
    logic [31:0] a_sel, b_sel, rs1_rev, pc4, target;
    assign pc4      = pc + 32'd4;
    assign target   = (is_jalr ? rs1_val : pc) + imm;
    assign is_aluop = is_op | is_opimm;
    assign is_sll   = is_aluop && funct3 == 3'b001;
    assign is_sra   = is_aluop && funct3 == 3'b101 && funct7[5];
    assign is_slt   = is_aluop && funct3 == 3'b010;           // signed SLT / SLTI
    assign alu_sub  = (is_aluop && f3_hi == 2'b01) || (is_op && funct7[5] && funct3 == 3'b000);
    genvar gi;
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_rev_a
            assign rs1_rev[gi] = rs1_val[31-gi];
        end
    endgenerate
    assign a_sel = is_lui ? 32'd0 :
                   (is_auipc | is_jal | is_jalr) ? pc :
                   is_sll ? rs1_rev : rs1_val;
    // A branch does not use the ALU: op_b carries its fall-through PC + 4.
    assign b_sel = is_op ? rs2_val : (is_jal | is_jalr) ? 32'd4 : is_branch ? pc4 : imm;

    // Branch decision in DECODE, beside the target adder, so the next PC is
    // known when EXEC starts and every instruction except VFENCE can fetch
    // its successor from EXEC. To keep the compare off the npc path, npc
    // always gets the target and the compare only sets nt_q: a not-taken
    // branch fetches op_b (PC + 4) from EXEC and copies it into npc there.
    // funct3: 000 BEQ, 001 BNE, 100 BLT, 101 BGE, 110 BLTU, 111 BGEU
    // (funct3[0] inverts, funct3[1] unsigned).
    logic        cmp_s, br_eq, br_lt, br_cond;
    assign cmp_s   = !funct3[1];
    assign br_eq   = rs1_val == rs2_val;
    assign br_lt   = $signed({cmp_s & rs1_val[31], rs1_val}) < $signed({cmp_s & rs2_val[31], rs2_val});
    assign br_cond = (funct3[2] ? br_lt : br_eq) ^ funct3[0];

    // Load/store alignment, from the low address bits only (a 2-bit add), so
    // that EXEC knows before its fetch whether the access will fault: a fetch
    // request, once raised, is never withdrawn.
    logic [1:0] ea_lo_d;
    logic       mis_d;
    assign ea_lo_d = rs1_val[1:0] + imm[1:0];
    assign mis_d   = (is_load | is_store) &&
                     ((f3_lo == 2'b01) ? ea_lo_d[0] : (f3_lo == 2'b10 && ea_lo_d != 2'b00));

    // ------------------------------------------------------------------
    // EXEC datapath (all inputs registered)
    // ------------------------------------------------------------------
    // One 33-bit adder: ADD/SUB, SLT/SLTU (sum[32]), LUI/AUIPC, the link
    // value and the load/store effective address. The adder result goes
    // through a single 2:1 select (bit 0: 3:1) into alu_q; the shift and
    // logic results are selected in parallel with the carry chain.
    logic [32:0] sum, sh_ext;
    logic [31:0] sh_rev, sh_right, a_lo, b_lo, lres, alu_res;
    logic        slt_bit;
    assign sum    = op_a + op_b + {32'd0, sub_q};
    assign sh_ext = $signed(op_a) >>> op_b[4:0];
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : g_rev_sh
            assign sh_rev[gi] = sh_ext[31-gi];
        end
    endgenerate
    assign sh_right = sh_ext[31:0];
    assign a_lo     = op_a[31:0];
    assign b_lo     = op_b[31:0];
    always_comb begin
        case (alu_f3_q)
            3'b001:  lres = sh_rev;                       // SLL (operand pre-reversed)
            3'b101:  lres = sh_right;                     // SRL / SRA
            3'b100:  lres = a_lo ^ b_lo;
            3'b110:  lres = a_lo | b_lo;
            3'b111:  lres = a_lo & b_lo;
            default: lres = 32'd0;                        // add, SLT, SLTU
        endcase
    end
    assign slt_bit = alu_f3_q[2:1] == 2'b01 && sum[32];
    assign alu_res = (alu_f3_q == 3'b000) ? sum[31:0] : {lres[31:1], lres[0] | slt_bit};

    // Store byte lanes from the low address bits.
    logic [3:0] lanes;
    logic [1:0] ea_lo;
    assign ea_lo = sum[1:0];
    assign lanes = (f3_lo == 2'b00) ? (4'b0001 << ea_lo) :
                   (f3_lo == 2'b01) ? (ea_lo[1] ? 4'b1100 : 4'b0011) : 4'b1111;

    // Load lane extraction in WB from the registered response word.
    logic [7:0]  lane_b;
    logic [15:0] lane_h;
    logic [31:0] load_ext, wb_data;
    // lf3_q: 000 LB, 001 LH, 010 LW, 100 LBU, 101 LHU ([2] = unsigned).
    assign lane_b   = alu_q[1] ? (alu_q[0] ? load_q[31:24] : load_q[23:16]) :
                                 (alu_q[0] ? load_q[15:8]  : load_q[7:0]);
    assign lane_h   = alu_q[1] ? load_q[31:16] : load_q[15:0];
    assign load_ext = lf3_q[1] ? load_q :
                      lf3_q[0] ? {{16{lane_h[15] & ~lf3_q[2]}}, lane_h} :
                                 {{24{lane_b[7] & ~lf3_q[2]}}, lane_b};
    assign wb_data  = ld_q ? load_ext : alu_q;

    // target[0] is dropped (JALR clears it, other targets are even), and the
    // shifter's bit 32 is only the fill bit.
    logic unused_bits;
    assign unused_bits = target[0] ^ sh_ext[32];

    // ------------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------------
    logic stopped, post_exec, i_hs, retire, nxt_have, nxt_err;
    assign stopped = state == S_IDLE || state == S_HALT || state == S_FAULT;
    assign running = !stopped;
    assign halted  = state == S_HALT;
    assign fault   = state == S_FAULT;

    // Instruction port. FETCH waits until a response abandoned by a fault
    // (i_out_q still set after the stop) has drained. EXEC fetches npc unless
    // the instruction is a VFENCE or will fault in EXEC (misaligned jump
    // target or load/store address); if that fetch is not accepted, MEM / WB
    // / VCMD keep requesting it until it is (WB is where a VFENCE fetches).
    assign post_exec   = state == S_MEM || state == S_WB || state == S_VCMD;
    assign i_req_valid = (state == S_FETCH && !i_out_q) ||
                         (state == S_EXEC && pf_q && !(npc[1] && !nt_q)) ||
                         (post_exec && !i_out_q && !i_have_q);
    assign i_req_addr  = (state == S_FETCH) ? pc : (state == S_EXEC && nt_q) ? b_lo : npc;
    assign i_hs        = i_req_valid && i_req_ready;

    // The data request waits in MEM until the fetch has been accepted, so a
    // data bus error never has to withdraw a pending fetch request.
    assign d_req_valid = state == S_MEM && (i_out_q || i_have_q);
    assign d_req_addr  = {alu_q[31:2], 2'b00};
    assign d_req_we    = st_q;
    assign d_req_wstrb = wstrb_q;
    assign d_req_wdata = wdata_q;

    assign vcmd_valid  = state == S_VCMD;
    // A vector command does not use the ALU: op_b holds its I-immediate
    // inst[31:20], i.e. op = op_b[9:5], vs1 = op_b[2:0], vs2[2:1] = op_b[11:10].
    assign vcmd_op     = b_lo[9:5];
    assign vcmd_vs1    = b_lo[2:0];
    assign vcmd_vs2    = {b_lo[11:10], vfld_q[0]};
    assign {vcmd_type, vcmd_vd} = vfld_q[5:1];
    assign vcmd_scalar = a_lo;

    // An instruction retires in WB, or in VCMD when the command is accepted.
    // The next instruction is then ready if its response arrived earlier
    // (i_have_q) or arrives now; only a fetch issued after DECODE can be
    // outstanding at that point, so any response belongs to it.
    assign retire   = state == S_WB || (state == S_VCMD && vcmd_ready);
    assign nxt_have = i_have_q || i_resp_valid;
    assign nxt_err  = i_have_q ? i_err_q : i_resp_err;

    // ------------------------------------------------------------------
    // Register file write port (write-back)
    // ------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_WB && wen_q) begin
            rf[rd_q] <= wb_data;
        end
    end

    // ------------------------------------------------------------------
    // Datapath registers (no reset, see the header)
    // ------------------------------------------------------------------
    always_ff @(posedge clk) begin
        // A response is always for the latest fetch, and instr is not needed
        // after EXEC (later stages use the registers below), so it can be
        // captured in any state; one that drains after a fault is overwritten
        // before the next DECODE. Same for the data port.
        if (i_resp_valid) instr <= i_resp_rdata;
        if (d_resp_valid) load_q <= d_resp_rdata;
        case (state)
            S_IDLE, S_HALT, S_FAULT: begin
                if (start) pc <= start_pc;
            end
            S_DECODE: begin
                op_a     <= {(is_slt | is_sra) & rs1_val[31], a_sel};
                op_b     <= alu_sub ? ~{is_slt & b_sel[31], b_sel} : {1'b0, b_sel};
                sub_q    <= alu_sub;
                alu_f3_q <= is_aluop ? funct3 : 3'b000;
                npc      <= (is_jal | is_jalr | is_branch) ? {target[31:1], 1'b0} : pc4;
                nt_q     <= is_branch && !br_cond;
                pf_q     <= !is_vfence && !mis_d;   // for a load/store: !misaligned
                rd_q     <= rd;
                wen_q    <= writes_rd;                 // x0 may be written: reads force 0
                ld_q     <= is_load;
                st_q     <= is_store;
                lf3_q    <= funct3;
                if (is_custom) vfld_q <= {instr[13:12], instr[9:7], instr[14]};
            end
            S_EXEC: begin
                alu_q   <= alu_res;
                wstrb_q <= is_store ? lanes : 4'b0000;
                // Store data straight from the regfile (rs2 is unchanged
                // since DECODE), so MEM / MWAIT do not need instr.
                if (is_store) wdata_q <= funct3[1] ? rs2_val :
                                         (funct3[0] ? {2{rs2_val[15:0]}} : {4{rs2_val[7:0]}});
                if (nt_q) npc <= b_lo;
            end
            default: ;
        endcase
        if (retire) pc <= npc;
    end

    // ------------------------------------------------------------------
    // Control state machine
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            i_out_q     <= 1'b0;
            i_have_q    <= 1'b0;
            i_err_q     <= 1'b0;
            halt_cause  <= 1'b0;
            fault_cause <= '0;
            fault_pc    <= '0;
            cycles      <= '0;
            instret     <= '0;
        end else begin
            if (running) begin
                cycles <= cycles + 32'd1;
            end

            // Fetch tracking. i_have_q latches a response that arrives before
            // the retire state; FETCH, IWAIT and retire consume it.
            if (i_resp_valid) begin
                i_out_q <= 1'b0;
            end else if (i_hs) begin
                i_out_q <= 1'b1;
            end
            if (i_resp_valid) begin
                i_have_q <= 1'b1;
                i_err_q  <= i_resp_err;
            end
            if (retire) begin
                instret  <= instret + 32'd1;
                i_have_q <= 1'b0;
                if (nxt_have) begin
                    if (nxt_err) begin
                        fault_cause <= FC_IBUS;
                        fault_pc    <= npc;
                        state       <= S_FAULT;
                    end else begin
                        state <= S_DECODE;
                    end
                end else begin
                    state <= (i_out_q || i_hs) ? S_IWAIT : S_FETCH;
                end
            end

            case (state)
                S_IDLE, S_HALT, S_FAULT: begin
                    if (start) begin
                        i_have_q    <= 1'b0;
                        halt_cause  <= 1'b0;
                        fault_cause <= '0;
                        cycles      <= '0;
                        instret     <= '0;
                        if (start_pc[1:0] != 2'b00) begin
                            fault_cause <= FC_FETCH_MIS;
                            fault_pc    <= start_pc;
                            state       <= S_FAULT;
                        end else begin
                            state <= S_FETCH;
                        end
                    end
                end

                S_FETCH, S_IWAIT: begin
                    // In FETCH only a response in the handshake cycle is ours
                    // (anything else is a fault leftover draining).
                    i_have_q <= 1'b0;
                    if (i_resp_valid && (state == S_IWAIT || i_hs)) begin
                        if (i_resp_err) begin
                            fault_cause <= FC_IBUS;
                            fault_pc    <= pc;
                            state       <= S_FAULT;
                        end else begin
                            state <= S_DECODE;
                        end
                    end else if (i_hs) begin
                        state <= S_IWAIT;
                    end
                end

                S_DECODE: begin
                    if (!legal) begin
                        fault_cause <= FC_ILLEGAL;
                        fault_pc    <= pc;
                        state       <= S_FAULT;
                    end else if (is_system) begin
                        halt_cause <= instr[20];          // 0 = ECALL, 1 = EBREAK
                        fault_pc   <= pc;
                        instret    <= instret + 32'd1;
                        state      <= S_HALT;
                    end else begin
                        state <= S_EXEC;
                    end
                end

                S_EXEC: begin
                    // npc[1] can only be set by a jump or branch target.
                    if (npc[1] && !nt_q) begin
                        fault_cause <= FC_FETCH_MIS;
                        fault_pc    <= pc;
                        state       <= S_FAULT;
                    end else if (is_load | is_store) begin
                        if (!pf_q) begin
                            fault_cause <= is_store ? FC_STORE_MIS : FC_LOAD_MIS;
                            fault_pc    <= pc;
                            state       <= S_FAULT;
                        end else begin
                            state <= S_MEM;
                        end
                    end else if (is_custom) begin
                        state <= is_vfence ? S_VFENCE : S_VCMD;
                    end else begin
                        state <= S_WB;
                    end
                end

                S_MEM, S_MWAIT: begin
                    if (d_resp_valid && (state == S_MWAIT || (d_req_valid && d_req_ready))) begin
                        if (d_resp_err) begin
                            fault_cause <= FC_DBUS;
                            fault_pc    <= pc;
                            state       <= S_FAULT;
                        end else begin
                            state <= S_WB;
                        end
                    end else if (state == S_MEM && d_req_valid && d_req_ready) begin
                        state <= S_MWAIT;
                    end
                end

                S_VFENCE: begin
                    if (!vec_busy) begin
                        state <= S_WB;
                    end
                end

                S_WB, S_VCMD: ;                         // retire (above)

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
