"""Reference instruction-set simulator for npu_scalar_core (issue #136).

Architectural contract shared with the RTL
(src/hw/rtl/npu_matrix/npu_hexagon/scalar/npu_scalar_core.sv):

- RV32I. FENCE / FENCE.I are NOPs. ECALL / EBREAK retire and halt
  (halt_cause 0 / 1). Every other SYSTEM encoding and every undefined encoding
  is an illegal instruction.
- Custom-0 (opcode 0b0001011) records a vector command
  ``(op, type, vd, vs1, vs2, scalar)`` with the architecture.md packing, except
  op 31 (VFENCE), which is recorded only in ``fences`` (the core waits for the
  vector unit instead of sending it).
- ``start(pc)`` keeps the register file, clears the halt/fault status and the
  ``cycles`` / ``instret`` counters.
- Faults leave no side effect; ``fault_pc`` is the PC of the faulting
  instruction, or the misaligned start PC. Causes are the FAULT_* constants.
- Memory is a flat little-endian byte array at address 0; any access at or
  beyond its size is a bus error (instruction or data).

``cycles`` follows the RTL cycle model with an ideal environment: both
memory ports always ready with a fixed response latency of ``latency`` cycles
after the handshake (0 = same cycle), vcmd_ready high, vec_busy low. It is
only meaningful under those conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FAULT_NONE = 0
FAULT_ILLEGAL = 1
FAULT_FETCH_MISALIGNED = 2
FAULT_LOAD_MISALIGNED = 3
FAULT_STORE_MISALIGNED = 4
FAULT_IBUS = 5
FAULT_DBUS = 6

HALT_ECALL = 0
HALT_EBREAK = 1

# Cycle model (see the RTL header), counted from one DECODE to the next, for
# a fixed memory latency L. DECODE computes the final next PC, so EXEC fetches
# the next instruction for everything except VFENCE (fetched from WB).


def cpi_simple(latency: int) -> int:
    """OP, OP-IMM, LUI, AUIPC, FENCE, JAL, JALR, branches, vector command."""
    return max(3, latency + 2)


def cpi_mem(latency: int) -> int:
    """Loads, stores and VFENCE."""
    return 4 + latency


CYCLES_STOP = 1         # ECALL / EBREAK / illegal: the DECODE cycle
CYCLES_EXEC_FAULT = 2   # fault detected in EXEC (misaligned access / jump)


MASK = 0xFFFFFFFF


def _sx(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _s32(value: int) -> int:
    return _sx(value, 32)


@dataclass
class VectorCommand:
    op: int
    vtype: int
    vd: int
    vs1: int
    vs2: int
    scalar: int

    def as_tuple(self):
        return (self.op, self.vtype, self.vd, self.vs1, self.vs2, self.scalar)


@dataclass
class RunResult:
    status: str            # "halt" or "fault"
    cause: int             # halt cause or fault cause
    stop_pc: int           # fault_pc register value
    instret: int
    cycles: int
    regs: list[int]


class BusError(Exception):
    pass


@dataclass
class RV32ISS:
    mem_size: int = 0x10000
    mem: bytearray = field(default=None)
    latency: int = 1      # memory response latency of the cycle model
    regs: list[int] = field(default_factory=lambda: [0] * 32)
    pc: int = 0
    vcmds: list[VectorCommand] = field(default_factory=list)
    fences: list[int] = field(default_factory=list)   # index into vcmds at each VFENCE
    written: set[int] = field(default_factory=set)    # word addresses stored to

    def __post_init__(self):
        if self.mem is None:
            self.mem = bytearray(self.mem_size)

    # ------------------------------------------------------------------ memory
    def load_words(self, image: dict[int, int]):
        for addr, word in image.items():
            self.mem[addr:addr + 4] = (word & MASK).to_bytes(4, "little")

    def word(self, addr: int) -> int:
        if addr < 0 or addr + 4 > self.mem_size:
            raise BusError
        return int.from_bytes(self.mem[addr:addr + 4], "little")

    def _store(self, addr: int, value: int, size: int):
        if addr < 0 or addr + size > self.mem_size:
            raise BusError
        self.mem[addr:addr + size] = (value & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
        self.written.add(addr & ~3)

    # ------------------------------------------------------------------ run
    def run(self, start_pc: int, max_steps: int = 1_000_000) -> RunResult:
        """Start at start_pc and run until halt or fault."""
        self.pc = start_pc & MASK
        instret = 0
        cycles = 1 + self.latency          # start to the first DECODE

        def stop(status, cause, pc, extra):
            return RunResult(status, cause, pc, instret, cycles + extra, list(self.regs))

        if self.pc & 3:
            return RunResult("fault", FAULT_FETCH_MISALIGNED, self.pc, 0, 0, list(self.regs))
        for _ in range(max_steps):
            pc = self.pc
            try:
                inst = self.word(pc)
            except BusError:
                # The fetch fails where the next DECODE would have been.
                return stop("fault", FAULT_IBUS, pc, 0)
            outcome = self.step(inst, pc)
            kind = outcome[0]
            if kind == "retire":
                instret += 1
                cycles += outcome[1]
            elif kind == "halt":
                instret += 1
                return stop("halt", outcome[1], pc, CYCLES_STOP)
            else:  # fault
                return stop("fault", outcome[1], pc, outcome[2])
        raise RuntimeError(f"program did not stop within {max_steps} instructions")

    def _wr(self, rd: int, value: int):
        if rd:
            self.regs[rd] = value & MASK

    def step(self, inst: int, pc: int):
        """Execute one instruction. Returns ('retire', cycles), ('halt', cause)
        or ('fault', cause, cycles_until_detected)."""
        opc = inst & 0x7F
        rd = (inst >> 7) & 0x1F
        f3 = (inst >> 12) & 7
        rs1 = (inst >> 15) & 0x1F
        rs2 = (inst >> 20) & 0x1F
        f7 = inst >> 25
        a = self.regs[rs1]
        b = self.regs[rs2]
        imm_i = _sx(inst >> 20, 12)
        imm_s = _sx(((inst >> 25) << 5) | ((inst >> 7) & 0x1F), 12)
        imm_b = _sx((((inst >> 31) & 1) << 12) | (((inst >> 7) & 1) << 11)
                    | (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1), 13)
        imm_u = inst & 0xFFFFF000
        imm_j = _sx((((inst >> 31) & 1) << 20) | (((inst >> 12) & 0xFF) << 12)
                    | (((inst >> 20) & 1) << 11) | (((inst >> 21) & 0x3FF) << 1), 21)
        illegal = ("fault", FAULT_ILLEGAL, CYCLES_STOP)
        npc = (pc + 4) & MASK
        cpi = cpi_simple(self.latency)
        cpi_m = cpi_mem(self.latency)
        dbus = ("fault", FAULT_DBUS, 3 + self.latency)

        def jump(target, link):
            target &= MASK
            if target & 2:
                return ("fault", FAULT_FETCH_MISALIGNED, CYCLES_EXEC_FAULT)
            if link is not None:
                self._wr(rd, link)
            self.pc = target
            return ("retire", cpi)

        if opc == 0b0110111:      # LUI
            self._wr(rd, imm_u)
        elif opc == 0b0010111:    # AUIPC
            self._wr(rd, pc + imm_u)
        elif opc == 0b1101111:    # JAL
            return jump(pc + imm_j, npc)
        elif opc == 0b1100111:    # JALR
            if f3:
                return illegal
            return jump((a + imm_i) & ~1, npc)
        elif opc == 0b1100011:    # BRANCH
            if f3 in (2, 3):
                return illegal
            cond = {0: a == b, 1: a != b, 4: _s32(a) < _s32(b), 5: _s32(a) >= _s32(b),
                    6: a < b, 7: a >= b}[f3]
            if cond:
                return jump(pc + imm_b, None)
            self.pc = npc
            return ("retire", cpi)
        elif opc == 0b0000011:    # LOAD
            if f3 not in (0, 1, 2, 4, 5):
                return illegal
            addr = (a + imm_i) & MASK
            size = 1 << (f3 & 3)
            if addr % size:
                return ("fault", FAULT_LOAD_MISALIGNED, CYCLES_EXEC_FAULT)
            try:
                word = self.word(addr & ~3)
            except BusError:
                return dbus
            value = (word >> (8 * (addr & 3))) & ((1 << (8 * size)) - 1)
            if f3 < 4 and size < 4:
                value = _sx(value, 8 * size)
            self._wr(rd, value)
            self.pc = npc
            return ("retire", cpi_m)
        elif opc == 0b0100011:    # STORE
            if f3 > 2:
                return illegal
            addr = (a + imm_s) & MASK
            size = 1 << f3
            if addr % size:
                return ("fault", FAULT_STORE_MISALIGNED, CYCLES_EXEC_FAULT)
            if (addr & ~3) + 4 > self.mem_size:
                return dbus
            self._store(addr, b, size)
            self.pc = npc
            return ("retire", cpi_m)
        elif opc == 0b0010011:    # OP-IMM
            if f3 == 1:
                if f7:
                    return illegal
                self._wr(rd, a << (rs2))
            elif f3 == 5:
                if f7 == 0:
                    self._wr(rd, a >> rs2)
                elif f7 == 0x20:
                    self._wr(rd, _s32(a) >> rs2)
                else:
                    return illegal
            else:
                self._wr(rd, self._alu(f3, a, imm_i & MASK, False))
        elif opc == 0b0110011:    # OP
            if f7 == 0x20 and f3 not in (0, 5) or f7 not in (0, 0x20):
                return illegal
            if f3 == 1:
                self._wr(rd, a << (b & 31))
            elif f3 == 5:
                self._wr(rd, (_s32(a) if f7 else a) >> (b & 31))
            else:
                self._wr(rd, self._alu(f3, a, b, f7 == 0x20))
        elif opc == 0b0001111:    # FENCE / FENCE.I
            if f3 > 1:
                return illegal
        elif opc == 0b1110011:    # SYSTEM
            if inst == 0x00000073:
                return ("halt", HALT_ECALL)
            if inst == 0x00100073:
                return ("halt", HALT_EBREAK)
            return illegal
        elif opc == 0b0001011:    # custom-0 vector command
            op = f7 & 0x1F
            self.pc = npc
            if op == 31:
                self.fences.append(len(self.vcmds))
                return ("retire", cpi_m)
            self.vcmds.append(VectorCommand(op, f3 & 3, rd & 7, rs2 & 7,
                                            ((f7 >> 5) << 1) | (f3 >> 2), a))
            return ("retire", cpi)
        else:
            return illegal
        self.pc = npc
        return ("retire", cpi)

    @staticmethod
    def _alu(f3: int, a: int, b: int, sub: bool) -> int:
        if f3 == 0:
            return (a - b) if sub else (a + b)
        if f3 == 2:
            return int(_s32(a) < _s32(b))
        if f3 == 3:
            return int(a < b)
        if f3 == 4:
            return a ^ b
        if f3 == 6:
            return a | b
        return a & b
