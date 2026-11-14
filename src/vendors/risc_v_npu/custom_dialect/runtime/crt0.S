# crt0.S - start-up code linked into every risc_v_npu program (RV32IMC).
#
# The program is one flat, position-independent image loaded at prog_addr
# (CVA6's boot address), entry _start at offset 0 (link.ld). Everything here
# is PC-relative (lla, call); only the MMIO window is absolute.
#
#   _start  mtvec = trap handler, sp = top of the stack link.ld reserves,
#           clear .bss, a0 = rvnpu_main(), store a0 to DONE (0 = success)
#   trap    store RVNPU_DONE_TRAP to DONE
#
# A DONE write ends the run: the hardware puts CVA6 back into reset, so both
# paths only spin afterwards. .data is not reloaded between runs (the driver
# copies the binary once), so nothing may rely on initialised writable data.
#
# No preprocessor: the back end assembles this with LLVM MC after a block of
# `.equ` lines generated from RvNpuMmio.h (mlir::rvnpu::mmioEquates), so the
# RVNPU_* names below are that header's values.

    .section .text.rvnpu.start, "ax", @progbits
    .globl  _start
    .type   _start, @function
_start:
    lla     t0, rvnpu_trap
    csrw    mtvec, t0
    lla     sp, __stack_top
    lla     t0, __bss_start
    lla     t1, __bss_end
1:  bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
2:  call    rvnpu_main
    li      t0, RVNPU_MMIO_BASE
    sw      a0, RVNPU_MMIO_DONE(t0)
3:  j       3b
    .size   _start, . - _start

    # mtvec in direct mode needs a 4-byte aligned handler.
    .p2align 2
    .type   rvnpu_trap, @function
rvnpu_trap:
    li      t0, RVNPU_MMIO_BASE
    li      t1, RVNPU_DONE_TRAP
    sw      t1, RVNPU_MMIO_DONE(t0)
4:  j       4b
    .size   rvnpu_trap, . - rvnpu_trap
