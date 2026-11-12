/* link.ld - layout of every risc_v_npu program.
 *
 * One flat image linked at 0 and loaded anywhere (prog_addr): crt0.S and
 * the generated code address their own sections PC-relative only, and the
 * back end refuses an object with absolute relocations. The back end turns
 * the ELF into a flat binary of the loadable bytes (bin_bytes) and records
 * how much memory the program uses from its load address (mem_bytes: up to
 * __stack_top), which the driver reserves and zeroes.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

RVNPU_STACK_BYTES = 1024;

SECTIONS
{
  . = 0;
  .text : {
    KEEP(*(.text.rvnpu.start))
    *(.text .text.*)
  }
  .rodata : ALIGN(8) {
    *(.rodata .rodata.* .srodata .srodata.*)
  }
  .data : ALIGN(8) {
    *(.data .data.* .sdata .sdata.*)
  }
  .bss (NOLOAD) : ALIGN(8) {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(8);
    __bss_end = .;
  }
  .stack (NOLOAD) : ALIGN(16) {
    . += RVNPU_STACK_BYTES;
    __stack_top = .;
  }
  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr .comment .note .note.* .riscv.attributes)
  }
}
