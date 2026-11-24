"""Hardware vendors. Each folder holds what one accelerator needs at run time:
its hardware facts (``spec/``), read by both export and runtime, and its
board-side driver (``driver/``). Compile-time code for a device belongs to the
export tools, not here. Export and runtime import from here; nothing here
imports either of them."""
