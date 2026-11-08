"""Hardware vendors. Each folder holds everything one accelerator needs: its
instruction set (``isa/``), its compiler backend (``compiler/``) and its
board-side driver (``driver/``). ``src/export`` and ``src/runtime`` import
from here; nothing here imports ``src.export`` or ``src.runtime``."""
