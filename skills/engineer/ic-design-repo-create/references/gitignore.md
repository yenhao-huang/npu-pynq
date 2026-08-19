# Ignore Rules

Baseline for a Vivado repository:

```gitignore
# Vivado project directories: regenerate from src/vivado_tcl/
vivado_projects/
*.cache/
*.gen/
*.hw/
*.ip_user_files/
*.runs/
*.sim/
*.xpr
.Xil/

# Vivado logs and journals
*.jou
*.log
*.str
vivado_pid*.*

# Build outputs
build/
results/
*.bit
*.bin
*.dcp
*.ltx
*.xsa

# Simulation
sim/waves/
sim/build/
*.vcd
*.fst
*.wdb
xsim.dir/
obj_dir/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
Thumbs.db
.DS_Store
```

## Bitstream exception

A `.bit` that is a *deploy payload* — already flashed, needed by a notebook,
paired with its `.hwh` — may be tracked deliberately with a negation such as
`!sw/overlay/*.bit`. Track it only when the board cannot be provisioned any
other way, and keep it out of the build output tree. Prefer a Release asset.

## Verifying

After installing the file, confirm nothing already-tracked is now ignored:

```bash
git ls-files -i -c --exclude-standard
```

Output here means those files are tracked despite matching a rule. Decide each
one; `.gitignore` does not untrack anything already committed.
