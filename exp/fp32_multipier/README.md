# FP12 run_eval

Read [report.md](report.md) for measurements and [reproduce.md](reproduce.md) to replay the best three frozen designs.

- `generate.py`, `architectures.py`: modular candidate RTL generation.
- `model/reference.py`, `test/`: exact arithmetic references and verification.
- `run_eval.py`: Verilator → Yosys/ABC → gate verification → OpenROAD adapter.
- `sta.py`: pinned Docker OpenROAD timing and power extraction.
- `campaign.py`, `refine.py`, `refine_timing.py`: recorded exploration rounds.
- `resume_sta.py`: validated checkpoint recovery.
- `reproduce.py`: frozen-source full-domain replay and numerical agreement gate.
- `summarize.py`: evidence-derived CSV, ranking and report.
- `runs/`: immutable original experiment snapshots, per-stage commands/logs and metrics.
- `.venv/`: requested local Linux Python environment; no third-party packages needed.

Numerical specification and PPA constraints: [docs/contract.md](docs/contract.md).
