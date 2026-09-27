# Environment

Python follows tools/ic/pyproject.toml (>=3.10; this run uses Python 3.12).
Use a repository-local .venv. Install the declared package and optional adapters;
prefer tools/ic/npm/requirements.lock for the runtime dependency baseline.
Do not install globally. pytest and PyYAML support tests and skill validation.

Icarus (iverilog and vvp) is needed for comb_check. Local licensed Vivado is
needed for ppa_measure. GNU Make, Verilator and Icarus support repository gates.
No server or board is required. Run ic doctor and record actual versions.
Do not change licenses, network adapters, board state or tool installations to
make an unavailable validation gate appear to pass.
