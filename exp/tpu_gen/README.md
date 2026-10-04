# TPU-Gen POC

Reproduction of the TPU-Gen pipeline (ICLAD 2025, [arXiv:2503.05951](https://arxiv.org/abs/2503.05951))
as a modular, runnable flow:

```
user prompt -> prompt_formatter -> LLM -> .vh -> retrieval -> full .v
            -> Yosys -> OpenROAD -> GDSII + PPA   (repeat)
```

**OpenROAD is the only backend.** Area, WNS and power come from a real
place-and-route run or the flow fails; nothing here estimates PPA.

See [plan.md](plan.md) for the design, the verified environment facts and the
open risks.

## Layout

| Path | What |
| --- | --- |
| `demo.ipynb` | the end-to-end walkthrough |
| `run_flow.py` | the same pipeline, headless |
| `lib/prompt_formatter/` | user request -> the trained prompt format |
| `lib/llm/` | backend registry: `replay`, `claude`, `codex`, `anthropic`, `openai` |
| `lib/retrieval/` | `.vh` -> the `.v` files the design needs |
| `lib/synthesis/` | staging and iverilog pre-flight |
| `lib/openroad/` | ORFS runner: RTL -> GDSII + PPA |
| `lib/validate/` | tool output -> feedback for the next round |
| `lib/flow/` | the multi-round orchestrator |
| `src_code/` | ignored upstream TPU-Gen checkout (RTL library, datasets) |
| `runs/` | per-run artifacts, untracked |

The local OpenROAD measurements and their limits are in
[docs/results.md](docs/results.md). They are ASIC Nangate45 results, not PYNQ
FPGA resource or timing results.

## Setup

```bash
git clone https://github.com/ACADLab/TPU_Gen.git src_code
git -C src_code checkout 0abb8511369f859d3286d31caab2c37ab6bddde9
mkdir -p src_code/rtl-20250603T193300Z-1-001
python3 -m zipfile -e src_code/rtl-20250603T193300Z-1-001.zip src_code/rtl-20250603T193300Z-1-001
docker image inspect openroad/orfs:26Q3-605-g2d29bdaf8
python3 -m venv .venv && .venv/bin/pip install ipykernel nbclient nbformat
```

Run these commands from `exp/tpu_gen/`. The upstream commit contains the RTL
archive and train/test JSON files; extraction creates the `rtl/` directory used
by this flow. Keep the upstream checkout in `src_code/`, which is ignored by Git.
The upstream repository and model datasets are not copied into this PR.

`lib/` itself is standard library only; the venv is for the notebook.
The validated default image is `openroad/orfs:26Q3-605-g2d29bdaf8`. To use
another local ORFS image, set `TPUGEN_ORFS_IMAGE` (for the notebook and CLI),
pass `--openroad-image` to `run_flow.py`, or set `OPENROAD_IMAGE` in the notebook.
The selected image must contain `/OpenROAD-flow-scripts/env.sh` and the
`nangate45` platform. No image is downloaded automatically.

## Run

```bash
python3 run_flow.py "4x4 INT8 TPU with DRUM_APTPU multiplier and APPROX5 adder"
python3 run_flow.py "..." --mode full          # skip retrieval, for comparison
python3 run_flow.py "..." --iterations 3 --tolerance 0.2
python3 run_flow.py "..." --backend claude      # Claude Code CLI, one-shot
python3 run_flow.py "..." --backend codex       # OpenAI Codex CLI, one-shot
python3 run_flow.py "..." --backend codex --model gpt-5.6-luna  # verified on this host
python3 run_flow.py "..." --openroad-image openroad/orfs:latest
python3 -m unittest discover -s tests          # everything that needs no OpenROAD
```

## LLM backends

`lib/llm/registry.py` is the registry; `get_backend(name)` builds one and
`available_backends()` lists them. Add one with
`register("name", factory, "one line")`.

| name | what runs | needs |
| --- | --- | --- |
| `replay` | nearest neighbour over `beta_train_all.json` | nothing |
| `claude` | `claude -p` (Claude Code CLI) | the CLI on PATH, signed in |
| `codex` | `codex exec` (OpenAI Codex CLI) | the CLI on PATH, signed in |
| `anthropic` | Anthropic Messages API | `anthropic` package, `ANTHROPIC_API_KEY` |
| `openai` | OpenAI Chat Completions API | `openai` package, `OPENAI_API_KEY` |

The two CLI backends run the agent non-interactively with its tools off
(`--permission-mode plan` / `--sandbox read-only`) in a throwaway directory:
the prompt is the whole task and the reply is the header text.

For `codex`, pass `--model` when the CLI's global default is not supported by
its logged-in account. On this host the global `gpt-6-sol` setting is rejected
by the CLI; `gpt-5.6-luna` responded to a bounded header-generation probe.
The notebook supplies that model automatically when `AGENT = "codex"`.

Each round leaves a self-contained directory under `runs/<run-id>/iter_NN/`:
the prompt, the generated header, what retrieval selected, the staged sources,
the full ORFS log, the GDSII and the metrics.
