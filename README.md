# NPU PYNQ

[![CI](https://github.com/yenhao-huang/npu_in_pynq/actions/workflows/ci.yml/badge.svg)](https://github.com/yenhao-huang/npu_in_pynq/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/yenhao-huang/npu_in_pynq?sort=semver)](https://github.com/yenhao-huang/npu_in_pynq/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Board](https://img.shields.io/badge/board-PYNQ--Z1-8A2BE2)
![FPGA](https://img.shields.io/badge/FPGA-xc7z020clg400--1-00599C)

An open-source neural-network accelerator stack for the PYNQ-Z1. It connects
bit-accurate quantized model contracts, deterministic export, a Python runtime,
DMA-driven FPGA execution, reproducible Vivado builds, and Jupyter notebook
validation. Status: research and development.

## Contents

- [Features](#features)
- [How to build](#how-to-build)
- [Software Stack](#software-stack)
- [Hardware Architecture](#hardware-architecture)
- [Supported target and contracts](#supported-target-and-contracts)
- [Agentic Components](#agentic-components)
- [Contributing](#contributing)
- [License](#license)

## Features

| Capability | Status |
| --- | --- |
| Support Quantized ResNet-18 | Done |
| Customized EDA Tools | Done |
| Transformers | Done |
| RTL Optimization by Design Space Exploration | Undone |

## How to build

This README uses Linux Bash. The main example guides provide `README.md` for
WSL and `README.macos.md` for macOS Terminal, including platform build
limitations. Run commands from
the repository root unless a step says otherwise.
The SmolLM build guide uses `dist/` for generated model assets and packages.

### Quick start

1. Download the package on your computer:
   [npu-resnet18-v1.0.6.zip](https://github.com/yenhao-huang/npu-pynq/releases/download/v1.0.6/npu-resnet18-v1.0.6.zip).
2. Copy it to the PYNQ-Z1 (`<user>` is the board account, `xilinx` by default):

   ```bash
   scp npu-resnet18-v1.0.6.zip <user>@192.168.2.99:~/jupyter_notebooks/
   ```

3. Log in to the board:

   ```bash
   ssh <user>@192.168.2.99
   ```

4. Extract the package:

   ```bash
   cd ~/jupyter_notebooks
   unzip npu-resnet18-v1.0.6.zip -d npu-resnet18
   ```

5. Open the notebook in Jupyter at
   <http://192.168.2.99:9090/notebooks/npu-resnet18/resnet18.ipynb> and run the
   cells in order. Pick a bundled picture or upload your own; the last cell
   prints the top-5 ImageNet labels, their scores, and a CORRECT or INCORRECT
   verdict.

### Build from scratch

Go to `examples/<model>/` and follow its README to export the model, build the
overlay, and package it for the board:
[ResNet-18](examples/resnet18/README.md),
[matrix multiplication](examples/matrix-multiplication/README.md), and
[SmolLM](examples/smollm-safetensors/README.md).

## Software Stack

```text
       ONNX / safetensors / TorchVision
                 |
                 v
 HOST       +----------+  MLIR  +------------+  LLVM-IR  +------+
 (compile)  | Frontend +------->|  NPU-MLIR  +---------->| LLVM |
            | (graph)  |        |            |           +--+---+
            +----------+        | Quant,     |              |
                                | Partition, |              |
                                | Tiling,    |              |
                                | Lowering   |              |
                                +-----+------+              |
                                      | custom ISA          | CPU code
 -------------------------------------+---------------------+-------
 BOARD                                v                     v
 (PYNQ-Z1)  +------------------------------------------------------+
            |                       Runtime                        |
            +---------------------+-----------------+--------------+
                                  v                 |
                           +-------------+          |
                           | NPU driver  |          |
                           |    (CPU)    |          |
                           +------+------+          |
                                  v                 v
                           +-------------+   +-------------+
                           |     NPU     |   |  Cortex-A9  |
                           +-------------+   +-------------+
```

Note:

- **Host compile:** `python -m src.export.cli export` turns a model into a
  package.
- **Board run:** `python -m src.runtime.cli llm` loads that package on the
  PYNQ-Z1 and runs inference.
- **NPU driver:** runs on the Cortex-A9. For each NPU matrix multiplication it
  packs the activations into tiles, writes the program address to the NPU's
  registers, starts it, polls until it finishes, and unpacks the result.

## Hardware Architecture

![PYNQ-Z1 NPU hardware architecture](docs/assets/npu-hardware-architecture.svg)

The Zynq processing system drives the accelerator (controller, operand memory,
systolic array) over DMA and AXI.

## Supported target and contracts

| Area | Supported boundary |
| --- | --- |
| Board | PYNQ-Z1 |
| FPGA part | Zynq-7020, `xc7z020clg400-1` |
| Overlay | `npu_matrix` with AXI DMA |
| Arithmetic | Signed INT8 operands, exact INT16 products, saturating INT32 accumulation |
| Requantization | Q1.31 with explicit rounding and saturation contracts |
| Host tooling | Python and Bash (Linux/WSL); Vivado only for hardware builds |

## Agentic Components

### IC design tools

Give Claude Code, Codex, pi-agent and other compatible agents tools for RTL
linting, simulation, waveform debugging and synthesis. Install in Linux/WSL
or macOS with Node.js 20.11+:

```bash
npm install -g --foreground-scripts @jony2156/ai-eda-tools@0.1.1
ic-tools doctor
```

Connect Claude Code from your project root:

```bash
claude mcp add --transport stdio --scope project ic-tools -- \
  "$(command -v ic-tools)" serve --project "$PWD"
claude mcp list
claude
```

See the [tool guide](tools/ic/README.md) for capabilities and
[agent installation guide](tools/ic/docs/manual/installation.md) for Codex, pi
and environment requirements.

### Harness Engineering

- **Agent instructions:** [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md)
  define repository context and working rules.
- **EDA skills:** [`.codex/skills/`](.codex/skills/) provides reusable
  development, deployment and IC design workflows, including hardware builds
  and PYNQ board delivery. Skills guide the agent's workflow; tools execute
  the design operations.
- **Git management:** [docs/rules/git/](docs/rules/git/) defines issue,
  branch, commit and pull-request workflows.
- **Environment and structure:** [environment.md](docs/rules/environment.md)
  and [filetree.md](docs/rules/filetree.md) define tool assumptions and where
  source, tests and generated artifacts belong.
- **Spec-Driven Development:** use the OpenSpec skills to propose, implement
  and archive changes under [openspec/](openspec/).

## Contributing

Work is issue-scoped: read [AGENTS.md](AGENTS.md) and the
[Git rules](docs/rules/git/) first. Pull requests target `dev`; promoting
`dev` to `main` is a human release decision.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
