# AI EDA Tools

Lint, simulate, inspect waveforms and synthesize RTL from your coding agent.
The npm package is `@jony2156/ai-eda-tools`; its CLI is `ic-tools`.
Claude Code connects through MCP, and pi loads a native extension.
Large waveforms and logs stay on disk; tools return summaries and artifact handles.

## Contents

- [Quick Start](#quick-start)
- [Tools](#tools)
- [Installation](#installation)
- [Use with Claude Code](#use-with-claude-code)
- [Use with pi](#use-with-pi)

## Quick Start

Requires Node.js 20.11+, npm and Claude Code in Linux/WSL or macOS.
Run from your RTL project root. Install the published package from the
[official npm registry](https://www.npmjs.com/package/@jony2156/ai-eda-tools):

```bash
npm install -g --foreground-scripts @jony2156/ai-eda-tools@0.1.0
ic-tools doctor
```

Register the installed tools with Claude Code from the same project root:

```bash
claude mcp add --transport stdio --scope project ic-tools -- \
  "$(command -v ic-tools)" serve --project "$PWD"
claude mcp list
claude
```

The registration command writes an `ic-tools` server entry to `.mcp.json`,
using the installed CLI's absolute path and the current project directory.
If `.mcp.json` already contains the legacy `ic-design-tools` server using
`ic-mcp`, replace that entry before registration and preserve other servers.
If you use a custom runtime cache, add `--env IC_MCP_CACHE="$IC_MCP_CACHE"`
before `--` in the command above.

Inside Claude Code, approve the project MCP server when prompted. Run `/mcp`
and confirm that `ic-tools` is connected. In an NPU repository checkout, ask:

```text
Use the IC lint tool on src/hw/rtl/systolic_array/npu_pe.sv,
top npu_pe. Report errors and warnings. Leave the source unchanged.
```

For other agents, such as Codex or pi, see [docs/manual/installation.md](docs/manual/installation.md).

## Tools

| Tool | Purpose | Main inputs |
| --- | --- | --- |
| `lint` | Check RTL and return errors and warnings | `files`, `top` |
| `sim` | Run a testbench; return its verdict, wave and log handles | `files`, `tb` |
| `signals` | List waveform signals and scopes | `wave`, `pattern` |
| `first_mismatch` | Find the first divergent cycle | `wave`, `ref`, `dut` |
| `value_at` | Read signals at one cycle or time | `wave`, `signals`, `cycle` |
| `value_range` | Inspect a signal's transitions over a range | `wave`, `signal`, `from_cycle`, `to_cycle` |
| `show_wave` | Prepare a GTKWave view for a person | `wave`, `signals`, `center_cycle`, `launch` |
| `synth` | Estimate utilization with Yosys or run full synthesis with Vivado | `files`, `top`, `mode` |

A typical workflow is `lint` → `sim` → `first_mismatch` → `value_at`.
Use the `wave` handle returned by your simulation for subsequent queries.
Set `show_wave.launch=false` on headless systems. Yosys `mode="estimate"`
does not provide timing; `mode="full"` requires a separate Vivado installation.

These are agent tool names. The npm CLI exposes `setup`, `doctor`, `serve`
and `init pi`; it does not accept `ic-tools lint`.

## Installation

### Runtime dependencies

npm's `postinstall` runs setup, downloading a pinned, checksum-verified OSS
CAD Suite and installing hash-locked Python dependencies. The first setup
downloads about 480–710 MiB plus Python packages; later runs reuse the cache.
No Python venv activation is required.

- Linux/WSL: provide a C++ compiler, Make, tar, zlib and LZ4 development files
  for simulation builds. On Debian/Ubuntu these include `build-essential`,
  `tar`, `zlib1g-dev` and `liblz4-dev`.
- macOS: provide a compatible native build toolchain; GUI viewing requires
  a desktop session.
- Native Windows is unsupported; run Node, npm and the agent inside WSL.
- Vivado is separately installed and licensed.

If install scripts were skipped, run `ic-tools setup` before starting the agent.
`ic-tools doctor` checks backend availability.

### Registry installation

Install version 0.1.0 from npm:

```bash
npm install -g --foreground-scripts @jony2156/ai-eda-tools@0.1.0
ic-tools doctor
```

### Existing runtime and uninstall

To reuse a completed runtime stored elsewhere, set `IC_MCP_CACHE` to its
cache root before installation and agent startup. Defaults are
`~/.cache/ic-tools` on Linux and `~/Library/Caches/ic-tools` on macOS.
Project results are stored separately under `.ic/` by default.

```bash
npm uninstall -g @jony2156/ai-eda-tools
```

Uninstalling preserves runtime caches and project results. Remove the agent's
IC registration as well. See [installation details](docs/manual/installation.md)
for cache reuse, migration and troubleshooting.

## Use with Claude Code

After installation, run from the RTL project's root:

```bash
claude mcp add --transport stdio --scope project ic-tools -- \
  "$(command -v ic-tools)" serve --project "$PWD"
claude mcp list
claude
```

This writes the project MCP configuration to `.mcp.json`. Inside Claude Code,
approve the project server when prompted and use `/mcp` to check the connection.
If this repository still has the old `ic-design-tools` entry using `ic-mcp`,
replace that entry before adding this server to avoid duplicate tools.
Preserve unrelated MCP entries.

For a custom runtime cache, add `--env IC_MCP_CACHE="$IC_MCP_CACHE"` before
`--` in the registration command. The saved executable and project paths are
machine-specific; register again after moving them.

Try this from the NPU repository root:

```text
Use the IC tools to simulate tools/ic/tests/fixtures/counter.sv and
tools/ic/tests/fixtures/tb_counter.sv with tb=tb_counter. Use the returned
wave handle to compare ref_count and dut_count with first_mismatch.
Report the first divergent cycle and both values. Keep the files unchanged.
```

The fixture intentionally fails: the expected first mismatch is cycle 8,
reference 8, DUT 7. Registration is successful when Claude actually calls the
tools and returns their results. See the
[Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

## Use with pi

From the repository root, with pi already installed:

```bash
pi install npm:@jony2156/ai-eda-tools@0.1.0 --local
pi
```

pi reads the package's `pi.extensions` declaration and registers the eight
tools. The extension prepares the runtime on first use if needed. Inside pi,
run `/ic` to inspect backends. A skill is optional workflow guidance.
Move any previous `.pi/extensions/ic-design-tools.ts` wrapper aside before
using package discovery to avoid registering the tools twice.

`--local` saves the registry package declaration in this project's settings.
See the
[pi acceptance guide](../../docs/goals/ic-design-tools/acceptance/pi-agent.md) for all eight checks.

## Verification demo

See [pi prompts and expected results](docs/manual/demo.md) to verify tool loading,
lint, simulation, waveform inspection and synthesis estimates.

## Paper-informed RTL optimization

The registry also includes `optimization_rules`, `width_advice`, `comb_check`,
`ppa_measure`, `ppa_provenance`, `ppa_compare`, `ppa_pareto`, `ppa_reward`,
`ppa_select`, and `rtl_evaluate`. The pipeline composes a bounded combinational
check with local Vivado measurements and a compatible comparison. Use `ic tools`
for their schemas. Complex CLI fields accept JSON objects or JSON object list
items; MCP and HTTP accept native structured values.

See the [exploration report](../../docs/goals/0928-tool-exploration/report.md)
and [reproduction guide](../../docs/goals/0928-tool-exploration/reproduce.md).
The checker does not prove sequential equivalence. Measurements report routed
FPGA LUT/FF counts and combinational datapath delay; power and board performance
are not measured by this flow.
