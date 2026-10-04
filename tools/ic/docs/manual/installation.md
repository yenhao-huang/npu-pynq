# AI EDA tools installation

The npm package is `@jony2156/ai-eda-tools`; its CLI command remains `ic-tools`.

Expose nine tools to your agent: `lint`, `sim`, `signals`, `first_mismatch`,
`value_at`, `value_range`, `show_wave`, `synth` and `ppa`.

- [MCP-based installation](#mcp-based-installation): Claude Code and Codex.
- [Others installation](#others-installation): pi.
- [Runtime and troubleshooting](#runtime-and-troubleshooting): shared environment setup.

For copyable verification prompts and expected results, see [the demo](demo.md).

## Prerequisites

Install Node.js 20.11+ and your chosen agent first, and configure its model
provider. Run the Bash commands below from the repository root, in the same
Linux/macOS environment as the agent. Windows users must use WSL.
See [platform requirements](#platforms-and-requirements).

For the fastest startup, install OpenROAD before installing the npm package and
confirm it is on PATH:

```bash
openroad -version
```

Use an official prebuilt package or follow the
[OpenROAD installation guide](https://openroad.readthedocs.io/en/latest/user/Build.html).
Alternatively install and start Docker. When native OpenROAD is absent, npm
automatically pulls a pinned official `openroad/orfs` image. Setup intentionally
does not invoke a system package manager or `sudo`.

If you already have a runtime, select its [existing cache](#reuse-an-existing-environment)
before installing. No Python venv activation is required.

## MCP-based installation

### Install ai-eda-tools

Install the published package from the
[official npm registry](https://www.npmjs.com/package/@jony2156/ai-eda-tools):

```bash
npm install -g --foreground-scripts @jony2156/ai-eda-tools@0.1.0
ic-tools doctor
```

npm runs `ic-tools setup` automatically through `postinstall`. Then register
it with your chosen agent below. The agent starts the MCP server when needed.

### Claude Code

```bash
claude mcp add --transport stdio --scope project ic-tools -- \
  "$(command -v ic-tools)" serve --project "$PWD"
claude mcp list
claude
```

Inside Claude Code, use `/mcp` to inspect the connection. Project configuration
is stored in `.mcp.json`. If it already contains the old `ic-design-tools`
server using `ic-mcp`, replace that entry to avoid exposing duplicate tools.

### Codex

```bash
codex mcp add ic-tools -- "$(command -v ic-tools)" serve --project "$PWD"
codex mcp list
codex
```

This registers the server in Codex's user configuration. The saved `--project`
path selects this checkout even when Codex is launched elsewhere.

For either agent, if you selected a custom `IC_MCP_CACHE`, also pass
`--env IC_MCP_CACHE="$IC_MCP_CACHE"` to the registration command before `--`.
This preserves the cache choice when the agent is launched from another shell.

Ask the agent to call `lint` on `src/hw/rtl/npu_matrix/npu_accelerator/npu_matrix_core/npu_matrix_datapath/systolic_array/npu_pe.sv`, top
`npu_pe`. See the [tool acceptance guide](../../../../docs/goals/ic-design-tools/acceptance/reproduce.md)
for more checks. Other MCP clients can launch the absolute `ic-tools` path with
`serve --project /absolute/path/to/project` using stdio transport.

## Others installation

### pi

Before installing, handle any [existing wrapper](#migrate-an-existing-pi-wrapper).
Install the package from npm:

```bash
pi install npm:@jony2156/ai-eda-tools@0.1.0 --local
pi
```

`--local` records the npm package in this project's `.pi/settings.json`;
the package is downloaded from the registry.

pi loads the extension declared in `package.json` and registers the tools.
No MCP configuration or `ic-tools init pi` command is needed. npm installation
prepares the runtime through `postinstall` and reuses completed caches.

Inside pi, run `/ic` to inspect backends, then ask it to call `lint` on
`src/hw/rtl/npu_matrix/npu_accelerator/npu_matrix_core/npu_matrix_datapath/systolic_array/npu_pe.sv`, top `npu_pe`.
See [pi acceptance](../../../../docs/goals/ic-design-tools/acceptance/pi-agent.md) for all eight tools.

### Migrate an existing pi wrapper

If `.pi/extensions/ic-design-tools.ts` exists, inspect it. If it is the unchanged
repository wrapper or a generated `ic-tools init pi` wrapper, move it aside
before `pi install` to avoid duplicate registration:

```bash
cat .pi/extensions/ic-design-tools.ts
mkdir -p .ic/disabled-extensions
mv -i .pi/extensions/ic-design-tools.ts .ic/disabled-extensions/
```

Skip this step when absent; preserve customized extensions.

## Runtime and troubleshooting

### Reuse an existing environment

Completed runtimes are reused automatically from `~/.cache/ic-tools` on Linux
or `~/Library/Caches/ic-tools` on macOS. To use another location, set
`IC_MCP_CACHE` to the cache root before installation and agent startup.

For the existing howard sandbox environment:

```bash
export IC_MCP_CACHE="$HOME/.cache/ic-tools/ic-design-mcp"
```

This is the parent of the `2026-09-23-linux-x64-b8812dcc213d8b72` runtime
directory, which contains `complete.json`, `oss-cad-suite/` and
`python-packages/`. This example applies inside that Linux sandbox.
The older directory name can remain; it does not change the command name.

### First setup and recovery

The current implementation first checks native OpenROAD. When it is absent,
setup verifies Docker and pulls
`openroad/orfs:26Q1-534-g510137693`, pinned by its Linux/amd64 manifest digest.
ARM hosts therefore need Docker's amd64 emulation support. Setup then downloads
a pinned
[OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) archive
(approximately 480–710 MiB) and hash-locked Python wheels. It still uses this
managed bundle. npm installs Node dependencies, then `postinstall` prepares
this runtime; system packages listed below must be installed separately.

| Command | Purpose |
| --- | --- |
| `ic-tools setup` | Prepare or retry runtime setup; reuse completed caches. |
| `ic-tools doctor` | Prepare the runtime if missing, then list available backends. |
| `ic-tools serve --project DIR` | Start the stdio MCP server for a project. |

For pi installations without a global CLI, run `/ic` inside pi to check
backends. To use the standalone commands, install the global npm package above.

- `--ignore-scripts` skips npm setup; first agent load can prepare the runtime.
- If setup fails, fix the reported cause and retry. If npm removed the CLI,
  rerun the install command.
- For an interrupted setup lock, confirm its process has exited before removing
  that specific lock and abandoned staging directory. Preserve active caches.
- `IC_MCP_ARCHIVE=/absolute/path/to/archive.tgz` supplies the pinned archive
  locally; size and SHA256 are checked. First setup still needs Python wheels
  from the network. Completed caches work offline.
- `IC_ROOT` selects the project artifact store independently of the runtime cache.
- `IC_OPENROAD_MODE=auto` prefers native OpenROAD and falls back to Docker.
  Set it to `host` or `docker` to require one provider explicitly.

### Uninstall

Remove the integration you installed:

```bash
# pi registry installation; run from the same project root.
pi remove npm:@jony2156/ai-eda-tools@0.1.0 --local

# Separately installed global CLI; use the same Node/NVM environment.
npm uninstall -g @jony2156/ai-eda-tools
```

If a different version was installed, use the exact source shown by `pi list`
with the same scope. Remove the IC server entry from each
MCP client's configuration, or disable a legacy pi wrapper, then restart the
agent. Runtime caches and project `.ic/` evidence remain on disk.

## Platforms and requirements

- Node.js >=20.11. Pinned manifests cover Linux x64/arm64 and macOS x64/arm64;
  see acceptance records for platforms actually tested.
- Native Windows is unsupported; run Node/npm and the agent inside WSL.
- OpenROAD is required either on PATH or through a running Docker daemon. The
  Docker fallback pulls a pinned official image and bind-mounts only the
  project, run, and referenced PDK directories. A standard-cell PDK is supplied
  separately when invoking `ppa`; npm does not install technology files.
- Linux simulation requires a C++ compiler, Make, and zlib/LZ4 development files
  (`build-essential`, `zlib1g-dev`, `liblz4-dev` on Debian). npm does not install them.
- macOS needs an upstream-supported OS version and may need Xcode command-line
  tools for Verilator builds. GUI waveform viewing requires a desktop/display.
- Vivado requires a separate licensed installation on PATH. Yosys estimates
  provide no timing signoff.
