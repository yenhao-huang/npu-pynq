## ADDED Requirements

### Requirement: Automatic runtime provisioning
The npm MCP entry point SHALL install missing runtime dependencies into a
user-owned cache without modifying system packages, using pinned URLs and hashes.

#### Scenario: First invocation
- WHEN a supported platform starts without a completed cache
- THEN download and verify its archive, extract it, install locked Python
  dependencies and start the MCP server only after preflight passes.

#### Scenario: Cached invocation
- WHEN the same runtime and dependency lock are already installed
- THEN start without accessing the network.

#### Scenario: Corruption or interruption
- WHEN a download checksum is wrong or setup fails
- THEN no completed cache is published and the invocation fails with an
  actionable diagnostic on stderr.

### Requirement: MCP transport integrity
The launcher SHALL expose the existing MCP server over stdio, preserve the
project working directory, keep setup messages off stdout, and forward shutdown.

#### Scenario: Stdio session
- WHEN an MCP client launches the package entry point in a project directory
- THEN initialize, tools/list and tools/call succeed over stdio while every
  bootstrap message stays on stderr and the project directory is preserved.

#### Scenario: Shutdown
- WHEN the client closes stdin or the launcher receives SIGINT or SIGTERM
- THEN the signal reaches the server process and the launcher exits with its
  status instead of orphaning it.

### Requirement: Platform and licensing boundaries
The launcher SHALL reject unsupported platforms before downloading and SHALL
not install Vivado or claim Yosys estimates provide signoff timing.

#### Scenario: Unsupported platform
- WHEN the package starts on a platform absent from the pinned manifest
- THEN it fails before any download and names the supported targets and the
  WSL path for Windows.

#### Scenario: Separately provisioned tools
- WHEN a caller expects Vivado, GUI waveform display or signoff timing
- THEN the package neither installs them nor presents Yosys estimates as
  signoff results.
