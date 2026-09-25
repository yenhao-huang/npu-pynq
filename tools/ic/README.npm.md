# IC design MCP

An npm entry point for the existing IC design MCP server. First startup downloads
a pinned [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) archive,
verifies its SHA256 and installs hash-locked Python dependencies in a user cache.
Python, Verilator, Icarus, Yosys and waveform tools come from that suite. No Docker,
global Python install, sudo or agent skill is needed. Vivado is not included.

The package is **not yet published to npm**. `ic-design-mcp` is the proposed name;
registry ownership and public availability are not assumed. Build a local tarball:

```bash
cd tools/ic
npm ci --ignore-scripts
npm test
npm pack --pack-destination /tmp
npm install --prefix /tmp/ic-client /tmp/ic-design-mcp-0.1.0.tgz --ignore-scripts
/tmp/ic-client/node_modules/.bin/ic-design-mcp setup
```

The installed `ic-design-mcp` binary starts a stdio MCP server by default. An MCP
client can configure its absolute path as `command` and pass
`["--project", "/absolute/path/to/your/rtl-project"]` as `args`. Once a version is
published, the equivalent configuration is:

```json
{
  "mcpServers": {
    "ic-design": {
      "command": "npx",
      "args": ["-y", "ic-design-mcp@0.1.0", "--project", "/absolute/path/to/project"]
    }
  }
}
```

MCP tools are discovered through `tools/list` and invoked with `tools/call`.
The server exposes lint, sim, signals, first_mismatch, value_at, value_range,
show_wave and synth. Source paths resolve against `--project`, or the launcher's
working directory when omitted. Setup output is stderr only. stdout carries MCP.

## Commands

- `ic-design-mcp setup`: download and prepare dependencies ahead of the client
  handshake. Recommended on first use because MCP clients may have short startup
  timeouts. Initial download is approximately 480–710 MiB plus Python wheels.
- `ic-design-mcp doctor`: automatically prepare dependencies if absent, then show
  the backend inventory. Vivado may be unavailable even when defaults are ready.
- `ic-design-mcp [serve] --project DIR`: reuse the cache or set it up, then serve.

There is no npm postinstall hook. `npm install --ignore-scripts` works. Provisioning
happens at first invocation so installation does not silently download large EDA
archives, and a failed download can be retried independently of npm installation.

## Platforms and requirements

- Node.js >=20.11. Linux x64/arm64 and macOS x64/arm64 have pinned manifests.
- Windows: run Node/npm and this command inside WSL. Native Windows is rejected
  before download; it is not advertised as compatible.
- macOS needs an upstream-supported OS version and may require Xcode command-line
  tools for Verilator's C++ build. GUI viewing requires a desktop/display.
- Linux requires a runnable native toolchain environment. Check the acceptance
  record for platforms actually tested; a manifest entry alone is not validation.
- Vivado must be separately installed/licensed and on PATH for full synthesis.
  Yosys estimates contain no timing signoff.

## Cache, failure recovery and offline use

`IC_MCP_CACHE` overrides the default user cache (`~/.cache/ic-design-mcp` on Linux,
`~/Library/Caches/ic-design-mcp` on macOS). `IC_ROOT` independently selects the
project run/artifact store. Runtime identity includes platform and dependency
hashes. Completed caches are reused without network access; failed setup never
publishes a completion marker. Concurrent clients serialize installation.

`IC_MCP_ARCHIVE=/absolute/path/to/archive.tgz` supplies the pinned upstream archive
locally; its size and SHA256 are still verified. Python wheels require network
during first setup. Completed caches work offline. Setuptools/source builds are
not accepted for Python dependencies: setup uses hash-checked binary wheels.

On checksum failure, remove only the reported corrupt archive and retry. A killed
installer may leave a lock and staging directory. After confirming its process
has exited, remove that specific lock and abandoned staging directory, then run
setup. Active caches and project artifacts are not automatically deleted. npm
uninstall removes the launcher, not its reusable runtime cache or project runs.

Upstream licenses remain in the extracted archive. Do not include Vivado in a
redistributed archive. Updating tools requires a reviewed manifest/digest update,
dependency lock refresh and fresh MCP acceptance, not a mutable `latest` URL.
