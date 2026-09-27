# Design

The npm package owns download, integrity checking, cache locking, extraction and
process startup. The Python registry remains the source of tool schemas and
behavior. Download metadata is pinned in the package, not fetched from latest.
OSS CAD Suite supplies Python and the open-source EDA executables. Python
requirements are pinned with hashes. No global installs or privileged operations.

Install into a unique staging directory and publish it by rename only after
dependency installation and preflight succeed. Cache identity includes platform,
archive digest and dependency lock digest. A lock serializes concurrent setup;
interrupted setup leaves no completion marker and can be retried. Never reclaim
an active lock automatically. All bootstrap output goes to stderr; stdout is
reserved for MCP. The child inherits stdin and signals are forwarded.

Preserve the caller's project directory. IC_ROOT controls artifacts independently
of the runtime cache. npm lifecycle scripts are deliberately unnecessary: setup
runs explicitly or on first server startup, including with --ignore-scripts.

Native Windows is not advertised until its Python extension ABI and simulator
subprocess behavior have been validated. WSL uses the Linux build. macOS targets
are present in the manifest but require platform acceptance before release.
