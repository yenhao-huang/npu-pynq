# npm entry point for IC MCP

Related to issue #74. Users need an MCP command that prepares its own open-source
toolchain rather than requiring a repository checkout and a configured Python.

Add an npm-distributed stdio launcher under tools/ic. On first invocation it
downloads a pinned OSS CAD Suite archive, verifies SHA256, installs hash-locked
Python dependencies into an isolated cache, and starts the existing MCP server.
Provide setup and doctor commands and reuse a completed installation offline.
Package the existing Python sources rather than duplicating or rewriting them.

Initial native targets are Linux and macOS x64/arm64; Windows users run the Linux
package in WSL. Vivado and GUI display services remain separately provisioned.
Publishing to npm and changing agent configuration are outside this change.
