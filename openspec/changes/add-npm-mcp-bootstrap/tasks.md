- [x] Add npm package, pinned toolchain manifest and Python dependency lock.
- [x] Implement setup, cache reuse, doctor and stdio MCP startup.
- [x] Test corrupt downloads, failed setup, concurrency, packaging and stdout.
- [ ] Validate a fresh downloaded runtime through MCP simulation/debug/synthesis.
  - Blocked on this development host: the pinned 708 MiB archive downloads at
    10-15 KB/s from the GitHub release and parallel range requests time out, so
    no cold provisioning run can complete here. Startup, tool discovery, lint,
    simulation with tracing, waveform queries, synthesis estimation and clean
    shutdown were validated through the launcher against an equivalent local
    toolchain with `npm/tests/mcp-smoke.mjs`, including from an unpacked
    `npm pack` tarball. Download, verification and extraction are covered by
    `npm/tests/bootstrap.test.mjs` with stubs. Repeat the cold run on a host
    with normal network throughput before publishing.
- [x] Document platform support, cache lifecycle and agent configuration.
