# pi-agent acceptance

Use pi's native IC extension to diagnose the counter fixture from
[PR #75](https://github.com/yenhao-huang/npu-pynq/pull/75).
Commands below run in Linux Bash, from the PR worktree.

## 1. Install with pi

Use Node.js 20.11+ and pi. In the existing experiment container, select pi 0.74.2:

```bash
export PATH="$HOME/.nvm/versions/node/v20.20.2/bin:$PATH"
cd /workspace/npu/worktrees/npu-issue74-a
```

Before switching, inspect `.pi/extensions/ic-design-tools.ts`. If it is the
original repository wrapper or the generated `ic-tools init pi` wrapper, move
it out of the extension directory to avoid duplicate tool registration:

```bash
mkdir -p .ic/disabled-extensions
mv -i .pi/extensions/ic-design-tools.ts .ic/disabled-extensions/
```

Skip that step when the file is absent. Preserve customized extensions.

### Install from npm

Use the published [npm package](https://www.npmjs.com/package/@jony2156/ai-eda-tools).
No venv, `IC_BIN`, or `ic-tools init pi` command is required.

```bash
pi install npm:@jony2156/ai-eda-tools@0.1.0 --local
pi
```

`--local` saves the npm package declaration in project settings.
The npm postinstall hook prepares the runtime. First setup downloads roughly
480–710 MiB plus Python dependencies; completed caches are reused. If scripts
were disabled, the extension prepares the runtime on first load. Native build
prerequisites still apply; see [platform requirements](../../../../tools/ic/docs/manual/installation.md#platforms-and-requirements).
Vivado is a separate licensed installation.

## 2. Check tool discovery

Pi reads `pi.extensions` from the package manifest and loads `managed.ts`,
which registers the eight tools. Start pi from the project root and enter
`/ic` to inspect backend availability. A skill is optional workflow guidance.
Configure a model provider in pi before the live acceptance prompt below.

## 3. Paste this acceptance prompt

```text
Use the eight native IC tools. Leave all source files unchanged.
1. lint: files=["src/hw/rtl/systolic_array/npu_pe.sv"], top="npu_pe".
2. sim: files=["tools/ic/tests/fixtures/counter.sv",
   "tools/ic/tests/fixtures/tb_counter.sv"], tb="tb_counter".
3. Use the returned wave handle for signals, pattern="*count*", then
   first_mismatch, ref="ref_count", dut="dut_count", context=3.
4. value_at: signals=["ref_count","dut_count","enable"] at the mismatch cycle.
   value_range: signal="dut_count", from_cycle=5, to_cycle=12, max_points=16.
5. show_wave: signals=["ref_count","dut_count","enable"], center_cycle set to
   the mismatch cycle, launch=false.
6. synth: files=["tools/ic/tests/fixtures/counter.sv"], top="counter_ref",
   mode="estimate".
Report actual tool results, run/wave handles and the cause of failure.
Use all eight native tools; do not replace them with Bash commands.
Do not read raw waveforms or whole logs. Do not repair this intentionally
broken fixture or claim timing closure.
```

## 4. Check the result

- The transcript contains all eight native calls; lint has zero errors.
- Simulation builds and returns a wave handle with an intentional failure.
- First mismatch: cycle **8**, reference **8**, DUT **7**. `value_at` agrees;
  `value_range` shows the DUT stalling at 7.
- `show_wave` returns a savefile and `launched: false`, as requested.
- `synth` returns utilization in estimate mode; timing is unavailable.

Record the commit, tool versions, verifier result, model, run/wave handles
and each failed or blocked check. SDK verification alone does not establish
that the model used all eight tools. See [full reproduction](reproduce.md)
for CLI examples, GUI checks and daemon jobs.

If tools are missing, check `pi list` and startup diagnostics, then restart pi.
Use only one package or wrapper registration. Do not set IC_BIN or activate
the old Python environment.

## Earlier wrapper verification (2026-09-27)

- `npm test` in `tools/ic`: 14 passed on Windows and Linux.
- Packed and installed the npm tarball into an isolated test prefix.
- Pi 0.74.2 SDK loaded the packaged adapter and executed all eight tools using
  the existing Python 3.11/HDL dependencies; mismatch cycle 8, ref 8, DUT 7.
  Local evidence: `.ic/pi-package-a44rrn/evidence.json`.
- Fresh managed-runtime acceptance was interrupted by very slow toolchain
  download. It is **not yet verified end to end**, nor was a live model tested.

To verify the complete managed path once downloads are available, run from
this repository root with pi visible to `npm root -g`:

```bash
ic-tools setup
node tools/ic/npm/tests/pi-smoke.mjs
```

Set `IC_SMOKE_PACKAGE_ROOT` to the installed npm package directory to test
that tarball instead of the source package. This SDK test calls all eight
native tools without model credentials, in a separate acceptance directory.

## Pi package release validation (2026-09-27)

The following checks supersede the managed-runtime limitation above:

- `node --test tools/ic/npm/tests/*.test.mjs`: 14 passed.
- Focused Python catalogue, registry and surface tests: 13 passed.
- Installed `ic-tools-0.1.0.tgz` into an isolated prefix. Its npm postinstall
  completed using an existing verified OSS CAD Suite runtime cache.
- `package-smoke.mjs`: `pi install --local` discovered all eight tools from
  the package manifest and executed lint without a project extension wrapper.
- `pi-smoke.mjs`: the installed managed adapter executed all eight tools;
  evidence is in `.ic/pi-package-F13nvs/evidence.json` on the experiment host.
- The pinned Verilator needs LZ4 development files for FST builds. This host
  lacked them; acceptance supplied Debian `liblz4-dev` 1.9.4-1 under
  `.ic/native-dev` through `CPLUS_INCLUDE_PATH` and `LIBRARY_PATH`.
  Installation requirements now include LZ4 development files.
- `npm publish --dry-run` passed for `ic-tools@0.1.0`.

These historical checks did not contact a model or publish to npm.
The current installation instructions above use the published scoped package.
The reusable integration check is:

```bash
node tools/ic/npm/tests/package-smoke.mjs
```

Use the same `IC_MCP_CACHE` as setup; `IC_SMOKE_PACKAGE_ROOT` optionally selects
an installed package. The test uses an isolated temporary project and pi settings.
