# CI and CD

## The constraint that shapes everything

GitHub-hosted runners cannot run Vivado, Quartus, or any licensed EDA tool. The
images are tens of gigabytes and the licences are node-locked. This splits the
pipeline in two:

| Stage | Tool | Runner |
| --- | --- | --- |
| Lint | verible, Verilator `--lint-only` | GitHub-hosted |
| Simulation | Icarus, Verilator, cocotb | GitHub-hosted |
| Synthesis, implementation, bitstream | Vivado | self-hosted |
| Board test | Vivado hardware server, SSH to board | self-hosted, board attached |

Never present a workflow as running synthesis unless it declares
`runs-on: self-hosted`.

## Without a self-hosted runner

Install `ci.yml` only. State plainly that synthesis and bitstream generation
stay manual, and that `build.yml` is a placeholder until a runner exists. Do not
fabricate a green build badge for a stage that never runs.

## Registering a self-hosted runner

Repository Settings -> Actions -> Runners -> New self-hosted runner. Run the
listed commands on the machine that has Vivado installed. On Windows install it
as a service so it survives reboot. Label it, for example `vivado`, and target
it with `runs-on: [self-hosted, vivado]`.

A self-hosted runner executes any workflow that reaches it. Do not enable one on
a public repository that accepts pull requests from forks.

## Artifacts

Bitstreams and `.xsa` handoff files are build outputs. Publish them with
`actions/upload-artifact` for a branch build, and attach them to a GitHub
Release for a tagged version. They never enter Git history.
