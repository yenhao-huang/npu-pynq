## Context

See `proposal.md` for motivation. The current `build.yml` is tag-push driven,
uses an optional dispatch input that is empty for tag events, and collects from
a directory different from the one produced by `build_overlay.tcl`. The Phase
1C notebook also depends on repository-relative shared runtime imports and has
no non-interactive board entrypoint. Vivado and physical-board work must remain
on trusted self-hosted runners, while generated payloads remain outside Git.

## Goals / Non-Goals

**Goals:**

- Make a published stable GitHub Release the sole production CD trigger.
- Verify the release tag before privileged jobs and preserve immutable source
  provenance through build, package, deployment, and evidence.
- Produce a standalone, allowlisted Phase 1C package and execute it without a
  repository checkout on the board.
- Keep packaging and board-case logic host-testable without Vivado or PYNQ.

**Non-Goals:**

- Automatically creating tags, merging `dev` into `main`, or publishing a
  Release.
- Running Vivado or physical-board work for pull requests or branch pushes.
- Committing generated overlays, board evidence, runner credentials, or Vivado
  projects.
- Changing RTL, the hardware ABI, numeric behavior, or adding K tiling.

## Decisions

### Use a release-published workflow with three ordered jobs

Rename the existing build workflow to a CD workflow triggered by
`release: types: [published]`. A GitHub-hosted validation job checks stable
version syntax and containment in `origin/main`; a trusted Vivado job builds,
verifies, packages, and publishes assets; a trusted PYNQ-Z1 job deploys and
executes the package under a protected environment. `concurrency` serializes
production board access without cancelling an active deployment.

This is preferred over `push: branches: [main]` because documentation-only
main changes must not consume Vivado or mutate board state. It is preferred
over tag-push triggering because a tag alone is not the explicit production
release decision requested by the repository workflow.

### Pass immutable packages through workflow artifacts

The Vivado job runs the repository Tcl unchanged, verifies
`build/vivado/npu_matrix/artifacts`, creates the standalone package under an
ignored staging directory, and uploads it with build evidence. The board job
downloads that exact workflow artifact rather than rebuilding or checking out
a moving branch. The same BIT, HWH, manifest, and evidence are uploaded to the
originating Release with deterministic names.

Using workflow artifacts between jobs avoids assuming that two self-hosted
jobs share a filesystem. Copying arbitrary worktree contents was rejected
because it could include secrets, caches, or unrelated files.

### Package a minimal repository-shaped runtime tree

The package keeps `runtime/matrix_multiplication.py` for example-specific
logic and `src/runtime/` for the shared Phase 1B runtime. This mirrors current
imports, avoids duplicating production runtime implementations, and lets the
notebook and non-interactive runner execute from the package root. A Python
package builder uses a fixed allowlist and refuses missing inputs.

For local validation, the package builder discovers the repository from its
own path and reads commit provenance from the verified overlay manifest. Its
zero-option command writes to `mount/matrix-multiplication/local-<commit>`.
Release CD continues passing the stable release tag, commit, and paths
explicitly so production inputs remain visible in workflow logs.

### Use a testable non-interactive board entrypoint

`run_on_board.py` separates pure case execution/evidence creation from the CLI
that verifies artifacts and loads PYNQ. Host tests inject a fake physical
runtime; the trusted board job exercises the real overlay. The notebook remains
the interactive presentation, while CD consumes only the deterministic CLI.

### Deploy versioned directories over preconfigured SSH

The board job deploys to a release-tagged directory below the configured PYNQ
deployment root, runs the entrypoint there, and retrieves the JSON evidence.
SSH host identity and credentials come from the protected environment or
runner configuration. The workflow never echoes credentials. A release-tagged
path preserves previous deployments and makes rollback a directory selection
rather than an untracked overwrite.

### Restore the existing RTL lint gate without changing hardware behavior

The PR exposed a baseline Verilator `-Wall` failure already present on `dev`.
Resolve arithmetic-width warnings with explicit zero-extension, initialize the
temporary combinational reduction index on every path, and annotate only the
AXI protection inputs and mixed reset observation that are intentional parts of
the existing interface. Do not suppress width or latch warnings globally and
do not change the register map, reset implementation, timing protocol, or
numeric behavior.

The existing controller, AXI-Lite, and accelerator testbenches are the behavior
regression. GitHub's `make -C src/test lint sim` run is the authoritative
Verilator gate because Verilator is unavailable on the local Windows host.

## Risks / Trade-offs

- [Release is already public while hardware jobs run] -> Release notes and
  workflow status remain authoritative; assets are attached only after their
  gates pass, and failures stay visible instead of being reported as success.
- [Self-hosted runner is offline or lacks Vivado/board access] -> Jobs remain
  queued or fail as blocked; no fallback claims hardware validation.
- [Repeated workflow run encounters an existing remote version directory] ->
  deployment uses a run-specific staging directory and updates the stable
  release path only after validation.
- [Long Vivado execution occupies the runner] -> concurrency serializes CD and
  does not cancel an in-progress hardware build.
- [Release asset upload is retried] -> deterministic names and explicit
  clobber semantics make reruns idempotent for the same immutable tag.

## Migration Plan

1. Land the new workflow, package builder, board runner, tests, and rules in
   `dev`, then merge them to `main` through the human deployment process.
2. Configure protected environment `pynq-z1-production`, trusted runner labels,
   non-interactive SSH, and required GitHub CLI/Python/Vivado tools.
3. Publish the next stable release and observe validation/build/package jobs
   before approving or enabling the physical-board job.
4. On failure, disable the board environment or workflow and retain the last
   known-good release directory; source rollback is a normal revert through
   `dev`.
