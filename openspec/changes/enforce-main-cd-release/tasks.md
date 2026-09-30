## 1. Contract tests

- [x] 1.1 Rewrite the CD workflow contract test for release-branch validation, draft-only output, gate ordering, write-permission scoping, and a dispatched publication gated on containment in `main`.
- [x] 1.2 Add release version selection tests for highest declared version, already-tagged no-op, empty or missing declaration, and workflow output format.
- [x] 1.3 Add real-image acceptance contract tests for physical-only passing, provenance-bound evidence, and host/board capture comparison.

## 2. Standalone ResNet-18 package

- [x] 2.1 Implement the allowlisted delivery archive with Vivado gate parsing, report manifest, reproducible output, and no host paths; verify `examples/resnet18/tests/test_delivery.py` passes.
- [x] 2.2 Implement package tree verification against the published archive digest, per-file digests, unexpected-file rejection, and overlay agreement before execution.
- [x] 2.3 Implement board acceptance over the acceptance bundle with recovery probing and provenance-bound evidence.
- [x] 2.4 Implement non-interactive real-image acceptance and publish its evidence atomically.
- [x] 2.5 Point the notebook at the package layout and keep the demonstration free of acceptance claims.

## 3. Main promotion, CD validation, and release

- [x] 3.1 Add declarative release version resolution from `changelog/`, and check the candidate tag against it before privileged work.
- [x] 3.2 Move the CD trigger to a `release/v*` branch that builds, packages, deploys, accepts, and publishes a draft in that order, and add the dispatched publication workflow.
- [x] 3.3 Add the enforced pre-merge gate on pull requests into `main`.
- [x] 3.4 Rewrite the deployment-and-acceptance scripts around the published archive, with atomic promotion after readable evidence.

## 4. Documentation

- [x] 4.1 Update `docs/rules/ci-cd.md` and `AGENTS.md` for release-branch CD, the four release states, approval boundaries, and failure recovery, and add the `release-npu-pynq` skill.
- [x] 4.4 Give the release skill the runner step: start this host's `actions.runner.*` services, confirm the `vivado` and `pynq-z1` labels are online, and refuse to push a release branch before they are.
- [x] 4.2 Document the three-step Quick Start, prerequisites, and known limitations in the package, the example README, and the root README.
- [x] 4.3 Record the release contents and boundaries in `changelog/v0.1.5.md`.

## 5. Validation

- [x] 5.1 Run the repository Python suites and record exact results.
- [ ] 5.2 Configure branch protection on `main` with `lint-and-simulate` and `release-readiness` required, and create the `pynq-z1-release` environment with required reviewers. Repository settings; blocked outside GitHub.
- [ ] 5.3 Push `release/v0.1.5`, run CD to completion including Vivado synthesis and physical PYNQ-Z1 image inference, confirm the draft, merge the promotion, publish, and link the run URL and commit SHA. Blocked until the trusted runners and the board are available.
- [ ] 5.4 Demonstrate a failed validation run producing no draft and no tag, and record the run URL.
