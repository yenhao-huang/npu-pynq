# Repository Rules

These files are the repository-wide operating rules for contributors and
agents. They were promoted from the `ic-design-repo-create` skill so that the
rules remain visible even when that skill is not active.

Read the rules that apply before making a change:

- [filetree.md](filetree.md): allowed repository paths and generated outputs.
- [environment.md](environment.md): host, tools, simulators, Vivado, and board
  assumptions.
- [generated-artifacts.md](generated-artifacts.md): ignored build products,
  bitstreams, and credential restrictions.
- [human-docs.md](human-docs.md): human confirmation boundary for feature
  lists, roadmaps, and weekly changelogs.
- [simulation.md](simulation.md): lint and simulation contracts.
- [ci-cd.md](ci-cd.md): hosted CI, self-hosted synthesis, and board testing.
- [git/branch.md](git/branch.md): `main`, `dev`, task branches, and releases.
- [git/changelog.md](git/changelog.md): release commit boundaries, roadmap
  progress, and ordered Issue/PR tables.
- [git/commit.md](git/commit.md): commit format and safety requirements.
- [git/issues.md](git/issues.md): issue creation, triage, and evidence.
- [git/pull-request.md](git/pull-request.md): review, validation, and merge
  boundaries.

When these rules conflict with a skill reference, this directory is the
authority for work in `npu_repo_in_pynq`.
