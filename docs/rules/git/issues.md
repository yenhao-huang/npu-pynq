# Issue Rules

Issues are the executable units of work. Major behavior changes first require
an OpenSpec change; its tracking issue links the implementation issues and pull
requests.

Before creating or modifying an issue:

1. Confirm the exact repository and authentication state.
2. Search open and closed issues for duplicates.
3. Read the existing issue before editing it.
4. Preserve unrelated body sections and metadata.

Issue titles are specific, actionable, and under 72 characters. A useful
hardware issue records what is known about:

- design, source commit, branch, or tag;
- simulation-only, board-only, or shared reproduction;
- triggering vector, tensor, register sequence, or model;
- expected and observed values and golden-model agreement;
- Vivado/tool version when synthesis or timing is involved;
- acceptance criteria and required validation level;
- owner, dependencies, blockers, and associated OpenSpec change.

Do not invent reproduction steps, severity, assignees, milestones, or results.
Never include credentials, board passwords, licences, tokens, private keys, or
unredacted sensitive logs. Never close, reopen, transfer, or delete an issue
without clear user intent.

Use `Related to #123`, `Blocked by #123`, and sub-issues to preserve the work
graph. After a write, read the issue back and report its number and URL.

## Claiming and ownership

Before creating a worktree, claim the issue through the repository-supported
assignee mechanism and a concise comment containing the stable `agent-id`.
Read the issue back after the claim; if another owner already claimed it or
ownership is ambiguous, stop before branching. Search for an existing branch,
worktree, and open PR for the issue and reuse the same active worktree.

One issue maps to branch `npu/issue<issue-id>-<agent-id>`, one active worktree,
and one PR to `dev`. Do not close the issue until the PR is confirmed merged,
the acceptance criteria are satisfied, and lifecycle closure is authorized.
