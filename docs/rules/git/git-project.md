# GitHub Project Rules

GitHub Projects (v2) boards track issues across their lifecycle. Every board
for this repository uses the same `Status` field with exactly these five
options, in this order:

| Status | Meaning | Entry condition |
| --- | --- | --- |
| To triage | A newly raised issue or request, waiting for initial assessment. | The issue exists and has been added to the board. |
| Incoming | Accepted for work, but not yet scheduled. | Someone has assessed it and decided it will be done. |
| Ready | Ready to develop. | Priority and owner are set, and dependencies and acceptance criteria are written in the issue. |
| In progress | The owner is working on it. | The issue is claimed (see [git/issues.md](git/issues.md)) and its branch `npu/issue<issue-id>-<agent-id>` exists. |
| Done | Completed. | The PR is merged into `dev`, the acceptance criteria are met, and the issue is closed. |

## Moving items

- An item moves forward one status at a time. Moving it back is allowed when
  its entry condition no longer holds; say why in an issue comment.
- New issues enter at `To triage`. Only issues created already assessed and
  accepted may enter at `Incoming`.
- An agent moves an item to `In progress` when it claims the issue, and to
  `Done` only after the PR merge is confirmed. Moving an item to `Incoming` or
  `Ready` is a planning decision; agents do it only when a person asks.
- A tracking issue is `In progress` while any of its sub-issues is, and `Done`
  when all of them are.
- Do not add, rename, or remove `Status` options without updating this file.

## Priority

Every board also has a single-select `Priority` field with `P0` (most urgent),
`P1` and `P2`. An item must have a priority before it moves to `Ready`.

## Views

Every board has an `All Issues Board` view: board layout, one column per
`Status`, cards sorted by `Priority` ascending, showing Title, Assignees,
Status and Priority. The GitHub API cannot set a view's sort order, so set
"Sort by: Priority" once in the web UI after creating the view.

## Boards

| Board | Scope |
| --- | --- |
| [Hexagon-like NPU](https://github.com/users/yenhao-huang/projects/4) ([All Issues Board](https://github.com/users/yenhao-huang/projects/4/views/2)) | #166 and its sub-issues |
| [PYNQ Profiler](https://github.com/users/yenhao-huang/projects/5) ([All Issues Board](https://github.com/users/yenhao-huang/projects/5/views/2)) | Kernel profiler plan (`docs/plans/2026-1112-002-kernel-profiler.md`): #172 and later phases |

## Commands

Changing a status needs the `project` scope on the `gh` token
(`gh auth refresh -h github.com -s project`).

```bash
gh project item-list <number> --owner yenhao-huang
gh project field-list <number> --owner yenhao-huang   # Status field and option IDs
gh project item-edit --project-id <project-id> --id <item-id> \
  --field-id <status-field-id> --single-select-option-id <option-id>
```
