# Human Documentation Rules

`docs/human/` is the human-owned view of confirmed features, direction, and
change history. Agents may read and summarize it without approval, but must
obtain explicit human confirmation before any mutation.

Mutation includes creating, editing, appending, formatting, renaming, moving,
or deleting files or directories. Finishing code, an issue, a pull request, a
merge, or a release does not implicitly authorize a roadmap or changelog update.

Before writing, an agent must show the exact paths and summarize the proposed
batch. Confirmation applies only to that batch and cannot be reused for later
changes. After writing, read the files back and report the result.

Required structure:

```text
docs/human/
|-- feature-list.md
|-- roadmap.md
`-- changelog/
    `-- <YYYY-Www>.md
```

- `feature-list.md` contains only confirmed features and evidence.
- `roadmap.md` contains only human-confirmed priorities, milestones, and
  non-goals; agents must not invent dates or commitments.
- `changelog/<YYYY-Www>.md` uses ISO week naming and evidence-based entries.

Changes to this rule itself require explicit human confirmation.
