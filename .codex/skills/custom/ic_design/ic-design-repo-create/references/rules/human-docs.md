# Human Documentation Rules

`docs/human/` is owned by humans. Agents may read, search, compare, summarize,
and propose changes without approval, but must obtain explicit human
confirmation before any mutation.

Mutation includes creating, editing, appending, formatting, renaming, moving,
or deleting any file or directory under `docs/human/`. It also includes
automatically updating a roadmap or changelog as a side effect of completing
code, an issue, a pull request, or a release.

## Confirmation protocol

1. Show the exact target paths and a concise summary of the proposed content or
   edits.
2. Ask the human to confirm that specific batch.
3. Treat confirmation as scoped to those paths and edits only. Do not reuse it
   for later updates or infer it from approval of code, an issue, a PR, or a
   release.
4. After writing, read the files back and report what changed.

The human may explicitly request and confirm creation in the same message. If a
request is incomplete or would require inventing priorities, dates, ownership,
status, or release claims, stop and clarify before writing.

## Required files

```text
docs/human/
|-- feature-list.md
|-- roadmap.md
`-- changelog/
    `-- <year-week>.md
```

- `feature-list.md` explains confirmed features, intended users, status, and
  supporting evidence in human-readable language.
- `roadmap.md` records only human-confirmed direction, priorities, milestones,
  and non-goals. Never invent dates or commitments.
- `changelog/<year-week>.md` uses ISO week format `YYYY-Www`, for example
  `2026-W34.md`. Keep entries evidence-based and do not rewrite earlier entries
  merely to improve style.

Changes to this confirmation rule itself also require explicit human approval.
