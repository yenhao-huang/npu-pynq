# Filetree Rules

Keep this skill in:

```text
npu_repo_in_pynq/.codex/skills/custom/ic_design/ic-design-repo-create/
|-- SKILL.md
|-- STATE.md
`-- references/
    |-- ci-cd.md
    |-- directory.md
    |-- gitignore.md
    |-- simulation.md
    |-- rules/
    |   |-- env.md
    |   |-- filetree.md
    |   |-- human-docs.md
    |   |-- state-rules.md
    |   `-- git/
    |       |-- git-branch.md
    |       |-- git-commit.md
    |       |-- git-issues.md
    |       `-- git-pr.md
    |-- template/
    |   |-- AGENTS.template.md
    |   |-- CHANGELOG-WEEK.template.md
    |   |-- FEATURE-LIST.template.md
    |   |-- ROADMAP.template.md
    |   `-- STATE.template.md
    `-- workflows/
        |-- build.yml
        `-- ci.yml
```

Do not add `agents/`, a top-level `scripts/`, README, installation guide, or
generated cache files. Human changelogs belong only under `docs/human/changelog/`
and require confirmation under `references/rules/human-docs.md`.
