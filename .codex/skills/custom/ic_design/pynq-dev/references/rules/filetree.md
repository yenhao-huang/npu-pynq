# Filetree Rules

This repo-local skill is currently discovered at:

```text
.codex/skills/custom/ic_design/pynq-dev/
|-- SKILL.md
|-- STATE.md
`-- references/
    |-- development-matrix.md
    |-- validation-gates.md
    |-- rules/
    |   |-- env.md
    |   |-- filetree.md
    |   `-- state-rules.md
    `-- template/
        `-- STATE.template.md
```

Do not add `agents/`, top-level `scripts/`, README, changelog, generated cache,
credentials, FPGA build output, or board output to this skill.

Before editing product files, read `docs/rules/filetree.md`; it is authoritative
for product paths and prohibited generated content. Repository-local skills
must stay under `.codex/skills/` and follow its `dev/`, `deploy/`, and
`custom/ic_design/` classification. Move a complete skill directory and update
all repository-local references in the same change when its category changes.
