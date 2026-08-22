# Filetree Rule

Keep this skill at:

```text
.codex/skills/custom/ic_design/build-mac-npu-on-pynq-z1/
|-- SKILL.md
|-- STATE.md
`-- references/
    |-- architecture.md
    |-- board-deploy-and-debug.md
    |-- end-to-end-runbook.md
    |-- pynq-integration.md
    |-- rtl-and-axi-contract.md
    |-- verification.md
    |-- vivado-overlay-flow.md
    |-- installation/
    |   `-- icarus-verilog-windows.md
    |-- rules/
    |   |-- env.md
    |   |-- filetree.md
    |   `-- state-rules.md
    `-- template/
        `-- STATE.template.md
```

Keep implementation under `mount/mac_npu/`. Do not add `agents/`, a top-level
skill `scripts/`, README, changelog, generated caches, or Vivado run products to
the skill directory.

Keep package installation workflows under `references/installation/`. Every
installation reference must be linked directly from `SKILL.md`; do not add an
extra nested index that forces reference chaining.
