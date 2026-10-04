# Filetree Rules

Keep this skill in `.codex/skills/custom/ic_design/hardware-architecture/`.
Required files are `SKILL.md`, `STATE.md`,
`references/rules/{env,filetree,state-rules}.md`, and
`references/template/STATE.template.md`. Put reusable helpers under
`references/scripts/` only when needed; do not add top-level scripts or a
README.

The active product image is selected by the Markdown image link in
`README.md`. At creation, that link points to
`docs/assets/npu-hardware-architecture.svg`. Treat
`src/hw/rtl/npu_matrix/` as the source of visible module and directory names.
The design wrapper `npu_matrix/` is not itself a functional block.

Follow `docs/rules/filetree.md` for repository paths and generated artifacts.
Keep render previews in ignored build output, not beside the source SVG.
