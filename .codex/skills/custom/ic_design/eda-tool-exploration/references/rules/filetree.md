# File placement

Repository rules override generic skill layouts. This skill lives under
.codex/skills/custom/ic_design/eda-tool-exploration/ and contains SKILL.md,
STATE.md, references/rules/{filetree,env,state-rules}.md and
references/template/STATE.template.md. Domain notes belong in references/.

Backend operations: tools/ic/ic_core/tools/<category>/<backend>.py.
Compositions: tools/ic/ic_core/tools/pipeline/<paper-abbreviation>/.
Tests: tools/ic/tests/. Experiments: exp/tool-exploration/exp-tool-<id>-<paper>/.
Raw output: ignored output/ or .ic/. Compact acceptance summaries and reports:
docs/goals/<goal>/. Never modify docs/human without exact human confirmation.
