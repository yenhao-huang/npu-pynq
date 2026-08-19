# Commit Rules

Conventional Commits. Determine type and scope from the actual diff, never from
the request wording.

```text
<type>(<scope>): <description>

<body>

<footer>
```

## Types

| Type | Use for |
| --- | --- |
| `feat` | new RTL module, new export or runtime capability |
| `fix` | corrected logic, timing, or numeric behaviour |
| `perf` | fewer cycles, fewer LUT/DSP/BRAM, higher Fmax |
| `test` | testbench, cocotb test, or test vectors |
| `refactor` | restructuring with no behavioural change |
| `docs` | specifications, rules, README |
| `build` | Makefile, Vivado Tcl, tool version |
| `ci` | workflow files |
| `chore` | maintenance that fits nothing above |
| `revert` | revert of an earlier commit |

## Scopes

Use the design name for hardware work (`feat(conv): ...`). Otherwise use the
directory: `hw`, `test`, `export`, `runtime`, `examples`, `docs`, `ci`.

## Description

Imperative mood, present tense, no trailing period, under 72 characters.
"add saturating accumulator", not "added" or "adds".

## Body

Explain why, not what; the diff already shows what. For hardware changes state
what a reader cannot see in the diff:

- the numeric behaviour that changed, and whether `src/test/model/` moved with it
- resource or timing impact, when a synthesis run measured it
- which testbench or cocotb test covers the change

A commit that changes RTL without touching a test must say in the body why
coverage is unchanged.

## Breaking changes

An RTL change is breaking when it alters the register map, the AXI interface,
or the numeric contract in `src/test/model/`. A previously exported model then
no longer runs correctly, so mark it:

```text
feat(conv)!: widen accumulator to 48 bits

BREAKING CHANGE: saturation bounds change; re-export existing models.
```

## Rules

- One logical change per commit.
- Never commit a bitstream, a Vivado project directory, or generated output.
  See [../../gitignore.md](../../gitignore.md).
- Never commit credentials, board passwords, licence files, or tokens.
- Reference issues in the footer: `Refs #123`, or `Closes #123` only when the
  commit fully resolves it.
- Never amend or rebase a commit that has been pushed.
- If a hook rejects the commit, fix the cause and make a new commit. Never pass
  `--no-verify`.
- Never run `git config`, `git reset --hard`, or `git push --force` without an
  explicit request naming that command.
