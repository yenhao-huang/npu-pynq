"""The runtime's one entry point: run a package with its task's command line.

    python -m src.runtime.cli llm <package> --prompt "Hello" [--chat --stream ...]
    python -m src.runtime.cli classifier --package <package> --image <jpg> --labels <txt>

Each task (src/runtime/tasks/<task>/cli.py) owns its arguments and output;
this module only picks the task.
"""

from __future__ import annotations

from importlib import import_module
import sys

TASKS = {
    'llm': 'generate text with an LLM package',
    'classifier': 'classify an image with a classifier package',
}


def usage() -> str:
    lines = ['usage: python -m src.runtime.cli {%s} ...' % ','.join(TASKS), '', 'tasks:']
    lines += [f'  {name:<12}{text}' for name, text in TASKS.items()]
    lines += ['', 'Run "python -m src.runtime.cli <task> -h" for its arguments.']
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print(usage())
        return 0 if argv else 2
    task, rest = argv[0], argv[1:]
    if task not in TASKS:
        print(f'unknown task {task!r}\n\n{usage()}', file=sys.stderr)
        return 2
    return import_module(f'src.runtime.tasks.{task}.cli').main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
