"""src/runtime/ layering: the shared layer knows no task, and tasks never import each other.

    cli.py               picks the task
    share/, delegate/    load and call a package; no task
    tasks/<task>/        one engine and cli.py per manifest graph kind
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

RUNTIME = Path(__file__).resolve().parents[2] / 'runtime'
TASKS = sorted(p.name for p in (RUNTIME / 'tasks').iterdir() if (p / '__init__.py').is_file())


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def task_imports(path: Path) -> set[str]:
    return {task for name in imports(path) for task in TASKS
            if name == f'src.runtime.tasks.{task}' or name.startswith(f'src.runtime.tasks.{task}.')}


class RuntimeLayeringTests(unittest.TestCase):
    def test_there_are_tasks(self) -> None:
        self.assertEqual(TASKS, ['classifier', 'llm'])

    def test_the_shared_layer_imports_no_task(self) -> None:
        for path in [*(RUNTIME / 'share').glob('*.py'), *(RUNTIME / 'delegate').glob('*.py')]:
            with self.subTest(path=path.relative_to(RUNTIME).as_posix()):
                self.assertEqual(task_imports(path), set())

    def test_tasks_do_not_import_each_other(self) -> None:
        for task in TASKS:
            for path in (RUNTIME / 'tasks' / task).glob('*.py'):
                with self.subTest(path=path.relative_to(RUNTIME).as_posix()):
                    self.assertLessEqual(task_imports(path), {task})

    def test_the_entry_point_knows_every_task(self) -> None:
        from src.runtime.cli import TASKS as ENTRY_TASKS

        self.assertEqual(sorted(ENTRY_TASKS), TASKS)
        for task in TASKS:
            self.assertTrue((RUNTIME / 'tasks' / task / 'cli.py').is_file(), task)


if __name__ == '__main__':
    unittest.main()
