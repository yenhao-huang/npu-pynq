"""The release workflow: check, export, build-bitstream, test-on-board, draft-release.

Every model in .github/release/models.json goes the way its example guide
does (export, src/deploy/deploy.sh, the guide's run command), and only a
draft Release comes out; publish.yml publishes it once it is in `main`.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / '.github' / 'workflows'
RELEASE = ROOT / '.github' / 'release'
SKILL = ROOT / '.codex' / 'skills' / 'deploy' / 'release-npu-pynq'
JOBS = ['check', 'export', 'build-bitstream', 'test-on-board', 'draft-release']
POSIX_BASH = sys.platform != 'win32' and shutil.which('bash')


def load(name: str) -> dict:
    workflow = yaml.safe_load((WORKFLOWS / name).read_text(encoding='utf-8'))
    workflow['on'] = workflow.pop(True, workflow.get('on'))
    return workflow


def needs(job: dict) -> set[str]:
    value = job.get('needs', [])
    return {value} if isinstance(value, str) else set(value)


class ReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.release = load('release.yml')
        self.jobs = self.release['jobs']

    def test_it_runs_on_release_branches(self) -> None:
        self.assertEqual(self.release['on']['push']['branches'], ['release/v*'])
        self.assertFalse((ROOT / '.github' / 'cd').exists())
        self.assertFalse((WORKFLOWS / 'cd.yml').exists())

    def test_the_jobs_and_their_order(self) -> None:
        self.assertEqual(list(self.jobs), JOBS)
        self.assertEqual(needs(self.jobs['export']), {'check'})
        self.assertEqual(needs(self.jobs['build-bitstream']), {'check'})
        self.assertEqual(needs(self.jobs['test-on-board']), {'check', 'export', 'build-bitstream'})
        self.assertEqual(needs(self.jobs['draft-release']), set(JOBS[:-1]))

    def test_only_the_draft_can_write_and_it_stays_a_draft(self) -> None:
        for name, job in self.jobs.items():
            writes = job.get('permissions', {}).get('contents') == 'write'
            self.assertEqual(writes, name == 'draft-release', name)
        script = '\n'.join(step.get('run', '') for step in self.jobs['draft-release']['steps'])
        self.assertIn('gh release create "$TAG" --draft', script)
        self.assertNotIn('--draft=false', script)

    def test_later_jobs_check_out_the_checked_commit(self) -> None:
        for name in JOBS[1:]:
            checkout = self.jobs[name]['steps'][0]
            self.assertEqual(checkout['uses'], 'actions/checkout@v4', name)
            self.assertEqual(checkout['with']['ref'], '${{ needs.check.outputs.commit }}', name)

    def test_privileged_work_stays_on_this_host(self) -> None:
        self.assertEqual(self.jobs['check']['runs-on'], ['self-hosted', 'vivado', 'pynq-z1'])
        self.assertEqual(self.jobs['build-bitstream']['runs-on'], ['self-hosted', 'vivado'])
        self.assertEqual(self.jobs['test-on-board']['runs-on'], ['self-hosted', 'pynq-z1'])
        for name in ('check', 'test-on-board'):
            self.assertEqual(self.jobs[name]['environment'], 'pynq-z1-production', name)

    def test_both_model_jobs_take_their_matrix_from_models_json(self) -> None:
        for name in ('export', 'test-on-board'):
            include = self.jobs[name]['strategy']['matrix']['include']
            self.assertEqual(include, '${{ fromJSON(needs.check.outputs.models) }}', name)
        self.assertEqual(self.jobs['test-on-board']['strategy']['max-parallel'], 1)

    def test_the_board_runs_what_deploy_sh_packages(self) -> None:
        script = '\n'.join(step.get('run', '') for step in self.jobs['test-on-board']['steps'])
        self.assertIn('bash src/deploy/deploy.sh dist/${{ matrix.model }} ${{ matrix.files }}', script)
        self.assertIn('bash .github/release/board_check.sh "$ARCHIVE"', script)
        self.assertIn("board['output_ids'] != host['output_ids']", script)

    def test_a_release_reuses_its_own_bitstream(self) -> None:
        script = json.dumps(self.jobs['build-bitstream']['steps'])
        for marker in ("workflow_id: 'release.yml'", "job.name === 'build-bitstream'",
                       'git merge-base --is-ancestor', 'src/hw src/vendors/npu_pynq/driver/overlay'):
            self.assertIn(marker, script)


class ModelListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models = json.loads((RELEASE / 'models.json').read_text(encoding='utf-8'))

    def test_the_release_models(self) -> None:
        self.assertEqual([m['model'] for m in self.models], ['resnet18', 'smollm'])

    def test_every_entry_names_files_that_exist(self) -> None:
        from src.runtime.cli import TASKS

        for model in self.models:
            with self.subTest(model=model['model']):
                self.assertEqual(set(model), {'model', 'fetch', 'config', 'files', 'run', 'evidence'})
                self.assertTrue((ROOT / model['config']).is_file())
                for path in model['files'].split():
                    self.assertTrue((ROOT / path).is_file(), path)
                task = model['run'].split()[0]
                self.assertIn(task, TASKS)
                self.assertEqual(model['evidence'], {'classifier': '--evidence', 'llm': '--json'}[task])

    def test_publish_requires_every_model_asset(self) -> None:
        publish = (WORKFLOWS / 'publish.yml').read_text(encoding='utf-8')
        for model in self.models:
            self.assertIn(f'"pynqo-{model["model"]}-${{TAG}}.tar.gz"', publish)
            self.assertIn(f'"board-{model["model"]}.json"', publish)


class PublishTests(unittest.TestCase):
    def test_publishing_is_manual_and_approved(self) -> None:
        publish = load('publish.yml')
        self.assertEqual(set(publish['on']), {'workflow_dispatch'})
        self.assertEqual(publish['jobs']['publish']['environment'], 'pynq-z1-release')
        self.assertIn('git merge-base --is-ancestor "$commit" origin/main',
                      (WORKFLOWS / 'publish.yml').read_text(encoding='utf-8'))

    def test_the_pre_merge_gate_protects_main(self) -> None:
        ci = (WORKFLOWS / 'ci.yml').read_text(encoding='utf-8')
        for marker in ('release-readiness:', "github.base_ref == 'main'", '.github/release/version.py',
                       '.github/workflows/release.yml', '.github/workflows/publish.yml'):
            self.assertIn(marker, ci)


@unittest.skipUnless(POSIX_BASH, 'needs a POSIX bash')
class ScriptTests(unittest.TestCase):
    def test_the_scripts_parse(self) -> None:
        for script in (RELEASE / 'board_check.sh', ROOT / 'src' / 'deploy' / 'deploy.sh'):
            subprocess.run(['bash', '-n', str(script)], check=True)

    def test_deploy_archive_holds_only_what_the_board_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'demo'
            for path in ('package/manifest.json', 'package/compile.json', 'package/ir/model.mlir',
                         'package/arm-cortex-a9/model.so', 'package/x86_64-host/model.so',
                         'overlay/npu_matrix.bit', 'model/weights.pth', 'model/demo.jpg'):
                (directory / path).parent.mkdir(parents=True, exist_ok=True)
                (directory / path).write_text('x')
            archive = Path(temporary) / 'out' / 'demo.tar.gz'
            subprocess.run(['bash', str(ROOT / 'src' / 'deploy' / 'deploy.sh'), str(directory)],
                           check=True, env={'PATH': '/usr/bin:/bin', 'ARCHIVE': str(archive)})
            with tarfile.open(archive) as tar:
                names = set(tar.getnames())
        for kept in ('demo/package/manifest.json', 'demo/package/arm-cortex-a9/model.so',
                     'demo/overlay/npu_matrix.bit', 'demo/model/demo.jpg', 'demo/src/runtime/cli.py'):
            self.assertIn(kept, names)
        for dropped in ('demo/package/compile.json', 'demo/package/ir/model.mlir',
                        'demo/package/x86_64-host/model.so', 'demo/model/weights.pth'):
            self.assertNotIn(dropped, names)


class ReleaseSkillTests(unittest.TestCase):
    def test_the_skill_documents_the_whole_path(self) -> None:
        body = (SKILL / 'SKILL.md').read_text(encoding='utf-8')
        for marker in ('release/vX.Y.Z', 'draft Release', 'merge commit', 'release.yml', 'publish.yml',
                       '.github/release/version.py', *JOBS):
            self.assertIn(marker, body)
        for gone in ('cd.yml', 'release-publish.yml', 'resolve_release_version.py'):
            self.assertNotIn(gone, body)

    def test_the_skill_starts_the_runners_itself(self) -> None:
        body = (SKILL / 'SKILL.md').read_text(encoding='utf-8')
        self.assertLess(body.index('Bring the runners online'), body.index('Cut the release branch'))
        self.assertIn('start_cd_runners.ps1', body)
        source = (SKILL / 'references' / 'scripts' / 'start_cd_runners.ps1').read_text(encoding='utf-8')
        for marker in ('actions.runner.*', 'Start-Service', 'SupportsShouldProcess', 'run.cmd',
                       'Start-Process', 'Start-ScheduledTask'):
            self.assertIn(marker, source)
        for forbidden in ('config.cmd', '--token', 'svc.cmd install', 'Remove-Service'):
            self.assertNotIn(forbidden, source)
        task = (SKILL / 'references' / 'scripts' / 'register_runner_task.ps1').read_text(encoding='utf-8')
        for marker in ('-AtLogOn', '[TimeSpan]::Zero', 'Set-Service', 'Disabled'):
            self.assertIn(marker, task)
        for reference in ('references/rules/env.md', 'references/runners.md'):
            self.assertTrue((SKILL / reference).is_file(), reference)


if __name__ == '__main__':
    unittest.main()
