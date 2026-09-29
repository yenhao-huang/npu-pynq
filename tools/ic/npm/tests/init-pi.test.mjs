import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { initPi } from '../init-pi.mjs';
import { packageRoot } from '../bootstrap.mjs';

async function fixture(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'ic pi project '));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

test('init preserves settings, quotes paths, and is repeatable across package relocation', async t => {
  const project = await fixture(t);
  await mkdir(path.join(project, '.pi'));
  const settings = '{"model":"keep-me","skills":["existing"]}\n';
  await writeFile(path.join(project, '.pi/settings.json'), settings);
  const root = path.join(project, 'installed package');
  const target = await initPi(project, root);
  const first = await readFile(target, 'utf8');
  assert.ok(first.includes(JSON.stringify(path.join(root, 'integrations/pi/managed.ts').replaceAll('\\', '/'))));
  await initPi(project, root);
  assert.equal(await readFile(target, 'utf8'), first);
  await initPi(project, root + '-moved');
  assert.notEqual(await readFile(target, 'utf8'), first);
  assert.equal(await readFile(path.join(project, '.pi/settings.json'), 'utf8'), settings);
});

test('custom extension is never overwritten, including edits to a generated wrapper', async t => {
  const project = await fixture(t);
  const target = await initPi(project);
  const edited = await readFile(target, 'utf8') + '// user customization\n';
  await writeFile(target, edited);
  await assert.rejects(initPi(project), /Refusing to overwrite/);
  assert.equal(await readFile(target, 'utf8'), edited);
});

test('CLI init pi needs no runtime download and rejects other agents', async t => {
  const project = await fixture(t);
  const cli = path.join(packageRoot, 'npm/cli.mjs');
  const result = spawnSync(process.execPath, [cli, 'init', 'pi', '--project', project], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(await readFile(path.join(project, '.pi/extensions/ic-design-tools.ts'), 'utf8'), /managed\.ts/);
  const bad = spawnSync(process.execPath, [cli, 'init', 'unknown'], { encoding: 'utf8' });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /Usage/);
});

test('init migrates the known repository extension without duplicating tool registration', async t => {
  const project = await fixture(t);
  const target = path.join(project, '.pi/extensions/ic-design-tools.ts');
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target,
    '// Auto-discovered by pi. The implementation lives with the tools it exposes,\n' +
    '// so it stays in step with them; this file only points at it.\n' +
    'export { default } from "../../tools/ic/integrations/pi/extension.ts";\n');
  await initPi(project);
  assert.match(await readFile(target, 'utf8'), /Managed by ic-tools/);
});
