import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import * as tar from 'tar';
import { platformSpec, verify, download, withLock, ensureRuntime, runtimeEnvironment, extract, packageRoot, run } from '../bootstrap.mjs';

async function fixture(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'ic-mcp-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}
const bytes = Buffer.from('verified archive');
const spec = { key: 'linux-x64', name: 'fixture.tgz', size: bytes.length,
  sha256: createHash('sha256').update(bytes).digest('hex'), url: 'https://example.test/suite.tgz' };

test('platform selection fails before download and gives Windows guidance', () => {
  assert.match(platformSpec('linux', 'arm64').url, /linux-arm64-20260923/);
  assert.match(platformSpec('darwin', 'arm64').url, /darwin-arm64/);
  assert.throws(() => platformSpec('win32', 'x64'), /WSL/);
  assert.throws(() => platformSpec('linux', 'ia32'), /Unsupported/);
});

test('download validates bytes and reuses a verified archive without network', async t => {
  const file = path.join(await fixture(t), 'download.tgz');
  let calls = 0;
  const fetchImpl = async () => { calls++; return new Response(bytes); };
  await download(spec, file, { fetchImpl });
  await download(spec, file, { fetchImpl });
  assert.equal(calls, 1);
  assert.deepEqual(await readFile(file), bytes);
});

test('bad checksum, failed HTTP and oversized response never publish an archive', async t => {
  const dir = await fixture(t);
  for (const response of [new Response(Buffer.alloc(bytes.length)), new Response('', { status: 503 }), new Response(Buffer.alloc(bytes.length + 1))]) {
    await assert.rejects(download(spec, path.join(dir, 'bad.tgz'), { fetchImpl: async () => response }));
    assert.deepEqual(await readdir(dir), []);
  }
  await writeFile(path.join(dir, 'bad.tgz'), 'broken');
  await assert.rejects(verify(path.join(dir, 'bad.tgz'), spec), /Integrity/);
});

test('concurrent setup serializes one install and completed runtime works offline', async t => {
  const root = await fixture(t);
  let installs = 0;
  const install = async stage => {
    installs++;
    await sleep(40);
    await mkdir(path.join(stage, 'oss-cad-suite', 'py3bin'), { recursive: true });
    await writeFile(path.join(stage, 'oss-cad-suite', 'py3bin', 'python3'), 'fixture');
    await mkdir(path.join(stage, 'python-packages', 'mcp'), { recursive: true });
    await writeFile(path.join(stage, 'python-packages', 'mcp', '__init__.py'), '');
  };
  const paths = await Promise.all([ensureRuntime({ root, spec, install }), ensureRuntime({ root, spec, install })]);
  assert.equal(paths[0], paths[1]);
  assert.equal(installs, 1);
  assert.equal(await ensureRuntime({ root, spec, install: () => { throw new Error('must not install'); } }), paths[0]);
});

test('failed setup cleans staging and lock and can be retried', async t => {
  const root = await fixture(t);
  await assert.rejects(ensureRuntime({ root, spec, install: async stage => {
    await writeFile(path.join(stage, 'partial'), 'incomplete'); throw new Error('interrupted');
  } }), /interrupted/);
  assert.deepEqual(await readdir(root), []);
});

test('an occupied lock times out without removing the owner lock', async t => {
  const root = await fixture(t);
  const lock = path.join(root, 'install.lock');
  await writeFile(lock, '{}');
  await assert.rejects(withLock(lock, () => {}, { timeoutMs: 5, pollMs: 2 }), /Timed out/);
  assert.equal(await readFile(lock, 'utf8'), '{}');
});

test('verified extraction preserves executable modes and contained links', async t => {
  const root = await fixture(t);
  const input = path.join(root, 'input');
  await mkdir(path.join(input, 'oss-cad-suite', 'bin'), { recursive: true });
  await writeFile(path.join(input, 'oss-cad-suite', 'bin', 'tool'), '#!/bin/sh\n', { mode: 0o755 });
  const archive = path.join(root, 'suite.tgz');
  await tar.c({ cwd: input, file: archive, gzip: true }, ['oss-cad-suite']);
  await extract(archive, path.join(root, 'output'));
  assert.equal(await readFile(path.join(root, 'output', 'oss-cad-suite', 'bin', 'tool'), 'utf8'), '#!/bin/sh\n');
});

test('runtime environment ignores ambient Python paths but preserves artifact location', () => {
  const env = runtimeEnvironment('/runtime', { PATH: '/usr/bin', PYTHONHOME: '/wrong', PYTHONPATH: '/wrong', IC_ROOT: '/project/artifacts' });
  assert.equal(env.PYTHONHOME, undefined);
  assert.equal(env.IC_ROOT, '/project/artifacts');
  assert.ok(env.PYTHONPATH.startsWith(packageRoot));
  assert.ok(!env.PYTHONPATH.includes('/wrong'));
});

test('help and invalid arguments do not trigger setup', () => {
  const cli = path.join(packageRoot, 'npm', 'cli.mjs');
  const help = spawnSync(process.execPath, [cli, '--help'], { encoding: 'utf8' });
  assert.equal(help.status, 0);
  assert.match(help.stdout, /setup\|doctor/);
  const bad = spawnSync(process.execPath, [cli, 'remove-everything'], { encoding: 'utf8' });
  assert.equal(bad.status, 1);
  assert.equal(bad.stdout, '');
  assert.match(bad.stderr, /Unknown command/);
});

test('child startup failure is surfaced', async () => {
  await assert.rejects(run('/definitely/not/an/executable', []), /ENOENT/);
});
