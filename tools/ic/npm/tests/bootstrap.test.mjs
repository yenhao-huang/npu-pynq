import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, stat, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import * as tar from 'tar';
import { platformSpec, verify, download, withLock, ensureRuntime, runtimeEnvironment, extract, packageRoot, openroadImage, openroadPlatform, requireOpenroad, configureOpenroad, run } from '../bootstrap.mjs';

async function fixture(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'ic-mcp-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}
const bytes = Buffer.from('verified archive');
const spec = { key: 'linux-x64', name: 'fixture.tgz', size: bytes.length,
  sha256: createHash('sha256').update(bytes).digest('hex'), url: 'https://example.test/suite.tgz' };

test('platform selection fails before download and gives Windows guidance', () => {
  assert.match(platformSpec('linux', 'arm64').url, /linux-arm64-20261004/);
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
  let checks = 0;
  const install = async stage => {
    installs++;
    await sleep(40);
    await mkdir(path.join(stage, 'oss-cad-suite', 'py3bin'), { recursive: true });
    await writeFile(path.join(stage, 'oss-cad-suite', 'py3bin', 'python3'), 'fixture');
    await mkdir(path.join(stage, 'python-packages', 'mcp'), { recursive: true });
    await writeFile(path.join(stage, 'python-packages', 'mcp', '__init__.py'), '');
  };
  const requireTools = async () => { checks++; };
  const paths = await Promise.all([
    ensureRuntime({ root, spec, install, requireTools }),
    ensureRuntime({ root, spec, install, requireTools }),
  ]);
  assert.equal(paths[0], paths[1]);
  assert.equal(installs, 1);
  assert.equal(await ensureRuntime({ root, spec, requireTools,
    install: () => { throw new Error('must not install'); } }), paths[0]);
  assert.equal(checks, 3, 'the required tool is checked even when the runtime is cached');
});

test('failed setup cleans staging and lock and can be retried', async t => {
  const root = await fixture(t);
  await assert.rejects(ensureRuntime({ root, spec, requireTools: async () => {}, install: async stage => {
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

test('native OpenROAD is preferred without contacting Docker', async () => {
  const env = { PATH: '/tools' };
  const calls = [];
  const provider = await requireOpenroad({ env, runImpl: async (...args) => {
    calls.push(args);
    return 0;
  } });
  assert.deepEqual(provider, { kind: 'host' });
  assert.deepEqual(calls, [['openroad', ['-version'], { env }]]);
  assert.equal(env.IC_OPENROAD_PROVIDER, 'host');
  assert.equal(env.IC_OPENROAD_DOCKER_IMAGE, undefined);
});

test('Docker fallback pulls and verifies the pinned official OpenROAD image', async () => {
  const env = { PATH: '/tools' };
  const calls = [];
  const runImpl = async (command, args) => {
    calls.push([command, ...args]);
    if (command === 'openroad') return 127;
    if (args[0] === 'image') return 1;
    return 0;
  };
  const provider = await requireOpenroad({ env, runImpl });
  assert.deepEqual(provider, { kind: 'docker', image: openroadImage, platform: openroadPlatform });
  assert.equal(env.IC_OPENROAD_PROVIDER, 'docker');
  assert.equal(env.IC_OPENROAD_DOCKER_IMAGE, openroadImage);
  assert.equal(env.IC_OPENROAD_DOCKER_PLATFORM, openroadPlatform);
  assert.deepEqual(calls, [
    ['openroad', '-version'],
    ['docker', 'version', '--format', '{{.Server.Version}}'],
    ['docker', 'image', 'inspect', openroadImage],
    ['docker', 'pull', '--platform', openroadPlatform, openroadImage],
    ['docker', 'run', '--rm', '--platform', openroadPlatform, openroadImage, 'openroad', '-version'],
  ]);
});

test('setup fails clearly when neither native OpenROAD nor Docker is available', async () => {
  await assert.rejects(
    requireOpenroad({ runImpl: async () => 127 }),
    /OpenROAD was not found.*Docker is unavailable/s,
  );
  await assert.rejects(
    requireOpenroad({ env: { IC_OPENROAD_MODE: 'host' }, runImpl: async () => 127 }),
    /requires `openroad` on PATH/,
  );
});

test('Docker provider installs an executable OpenROAD wrapper into cached runtimes', async t => {
  const runtime = await fixture(t);
  const bin = path.join(runtime, 'oss-cad-suite', 'bin');
  await mkdir(bin, { recursive: true });
  await configureOpenroad(runtime, { kind: 'docker', image: openroadImage, platform: openroadPlatform });
  const wrapper = path.join(bin, 'openroad');
  const body = await readFile(wrapper, 'utf8');
  assert.match(body, /docker.*run/s);
  assert.match(body, /IC_OPENROAD_MOUNTS/);
  if (process.platform !== 'win32') assert.ok((await stat(wrapper)).mode & 0o100);
  await configureOpenroad(runtime, { kind: 'host' });
  await assert.rejects(readFile(wrapper), /ENOENT/);
});

test('help and invalid arguments do not trigger setup', () => {
  const cli = path.join(packageRoot, 'npm', 'cli.mjs');
  const help = spawnSync(process.execPath, [cli, '--help'], { encoding: 'utf8' });
  assert.equal(help.status, 0);
  assert.match(help.stdout, /setup\|doctor/);
  assert.match(help.stdout, /OpenROAD/);
  const bad = spawnSync(process.execPath, [cli, 'remove-everything'], { encoding: 'utf8' });
  assert.equal(bad.status, 1);
  assert.equal(bad.stdout, '');
  assert.match(bad.stderr, /Unknown command/);
});

test('child startup failure is surfaced', async () => {
  await assert.rejects(run('/definitely/not/an/executable', []), /ENOENT/);
});
