// Run from the repository root with pi installed and a completed IC_MCP_CACHE.
// IC_SMOKE_PACKAGE_ROOT selects an installed tarball instead of the source tree.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { packageRoot } from '../bootstrap.mjs';

const repo = process.cwd();
await mkdir(path.join(repo, '.ic'), { recursive: true });
const project = await mkdtemp(path.join(repo, '.ic/pi-package-'));
process.env.IC_ROOT = path.join(project, '.ic');
const installed = process.env.IC_SMOKE_PACKAGE_ROOT || packageRoot;
const cli = path.join(installed, 'npm/cli.mjs');
execFileSync(process.execPath, [cli, 'init', 'pi', '--project', project]);
const npmRoot = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
const { createAgentSession, SessionManager } = await import(pathToFileURL(
  path.join(npmRoot, '@earendil-works/pi-coding-agent/dist/index.js')).href);

// Ambient legacy settings must not be needed by the managed integration.
process.env.IC_BIN = '/invalid/legacy/ic';
process.env.IC_CWD = '/invalid/legacy/project';
process.env.IC_DAEMON_URL = 'http://127.0.0.1:1';
process.chdir(project);
const { session } = await createAgentSession({
  sessionManager: SessionManager.inMemory(),
  extensions: [path.join(project, '.pi/extensions/ic-design-tools.ts')],
  noTools: 'builtin',
});
const tools = session.agent.state.tools;
assert.deepEqual(tools.map(t => t.name).sort(),
  ['first_mismatch', 'lint', 'show_wave', 'signals', 'sim', 'synth', 'value_at', 'value_range']);
const transcript = [];
async function call(name, params) {
  const result = await tools.find(t => t.name === name).execute(
    `acceptance-${name}`, params, new AbortController().signal, () => {});
  const value = JSON.parse(result.content.find(c => c.type === 'text').text);
  assert.ok(!value.error, JSON.stringify(value));
  transcript.push({ name, params, result: value });
  return value;
}
const fixture = path.join(repo, 'tools/ic/tests/fixtures');
assert.equal((await call('lint', {
  files: [path.join(repo, 'src/hw/rtl/systolic_array/npu_pe.sv')], top: 'npu_pe',
})).error_count, 0);
const sim = await call('sim', {
  files: [path.join(fixture, 'counter.sv'), path.join(fixture, 'tb_counter.sv')], tb: 'tb_counter',
});
assert.equal(sim.built, true);
assert.equal(sim.ok, false);
assert.ok(sim.wave);
const wave = sim.wave;
await call('signals', { wave, pattern: '*count*' });
const mismatch = await call('first_mismatch', { wave, ref: 'ref_count', dut: 'dut_count' });
assert.equal(mismatch.cycle, 8);
assert.equal(mismatch.ref_value.dec, 8);
assert.equal(mismatch.dut_value.dec, 7);
const at = await call('value_at', { wave, signals: ['ref_count', 'dut_count'], cycle: 8 });
assert.deepEqual(at.missing, []);
assert.equal(at.values.dut_count.dec, 7);
await call('value_range', { wave, signal: 'dut_count', from_cycle: 5, to_cycle: 12 });
const view = await call('show_wave', { wave, signals: ['ref_count', 'dut_count'], center_cycle: 8, launch: false });
assert.equal(view.launched, false);
assert.ok(view.savefile);
assert.equal((await call('synth', { files: [path.join(fixture, 'counter.sv')], top: 'counter_ref', mode: 'estimate' })).ok, true);
await writeFile(path.join(project, 'evidence.json'), JSON.stringify(transcript, null, 2));
console.log(`PASS: installed pi extension registered and executed all eight tools; evidence ${project}`);
process.exit(0);
