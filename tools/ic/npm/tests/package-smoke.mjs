// Linux acceptance: use installed pi and an existing completed IC_MCP_CACHE.
// IC_SMOKE_PACKAGE_ROOT may select an extracted release tarball.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { packageRoot } from '../bootstrap.mjs';

const repo = process.cwd();
const project = await mkdtemp(path.join(tmpdir(), 'ic-tools-pi-install-'));
const agentDir = path.join(project, 'agent');
const installed = process.env.IC_SMOKE_PACKAGE_ROOT || packageRoot;
const npmRoot = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
const piRoot = path.join(npmRoot, '@earendil-works/pi-coding-agent');
const env = { ...process.env, PI_CODING_AGENT_DIR: agentDir, PI_TELEMETRY: '0' };
execFileSync(process.execPath, [path.join(piRoot, 'dist/cli.js'), 'install', '--local', installed],
  { cwd: project, env, stdio: 'inherit' });
const settings = JSON.parse(await readFile(path.join(project, '.pi/settings.json'), 'utf8'));
assert.equal(settings.packages.length, 1);
await assert.rejects(access(path.join(project, '.pi/extensions/ic-design-tools.ts')));

process.chdir(project);
process.env.IC_ROOT = path.join(project, '.ic');
const { createAgentSession, SessionManager } = await import(pathToFileURL(path.join(piRoot, 'dist/index.js')).href);
const { session, extensionsResult } = await createAgentSession({
  cwd: project, agentDir, sessionManager: SessionManager.inMemory(), noTools: 'builtin',
});
try {
  assert.deepEqual(extensionsResult.errors, []);
  const tools = session.agent.state.tools;
  assert.deepEqual(tools.map(t => t.name).sort(),
    ['first_mismatch', 'lint', 'ppa', 'show_wave', 'signals', 'sim', 'synth', 'value_at', 'value_range']);
  const result = await tools.find(t => t.name === 'lint').execute('package-smoke', {
    files: [path.join(repo, 'src/hw/rtl/npu_matrix/npu_accelerator/npu_matrix_core/npu_matrix_datapath/systolic_array/npu_pe.sv')], top: 'npu_pe',
  }, new AbortController().signal, () => {});
  const value = JSON.parse(result.content.find(c => c.type === 'text').text);
  assert.equal(value.ok, true);
  assert.equal(value.error_count, 0);
  console.log(`PASS: pi install discovered nine tools and executed lint without a wrapper; ${project}`);
} finally {
  session.dispose();
}
