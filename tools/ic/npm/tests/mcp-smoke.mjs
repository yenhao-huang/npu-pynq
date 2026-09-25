// Explicit integration test: provisions the real runtime and exercises MCP wire I/O.
// Run from the repository root, with IC_MCP_CACHE pointing at an isolated cache.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { packageRoot } from '../bootstrap.mjs';

const root = process.cwd();
const output = path.join(root, 'src/test/build/npm-mcp-acceptance');
await mkdir(output, { recursive: true });
const cli = path.join(process.env.IC_SMOKE_PACKAGE_ROOT || packageRoot, 'npm/cli.mjs');
const child = spawn(process.execPath, [cli, '--project', root], {
  env: { ...process.env, IC_ROOT: path.join(output, 'ic'), PYTHONPATH: '/invalid-ambient-python-path' },
  stdio: ['pipe', 'pipe', 'pipe'],
});
let stderr = '';
child.stderr.on('data', data => { stderr += data; });
const pending = new Map();
let id = 0;
const transcript = [];
const lines = createInterface({ input: child.stdout });
lines.on('line', line => {
  try {
    const message = JSON.parse(line); // Any bootstrap chatter on stdout fails this test.
    if (pending.has(message.id)) {
      const { resolve, reject, timer } = pending.get(message.id);
      clearTimeout(timer); pending.delete(message.id);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    }
  } catch (error) { for (const p of pending.values()) p.reject(error); child.kill(); }
});
const exited = new Promise(resolve => child.once('exit', (code, signal) => {
  for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error(`MCP exited ${code}/${signal}: ${stderr}`)); }
  pending.clear(); resolve({ code, signal });
}));
function request(method, params) {
  return new Promise((resolve, reject) => {
    const requestId = ++id;
    const timer = setTimeout(() => { pending.delete(requestId); reject(new Error(`Timeout: ${method}`)); }, 180000);
    pending.set(requestId, { resolve, reject, timer });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: requestId, method, params }) + '\n');
  });
}
async function call(name, args) {
  const result = await request('tools/call', { name, arguments: args });
  assert.ok(!result.isError, JSON.stringify(result));
  const value = result.structuredContent || JSON.parse(result.content.find(x => x.type === 'text').text);
  assert.ok(!value.error, JSON.stringify(value));
  transcript.push({ name, args, result: value });
  return value;
}
try {
  const initialization = await request('initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'npm-acceptance', version: '1.0' } });
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
  const catalogue = await request('tools/list', {});
  assert.deepEqual(catalogue.tools.map(t => t.name).sort(), ['lint', 'sim', 'signals', 'first_mismatch', 'value_at', 'value_range', 'show_wave', 'synth'].sort());
  const files = ['src/hw/rtl/systolic_array/npu_pe.sv', 'src/hw/rtl/systolic_array/npu_systolic_array.sv'];
  const lint = await call('lint', { files, top: 'npu_systolic_array' });
  assert.equal(lint.error_count, 0);
  const sim = await call('sim', { files: [...files, 'src/hw/tb/systolic_array/tb_npu_systolic_array.sv'], tb: 'tb_npu_systolic_array', top: 'npu_systolic_array', trace: true, timeout_s: 120 });
  assert.equal(sim.ok, true, JSON.stringify(sim));
  assert.ok(sim.wave);
  await call('signals', { wave: sim.wave, pattern: 'accumulators', limit: 10 });
  const at = await call('value_at', { wave: sim.wave, signals: ['TOP.tb_npu_systolic_array.accumulators'], cycle: 8 });
  const packed = BigInt(at.values['TOP.tb_npu_systolic_array.accumulators'].hex);
  const lanes = [0, 1, 2, 3].map(i => Number(BigInt.asIntN(32, packed >> BigInt(i * 32))));
  assert.deepEqual(lanes, [-8, 48, 83, 10]);
  const clear = await call('value_at', { wave: sim.wave, signals: ['TOP.tb_npu_systolic_array.accumulators'], cycle: 9 });
  assert.equal(BigInt(clear.values['TOP.tb_npu_systolic_array.accumulators'].hex), 0n);
  await call('value_range', { wave: sim.wave, signal: 'TOP.tb_npu_systolic_array.enable', from_cycle: 0, to_cycle: 12 });
  const synth = await call('synth', { files, top: 'npu_systolic_array', mode: 'estimate', timeout_s: 120 });
  assert.equal(synth.ok, true, JSON.stringify(synth));
  await writeFile(path.join(output, 'mcp-smoke.json'), JSON.stringify({ initialization, tools: catalogue.tools.map(t => t.name), transcript }, null, 2));
  child.stdin.end();
  const watchdog = setTimeout(() => child.kill('SIGTERM'), 10000);
  const exit = await exited;
  clearTimeout(watchdog);
  assert.equal(exit.code, 0, stderr);
  console.log(`PASS MCP initialize/list/lint/sim/debug/synth/EOF: ${output}`);
} finally {
  child.kill('SIGTERM');
  await writeFile(path.join(output, 'mcp.stderr'), stderr);
}
