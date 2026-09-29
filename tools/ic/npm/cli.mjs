#!/usr/bin/env node
import path from 'node:path';
import { initPi } from './init-pi.mjs';
import { ensureRuntime, runtimeEnvironment, pythonPath, run, log, cacheRoot, platformSpec } from './bootstrap.mjs';

async function main() {
  const args = process.argv.slice(2);
  let command = 'serve';
  let project = process.cwd();
  if (args[0] && !args[0].startsWith('-')) command = args.shift();
  if (command === 'init') {
    if (args.shift() !== 'pi') throw new Error('Usage: ic-tools init pi [--project DIR]');
    command = 'init-pi';
  }
  while (args.length) {
    const option = args.shift();
    if (option === '--project' && args[0]) project = path.resolve(args.shift());
    else if (option === '--help' || option === '-h') { command = 'help'; }
    else throw new Error(`Unknown or incomplete option: ${option}`);
  }
  if (command === 'help') {
    process.stdout.write('Usage: ic-tools [serve|setup|doctor] [--project DIR]\n       ic-tools init pi [--project DIR]\nDefault: stdio MCP server; prepares bundled dependencies after verifying OpenROAD on PATH.\nIC_MCP_CACHE: runtime cache. IC_ROOT: project artifacts. IC_MCP_ARCHIVE: verified offline toolchain archive.\n');
    return 0;
  }
  if (!['serve', 'setup', 'doctor', 'init-pi'].includes(command)) throw new Error(`Unknown command: ${command}`);
  // Reject bad project/platform before any large download.
  const { stat } = await import('node:fs/promises');
  if (!(await stat(project)).isDirectory()) throw new Error(`Not a project directory: ${project}`);
  if (command === 'init-pi') {
    log(`pi extension ready: ${await initPi(project)}. Start pi from ${project}.`);
    return 0;
  }
  const spec = platformSpec();
  const runtime = await ensureRuntime({ spec });
  if (command === 'setup') { log(`Setup complete: ${runtime}`); return 0; }
  const env = runtimeEnvironment(runtime);
  if (command === 'doctor') {
    log(`Platform: ${spec.key}; cache: ${cacheRoot()}; project: ${project}`);
    return run(pythonPath(runtime), ['-m', 'ic_cli.main', 'doctor'], { env, cwd: project, mcp: true });
  }
  return run(pythonPath(runtime), ['-m', 'ic_mcp.server'], { env, cwd: project, mcp: true });
}

try { process.exitCode = await main(); }
catch (error) {
  log(error.message);
  if (process.argv[2] === 'setup') {
    log('Setup failed. Fix the error above, then retry ic-tools setup. If npm did not retain the CLI, rerun npm install.');
  }
  process.exitCode = 1;
}
