#!/usr/bin/env node
import path from 'node:path';
import { ensureRuntime, runtimeEnvironment, pythonPath, run, log, cacheRoot, platformSpec } from './bootstrap.mjs';

async function main() {
  const args = process.argv.slice(2);
  let command = 'serve';
  let project = process.cwd();
  if (args[0] && !args[0].startsWith('-')) command = args.shift();
  while (args.length) {
    const option = args.shift();
    if (option === '--project' && args[0]) project = path.resolve(args.shift());
    else if (option === '--help' || option === '-h') { command = 'help'; }
    else throw new Error(`Unknown or incomplete option: ${option}`);
  }
  if (command === 'help') {
    process.stdout.write('Usage: ic-design-mcp [serve|setup|doctor] [--project DIR]\nDefault: stdio MCP server; automatically installs missing dependencies.\nIC_MCP_CACHE: runtime cache. IC_ROOT: project artifacts. IC_MCP_ARCHIVE: verified offline toolchain archive.\n');
    return 0;
  }
  if (!['serve', 'setup', 'doctor'].includes(command)) throw new Error(`Unknown command: ${command}`);
  // Reject bad project/platform before any large download.
  const { stat } = await import('node:fs/promises');
  if (!(await stat(project)).isDirectory()) throw new Error(`Not a project directory: ${project}`);
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
catch (error) { log(error.message); process.exitCode = 1; }
