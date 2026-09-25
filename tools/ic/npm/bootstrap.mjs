import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { access, mkdir, open, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { homedir, hostname } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import * as tar from 'tar';

export const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(await readFile(new URL('./toolchains.json', import.meta.url), 'utf8'));
const dependencyLock = await readFile(new URL('./requirements.lock', import.meta.url));
export const log = message => process.stderr.write(`[ic-design-mcp] ${message}\n`);
export const exists = async p => access(p).then(() => true, () => false);

export function platformSpec(platform = process.platform, arch = process.arch) {
  const key = `${platform}-${arch}`;
  const asset = manifest.platforms[key];
  if (!asset) throw new Error(`Unsupported platform ${key}. On Windows, install Node.js and run this package inside WSL. Supported: ${Object.keys(manifest.platforms).join(', ')}.`);
  const name = `oss-cad-suite-${key}-${manifest.release.replaceAll('-', '')}.tgz`;
  return { ...asset, key, name, url: `${manifest.source}/releases/download/${manifest.release}/${name}` };
}

export function cacheRoot(env = process.env) {
  return path.resolve(env.IC_MCP_CACHE || (process.platform === 'darwin'
    ? path.join(homedir(), 'Library', 'Caches', 'ic-design-mcp')
    : path.join(env.XDG_CACHE_HOME || path.join(homedir(), '.cache'), 'ic-design-mcp')));
}

export function runtimeKey(spec) {
  return `${manifest.release}-${spec.key}-${createHash('sha256').update(spec.sha256).update(dependencyLock).update('bootstrap-v1').digest('hex').slice(0, 16)}`;
}

export async function sha256(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

export async function verify(file, spec) {
  if ((await stat(file)).size !== spec.size || await sha256(file) !== spec.sha256)
    throw new Error(`Integrity check failed for ${file}. Expected SHA256 ${spec.sha256}. Remove the corrupt archive and retry setup.`);
}

export async function download(spec, destination, { fetchImpl = fetch } = {}) {
  if (await exists(destination)) { await verify(destination, spec); return destination; }
  await mkdir(path.dirname(destination), { recursive: true });
  const temp = `${destination}.${randomUUID()}.part`;
  try {
    log(`Downloading ${spec.name} (${Math.ceil(spec.size / 1024 / 1024)} MiB)`);
    if (new URL(spec.url).protocol !== 'https:') throw new Error('Toolchain downloads require HTTPS.');
    const response = await fetchImpl(spec.url, { signal: AbortSignal.timeout(20 * 60 * 1000) });
    if (!response.ok || !response.body) throw new Error(`Download failed: HTTP ${response.status}`);
    if (response.url && new URL(response.url).protocol !== 'https:') throw new Error('Refusing a non-HTTPS redirect.');
    let bytes = 0;
    let lastNotice = Date.now();
    const meter = new Transform({ transform(chunk, _, callback) {
      bytes += chunk.length;
      if (bytes > spec.size) return callback(new Error('Archive exceeds its pinned size.'));
      if (Date.now() - lastNotice > 15000) { log(`Downloaded ${Math.floor(bytes / 1024 / 1024)} MiB`); lastNotice = Date.now(); }
      callback(null, chunk);
    } });
    await pipeline(Readable.fromWeb(response.body), meter, createWriteStream(temp, { flags: 'wx' }));
    await verify(temp, spec);
    await rename(temp, destination);
    return destination;
  } finally { await rm(temp, { force: true }); }
}

export async function withLock(lockPath, action, { timeoutMs = 30 * 60 * 1000, pollMs = 500 } = {}) {
  await mkdir(path.dirname(lockPath), { recursive: true });
  const start = Date.now();
  let handle;
  while (!handle) {
    try { handle = await open(lockPath, 'wx', 0o600); }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      const owner = await readFile(lockPath, 'utf8').then(JSON.parse).catch(() => null);
      if (owner?.host === hostname() && Number.isInteger(owner.pid)) {
        try { process.kill(owner.pid, 0); }
        catch (e) { if (e.code === 'ESRCH') throw new Error(`Interrupted setup lock: ${lockPath}. Its process ${owner.pid} has exited. Remove this lock file and retry setup.`); }
      }
      if (Date.now() - start >= timeoutMs) throw new Error(`Timed out waiting for setup lock ${lockPath}. Check the owning installer before removing it.`);
      await sleep(pollMs);
    }
  }
  try {
    await handle.writeFile(JSON.stringify({ pid: process.pid, host: hostname() }));
    return await action();
  } finally { await handle.close(); await rm(lockPath, { force: true }); }
}

export async function extract(archive, destination) {
  await mkdir(destination, { recursive: true });
  const invalid = [];
  // Validate the complete archive before invoking the platform tar. OSS CAD Suite
  // contains directory aliases that node-tar's extractor deliberately refuses.
  await tar.t({ file: archive, strict: true,
    filter: (name, entry) => {
      const normalized = path.posix.normalize(name);
      if (!(normalized === 'oss-cad-suite' || normalized.startsWith('oss-cad-suite/')) || name.includes('\\'))
        { invalid.push(`Invalid archive member: ${name}`); return false; }
      if (entry.linkpath) {
        const target = entry.type === 'Link' ? entry.linkpath : path.posix.join(path.posix.dirname(name), entry.linkpath);
        const resolved = path.posix.normalize(target);
        if (path.posix.isAbsolute(entry.linkpath) || !(resolved === 'oss-cad-suite' || resolved.startsWith('oss-cad-suite/')))
          { invalid.push(`Archive link escapes the toolchain: ${name} -> ${entry.linkpath}`); return false; }
      }
      return true;
    },
  });
  if (invalid.length) throw new Error(invalid.slice(0, 3).join('; '));
  await checked('tar', ['-xzf', archive, '-C', destination]);
}

export function runtimeEnvironment(runtime, env = process.env) {
  const suite = path.join(runtime, 'oss-cad-suite');
  const result = { ...env,
    PATH: [path.join(suite, 'bin'), path.join(suite, 'py3bin'), env.PATH || ''].join(path.delimiter),
    PYTHONPATH: [packageRoot, path.join(runtime, 'python-packages')].join(path.delimiter),
    PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1',
    VERILATOR_ROOT: path.join(suite, 'share', 'verilator'),
    VIRTUAL_ENV: suite,
  };
  delete result.PYTHONHOME;
  return result;
}

export function pythonPath(runtime) { return path.join(runtime, 'oss-cad-suite', 'py3bin', 'python3'); }

// Bootstrap child output goes to stderr; only the final MCP process gets stdout.
export function run(command, args, { env = process.env, cwd = process.cwd(), mcp = false } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env, cwd, stdio: mcp ? 'inherit' : ['ignore', 2, 2] });
    const onInt = () => child.kill('SIGINT');
    const onTerm = () => child.kill('SIGTERM');
    process.on('SIGINT', onInt); process.on('SIGTERM', onTerm);
    const cleanup = () => { process.off('SIGINT', onInt); process.off('SIGTERM', onTerm); };
    child.once('error', error => { cleanup(); reject(error); });
    child.once('exit', (code, signal) => { cleanup(); resolve(code ?? (signal === 'SIGINT' ? 130 : 143)); });
  });
}

export async function checked(command, args, options) {
  const code = await run(command, args, options);
  if (code !== 0) throw new Error(`${path.basename(command)} exited ${code}. See diagnostics above.`);
}

export async function completed(runtime, key) {
  const marker = await readFile(path.join(runtime, 'complete.json'), 'utf8').then(JSON.parse).catch(() => null);
  return marker?.key === key && await exists(pythonPath(runtime)) && await exists(path.join(runtime, 'python-packages', 'mcp', '__init__.py'));
}

export async function ensureRuntime({ root = cacheRoot(), spec = platformSpec(), install = installRuntime } = {}) {
  const key = runtimeKey(spec);
  const runtime = path.join(root, key);
  if (await completed(runtime, key)) return runtime;
  return withLock(path.join(root, `${key}.lock`), async () => {
    if (await completed(runtime, key)) return runtime;
    if (await exists(runtime)) throw new Error(`Incomplete runtime at ${runtime}. Move it aside and rerun setup.`);
    const stage = path.join(root, `${key}.staging-${randomUUID()}`);
    await mkdir(stage, { recursive: true });
    try {
      await install(stage, root, spec);
      await writeFile(path.join(stage, 'complete.json'), JSON.stringify({ key, release: manifest.release, platform: spec.key }));
      await rename(stage, runtime);
      return runtime;
    } finally { await rm(stage, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }).catch(error => log(`Could not remove staging directory ${stage}: ${error.message}`)); }
  });
}

async function installRuntime(stage, root, spec) {
  let archive = process.env.IC_MCP_ARCHIVE;
  if (archive) { archive = path.resolve(archive); await verify(archive, spec); }
  else archive = await download(spec, path.join(root, 'downloads', spec.name));
  log('Extracting the verified toolchain');
  await extract(archive, stage);
  const env = runtimeEnvironment(stage);
  const python = pythonPath(stage);
  log('Installing hash-locked Python dependencies');
  await checked(python, ['-m', 'pip', '--isolated', '--disable-pip-version-check', 'install', '--no-user',
    '--require-hashes', '--only-binary=:all:', '--target', path.join(stage, 'python-packages'),
    '-r', path.join(packageRoot, 'npm', 'requirements.lock')], { env });
  await checked(python, ['-c', 'import mcp, pydantic; from ic_mcp.server import build_server; build_server()'], { env });
  await checked(python, ['-m', 'ic_cli.main', 'doctor'], { env });
  log('Runtime ready');
}
