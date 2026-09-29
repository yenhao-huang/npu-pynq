import path from 'node:path';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { packageRoot } from './bootstrap.mjs';

const marker = '// Managed by ic-tools init pi.';
const legacy = '// Auto-discovered by pi. The implementation lives with the tools it exposes,\n' +
  '// so it stays in step with them; this file only points at it.\n' +
  'export { default } from "../../tools/ic/integrations/pi/extension.ts";';

export async function initPi(project, root = packageRoot) {
  const target = path.join(project, '.pi', 'extensions', 'ic-design-tools.ts');
  // Use an absolute filesystem specifier: pi's TS loader resolves this installed package.
  const source = path.join(root, 'integrations', 'pi', 'managed.ts').replaceAll('\\', '/');
  const content = `${marker}\nexport { default } from ${JSON.stringify(source)};\n`;
  const existing = await readFile(target, 'utf8').catch(error => {
    if (error.code !== 'ENOENT') throw error;
    return null;
  });
  if (existing === content) return target;
  const normalized = existing?.replaceAll('\r\n', '\n').trim();
  const managed = normalized && /^\/\/ Managed by ic-tools init pi\.\nexport \{ default \} from "(?:[^"\\]|\\.)*";$/.test(normalized);
  if (existing !== null && normalized !== legacy && !managed)
    throw new Error(`Refusing to overwrite custom extension ${target}. Move it aside before running init pi.`);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, content, { flag: existing === null ? 'wx' : 'w' });
  return target;
}
