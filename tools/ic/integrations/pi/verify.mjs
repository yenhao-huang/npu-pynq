/**
 * Prove the pi extension registers the ic tools, without contacting a model.
 *
 * pi's own SDK loads the extension exactly as the CLI does, so the tool list
 * this prints is the list an agent would see. It then invokes one tool
 * directly, which exercises the whole path: pi -> extension -> `ic` -> daemon
 * or in-process dispatch.
 */
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

// Resolve pi from wherever it is installed -- usually globally, since it is a
// CLI. ESM ignores NODE_PATH, so the global root is looked up and imported by
// absolute URL rather than by bare specifier.
function resolvePi() {
  const candidates = [];
  try {
    candidates.push(execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim());
  } catch {}
  candidates.push("node_modules");
  for (const root of candidates) {
    const entry = `${root}/@earendil-works/pi-coding-agent/dist/index.js`;
    try {
      return pathToFileURL(entry).href;
    } catch {}
  }
  throw new Error("pi-coding-agent not found; npm install -g @earendil-works/pi-coding-agent");
}

const pi = await import(resolvePi());
const { createAgentSession, SessionManager, loadSkillsFromDir } = pi;

const extension = process.argv[2] ?? ".pi/extensions/ic-design-tools.ts";
const { session } = await createAgentSession({
  sessionManager: SessionManager.inMemory(),
  extensions: [extension],
  noTools: "builtin",
});

const tools = session.agent.state.tools ?? [];
const names = tools.map((t) => t.name).sort();
console.log("REGISTERED:", JSON.stringify(names));

const lint = tools.find((t) => t.name === "lint");
if (!lint) {
  console.error("FAIL: lint tool was not registered");
  process.exit(1);
}
console.log("lint schema required:", JSON.stringify(lint.parameters?.required ?? null));

const result = await lint.execute("verify-1", {
  files: ["src/hw/rtl/systolic_array/npu_pe.sv"],
  top: "npu_pe",
}, new AbortController().signal, () => {});
const text = result.content.find((c) => c.type === "text")?.text ?? "";
console.log("lint result:", text.slice(0, 200));
// The skill is the other half of the pi integration: the tools are callable,
// and the skill tells the model when and how to call them. pi discovers it
// through the `skills` entry in .pi/settings.json, so that is what is read
// here rather than a path hard-coded into this script.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

let skillDirs = [];
try {
  skillDirs = JSON.parse(readFileSync(".pi/settings.json", "utf8")).skills ?? [];
} catch {
  console.error("FAIL: .pi/settings.json is missing or has no `skills` entry");
}

const skillNames = [];
const diagnostics = [];
for (const dir of skillDirs) {
  const loaded = loadSkillsFromDir({ dir: resolve(process.cwd(), dir), source: "project" });
  skillNames.push(...loaded.skills.map((s) => s.name));
  diagnostics.push(...loaded.diagnostics);
}
console.log("SKILLS:", JSON.stringify(skillNames));
if (diagnostics.length) console.log("SKILL DIAGNOSTICS:", JSON.stringify(diagnostics));

const toolsOk = names.length >= 8;
const lintOk = text.includes('"ok": true') || text.includes('"ok":true');
const skillOk = skillNames.includes("ic-design-tools");
console.log(`tools=${toolsOk} lint=${lintOk} skill=${skillOk}`);
process.exit(toolsOk && lintOk && skillOk ? 0 : 1);
