/**
 * pi extension: registers the ic design tools as native pi tools.
 *
 * pi has no built-in MCP support -- that is an explicit upstream design
 * choice, not an oversight -- so the integration is an extension rather than
 * an `mcp.json` entry.
 *
 * The tool list is not written here. It is fetched at startup by running
 * `ic tools`, whose JSON Schemas come straight from the categories' pydantic
 * models. A new category therefore shows up in pi with no change to this
 * file, exactly as it does in MCP, the CLI and the HTTP API. TypeBox schemas
 * are plain JSON Schema objects at runtime, so the schema can be handed to
 * `registerTool` as-is.
 */

import { spawn } from "node:child_process";
import { execFileSync } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

type IcTool = {
  name: string;
  category: string;
  summary: string;
  long_running: boolean;
  input_schema: Record<string, unknown>;
};

/** How `ic` is invoked. Override when it is not on PATH. */
const IC_BIN = process.env.IC_BIN ?? "ic";
const IC_CWD = process.env.IC_CWD ?? process.cwd();

/** Long-running ops answer inline when they finish inside this window. */
const WAIT_S = process.env.IC_WAIT_S ?? "240";

function catalogue(): IcTool[] {
  try {
    const raw = execFileSync(IC_BIN, ["tools", "--compact"], {
      cwd: IC_CWD,
      encoding: "utf8",
      maxBuffer: 32 * 1024 * 1024,
    });
    return JSON.parse(raw).tools as IcTool[];
  } catch (error) {
    // A missing `ic` must not stop pi from starting; the notice below tells
    // the user what to install, and every other tool keeps working.
    return [];
  }
}

function runIc(tool: IcTool, params: Record<string, unknown>, signal: AbortSignal): Promise<string> {
  const argv: string[] = [tool.name, "--compact", "--wait-s", WAIT_S];
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    const flag = `--${key.replace(/_/g, "-")}`;
    if (typeof value === "boolean") {
      argv.push(value ? flag : `--no-${key.replace(/_/g, "-")}`);
    } else if (Array.isArray(value)) {
      if (value.length === 0) continue;
      argv.push(flag, ...value.map(String));
    } else {
      argv.push(flag, String(value));
    }
  }

  return new Promise((resolve) => {
    const child = spawn(IC_BIN, argv, { cwd: IC_CWD });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    const onAbort = () => child.kill("SIGTERM");
    signal.addEventListener("abort", onAbort, { once: true });
    child.on("close", () => {
      signal.removeEventListener("abort", onAbort);
      // `ic` prints JSON on stdout for every outcome, including errors, and
      // uses stderr only when it could not produce any. Both are results the
      // model should see rather than exceptions that hide them.
      resolve(stdout.trim() || stderr.trim() || "{}");
    });
    child.on("error", (error) => {
      resolve(JSON.stringify({ error: { code: "spawn_failed", message: String(error) } }));
    });
  });
}

export default function (pi: ExtensionAPI) {
  const tools = catalogue();

  if (tools.length === 0) {
    pi.on("session_start", async (_event, ctx) => {
      ctx.ui.notify(
        `ic design tools unavailable: could not run \`${IC_BIN} tools\`. ` +
          "Install with `pip install -e tools/ic`, or set IC_BIN.",
        "warn",
      );
    });
    return;
  }

  for (const tool of tools) {
    pi.registerTool({
      name: tool.name,
      label: `ic ${tool.category}`,
      description: tool.long_running
        ? `${tool.summary} May take minutes; a slow run returns a job_id to wait on.`
        : tool.summary,
      parameters: tool.input_schema as never,
      async execute(_toolCallId, params, signal) {
        const text = await runIc(tool, params as Record<string, unknown>, signal);
        return { content: [{ type: "text", text }], details: {} };
      },
    });
  }

  pi.registerCommand("ic", {
    description: "Show which ic design tool backends are installed",
    handler: async (_args, ctx) => {
      const report = execFileSync(IC_BIN, ["doctor"], { cwd: IC_CWD, encoding: "utf8" });
      ctx.ui.notify(report, "info");
    },
  });
}
