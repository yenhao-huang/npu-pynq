import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { ensureRuntime, pythonPath, runtimeEnvironment } from "../../npm/bootstrap.mjs";
import { createExtension } from "./extension.ts";

// The npm runtime owns Python and HDL dependencies; no activated venv is needed.
export default async function (pi: ExtensionAPI) {
  const runtime = await ensureRuntime();
  const env = runtimeEnvironment(runtime);
  // This integration always executes in the current project, using its local store.
  delete env.IC_DAEMON_URL;
  return createExtension({
    command: pythonPath(runtime), args: ["-m", "ic_cli.main"], env,
  })(pi);
}
