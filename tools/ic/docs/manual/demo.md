# pi tool verification demo

Use these prompts after following [installation](installation.md). Commands run
in Linux Bash from the NPU repository root. Configure pi's model provider first.
The prompts below are verification instructions; expected results are not a
claim that your installed version or model has passed them.

## 1. Check the installed package and start pi

```bash
pi list
# Reuse this cache only in the existing howard experiment sandbox.
export IC_MCP_CACHE="$HOME/.cache/ic-tools/ic-design-mcp"
pi
```

For other environments, use the default cache or your own completed cache root;
see [environment reuse](installation.md#reuse-an-existing-environment).

To test the published package instead of local source, exit pi and replace the
local registration when `pi list` shows `./tools/ic`:

```bash
pi remove --local ./tools/ic
pi install npm:@jony2156/ai-eda-tools@0.1.1 --local
pi
```

Use the exact source and scope shown by `pi list` if they differ. Keep only one
IC package or wrapper registration. See [wrapper migration](installation.md#migrate-an-existing-pi-wrapper)
if startup reports conflicting tools.

Inside pi, enter:

```text
/ic
```

**Expected:** backend inventory appears, with no extension loading or duplicate
tool errors. The lint/simulation/estimate backends needed below are available.
Vivado can be unavailable for this demo. Inventory alone does not prove that
the model can call the tools.

## 2. Quick lint check

Paste this prompt into pi:

```text
Actually call the native lint tool on
src/hw/rtl/npu_matrix/npu_accelerator/npu_matrix_core/npu_matrix_datapath/systolic_array/npu_pe.sv, with top=npu_pe.
Leave the source unchanged. Report ok, error_count, and warning_count from
the returned result. Explain any warnings. Do not replace the tool with Bash.
```

**Expected:** the transcript shows a native `lint` call, returning `ok: true`
and `error_count: 0`. Report the actual warning count; warnings need not be zero.
A text-only claim without a tool call does not pass.

## 3. Verify all eight tools

The counter fixture is intentionally incorrect. This demo checks whether the
tools can locate its mismatch; a simulation failure is expected.

Paste this prompt into pi:

```text
Use all eight native IC tools and leave all source files unchanged.
1. lint: files=["src/hw/rtl/npu_matrix/npu_accelerator/npu_matrix_core/npu_matrix_datapath/systolic_array/npu_pe.sv"], top="npu_pe".
2. sim: files=["tools/ic/tests/fixtures/counter.sv",
   "tools/ic/tests/fixtures/tb_counter.sv"], tb="tb_counter".
3. Use the wave handle returned by this sim for every waveform tool below.
   signals: pattern="*count*".
   first_mismatch: ref="ref_count", dut="dut_count", context=3.
4. value_at: signals=["ref_count","dut_count","enable"] at the mismatch cycle.
   value_range: signal="dut_count", from_cycle=5, to_cycle=12, max_points=16.
5. show_wave: signals=["ref_count","dut_count","enable"], center_cycle set
   to the mismatch cycle, launch=false.
6. synth: files=["tools/ic/tests/fixtures/counter.sv"], top="counter_ref",
   mode="estimate".
Report each tool as passed, failed, or not verified, with actual results and
run/wave handles. Explain the cause of the counter mismatch.
Do not use Bash instead of native tools, read raw waveforms or whole logs,
repair the fixture, or claim timing closure. If a prerequisite fails, report
blocked downstream checks as not verified; do not invent results or handles.
```

### Expected results

| Tool | Expected evidence |
| --- | --- |
| `lint` | Zero lint errors for `npu_pe`. |
| `sim` | Build completes and a wave handle is returned, with the fixture's intentional test failure. A compiler/dependency failure does not pass. |
| `signals` | The returned signal catalogue includes the reference and DUT count signals. |
| `first_mismatch` | First mismatch at cycle **8**: reference **8**, DUT **7**. |
| `value_at` | Count values agree with the mismatch result; report the actual enable value. |
| `value_range` | DUT count stalls at **7** within the requested interval. |
| `show_wave` | A savefile is returned with `launched: false`. Opening a GUI is not part of this check. |
| `synth` | Utilization is returned in estimate mode; timing is unavailable. |

All eight native calls must appear in the transcript. Check tool responses
against this table rather than relying only on the agent's summary. Missing
backends, model authentication errors, or dependency failures must be recorded.
Do not count skipped or blocked checks as passed.

## 4. Keep the evidence

Ask pi:

```text
Summarize this run in a table: tool, passed/failed/not verified, actual result,
and run/wave handle where available. Separate the expected counter test failure
from infrastructure errors. State the installed package version, model, and any
checks you could not complete. Do not present expected values as measured results.
```

Keep the transcript and `.ic/` run artifacts. For detailed reproduction and
historical verification, see [pi acceptance](../../../../docs/goals/ic-design-tools/acceptance/pi-agent.md).
The fixture and acceptance files require the repository checkout; the npm package
alone does not include the repository's RTL and test fixtures.
