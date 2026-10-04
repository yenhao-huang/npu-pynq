"""Fast checks for everything that does not need OpenROAD.

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import paths
from llm import available_backends, describe_backends, get_backend, register
from llm.base import LLMBackend, clean_vh
from llm.cli import ClaudeCLIBackend, CodexCLIBackend
from prompt_formatter import DesignSpec, format_prompt, parse_user_prompt
from retrieval import RTLLibrary, collect_defines, retrieve
from tpugen_types import PPA, FlowError, PPATarget
from validate import check_macros, summarize_elaboration
from vh_template import render_reference_vh


class TestPPAContract(unittest.TestCase):
    def test_only_openroad_may_produce_ppa(self):
        PPA(1.0, -0.5, 0.01, "openroad")
        for bogus in ("yosys", "estimate", ""):
            with self.assertRaises(FlowError):
                PPA(1.0, -0.5, 0.01, bogus)

    def test_distance_is_relative(self):
        ppa = PPA(area=200, wns=-2, total_power=0.02, source="openroad")
        d = ppa.distance(PPATarget(area=100, wns=-1, total_power=0.01))
        self.assertEqual(d, {"area": 1.0, "wns": 1.0, "total_power": 1.0})


class TestPromptFormatter(unittest.TestCase):
    def test_parses_size_multiplier_adder_and_width(self):
        spec = parse_user_prompt(
            "4x4 systolic array, INT8, BAM multiplier, LZTA adder, coefficient of 3"
        )
        self.assertEqual((spec.m, spec.n), (4, 4))
        self.assertEqual(spec.multiplier, "BAM")
        self.assertEqual(spec.adder, "LZTA")
        self.assertEqual((spec.dw, spec.ww), (8, 8))
        self.assertEqual(spec.mult_dw, 3)

    def test_longer_selector_name_wins(self):
        spec = parse_user_prompt("8x8 TPU with an ALM_LOA multiplier and LOA adder")
        self.assertEqual(spec.multiplier, "ALM_LOA")

    def test_rejects_impossible_coefficient(self):
        with self.assertRaises(FlowError):
            DesignSpec(dw=4, ww=4, mult_dw=8).validate()

    def test_feedback_is_appended_on_later_rounds(self):
        spec = DesignSpec()
        target = PPATarget(100, -1, 0.01)
        plain = format_prompt(spec, target)
        with_fb = format_prompt(
            spec, target,
            previous_ppa=PPA(200, -2, 0.02, "openroad"),
            errors=["module LZTA not found"],
        )
        self.assertNotIn("previous attempt", plain)
        self.assertIn("module LZTA not found", with_fb)


class TestPreprocessor(unittest.TestCase):
    def test_implied_macros_reach_fixed_point(self):
        defines = collect_defines(paths.OPTIONS_VH.read_text())
        self.assertIn("ALM", defines)
        self.assertIn("SHARED_PRE_APPROX", defines)  # implied by ALM
        self.assertEqual(defines["M"], "16")

    def test_elsif_branches_are_exclusive(self):
        text = (
            "`define A\n"
            "`ifdef A\n`define TOOK_A\n"
            "`elsif B\n`define TOOK_B\n"
            "`else\n`define TOOK_ELSE\n`endif\n"
        )
        defines = collect_defines(text)
        self.assertIn("TOOK_A", defines)
        self.assertNotIn("TOOK_B", defines)
        self.assertNotIn("TOOK_ELSE", defines)

    def test_unterminated_ifdef_is_an_error(self):
        with self.assertRaises(FlowError):
            collect_defines("`ifdef A\n`define X\n")


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths.check()
        cls.library = RTLLibrary.load(paths.RTL_DIR)

    def test_library_loads_every_module(self):
        self.assertGreater(len(self.library.module_file), 90)

    def test_closure_is_a_strict_subset_of_the_library(self):
        vh = render_reference_vh(
            DesignSpec(m=2, n=2, multiplier="DRUM_APTPU", adder="APPROX5",
                       dw=8, ww=8, mult_dw=4)
        )
        rag = retrieve(vh, self.library, paths.TOP_MODULE, mode="rag")
        full = retrieve(vh, self.library, paths.TOP_MODULE, mode="full")
        self.assertEqual(rag.missing, [])
        self.assertIn(paths.RTL_DIR / "systolic_array_top.v", rag.filelist)
        self.assertLess(len(rag.filelist), len(full.filelist))
        self.assertTrue(set(rag.filelist) <= set(full.filelist))

    def test_the_selected_multiplier_decides_what_is_pulled_in(self):
        base = DesignSpec(m=2, n=2, adder="APPROX5", dw=8, ww=8, mult_dw=4)
        drum = retrieve(
            render_reference_vh(DesignSpec(**{**vars(base), "multiplier": "DRUM_APTPU"})),
            self.library, paths.TOP_MODULE,
        )
        bam = retrieve(
            render_reference_vh(DesignSpec(**{**vars(base), "multiplier": "BAM"})),
            self.library, paths.TOP_MODULE,
        )
        drum_names = {p.name for p in drum.filelist}
        bam_names = {p.name for p in bam.filelist}
        self.assertIn("BAM_top.v", bam_names)
        self.assertNotIn("BAM_top.v", drum_names)

    def test_numeric_macros_do_not_pull_files(self):
        """The upstream matcher pulled files whose name merely contained DW/M/N."""
        vh = render_reference_vh(
            DesignSpec(m=4, n=4, multiplier="BAM", adder="LZTA",
                       dw=8, ww=8, mult_dw=4)
        )
        names = {p.name for p in retrieve(vh, self.library, paths.TOP_MODULE).filelist}
        self.assertNotIn("Dynamic_truncation.v", names)
        self.assertNotIn("EIM16x16.v", names)


class TestCleanVH(unittest.TestCase):
    def test_strips_fences_and_dataset_marker(self):
        out = clean_vh("{.vh}\n```verilog\nchatter\n`define DW 8\n```\n")
        self.assertTrue(out.startswith("`define DW 8"))

    def test_rejects_output_without_macros(self):
        with self.assertRaises(FlowError):
            clean_vh("I am afraid I cannot do that.")


class TestMacroCheck(unittest.TestCase):
    def test_reference_header_is_complete(self):
        vh = render_reference_vh(
            DesignSpec(m=4, n=4, multiplier="BAM", adder="LZTA",
                       dw=8, ww=8, mult_dw=4)
        )
        self.assertEqual(check_macros(vh), [])

    def test_reports_every_missing_piece(self):
        problems = check_macros("`define DW 8\n`define WW 8\n`define MULT_DW 16\n")
        joined = " ".join(problems)
        self.assertIn("missing `define M", problems)
        self.assertIn("no multiplier selected", joined)
        self.assertIn("no adder selected", joined)
        self.assertIn("MULT_DW (16) exceeds", joined)

    def test_unpreprocessable_header_is_reported_not_raised(self):
        problems = check_macros("`ifdef A\n`define DW 8\n")
        self.assertTrue(problems[0].startswith("header does not preprocess"))

    def test_shared_pre_approx_is_required_for_drum_data_path(self):
        vh = render_reference_vh(DesignSpec())
        missing_shared = vh.replace("    `define SHARED_PRE_APPROX\n", "")
        self.assertIn(
            "selected multiplier requires `define SHARED_PRE_APPROX",
            check_macros(missing_shared),
        )


class TestValidate(unittest.TestCase):
    def test_keeps_errors_drops_warnings(self):
        log = (
            "/x/pe.v:12: error: Unknown module type: lzta_adder\n"
            "/x/pe.v:57: warning: Port 2 expects 8 bits\n"
        )
        self.assertEqual(
            summarize_elaboration(log),
            ["pe.v:12: Unknown module type: lzta_adder"],
        )


class TestBackendRegistry(unittest.TestCase):
    def test_claude_and_codex_are_registered(self):
        for name in ("claude", "codex"):
            self.assertIn(name, available_backends())
            self.assertIn(name, describe_backends())

    def test_unknown_backend_names_the_available_ones(self):
        with self.assertRaises(FlowError) as ctx:
            get_backend("gpt5-by-vibes")
        self.assertIn("claude", str(ctx.exception))

    def test_register_adds_a_backend(self):
        class Stub(LLMBackend):
            name = "stub"

            def generate(self, prompt: str) -> str:
                return "`define DW 8\n"

        try:
            register("stub", lambda **kw: Stub(), "test double")
            self.assertIsInstance(get_backend("stub"), Stub)
        finally:
            from llm import registry

            registry._BACKENDS.pop("stub", None)
            registry._DESCRIPTIONS.pop("stub", None)

    def test_missing_cli_is_a_flow_error_not_a_crash(self):
        class Missing(ClaudeCLIBackend):
            executable = "definitely-not-on-path-xyzzy"

        with self.assertRaises(FlowError):
            Missing()

    def test_cli_commands_run_the_agent_without_tools(self):
        claude = ClaudeCLIBackend.__new__(ClaudeCLIBackend)
        claude.path, claude.model, claude.timeout = "/bin/claude", "m", 60
        cmd = claude._command("PROMPT", Path("/tmp/reply.txt"))
        self.assertIn("-p", cmd)
        self.assertIn("PROMPT", cmd)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "plan")

        codex = CodexCLIBackend.__new__(CodexCLIBackend)
        codex.path, codex.model, codex.timeout = "/bin/codex", None, 60
        out_file = Path("/tmp/reply.txt")
        cmd = codex._command("PROMPT", out_file)
        self.assertEqual(cmd[1], "exec")
        self.assertEqual(cmd[cmd.index("--sandbox") + 1], "read-only")
        self.assertEqual(cmd[cmd.index("--output-last-message") + 1],
                         str(out_file))
        self.assertTrue(cmd[-1].endswith("PROMPT"))

    def test_codex_prefers_the_last_message_file_over_the_session_log(self):
        codex = CodexCLIBackend.__new__(CodexCLIBackend)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reply.txt"
            self.assertEqual(codex._read_reply("session log", out), "session log")
            out.write_text("`define DW 8\n")
            self.assertEqual(codex._read_reply("session log", out),
                             "`define DW 8\n")


if __name__ == "__main__":
    unittest.main()
