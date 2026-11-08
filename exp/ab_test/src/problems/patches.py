"""Known dataset corrections used only by the grader."""
REF_TOP_RENAME = {"adder_pipe_64bit": ("adder_64bit", "adder_pipe_64bit"),
                  "multi_pipe_4bit": ("multi_pipe", "multi_pipe_4bit")}

TB_PATCHES = {
    # An output cannot drive a reg; and sampling on the toggle instant is a race,
    # so sample one time unit before each edge (expected values unchanged).
    "clkgenerator": [("reg clk_tb;", "wire clk_tb;"),
                     ("#5; // Time delay between clock cycles", "#4; // sample before the edge"),
                     ("res = res + 1;", "res = res + 1; #1;")],
    # SV array literal for iverilog; `i == 9` races the always block that increments i.
    "ring_counter": [("[0:9] = {", "[0:9] = '{"), ("if (i == 9)", "if (i >= 9)")],
}

HELDOUT_SKIP = {"radix2_div"}
