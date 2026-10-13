# Out-of-context PPA run for one RTL module on the PYNQ-Z1 part.
#
# Usage: vivado -mode batch -source ppa_module.tcl -tclargs \
#          --top <module> ?--name <label>? ?--generic NAME:VALUE ...? \
#          ?--clock-port clk? ?--clock-ns 10.0? ?--out <dir>?
#
# Synthesizes <module> out of context from every .sv under src/hw/rtl, then
# places and routes it against a single clock constraint and writes:
#   <out>/<label>/utilization.rpt        report_utilization (hierarchical too)
#   <out>/<label>/timing_summary.rpt     report_timing_summary, routed
#   <out>/<label>/power.rpt              report_power, vectorless defaults
#   <out>/<label>/summary.json           the numbers the PPA tables quote
# Generics are written NAME:VALUE (NAME=VALUE also works on Linux): the Windows
# vivado.bat launcher splits arguments at '='.
# Fmax is 1000 / (clock-ns - WNS). Power is vectorless (default 12.5 % toggle
# rate on inputs), so it compares designs; it does not predict the board.

set script_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set top ""
set name ""
set generics [list]
set clock_port clk
set clock_ns 10.0
set out_root [file join $repo_root build ppa]

for {set i 0} {$i < [llength $argv]} {incr i} {
    set argument [lindex $argv $i]
    switch -- $argument {
        --top        { incr i; set top [lindex $argv $i] }
        --name       { incr i; set name [lindex $argv $i] }
        --generic    {
            incr i
            set g [lindex $argv $i]
            if {[string first = $g] < 0} { set g [regsub {:} $g =] }
            lappend generics $g
        }
        --clock-port { incr i; set clock_port [lindex $argv $i] }
        --clock-ns   { incr i; set clock_ns [lindex $argv $i] }
        --out        { incr i; set out_root [file normalize [lindex $argv $i]] }
        default      { error "unknown argument '$argument'" }
    }
}
if {$top eq ""} { error "--top is required" }
if {$name eq ""} { set name $top }

proc collect_sv {dir} {
    set found [lsort [glob -nocomplain -type f -directory $dir *.sv]]
    foreach sub [lsort [glob -nocomplain -type d -directory $dir *]] {
        set found [concat $found [collect_sv $sub]]
    }
    return $found
}

set out_dir [file join $out_root $name]
file delete -force $out_dir
file mkdir $out_dir

set rtl_files [collect_sv [file join $repo_root src hw rtl]]
if {[llength $rtl_files] == 0} { error "no RTL found under src/hw/rtl" }
read_verilog -sv $rtl_files

set generic_args [list]
foreach g $generics { lappend generic_args -generic $g }
synth_design -top $top -part xc7z020clg400-1 -mode out_of_context \
    -flatten_hierarchy rebuilt {*}$generic_args

set clock_ports [get_ports -quiet $clock_port]
if {[llength $clock_ports] == 0} { error "clock port '$clock_port' not found on $top" }
create_clock -name clk -period $clock_ns $clock_ports
# OOC: no I/O buffers; constrain the boundary like a register-to-register
# neighbour so only the module's own logic sets Fmax.
set data_inputs [lsearch -all -inline -not -exact [all_inputs] $clock_ports]
if {[llength $data_inputs] > 0} { set_input_delay -clock clk 0.0 $data_inputs }
if {[llength [all_outputs]] > 0} { set_output_delay -clock clk 0.0 [all_outputs] }

opt_design
place_design
phys_opt_design
route_design

report_utilization -file [file join $out_dir utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $out_dir utilization_hier.rpt]
report_timing_summary -max_paths 5 -file [file join $out_dir timing_summary.rpt]
report_power -file [file join $out_dir power.rpt]

# Read the counts back from the utilization report, so the JSON quotes the
# same numbers a person reading utilization.rpt sees.
set util [report_utilization -return_string]
proc util_used {text label} {
    if {[regexp -line "^\\|\\s*[string map {( \( ) \) * \*} $label]\\s*\\|\\s*(\[0-9.\]+)" $text -> used]} {
        return $used
    }
    return 0
}
set luts [util_used $util {Slice LUTs}]
set lutram [util_used $util {LUT as Memory}]
set ffs [util_used $util {Slice Registers}]
set dsps [util_used $util {DSPs}]
set bram_tiles [util_used $util {Block RAM Tile}]
set ramb36 [util_used $util {RAMB36/FIFO*}]
set ramb18 [util_used $util {RAMB18}]
set carry [llength [get_cells -quiet -hierarchical -filter {REF_NAME == CARRY4}]]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
if {$wns eq ""} { set wns 0.0 }
set fmax [format %.1f [expr {1000.0 / ($clock_ns - $wns)}]]
set power_total [get_property TOTAL_POWER [current_design]]
set power_dynamic [get_property DYNAMIC_POWER [current_design]]
set power_static [get_property STATIC_POWER [current_design]]
if {$power_total eq ""} {
    # Older Vivado: read the numbers back from the report.
    set fh [open [file join $out_dir power.rpt] r]; set text [read $fh]; close $fh
    regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $text -> power_total
    regexp {Dynamic \(W\)\s*\|\s*([0-9.]+)} $text -> power_dynamic
    regexp {Device Static \(W\)\s*\|\s*([0-9.]+)} $text -> power_static
}

set fh [open [file join $out_dir summary.json] w]
puts $fh "{"
puts $fh "  \"name\": \"$name\", \"top\": \"$top\", \"generics\": \"[join $generics { }]\","
puts $fh "  \"vivado\": \"[version -short]\", \"part\": \"xc7z020clg400-1\", \"clock_ns\": $clock_ns,"
puts $fh "  \"lut\": $luts, \"lutram\": $lutram, \"ff\": $ffs, \"dsp\": $dsps,"
puts $fh "  \"bram_tiles\": $bram_tiles, \"ramb36\": $ramb36, \"ramb18\": $ramb18, \"carry4\": $carry,"
puts $fh "  \"wns_ns\": $wns, \"fmax_mhz\": $fmax,"
puts $fh "  \"power_total_w\": $power_total, \"power_dynamic_w\": $power_dynamic, \"power_static_w\": $power_static"
puts $fh "}"
close $fh
puts "PPA $name: lut=$luts ff=$ffs dsp=$dsps ramb36=$ramb36 ramb18=$ramb18 wns=$wns fmax=$fmax dyn=$power_dynamic"
