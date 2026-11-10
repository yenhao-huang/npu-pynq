set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set build_root [file join $repo_root build vivado pe_dsp_check]
set part_name xc7z020clg400-1

file delete -force $build_root
file mkdir $build_root

create_project pe_dsp_check $build_root -part $part_name -force
read_verilog -sv [file join $repo_root src hw rtl npu_accelerator matrix_core systolic_array pe.sv]
synth_design -top pe -mode out_of_context -part $part_name

set dsp_cells [get_cells -hier -filter {REF_NAME =~ DSP48*}]
report_utilization -file [file join $build_root utilization.rpt]

if {[llength $dsp_cells] != 1} {
    error "Expected exactly one DSP48 cell for pe, found [llength $dsp_cells]"
}

puts "PASS: pe inferred exactly one DSP48 cell"
close_project
