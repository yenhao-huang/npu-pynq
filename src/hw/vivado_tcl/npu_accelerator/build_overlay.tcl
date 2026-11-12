# Recreate and optionally implement the PYNQ-Z1 NPU matrix overlay.
# Usage: vivado -mode batch -source build_overlay.tcl -tclargs
#        ?--array-size 2|8|16? ?--scalar-impl CUSTOM|CVA6? ?--elaborate-only?
#        ?--allow-dirty? ?--out <dir>?
#
# --scalar-impl CVA6 (default CUSTOM) builds the CVA6 RISC-V scalar core: it
# reads src/hw/rtl/npu_accelerator/scalar_core/cva6/cva6.f (needs the CVA6
# submodule), defines NPU_SCALAR_CVA6 and sets SCALAR_IMPL. That path has not
# been run in Vivado yet, and the npu_pynq overlay verifier does not describe a
# CVA6 overlay, so it requires --elaborate-only or --allow-dirty.
#
# --out also copies the verified BIT, HWH and manifest to <dir>,
# for example dist/smollm/overlay.

set script_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set array_size 16
set elaborate_only 0
set allow_dirty 0
set out_dir ""
set scalar_impl CUSTOM
for {set argument_index 0} {$argument_index < [llength $argv]} {incr argument_index} {
    set argument [lindex $argv $argument_index]
    switch -- $argument {
        --array-size {
            incr argument_index
            if {$argument_index >= [llength $argv]} {
                error "--array-size requires 2, 8, or 16"
            }
            set array_size [lindex $argv $argument_index]
            if {$array_size ni {2 8 16}} {
                error "unsupported array size '$array_size': expected 2, 8, or 16"
            }
        }
        --scalar-impl {
            incr argument_index
            if {$argument_index >= [llength $argv]} {
                error "--scalar-impl requires CUSTOM or CVA6"
            }
            set scalar_impl [lindex $argv $argument_index]
            if {$scalar_impl ni {CUSTOM CVA6}} {
                error "unsupported scalar core '$scalar_impl': expected CUSTOM or CVA6"
            }
        }
        --elaborate-only { set elaborate_only 1 }
        --allow-dirty { set allow_dirty 1 }
        --out {
            incr argument_index
            if {$argument_index >= [llength $argv]} {
                error "--out requires a directory"
            }
            set out_dir [file normalize [lindex $argv $argument_index]]
        }
        default { error "unknown argument '$argument'" }
    }
}

if {$scalar_impl eq "CVA6" && !$elaborate_only && !$allow_dirty} {
    error "--scalar-impl CVA6 is exploratory: pass --elaborate-only or --allow-dirty"
}

set build_name npu_accelerator
if {$array_size != 2} {
    set build_name "npu_accelerator_${array_size}x${array_size}"
}
if {$scalar_impl eq "CVA6"} {
    append build_name _cva6
}
set build_root [file normalize [file join $repo_root build vivado $build_name]]
set project_dir [file join $build_root project]
set report_dir [file join $build_root reports]
set artifact_dir [file join $build_root artifacts]
set jobs 4

# The manifest records HEAD as the overlay's source commit, so src/ must match
# it. Changes elsewhere (documentation, examples) do not reach the bitstream
# or its verifier and are allowed.
if {!$elaborate_only && !$allow_dirty} {
    if {[catch {exec git -C $repo_root status --porcelain --untracked-files=normal -- src} source_status]} {
        error "cannot verify source tree cleanliness: $source_status"
    }
    if {[string trim $source_status] ne ""} {
        error "refusing production artifact build: src/ has uncommitted changes:\n$source_status"
    }
}

file delete -force $build_root
file mkdir $project_dir
file mkdir $report_dir
file mkdir $artifact_dir

create_project npu_accelerator $project_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

set rtl_files [list \
    [file join $repo_root src hw rtl npu_accelerator matrix_core systolic_array pe.sv] \
    [file join $repo_root src hw rtl npu_accelerator matrix_core systolic_array systolic_array.sv] \
    [file join $repo_root src hw rtl npu_accelerator csr csr.sv] \
    [file join $repo_root src hw rtl npu_accelerator matrix_core memory operand_buffer.sv] \
    [file join $repo_root src hw rtl npu_accelerator matrix_core controller matrix_core_controller.sv] \
    [file join $repo_root src hw rtl npu_accelerator matrix_core matrix_core.sv] \
    [file join $repo_root src hw rtl npu_accelerator axis_skid.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core custom custom_sync_fifo.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core custom custom_fetch.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core custom custom_decode.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core custom custom_dma.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core custom custom_controller.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core custom custom_scalar_core.sv] \
    [file join $repo_root src hw rtl npu_accelerator scalar_core scalar_core.sv] \
    [file join $repo_root src hw rtl npu_accelerator npu_accelerator.sv]]
foreach rtl_file $rtl_files {
    if {![file isfile $rtl_file]} {
        error "missing source-controlled RTL: $rtl_file"
    }
}
add_files -norecurse $rtl_files
set_property file_type {SystemVerilog} [get_files $rtl_files]

# CVA6 scalar core: the file list, include directories and define of cva6.f
# (paths relative to it, "//" comments), plus our two plain cva6_* files that
# cva6.f leaves to the default list.
if {$scalar_impl eq "CVA6"} {
    set cva6_dir [file join $repo_root src hw rtl npu_accelerator scalar_core cva6]
    set cva6_files [list [file join $cva6_dir cva6_dma.sv] [file join $cva6_dir cva6_npu_mmio.sv]]
    set cva6_incdirs {}
    set cva6_defines {}
    set flist [open [file join $cva6_dir cva6.f] r]
    foreach line [split [read $flist] "\n"] {
        set line [string trim [regsub {//.*$} $line ""]]
        if {$line eq ""} { continue }
        if {[string match "+define+*" $line]} {
            lappend cva6_defines [string range $line 8 end]
        } elseif {[string match "+incdir+*" $line]} {
            lappend cva6_incdirs [file normalize [file join $cva6_dir [string range $line 8 end]]]
        } else {
            lappend cva6_files [file normalize [file join $cva6_dir $line]]
        }
    }
    close $flist
    foreach cva6_file $cva6_files {
        if {![file isfile $cva6_file]} {
            error "missing CVA6 source (is the submodule initialized?): $cva6_file"
        }
    }
    add_files -norecurse $cva6_files
    set_property file_type {SystemVerilog} [get_files $cva6_files]
    set_property include_dirs $cva6_incdirs [current_fileset]
    set_property verilog_define $cva6_defines [current_fileset]
}
update_compile_order -fileset sources_1

create_bd_design npu_accelerator_bd
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* ps7]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000}] $ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
    $ps7

set reset0 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset0]
set reset_inverter [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* reset_inverter]
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $reset_inverter
set control_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* control_ic]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $control_ic
set memory_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* memory_ic]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $memory_ic
set accelerator [create_bd_cell -type module -reference npu_accelerator npu_accelerator_0]
set_property -dict [list \
    CONFIG.ROWS $array_size \
    CONFIG.COLUMNS $array_size \
    CONFIG.MAX_K {256} \
    CONFIG.IN_BYTES {8}] $accelerator
if {$scalar_impl eq "CVA6"} {
    set_property CONFIG.SCALAR_IMPL {"CVA6"} $accelerator
}

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_ic/M00_AXI] [get_bd_intf_pins npu_accelerator_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins npu_accelerator_0/m_axi] [get_bd_intf_pins memory_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_ic/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]

connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
    [get_bd_pins reset0/slowest_sync_clk] \
    [get_bd_pins control_ic/ACLK] [get_bd_pins control_ic/S00_ACLK] \
    [get_bd_pins control_ic/M00_ACLK] \
    [get_bd_pins memory_ic/ACLK] [get_bd_pins memory_ic/S00_ACLK] \
    [get_bd_pins memory_ic/M00_ACLK] \
    [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
    [get_bd_pins npu_accelerator_0/s_axi_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins reset_inverter/Op1]
connect_bd_net [get_bd_pins reset_inverter/Res] [get_bd_pins reset0/ext_reset_in]
connect_bd_net [get_bd_pins reset0/interconnect_aresetn] \
    [get_bd_pins control_ic/ARESETN] [get_bd_pins control_ic/S00_ARESETN] \
    [get_bd_pins control_ic/M00_ARESETN] \
    [get_bd_pins memory_ic/ARESETN] [get_bd_pins memory_ic/S00_ARESETN] \
    [get_bd_pins memory_ic/M00_ARESETN]
connect_bd_net [get_bd_pins reset0/peripheral_aresetn] \
    [get_bd_pins npu_accelerator_0/s_axi_aresetn]
connect_bd_net [get_bd_pins npu_accelerator_0/irq] [get_bd_pins ps7/IRQ_F2P]

create_bd_addr_seg -range 0x00010000 -offset 0x43C00000 \
    [get_bd_addr_spaces ps7/Data] \
    [get_bd_addr_segs npu_accelerator_0/s_axi/reg0] SEG_npu_accelerator
assign_bd_address

validate_bd_design
save_bd_design
set bd_file [get_files [file join $project_dir npu_accelerator.srcs sources_1 bd npu_accelerator_bd npu_accelerator_bd.bd]]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrapper_file [file join $project_dir npu_accelerator.gen sources_1 bd npu_accelerator_bd hdl npu_accelerator_bd_wrapper.v]
add_files -norecurse $wrapper_file
set_property top npu_accelerator_bd_wrapper [current_fileset]
update_compile_order -fileset sources_1

set address_file [open [file join $report_dir address_map.txt] w]
puts $address_file "accelerator=0x43C00000 range=0x00010000"
foreach segment [get_bd_addr_segs] {
    puts $address_file "[get_property NAME $segment] offset=[get_property OFFSET $segment] range=[get_property RANGE $segment]"
}
close $address_file

if {$elaborate_only} {
    puts "PASS: npu_accelerator ${array_size}x${array_size} Vivado block-design elaboration"
    close_project
    exit 0
}

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "synthesis did not complete"
}
open_run synth_1
report_utilization -file [file join $report_dir utilization_synth.rpt]
close_design

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "implementation or write_bitstream did not complete"
}
open_run impl_1
report_utilization -file [file join $report_dir utilization_impl.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -max_paths 10 -file [file join $report_dir timing_summary_routed.rpt]
report_drc -file [file join $report_dir drc_routed.rpt]
set route_report [file join $report_dir route_status.rpt]
report_route_status -file $route_report

set route_handle [open $route_report r]
set route_text [read $route_handle]
close $route_handle
if {![regexp {# of routable nets\.*[[:space:]]*:[[:space:]]*([0-9]+)} \
      $route_text -> routable_nets] ||
    ![regexp {# of fully routed nets\.*[[:space:]]*:[[:space:]]*([0-9]+)} \
      $route_text -> fully_routed_nets] ||
    ![regexp {# of nets with routing errors\.*[[:space:]]*:[[:space:]]*([0-9]+)} \
      $route_text -> routing_errors]} {
    error "route-status report format was not recognized"
}
if {$routable_nets != $fully_routed_nets || $routing_errors != 0} {
    error "route-status gate failed: routable=$routable_nets fully_routed=$fully_routed_nets errors=$routing_errors"
}

set worst_path [get_timing_paths -setup -max_paths 1]
if {[llength $worst_path] == 0} {
    error "no routed setup timing path was found"
}
set wns [get_property SLACK $worst_path]
set failing_paths [get_timing_paths -setup -slack_lesser_than 0 -max_paths 100000]
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
if {$wns < 0.0 || [llength $failing_paths] != 0} {
    error "routed timing failed: WNS=$wns failing_setup_paths=[llength $failing_paths]"
}
if {[llength $drc_errors] != 0} {
    error "routed DRC has [llength $drc_errors] error violations"
}

set bit_source [file join $project_dir npu_accelerator.runs impl_1 npu_accelerator_bd_wrapper.bit]
set hwh_source [file join $project_dir npu_accelerator.gen sources_1 bd npu_accelerator_bd hw_handoff npu_accelerator_bd.hwh]
if {![file isfile $bit_source] || ![file isfile $hwh_source]} {
    error "same-build BIT or HWH artifact is missing"
}
if {$allow_dirty} {
    puts "PASS: npu_accelerator dirty exploratory implementation; artifacts not published"
    close_project
    exit 0
}
file copy -force $bit_source [file join $artifact_dir npu_accelerator.bit]
file copy -force $hwh_source [file join $artifact_dir npu_accelerator.hwh]

set source_commit unknown
catch {set source_commit [string trim [exec git -C $repo_root rev-parse HEAD]]}
set verifier [file join $repo_root src vendors npu_pynq driver overlay build.py]
if {[catch {exec python $verifier \
    --source-commit $source_commit --vivado-version [version -short] \
    --array-size $array_size $artifact_dir} verify_output]} {
    error "overlay provenance verification failed: $verify_output"
}
puts $verify_output

if {$out_dir ne ""} {
    file mkdir $out_dir
    foreach artifact [glob -directory $artifact_dir *] {
        file copy -force $artifact $out_dir
    }
    puts "Copied the overlay to $out_dir"
}

set evidence [open [file join $report_dir build_evidence.txt] w]
puts $evidence "vivado=[version -short]"
puts $evidence "part=xc7z020clg400-1"
puts $evidence "rows=$array_size"
puts $evidence "columns=$array_size"
puts $evidence "wns=$wns"
puts $evidence "setup_failing_paths=[llength $failing_paths]"
puts $evidence "drc_errors=[llength $drc_errors]"
puts $evidence "bit=[file join $artifact_dir npu_accelerator.bit]"
puts $evidence "hwh=[file join $artifact_dir npu_accelerator.hwh]"
puts $evidence "source_commit=$source_commit"
close $evidence
puts "PASS: npu_accelerator ${array_size}x${array_size} Vivado implementation and bitstream"
close_project
exit 0
