set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir "../../.."]]
set build_dir [file normalize [file join $repo_root "vivado_projects/mac_npu"]]
set overlay_dir [file normalize [file join $repo_root "mount/mac_npu/overlay"]]

file mkdir $build_dir
file mkdir $overlay_dir

create_project -force mac_npu $build_dir -part xc7z020clg400-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

add_files -norecurse [list \
    [file join $repo_root "src/rtl/mac_npu/mac_unit.sv"] \
    [file join $repo_root "src/rtl/mac_npu/mac_axi_lite.sv"]]
update_compile_order -fileset sources_1

create_bd_design mac_npu_bd
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {apply_board_preset "0" make_external "FIXED_IO, DDR"} \
    [get_bd_cells processing_system7_0]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1}] \
    [get_bd_cells processing_system7_0]

create_bd_cell -type module -reference mac_axi_lite mac_axi_lite_0

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/processing_system7_0/M_AXI_GP0" Clk "Auto"} \
    [get_bd_intf_pins mac_axi_lite_0/s_axi]

assign_bd_address -offset 0x43C00000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    [get_bd_addr_segs mac_axi_lite_0/s_axi/reg0] -force

validate_bd_design
save_bd_design

set bd_file [get_files mac_npu_bd.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrapper_file [file join $build_dir "mac_npu.gen/sources_1/bd/mac_npu_bd/hdl/mac_npu_bd_wrapper.v"]
add_files -norecurse $wrapper_file
set_property top mac_npu_bd_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation did not reach 100%"
}

set bit_source [file join $build_dir "mac_npu.runs/impl_1/mac_npu_bd_wrapper.bit"]
set hwh_source [file join $build_dir "mac_npu.gen/sources_1/bd/mac_npu_bd/hw_handoff/mac_npu_bd.hwh"]
if {![file exists $bit_source]} { error "Missing bitstream: $bit_source" }
if {![file exists $hwh_source]} { error "Missing HWH: $hwh_source" }

file copy -force $bit_source [file join $overlay_dir "mac_npu.bit"]
file copy -force $hwh_source [file join $overlay_dir "mac_npu.hwh"]
puts "OVERLAY_READY [file join $overlay_dir mac_npu.bit]"
