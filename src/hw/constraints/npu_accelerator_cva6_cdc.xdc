# Clock domain crossing of the CVA6 scalar core (--scalar-impl CVA6 only).
#
# clk_fpga_0 (FCLK_CLK0, 100 MHz) clocks the NPU; clk_fpga_1 (FCLK_CLK1,
# 50 MHz) clocks the cva6 core alone. Every path between them is one of:
#   - the gray-code pointers and the FIFO data of axi_cdc (pulp cdc_fifo_gray),
#   - cpu_rst_n into the scalar_clk reset synchronizer,
#   - boot_addr, which only changes while the core is held in reset.
# cdc_fifo_gray asks for the data and pointer delay to be bounded by the
# smaller of the two periods and for no hold check; -datapath_only does both.
# Read late, after the PS7 constraints have created both clocks.
set_max_delay -datapath_only -from [get_clocks clk_fpga_0] -to [get_clocks clk_fpga_1] 10.000
set_max_delay -datapath_only -from [get_clocks clk_fpga_1] -to [get_clocks clk_fpga_0] 10.000
