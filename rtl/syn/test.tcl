set_host_options -max_cores 16

# Set design variables
set design_name "mulf_bf16_bf16"
set search_path [list /home/tools/freepdk-45nm $search_path]
set target_library "/home/tools/freepdk-45nm/stdcells.db"
set link_library "* $target_library"
set report_path "./report"
set syn_path "./syn"
set tmp_path "./tmp"
set rtl_src [glob ../*.sv]

cd $syn_path

if { ! [file exists $report_path] } {
    file mkdir $report_path
}

analyze -format sverilog $rtl_src


elaborate $design_name

current_design $design_name

set_ideal_network -no_propagate [get_nets -hierarchical *clk*]
set_dont_touch_network [get_nets -hierarchical *clk*]

create_clock -period 1.0 -name clk [get_ports clk]
set_input_delay 0 -clock clk [all_inputs]
set_output_delay 0 -clock clk [all_outputs]

# check_design > "$report_path/check_design.rpt"


compile_ultra

write -hierarchy -format ddc -output "$syn_path/$design_name.ddc"
write -hierarchy -format verilog -output "$syn_path/$design_name.v"
write_sdc "$syn_path/$design_name.sdc"

report_timing > "$report_path/timing_report.rpt"
report_hierarchy -timing > "$report_path/hierarchy.rpt"
report_area >  "$report_path/area_report.rpt"
report_power >  "$report_path/power_report.rpt"
