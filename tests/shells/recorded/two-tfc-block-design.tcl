create_bd_cell -type ip -vlnv $smartconnect_vlnv control_0_0
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 4 CONFIG.STRATEGY LOW_AREA] [get_bd_cells control_0_0]
connect_bd_intf_net [get_bd_intf_pins control_0_0/S00_AXI] [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD]
connect_bd_net [get_bd_pins control_0_0/aclk] [get_bd_pins zynq_ps/pl_clk0]
connect_bd_net [get_bd_pins control_0_0/aresetn] [get_bd_pins rst_ps/peripheral_aresetn]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "/in0"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv v:in0 in0
connect_bd_intf_net [get_bd_intf_pins in0/m_axi_gmem] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins in0/s_axi_control] [get_bd_intf_pins control_0_0/M00_AXI]
assign_bd_address [get_bd_addr_segs in0/s_axi_control/Reg*] -offset 0xa0000000 -range 4096
connect_bd_net [get_bd_pins in0/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins in0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "/DataflowCore_0"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv v:DataflowCore_0 DataflowCore_0
connect_bd_net [get_bd_pins DataflowCore_0/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins DataflowCore_0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins DataflowCore_0/s_axis_0] [get_bd_intf_pins in0/out0_V]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "/out0"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv v:out0 out0
connect_bd_intf_net [get_bd_intf_pins out0/m_axi_gmem] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins out0/s_axi_control] [get_bd_intf_pins control_0_0/M01_AXI]
assign_bd_address [get_bd_addr_segs out0/s_axi_control/Reg*] -offset 0xa0001000 -range 4096
connect_bd_net [get_bd_pins out0/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins out0/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins out0/in0_V] [get_bd_intf_pins DataflowCore_0/m_axis_0]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "/in1"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv v:in1 in1
connect_bd_intf_net [get_bd_intf_pins in1/m_axi_gmem] [get_bd_intf_pins smartconnect_0/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins in1/s_axi_control] [get_bd_intf_pins control_0_0/M02_AXI]
assign_bd_address [get_bd_addr_segs in1/s_axi_control/Reg*] -offset 0xa0002000 -range 4096
connect_bd_net [get_bd_pins in1/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins in1/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "/DataflowCore_1"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv v:DataflowCore_1 DataflowCore_1
connect_bd_net [get_bd_pins DataflowCore_1/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins DataflowCore_1/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins DataflowCore_1/s_axis_0] [get_bd_intf_pins in1/out0_V]
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] [list "/out1"]] [current_project]
update_ip_catalog -rebuild -scan_changes
create_bd_cell -type ip -vlnv v:out1 out1
connect_bd_intf_net [get_bd_intf_pins out1/m_axi_gmem] [get_bd_intf_pins smartconnect_0/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins out1/s_axi_control] [get_bd_intf_pins control_0_0/M03_AXI]
assign_bd_address [get_bd_addr_segs out1/s_axi_control/Reg*] -offset 0xa0003000 -range 4096
connect_bd_net [get_bd_pins out1/ap_clk] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins out1/ap_rst_n] [get_bd_pins smartconnect_0/aresetn]
connect_bd_intf_net [get_bd_intf_pins out1/in0_V] [get_bd_intf_pins DataflowCore_1/m_axis_0]
