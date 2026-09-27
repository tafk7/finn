module observe_mvau(
input wire ap_clk,
input wire ap_clk2x,
input wire ap_rst_n,
input wire [7:0] in0_V_tdata,
output wire in0_V_tready,
input wire in0_V_tvalid,
output wire [23:0] out0_V_tdata,
input wire out0_V_tready,
output wire out0_V_tvalid,
output wire [5:0] observe_replay_data,
output wire observe_replay_valid,
output wire observe_replay_ready,
output wire observe_replay_last,
output wire [23:0] observe_weights_data,
output wire observe_weights_valid,
output wire observe_weights_ready);
finn_mvau_cyclic__e628e25bd87ad3411a63146741f0d61bd99bc3ddc3442eb77cd3f805b02db976 dut (.ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n), .in0_V_tdata(in0_V_tdata), .in0_V_tready(in0_V_tready), .in0_V_tvalid(in0_V_tvalid), .out0_V_tdata(out0_V_tdata), .out0_V_tready(out0_V_tready), .out0_V_tvalid(out0_V_tvalid));
assign observe_replay_data = dut.u_replay.odat;
assign observe_replay_valid = dut.u_replay.ovld;
assign observe_replay_ready = dut.u_replay.ordy;
assign observe_replay_last = dut.u_replay.olast;
assign observe_weights_data = dut.u_compute.s_axis_weights_tdata;
assign observe_weights_valid = dut.u_compute.s_axis_weights_tvalid;
assign observe_weights_ready = dut.u_compute.s_axis_weights_tready;
endmodule
