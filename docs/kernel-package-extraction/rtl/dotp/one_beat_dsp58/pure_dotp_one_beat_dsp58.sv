module pure_dotp_one_beat_dsp58(
    input logic ap_clk, input logic ap_clk2x, input logic ap_rst_n,
    input logic [15:0] in0_V_tdata,
    input logic in0_V_tvalid, output logic in0_V_tready,
    input logic [23:0] in1_V_tdata,
    input logic in1_V_tvalid, output logic in1_V_tready,
    output logic [15:0] out0_V_tdata,
    output logic out0_V_tvalid, input logic out0_V_tready
);
    integer unsigned synapse_fold = 0;
    wire input_last = synapse_fold == 0;
    always_ff @(posedge ap_clk) begin
        if (!ap_rst_n) synapse_fold <= 0;
        else if (in0_V_tvalid && in0_V_tready) begin
            if (input_last) synapse_fold <= 0;
            else synapse_fold <= synapse_fold + 1;
        end
    end
    dotp_axi #(
        .ACCU_WIDTH(8),
        .ACTIVATION_BROADCASTING(1),
        .ACTIVATION_WIDTH(3),
        .FORCE_BEHAVIORAL(0),
        .NARROW_WEIGHTS(0),
        .PE(2),
        .PUMPED_COMPUTE(0),
        .SEGMENTLEN(0),
        .SIGNED_ACTIVATIONS(1),
        .SIMD(4),
        .VERSION(3),
        .WEIGHT_WIDTH(3)
    ) dut (
        .ap_clk(ap_clk), .ap_clk2x(ap_clk2x), .ap_rst_n(ap_rst_n),
        .s_axis_weights_tdata(in1_V_tdata),
        .s_axis_weights_tvalid(in1_V_tvalid),
        .s_axis_weights_tready(in1_V_tready),
        .s_axis_input_tdata(in0_V_tdata),
        .s_axis_input_tvalid(in0_V_tvalid),
        .s_axis_input_tlast(input_last),
        .s_axis_input_tready(in0_V_tready),
        .m_axis_output_tdata(out0_V_tdata),
        .m_axis_output_tvalid(out0_V_tvalid),
        .m_axis_output_tready(out0_V_tready)
    );
endmodule
