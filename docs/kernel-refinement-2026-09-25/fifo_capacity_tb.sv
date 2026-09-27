// Scoped experiment: measure the native FIFO's effective storage and capacity.
module fifo_capacity_tb;
    logic clk = 0;
    always #5 clk = !clk;
    logic rst = 1;
    logic [6:0] idat = 0;
    logic ivld = 0;
    wire irdy;
    wire [6:0] odat;
    wire ovld;
    int accepted = 0;

    fifo #(.DATA_WIDTH(7), .DEPTH(2), .RAM_STYLE("ultra")) dut (
        .clk, .rst, .idat, .ivld, .irdy, .odat, .ovld, .ordy(1'b0)
    );
    always @(posedge clk) begin
        if (!rst && ivld && irdy) accepted <= accepted + 1;
    end
    initial begin
        repeat (3) @(negedge clk);
        rst = 0;
        ivld = 1;
        repeat (20) begin
            @(negedge clk);
            idat = accepted[6:0];
        end
        $display("FIFO_PROBE requested_depth=2 requested_style=ultra effective_style=%s accepted_while_stalled=%0d", dut.RAM_STYLE_EFF, accepted);
        if (accepted != 5) $fatal(1, "Unexpected FIFO capacity");
        $finish;
    end
endmodule
