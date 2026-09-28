module probe;
  logic clk = 0, rst = 1;
  logic [7:0] mark1_idat = 0; logic mark1_ivld = 0, mark1_ordy = 1;
  wire mark1_irdy, mark1_ovld; wire [7:0] mark1_odat; wire [0:0] mark1_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(1), .D(1), .DIMS('{1}), .COEFS('{1})) mark1 (
    .clk, .rst, .idat(mark1_idat), .ivld(mark1_ivld), .irdy(mark1_irdy), .odat(mark1_odat), .ovld(mark1_ovld), .ordy(mark1_ordy), .olst(mark1_olst));
  logic [7:0] mark3_idat = 0; logic mark3_ivld = 0, mark3_ordy = 1;
  wire mark3_irdy, mark3_ovld; wire [7:0] mark3_odat; wire [0:0] mark3_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(3), .D(1), .DIMS('{3}), .COEFS('{1})) mark3 (
    .clk, .rst, .idat(mark3_idat), .ivld(mark3_ivld), .irdy(mark3_irdy), .odat(mark3_odat), .ovld(mark3_ovld), .ordy(mark3_ordy), .olst(mark3_olst));
  logic [7:0] mark8_idat = 0; logic mark8_ivld = 0, mark8_ordy = 1;
  wire mark8_irdy, mark8_ovld; wire [7:0] mark8_odat; wire [0:0] mark8_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(8), .D(1), .DIMS('{8}), .COEFS('{1})) mark8 (
    .clk, .rst, .idat(mark8_idat), .ivld(mark8_ivld), .irdy(mark8_irdy), .odat(mark8_odat), .ovld(mark8_ovld), .ordy(mark8_ordy), .olst(mark8_olst));
  logic [7:0] mark64_idat = 0; logic mark64_ivld = 0, mark64_ordy = 1;
  wire mark64_irdy, mark64_ovld; wire [7:0] mark64_odat; wire [0:0] mark64_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(64), .D(1), .DIMS('{64}), .COEFS('{1})) mark64 (
    .clk, .rst, .idat(mark64_idat), .ivld(mark64_ivld), .irdy(mark64_irdy), .odat(mark64_odat), .ovld(mark64_ovld), .ordy(mark64_ordy), .olst(mark64_olst));
  logic [7:0] rep2x2_idat = 0; logic rep2x2_ivld = 0, rep2x2_ordy = 1;
  wire rep2x2_irdy, rep2x2_ovld; wire [7:0] rep2x2_odat; wire [1:0] rep2x2_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(2), .D(2), .DIMS('{2, 2}), .COEFS('{0, 1})) rep2x2 (
    .clk, .rst, .idat(rep2x2_idat), .ivld(rep2x2_ivld), .irdy(rep2x2_irdy), .odat(rep2x2_odat), .ovld(rep2x2_ovld), .ordy(rep2x2_ordy), .olst(rep2x2_olst));
  logic [7:0] rep4x8_idat = 0; logic rep4x8_ivld = 0, rep4x8_ordy = 1;
  wire rep4x8_irdy, rep4x8_ovld; wire [7:0] rep4x8_odat; wire [1:0] rep4x8_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(8), .D(2), .DIMS('{4, 8}), .COEFS('{0, 1})) rep4x8 (
    .clk, .rst, .idat(rep4x8_idat), .ivld(rep4x8_ivld), .irdy(rep4x8_irdy), .odat(rep4x8_odat), .ovld(rep4x8_ovld), .ordy(rep4x8_ordy), .olst(rep4x8_olst));
  logic [7:0] rep4x1_idat = 0; logic rep4x1_ivld = 0, rep4x1_ordy = 1;
  wire rep4x1_irdy, rep4x1_ovld; wire [7:0] rep4x1_odat; wire [1:0] rep4x1_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(1), .D(2), .DIMS('{4, 1}), .COEFS('{0, 1})) rep4x1 (
    .clk, .rst, .idat(rep4x1_idat), .ivld(rep4x1_ivld), .irdy(rep4x1_irdy), .odat(rep4x1_odat), .ovld(rep4x1_ovld), .ordy(rep4x1_ordy), .olst(rep4x1_olst));
  logic [7:0] rep3x16_idat = 0; logic rep3x16_ivld = 0, rep3x16_ordy = 1;
  wire rep3x16_irdy, rep3x16_ovld; wire [7:0] rep3x16_odat; wire [1:0] rep3x16_olst;
  input_gen #(.DATA_WIDTH(8), .FM_SIZE(16), .D(2), .DIMS('{3, 16}), .COEFS('{0, 1})) rep3x16 (
    .clk, .rst, .idat(rep3x16_idat), .ivld(rep3x16_ivld), .irdy(rep3x16_irdy), .odat(rep3x16_odat), .ovld(rep3x16_ovld), .ordy(rep3x16_ordy), .olst(rep3x16_olst));
  initial begin
    $display("MEASURE mark1 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", mark1.BUF_SIZE, mark1.MAX_OCCUPANCY);
    $display("MEASURE mark3 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", mark3.BUF_SIZE, mark3.MAX_OCCUPANCY);
    $display("MEASURE mark8 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", mark8.BUF_SIZE, mark8.MAX_OCCUPANCY);
    $display("MEASURE mark64 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", mark64.BUF_SIZE, mark64.MAX_OCCUPANCY);
    $display("MEASURE rep2x2 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", rep2x2.BUF_SIZE, rep2x2.MAX_OCCUPANCY);
    $display("MEASURE rep4x8 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", rep4x8.BUF_SIZE, rep4x8.MAX_OCCUPANCY);
    $display("MEASURE rep4x1 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", rep4x1.BUF_SIZE, rep4x1.MAX_OCCUPANCY);
    $display("MEASURE rep3x16 BUF_SIZE=%0d MAX_OCCUPANCY=%0d", rep3x16.BUF_SIZE, rep3x16.MAX_OCCUPANCY);
    $finish;
  end
endmodule
