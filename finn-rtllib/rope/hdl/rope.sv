module rope #(
	int unsigned  SEQ_LEN,
	int unsigned  HIDDEN_DIM,
	int unsigned  SIMD,
	int unsigned  ELEM_BITS,

	// INITIALIZE WEIGHTS
	parameter COS_INIT_FILE = "",
	parameter SIN_INIT_FILE = "",

	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8),
	localparam int unsigned  WEIGHT_DEPTH = SEQ_LEN * HIDDEN_DIM
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [STREAM_BITS-1:0]  s_qaxis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_tdata
);

	logic m_axis_dup_0_tready;
	logic m_axis_dup_0_tvalid;
	logic [STREAM_BITS-1:0] m_axis_dup_0_tdata;

	logic m_axis_dup_1_tready;
	logic m_axis_dup_1_tvalid;
	logic [STREAM_BITS-1:0] m_axis_dup_1_tdata;

	duplicate_stream #(
	  .SIMD(SIMD),
	  .ELEM_BITS(ELEM_BITS),
	  .HIDDEN_DIM(HIDDEN_DIM)
	) duplicate_stream_inst (
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),

		.s_axis_tready(s_axis_tready),
		.s_axis_tvalid(s_axis_tvalid),
		.s_axis_tdata(s_axis_tdata),

		.m_axis_0_tready(m_axis_dup_0_tready),
		.m_axis_0_tvalid(m_axis_dup_0_tvalid),
		.m_axis_0_tdata(m_axis_dup_0_tdata),

		.m_axis_1_tready(m_axis_dup_1_tready),
		.m_axis_1_tvalid(m_axis_dup_1_tvalid),
		.m_axis_1_tdata(m_axis_dup_1_tdata)
	);


	logic m_axis_0_to_half_tready;
	logic m_axis_0_to_half_tvalid;
	logic [STREAM_BITS-1:0] m_axis_0_to_half_tdata;

	logic m_axis_half_to_end_tready;
	logic m_axis_half_to_end_tvalid;
	logic [STREAM_BITS-1:0] m_axis_half_to_end_tdata;

	slice #(
		.SIMD(SIMD),
		.ELEM_BITS(ELEM_BITS),
		.SLICE_0_STARTS(0),
	    .SLICE_0_ENDS((HIDDEN_DIM + 1)/2 - 1),
        .SLICE_1_STARTS((HIDDEN_DIM + 1)/2),
		.SLICE_1_ENDS(HIDDEN_DIM-1),
		.SLICE_DIM_SIZE(HIDDEN_DIM)
	) slice_inst (
		//- Global Control ------------------
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),

		//- AXI Stream - Input --------------
		.s_axis_tready(m_axis_dup_0_tready),
		.s_axis_tvalid(m_axis_dup_0_tvalid),
		.s_axis_tdata(m_axis_dup_0_tdata),

		//- AXI Stream - Output -------------
		.m_axis_slice_0_tready(m_axis_0_to_half_tready),
		.m_axis_slice_0_tvalid(m_axis_0_to_half_tvalid),
		.m_axis_slice_0_tdata(m_axis_0_to_half_tdata),

		//- AXI Stream - Output -------------
		.m_axis_slice_1_tready(m_axis_half_to_end_tready),
		.m_axis_slice_1_tvalid(m_axis_half_to_end_tvalid),
		.m_axis_slice_1_tdata(m_axis_half_to_end_tdata)
	);

	localparam int unsigned FIFO_COUNT_WIDTH = $clog2(2*HIDDEN_DIM) + 1;
	logic [FIFO_COUNT_WIDTH-1:0] count;
	logic [FIFO_COUNT_WIDTH-1:0] maxcount;

	logic m_axis_fifo_tready;
	logic m_axis_fifo_tvalid;
	logic [STREAM_BITS-1:0] m_axis_fifo_tdata;

	Q_srl #(
		.depth(2*HIDDEN_DIM),
		.width(STREAM_BITS)
	) fifo_impl (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(count),
		.maxcount(maxcount),
		.i_d(m_axis_0_to_half_tdata),
		.i_v(m_axis_0_to_half_tvalid),
		.i_r(m_axis_0_to_half_tready),
		.o_d(m_axis_fifo_tdata),
		.o_v(m_axis_fifo_tvalid),
		.o_r(m_axis_fifo_tready)
	);


	logic  m_axis_neg_tready;
	logic  m_axis_neg_tvalid;
	logic [STREAM_BITS-1:0]  m_axis_neg_tdata;

	v_unary_op #(
		.SIMD(SIMD),
		.ELEM_BITS(ELEM_BITS),
		.OP("neg")
	) negative_op (
		//- Global Control ------------------
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),

		//- AXI Stream - Input --------------
		.s_axis_tready(m_axis_half_to_end_tready),
		.s_axis_tvalid(m_axis_half_to_end_tvalid),
		.s_axis_tdata(m_axis_half_to_end_tdata),

		//- AXI Stream - Output -------------
		.m_axis_tready(m_axis_neg_tready),
		.m_axis_tvalid(m_axis_neg_tvalid),
		.m_axis_tdata(m_axis_neg_tdata)
	);

	logic m_axis_concat_0_tready;
	logic m_axis_concat_0_tvalid;
    logic [STREAM_BITS-1:0] m_axis_concat_0_tdata;

    concat #(
    	.SIMD(SIMD),
	    .ELEM_BITS(ELEM_BITS),
		.SLICE_0_STARTS(0),
	    .SLICE_0_ENDS((HIDDEN_DIM + 1)/2 - 1),
        .SLICE_1_STARTS((HIDDEN_DIM + 1)/2),
	    .SLICE_1_ENDS(HIDDEN_DIM),
	    .SLICE_DIM_SIZE(HIDDEN_DIM)
	) concat_0 (
		//- Global Control ------------------
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),

		//- AXI Stream - Input --------------
		.s_axis_slice_0_tready(m_axis_neg_tready),
		.s_axis_slice_0_tvalid(m_axis_neg_tvalid),
		.s_axis_slice_0_tdata(m_axis_neg_tdata),

		.s_axis_slice_1_tready(m_axis_fifo_tready),
		.s_axis_slice_1_tvalid(m_axis_fifo_tvalid),
		.s_axis_slice_1_tdata(m_axis_fifo_tdata),

		//- AXI Stream - Output -------------
		.m_axis_tready(m_axis_concat_0_tready),
		.m_axis_tvalid(m_axis_concat_0_tvalid),
		.m_axis_tdata(m_axis_concat_0_tdata)
	);

	logic  m_axis_c_weights_tready;
	logic  m_axis_c_weights_tvalid;
	logic [STREAM_BITS-1:0]  m_axis_c_weights_tdata;

  memstream #(
	.DEPTH(WEIGHT_DEPTH),
	.WIDTH(STREAM_BITS),
	.INIT_FILE(COS_INIT_FILE),
	.RAM_STYLE("auto")
  ) c_weight (
	.clk(ap_clk),
	.rst(~ap_rst_n),

	// Configuration and readback interface - compatible with ap_memory
	.config_ce(),
	.config_we(),
	.config_address(),
	.config_d0(),

	.config_rack(),
	.config_q0(),

	// Continuous output stream
	.ordy(m_axis_c_weights_tready),
	.ovld(m_axis_c_weights_tvalid),
	.odat(m_axis_c_weights_tdata)
);

  logic  m_axis_c_mul_tready;
  logic  m_axis_c_mul_tvalid;
  logic [STREAM_BITS-1:0]  m_axis_c_mul_tdata;

  vv_op #(
	.SIMD(SIMD),
	.ELEM_BITS(ELEM_BITS),
	.OP("mul")
  ) vv_c_mul_array (
	//- Global Control ------------------
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),

	//- AXI Stream - Input --------------
	.s_axis_a_tready(m_axis_dup_1_tready),
	.s_axis_a_tvalid(m_axis_dup_1_tvalid),
	.s_axis_a_tdata(m_axis_dup_1_tdata),

	//- AXI Stream - Input --------------
	.s_axis_b_tready(m_axis_c_weights_tready),
	.s_axis_b_tvalid(m_axis_c_weights_tvalid),
	.s_axis_b_tdata(m_axis_c_weights_tdata),

	//- AXI Stream - Output -------------
	.m_axis_c_tready(m_axis_c_mul_tready),
	.m_axis_c_tvalid(m_axis_c_mul_tvalid),
	.m_axis_c_tdata(m_axis_c_mul_tdata)
  );

  logic m_axis_c_fifo_tready;
  logic m_axis_c_fifo_tvalid;
  logic [STREAM_BITS-1:0] m_axis_c_fifo_tdata;

  Q_srl #(
		.depth(2*HIDDEN_DIM),
		.width(STREAM_BITS)
	) c_fifo_impl (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(count),
		.maxcount(maxcount),
		.i_d(m_axis_c_mul_tdata),
		.i_v(m_axis_c_mul_tvalid),
		.i_r(m_axis_c_mul_tready),
		.o_d(m_axis_c_fifo_tdata),
		.o_v(m_axis_c_fifo_tvalid),
		.o_r(m_axis_c_fifo_tready)
	);

  logic  m_axis_s_weights_tready;
  logic  m_axis_s_weights_tvalid;
  logic [STREAM_BITS-1:0]  m_axis_s_weights_tdata;

  memstream #(
	.DEPTH(WEIGHT_DEPTH),
	.WIDTH(STREAM_BITS),
	.INIT_FILE(SIN_INIT_FILE),
	.RAM_STYLE("auto")
  ) s_weight (
	.clk(ap_clk),
	.rst(~ap_rst_n),

	// Configuration and readback interface - compatible with ap_memory
	.config_ce(),
	.config_we(),
	.config_address(),
	.config_d0(),

	.config_rack(),
	.config_q0(),

	// Continuous output stream
	.ordy(m_axis_s_weights_tready),
	.ovld(m_axis_s_weights_tvalid),
	.odat(m_axis_s_weights_tdata)
);

  logic  m_axis_s_mul_tready;
  logic  m_axis_s_mul_tvalid;
  logic [STREAM_BITS-1:0]  m_axis_s_mul_tdata;

  vv_op #(
	.SIMD(SIMD),
	.ELEM_BITS(ELEM_BITS),
	.OP("mul")
  ) vv_s_mul_array (
	//- Global Control ------------------
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),

	//- AXI Stream - Input --------------
	.s_axis_a_tready(m_axis_concat_0_tready),
	.s_axis_a_tvalid(m_axis_concat_0_tvalid),
	.s_axis_a_tdata(m_axis_concat_0_tdata),

	//- AXI Stream - Input --------------
	.s_axis_b_tready(m_axis_s_weights_tready),
	.s_axis_b_tvalid(m_axis_s_weights_tvalid),
	.s_axis_b_tdata(m_axis_s_weights_tdata),

	//- AXI Stream - Output -------------
	.m_axis_c_tready(m_axis_s_mul_tready),
	.m_axis_c_tvalid(m_axis_s_mul_tvalid),
	.m_axis_c_tdata(m_axis_s_mul_tdata)
  );

  vv_op #(
	.SIMD(SIMD),
	.ELEM_BITS(ELEM_BITS),
	.OP("add")
  ) vv_add_array (
	//- Global Control ------------------
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),

	//- AXI Stream - Input --------------
	.s_axis_a_tready(m_axis_c_fifo_tready),
	.s_axis_a_tvalid(m_axis_c_fifo_tvalid),
	.s_axis_a_tdata(m_axis_c_fifo_tdata),

	//- AXI Stream - Input --------------
	.s_axis_b_tready(m_axis_s_mul_tready),
	.s_axis_b_tvalid(m_axis_s_mul_tvalid),
	.s_axis_b_tdata(m_axis_s_mul_tdata),

	//- AXI Stream - Output -------------
	.m_axis_c_tready(m_axis_tready),
	.m_axis_c_tvalid(m_axis_tvalid),
	.m_axis_c_tdata(m_axis_tdata)
);


endmodule