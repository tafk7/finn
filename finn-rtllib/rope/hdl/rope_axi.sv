/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Rotational Positional Embedding.
 * @author	Josh Monson <joshmonson@microsoft.com>
 *****************************************************************************/

module range_indicator #(
	parameter int unsigned RANGE_STEP = 1,
	parameter int unsigned RANGE_MIN  = 0,
	parameter int unsigned RANGE_MAX  = 1,
	parameter int unsigned RANGE_COUNT = 4,
	parameter int unsigned RANGE_0_START = 0,
	parameter int unsigned RANGE_0_END   = 1,
	parameter int unsigned RANGE_1_START = 0,
	parameter int unsigned RANGE_1_END   = 1,
	parameter int unsigned RANGE_2_START = 0,
	parameter int unsigned RANGE_2_END   = 1,
	parameter int unsigned RANGE_3_START = 0,
	parameter int unsigned RANGE_3_END   = 1
) (
	input logic ap_clk,
	input logic ap_rst_n,
	input logic increment_range_indices,
	output logic [RANGE_COUNT-1:0] range_indicators,
	output logic [31:0] range_index_low
);

   logic [31:0] c_range_index_low, n_range_index_low;
   logic [31:0] c_range_index_high, n_range_index_high;

	localparam int unsigned RANGE_STARTS[3:0] = '{ RANGE_3_START, RANGE_2_START, RANGE_1_START, RANGE_0_START };
	localparam int unsigned RANGE_ENDS[3:0]   = '{ RANGE_3_END, RANGE_2_END, RANGE_1_END, RANGE_0_END };

	genvar i;
	generate
		for (i=0; i<RANGE_COUNT; i=i+1) begin : range_indicator
   			assign range_indicators[i] = (c_range_index_low >= RANGE_STARTS[i]) && (c_range_index_high <= RANGE_ENDS[i]);
		end
	endgenerate

	always_comb begin
		n_range_index_low  = c_range_index_low;
		n_range_index_high = c_range_index_high;

		if (increment_range_indices) begin
			if (c_range_index_high + RANGE_STEP < RANGE_MAX) begin
				n_range_index_low  = c_range_index_low + RANGE_STEP;
				n_range_index_high = c_range_index_high + RANGE_STEP;
			end else if(c_range_index_high + RANGE_STEP >= RANGE_MAX)  begin
				n_range_index_low  = RANGE_MIN;
				n_range_index_high = RANGE_MIN + RANGE_STEP - 1;
			end
		end
   end

   always_ff @(posedge ap_clk) begin
		c_range_index_low <= n_range_index_low;
		c_range_index_high <= n_range_index_high;

		if (!ap_rst_n) begin
			c_range_index_low  <= RANGE_MIN;
			c_range_index_high <= RANGE_MIN + RANGE_STEP - 1;
		end
   end

	assign range_index_low = c_range_index_low;
endmodule

module concat #(
    parameter int unsigned  SIMD = 1,
	parameter int unsigned  ELEM_BITS = 32,
	parameter int unsigned  SLICE_0_STARTS = 0,
	parameter int unsigned  SLICE_0_ENDS = 1,
    parameter int unsigned  SLICE_1_STARTS = 0,
	parameter int unsigned  SLICE_1_ENDS = 1,
	parameter int unsigned  SLICE_DIM_SIZE = 1,
	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8),
	localparam int unsigned SLICE_DIM_WIDTH = $clog2(SLICE_DIM_SIZE) + 1
) (
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

    //- AXI Stream - Input --------------
	output	logic  s_axis_slice_0_tready,
	input	logic  s_axis_slice_0_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_slice_0_tdata,

	output	logic  s_axis_slice_1_tready,
	input	logic  s_axis_slice_1_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_slice_1_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_tdata
);

	logic [31:0]            range_index_low;
	logic [1:0]             range_indicators;

	logic [STREAM_BITS-1:0] stream_fifo_0_tdata;
	logic                   stream_fifo_0_tvalid;
	logic                   stream_fifo_0_tready;

	logic [SIMD-1:0]        element_tvalid;

	genvar i;
	generate
		for (i=0; i<SIMD; i=i+1) begin : concat
			always_comb begin
				if (range_index_low + i <= SLICE_0_ENDS) begin
					stream_fifo_0_tdata[i*ELEM_BITS +: ELEM_BITS] = s_axis_slice_0_tdata[i*ELEM_BITS +: ELEM_BITS];
					element_tvalid[i] = s_axis_slice_0_tvalid;
				end else begin
					stream_fifo_0_tdata[i*ELEM_BITS +: ELEM_BITS]  = s_axis_slice_1_tdata[i*ELEM_BITS +: ELEM_BITS];
					element_tvalid[i] = s_axis_slice_1_tvalid;
				end
			end
		end
	endgenerate

	assign stream_fifo_0_tvalid = &element_tvalid;

	logic increment_indices;

	always_comb begin
		increment_indices =  stream_fifo_0_tvalid & stream_fifo_0_tready;

		s_axis_slice_0_tready = range_indicators[0] & stream_fifo_0_tready;
		s_axis_slice_1_tready = range_indicators[1] & stream_fifo_0_tready;
	end

	range_indicator #(
		.RANGE_STEP(SIMD),
		.RANGE_MIN(0),
		.RANGE_MAX(SLICE_DIM_SIZE),
		.RANGE_COUNT(2),
		.RANGE_0_START(SLICE_0_STARTS),
		.RANGE_0_END(SLICE_0_ENDS),
		.RANGE_1_START(SLICE_1_STARTS),
		.RANGE_1_END(SLICE_1_ENDS)
	) range_indicator_inst(
			.ap_clk(ap_clk),
			.ap_rst_n(ap_rst_n),
			.increment_range_indices(increment_indices),
			.range_indicators(range_indicators),
			.range_index_low(range_index_low)
	);

	Q_srl #(
		.depth(16),
		.width(STREAM_BITS)
	) stream_fifo_0 (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(),
		.maxcount(),
		.i_d(stream_fifo_0_tdata),
		.i_v(stream_fifo_0_tvalid),
		.i_r(stream_fifo_0_tready),
		.o_d(m_axis_tdata),
		.o_v(m_axis_tvalid),
		.o_r(m_axis_tready)
	);

endmodule

module slice #(
	parameter int unsigned  SIMD = 1,
	parameter int unsigned  ELEM_BITS = 32,
	parameter int unsigned  SLICE_0_STARTS = 0,
	parameter int unsigned  SLICE_0_ENDS = 1,
    parameter int unsigned  SLICE_1_STARTS = 0,
	parameter int unsigned  SLICE_1_ENDS = 1,
	parameter int unsigned  SLICE_DIM_SIZE = 1,
	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
) (
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_slice_0_tready,
	output	logic  m_axis_slice_0_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_slice_0_tdata,

	input	logic  m_axis_slice_1_tready,
	output	logic  m_axis_slice_1_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_slice_1_tdata
);

   localparam SLICE_DIM_WIDTH  = $clog2(SLICE_DIM_SIZE + 1);

   logic  in_slice_0_range, in_slice_1_range;
   logic  increment_indices;

   assign m_axis_slice_0_tdata = s_axis_tdata;
   assign m_axis_slice_1_tdata = s_axis_tdata;

   range_indicator #(
	.RANGE_STEP(SIMD),
	.RANGE_MIN(0),
	.RANGE_MAX(SLICE_DIM_SIZE),
	.RANGE_COUNT(2),
	.RANGE_0_START(SLICE_0_STARTS),
	.RANGE_0_END(SLICE_0_ENDS),
	.RANGE_1_START(SLICE_1_STARTS),
	.RANGE_1_END(SLICE_1_ENDS)
   ) range_indicator_inst(
		.ap_clk(ap_clk),
		.ap_rst_n(ap_rst_n),
		.increment_range_indices(increment_indices),
		.range_indicators({in_slice_1_range, in_slice_0_range}),
		.range_index_low()
	);

   always_comb begin
	  s_axis_tready         = 1'b0;
	  increment_indices     = 1'b0;
	  m_axis_slice_0_tvalid = 1'b0;
	  m_axis_slice_1_tvalid = 1'b0;

	  if(s_axis_tvalid) begin
		  case ({in_slice_1_range, in_slice_0_range})
		    2'b00: begin
			  s_axis_tready         = 1'b0;
  			  increment_indices     = 1'b0;
	  		  m_axis_slice_0_tvalid = 1'b0;
	          m_axis_slice_1_tvalid = 1'b0;
			end
			2'b01: begin
				if (m_axis_slice_0_tready) begin
					s_axis_tready         = 1'b1;
					increment_indices     = 1'b1;
					m_axis_slice_0_tvalid = 1'b1;
				end
			end
			2'b10: begin
				if (m_axis_slice_1_tready) begin
					s_axis_tready         = 1'b1;
					increment_indices     = 1'b1;
					m_axis_slice_1_tvalid = 1'b1;
				end
			end
			2'b11: begin
				if (m_axis_slice_0_tready && m_axis_slice_1_tready) begin
					s_axis_tready         = 1'b1;
					increment_indices     = 1'b1;
					m_axis_slice_0_tvalid = 1'b1;
					m_axis_slice_1_tvalid = 1'b1;
				end
			end
		  endcase
	  end
   end

endmodule

module v_unary_op #(
	parameter int unsigned  SIMD = 1,
	parameter int unsigned  ELEM_BITS = 32,
	parameter               OP = "neg",
	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
) (
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_tdata
);
	logic [STREAM_BITS-1:0] results;
	genvar i;
	generate
		for (i=0; i<SIMD; i=i+1) begin : unary_op
			if (OP == "neg") begin
				assign results[i*ELEM_BITS +: ELEM_BITS] = -s_axis_tdata[i*ELEM_BITS +: ELEM_BITS];
			end else begin
				assign results[i*ELEM_BITS +: ELEM_BITS] = 0;
			end
		end
	endgenerate

	Q_srl #(
		.depth(16),
		.width(STREAM_BITS)
	) stream_0_fifo (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(),
		.maxcount(),
		.i_d(results),
		.i_v(s_axis_tvalid),
		.i_r(s_axis_tready),
		.o_d(m_axis_tdata),
		.o_v(m_axis_tvalid),
		.o_r(m_axis_tready)
	);

endmodule


module vv_op #(
	parameter int unsigned  SIMD = 1,
	parameter int unsigned  ELEM_BITS = 32,
	parameter               OP = "mul",
	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
) (
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_a_tready,
	input	logic  s_axis_a_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_a_tdata,

	//- AXI Stream - Input --------------
	output	logic  s_axis_b_tready,
	input	logic  s_axis_b_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_b_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_c_tready,
	output	logic  m_axis_c_tvalid,
	output	logic [STREAM_BITS-1:0] m_axis_c_tdata
);

	logic [STREAM_BITS-1:0] stream_fifo_0_tdata;
	logic stream_fifo_0_tvalid;
	logic stream_fifo_0_tready;

	assign stream_fifo_0_tvalid = s_axis_a_tvalid & s_axis_b_tvalid;
	assign s_axis_a_tready        = stream_fifo_0_tvalid & stream_fifo_0_tready;
	assign s_axis_b_tready		  = stream_fifo_0_tvalid & stream_fifo_0_tready;
    genvar i;
	generate
		for (i=0; i<SIMD; i=i+1) begin : vv_mul
			if (OP == "mul") begin
				assign stream_fifo_0_tdata[i*ELEM_BITS +: ELEM_BITS] = s_axis_a_tdata[i*ELEM_BITS +: ELEM_BITS] * s_axis_b_tdata[i*ELEM_BITS +: ELEM_BITS];
			end else if (OP == "add") begin
				assign stream_fifo_0_tdata[i*ELEM_BITS +: ELEM_BITS] = s_axis_a_tdata[i*ELEM_BITS +: ELEM_BITS] + s_axis_b_tdata[i*ELEM_BITS +: ELEM_BITS];
			end else  begin
				assign stream_fifo_0_tdata[i*ELEM_BITS +: ELEM_BITS] = 0;
			end
		end
	endgenerate

	Q_srl #(
		.depth(16),
		.width(STREAM_BITS)
	) stream_fifo_0 (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(),
		.maxcount(),
		.i_d(stream_fifo_0_tdata),
		.i_v(stream_fifo_0_tvalid),
		.i_r(stream_fifo_0_tready),
		.o_d(m_axis_c_tdata),
		.o_v(m_axis_c_tvalid),
		.o_r(m_axis_c_tready)
	);

endmodule

module duplicate_stream #(
	parameter int unsigned   SIMD = 1,
	parameter int unsigned   ELEM_BITS = 32,
	parameter int unsigned   HIDDEN_DIM = 1,
	localparam int unsigned  STREAM_BITS = 8*(1 + (SIMD*ELEM_BITS-1)/8)
) (
	input logic ap_clk,
	input logic ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_tdata,

	input	logic  m_axis_0_tready,
	output	logic  m_axis_0_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_0_tdata,

	input	logic  m_axis_1_tready,
	output	logic  m_axis_1_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_1_tdata
);
	logic stream_0_fifo_tready;
	logic stream_1_fifo_tready;

	assign s_axis_tready = stream_0_fifo_tready & stream_1_fifo_tready;

	Q_srl #(
		.depth(HIDDEN_DIM),
		.width(STREAM_BITS)
	) stream_0_fifo (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(),
		.maxcount(),
		.i_d(s_axis_tdata),
		.i_v(s_axis_tvalid),
		.i_r(stream_0_fifo_tready),
		.o_d(m_axis_0_tdata),
		.o_v(m_axis_0_tvalid),
		.o_r(m_axis_0_tready)
	);

	Q_srl #(
		.depth(HIDDEN_DIM),
		.width(STREAM_BITS)
	) stream_1_fifo (
		.clock(ap_clk),
		.reset(!ap_rst_n),
		.count(),
		.maxcount(),
		.i_d(s_axis_tdata),
		.i_v(s_axis_tvalid),
		.i_r(stream_1_fifo_tready),
		.o_d(m_axis_1_tdata),
		.o_v(m_axis_1_tvalid),
		.o_r(m_axis_1_tready)
	);

endmodule

module rope_axi #(
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
	output	logic  s_axis_q_tready,
	input	logic  s_axis_q_tvalid,
	input	logic [STREAM_BITS-1:0]  s_qaxis_q_tdata,

	output	logic  s_axis_k_tready,
	input	logic  s_axis_k_tvalid,
	input	logic [STREAM_BITS-1:0]  s_axis_k_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_q_tready,
	output	logic  m_axis_q_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_q_tdata,

	input	logic  m_axis_k_tready,
	output	logic  m_axis_k_tvalid,
	output	logic [STREAM_BITS-1:0]  m_axis_k_tdata
);

  rope #(
	.SEQ_LEN(SEQ_LEN),
	.HIDDEN_DIM(HIDDEN_DIM),
	.SIMD(SIMD),
	.ELEM_BITS(ELEM_BITS),

	// INITIALIZE WEIGHTS
	.COS_INIT_FILE(COS_INIT_FILE),
	.SIN_INIT_FILE(SIN_INIT_FILE)
  ) rope_q_inst (
	//- Global Control ------------------
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),

	//- AXI Stream - Input --------------
	.s_axis_tready(s_axis_q_tready),
	.s_axis_tvalid(s_axis_q_tvalid),
	.s_qaxis_tdata(s_qaxis_q_tdata),

	//- AXI Stream - Output -------------
	.m_axis_tready(m_axis_q_tready),
	.m_axis_tvalid(m_axis_q_tvalid),
	.m_axis_tdata(m_axis_q_tdata)
  );

  rope #(
	.SEQ_LEN(SEQ_LEN),
	.HIDDEN_DIM(HIDDEN_DIM),
	.SIMD(SIMD),
	.ELEM_BITS(ELEM_BITS),

	// INITIALIZE WEIGHTS
	.COS_INIT_FILE(COS_INIT_FILE),
	.SIN_INIT_FILE(SIN_INIT_FILE)
  ) rope_k_inst (
	//- Global Control ------------------
	.ap_clk(ap_clk),
	.ap_rst_n(ap_rst_n),

	//- AXI Stream - Input --------------
	.s_axis_tready(s_axis_k_tready),
	.s_axis_tvalid(s_axis_k_tvalid),
	.s_qaxis_tdata(s_qaxis_k_tdata),

	//- AXI Stream - Output -------------
	.m_axis_tready(m_axis_k_tready),
	.m_axis_tvalid(m_axis_k_tvalid),
	.m_axis_tdata(m_axis_k_tdata)
  );

endmodule : rope_axi
