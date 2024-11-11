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
	input	logic [STREAM_BITS-1:0]  s_axis_q_tdata,

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
