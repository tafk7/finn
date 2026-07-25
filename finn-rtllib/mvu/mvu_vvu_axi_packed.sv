/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
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
 * @brief	Standalone DSP58 INT8-packed MVU/VVU AXI-lite wrapper.
 * @details
 *	 Base plumbing (mvu_vvu_axi_base_head.svh) + the DSP58 INT8-packed
 *	 `mvu_vvu_8sx9_dsp58` compute core + output stage
 *	 (mvu_vvu_axi_base_tail.svh). One of the two per-core wrappers that replace
 *	 the former fused mvu_vvu_axi.sv; this one owns `mvu_vvu_8sx9_dsp58.sv` alone
 *	 and never references the soft-vectorized core. It serves both MVU-packed
 *	 (IS_MVU=1) and all VVU (IS_MVU=0), which is DSP58-packed only.
 *****************************************************************************/

module mvu_vvu_axi_packed
`include "mvu_vvu_axi_base_head.svh"

		//- DSP58 INT8-packed compute core (was genINT8 in the fused wrapper) ---
		mvu_vvu_8sx9_dsp58 #(
			.IS_MVU(IS_MVU),
			.PE(PE), .SIMD(DSP_SIMD),
			.WEIGHT_WIDTH(WEIGHT_WIDTH), .ACTIVATION_WIDTH(ACTIVATION_WIDTH), .ACCU_WIDTH(ACCU_WIDTH),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
			.SEGMENTLEN(SEGMENTLEN),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) core (
			.clk(PUMPED_COMPUTE? ap_clk2x : ap_clk), .rst, .en('1),
			.last(dsp_last), .zero(dsp_zero), .w(dsp_w), .a(dsp_a),
			.vld(dsp_vld), .p(dsp_p)
		);

`include "mvu_vvu_axi_base_tail.svh"

endmodule : mvu_vvu_axi_packed
