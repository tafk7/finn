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
 * @brief	Shared output stage for the per-core MVU/VVU AXI-lite wrappers.
 * @details
 *	 `include`d by the per-core wrappers immediately after their compute-core
 *	 instantiation. Closes the `blkDsp` block opened in
 *	 mvu_vvu_axi_base_head.svh and appends the output queue / AXI-Stream output.
 *****************************************************************************/

	end : blkDsp

	if(1) begin : blkOutput
		localparam int unsigned  CORE_PIPELINE_DEPTH =
			VERSION == 3? 3 + (SEGMENTLEN == 0? 0 : ((SIMD+2)/3 -1)/SEGMENTLEN) :
			/* else */    3 + $clog2(SIMD+1) + (SIMD == 1);

		// This is conservative and could be divided by a guaranteed minimum output interval, e.g. MW/SIMD.
		localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH;
		typedef logic [PE-1:0][ACCU_WIDTH-1:0]  output_t;

		logic signed [$clog2(MAX_IN_FLIGHT+1):0]  OPtr = '1;	// -1 | 0, 1, ..., MAX_IN_FLIGHT
		(* SHREG_EXTRACT = "YES" *)
		output_t  OBuf[0:MAX_IN_FLIGHT];
		logic     OVld  =  0;
		output_t  OReg  = 'x;
		logic     OLock =  0;	// Lock upon backpressure (second entry into queue)

		// Catch every output into (SRL) Output Queue
		always_ff @(posedge ap_clk) begin
			if(ovld)  OBuf <= { odat, OBuf[0:MAX_IN_FLIGHT-1] };
		end

		always_ff @(posedge ap_clk) begin
			if(rst) begin
				OPtr  <= '1;
				OVld  <=  0;
				OReg  <= 'x;
				OLock <=  0;
			end
			else begin
				automatic logic  push = ovld;
				automatic logic  pop  = (m_axis_output_tready || !OVld) && !OPtr[$left(OPtr)];
				assert(pop || !push || (OPtr < $signed(MAX_IN_FLIGHT))) else begin
					$error("%m: Overflowing output queue.");
				end
				OPtr <= OPtr + $signed(push == pop? 0 : push? 1 : -1);

				if(OPtr[$left(OPtr)])                   OLock <= 0;
				else if(OVld && !m_axis_output_tready)  OLock <= 1;

				if(m_axis_output_tready || !OVld) begin
					OVld <= !OPtr[$left(OPtr)];
					OReg <= OBuf[OPtr[$left(OPtr)-1:0]];
				end
			end
		end
		assign	idle = OLock;

		assign	m_axis_output_tvalid = OVld;
		// Why would we need a sign extension here potentially creating a higher signal load into the next FIFO?
		// These extra bits should never be used. Why not 'x them out?
		assign	m_axis_output_tdata = { {(OUTPUT_STREAM_WIDTH_BA-OUTPUT_STREAM_WIDTH){OReg[PE-1][ACCU_WIDTH-1]}}, OReg };

	end : blkOutput
