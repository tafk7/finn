// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.
#ifndef RMSNORM_HPP
#define RMSNORM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>
#include "utils.hpp"

constexpr float epsilon = 1e-5;

// First pipeline stage
//
// Trigger: Data available on src input stream
//
// Desc: Performs elemwise-square and calculate mean for N elements

template<typename TI, typename TO, unsigned N, unsigned SIMD>
void square_mean_stage(
	hls::stream<hls::vector<TI, SIMD>> &in_s,
	hls::stream<hls::vector<TO, SIMD>> &out_s,
	hls::stream<TO> &mean_s
) {
#pragma HLS pipeline II=1 style=flp

	static ap_uint<clog2(N)> count = 0;
	static TO sum = TO(0.0);
	static TO mean = TO(0.0);
#pragma HLS reset variable=count
#pragma HLS reset variable=sum
#pragma HLS reset variable=mean
	
	if (!in_s.empty()) {
		hls::vector<TI,SIMD> const in = in_s.read();
		hls::vector<TO, SIMD> sq_res;
		hls::vector<TO, SIMD> out;

        // Elemwise square and pass along input
		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
            sq_res[i] = TO(in[i]*in[i]);
			out[i] = TO(in[i]);
		}
		out_s.write(out);

		// Mean calc
		sum += TreeReduction<SIMD>::reduce(sq_res);
		count += SIMD;
		mean = sum / count;

		if (count == N) {
			count = 0;
			mean_s.write(mean); 
			mean = TO(0.0);
			sum = TO(0.0);
		}
	}
}

// Second pipeline stage
//
// Trigger: On data being available on the square-mean value stream 
//
// Desc: Divide by sqroot of square-mean
template<typename T, unsigned N, unsigned SIMD>
void inv_sqrt_stage(
	hls::stream<hls::vector<T, SIMD>> &in_s,
	hls::stream<hls::vector<T, SIMD>> &out_s,
	hls::stream<T> &mean_s
) {
#pragma HLS pipeline II=1 style=flp

	static ap_uint<clog2(N/SIMD)+1> count = 0; 
	static bool valid = false;
	static float sqm;
#pragma HLS reset variable=count
#pragma HLS reset variable=valid
#pragma HLS reset variable=sqm

	if(count == (N/SIMD)) {
		count = 0; 
		valid = false;
		return;
	}

	if (valid && !in_s.empty()) {
		hls::vector<T, SIMD> const in = in_s.read();
		hls::vector<T, SIMD> out;
		for (unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = in[i] / hls::sqrt(sqm + epsilon);  
		}
		out_s.write(out);
		count++;
	}

	if (!mean_s.empty() && !valid) {
		sqm = mean_s.read();
		valid = true;
	}
}

template<typename TI, typename TO, unsigned N, unsigned SIMD>
void rmsnorm_pipeline(
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, SIMD>> &dst
) {
#pragma HLS DATAFLOW disable_start_propagation

	static hls::stream<hls::vector<TI, SIMD>> stage1_s;
#pragma HLS stream variable=stage1_s depth=N
	static hls::stream<TO> mean_s;
#pragma HLS stream variable=mean_s depth=2

	square_mean_stage<TI, TO, N, SIMD>(src, stage1_s, mean_s);
	inv_sqrt_stage<TO, N, SIMD>(stage1_s, dst, mean_s);
}

#endif
