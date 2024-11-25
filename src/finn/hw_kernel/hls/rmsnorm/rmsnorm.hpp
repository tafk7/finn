// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef RMSNORM_HPP
#define RMSNORM_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>
#include "rn_utils.hpp"


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
	const T epsilon,
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

template<typename TI, // Input type
         typename TO, // Output type
         unsigned N, 
         unsigned SIMD>
void rmsnorm_pipeline(
    const TO epsilon,
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, SIMD>> &dst
) {
#pragma HLS DATAFLOW disable_start_propagation

	static hls::stream<hls::vector<TI, SIMD>> stage1_s;
#pragma HLS stream variable=stage1_s depth=N
	static hls::stream<TO> mean_s;
#pragma HLS stream variable=mean_s depth=2

	square_mean_stage<TI, TO, N, SIMD>(src, stage1_s, mean_s);
	inv_sqrt_stage<TO, N, SIMD>(epsilon, stage1_s, dst, mean_s);
}

#endif
