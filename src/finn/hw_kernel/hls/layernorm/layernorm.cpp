// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "layernorm.hpp"
#include "ln_utils.hpp"

constexpr unsigned SIMD = 4;
constexpr unsigned W = 384;
constexpr float epsilon = 1e-5;
using  TI = float;
using  TO = float;

void layernorm(
	hls::stream<hls::vector<TI,SIMD>> &src,
	hls::stream<hls::vector<TO,SIMD>> &dst
) {
#pragma HLS interface AXIS port=src
#pragma HLS interface AXIS port=dst
#pragma HLS aggregate variable=src compact=bit
#pragma HLS aggregate variable=dst compact=bit

#pragma HLS interface ap_ctrl_none port=return
#pragma HLS dataflow disable_start_propagation

	layernorm_pipeline<TI, TO, W, SIMD>(epsilon, src, dst);

} 
