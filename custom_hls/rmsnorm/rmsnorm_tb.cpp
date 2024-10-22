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
#include "rmsnorm.hpp"
#include <cstdlib>
#include <ctime>

// how much to test
constexpr unsigned ROUNDS = 3;
constexpr unsigned W = 128;

// Function to calculate square-mean
float sq_mean(float* array, int size) {  
    float sum = 0;  
    for (int i = 0; i < size; i++) {  
        sum += array[i]*array[i];
    }  
    return sum / size;  
}  
  
void ref_rmsnorm(float* input, float* output, int length) {  
    float mean_val = sq_mean(input, length);  
    for (int i = 0; i < length; i++) {  
        output[i] = input[i] / sqrt(mean_val + 1e-5);  
    }  
}  

bool closeEnough(double num1, double num2, double tolerance) {  
    return std::abs(num1 - num2) <= tolerance;  
}

void rmsnorm(
    //hls::stream<hls::vector<ap_int<${TL_Activation_width}>, ${p1_RMSnorm_0_PE}>> &src,
	// hls::stream<hls::vector<float, ${p1_RMSnorm_0_PE}>> &dst
	hls::stream<hls::vector<ap_int<4>, 32>> &src,
	hls::stream<hls::vector<float, 32>> &dst
);

template<unsigned W, unsigned SIMD, typename T>
bool test() {
	hls::stream<hls::vector<T,SIMD>> src;
	hls::stream<hls::vector<float,SIMD>> dst;

	// Reference input and output
	float ref_in[W*ROUNDS];
	float ref_out[W*ROUNDS];

	// Create the input stream (and test stream)
	std::srand(static_cast<unsigned int>(std::time(0)));
	T ref_val = 0;
	for(unsigned r=0; r<ROUNDS; r++){
		for (unsigned i=0; i<W; i+=SIMD) {
			hls::vector<T, SIMD> t;
			for(unsigned j=0; j<SIMD; j++) {
				ref_val = i + j;
				t[j] = ref_val;
				ref_in[i+j] = float(ref_val);
			}
			src.write(t);
		}
	}
	ref_rmsnorm(ref_in, ref_out, W);

	bool ok = true;
	for(unsigned i=0; i < 3*(ROUNDS*W); i++) {
		rmsnorm(src, dst);
	}
	unsigned out_count=0;
	std::cout << "----- Results from CSim ------ \n";
	unsigned total = 0;

	unsigned timeout=0;

	while(timeout < W*ROUNDS) { 
		while(!dst.empty()) 
		{
			hls::vector<float, SIMD> y = dst.read();
			for (unsigned j=0; j<SIMD; j++) {
				if (!closeEnough(y[j],ref_out[out_count], 1e-5)) {
					std::cout << "Error: "  << y[j] << " !=  " << ref_out[out_count] << "\n";
					ok = false;
				}
				out_count = (out_count + 1) % W;
				total++;
			}
		}
		timeout++;
	}
	std::cout << "Total of data items at the output: "<< total <<"\n";

	if(total != W*ROUNDS )
		ok = false;

	return ok;
}

int main() {
	
	// bool ok = test<W, ${p1_RMSnorm_0_PE}, ap_int<${TL_Activation_width}>>();
    bool ok = test<W, 32, ap_int<4>>();
	if (ok) {
		std::cout << "Test completed okay\n";
		return 0;
	} else {
		std::cout << "Test failed\n";
		return 1;
	}

}
