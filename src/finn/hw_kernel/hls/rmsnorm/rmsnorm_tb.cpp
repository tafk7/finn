// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rmsnorm.hpp"
#include <cstdlib>
#include <ctime>
#include "_eltwise_affine.hpp"

// how much to test
constexpr unsigned ROUNDS = 3;
constexpr unsigned W = 384;
constexpr unsigned SIMD = 4;
using TI = float; 
using TO = float; 

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
        output[i] = (input[i] / sqrt(mean_val + 1e-5)) * _eltwise_affine[int(i/4)][i%4];  
    }  
}  

bool closeEnough(double num1, double num2, double tolerance) {  
    return std::abs(num1 - num2) <= tolerance;  
}

void rmsnorm(
	hls::stream<hls::vector<TI, SIMD>> &src,
	hls::stream<hls::vector<TO, SIMD>> &dst
);

template<typename TI, typename TO, unsigned W, unsigned SIMD>
bool test() {
	hls::stream<hls::vector<TI,SIMD>> src;
	hls::stream<hls::vector<TO,SIMD>> dst;

	// Reference input and output
	float ref_in[W*ROUNDS];
	float ref_out[W*ROUNDS];

	// Create the input stream (and test stream)
	std::srand(static_cast<unsigned int>(std::time(0)));
	TI ref_val = TI(0);
	for(unsigned r=0; r<ROUNDS; r++){
		for (unsigned i=0; i<W; i+=SIMD) {
			hls::vector<TI, SIMD> t;
			for(unsigned j=0; j<SIMD; j++) {
				ref_val = i + j;
				t[j] = TI(ref_val);
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
				if (!closeEnough(y[j],ref_out[out_count], 1e-4)) {
					std::cout << "Error: "  << y[j] << " !=  " << ref_out[out_count] << "\n";
					ok = false;
				} else {
					std::cout << "OK: " << y[j] << " == " << ref_out[out_count] << "\n";
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
	
    bool ok = test<TI, TO,W, SIMD>();
	if (ok) {
		std::cout << "Test completed okay\n";
		return 0;
	} else {
		std::cout << "Test failed\n";
		return 1;
	}

}
