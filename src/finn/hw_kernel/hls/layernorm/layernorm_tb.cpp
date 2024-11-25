// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "layernorm.hpp"
#include <cstdlib>
#include <ctime>

// how much to test
constexpr unsigned ROUNDS = 3;
constexpr unsigned W = 384;
constexpr float epsilon = 1e-5;

// Function to calculate mean  
float mean(float* array, int size) {  
    float sum = 0;  
    for (int i = 0; i < size; i++) {  
        sum += array[i];  
    }  
    return sum / size;  
}  
  
// Function to calculate variance  
float variance(float* array, int size, float mean) {  
    float sum = 0;  
    for (int i = 0; i < size; i++) {  
        sum += pow(array[i] - mean, 2);  
    }  
    return sum / size;  
}  
  

void ref_layernorm(float* input, float* output, int length) {  
    float mean_val = mean(input, length);  
    float variance_val = variance(input, length, mean_val);  
  
    for (int i = 0; i < length; i++) {  
        output[i] = (input[i] - mean_val) / sqrt(variance_val + 1e-5);  
    }  
}  

bool closeEnough(double num1, double num2, double tolerance) {  
    return std::abs(num1 - num2) <= tolerance;  
}

void layernorm(
	hls::stream<hls::vector<float, 4>> &src,
	hls::stream<hls::vector<float, 4>> &dst
);

template<typename T, unsigned W, unsigned SIMD>
bool test() {
	hls::stream<hls::vector<float,SIMD>> src;
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
	ref_layernorm(ref_in, ref_out, W);

	bool ok = true;
	for(unsigned i=0; i < 3*(ROUNDS*W); i++) {
		layernorm(src, dst);
	}
	unsigned out_count=0;
	std::cout << "----- Results from Sim ------ \n";
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
	
	bool ok = test<float, W, 4>();
	if (ok) {
		std::cout << "Test completed okay\n";
		return 0;
	} else {
		std::cout << "Test failed\n";
		return 1;
	}

}
