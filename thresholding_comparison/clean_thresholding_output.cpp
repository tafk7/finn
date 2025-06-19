// Clean HLS backend: CG_Thresholding_hls
// Template-based code generation

// Template values keys: ['function_name', 'ap_int_max_w', 'global_includes', 'defines', 'input_ports', 'output_ports', 'pragmas', 'compute_body', 'include_wrapper']

// Template values debug:
// function_name: '_compute'
// ap_int_max_w: 8192
// global_includes: ['#include "activations.hpp"', '#include "params.h"']
// defines: ['#define NumChannels 4', '#define PE 2', '#define numReps 1', '#define numSteps 3', '#define TMEM 2']
// input_ports: [{'name': 'in0_V', 'type': 'ap_uint<16>'}]
// output_ports: [{'name': 'out0_V', 'type': 'ap_uint<4>'}]
// pragmas: ['#pragma HLS INTERFACE axis port=in0_V', '#pragma HLS INTERFACE axis port=out0_V', '#pragma HLS INTERFACE ap_ctrl_none port=return', '#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1', '#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=3']
// compute_body: <150 chars>
// include_wrapper: False

// === Rendered docompute.cpp ===
// Basic HLS Template for FINN Operations
// This is a simplified, self-contained template for HLS code generation
// Variables are directly substituted - no complex macros or includes needed

#define AP_INT_MAX_W 8192
#include <ap_int.h>
#include <hls_stream.h>
#include "bnn-library.h"

// Global includes for parameters, weights, etc.
#include "activations.hpp"
#include "params.h"

// Parameter defines
#define NumChannels 4
#define PE 2
#define numReps 1
#define numSteps 3
#define TMEM 2

// Main compute function
void _compute(
    hls::stream<ap_uint<16>> &in0_V
    ,
    hls::stream<ap_uint<4>> &out0_V
) {
    // HLS pragmas for interface and optimization
    #pragma HLS INTERFACE axis port=in0_V
    #pragma HLS INTERFACE axis port=out0_V
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1
    #pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=3
    
    // Main computation
    // Thresholding computation
    Thresholding_Batch<ap_int<8>, ap_uint<2>, ap_int<8>, NumChannels, PE, 2>(
        in0_V, out0_V, threshs.parameters, numReps);
}

