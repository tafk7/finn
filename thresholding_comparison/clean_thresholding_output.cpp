// Clean HLS backend: CG_ThresholdingHLS
// Template-based code generation

// Template values keys: ['NODE_NAME', 'OP_TYPE', 'INSTREAM_WIDTH', 'OUTSTREAM_WIDTH', 'INPUT_HLS_TYPE', 'OUTPUT_HLS_TYPE', 'INPUT_SHAPE', 'OUTPUT_SHAPE', 'HLS_PRAGMAS', 'GLOBAL_INCLUDES', 'PE', 'NUM_CHANNELS', 'NUM_STEPS', 'CODE_GEN_DIR_CPPSIM', 'CODE_GEN_DIR_IPGEN', 'AP_INT_MAX_W', 'INCLUDES', 'PRAGMAS', 'STREAM_DECLARATIONS', 'GLOBALS', 'DEFINES', 'DOCOMPUTE', 'READNPYDATA', 'DATAOUTSTREAM', 'SAVEASCNPY']

// Template values debug:
// NODE_NAME: ''
// OP_TYPE: 'Thresholding_hls'
// INSTREAM_WIDTH: '16'
// OUTSTREAM_WIDTH: '4'
// INPUT_HLS_TYPE: 'ap_int<8>'
// OUTPUT_HLS_TYPE: 'ap_uint<2>'
// INPUT_SHAPE: '(1, 2, 2)'
// OUTPUT_SHAPE: '(1, 2, 2)'
// HLS_PRAGMAS: <265 chars>
// GLOBAL_INCLUDES: '#include "activations.hpp"\n#include "params.h"'
// PE: '2'
// NUM_CHANNELS: '4'
// NUM_STEPS: '3'
// CODE_GEN_DIR_CPPSIM: ''
// CODE_GEN_DIR_IPGEN: ''
// AP_INT_MAX_W: '16'
// INCLUDES: <167 chars>
// PRAGMAS: <265 chars>
// STREAM_DECLARATIONS: 'hls::stream<ap_uint<16>> in0_V ("in0_V");\nhls::stream<ap_uint<4>> out0_V ("out0_V");'
// GLOBALS: '#include "activations.hpp"\n#include "params.h"'
// DEFINES: '#define NumChannels1 4\n#define PE1 2\n#define numReps 1\n#define numSteps 3\n#define TMEM 2'
// DOCOMPUTE: <128 chars>
// READNPYDATA: 'npy2apintstream<ap_uint<16>, ap_int<8>, 8, float>("/input_0.npy", in0_V);'
// DATAOUTSTREAM: 'apintstream2npy<ap_uint<4>, ap_uint<2>, 2, float>(out0_V, {1, 2, 2}, "/output_0.npy");'
// SAVEASCNPY: '// Save functionality handled by dataoutstream'

// === Rendered docompute.cpp ===
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W 16
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
#include "activations.hpp"
#include "params.h"

// defines for network parameters
#define NumChannels1 4
#define PE1 2
#define numReps 1
#define numSteps 3
#define TMEM 2

int main(){
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out0_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=3



npy2apintstream<ap_uint<16>, ap_int<8>, 8, float>("/input_0.npy", in0_V);

Thresholding_Batch<ap_int<8>, ap_uint<2>, ap_int<8>, NumChannels1, PE1, 2>
        (in0_V, out0_V, threshs.parameters, numReps);

apintstream2npy<ap_uint<4>, ap_uint<2>, 2, float>(out0_V, {1, 2, 2}, "/output_0.npy");

// Save functionality handled by dataoutstream

}