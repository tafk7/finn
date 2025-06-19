// Legacy HLS backend: Thresholding_hls
// Generated code from populated code_gen_dict
// Keys found: ['$DEFINES$', '$READNPYDATA$', '$STREAMDECLARATIONS$', '$DOCOMPUTE$', '$PRAGMAS$', '$DATAOUTSTREAM$', '$SAVEASCNPY$', '$BLACKBOXFUNCTION$']


// === Defines ($DEFINES$) ===
#define NumChannels1 4
 #define PE1 2
 #define numReps 1

               #define ImgDim1 1
#define ActVal1 0
#define ThresType1 ap_int<8>
#define NumSteps1 3

// === HLS Pragmas ($PRAGMAS$) ===
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out0_V
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=in1_V

// === Stream declarations ($STREAMDECLARATIONS$) ===
hls::stream<ap_uint<16>> in0_V ("in0_V");
hls::stream<ap_uint<4>> out0_V ("out0_V");
hls::stream<ap_uint<48>> in1_V ("in1_V");

// === Read NPY data ($READNPYDATA$) ===
npy2apintstream<ap_uint<16>, ap_int<8>, 8, float>("/input_0.npy", in0_V, false);
npy2apintstream<ap_uint<48>, ap_int<8>, 8, float>("/thresholds.npy", in1_V, false, ImgDim1);

// === Compute function ($DOCOMPUTE$) ===
Thresholding_Stream_Batch<ImgDim1, NumChannels1, PE1, Slice<ap_int<8>>, Slice<ap_uint<2>>, ActVal1, ThresType1, NumSteps1>
                (in0_V, out0_V, in1_V, numReps);

// === Data output ($DATAOUTSTREAM$) ===
apintstream2npy<ap_uint<4>, ap_uint<2>, 2, float>(out0_V, {1, 2, 2}, "/output_0.npy", false);

// === Blackbox function ($BLACKBOXFUNCTION$) ===
void (hls::stream<ap_uint<16>> &in0_V,
                    hls::stream<ap_uint<48>> &in1_V,
                    hls::stream<ap_uint<4>> &out0_V
                    )
