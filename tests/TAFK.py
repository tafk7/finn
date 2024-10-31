import fpgadataflow.test_fpgadataflow_layernorm as t
import transformation.test_batchnorm_to_affine_bnn_pynq as tb
# ["cppsim", "rtlsim", "stitched_ip"]
# ["simd1", "simd2", "simd3", "simd4"]
exec_mode = "cppsim"
impl_style = "hls"
simd = "simd1"
#idt = ["FLOAT16"]
idt = "FLOAT16"
wdt = "FLOAT16"
bdt = "FLOAT16"
odt = "FLOAT16"
ifm_dim = (1, 1, 128, 384)
# t.test_fpga_dataflow_layernorm(impl_style, simd, idt, wdt, bdt, odt, ifm_dim)
# t.test_convert_to_hw_layernorm_layer(exec_mode, simd)

tb.test_batchnorm_to_affine_cnv_w1a1()