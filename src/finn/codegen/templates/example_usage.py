#!/usr/bin/env python3
"""
Example usage of simplified templates with FINN codegen.
This demonstrates how to use the new simplified templates for both HLS and RTL.
"""

from finn.codegen.codegen import Codegen
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from typing import Dict, Any


class SimpleThresholdingOp(HWCustomOp):
    """Example custom operation using simplified templates."""
    
    def get_nodeattr_types(self):
        return {
            "NumChannels": ("i", True, 0),
            "PE": ("i", True, 1),
            "NumSteps": ("i", True, 1),
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        }


class CG_SimpleThresholding_hls(SimpleThresholdingOp, Codegen):
    """Clean HLS backend using simplified template."""
    
    # Use the basic HLS template
    TEMPLATE_NAME = "hls_basic.cpp.j2"
    
    def get_template_name(self) -> str:
        return self.TEMPLATE_NAME
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate values for basic HLS template."""
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        
        return {
            # Basic information
            'function_name': f'{self.onnx_node.name}_compute',
            'ap_int_max_w': 512,
            
            # Includes and defines
            'global_includes': [
                '#include "thresholds.h"',
                '#include "activation.h"'
            ],
            'defines': [
                f'#define NUM_CHANNELS {num_channels}',
                f'#define PE {pe}',
                f'#define FOLD_FACTOR {num_channels // pe}'
            ],
            
            # Ports
            'input_ports': [
                {'name': 'in_stream', 'type': 'ap_uint<8>'}
            ],
            'output_ports': [
                {'name': 'out_stream', 'type': 'ap_uint<4>'}
            ],
            
            # HLS pragmas
            'pragmas': [
                '#pragma HLS INTERFACE axis port=in_stream',
                '#pragma HLS INTERFACE axis port=out_stream',
                '#pragma HLS INTERFACE ap_ctrl_none port=return',
                '#pragma HLS PIPELINE II=1'
            ],
            
            # Main computation
            'compute_body': self._generate_compute_body(),
            
            # Optional wrapper
            'include_wrapper': True,
            'top_function_name': self.onnx_node.name,
            'input_width': 8,
            'output_width': 4
        }
    
    def _generate_compute_body(self) -> str:
        """Generate the main computation code."""
        return """// Threshold computation
for (int fold = 0; fold < FOLD_FACTOR; fold++) {
    #pragma HLS PIPELINE II=PE
    
    for (int p = 0; p < PE; p++) {
        #pragma HLS UNROLL
        
        auto in_val = in_stream.read();
        auto out_val = apply_thresholds(in_val, thresholds[fold][p]);
        out_stream.write(out_val);
    }
}"""


class CG_SimpleThresholding_rtl(SimpleThresholdingOp, Codegen):
    """Clean RTL backend using simplified template."""
    
    # Use the thresholding RTL template
    TEMPLATE_NAME = "thresholding_rtl.v.j2"
    
    def get_template_name(self) -> str:
        return self.TEMPLATE_NAME
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate values for RTL template."""
        c = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        
        # Calculate stream widths
        input_width = pe * 8  # 8-bit inputs
        output_width = pe * 4  # 4-bit outputs
        
        # Pad to byte boundaries
        input_stream_width = ((input_width + 7) // 8) * 8
        output_stream_width = ((output_width + 7) // 8) * 8
        
        return {
            # Module naming
            'module_name': self.get_verilog_top_module_name(),
            
            # Core parameters
            'n': 4,    # output bits
            'wi': 8,   # input bits
            'wt': 16,  # threshold bits
            'c': c,
            'pe': pe,
            
            # Operation parameters
            'signed': 0,
            'fparg': 0,
            'bias': 0,
            
            # Paths and config
            'thresholds_path': f'"{self.get_nodeattr("code_gen_dir_ipgen")}/thresholds"',
            'use_axilite': 0,  # No runtime reconfig for now
            'depth_trigger_uram': 0,
            'depth_trigger_bram': 0,
            'deep_pipeline': 1,
            
            # Derived parameters
            'o_bits': 4,
            'axilite_addr_width': 14,  # Not used when use_axilite=0
            'input_stream_width': input_stream_width,
            'output_stream_width': output_stream_width
        }


# Example of using the improved template for more complex operations
class CG_MVAU_hls(HWCustomOp, Codegen):
    """Example of using the improved HLS template for MVAU."""
    
    TEMPLATE_NAME = "hls_improved.cpp.j2"
    
    def get_template_name(self) -> str:
        return self.TEMPLATE_NAME
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate values for improved HLS template."""
        return {
            # Precisions
            'input_precision': 8,
            'output_precision': 32,
            'weight_precision': 8,
            'weight_signed': True,
            'accumulator_precision': 32,
            'acc_signed': True,
            
            # Architecture parameters
            'num_channels': 256,
            'pe': 16,
            'simd': 8,
            'function_name': 'mvau_compute',
            
            # Extra constants
            'extra_constants': [
                {'name': 'MW', 'value': 512},
                {'name': 'MH', 'value': 256}
            ],
            
            # Weight configuration
            'weights_const': True,
            'weight_size': 131072,  # MW * MH
            'weights_file': 'mvau_weights.dat',
            
            # Optimization
            'use_dataflow': True,
            'array_partitions': [
                {'var': 'weights', 'type': 'cyclic', 'factor': 16}
            ],
            
            # Loop structure
            'num_iterations': 256,
            'outer_loop_pipeline': True,
            'outer_loop_ii': 1,
            
            # Computation
            'main_compute_body': self._generate_mvau_compute(),
            
            # Options
            'generate_wrapper': True,
            'top_function_name': 'mvau_top',
            'input_width': 64,
            'output_width': 512,
            'total_inputs': 32768,
            'total_outputs': 512
        }
    
    def _generate_mvau_compute(self) -> str:
        return """// Matrix-Vector-Activation Unit
AccType acc[PE];
#pragma HLS ARRAY_PARTITION variable=acc complete

// Initialize accumulators
for (int p = 0; p < PE; p++) {
    #pragma HLS UNROLL
    acc[p] = 0;
}

// Compute dot products
for (int s = 0; s < MW/SIMD; s++) {
    InputType in_block = in.read();
    
    for (int p = 0; p < PE; p++) {
        #pragma HLS UNROLL
        for (int simd = 0; simd < SIMD; simd++) {
            #pragma HLS UNROLL
            
            InputType in_val = in_block.range((simd+1)*8-1, simd*8);
            WeightType w = weights[i*MW + s*SIMD + simd][p];
            acc[p] += in_val * w;
        }
    }
}

// Apply activation and output
OutputType out_pack = 0;
for (int p = 0; p < PE; p++) {
    #pragma HLS UNROLL
    OutputType activated = activation_function(acc[p]);
    out_pack.range((p+1)*32-1, p*32) = activated;
}
out.write(out_pack);"""


if __name__ == "__main__":
    print("This file demonstrates how to use simplified templates.")
    print("Import these classes in your FINN operations to use the new templates.")
    print("\nKey features:")
    print("- Direct, readable template value generation")
    print("- No complex inheritance or macros")
    print("- Easy to understand and modify")
    print("- Suitable for open source contributors")