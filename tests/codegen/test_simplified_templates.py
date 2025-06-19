#!/usr/bin/env python3
"""Test simplified templates for FINN codegen."""

import os
import sys
import pytest

# Set and add FINN root to path
finn_root = '/home/tafk/dev/tafk-finn-1'
os.environ['FINN_ROOT'] = finn_root
sys.path.insert(0, finn_root)

from finn.codegen.template_engine import TemplateEngine


def test_hls_basic_template():
    """Test basic HLS template rendering."""
    engine = TemplateEngine()
    
    # Define simple template values
    values = {
        'function_name': 'threshold_compute',
        'ap_int_max_w': 1024,
        'global_includes': ['#include "weights.h"'],
        'defines': ['#define NUM_THRESHOLDS 16', '#define PE 4'],
        'input_ports': [
            {'name': 'in_stream', 'type': 'ap_uint<8>'}
        ],
        'output_ports': [
            {'name': 'out_stream', 'type': 'ap_uint<4>'}
        ],
        'pragmas': [
            '#pragma HLS INTERFACE axis port=in_stream',
            '#pragma HLS INTERFACE axis port=out_stream',
            '#pragma HLS PIPELINE II=1'
        ],
        'compute_body': '''// Simple thresholding
for (int i = 0; i < NUM_ITERATIONS; i++) {
    auto val = in_stream.read();
    auto result = threshold_function(val);
    out_stream.write(result);
}''',
        'include_wrapper': True,
        'top_function_name': 'threshold_top',
        'input_width': 8,
        'output_width': 4
    }
    
    # Render template
    result = engine.render('hls_basic.cpp.j2', values)
    
    # Verify key elements are present
    assert 'threshold_compute' in result
    assert '#define NUM_THRESHOLDS 16' in result
    assert 'ap_uint<8>' in result
    assert 'ap_uint<4>' in result
    assert '#pragma HLS PIPELINE II=1' in result
    assert 'threshold_top' in result
    
    print("Basic HLS template test passed!")
    return result


def test_thresholding_rtl_template():
    """Test thresholding RTL template rendering."""
    engine = TemplateEngine()
    
    # Define RTL template values
    values = {
        'module_name': 'Thresholding_rtl_0',
        'n': 4,
        'wi': 8,
        'wt': 16,
        'c': 128,
        'pe': 4,
        'signed': 0,
        'fparg': 0,
        'bias': 0,
        'thresholds_path': './thresholds',
        'use_axilite': 0,
        'depth_trigger_uram': 0,
        'depth_trigger_bram': 0,
        'deep_pipeline': 1,
        'o_bits': 4,
        'axilite_addr_width': 14,
        'input_stream_width': 32,
        'output_stream_width': 16
    }
    
    # Render template
    result = engine.render('thresholding_rtl.v.j2', values)
    
    # Verify key elements
    assert 'module Thresholding_rtl_0' in result
    assert 'parameter  N   = 4' in result
    assert 'parameter  PE  = 4' in result
    assert 'thresholding_axi #(' in result
    assert 'in0_V_TDATA' in result
    assert 'out0_V_TDATA' in result
    
    print("Thresholding RTL template test passed!")
    return result


def test_hls_improved_template():
    """Test improved HLS template rendering."""
    engine = TemplateEngine()
    
    # Define improved template values
    values = {
        'ap_int_max_w': 4096,
        'function_name': 'mvau_compute',
        'input_precision': 8,
        'output_precision': 32,
        'weight_precision': 8,
        'weight_signed': True,
        'accumulator_precision': 32,
        'acc_signed': True,
        'num_channels': 256,
        'pe': 16,
        'simd': 8,
        'extra_constants': [
            {'name': 'TILE_SIZE', 'value': 64},
            {'name': 'BUFFER_DEPTH', 'value': 512}
        ],
        'weights_const': True,
        'weight_size': 2048,
        'weights_file': 'mvau_weights.h',
        'use_dataflow': True,
        'array_partitions': [
            {'var': 'weights', 'type': 'cyclic', 'factor': 16},
            {'var': 'buffer', 'type': 'complete', 'factor': 0}
        ],
        'num_iterations': 1024,
        'outer_loop_pipeline': True,
        'outer_loop_ii': 1,
        'main_compute_body': '''// MVAU computation
AccType acc = 0;
for (int j = 0; j < SIMD; j++) {
    #pragma HLS UNROLL
    InputType in_val = in.read();
    WeightType w = weights[i*SIMD + j];
    acc += in_val * w;
}
OutputType result = activation(acc);
out.write(result);''',
        'generate_wrapper': False,
        'generate_testbench': False
    }
    
    # Render template
    result = engine.render('hls_improved.cpp.j2', values)
    
    # Verify key elements
    assert 'mvau_compute' in result
    assert 'typedef ap_int<8> WeightType' in result
    assert 'const unsigned int PE = 16' in result
    assert '#pragma HLS DATAFLOW' in result
    assert '#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16' in result
    assert 'MVAU computation' in result
    
    print("Improved HLS template test passed!")
    return result


def test_all_templates():
    """Test all simplified templates."""
    print("\nTesting simplified FINN templates...")
    print("=" * 60)
    
    # Test each template
    hls_basic = test_hls_basic_template()
    print("-" * 60)
    
    rtl = test_thresholding_rtl_template()  
    print("-" * 60)
    
    hls_improved = test_hls_improved_template()
    print("-" * 60)
    
    print("\nAll template tests passed! ✓")
    print(f"Generated {len(hls_basic)} chars of basic HLS code")
    print(f"Generated {len(rtl)} chars of RTL code")
    print(f"Generated {len(hls_improved)} chars of improved HLS code")


if __name__ == "__main__":
    test_all_templates()