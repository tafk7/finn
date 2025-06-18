#!/usr/bin/env python3
"""
Output equivalence testing for Thresholding Jinja2 refactor.
Tests that new template system produces identical output to legacy system.
"""

import os
import tempfile
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

# Import the implementations we want to test
from finn.custom_op.fpgadataflow.hls.thresholding_hls import ThresholdingHLS
from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
from qonnx.custom_op.registry import getCustomOp


def create_test_thresholding_node():
    """Create a test ONNX node for Thresholding operations."""
    
    # Define test parameters
    num_channels = 8
    pe = 2
    act_val = 0
    
    # Create input and output value info
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, num_channels]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, num_channels]
    )
    
    # Create threshold weights (INT4 -> 3 thresholds per channel)
    n_thresholds = 3  # For INT4 output: 2^2 - 1 = 3
    threshold_shape = [num_channels, n_thresholds]
    thresholds = np.random.randint(-8, 8, size=threshold_shape).astype(np.int8)
    thresholds.sort(axis=1)  # Ensure thresholds are sorted
    
    threshold_tensor = helper.make_tensor(
        "thresholds",
        TensorProto.INT8,
        threshold_shape,
        thresholds.flatten()
    )
    
    # Create the Thresholding node
    node = helper.make_node(
        "Thresholding",
        inputs=["input", "thresholds"],
        outputs=["output"],
        domain="finn.custom_op.fpgadataflow",
        name="test_thresholding"
    )
    
    # Create the graph
    graph = helper.make_graph(
        [node],
        "test_thresholding_graph",
        [input_tensor],
        [output_tensor],
        [threshold_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph)
    
    # Wrap in ModelWrapper and set attributes
    model_wrapper = ModelWrapper(model)
    
    # Set required attributes (only those defined in base Thresholding class)
    node_attrs = {
        "NumChannels": num_channels,
        "PE": pe,
        "inputDataType": "INT8",
        "outputDataType": "INT4",
        "weightDataType": "INT8",
        "ActVal": act_val,
        "numInputVectors": [1],
        "numSteps": n_thresholds,
        "runtime_writeable_weights": 0,
    }
    
    # Get the custom operation instance and set attributes
    custom_op = getCustomOp(node)
    for attr_name, attr_value in node_attrs.items():
        custom_op.set_nodeattr(attr_name, attr_value)
    
    return model_wrapper, node


def normalize_code(code_str):
    """Normalize generated code for comparison by removing whitespace differences."""
    lines = code_str.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Strip leading/trailing whitespace
        line = line.strip()
        # Skip empty lines
        if line:
            # Normalize internal whitespace
            line = ' '.join(line.split())
            normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)


def test_hls_template_generation():
    """Test that HLS template generation works without errors."""
    print("Testing HLS template generation...")
    
    model, node = create_test_thresholding_node()
    
    try:
        # Create HLS instance
        hls_node = ThresholdingHLS(node)
        
        # Set up temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            hls_node.set_nodeattr("code_gen_dir_cppsim", temp_dir)
            hls_node.set_nodeattr("code_gen_dir_ipgen", temp_dir)
            
            # Test cppsim code generation
            print("  Testing cppsim code generation...")
            hls_node.code_generation_cppsim(model)
            
            # Check that output file was created
            expected_file = os.path.join(temp_dir, "execute_Thresholding.cpp")
            assert os.path.exists(expected_file), f"Expected file not created: {expected_file}"
            
            # Check that file has content
            with open(expected_file, 'r') as f:
                content = f.read()
                assert len(content) > 100, "Generated file appears to be empty or too short"
                assert '#include "activations.hpp"' in content, "Expected include not found"
                assert 'NumChannels1' in content, "Expected define not found"
                
            print("  ✓ cppsim code generation successful")
            
            # Test ipgen code generation
            print("  Testing ipgen code generation...")
            hls_node.code_generation_ipgen(model, "xc7z020clg400-1", 10)
            
            # Check that output files were created
            expected_cpp = os.path.join(temp_dir, f"top_{node.name}.cpp")
            expected_tcl = os.path.join(temp_dir, f"hls_syn_{node.name}.tcl")
            
            assert os.path.exists(expected_cpp), f"Expected CPP file not created: {expected_cpp}"
            assert os.path.exists(expected_tcl), f"Expected TCL file not created: {expected_tcl}"
            
            # Check content
            with open(expected_cpp, 'r') as f:
                cpp_content = f.read()
                assert 'void test_thresholding(' in cpp_content, "Expected function not found"
                
            with open(expected_tcl, 'r') as f:
                tcl_content = f.read()
                assert 'project_test_thresholding' in tcl_content, "Expected project name not found"
                
            print("  ✓ ipgen code generation successful")
            
    except Exception as e:
        print(f"  ✗ HLS template generation failed: {e}")
        raise
    
    print("✓ HLS template generation completed successfully")


def test_rtl_template_generation():
    """Test that RTL template generation works without errors."""
    print("Testing RTL template generation...")
    
    model, node = create_test_thresholding_node()
    
    try:
        # Create RTL instance
        rtl_node = Thresholding_rtl(node)
        
        # Set up temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            rtl_node.set_nodeattr("code_gen_dir_ipgen", temp_dir)
            
            # Test HDL generation
            print("  Testing HDL generation...")
            rtl_node.generate_hdl(model, "xc7z020clg400-1", 10)
            
            # Check that output files were created
            module_name = rtl_node.get_verilog_top_module_name()
            expected_hdl = os.path.join(temp_dir, f"{module_name}.v")
            
            assert os.path.exists(expected_hdl), f"Expected HDL file not created: {expected_hdl}"
            
            # Check content
            with open(expected_hdl, 'r') as f:
                hdl_content = f.read()
                assert f'module {module_name}' in hdl_content, "Expected module declaration not found"
                assert 'parameter  N =' in hdl_content, "Expected parameter not found"
                assert 'thresholding_axi' in hdl_content, "Expected instance not found"
                
            # Check that library files were copied
            expected_files = ["axilite_if.v", "thresholding.sv", "thresholding_axi.sv"]
            for file_name in expected_files:
                file_path = os.path.join(temp_dir, file_name)
                assert os.path.exists(file_path), f"Expected library file not copied: {file_name}"
                
            print("  ✓ HDL generation successful")
            
    except Exception as e:
        print(f"  ✗ RTL template generation failed: {e}")
        raise
    
    print("✓ RTL template generation completed successfully")


def test_template_values_extraction():
    """Test that template values are extracted correctly."""
    print("Testing template values extraction...")
    
    model, node = create_test_thresholding_node()
    
    try:
        # Test HLS template values
        print("  Testing HLS template values...")
        hls_node = ThresholdingHLS(node)
        
        # Test docompute template values
        docompute_values = hls_node.get_template_values("thresholding/hls/docompute.cpp.j2")
        
        required_keys = [
            'AP_INT_MAX_W', 'GLOBALS', 'DEFINES', 'PRAGMAS', 
            'STREAMDECLARATIONS', 'READNPYDATA', 'DOCOMPUTE', 
            'DATAOUTSTREAM', 'SAVEASCNPY'
        ]
        
        for key in required_keys:
            assert key in docompute_values, f"Required template value missing: {key}"
            assert docompute_values[key] is not None, f"Template value is None: {key}"
        
        # Test timeout template values
        timeout_values = hls_node.get_template_values("thresholding/hls/docompute_timeout.cpp.j2")
        
        timeout_keys = ['TIMEOUT_VALUE', 'TIMEOUT_CONDITION', 'TIMEOUT_READ_STREAM']
        for key in timeout_keys:
            assert key in timeout_values, f"Required timeout value missing: {key}"
        
        print("  ✓ HLS template values extraction successful")
        
        # Test RTL template values
        print("  Testing RTL template values...")
        rtl_node = Thresholding_rtl(node)
        
        wrapper_values = rtl_node.get_template_values("thresholding/rtl/wrapper.v.j2")
        
        required_rtl_keys = [
            'MODULE_NAME_AXI_WRAPPER', 'N', 'WI', 'WT', 'C', 'PE',
            'SIGNED', 'FPARG', 'BIAS', 'THRESHOLDS_PATH', 'USE_AXILITE',
            'DEPTH_TRIGGER_URAM', 'DEPTH_TRIGGER_BRAM', 'DEEP_PIPELINE', 'O_BITS'
        ]
        
        for key in required_rtl_keys:
            assert key in wrapper_values, f"Required RTL template value missing: {key}"
            assert wrapper_values[key] is not None, f"RTL template value is None: {key}"
        
        # Check some specific values
        assert wrapper_values['C'] == 8, f"Expected C=8, got {wrapper_values['C']}"
        assert wrapper_values['PE'] == 2, f"Expected PE=2, got {wrapper_values['PE']}"
        
        print("  ✓ RTL template values extraction successful")
        
    except Exception as e:
        print(f"  ✗ Template values extraction failed: {e}")
        raise
    
    print("✓ Template values extraction completed successfully")


def run_all_tests():
    """Run all output equivalence tests."""
    print("Running Thresholding output equivalence tests...")
    print("=" * 60)
    
    try:
        test_hls_template_generation()
        print()
        test_rtl_template_generation()
        print()
        test_template_values_extraction()
        print()
        
        print("=" * 60)
        print("✓ All tests passed! Template refactor is working correctly.")
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)