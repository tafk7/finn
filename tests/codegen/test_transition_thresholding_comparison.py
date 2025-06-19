#!/usr/bin/env python3
"""Test script comparing legacy, clean, and transition HLS implementations."""

import os
import tempfile
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor

# Import the three implementations
from finn.custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
from finn.custom_op.fpgadataflow.hls.transition_thresholding_hls import CG_Thresholding_hls_Full


def create_test_model():
    """Create a simple test model with thresholding operation."""
    import onnx
    from onnx import helper, TensorProto
    
    # Model parameters
    input_shape = [1, 4]  # Batch size 1, 4 channels
    output_shape = [1, 4]
    
    # Create input/output value infos
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    
    # Create threshold tensor (4 channels, 3 thresholds each for 2-bit output)
    thresholds = np.array([
        [-2.0, 0.0, 2.0],  # Channel 0
        [-1.5, 0.5, 1.5],  # Channel 1  
        [-1.0, 1.0, 3.0],  # Channel 2
        [-0.5, 0.0, 0.5]   # Channel 3
    ], dtype=np.float32)
    
    # Create threshold initializer
    thresh_tensor = helper.make_tensor(
        "thresholds", TensorProto.FLOAT, [4, 3], thresholds.flatten()
    )
    
    # Create thresholding node
    thresh_node = helper.make_node(
        "Thresholding",
        inputs=["input", "thresholds"],
        outputs=["output"],
        domain="finn.custom_op.fpgadataflow",
        # Node attributes
        PE=2,
        NumChannels=4,
        numSteps=3,
        inputDataType="INT8",
        outputDataType="UINT2",
        weightDataType="INT8",  # Add missing attribute
        numInputVectors=[1],
        ActVal=0
    )
    
    # Create graph and model
    graph = helper.make_graph(
        [thresh_node], "thresholding_test", [input_vi], [output_vi], [thresh_tensor]
    )
    
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 11
    
    return ModelWrapper(model)


def test_template_generation():
    """Test that all three implementations can generate template values."""
    print("Transition Thresholding HLS Implementation Test")
    print("=" * 50)
    
    # Create test model
    model = create_test_model()
    thresh_node = model.graph.node[0]
    
    print("✓ Model created with thresholding node")
    
    # Test each implementation
    results = {}
    
    implementations = [
        ("Legacy", Thresholding_hls),
        ("Clean", CG_Thresholding_hls), 
        ("Transition", CG_Thresholding_hls_Full)
    ]
    
    for name, impl_class in implementations:
        print(f"\n=== {name.upper()} IMPLEMENTATION ===")
        
        try:
            # Create backend instance
            backend = impl_class(thresh_node)
            print(f"✓ {name} backend created: {backend.__class__.__name__}")
            
            # Test node attributes
            attrs = backend.get_nodeattr_types()
            print(f"✓ Node attributes: {len(attrs)} total")
            
            # Test memory mode attribute (should exist in legacy and transition)
            if hasattr(backend, 'get_nodeattr') and name != "Clean":
                try:
                    mem_mode = backend.get_nodeattr('mem_mode')
                    print(f"✓ Memory mode: {mem_mode}")
                except:
                    print("! Memory mode not accessible")
            
            # Test template generation (clean and transition only)
            if hasattr(backend, 'get_template_values'):
                try:
                    template_values = backend.get_template_values("hls_basic.cpp.j2")
                    print(f"✓ Template values generated: {len(template_values)} keys")
                    print(f"  Template keys: {list(template_values.keys())}")
                    
                    # Test specific values
                    if 'mem_mode' in template_values:
                        print(f"  Memory mode in template: {template_values['mem_mode']}")
                    if 'input_ports' in template_values:
                        print(f"  Input ports: {len(template_values['input_ports'])}")
                    if 'output_ports' in template_values:
                        print(f"  Output ports: {len(template_values['output_ports'])}")
                        
                except Exception as e:
                    print(f"✗ Template generation failed: {e}")
                    results[name] = f"Template error: {e}"
                    continue
            else:
                print("! No template generation method (legacy implementation)")
            
            # Test resource estimation (should exist in legacy and transition)
            if hasattr(backend, 'bram_estimation'):
                try:
                    bram_est = backend.bram_estimation()
                    lut_est = backend.lut_estimation()
                    print(f"✓ Resource estimation - BRAM: {bram_est}, LUT: {lut_est}")
                except Exception as e:
                    print(f"✗ Resource estimation failed: {e}")
            else:
                print("! No resource estimation methods")
                
            results[name] = "Success"
            
        except Exception as e:
            print(f"✗ {name} implementation failed: {e}")
            results[name] = f"Error: {e}"
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY:")
    for name, result in results.items():
        status = "✓" if result == "Success" else "✗"
        print(f"{status} {name}: {result}")
    
    # Test transition implementation features
    print(f"\n{'='*50}")
    print("TRANSITION IMPLEMENTATION FEATURES:")
    
    if "Transition" in results and results["Transition"] == "Success":
        backend = CG_Thresholding_hls_Full(thresh_node)
        
        # Test memory modes
        for mem_mode in ["internal_embedded", "internal_decoupled"]:
            print(f"\nTesting memory mode: {mem_mode}")
            backend.set_nodeattr("mem_mode", mem_mode)
            
            try:
                template_values = backend.get_template_values("hls_basic.cpp.j2")
                
                print(f"✓ Memory mode: {template_values['mem_mode']}")
                print(f"✓ Input ports: {len(template_values['input_ports'])}")
                print(f"✓ Includes: {len(template_values['global_includes'])}")
                print(f"✓ Compute body contains: {'embedded' if 'Batch<' in template_values['compute_body'] else 'streaming'}")
                
                # Check for weight stream in decoupled mode
                if mem_mode == "internal_decoupled":
                    weight_port = any(port['name'] == 'in1_V' for port in template_values['input_ports'])
                    print(f"✓ Weight stream port: {'present' if weight_port else 'missing'}")
                
            except Exception as e:
                print(f"✗ Memory mode {mem_mode} failed: {e}")


if __name__ == "__main__":
    test_template_generation()