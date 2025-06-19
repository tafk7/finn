#!/usr/bin/env python3
"""
Generate actual Thresholding code outputs from legacy backend.

This script creates a proper FINN model with Thresholding and generates
the actual C++ code that would be synthesized.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add FINN to path
finn_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(finn_root))

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from onnx import TensorProto, helper
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths


def create_thresholding_model():
    """Create a Thresholding model for code generation."""
    
    # Model parameters
    n_chans = 8
    pe = 2  # Parallelism
    n_steps = 3  # Number of threshold levels
    idt = DataType["INT8"]
    odt = DataType["UINT2"]  # 2-bit output for 3 thresholds
    
    # Create ONNX graph
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, n_chans])
    outp = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, [1, n_chans])
    
    # Create Thresholding node with detailed attributes
    thresh_node = helper.make_node(
        "Thresholding",
        ["global_in"],
        ["global_out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        preferred_impl_style="hls",
        NumChannels=n_chans,
        PE=pe,
        numSteps=n_steps,
        inputDataType=idt.name,
        outputDataType=odt.name,
        weightDataType="INT8",
        ActVal=0,
        mem_mode="const",  # Embed thresholds in code
        ram_style="distributed"
    )
    
    graph = helper.make_graph(
        [thresh_node], 
        "thresholding_graph",
        [inp], 
        [outp]
    )
    
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Set threshold values (must be in ascending order per channel)
    thresholds = np.zeros((n_chans, n_steps), dtype=np.float32)
    for ch in range(n_chans):
        # Create ascending thresholds for each channel
        base = -10 + ch * 2
        thresholds[ch] = [base, base + 10, base + 20]
    
    model.set_initializer("global_out_thresholds", thresholds)
    model.set_tensor_datatype("global_in", idt)
    model.set_tensor_datatype("global_out", odt)
    
    return model


def generate_code_outputs(model, output_dir):
    """Generate all code outputs from the model."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # First specialize to HLS
    print("Specializing model to HLS backend...")
    model = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))  # Default PYNQ part
    
    # Get the specialized node
    node = model.graph.node[0]
    print(f"Specialized node type: {node.op_type}")
    
    # Get the custom operation instance
    from qonnx.custom_op.registry import getCustomOp
    inst = getCustomOp(node)
    
    print(f"Backend class: {inst.__class__.__name__}")
    print(f"Node name: {inst.onnx_node.name}")
    
    # Generate various code components
    print("\nGenerating code components...")
    
    # 1. Defines (constants and parameters)
    if hasattr(inst, 'defines'):
        defines_code = inst.defines()
        if defines_code:
            with open(os.path.join(output_dir, "defines.hpp"), "w") as f:
                f.write(defines_code)
            print("✓ Generated defines.hpp")
    
    # 2. Pragmas (HLS optimization directives)
    if hasattr(inst, 'pragmas'):
        pragmas_code = inst.pragmas()
        if pragmas_code:
            with open(os.path.join(output_dir, "pragmas.hpp"), "w") as f:
                f.write(pragmas_code)
            print("✓ Generated pragmas.hpp")
    
    # 3. Read threshold data function
    if hasattr(inst, 'read_npy_data'):
        read_npy_code = inst.read_npy_data()
        if read_npy_code:
            with open(os.path.join(output_dir, "read_npy_data.hpp"), "w") as f:
                f.write(read_npy_code)
            print("✓ Generated read_npy_data.hpp")
    
    # 4. Stream declarations
    if hasattr(inst, 'strm_decl'):
        strm_decl_code = inst.strm_decl()
        if strm_decl_code:
            with open(os.path.join(output_dir, "stream_decl.hpp"), "w") as f:
                f.write(strm_decl_code)
            print("✓ Generated stream_decl.hpp")
    
    # 5. DoCompute function (main processing)
    if hasattr(inst, 'docompute'):
        docompute_code = inst.docompute()
        if docompute_code:
            with open(os.path.join(output_dir, "docompute.hpp"), "w") as f:
                f.write(docompute_code)
            print("✓ Generated docompute.hpp")
    
    # 6. Blackbox function (top-level wrapper)
    if hasattr(inst, 'blackboxfunction'):
        blackbox_code = inst.blackboxfunction()
        if blackbox_code:
            with open(os.path.join(output_dir, "blackbox_function.cpp"), "w") as f:
                f.write(blackbox_code)
            print("✓ Generated blackbox_function.cpp")
    
    # 7. Get dataflow function call
    if hasattr(inst, 'get_dataflow_function_call'):
        dataflow_call = inst.get_dataflow_function_call()
        if dataflow_call:
            with open(os.path.join(output_dir, "dataflow_function_call.txt"), "w") as f:
                f.write(dataflow_call)
            print("✓ Generated dataflow_function_call.txt")
    
    # 8. Check for code_gen_dict (legacy pattern)
    if hasattr(inst, 'code_gen_dict'):
        print("\n⚠️  Note: This backend uses legacy code_gen_dict pattern")
        code_dict = inst.code_gen_dict()
        print(f"   Available keys: {list(code_dict.keys())}")
        
        # Save the entire dict for reference
        with open(os.path.join(output_dir, "code_gen_dict_keys.txt"), "w") as f:
            f.write("Keys in code_gen_dict:\n")
            for key in sorted(code_dict.keys()):
                f.write(f"  - {key}\n")
    
    # 9. Get node attributes for reference
    attrs = {}
    for attr_name in inst.get_nodeattr_types():
        attrs[attr_name] = inst.get_nodeattr(attr_name)
    
    with open(os.path.join(output_dir, "node_attributes.txt"), "w") as f:
        f.write("Node Attributes:\n")
        for k, v in sorted(attrs.items()):
            f.write(f"  {k}: {v}\n")
    print("✓ Generated node_attributes.txt")
    
    # 10. Try to generate the complete HLS code
    try:
        # This would normally be done by PrepareIP transform
        code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
        if code_gen_dir == "":
            code_gen_dir = output_dir
            inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        
        # Generate IP block files
        inst.generate_hdl()
        print("✓ Generated HDL files")
    except Exception as e:
        print(f"⚠️  Could not generate HDL: {e}")
    
    return inst


def main():
    """Main execution."""
    print("FINN Thresholding Code Generation")
    print("="*60)
    
    output_base = "/tmp/finn_thresholding_code"
    os.makedirs(output_base, exist_ok=True)
    
    try:
        # Create model
        print("\nCreating Thresholding model...")
        model = create_thresholding_model()
        print("✓ Model created")
        
        # Save model for reference
        model.save(os.path.join(output_base, "thresholding_model.onnx"))
        print(f"✓ Model saved to {output_base}/thresholding_model.onnx")
        
        # Generate legacy backend code
        print("\nGenerating code with LEGACY backend...")
        legacy_output = os.path.join(output_base, "legacy_backend")
        inst = generate_code_outputs(model, legacy_output)
        
        # Print summary
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Output directory: {output_base}")
        print("\nGenerated files:")
        
        for root, dirs, files in os.walk(output_base):
            level = root.replace(output_base, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = "  " * (level + 1)
            for file in sorted(files):
                size = os.path.getsize(os.path.join(root, file))
                print(f"{subindent}{file} ({size} bytes)")
        
        print("\nTo examine the generated code:")
        print(f"  cat {legacy_output}/defines.hpp")
        print(f"  cat {legacy_output}/docompute.hpp")
        print(f"  cat {legacy_output}/blackbox_function.cpp")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())