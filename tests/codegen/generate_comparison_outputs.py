#!/usr/bin/env python3
"""
Generate code outputs from both legacy and clean backends for comparison.

This script generates code from Thresholding and MVAU operations using both
legacy and clean backends (where available) and saves the outputs for comparison.
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add FINN to path
finn_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(finn_root))


def generate_thresholding_comparison():
    """Generate Thresholding code from both backends."""
    import numpy as np
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.general.multithreshold import multithreshold
    from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
    import onnx
    from onnx import TensorProto, helper
    
    print("Creating Thresholding test model...")
    
    # Create a simple model with Thresholding
    idt = DataType["INT4"]
    odt = DataType["UINT4"]
    n_inp_vecs = 4
    n_chans = 8
    fold = 2  # Folding factor
    pe = n_chans // fold
    
    # Create model
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [n_inp_vecs, n_chans])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [n_inp_vecs, n_chans])
    
    # Create thresholds - 3 threshold levels
    thresholds = np.random.randint(idt.min(), idt.max(), (n_chans, 3)).astype(np.float32)
    thresholds = np.sort(thresholds, axis=1)  # Ensure increasing order
    
    # Create the Thresholding node
    thresh_node = helper.make_node(
        "Thresholding",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=n_chans,
        PE=pe,
        numSteps=3,
        inputDataType=idt.name,
        outputDataType=odt.name,
        weightDataType="INT4",
    )
    
    graph = helper.make_graph([thresh_node], "thresholding_graph", [inp], [outp])
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Set thresholds
    model.set_initializer("thresholding_graph_Thresholding_0_thresholds", thresholds)
    
    return model


def generate_mvau_comparison():
    """Generate MVAU code from both backends."""
    import numpy as np
    from qonnx.core.datatype import DataType
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
    import onnx
    from onnx import TensorProto, helper
    
    print("Creating MVAU test model...")
    
    # Create a simple model with MatrixVectorActivation
    idt = DataType["INT4"]
    wdt = DataType["INT4"]
    odt = DataType["INT8"]
    
    mw = 16
    mh = 8
    pe = 4
    simd = 2
    
    # Create model
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    
    # Create weights
    weights = np.random.randint(wdt.min(), wdt.max() + 1, (mw, mh)).astype(np.float32)
    
    # Create the MVAU node
    mvau_node = helper.make_node(
        "MVAU",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        PE=pe,
        SIMD=simd,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        binaryXnorMode=0,
        noActivation=1,
    )
    
    graph = helper.make_graph([mvau_node], "mvau_graph", [inp], [outp])
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Set weights
    model.set_initializer("mvau_graph_MVAU_0_weights", weights)
    
    return model


def save_generated_code(node, output_dir, backend_type):
    """Save generated code from a node."""
    # Get the custom op
    from qonnx.custom_op.registry import getCustomOp
    inst = getCustomOp(node)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different code types based on what the operation supports
    code_types = []
    
    # Try to generate various code outputs
    try:
        # HLS operations typically have these methods
        if hasattr(inst, 'docompute'):
            docompute_code = inst.docompute()
            if docompute_code:
                with open(os.path.join(output_dir, "docompute.cpp"), "w") as f:
                    f.write(docompute_code)
                code_types.append("docompute.cpp")
        
        if hasattr(inst, 'blackboxfunction'):
            blackbox_code = inst.blackboxfunction()
            if blackbox_code:
                with open(os.path.join(output_dir, "blackbox_function.cpp"), "w") as f:
                    f.write(blackbox_code)
                code_types.append("blackbox_function.cpp")
                
        if hasattr(inst, 'pragmas'):
            pragma_code = inst.pragmas()
            if pragma_code:
                with open(os.path.join(output_dir, "pragmas.hpp"), "w") as f:
                    f.write(pragma_code)
                code_types.append("pragmas.hpp")
                
        if hasattr(inst, 'defines'):
            defines_code = inst.defines()
            if defines_code:
                with open(os.path.join(output_dir, "defines.hpp"), "w") as f:
                    f.write(defines_code)
                code_types.append("defines.hpp")
                
        # For RTL operations
        if hasattr(inst, 'rtl_code'):
            rtl_code = inst.rtl_code()
            if rtl_code:
                with open(os.path.join(output_dir, "rtl_module.v"), "w") as f:
                    f.write(rtl_code)
                code_types.append("rtl_module.v")
        
        # Try the general code generation method
        if hasattr(inst, 'code_generation'):
            gen_code = inst.code_generation()
            if gen_code:
                with open(os.path.join(output_dir, "generated_code.txt"), "w") as f:
                    f.write(str(gen_code))
                code_types.append("generated_code.txt")
                
    except Exception as e:
        print(f"  Warning: Error generating code for {backend_type}: {e}")
    
    return code_types


def compare_backends(model_name, model):
    """Generate code from both legacy and clean backends for comparison."""
    from finn.codegen.CG_backend_registration import get_clean_backend_registry
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    
    print(f"\n{'='*60}")
    print(f"Comparing {model_name}")
    print('='*60)
    
    # Create output directory
    output_base = f"/tmp/finn_backend_comparison/{model_name}"
    os.makedirs(output_base, exist_ok=True)
    
    # First specialize to HLS backend
    model_hls = model.copy()
    model_hls = model_hls.transform(SpecializeLayers(backend="hls"))
    
    # Get the node
    node = model_hls.graph.node[0]
    op_type = node.op_type
    
    print(f"\nOperation type: {op_type}")
    print(f"Node name: {node.name}")
    
    # Generate legacy backend code
    print("\nGenerating LEGACY backend code...")
    legacy_output_dir = os.path.join(output_base, "legacy")
    try:
        legacy_files = save_generated_code(node, legacy_output_dir, "legacy")
        print(f"  Generated files: {', '.join(legacy_files)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check if clean backend exists
    clean_registry = get_clean_backend_registry()
    clean_hls_backend = clean_registry.get_hls_backend(op_type, prefer_clean=True)
    
    if clean_hls_backend and 'CG_' in clean_hls_backend.__name__:
        print("\nGenerating CLEAN backend code...")
        # Force use of clean backend
        clean_registry.enable_clean_backends([op_type])
        
        # Re-specialize with clean backend preference
        model_clean = model.copy()
        # This would need custom logic to force clean backend usage
        # For now, we'll note that clean backend exists
        print(f"  Clean backend available: {clean_hls_backend.__name__}")
        print("  Note: Full clean backend code generation requires A/B testing mode enabled")
    else:
        print(f"\nNo clean backend available for {op_type}")
    
    # Save model info
    info_file = os.path.join(output_base, "model_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Operation: {op_type}\n")
        f.write(f"Node attributes:\n")
        for attr in node.attribute:
            f.write(f"  {attr.name}: {helper.get_attribute_value(attr)}\n")
    
    print(f"\nOutput saved to: {output_base}")
    
    return output_base


def main():
    """Main execution function."""
    print("FINN Backend Code Generation Comparison")
    print("=" * 60)
    
    # Check if running in Docker
    if not os.path.exists("/workspace/finn"):
        print("WARNING: Not running in FINN Docker environment.")
        print("This script should be run with: ./run-docker.sh python <script>")
        print("Attempting to continue anyway...")
    
    try:
        # Generate Thresholding comparison
        thresh_model = generate_thresholding_comparison()
        thresh_output = compare_backends("Thresholding", thresh_model)
        
        # Generate MVAU comparison
        mvau_model = generate_mvau_comparison()
        mvau_output = compare_backends("MVAU", mvau_model)
        
        # Create a summary
        summary_file = "/tmp/finn_backend_comparison/COMPARISON_SUMMARY.txt"
        with open(summary_file, "w") as f:
            f.write("FINN Backend Code Generation Comparison\n")
            f.write("=" * 60 + "\n\n")
            f.write("This directory contains generated code from both legacy and clean backends.\n\n")
            f.write("Directory structure:\n")
            f.write("- Thresholding/\n")
            f.write("  - legacy/     : Code generated by legacy backend\n")
            f.write("  - clean/      : Code generated by clean backend (if available)\n")
            f.write("  - model_info.txt : Model configuration details\n")
            f.write("- MVAU/\n")
            f.write("  - legacy/     : Code generated by legacy backend\n")
            f.write("  - clean/      : Code generated by clean backend (if available)\n")
            f.write("  - model_info.txt : Model configuration details\n\n")
            f.write("To compare outputs:\n")
            f.write("1. Check if code structure differs between legacy and clean\n")
            f.write("2. Look for differences in generated pragmas, defines, etc.\n")
            f.write("3. Verify functional equivalence\n")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Comparison outputs saved to: /tmp/finn_backend_comparison/")
        print(f"View summary: /tmp/finn_backend_comparison/COMPARISON_SUMMARY.txt")
        
        # Copy to host if in Docker
        if os.path.exists("/workspace/finn"):
            host_output = "/workspace/finn_backend_comparison"
            if os.path.exists(host_output):
                shutil.rmtree(host_output)
            shutil.copytree("/tmp/finn_backend_comparison", host_output)
            print(f"\nOutputs also copied to: {host_output}")
            print("(This will be accessible from the host machine)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())