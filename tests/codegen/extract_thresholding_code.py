#!/usr/bin/env python3
"""
Extract Thresholding code generation outputs directly from the backend.
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


def create_thresholding_model():
    """Create a Thresholding model."""
    
    # Model parameters
    n_chans = 4
    pe = 2
    n_steps = 3
    idt = DataType["INT8"]
    odt = DataType["UINT2"]
    
    # Create ONNX graph
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, n_chans])
    outp = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, [1, n_chans])
    
    # Create Thresholding node
    thresh_node = helper.make_node(
        "Thresholding",
        ["global_in"],
        ["global_out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=n_chans,
        PE=pe,
        numSteps=n_steps,
        inputDataType=idt.name,
        outputDataType=odt.name,
        weightDataType="INT8",
        ActVal=0,
        mem_mode="const",
        ram_style="distributed"
    )
    
    graph = helper.make_graph([thresh_node], "thresholding_graph", [inp], [outp])
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Set thresholds
    thresholds = np.array([[-10, 0, 10], [-5, 5, 15], [-8, 2, 12], [-3, 7, 17]], dtype=np.float32)
    model.set_initializer("global_out_thresholds", thresholds)
    model.set_tensor_datatype("global_in", idt)
    model.set_tensor_datatype("global_out", odt)
    
    return model


def extract_code_snippets(inst, output_dir):
    """Extract code snippets from the backend instance."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nExtracting code generation outputs...")
    
    # 1. Get code_gen_dict if available
    if hasattr(inst, 'code_gen_dict'):
        print("\nUsing code_gen_dict() method:")
        try:
            code_dict = inst.code_gen_dict()
            print(f"Found {len(code_dict)} code generation keys")
            
            # Show first 20 keys
            keys = sorted(code_dict.keys())[:20]
            print("\nSample keys:")
            for key in keys:
                print(f"  - {key}")
            
            # Extract some key code snippets
            snippets = {
                "$DEFINES$": "defines.hpp",
                "$DOCOMPUTE$": "docompute.cpp", 
                "$PRAGMAS$": "pragmas.hpp",
                "$READNPYDATA$": "read_npy_data.hpp",
                "$BLACKBOXFUNCTION$": "blackbox_function.cpp",
                "$STREAMDECLARATIONS$": "stream_declarations.hpp",
                "$MYNAME$": "myname.txt",
                "$LAYER_NAME$": "layer_name.txt",
                "$WRITE_WEIGHTS_EXTERNAL$": "write_weights_external.cpp",
                "$THRESHOLDS_INIT$": "thresholds_init.cpp"
            }
            
            for key, fname in snippets.items():
                if key in code_dict:
                    content = str(code_dict[key])
                    if content and content != key:  # Not empty or just the key
                        with open(os.path.join(output_dir, fname), "w") as f:
                            f.write(f"// Generated from {key}\n")
                            f.write(content)
                        print(f"  ✓ Saved {fname} ({len(content)} chars)")
            
            # Save all keys for reference
            with open(os.path.join(output_dir, "all_keys.txt"), "w") as f:
                f.write("All code_gen_dict keys:\n")
                f.write("="*50 + "\n")
                for key in sorted(code_dict.keys()):
                    f.write(f"{key}\n")
            
        except Exception as e:
            print(f"  Error with code_gen_dict: {e}")
    
    # 2. Try direct method calls
    print("\nTrying direct method calls:")
    
    # Get node attributes
    attrs = {}
    for attr_name in inst.get_nodeattr_types():
        attrs[attr_name] = inst.get_nodeattr(attr_name)
    
    with open(os.path.join(output_dir, "node_attributes.txt"), "w") as f:
        f.write("Node Attributes:\n")
        f.write("="*50 + "\n")
        for k, v in sorted(attrs.items()):
            f.write(f"{k}: {v}\n")
    print("  ✓ Saved node_attributes.txt")
    
    # Backend type
    print(f"\nBackend type: {inst.__class__.__name__}")
    print(f"Base classes: {[c.__name__ for c in inst.__class__.__bases__]}")
    
    # Check if it's using legacy pattern
    if 'code_gen_dict' in dir(inst):
        print("  ⚠️  Uses legacy code_gen_dict pattern")
    else:
        print("  ✓ Modern backend (no code_gen_dict)")


def main():
    """Main execution."""
    print("FINN Thresholding Code Extraction")
    print("="*60)
    
    output_base = "/tmp/finn_thresholding_extract"
    os.makedirs(output_base, exist_ok=True)
    
    try:
        # Create model
        print("\nCreating model...")
        model = create_thresholding_model()
        model.save(os.path.join(output_base, "model.onnx"))
        
        # Specialize to HLS
        print("\nSpecializing to HLS...")
        model_hls = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
        
        # Get the node and instance
        node = model_hls.graph.node[0]
        print(f"Specialized node type: {node.op_type}")
        
        from qonnx.custom_op.registry import getCustomOp
        inst = getCustomOp(node)
        
        # Extract code for HLS backend
        hls_output = os.path.join(output_base, node.op_type)
        extract_code_snippets(inst, hls_output)
        
        # Also try forcing RTL specialization
        print("\n" + "-"*60)
        print("Trying RTL specialization...")
        
        # Set preferred implementation style
        model_rtl = model.copy()
        graph = model_rtl.graph
        for n in graph.node:
            if n.op_type == "Thresholding":
                for attr in n.attribute:
                    if attr.name == "preferred_impl_style":
                        n.attribute.remove(attr)
                # Add RTL preference
                attr = helper.make_attribute("preferred_impl_style", "rtl")
                n.attribute.append(attr)
        
        model_rtl = model_rtl.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
        node_rtl = model_rtl.graph.node[0]
        print(f"RTL specialized node type: {node_rtl.op_type}")
        
        if node_rtl.op_type != node.op_type:
            inst_rtl = getCustomOp(node_rtl)
            rtl_output = os.path.join(output_base, node_rtl.op_type)
            extract_code_snippets(inst_rtl, rtl_output)
        
        # Summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Output directory: {output_base}")
        print("\nGenerated directories:")
        for item in sorted(os.listdir(output_base)):
            path = os.path.join(output_base, item)
            if os.path.isdir(path):
                files = os.listdir(path)
                print(f"  {item}/ ({len(files)} files)")
        
        print("\nTo view extracted code:")
        print(f"  ls -la {output_base}/*/")
        print(f"  cat {output_base}/*/defines.hpp")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())