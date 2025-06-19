#!/usr/bin/env python3
"""
Generate Thresholding code using FINN's standard flow.
"""

import os
import sys
from pathlib import Path
import numpy as np
import shutil

# Add FINN to path
finn_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(finn_root))

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from onnx import TensorProto, helper
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP


def create_simple_thresholding_model():
    """Create a simple Thresholding model."""
    
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


def main():
    """Main execution."""
    print("FINN Thresholding Code Generation (Simple)")
    print("="*60)
    
    output_base = "/tmp/finn_thresholding_simple"
    os.makedirs(output_base, exist_ok=True)
    
    try:
        # Create model
        print("\nCreating model...")
        model = create_simple_thresholding_model()
        model.save(os.path.join(output_base, "model_original.onnx"))
        print("✓ Original model saved")
        
        # Specialize to HLS
        print("\nSpecializing to HLS...")
        model_spec = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
        model_spec.save(os.path.join(output_base, "model_specialized.onnx"))
        print("✓ Specialized model saved")
        
        # Get the node
        node = model_spec.graph.node[0]
        print(f"\nSpecialized node type: {node.op_type}")
        
        # Set code generation directory
        code_gen_dir = os.path.join(output_base, "code_gen")
        model_spec.set_metadata_prop("code_gen_dir", code_gen_dir)
        
        # Prepare for IP generation
        print("\nPreparing IP generation...")
        model_ip = model_spec.transform(PrepareIP(fpgapart="xc7z020clg400-1", clk_ns=10))
        
        # Get the custom op
        from qonnx.custom_op.registry import getCustomOp
        inst = getCustomOp(model_ip.graph.node[0])
        
        print(f"Backend class: {inst.__class__.__name__}")
        
        # Check code generation directory
        ip_dir = inst.get_nodeattr("code_gen_dir_ipgen")
        print(f"IP generation directory: {ip_dir}")
        
        # List generated files
        if os.path.exists(ip_dir):
            print("\nGenerated files:")
            for root, dirs, files in os.walk(ip_dir):
                level = root.replace(ip_dir, "").count(os.sep)
                indent = "  " * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = "  " * (level + 1)
                for file in sorted(files):
                    print(f"{subindent}{file}")
            
            # Copy some key files to output
            key_files = ["thresholding.h", "thresholding.cpp", "thresholding_tb.cpp"]
            for fname in key_files:
                src = os.path.join(ip_dir, fname)
                if os.path.exists(src):
                    dst = os.path.join(output_base, fname)
                    shutil.copy2(src, dst)
                    print(f"\nCopied {fname} to output directory")
        
        # Try C++ simulation preparation
        print("\nPreparing C++ simulation...")
        model_sim = model_ip.transform(PrepareCppSim())
        model_sim = model_sim.transform(CompileCppSim())
        model_sim = model_sim.transform(SetExecMode("cppsim"))
        print("✓ C++ simulation prepared")
        
        # Get some code snippets using the legacy interface
        print("\nExtracting code snippets...")
        snippets_dir = os.path.join(output_base, "snippets")
        os.makedirs(snippets_dir, exist_ok=True)
        
        # Try to get various code generation outputs
        if hasattr(inst, 'code_gen_dict'):
            print("Found code_gen_dict() method")
            try:
                code_dict = inst.code_gen_dict()
                print(f"Available keys: {list(code_dict.keys())[:10]}...")  # Show first 10
                
                # Save some key snippets
                for key in ["$DOCOMPUTE$", "$DEFINES$", "$PRAGMAS$", "$READNPYDATA$"]:
                    if key in code_dict:
                        fname = key.replace("$", "").lower() + ".txt"
                        with open(os.path.join(snippets_dir, fname), "w") as f:
                            f.write(f"// {key}\n")
                            f.write(str(code_dict[key])[:2000])  # First 2000 chars
                        print(f"  Saved {fname}")
            except Exception as e:
                print(f"  Error accessing code_gen_dict: {e}")
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Output directory: {output_base}")
        print("\nTo view generated code:")
        print(f"  ls -la {output_base}")
        print(f"  cat {output_base}/thresholding.h")
        print(f"  cat {output_base}/snippets/defines.txt")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())