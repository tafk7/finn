#!/usr/bin/env python3
"""
Simple Thresholding code generation comparison.
Create node, run codegen with legacy and clean backends, save outputs.
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


def create_model():
    """Create simple Thresholding model."""
    # Basic parameters
    n_chans = 4
    pe = 2
    n_steps = 3
    
    # Create node with threshold input
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, n_chans])
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [n_chans, n_steps])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, n_chans])
    
    node = helper.make_node(
        "Thresholding",
        ["inp", "thresh"], ["outp"],  # Two inputs: data and thresholds
        domain="finn.custom_op.fpgadataflow",
        preferred_impl_style="hls",  # Force HLS specialization
        NumChannels=n_chans,
        PE=pe,
        numSteps=n_steps,
        inputDataType="INT8",
        outputDataType="UINT2",
        weightDataType="INT8"
    )
    
    graph = helper.make_graph([node], "thresh", [inp], [outp], value_info=[thresh])
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)
    
    # Add thresholds
    thresholds = np.array([[-10, 0, 10], [-5, 5, 15], [-8, 2, 12], [-3, 7, 17]], dtype=np.float32)
    model.set_initializer("thresh", thresholds)
    
    return model


def save_legacy_output(model):
    """Generate and save legacy backend output."""
    print("=== LEGACY BACKEND ===")
    
    # Specialize to HLS
    model_hls = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
    node = model_hls.graph.node[0]
    print(f"Node type: {node.op_type}")
    
    # Get backend instance
    from qonnx.custom_op.registry import getCustomOp
    inst = getCustomOp(node)
    print(f"Backend: {inst.__class__.__name__}")
    
    # Extract code from legacy backend
    output = ""
    try:
        # Initialize code_gen_dict if not already done
        if not hasattr(inst, 'code_gen_dict'):
            inst.code_gen_dict = {}
        
        # Call methods to populate code_gen_dict
        print("  Populating code_gen_dict...")
        
        # These methods populate different parts of code_gen_dict
        # Some methods need specific arguments
        if hasattr(inst, 'defines'):
            inst.defines(var="")  # Legacy API requires var parameter
        if hasattr(inst, 'read_npy_data'):
            inst.read_npy_data()
        if hasattr(inst, 'strm_decl'):
            inst.strm_decl()
        if hasattr(inst, 'docompute'):
            inst.docompute()
        if hasattr(inst, 'pragmas'):
            inst.pragmas()
        if hasattr(inst, 'dataoutstrm'):
            inst.dataoutstrm()
        if hasattr(inst, 'save_as_npy'):
            inst.save_as_npy()
        if hasattr(inst, 'blackboxfunction'):
            inst.blackboxfunction()
            
        # Now extract the populated code
        code_dict = inst.code_gen_dict
        output += f"// Legacy HLS backend: {inst.__class__.__name__}\n"
        output += f"// Generated code from populated code_gen_dict\n"
        output += f"// Keys found: {list(code_dict.keys())}\n\n"
        
        # Extract main code sections in order
        sections = [
            ("$GLOBALS$", "Global includes"),
            ("$DEFINES$", "Defines"), 
            ("$PRAGMAS$", "HLS Pragmas"),
            ("$STREAMDECLARATIONS$", "Stream declarations"),
            ("$READNPYDATA$", "Read NPY data"),
            ("$DOCOMPUTE$", "Compute function"),
            ("$DATAOUTSTREAM$", "Data output"),
            ("$SAVEASCNPY$", "Save as NPY"),
            ("$BLACKBOXFUNCTION$", "Blackbox function")
        ]
        
        for key, desc in sections:
            if key in code_dict:
                content = code_dict[key]
                if isinstance(content, list):
                    content = '\n'.join(content)
                if content:
                    output += f"\n// === {desc} ({key}) ===\n"
                    output += f"{content}\n"
                    
    except Exception as e:
        output = f"// Error generating legacy code: {e}\n"
        import traceback
        output += f"// Traceback:\n// {traceback.format_exc().replace(chr(10), chr(10) + '// ')}\n"
    
    # Save output
    output_dir = "/home/tafk/dev/tafk-finn-1/thresholding_comparison"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/legacy_thresholding_output.cpp", "w") as f:
        f.write(output)
    
    print(f"Saved: {output_dir}/legacy_thresholding_output.cpp")
    return len(output)


def save_clean_output(model):
    """Generate and save clean backend output."""
    print("=== CLEAN BACKEND ===")
    
    try:
        from finn.custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_Thresholding_hls
        from finn.codegen.template_engine import TemplateEngine
        
        # Specialize to HLS with clean backend
        model_hls = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
        node = model_hls.graph.node[0]
        print(f"Node type: {node.op_type}")
        
        # Create clean backend instance directly
        # Don't use getCustomOp since it returns the legacy backend
        inst = CG_Thresholding_hls(node)
        print(f"Backend: {inst.__class__.__name__} (clean implementation)")
        
        output = f"// Clean HLS backend: {inst.__class__.__name__}\n"
        output += f"// Template-based code generation\n\n"
        
        # Try to generate actual code using template engine
        try:
            engine = TemplateEngine()
            
            # Get template values
            if hasattr(inst, 'get_template_values'):
                print("  Getting template values...")
                template_values = inst.get_template_values("docompute.cpp.j2")
                output += f"// Template values keys: {list(template_values.keys())}\n\n"
                
                # Debug: Show actual values
                output += "// Template values debug:\n"
                for key, value in template_values.items():
                    if isinstance(value, str) and len(value) > 100:
                        output += f"// {key}: <{len(value)} chars>\n"
                    else:
                        output += f"// {key}: {repr(value)}\n"
                output += "\n"
                
                # Render the template - now using simplified template
                print("  Rendering template...")
                rendered = engine.render("hls_basic.cpp.j2", template_values)
                output += "// === Rendered docompute.cpp ===\n"
                output += rendered
            else:
                # Fall back to showing template
                template_dir = finn_root / "src/finn/codegen/templates/thresholding/hls"
                template_path = template_dir / "docompute.cpp.j2"
                if template_path.exists():
                    with open(template_path) as f:
                        output += "// === Template (not rendered) ===\n"
                        output += f.read()
                        
        except Exception as e:
            output += f"\n// Error rendering template: {e}\n"
            import traceback
            output += f"// Traceback:\n// {traceback.format_exc().replace(chr(10), chr(10) + '// ')}\n"
        
        # Save output
        output_dir = "/home/tafk/dev/tafk-finn-1/thresholding_comparison"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/clean_thresholding_output.cpp", "w") as f:
            f.write(output)
        
        print(f"Saved: {output_dir}/clean_thresholding_output.cpp")
        return len(output)
        
    except Exception as e:
        print(f"Error with clean backend: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print("Simple Thresholding Code Generation Comparison")
    print("=" * 50)
    
    # Create model
    model = create_model()
    print("✓ Model created")
    
    # Generate outputs
    legacy_size = save_legacy_output(model)
    clean_size = save_clean_output(model)
    
    print(f"\nLegacy output: {legacy_size} chars")
    print(f"Clean output: {clean_size} chars")
    print("\nFiles saved:")
    print("- /home/tafk/dev/tafk-finn-1/thresholding_comparison/legacy_thresholding_output.cpp")
    print("- /home/tafk/dev/tafk-finn-1/thresholding_comparison/clean_thresholding_output.cpp")


if __name__ == "__main__":
    main()