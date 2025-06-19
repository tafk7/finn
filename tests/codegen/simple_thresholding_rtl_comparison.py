#!/usr/bin/env python3
"""
Simple Thresholding RTL code generation comparison.
Create node, run codegen with legacy and clean RTL backends, save outputs.
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
    """Create simple Thresholding model for RTL."""
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
        preferred_impl_style="rtl",  # Force RTL specialization
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


def save_legacy_rtl_output(model):
    """Generate and save legacy RTL backend output."""
    print("=== LEGACY RTL BACKEND ===")
    
    # Specialize to RTL
    model_rtl = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
    node = model_rtl.graph.node[0]
    print(f"Node type: {node.op_type}")
    
    # Get backend instance
    from qonnx.custom_op.registry import getCustomOp
    inst = getCustomOp(node)
    print(f"Backend: {inst.__class__.__name__}")
    
    # Extract code from legacy RTL backend
    output = ""
    try:
        # Initialize code_gen_dict if not already done
        if not hasattr(inst, 'code_gen_dict'):
            inst.code_gen_dict = {}
        
        # Call RTL-specific generation methods
        print("  Generating RTL code...")
        
        # Set required attributes for RTL generation
        import os
        os.makedirs("/tmp/rtl_gen", exist_ok=True)
        inst.set_nodeattr("code_gen_dir_ipgen", "/tmp/rtl_gen")
        
        # RTL backends typically use generate_hdl method
        if hasattr(inst, 'generate_hdl'):
            inst.generate_hdl(model_rtl, "xc7z020clg400-1", 10.0)
        
        # Check if code_gen_dict was populated
        code_dict = inst.code_gen_dict
        if code_dict:
            output += f"// Legacy RTL backend: {inst.__class__.__name__}\n"
            output += f"// Generated code from code_gen_dict\n"
            output += f"// Keys found: {list(code_dict.keys())}\n\n"
            
            for key, content in code_dict.items():
                if isinstance(content, list):
                    content = '\n'.join(content)
                if content:
                    output += f"\n// === {key} ===\n"
                    output += f"{content}\n"
        else:
            # Try to get generated files directly
            rtl_dir = inst.get_nodeattr("code_gen_dir_ipgen")
            if os.path.exists(rtl_dir):
                # Look for generated Verilog files
                verilog_files = [f for f in os.listdir(rtl_dir) if f.endswith('.v')]
                if verilog_files:
                    output += f"// Legacy RTL backend: {inst.__class__.__name__}\n"
                    output += f"// Found generated Verilog files: {verilog_files}\n\n"
                    
                    # Read the main wrapper file if it exists
                    wrapper_file = f"{inst.get_verilog_top_module_name()}_wrapper.v"
                    if wrapper_file in verilog_files:
                        with open(os.path.join(rtl_dir, wrapper_file)) as f:
                            output += f.read()
                    elif verilog_files:
                        # Just read the first file found
                        with open(os.path.join(rtl_dir, verilog_files[0])) as f:
                            output += f.read()
            else:
                output = f"// Legacy RTL backend: {inst.__class__.__name__}\n"
                output += "// No RTL code generated (directory not found)\n"
                    
    except Exception as e:
        output = f"// Error generating legacy RTL code: {e}\n"
        import traceback
        output += f"// Traceback:\n// {traceback.format_exc().replace(chr(10), chr(10) + '// ')}\n"
    
    # Save output
    output_dir = "/home/tafk/dev/tafk-finn-1/thresholding_comparison"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/legacy_thresholding_rtl_output.v", "w") as f:
        f.write(output)
    
    print(f"Saved: {output_dir}/legacy_thresholding_rtl_output.v")
    return len(output)


def save_clean_rtl_output(model):
    """Generate and save clean RTL backend output."""
    print("=== CLEAN RTL BACKEND ===")
    
    try:
        from finn.custom_op.fpgadataflow.rtl.CG_thresholding_rtl import CG_Thresholding_rtl
        from finn.codegen.template_engine import TemplateEngine
        
        # Specialize to RTL
        model_rtl = model.transform(SpecializeLayers(fpgapart="xc7z020clg400-1"))
        node = model_rtl.graph.node[0]
        print(f"Node type: {node.op_type}")
        
        # Create clean RTL backend instance directly
        inst = CG_Thresholding_rtl(node)
        print(f"Backend: {inst.__class__.__name__} (clean RTL implementation)")
        
        output = f"// Clean RTL backend: {inst.__class__.__name__}\n"
        output += f"// Template-based code generation\n\n"
        
        # Try to generate actual RTL code using template engine
        try:
            engine = TemplateEngine()
            
            # Check what template to use - now simplified
            template_name = "thresholding_rtl.v.j2"
            
            # Get template values - prioritize get_template_values for clean backend
            if hasattr(inst, 'get_template_values'):
                print("  Getting template values...")
                template_values = inst.get_template_values(template_name)
            elif hasattr(inst, 'get_rtl_wrapper_values'):
                print("  Getting RTL wrapper values...")
                template_values = inst.get_rtl_wrapper_values()
            else:
                raise Exception("No template value generation method found")
                
            output += f"// Template values keys: {list(template_values.keys())}\n\n"
            
            # Debug: Show actual values
            output += "// Template values debug:\n"
            for key, value in template_values.items():
                if isinstance(value, str) and len(value) > 100:
                    output += f"// {key}: <{len(value)} chars>\n"
                else:
                    output += f"// {key}: {repr(value)}\n"
            output += "\n"
            
            # Render the template
            print("  Rendering RTL template...")
            rendered = engine.render(template_name, template_values)
            output += "// === Rendered wrapper.v ===\n"
            output += rendered
                    
        except Exception as e:
            output += f"\n// Error rendering template: {e}\n"
            import traceback
            output += f"// Traceback:\n// {traceback.format_exc().replace(chr(10), chr(10) + '// ')}\n"
        
        # Save output
        output_dir = "/home/tafk/dev/tafk-finn-1/thresholding_comparison"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/clean_thresholding_rtl_output.v", "w") as f:
            f.write(output)
        
        print(f"Saved: {output_dir}/clean_thresholding_rtl_output.v")
        return len(output)
        
    except Exception as e:
        print(f"Error with clean RTL backend: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print("Simple Thresholding RTL Code Generation Comparison")
    print("=" * 50)
    
    # Create model
    model = create_model()
    print("✓ Model created with RTL preference")
    
    # Generate outputs
    legacy_size = save_legacy_rtl_output(model)
    clean_size = save_clean_rtl_output(model)
    
    print(f"\nLegacy RTL output: {legacy_size} chars")
    print(f"Clean RTL output: {clean_size} chars")
    print("\nFiles saved:")
    print("- /home/tafk/dev/tafk-finn-1/thresholding_comparison/legacy_thresholding_rtl_output.v")
    print("- /home/tafk/dev/tafk-finn-1/thresholding_comparison/clean_thresholding_rtl_output.v")


if __name__ == "__main__":
    main()