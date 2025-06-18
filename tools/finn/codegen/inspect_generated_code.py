#!/usr/bin/env python3
"""
Generated Code Inspection Tool
Shows full generated code output from real backend invocation.
"""

import sys
import os
sys.path.insert(0, '/home/tafk/dev/tafk-finn-1/src')

def inspect_generated_code():
    """Generate and save full code samples for inspection."""
    
    try:
        from finn.codegen.backend_instance_manager import BackendInstanceManager
        from finn.codegen.test_node_factory import TestNodeFactory
        
        print("🔍 Generating Code Samples for Inspection")
        print("=" * 50)
        
        manager = BackendInstanceManager()
        factory = TestNodeFactory()
        
        # Create test node with realistic attributes
        test_node = factory.create_thresholding_node(
            NumChannels=64, 
            PE=8, 
            NumSteps=16,
            ram_style='block'
        )
        
        print(f"📋 Test Node Attributes:")
        for key, value in test_node.attribute.items():
            print(f"   {key}: {value}")
        print()
        
        # 1. Clean Backend Simulation (Template-based)
        class MockCleanBackend:
            def __init__(self, onnx_node):
                self.onnx_node = onnx_node
                
            def get_template_values(self):
                pe = self.onnx_node.get_nodeattr("PE", 4)
                channels = self.onnx_node.get_nodeattr("NumChannels", 32)
                steps = self.onnx_node.get_nodeattr("NumSteps", 8)
                return {
                    'INCLUDES': [
                        '#include "activations.hpp"',
                        '#include "hls_stream.h"', 
                        '#include "ap_int.h"'
                    ],
                    'DEFINES': [
                        f'#define PE {pe}',
                        f'#define NumChannels {channels}',
                        f'#define NumSteps {steps}'
                    ],
                    'PRAGMAS': [
                        '#pragma HLS INTERFACE axis port=in0_V',
                        '#pragma HLS INTERFACE axis port=out0_V',
                        '#pragma HLS INTERFACE ap_ctrl_none port=return',
                        '#pragma HLS PIPELINE II=1'
                    ]
                }
        
        # 2. Legacy Backend Simulation (Direct generation)
        class MockLegacyBackend:
            def __init__(self, onnx_node):
                self.onnx_node = onnx_node
                
            def code_generation_cppsim(self):
                pe = self.onnx_node.get_nodeattr("PE", 4)
                channels = self.onnx_node.get_nodeattr("NumChannels", 32)
                return f"""// Legacy Backend Direct Generation
#include "bnn-library.h"
#include "activations.hpp"

#define PE_LEGACY {pe}
#define CHANNELS_LEGACY {channels}

void Thresholding_Batch_Legacy(
    hls::stream<ap_uint<64>>& in0_V,
    hls::stream<ap_uint<64>>& out0_V,
    const ap_uint<64> numReps
) {{
#pragma HLS INTERFACE axis port=in0_V
#pragma HLS INTERFACE axis port=out0_V
#pragma HLS INTERFACE ap_ctrl_none port=return

    const unsigned int fold = PE_LEGACY;
    
    for(unsigned int rep = 0; rep < numReps; rep++) {{
        for(unsigned int ch = 0; ch < CHANNELS_LEGACY / fold; ch++) {{
#pragma HLS PIPELINE II=1
            ap_uint<64> input_data = in0_V.read();
            ap_uint<64> output_data = 0;
            
            // Legacy thresholding logic
            for(unsigned int pe = 0; pe < fold; pe++) {{
                ap_uint<8> current_elem = input_data.range(8*(pe+1)-1, 8*pe);
                ap_uint<8> threshold_result = (current_elem > 127) ? 255 : 0;
                output_data.range(8*(pe+1)-1, 8*pe) = threshold_result;
            }}
            
            out0_V.write(output_data);
        }}
    }}
}}"""
        
        # Generate code with both backends
        print("🔧 Generating Clean Backend Code...")
        clean_instance = manager.create_backend_instance(MockCleanBackend, test_node)
        clean_code = manager.call_backend_generation(clean_instance, 'template')
        
        print("🔧 Generating Legacy Backend Code...")
        legacy_instance = manager.create_backend_instance(MockLegacyBackend, test_node)
        legacy_code = manager.call_backend_generation(legacy_instance, 'template')
        
        # Save to files for inspection
        clean_file = 'generated_clean_backend.cpp'
        legacy_file = 'generated_legacy_backend.cpp'
        
        with open(clean_file, 'w') as f:
            f.write(clean_code)
        print(f"✅ Clean backend code saved to: {clean_file}")
        
        with open(legacy_file, 'w') as f:
            f.write(legacy_code)
        print(f"✅ Legacy backend code saved to: {legacy_file}")
        
        # Display file contents
        print(f"\n📄 Clean Backend Code ({len(clean_code)} characters):")
        print("=" * 60)
        print(clean_code)
        
        print(f"\n📄 Legacy Backend Code ({len(legacy_code)} characters):")
        print("=" * 60)
        print(legacy_code)
        
        # Analysis
        print(f"\n🔍 Code Analysis:")
        print("=" * 30)
        print(f"Clean code lines: {len(clean_code.splitlines())}")
        print(f"Legacy code lines: {len(legacy_code.splitlines())}")
        print(f"Clean includes 'activations.hpp': {'activations.hpp' in clean_code}")
        print(f"Legacy includes 'activations.hpp': {'activations.hpp' in legacy_code}")
        print(f"Clean has HLS pragmas: {'#pragma HLS' in clean_code}")
        print(f"Legacy has HLS pragmas: {'#pragma HLS' in legacy_code}")
        print(f"Clean uses template approach: {'DEFINES' in str(MockCleanBackend.__dict__)}")
        print(f"Legacy uses direct generation: {'code_generation_cppsim' in str(MockLegacyBackend.__dict__)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Code inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_file_locations():
    """Show where generated files are located."""
    
    current_dir = os.getcwd()
    clean_file = os.path.join(current_dir, 'generated_clean_backend.cpp')
    legacy_file = os.path.join(current_dir, 'generated_legacy_backend.cpp')
    
    print(f"\n📁 Generated Files Location:")
    print(f"   Clean:  {clean_file}")
    print(f"   Legacy: {legacy_file}")
    
    if os.path.exists(clean_file):
        size = os.path.getsize(clean_file)
        print(f"   Clean file size: {size} bytes")
    
    if os.path.exists(legacy_file):
        size = os.path.getsize(legacy_file)
        print(f"   Legacy file size: {size} bytes")
    
    print(f"\n🔍 To examine files:")
    print(f"   cat {clean_file}")
    print(f"   cat {legacy_file}")
    print(f"   code {clean_file} {legacy_file}  # VS Code")
    print(f"   diff {clean_file} {legacy_file}  # Compare")


if __name__ == "__main__":
    print("🔍 FINN Generated Code Inspector")
    print("=" * 40)
    
    success = inspect_generated_code()
    
    if success:
        show_file_locations()
        print(f"\n🎯 Inspection Complete: ✅ SUCCESS")
    else:
        print(f"\n🎯 Inspection Failed: ❌ ERROR")
    
    exit(0 if success else 1)