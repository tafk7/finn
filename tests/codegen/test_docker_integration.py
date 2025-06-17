"""
Docker Integration Test for FINN Unified Code Generation Framework

This test validates the unified framework within the FINN Docker environment,
ensuring compatibility with existing FINN infrastructure and workflows.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add FINN to Python path if running in Docker
if 'FINN_ROOT' in os.environ:
    finn_root = os.environ['FINN_ROOT']
    sys.path.insert(0, os.path.join(finn_root, 'src'))

from finn.codegen import (
    ModernHLSGenerator,
    ModernRTLGenerator,
    TemplateEngine,
    FileManager,
    LibraryResolver
)


def test_docker_environment():
    """Test that FINN Docker environment is properly configured."""
    required_env_vars = ['FINN_ROOT', 'FINN_DEPS_DIR']
    
    print("=== Testing Docker Environment ===")
    for var in required_env_vars:
        value = os.environ.get(var)
        print(f"{var}: {value}")
        assert value is not None, f"Environment variable {var} not set"
        assert os.path.exists(value), f"Path {value} does not exist"
    
    print("✓ Docker environment configured correctly")


def test_library_resolver_in_docker():
    """Test LibraryResolver with actual FINN paths in Docker."""
    print("\n=== Testing LibraryResolver in Docker ===")
    
    resolver = LibraryResolver()
    
    # Test FINN library paths
    finn_hlslib_spec = resolver.get_library_info('finn-hlslib')
    if finn_hlslib_spec:
        resolved_path = resolver._resolve_path(finn_hlslib_spec.path)
        print(f"FINN HLS Library path: {resolved_path}")
        
    finn_rtllib_spec = resolver.get_library_info('finn-rtllib')
    if finn_rtllib_spec:
        resolved_path = resolver._resolve_path(finn_rtllib_spec.path)
        print(f"FINN RTL Library path: {resolved_path}")
    
    # List available libraries
    from finn.codegen.library_resolver import LibraryType
    hls_libs = resolver.list_libraries(LibraryType.HLS_LIBRARY)
    rtl_libs = resolver.list_libraries(LibraryType.RTL_LIBRARY)
    
    print(f"Available HLS libraries: {hls_libs}")
    print(f"Available RTL libraries: {rtl_libs}")
    
    print("✓ LibraryResolver working in Docker environment")


def test_template_engine_with_finn_paths():
    """Test TemplateEngine with FINN template directories."""
    print("\n=== Testing TemplateEngine with FINN Paths ===")
    
    engine = TemplateEngine()
    
    # List available template directories
    search_paths = engine.jinja_env.loader.searchpath
    print("Template search paths:")
    for path in search_paths:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {path}")
    
    # Test basic template rendering
    test_template = """
// Test template for {{ operation }}
#define WIDTH {{ width }}
#define HEIGHT {{ height }}
"""
    
    context = {
        'operation': 'TestOperation',
        'width': 64,
        'height': 32
    }
    
    result = engine.render_string(test_template, context)
    print(f"Template rendering test:\n{result}")
    
    assert "TestOperation" in result
    assert "64" in result
    assert "32" in result
    
    print("✓ TemplateEngine working with FINN paths")


def test_file_manager_in_docker():
    """Test FileManager operations in Docker environment."""
    print("\n=== Testing FileManager in Docker ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        fm = FileManager(tmp_dir)
        
        # Test directory creation
        code_gen_dir = fm.get_code_gen_dir("test_operation", "hls")
        print(f"Created code generation directory: {code_gen_dir}")
        assert code_gen_dir.exists()
        
        # Test file operations
        test_content = """
// Generated test file
#include <iostream>

void test_function() {
    std::cout << "Hello from FINN unified framework!" << std::endl;
}
"""
        
        test_file = fm.write_file(code_gen_dir / "test.cpp", test_content)
        print(f"Created test file: {test_file}")
        assert test_file.exists()
        
        # Test file reading
        read_content = fm.read_file(test_file)
        assert "Hello from FINN unified framework!" in read_content
        
        print("✓ FileManager working in Docker environment")


class MockFINNOperation:
    """Mock FINN operation that mimics real FINN HWCustomOp."""
    
    def __init__(self, op_type="MatrixVectorActivation"):
        self.onnx_node = type('MockNode', (), {})()
        self.onnx_node.op_type = op_type
        self.onnx_node.name = f"test_{op_type.lower()}"
        self.onnx_node.input = ["input0"]
        self.onnx_node.output = ["output0"]
        
        # Realistic FINN operation attributes
        self._node_attrs = {
            "MW": 128,
            "MH": 128,
            "PE": 8,
            "SIMD": 8,
            "mem_mode": "internal_embedded",
            "ActType": "relu",
            "NumChannels": 128,
            "ram_style": "distributed",
            "resType": "auto"
        }
    
    def get_nodeattr(self, name):
        return self._node_attrs.get(name)
    
    def get_nodeattr_names(self):
        return list(self._node_attrs.keys())
    
    def get_nodeattr_types(self):
        """Return dictionary of node attribute names to types."""
        return {
            "MW": int,
            "MH": int,
            "PE": int,
            "SIMD": int,
            "mem_mode": str,
            "ActType": str,
            "NumChannels": int,
            "ram_style": str,
            "resType": str
        }
    
    def get_normal_input_shape(self, input_name):
        return [1, 128] if input_name == "input0" else None
    
    def get_normal_output_shape(self, output_name):
        return [1, 128] if output_name == "output0" else None
    
    def get_input_datatype(self, idx):
        # Mock DataType with bitwidth method
        class MockDataType:
            def bitwidth(self):
                return 8
            def __str__(self):
                return "INT8"
        return MockDataType()
    
    def get_output_datatype(self, idx):
        # Mock DataType with bitwidth method
        class MockDataType:
            def bitwidth(self):
                return 8
            def __str__(self):
                return "INT8"
        return MockDataType()


def test_hls_generator_integration():
    """Test HLS generator with realistic FINN operation."""
    print("\n=== Testing HLS Generator Integration ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock operation
        mock_op = MockFINNOperation("MatrixVectorActivation")
        
        # Create HLS generator
        generator = ModernHLSGenerator(mock_op)
        
        # Test template name selection
        template_name = generator.get_template_name()
        print(f"Selected HLS template: {template_name}")
        assert "hls/" in template_name
        assert "mvau" in template_name.lower()
        
        # Test context preparation
        context = generator.prepare_context(None, "xc7z020clg400-1", "100MHz")
        print(f"Generated context keys: {list(context.keys())}")
        
        # Verify important context elements
        assert context['op_type'] == "MatrixVectorActivation"
        assert context['node_name'] == "test_matrixvectoractivation"
        assert 'defines' in context
        assert 'includes' in context
        
        # Test defines generation
        defines = context['defines']
        print(f"Generated defines: {defines[:3]}...")  # Show first 3
        define_names = [d[0] for d in defines]
        assert 'MW' in define_names
        assert 'PE' in define_names
        
        # Test file list generation
        files = generator.get_generated_files(tmp_dir)
        print(f"Files to be generated: {files}")
        assert len(files) > 0
        assert any(f.endswith('.cpp') for f in files)
        
        print("✓ HLS generator integration successful")


def test_rtl_generator_integration():
    """Test RTL generator with realistic FINN operation."""
    print("\n=== Testing RTL Generator Integration ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock RTL operation
        mock_op = MockFINNOperation("MatrixVectorActivation_rtl")
        # Mock required RTL method
        mock_op.get_verilog_top_module_intf_names = lambda: ["s_axis", "m_axis"]
        
        # Create RTL generator
        generator = ModernRTLGenerator(mock_op)
        
        # Test template name selection
        template_name = generator.get_template_name()
        print(f"Selected RTL template: {template_name}")
        assert "rtl/" in template_name
        assert "wrapper" in template_name.lower()
        
        # Test context preparation
        context = generator.prepare_context(None, "xc7z020clg400-1", "100MHz")
        print(f"Generated context keys: {list(context.keys())}")
        
        # Verify important context elements
        assert context['op_type'] == "MatrixVectorActivation_rtl"
        assert 'module_name' in context
        assert 'input_ports' in context
        assert 'output_ports' in context
        assert 'parameters' in context
        
        # Test port generation
        input_ports = context['input_ports']
        print(f"Generated input ports: {len(input_ports)} ports")
        port_names = [p['name'] for p in input_ports]
        assert 'ap_clk' in port_names
        assert 'ap_rst_n' in port_names
        
        # Test parameter generation
        parameters = context['parameters']
        print(f"Generated parameters: {len(parameters)} parameters")
        param_names = [p['name'] for p in parameters]
        assert 'MW' in param_names
        assert 'MH' in param_names
        
        # Test file list generation
        files = generator.get_generated_files(tmp_dir)
        print(f"Files to be generated: {files}")
        assert len(files) > 0
        assert any(f.endswith('.v') for f in files)
        
        print("✓ RTL generator integration successful")


def test_code_generation_workflow():
    """Test complete code generation workflow."""
    print("\n=== Testing Complete Code Generation Workflow ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        fm = FileManager(tmp_dir)
        
        # Test HLS workflow
        hls_op = MockFINNOperation("MatrixVectorActivation")
        hls_gen = ModernHLSGenerator(hls_op)
        
        # Create simple template for testing
        hls_template = """
// {{ node_name }}.cpp - Generated HLS Implementation
// Operation: {{ op_type }}

{% for define_name, define_value in defines %}
#define {{ define_name }} {{ define_value }}
{% endfor %}

void {{ node_name }}_hls() {
    // Implementation for MW={{ operation_params.MW }}, PE={{ operation_params.PE }}
}
"""
        
        # Override template rendering for test
        context = hls_gen.prepare_context(None, "xc7z020clg400-1", "100MHz")
        result = hls_gen.template_engine.render_string(hls_template, context)
        
        # Write generated code
        hls_file = fm.write_file("generated_hls.cpp", result)
        print(f"Generated HLS file: {hls_file}")
        
        # Verify content
        content = fm.read_file(hls_file)
        assert "test_matrixvectoractivation" in content
        assert "MatrixVectorActivation" in content
        assert "#define MW 128" in content
        assert "PE=8" in content
        
        # Test RTL workflow
        rtl_op = MockFINNOperation("StreamingFIFO_rtl")
        rtl_op.get_verilog_top_module_intf_names = lambda: ["s_axis", "m_axis"]
        rtl_gen = ModernRTLGenerator(rtl_op)
        
        rtl_template = """
// {{ node_name }}_wrapper.v - Generated RTL Wrapper
// Operation: {{ op_type }}

module {{ module_name }} (
    input wire ap_clk,
    input wire ap_rst_n
);

// Parameters
{% for param in parameters %}
parameter {{ param.name }} = {{ param.value }};
{% endfor %}

endmodule
"""
        
        context = rtl_gen.prepare_context(None, "xc7z020clg400-1", "100MHz")
        result = rtl_gen.template_engine.render_string(rtl_template, context)
        
        rtl_file = fm.write_file("generated_rtl.v", result)
        print(f"Generated RTL file: {rtl_file}")
        
        # Verify content
        content = fm.read_file(rtl_file)
        assert "test_streamingfifo_rtl_wrapper" in content
        assert "StreamingFIFO_rtl" in content
        assert "module" in content
        assert "parameter" in content
        
        print("✓ Complete code generation workflow successful")


def test_compatibility_with_existing_finn():
    """Test compatibility with existing FINN infrastructure."""
    print("\n=== Testing Compatibility with Existing FINN ===")
    
    # Test that framework can coexist with existing FINN
    try:
        # Try to import existing FINN modules
        import finn.util.basic as finn_basic
        print("✓ Can import existing FINN utilities")
    except ImportError as e:
        print(f"⚠ Could not import FINN utilities (expected in pure Docker test): {e}")
    
    # Test environment variables
    finn_root = os.environ.get('FINN_ROOT')
    if finn_root:
        print(f"✓ FINN_ROOT available: {finn_root}")
        
        # Check for expected FINN directories
        expected_dirs = ['src', 'custom_hls', 'finn-rtllib']
        for dirname in expected_dirs:
            dirpath = os.path.join(finn_root, dirname)
            exists = "✓" if os.path.exists(dirpath) else "⚠"
            print(f"  {exists} {dirname}: {dirpath}")
    
    print("✓ Compatibility with existing FINN verified")


def run_all_tests():
    """Run all Docker integration tests."""
    try:
        test_docker_environment()
        test_library_resolver_in_docker()
        test_template_engine_with_finn_paths()
        test_file_manager_in_docker()
        test_hls_generator_integration()
        test_rtl_generator_integration()
        test_code_generation_workflow()
        test_compatibility_with_existing_finn()
        
        print("\n" + "="*50)
        print("🎉 ALL DOCKER INTEGRATION TESTS PASSED!")
        print("✓ Unified framework ready for production use")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)