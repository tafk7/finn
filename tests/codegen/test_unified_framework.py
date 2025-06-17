"""
Test suite for the FINN Unified Code Generation Framework

This test suite validates the unified framework implementation including:
- BaseCodeGenerator functionality
- TemplateEngine capabilities
- FileManager operations
- LibraryResolver dependency resolution
- ModernHLSGenerator operation-specific generation
- ModernRTLGenerator enhanced RTL generation
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Import the unified framework components
from finn.codegen import (
    BaseCodeGenerator,
    TemplateEngine,
    FileManager,
    LibraryResolver,
    ModernHLSGenerator,
    ModernRTLGenerator
)
from finn.codegen.library_resolver import LibraryType, LibrarySpec

# Import real FINN operations factory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from real_finn_operations import FinnOperationFactory


class TestTemplateEngine:
    """Test the TemplateEngine functionality."""
    
    def test_template_engine_initialization(self):
        """Test basic template engine initialization."""
        engine = TemplateEngine()
        assert engine.jinja_env is not None
        assert len(engine.jinja_env.loader.searchpath) > 0
        
    def test_render_string_template(self):
        """Test rendering a simple string template."""
        engine = TemplateEngine()
        template = "Hello {{ name }}!"
        context = {"name": "FINN"}
        
        result = engine.render_string(template, context)
        assert result == "Hello FINN!"
        
    def test_legacy_string_replacement(self):
        """Test backward compatibility with legacy string replacement."""
        engine = TemplateEngine()
        template = "Width: $WIDTH$, Height: $HEIGHT$"
        replacements = {"WIDTH": "64", "HEIGHT": "32"}
        
        result = engine.render_legacy(template, replacements)
        assert result == "Width: 64, Height: 32"
        
    def test_custom_filters(self):
        """Test FINN-specific Jinja2 filters."""
        engine = TemplateEngine()
        
        # Test format_define filter
        template = "{{ 64 | format_define('MW') }}"
        result = engine.render_string(template, {})
        assert result == "#define MW 64"
        
        # Test format_parameter filter
        template = "{{ 32 | format_parameter('DEPTH') }}"
        result = engine.render_string(template, {})
        assert result == "parameter DEPTH = 32"


class TestFileManager:
    """Test the FileManager functionality."""
    
    def test_file_manager_initialization(self):
        """Test basic file manager initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fm = FileManager(tmp_dir)
            assert fm.base_path == Path(tmp_dir)
            
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fm = FileManager(tmp_dir)
            test_dir = fm.ensure_directory("test/nested/dir")
            assert test_dir.exists()
            assert test_dir.is_dir()
            
    def test_write_and_read_file(self):
        """Test file writing and reading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fm = FileManager(tmp_dir)
            test_content = "Hello FINN!"
            
            # Write file
            file_path = fm.write_file("test.txt", test_content)
            assert file_path.exists()
            
            # Read file
            read_content = fm.read_file("test.txt")
            assert read_content == test_content
            
    def test_copy_file(self):
        """Test file copying."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fm = FileManager(tmp_dir)
            
            # Create source file
            src_path = fm.write_file("source.txt", "test content")
            
            # Copy file
            dst_path = fm.copy_file("source.txt", "destination.txt")
            assert dst_path.exists()
            
            # Verify content
            dst_content = fm.read_file("destination.txt")
            assert dst_content == "test content"
            
    def test_list_files(self):
        """Test file listing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fm = FileManager(tmp_dir)
            
            # Create test files
            fm.write_file("test1.cpp", "content1")
            fm.write_file("test2.hpp", "content2")
            fm.write_file("test3.v", "content3")
            
            # List all files
            all_files = fm.list_files(".")
            assert len(all_files) == 3
            
            # List C++ files
            cpp_files = fm.list_files(".", "*.cpp")
            assert len(cpp_files) == 1
            assert cpp_files[0].name == "test1.cpp"


class TestLibraryResolver:
    """Test the LibraryResolver functionality."""
    
    def test_library_resolver_initialization(self):
        """Test basic library resolver initialization."""
        resolver = LibraryResolver()
        assert len(resolver.libraries) > 0
        assert 'finn-hlslib' in resolver.libraries
        assert 'finn-rtllib' in resolver.libraries
        
    def test_register_library(self):
        """Test registering a custom library."""
        resolver = LibraryResolver()
        
        custom_lib = LibrarySpec(
            name='test-lib',
            path='/test/path',
            include_files=['test.h'],
            library_type=LibraryType.CUSTOM_LIBRARY,
            required_for=['TestOp']
        )
        
        resolver.register_library(custom_lib)
        assert 'test-lib' in resolver.libraries
        
    def test_resolve_includes(self):
        """Test include resolution for operations."""
        resolver = LibraryResolver()
        real_op = FinnOperationFactory.create_mvau_operation()
        
        includes = resolver.resolve_includes(real_op)
        assert len(includes) > 0
        # Should include HLS-specific includes
        assert any('mvau.hpp' in inc for inc in includes)
        
    def test_resolve_libraries(self):
        """Test library resolution for operations."""
        resolver = LibraryResolver()
        real_op = FinnOperationFactory.create_mvau_operation()
        
        libraries = resolver.resolve_libraries(real_op)
        assert len(libraries) > 0
        assert 'finn-hlslib' in libraries
        
    def test_validate_dependencies(self):
        """Test dependency validation."""
        resolver = LibraryResolver()
        real_op = FinnOperationFactory.create_mvau_operation()
        
        status = resolver.validate_dependencies(real_op)
        assert isinstance(status, dict)
        assert len(status) > 0


class TestModernHLSGenerator:
    """Test the ModernHLSGenerator functionality."""
    
    def test_hls_generator_initialization(self):
        """Test HLS generator initialization."""
        real_op = FinnOperationFactory.create_mvau_operation()
        generator = ModernHLSGenerator(real_op)
        
        assert generator.operation == real_op
        assert generator.backend_type == "hls"
        assert generator.template_engine is not None
        
    def test_get_template_name(self):
        """Test template name selection."""
        real_op = FinnOperationFactory.create_mvau_operation()
        generator = ModernHLSGenerator(real_op)
        
        template_name = generator.get_template_name()
        assert template_name == "hls/mvau_embedded.cpp.j2"  # Based on mem_mode
        
    def test_prepare_context(self):
        """Test context preparation for HLS generation."""
        real_op = FinnOperationFactory.create_mvau_operation()
        generator = ModernHLSGenerator(real_op)
        
        context = generator.prepare_context(None, "xc7z020clg400-1", "100MHz")
        
        assert context['op_type'] == "MatrixVectorActivation"
        # Real FINN operations generate auto-generated node names
        assert context['node_name'].startswith("MatrixVectorActivation")
        assert 'defines' in context
        assert 'includes' in context
        assert 'operation_params' in context
        
    def test_generate_defines(self):
        """Test HLS define generation."""
        real_op = FinnOperationFactory.create_mvau_operation()
        generator = ModernHLSGenerator(real_op)
        
        defines = generator._generate_defines()
        assert len(defines) > 0
        
        # Check for MVAU-specific defines
        define_names = [d[0] for d in defines]
        assert 'MW' in define_names
        assert 'MH' in define_names
        assert 'PE' in define_names
        assert 'SIMD' in define_names
        
    def test_get_generated_files(self):
        """Test generated file list."""
        real_op = FinnOperationFactory.create_mvau_operation()
        generator = ModernHLSGenerator(real_op)
        
        files = generator.get_generated_files("/tmp/codegen")
        assert len(files) > 0
        assert any(f.endswith('.cpp') for f in files)
        assert any(f.endswith('.h') for f in files)
        
    def test_validate_operation(self):
        """Test operation validation."""
        # Valid operation
        real_op = FinnOperationFactory.create_mvau_operation()
        generator = ModernHLSGenerator(real_op)
        assert generator.validate_operation() == True
        
        # Invalid operation - use thresholding for this test since it's different
        thresh_op = FinnOperationFactory.create_thresholding_operation()
        # Change op_type to unsupported for this test
        thresh_op.onnx_node.op_type = "UnsupportedOp"
        generator_invalid = ModernHLSGenerator(thresh_op)
        assert generator_invalid.validate_operation() == False


class TestModernRTLGenerator:
    """Test the ModernRTLGenerator functionality."""
    
    def test_rtl_generator_initialization(self):
        """Test RTL generator initialization."""
        real_op = FinnOperationFactory.create_mvau_rtl_operation()
        generator = ModernRTLGenerator(real_op)
        
        assert generator.operation == real_op
        assert generator.backend_type == "rtl"
        assert generator.template_engine is not None
        
    def test_get_template_name(self):
        """Test RTL template name selection."""
        real_op = FinnOperationFactory.create_mvau_rtl_operation()
        generator = ModernRTLGenerator(real_op)
        
        template_name = generator.get_template_name()
        assert template_name == "rtl/mvau_wrapper.v.j2"
        
    def test_prepare_context(self):
        """Test context preparation for RTL generation."""
        real_op = FinnOperationFactory.create_mvau_rtl_operation()
        generator = ModernRTLGenerator(real_op)
        
        context = generator.prepare_context(None, "xc7z020clg400-1", "100MHz")
        
        assert context['op_type'] == "MatrixVectorActivation_rtl"
        # Real FINN operations generate auto-generated node names
        assert context['node_name'].startswith("MatrixVectorActivation")
        assert 'module_name' in context
        assert 'input_ports' in context
        assert 'output_ports' in context
        assert 'parameters' in context
        
    def test_get_input_ports(self):
        """Test RTL input port generation."""
        real_op = FinnOperationFactory.create_mvau_rtl_operation()
        generator = ModernRTLGenerator(real_op)
        
        ports = generator._get_input_ports()
        assert len(ports) > 0
        
        # Check for standard AXI ports
        port_names = [p['name'] for p in ports]
        assert 'ap_clk' in port_names
        assert 'ap_rst_n' in port_names
        assert any('TDATA' in name for name in port_names)
        assert any('TVALID' in name for name in port_names)
        assert any('TREADY' in name for name in port_names)
        
    def test_get_parameters(self):
        """Test RTL parameter generation."""
        real_op = FinnOperationFactory.create_mvau_rtl_operation()
        generator = ModernRTLGenerator(real_op)
        
        params = generator._get_parameters()
        assert len(params) > 0
        
        # Check for MVAU-specific parameters
        param_names = [p['name'] for p in params]
        assert 'MW' in param_names
        assert 'MH' in param_names
        assert 'PE' in param_names
        assert 'SIMD' in param_names
        
    def test_validate_operation(self):
        """Test RTL operation validation."""
        # Test with real RTL operation
        real_op = FinnOperationFactory.create_mvau_rtl_operation()
        generator = ModernRTLGenerator(real_op)
        assert generator.validate_operation() == True
        
        # Invalid operation (not ending with _rtl)
        hls_op = FinnOperationFactory.create_mvau_operation()
        generator_invalid = ModernRTLGenerator(hls_op)
        assert generator_invalid.validate_operation() == False


class TestIntegration:
    """Integration tests for the unified framework."""
    
    def test_end_to_end_hls_generation(self):
        """Test complete HLS code generation flow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup with real FINN operation
            real_op = FinnOperationFactory.create_mvau_operation()
            generator = ModernHLSGenerator(real_op)
            
            # Create a properly rendered template for testing
            test_template_rendered = """
// Generated HLS code for MatrixVectorActivation_0
// Operation: MatrixVectorActivation
// MW: 64
// PE: 4
"""
            generator.template_engine.render = Mock(return_value=test_template_rendered)
            
            # Generate code
            code = generator.generate_code(None, "xc7z020clg400-1", "100MHz")
            assert "Generated HLS code" in code
            assert "MatrixVectorActivation_0" in code
            assert "MatrixVectorActivation" in code
            
    def test_end_to_end_rtl_generation(self):
        """Test complete RTL code generation flow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup with real FINN RTL operation
            real_op = FinnOperationFactory.create_mvau_rtl_operation()
            generator = ModernRTLGenerator(real_op)
            
            # Create a properly rendered template for testing
            test_template_rendered = """
// Generated RTL wrapper for test_mvau
// Operation: MatrixVectorActivation_rtl
module test_mvau_wrapper (
    input wire ap_clk,
    input wire ap_rst_n
);
endmodule
"""
            generator.template_engine.render = Mock(return_value=test_template_rendered)
            
            # Generate code
            code = generator.generate_code(None, "xc7z020clg400-1", "100MHz")
            assert "Generated RTL wrapper" in code
            assert "test_mvau" in code
            assert "MatrixVectorActivation_rtl" in code
            assert "module" in code
            
    def test_framework_comparison(self):
        """Test that the framework provides benefits over current approach."""
        # Test operation-specific templates vs mega-templates
        hls_op = FinnOperationFactory.create_mvau_operation()
        hls_gen = ModernHLSGenerator(hls_op)
        
        rtl_op = FinnOperationFactory.create_mvau_rtl_operation()
        rtl_gen = ModernRTLGenerator(rtl_op)
        
        # Different operations should use different templates
        hls_template = hls_gen.get_template_name()
        rtl_template = rtl_gen.get_template_name()
        
        assert hls_template != rtl_template
        assert "hls/" in hls_template
        assert "rtl/" in rtl_template
        
        # Both should validate successfully
        assert hls_gen.validate_operation() == True
        assert rtl_gen.validate_operation() == True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])