"""
Tests for Enhanced Template Engine - Phase 2

Tests the enhanced template engine functionality including placeholder detection,
dependency resolution, validation, caching, and performance features.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
try:
    from finn.codegen.template_engine import TemplateEngine
    from finn.codegen.template_registry import TemplateRegistry, TemplateMetadata, TemplateTarget, TemplateComplexity
    from finn.codegen.codegen_config import CodegenConfigManager, CodegenConfig, LogLevel
    from finn.codegen.codegen_utils import validate_template_values, validate_identifier, format_cpp_code
    import jinja2
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Module imports failed: {import_error}")
class TestEnhancedTemplateEngine:
    """Test enhanced template engine features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = os.path.join(self.temp_dir, 'templates')
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Create test templates
        self._create_test_templates()
        
        # Initialize template engine
        self.engine = TemplateEngine([self.template_dir])
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_templates(self):
        """Create test templates for testing."""
        templates = {
            'simple.cpp.j2': '''
// Simple template
#define {{ name | upper }} {{ value }}
int main() {
    return {{ exit_code | default(0) }};
}
''',
            'with_dependencies.cpp.j2': '''
{% extends "base.cpp.j2" %}
{% block content %}
// Operation: {{ operation_name }}
// Parameters: {{ pe_factor }}, {{ simd_factor }}
{% endblock %}
''',
            'base.cpp.j2': '''
// Base template
{% block content %}
// Default content
{% endblock %}
''',
            'complex.cpp.j2': '''
// Complex template with loops and conditions
{% for i in range(num_ports) %}
input_{{ i }}: {{ data_width }} bits
{% endfor %}

{% if memory_mode == "streaming" %}
// Streaming implementation
{% elif memory_mode == "parallel" %}
// Parallel implementation
{% else %}
// Default implementation
{% endif %}
''',
            'invalid_syntax.cpp.j2': '''
// Template with syntax error
{{ unclosed_expression
''',
        }
        
        for filename, content in templates.items():
            with open(os.path.join(self.template_dir, filename), 'w') as f:
                f.write(content.strip())
    
    def test_get_required_placeholders_simple(self):
        """Test placeholder detection for simple template."""
        placeholders = self.engine.get_required_placeholders('simple.cpp.j2')
        
        expected = {'name', 'value', 'exit_code'}
        assert placeholders == expected
    
    def test_get_required_placeholders_complex(self):
        """Test placeholder detection for complex template."""
        placeholders = self.engine.get_required_placeholders('complex.cpp.j2')
        
        expected = {'num_ports', 'data_width', 'memory_mode'}
        assert placeholders == expected
    
    def test_get_required_placeholders_caching(self):
        """Test that placeholder detection results are cached."""
        # First call
        placeholders1 = self.engine.get_required_placeholders('simple.cpp.j2')
        
        # Second call should use cache
        with patch.object(self.engine.jinja_env, 'parse') as mock_parse:
            placeholders2 = self.engine.get_required_placeholders('simple.cpp.j2')
            
        # Parse should not be called on second access (cached)
        mock_parse.assert_not_called()
        assert placeholders1 == placeholders2
    
    def test_get_template_dependencies(self):
        """Test template dependency detection."""
        dependencies = self.engine.get_template_dependencies('with_dependencies.cpp.j2')
        
        expected = {'base.cpp.j2'}
        assert dependencies == expected
    
    def test_validate_template_valid(self):
        """Test template validation with valid inputs."""
        context = {
            'name': 'TEST_DEFINE',
            'value': 42,
            'exit_code': 0
        }
        
        is_valid, errors = self.engine.validate_template('simple.cpp.j2', context)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_template_missing_placeholders(self):
        """Test template validation with missing placeholders."""
        context = {
            'name': 'TEST_DEFINE'
            # Missing 'value' and 'exit_code'
        }
        
        is_valid, errors = self.engine.validate_template('simple.cpp.j2', context)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any('Missing required placeholders' in error for error in errors)
    
    def test_validate_template_missing_dependencies(self):
        """Test template validation with missing dependencies."""
        # Remove base template to create missing dependency
        base_path = os.path.join(self.template_dir, 'base.cpp.j2')
        os.remove(base_path)
        
        context = {
            'operation_name': 'TestOp',
            'pe_factor': 4,
            'simd_factor': 1
        }
        
        is_valid, errors = self.engine.validate_template('with_dependencies.cpp.j2', context)
        
        assert is_valid is False
        assert any('Missing template dependency' in error for error in errors)
    
    def test_validate_template_syntax_error(self):
        """Test template validation with syntax errors."""
        context = {}
        
        is_valid, errors = self.engine.validate_template('invalid_syntax.cpp.j2', context)
        
        assert is_valid is False
        assert any('Template syntax error' in error for error in errors)
    
    def test_render_with_validation_success(self):
        """Test successful render with validation."""
        context = {
            'name': 'TEST_VALUE',
            'value': 123,
            'exit_code': 0
        }
        
        result = self.engine.render_with_validation('simple.cpp.j2', context)
        
        assert 'TEST_VALUE' in result
        assert '123' in result
        assert 'return 0' in result
    
    def test_render_with_validation_failure(self):
        """Test render with validation failure."""
        context = {
            'name': 'TEST_VALUE'
            # Missing required placeholders
        }
        
        with pytest.raises(ValueError) as exc_info:
            self.engine.render_with_validation('simple.cpp.j2', context)
        
        assert 'Template validation failed' in str(exc_info.value)
    
    def test_performance_logging(self):
        """Test that performance metrics are logged."""
        context = {
            'name': 'PERF_TEST',
            'value': 999,
            'exit_code': 1
        }
        
        with patch.object(self.engine, 'logger') as mock_logger:
            result = self.engine.render_with_validation('simple.cpp.j2', context)
            
        # Check that performance was logged
        debug_calls = [call for call in mock_logger.debug.call_args_list 
                      if 'rendered in' in str(call)]
        assert len(debug_calls) > 0
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Populate caches
        self.engine.get_required_placeholders('simple.cpp.j2')
        self.engine.get_template_dependencies('with_dependencies.cpp.j2')
        
        assert len(self.engine._placeholder_cache) > 0
        assert len(self.engine._dependency_cache) > 0
        
        # Clear caches
        self.engine.clear_cache()
        
        assert len(self.engine._placeholder_cache) == 0
        assert len(self.engine._dependency_cache) == 0
    
    def test_custom_filters(self):
        """Test FINN-specific Jinja2 filters."""
        template_content = '''
{{ value | format_define("TEST_VALUE") }}
{{ width | format_port("input", "test_port") }}
{{ "INT8" | cpp_type_name }}
'''
        
        context = {
            'value': 42,
            'width': 8
        }
        
        result = self.engine.render_string(template_content, context)
        
        assert '#define TEST_VALUE 42' in result
        assert 'input logic [7:0] test_port' in result
        assert 'ap_int<8>' in result


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Module imports failed: {import_error}")
class TestTemplateRegistry:
    """Test template registry functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = os.path.join(self.temp_dir, 'templates')
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Create test templates with metadata
        self._create_test_templates_with_metadata()
        
        # Initialize registry
        self.registry = TemplateRegistry()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_templates_with_metadata(self):
        """Create test templates with metadata files."""
        templates = {
            'hls_thresholding_basic.cpp.j2': {
                'content': '''
// @description: Basic HLS thresholding template
// @operations: Thresholding
// @requires: pe_factor, mem_mode
#include "thresholding.hpp"
void thresholding_hls(/* parameters */) {
    // Implementation
}
''',
                'metadata': {
                    'required_placeholders': ['pe_factor', 'mem_mode', 'ram_style'],
                    'optional_placeholders': ['simd_factor'],
                    'supported_operations': ['Thresholding'],
                    'dependencies': [],
                    'description': 'Basic HLS template for thresholding operations',
                    'performance': {'latency': 'low', 'throughput': 'medium'},
                    'resources': {'lut': 100, 'ff': 50},
                    'constraints': {'memory_type': 'distributed'}
                }
            },
            'rtl_mvau_streaming.v.j2': {
                'content': '''
// @description: Streaming RTL MVAU template
// @operations: MatrixVectorActivation, MVAU
// @requires: pe_factor, simd_factor, weights
module mvau_streaming (
    input clk,
    input rst_n
);
    // Implementation
endmodule
''',
                'metadata': {
                    'required_placeholders': ['pe_factor', 'simd_factor', 'weights'],
                    'optional_placeholders': ['activation_type'],
                    'supported_operations': ['MatrixVectorActivation', 'MVAU'],
                    'dependencies': [],
                    'description': 'Streaming RTL template for MVAU operations',
                    'performance': {'latency': 'high', 'throughput': 'high'},
                    'resources': {'lut': 500, 'bram': 10},
                    'constraints': {'memory_type': 'block'}
                }
            }
        }
        
        for filename, template_info in templates.items():
            # Write template file
            template_path = os.path.join(self.template_dir, filename)
            with open(template_path, 'w') as f:
                f.write(template_info['content'].strip())
            
            # Write metadata file
            metadata_path = template_path + '.meta.json'
            import json
            with open(metadata_path, 'w') as f:
                json.dump(template_info['metadata'], f, indent=2)
    
    def test_register_template(self):
        """Test manual template registration."""
        metadata = TemplateMetadata(
            name='test_template.cpp.j2',
            target=TemplateTarget.HLS,
            complexity=TemplateComplexity.BASIC,
            required_placeholders={'param1', 'param2'},
            supported_operations={'TestOp'}
        )
        
        self.registry.register_template(metadata)
        
        # Check registration
        registered = self.registry.get_template_metadata('test_template.cpp.j2')
        assert registered is not None
        assert registered.name == 'test_template.cpp.j2'
        assert registered.target == TemplateTarget.HLS
    
    def test_scan_template_directories(self):
        """Test automatic template discovery."""
        self.registry.scan_template_directories([self.template_dir])
        
        # Check that templates were discovered
        all_templates = self.registry.list_all_templates()
        assert len(all_templates) >= 2
        
        # Check specific templates
        hls_templates = self.registry.find_templates_by_target(TemplateTarget.HLS)
        rtl_templates = self.registry.find_templates_by_target(TemplateTarget.RTL)
        
        assert any('hls_thresholding_basic.cpp.j2' in template for template in hls_templates)
        assert any('rtl_mvau_streaming.v.j2' in template for template in rtl_templates)
    
    def test_find_templates_for_operation(self):
        """Test finding templates by operation."""
        self.registry.scan_template_directories([self.template_dir])
        
        # Find Thresholding templates
        thresholding_templates = self.registry.find_templates_for_operation('Thresholding')
        assert len(thresholding_templates) > 0
        
        # Find MVAU templates
        mvau_templates = self.registry.find_templates_for_operation('MatrixVectorActivation')
        assert len(mvau_templates) > 0
    
    def test_assess_template_compatibility(self):
        """Test template compatibility assessment."""
        self.registry.scan_template_directories([self.template_dir])
        
        # Test compatible values
        available_values = {'pe_factor', 'mem_mode', 'ram_style', 'simd_factor'}
        compatibility = self.registry.assess_template_compatibility(
            'hls_thresholding_basic.cpp.j2',
            available_values
        )
        
        assert compatibility.is_compatible is True
        assert compatibility.score > 0.5
        
        # Test incompatible values (missing required)
        incomplete_values = {'pe_factor'}  # Missing mem_mode, ram_style
        compatibility = self.registry.assess_template_compatibility(
            'hls_thresholding_basic.cpp.j2',
            incomplete_values
        )
        
        assert compatibility.is_compatible is False
        assert 'Missing required placeholders' in ' '.join(compatibility.reasons)
    
    def test_get_best_template(self):
        """Test intelligent template selection."""
        self.registry.scan_template_directories([self.template_dir])
        
        # Test best template for Thresholding
        available_values = {'pe_factor', 'mem_mode', 'ram_style', 'simd_factor'}
        best_template = self.registry.get_best_template(
            'Thresholding',
            TemplateTarget.HLS,
            available_values
        )
        
        assert best_template is not None
        assert 'thresholding' in best_template.lower()
        assert 'hls' in best_template.lower()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Module imports failed: {import_error}")
class TestCodegenConfig:
    """Test codegen configuration system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration loading."""
        config_manager = CodegenConfigManager()
        config = config_manager.get_config()
        
        assert isinstance(config, CodegenConfig)
        assert config.template_cache_enabled is True
        assert config.log_level == LogLevel.INFO
        assert config.debug_mode is False
    
    def test_config_from_file(self):
        """Test configuration loading from file."""
        # Create test config file
        config_data = {
            'debug_mode': True,
            'log_level': 'DEBUG',
            'template_cache_enabled': False,
            'strict_validation': True
        }
        
        import json
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = CodegenConfigManager(self.config_file)
        config = config_manager.get_config()
        
        assert config.debug_mode is True
        assert config.log_level == LogLevel.DEBUG
        assert config.template_cache_enabled is False
        assert config.strict_validation is True
    
    def test_environment_variable_override(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'FINN_CODEGEN_DEBUG': 'true',
            'FINN_CODEGEN_LOG_LEVEL': 'ERROR',
            'FINN_CODEGEN_TEMPLATE_CACHE': 'false'
        }):
            config_manager = CodegenConfigManager()
            config = config_manager.get_config()
            
            assert config.debug_mode is True
            assert config.log_level == LogLevel.ERROR
            assert config.template_cache_enabled is False
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config_manager = CodegenConfigManager()
        
        # Update configuration
        config_manager.update_config(
            debug_mode=True,
            log_level=LogLevel.WARNING,
            max_render_threads=8
        )
        
        # Save to file
        config_manager.save_to_file(self.config_file)
        
        # Load from file in new manager
        new_config_manager = CodegenConfigManager(self.config_file)
        new_config = new_config_manager.get_config()
        
        assert new_config.debug_mode is True
        assert new_config.log_level == LogLevel.WARNING
        assert new_config.max_render_threads == 8


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Module imports failed: {import_error}")
class TestCodegenUtils:
    """Test codegen utility functions."""
    
    def test_validate_template_values_success(self):
        """Test successful template value validation."""
        values = {
            'name': 'test_value',
            'count': 42,
            'enabled': True
        }
        
        required_keys = {'name', 'count'}
        type_constraints = {
            'name': str,
            'count': int,
            'enabled': bool
        }
        
        is_valid, errors = validate_template_values(values, required_keys, type_constraints)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_template_values_missing_keys(self):
        """Test validation with missing required keys."""
        values = {
            'name': 'test_value'
            # Missing 'count'
        }
        
        required_keys = {'name', 'count'}
        
        is_valid, errors = validate_template_values(values, required_keys)
        
        assert is_valid is False
        assert any('Missing required keys' in error for error in errors)
        assert 'count' in str(errors)
    
    def test_validate_template_values_type_mismatch(self):
        """Test validation with type mismatches."""
        values = {
            'name': 123,  # Should be string
            'count': 'not_a_number'  # Should be int
        }
        
        type_constraints = {
            'name': str,
            'count': int
        }
        
        is_valid, errors = validate_template_values(values, type_constraints=type_constraints)
        
        assert is_valid is False
        assert len(errors) >= 2
    
    def test_validate_identifier_valid(self):
        """Test valid identifier validation."""
        valid_identifiers = [
            'valid_name',
            'ValidName',
            '_private_var',
            'name123',
            'a',
            '_',
            'CamelCase'
        ]
        
        for identifier in valid_identifiers:
            assert validate_identifier(identifier) is True, f"Failed for: {identifier}"
    
    def test_validate_identifier_invalid(self):
        """Test invalid identifier validation."""
        invalid_identifiers = [
            '',  # Empty
            '123name',  # Starts with number
            'name-with-dash',  # Contains dash
            'name with space',  # Contains space
            'class',  # Reserved keyword
            'int',  # Reserved keyword
            'module',  # SystemVerilog keyword
            'name$special',  # Special character
        ]
        
        for identifier in invalid_identifiers:
            assert validate_identifier(identifier) is False, f"Should fail for: {identifier}"
    
    def test_format_cpp_code(self):
        """Test C++ code formatting."""
        unformatted_code = '''
class TestClass {
public:
int getValue() {
return 42;
}
private:
int value;
};
'''
        
        formatted = format_cpp_code(unformatted_code)
        
        # Check basic indentation
        lines = formatted.split('\n')
        # First brace should increase indentation
        assert lines[1].startswith('  public:')  # 2 spaces
        assert lines[2].startswith('    int getValue()') # 4 spaces
    
    def test_format_cpp_code_custom_indent(self):
        """Test C++ code formatting with custom indentation."""
        code = '''
if (condition) {
doSomething();
}
'''
        
        formatted = format_cpp_code(code, indent_size=4)
        
        lines = formatted.split('\n')
        # Should use 4 spaces per level
        assert '    doSomething();' in formatted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])