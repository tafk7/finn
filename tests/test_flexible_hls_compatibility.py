############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN FlexibleHLS Compatibility Tests
############################################################################

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add the finn paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import FINN modules
from finn.util.flexible_hls import (
    FlexibleHLSBackend, 
    FINNHLSConfig,
    FINNPathResolver,
    FINNTemplateEngine,
    FINNCompilerBuilder,
    validate_finn_environment
)
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow import templates


class TestFlexibleHLSCompatibility:
    """Test compatibility between FlexibleHLSBackend and original HLSBackend."""
    
    def test_hlsbackend_is_flexible_backend(self):
        """Test that HLSBackend is now FlexibleHLSBackend."""
        # HLSBackend should now inherit from FlexibleHLSBackend
        assert issubclass(HLSBackend, FlexibleHLSBackend)
    
    def test_flexible_backend_api_compatibility(self):
        """Test that FlexibleHLSBackend has all required HLSBackend methods."""
        required_methods = [
            'get_nodeattr_types',
            'get_all_verilog_paths', 
            'get_all_verilog_filenames',
            'prepare_rtlsim',
            'code_generation_ipgen',
            'ipgen_default_directives',
            'ipgen_extra_directives',
            'ipgen_singlenode_code',
            'code_generation_cppsim',
            'code_generation_ipi',
            'compile_singlenode_code',
            'npy_to_dynamic_output',
            'exec_precompiled_singlenode_model',
            'execute_node',
            'global_includes',
            'defines',
            'read_npy_data',
            'strm_decl',
            'docompute',
            'dataoutstrm',
            'save_as_npy',
            'blackboxfunction',
            'pragmas',
            'get_ap_int_max_w',
            'timeout_value',
            'timeout_condition',
            'timeout_read_stream'
        ]
        
        for method_name in required_methods:
            assert hasattr(FlexibleHLSBackend, method_name), f"Missing method: {method_name}"
    
    def test_node_attr_types_compatibility(self):
        """Test that nodeattr types are compatible."""
        # Create a mock backend to test (since FlexibleHLSBackend is abstract)
        class MockFlexibleBackend(FlexibleHLSBackend):
            def global_includes(self): pass
            def defines(self, var): pass
            def docompute(self): pass
            def blackboxfunction(self): pass
        
        backend = MockFlexibleBackend()
        attr_types = backend.get_nodeattr_types()
        
        expected_attrs = {
            "code_gen_dir_cppsim",
            "executable_path", 
            "res_hls",
            "cpp_interface"
        }
        
        for attr in expected_attrs:
            assert attr in attr_types, f"Missing node attribute type: {attr}"


class TestFINNEnvironmentDetection:
    """Test FINN environment detection and path resolution."""
    
    def test_finn_environment_validation(self):
        """Test FINN environment validation."""
        report = validate_finn_environment()
        
        assert "valid" in report
        assert "issues" in report
        assert "summary" in report
        assert isinstance(report["issues"], list)
    
    def test_path_resolver_initialization(self):
        """Test that FINNPathResolver initializes without errors."""
        resolver = FINNPathResolver()
        
        assert hasattr(resolver, 'env_config')
        assert hasattr(resolver, 'registry')
        assert hasattr(resolver, 'discovery')
    
    def test_path_resolver_methods(self):
        """Test FINNPathResolver key methods."""
        resolver = FINNPathResolver()
        
        # Test method existence
        assert hasattr(resolver, 'get_finn_include_paths')
        assert hasattr(resolver, 'get_finn_source_paths')
        assert hasattr(resolver, 'validate_finn_environment')
        
        # Test method calls don't error
        include_paths = resolver.get_finn_include_paths()
        source_paths = resolver.get_finn_source_paths()
        env_issues = resolver.validate_finn_environment()
        
        assert isinstance(include_paths, list)
        assert isinstance(source_paths, list)
        assert isinstance(env_issues, list)


class TestTemplateEngineCompatibility:
    """Test template engine compatibility with FINN templates."""
    
    def test_template_engine_initialization(self):
        """Test that FINNTemplateEngine initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = FINNTemplateEngine([temp_dir])
            
            assert hasattr(engine, 'loader')
            assert hasattr(engine, 'resolver')
    
    def test_template_migration_compatibility(self):
        """Test template migration maintains compatibility."""
        # Test that templates module has required attributes
        assert hasattr(templates, 'docompute_template')
        assert hasattr(templates, 'docompute_template_timeout')
        assert hasattr(templates, 'ipgen_template')
        assert hasattr(templates, 'ipgentcl_template')
        
        # Test that template properties work
        docompute = templates.docompute_template
        assert isinstance(docompute, str)
        assert "$AP_INT_MAX_W$" in docompute


class TestCompilerBuilderCompatibility:
    """Test compiler builder compatibility."""
    
    def test_compiler_builder_initialization(self):
        """Test FINNCompilerBuilder initializes correctly."""
        builder = FINNCompilerBuilder()
        
        assert hasattr(builder, 'path_resolver')
        assert hasattr(builder, 'include_manager')
    
    def test_build_configuration(self):
        """Test build configuration works."""
        from finn.util.flexible_hls import FINNBuildConfiguration
        
        config = FINNBuildConfiguration()
        assert hasattr(config, 'strategy')
        assert hasattr(config, 'optimization_level')
        assert hasattr(config, 'debug_mode')


class TestConfigurationCompatibility:
    """Test configuration system compatibility."""
    
    def test_finn_config_initialization(self):
        """Test FINNHLSConfig initializes correctly."""
        config = FINNHLSConfig()
        
        assert hasattr(config, 'compiler')
        assert hasattr(config, 'templates')
        assert hasattr(config, 'paths')
    
    def test_config_expanded_paths(self):
        """Test configuration path expansion."""
        config = FINNHLSConfig()
        
        # Should not error even if environment variables are missing
        expanded = config.get_expanded_paths()
        assert hasattr(expanded, 'compiler')
        assert hasattr(expanded, 'templates')
        assert hasattr(expanded, 'paths')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = FINNHLSConfig()
        issues = config.validate_finn_environment()
        
        assert isinstance(issues, list)
        # Issues are expected since we may not have full FINN environment


class TestIntegrationCompatibility:
    """Test overall integration compatibility."""
    
    def test_flexible_backend_instantiation(self):
        """Test that FlexibleHLSBackend can be instantiated with config."""
        class MockFlexibleBackend(FlexibleHLSBackend):
            def global_includes(self): pass
            def defines(self, var): pass
            def docompute(self): pass
            def blackboxfunction(self): pass
        
        config = FINNHLSConfig()
        backend = MockFlexibleBackend(config=config)
        
        assert backend.config is not None
        assert hasattr(backend, 'path_resolver')
        assert hasattr(backend, 'template_engine')
        assert hasattr(backend, 'compiler_builder')
    
    def test_code_gen_dict_compatibility(self):
        """Test that code_gen_dict works as expected."""
        class MockFlexibleBackend(FlexibleHLSBackend):
            def global_includes(self): pass
            def defines(self, var): pass
            def docompute(self): pass
            def blackboxfunction(self): pass
        
        backend = MockFlexibleBackend()
        
        # Should have code_gen_dict for FINN compatibility
        assert hasattr(backend, 'code_gen_dict')
        assert isinstance(backend.code_gen_dict, dict)
        
        # Should be able to add entries
        backend.code_gen_dict["$TEST$"] = ["test_value"]
        assert backend.code_gen_dict["$TEST$"] == ["test_value"]


# Test fixtures and utilities
@pytest.fixture
def temp_finn_env():
    """Create a temporary FINN-like environment for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create basic FINN directory structure
        finn_root = os.path.join(temp_dir, "finn")
        finn_deps = os.path.join(temp_dir, "deps")
        
        os.makedirs(os.path.join(finn_root, "src", "finn", "qnn-data", "cpp"))
        os.makedirs(os.path.join(finn_root, "custom_hls"))
        os.makedirs(os.path.join(finn_deps, "cnpy"))
        os.makedirs(os.path.join(finn_deps, "finn-hlslib"))
        
        # Create a dummy cnpy.cpp file
        with open(os.path.join(finn_deps, "cnpy", "cnpy.cpp"), "w") as f:
            f.write("// Dummy cnpy source\n")
        
        yield {
            "FINN_ROOT": finn_root,
            "FINN_DEPS_DIR": finn_deps,
        }


class TestEnvironmentIntegration:
    """Test integration with FINN environment."""
    
    def test_path_resolver_with_environment(self, temp_finn_env):
        """Test path resolver with mock FINN environment."""
        with patch.dict(os.environ, temp_finn_env):
            resolver = FINNPathResolver()
            
            include_paths = resolver.get_finn_include_paths()
            source_paths = resolver.get_finn_source_paths()
            
            # Should find at least some paths
            assert len(include_paths) > 0
            
            # Should find cnpy source
            cnpy_sources = [s for s in source_paths if "cnpy.cpp" in s]
            assert len(cnpy_sources) > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])