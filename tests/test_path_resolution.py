############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Path Resolution Tests
############################################################################

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add the finn paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finn.util.flexible_hls import (
    FINNPathResolver,
    FINNPathType,
    FINNConfig
)


class TestFINNPathResolver:
    """Test FINN-specific path resolution functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN directory structure
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        self.hls_path = os.path.join(self.temp_dir, "hls")
        self.vitis_path = os.path.join(self.temp_dir, "vitis")
        
        # Create directories
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        os.makedirs(self.hls_path)
        os.makedirs(self.vitis_path)
        
        # Create mock files
        self._create_mock_files()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_files(self):
        """Create mock FINN files for testing."""
        # Create finn-hlslib files
        hlslib_files = [
            "bnn-library.h",
            "cnpy.h",
            "npy2apintstream.hpp",
            "npy2vectorstream.hpp"
        ]
        
        for filename in hlslib_files:
            filepath = os.path.join(self.finn_hlslib, filename)
            with open(filepath, 'w') as f:
                f.write(f"// Mock {filename} file\n")
        
        # Create custom HLS directory
        custom_hls_dir = os.path.join(self.finn_root, "custom_hls")
        os.makedirs(custom_hls_dir)
        
        custom_hls_files = [
            "matrix_vector_activate.h",
            "sliding_window.h",
            "streamtools.h"
        ]
        
        for filename in custom_hls_files:
            filepath = os.path.join(custom_hls_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"// Mock {filename} file\n")
        
        # Create HLS/Vitis mock files
        hls_include_dir = os.path.join(self.hls_path, "include")
        os.makedirs(hls_include_dir)
        
        with open(os.path.join(hls_include_dir, "hls_stream.h"), 'w') as f:
            f.write("// Mock HLS stream header\n")
        
        vitis_include_dir = os.path.join(self.vitis_path, "include")
        os.makedirs(vitis_include_dir)
        
        with open(os.path.join(vitis_include_dir, "ap_int.h"), 'w') as f:
            f.write("// Mock AP int header\n")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_default_environment_detection(self):
        """Test detection of FINN environment variables."""
        # Test with no environment variables set
        resolver = FINNPathResolver()
        
        # Should use defaults
        assert resolver.get_finn_root() is not None
        assert resolver.get_finn_deps_dir() is not None
        assert resolver.get_finn_hlslib_dir() is not None
    
    @patch.dict(os.environ, {
        'FINN_ROOT': '/custom/finn/root',
        'FINN_DEPS_DIR': '/custom/finn/deps',
        'FINN_HLSLIB_DIR': '/custom/finn/hlslib',
        'HLS_PATH': '/custom/hls',
        'VITIS_PATH': '/custom/vitis'
    }, clear=True)
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        resolver = FINNPathResolver()
        
        assert resolver.get_finn_root() == '/custom/finn/root'
        assert resolver.get_finn_deps_dir() == '/custom/finn/deps'
        assert resolver.get_finn_hlslib_dir() == '/custom/finn/hlslib'
        assert resolver.get_hls_path() == '/custom/hls'
        assert resolver.get_vitis_path() == '/custom/vitis'
    
    @patch.dict(os.environ, {
        'FINN_ROOT': None,
        'FINN_DEPS_DIR': None,
        'FINN_HLSLIB_DIR': None
    }, clear=True)
    def test_path_resolution_with_mock_env(self):
        """Test path resolution with mock environment."""
        # Set up environment for testing
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib,
            'HLS_PATH': self.hls_path,
            'VITIS_PATH': self.vitis_path
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test basic path resolution
            assert resolver.get_finn_root() == self.finn_root
            assert resolver.get_finn_deps_dir() == self.finn_deps
            assert resolver.get_finn_hlslib_dir() == self.finn_hlslib
            assert resolver.get_hls_path() == self.hls_path
            assert resolver.get_vitis_path() == self.vitis_path
    
    def test_path_type_resolution(self):
        """Test resolution of different FINN path types."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib,
            'HLS_PATH': self.hls_path,
            'VITIS_PATH': self.vitis_path
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test different path types
            finn_path = resolver.resolve_path("custom_hls", FINNPathType.FINN_ROOT)
            expected_path = os.path.join(self.finn_root, "custom_hls")
            assert finn_path == expected_path
            
            hlslib_path = resolver.resolve_path("bnn-library.h", FINNPathType.FINN_HLSLIB)
            expected_path = os.path.join(self.finn_hlslib, "bnn-library.h")
            assert hlslib_path == expected_path
            
            deps_path = resolver.resolve_path("some_dependency", FINNPathType.FINN_DEPS)
            expected_path = os.path.join(self.finn_deps, "some_dependency")
            assert deps_path == expected_path
    
    def test_include_path_generation(self):
        """Test generation of include paths for compilation."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib,
            'HLS_PATH': self.hls_path,
            'VITIS_PATH': self.vitis_path
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test include path generation
            include_paths = resolver.get_include_paths()
            
            assert isinstance(include_paths, list)
            assert len(include_paths) > 0
            
            # Should include FINN HLS lib
            assert self.finn_hlslib in include_paths
            
            # Should include custom HLS directory
            custom_hls_path = os.path.join(self.finn_root, "custom_hls")
            assert custom_hls_path in include_paths
            
            # Should include HLS include directory if it exists
            hls_include_path = os.path.join(self.hls_path, "include")
            if os.path.exists(hls_include_path):
                assert hls_include_path in include_paths
    
    def test_library_path_generation(self):
        """Test generation of library paths for linking."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib,
            'HLS_PATH': self.hls_path,
            'VITIS_PATH': self.vitis_path
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test library path generation
            library_paths = resolver.get_library_paths()
            
            assert isinstance(library_paths, list)
            
            # Should include relevant library directories
            if self.hls_path:
                hls_lib_path = os.path.join(self.hls_path, "lib")
                if os.path.exists(hls_lib_path):
                    assert hls_lib_path in library_paths
    
    def test_path_validation(self):
        """Test validation of resolved paths."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps, 
            'FINN_HLSLIB_DIR': self.finn_hlslib,
            'HLS_PATH': self.hls_path,
            'VITIS_PATH': self.vitis_path
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test existing path validation
            assert resolver.validate_path(self.finn_root)
            assert resolver.validate_path(self.finn_hlslib)
            
            # Test non-existing path validation
            non_existing_path = os.path.join(self.temp_dir, "non_existing")
            assert not resolver.validate_path(non_existing_path)
    
    def test_file_existence_checking(self):
        """Test checking for existence of specific FINN files."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib,
            'HLS_PATH': self.hls_path,
            'VITIS_PATH': self.vitis_path
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test checking for existing files
            assert resolver.file_exists("bnn-library.h", FINNPathType.FINN_HLSLIB)
            assert resolver.file_exists("cnpy.h", FINNPathType.FINN_HLSLIB)
            
            # Test checking for non-existing files
            assert not resolver.file_exists("non_existing.h", FINNPathType.FINN_HLSLIB)
    
    def test_relative_path_resolution(self):
        """Test resolution of relative paths within FINN structure."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test relative path resolution
            rel_path = "custom_hls/matrix_vector_activate.h"
            abs_path = resolver.resolve_path(rel_path, FINNPathType.FINN_ROOT)
            expected_path = os.path.join(self.finn_root, rel_path)
            assert abs_path == expected_path
    
    def test_path_normalization(self):
        """Test path normalization for cross-platform compatibility."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test path normalization with different separators
            test_paths = [
                "custom_hls/test.h",
                "custom_hls\\test.h",  # Windows-style
                "custom_hls//test.h",  # Double separator
                "./custom_hls/test.h"  # Relative with dot
            ]
            
            for path in test_paths:
                normalized = resolver.normalize_path(path)
                assert os.path.normpath(path) == normalized


class TestFINNPathIntegration:
    """Test integration of path resolver with FINN configuration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN structure
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        
        # Create config files
        with open(os.path.join(self.finn_hlslib, "bnn-library.h"), 'w') as f:
            f.write("// Mock BNN library\n")
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_path_integration(self):
        """Test integration between path resolver and configuration."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Create config with path resolver
            config = FINNConfig()
            resolver = FINNPathResolver()
            
            # Test that config can use resolved paths
            include_paths = resolver.get_include_paths()
            config.set_include_paths(include_paths)
            
            retrieved_paths = config.get_include_paths()
            assert retrieved_paths == include_paths
            assert self.finn_hlslib in retrieved_paths
    
    def test_template_path_resolution(self):
        """Test path resolution for template files."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            # Test template directory resolution
            template_paths = resolver.get_template_search_paths()
            assert isinstance(template_paths, list)
            assert len(template_paths) > 0
            
            # Should include standard template locations
            finn_template_path = resolver.resolve_path("util/flexible_hls/templates", FINNPathType.FINN_ROOT)
            assert finn_template_path in template_paths


class TestFINNEnvironmentDetection:
    """Test detection and validation of FINN environment setup."""
    
    def test_finn_environment_validation(self):
        """Test validation of complete FINN environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal FINN environment
            finn_root = os.path.join(temp_dir, "finn")
            finn_deps = os.path.join(temp_dir, "finn_deps")
            finn_hlslib = os.path.join(finn_deps, "finn-hlslib")
            
            os.makedirs(finn_root)
            os.makedirs(finn_hlslib)
            
            # Create essential files
            with open(os.path.join(finn_hlslib, "bnn-library.h"), 'w') as f:
                f.write("// BNN library")
            
            test_env = {
                'FINN_ROOT': finn_root,
                'FINN_DEPS_DIR': finn_deps,
                'FINN_HLSLIB_DIR': finn_hlslib
            }
            
            with patch.dict(os.environ, test_env):
                resolver = FINNPathResolver()
                
                # Test environment validation
                is_valid = resolver.validate_finn_environment()
                assert is_valid, "FINN environment should be valid"
                
                # Test individual component validation
                assert resolver.validate_path(resolver.get_finn_root())
                assert resolver.validate_path(resolver.get_finn_deps_dir())
                assert resolver.validate_path(resolver.get_finn_hlslib_dir())
    
    def test_incomplete_environment_detection(self):
        """Test detection of incomplete FINN environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create incomplete environment (missing hlslib)
            finn_root = os.path.join(temp_dir, "finn")
            os.makedirs(finn_root)
            
            test_env = {
                'FINN_ROOT': finn_root,
                'FINN_DEPS_DIR': '/non/existing/path',
                'FINN_HLSLIB_DIR': '/non/existing/hlslib'
            }
            
            with patch.dict(os.environ, test_env):
                resolver = FINNPathResolver()
                
                # Should detect incomplete environment
                is_valid = resolver.validate_finn_environment()
                assert not is_valid, "Incomplete FINN environment should be invalid"
    
    def test_environment_diagnostics(self):
        """Test diagnostic information for environment issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_env = {
                'FINN_ROOT': '/non/existing/root',
                'FINN_DEPS_DIR': '/non/existing/deps',
                'FINN_HLSLIB_DIR': '/non/existing/hlslib'
            }
            
            with patch.dict(os.environ, test_env):
                resolver = FINNPathResolver()
                
                # Get diagnostic information
                diagnostics = resolver.get_environment_diagnostics()
                
                assert isinstance(diagnostics, dict)
                assert 'finn_root' in diagnostics
                assert 'finn_deps_dir' in diagnostics
                assert 'finn_hlslib_dir' in diagnostics
                
                # Should indicate missing paths
                assert not diagnostics['finn_root']['exists']
                assert not diagnostics['finn_deps_dir']['exists'] 
                assert not diagnostics['finn_hlslib_dir']['exists']


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])