############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""Tests for FINN configuration system."""

import os
import tempfile
from pathlib import Path

import pytest

from finn.config import FINNConfig, get_config, reset_config
from finn.util.basic import (
    get_deps_dir,
    get_finn_root,
    get_liveness_threshold_cycles,
    get_rtlsim_trace_depth,
    get_vivado_root,
    make_build_dir,
)


class TestFINNConfig:
    """Test FINN configuration functionality."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()
        # Save original env vars
        self.orig_env = {}
        for key in ["FINN_ROOT", "FINN_DEPS_DIR", "FINN_BUILD_DIR", 
                    "XILINX_VIVADO", "RTLSIM_TRACE_DEPTH", "LIVENESS_THRESHOLD"]:
            self.orig_env[key] = os.environ.get(key)
    
    def teardown_method(self):
        """Restore original env vars after each test."""
        reset_config()
        for key, value in self.orig_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def test_auto_detect_finn_root(self):
        """Test that FINN_ROOT is auto-detected when not set."""
        # Clear FINN_ROOT
        os.environ.pop("FINN_ROOT", None)
        
        # Get config should auto-detect
        config = get_config()
        
        # Should have set FINN_ROOT
        assert config.finn_root.exists()
        assert (config.finn_root / "src" / "finn").exists()
        assert os.environ["FINN_ROOT"] == str(config.finn_root)
    
    def test_default_paths(self):
        """Test default path settings."""
        # Set only FINN_ROOT
        finn_root = Path(__file__).parent.parent.resolve()
        os.environ["FINN_ROOT"] = str(finn_root)
        os.environ.pop("FINN_DEPS_DIR", None)
        os.environ.pop("FINN_BUILD_DIR", None)
        
        config = get_config()
        
        # Check defaults
        assert config.finn_deps_dir == finn_root / "deps"
        assert config.finn_build_dir == Path("/tmp/finn_build")
        
        # Check env vars were set
        assert os.environ["FINN_DEPS_DIR"] == str(finn_root / "deps")
        assert os.environ["FINN_BUILD_DIR"] == "/tmp/finn_build"
    
    def test_xilinx_tools_config(self):
        """Test Xilinx tools configuration."""
        # Clear any existing tool paths
        os.environ.pop("HLS_PATH", None)
        os.environ.pop("XILINX_XRT", None)
        os.environ.pop("PLATFORM_REPO_PATHS", None)
        
        # Set some tool paths
        os.environ["XILINX_VIVADO"] = "/opt/xilinx/Vivado/2022.2"
        os.environ["VITIS_PATH"] = "/opt/xilinx/Vitis/2022.2"
        
        config = get_config()
        
        assert config.xilinx_tools.vivado_path == Path("/opt/xilinx/Vivado/2022.2")
        assert config.xilinx_tools.vitis_path == Path("/opt/xilinx/Vitis/2022.2")
        assert config.xilinx_tools.hls_path is None
    
    def test_runtime_config_defaults(self):
        """Test runtime configuration defaults."""
        # Clear runtime env vars
        for key in ["RTLSIM_TRACE_DEPTH", "LIVENESS_THRESHOLD", "NUM_DEFAULT_WORKERS"]:
            os.environ.pop(key, None)
        
        config = get_config()
        
        assert config.runtime.rtlsim_trace_depth == 1
        assert config.runtime.liveness_threshold == 1000000
        assert config.runtime.num_default_workers == 4
        
        # Check env vars were set
        assert os.environ["RTLSIM_TRACE_DEPTH"] == "1"
        assert os.environ["LIVENESS_THRESHOLD"] == "1000000"
        assert os.environ["NUM_DEFAULT_WORKERS"] == "4"
    
    def test_runtime_config_from_env(self):
        """Test runtime configuration from environment."""
        os.environ["RTLSIM_TRACE_DEPTH"] = "3"
        os.environ["LIVENESS_THRESHOLD"] = "5000"
        os.environ["NUM_DEFAULT_WORKERS"] = "8"
        
        config = get_config()
        
        assert config.runtime.rtlsim_trace_depth == 3
        assert config.runtime.liveness_threshold == 5000
        assert config.runtime.num_default_workers == 8
    
    def test_util_basic_functions(self):
        """Test that util.basic functions work with new config."""
        # Test get_finn_root
        assert Path(get_finn_root()).exists()
        
        # Test get_deps_dir
        assert get_deps_dir() == os.environ["FINN_DEPS_DIR"]
        
        # Test get_rtlsim_trace_depth
        os.environ["RTLSIM_TRACE_DEPTH"] = "2"
        reset_config()
        assert get_rtlsim_trace_depth() == 2
        
        # Test get_liveness_threshold_cycles
        os.environ["LIVENESS_THRESHOLD"] = "2000"
        reset_config()
        assert get_liveness_threshold_cycles() == 2000
        
        # Test make_build_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["FINN_BUILD_DIR"] = tmpdir
            reset_config()
            build_dir = make_build_dir(prefix="test_")
            assert build_dir.startswith(tmpdir)
            assert os.path.exists(build_dir)
    
    def test_get_vivado_root_error(self):
        """Test get_vivado_root raises error when not configured."""
        os.environ.pop("XILINX_VIVADO", None)
        reset_config()
        
        with pytest.raises(Exception, match="Vivado path not configured"):
            get_vivado_root()
    
    def test_invalid_finn_root(self):
        """Test error when FINN_ROOT is invalid."""
        os.environ["FINN_ROOT"] = "/nonexistent/path"
        reset_config()
        
        with pytest.raises(ValueError, match="FINN_ROOT does not exist"):
            get_config()
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test relative paths are rejected
        with pytest.raises(ValueError, match="Path must be absolute"):
            FINNConfig(
                finn_root=Path("relative/path"),
                finn_deps_dir=Path("/absolute/deps"),
                finn_build_dir=Path("/tmp/build")
            )
    
    def test_config_singleton(self):
        """Test that get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_import_finn_sets_env(self):
        """Test that importing finn sets environment variables."""
        # This test is tricky because finn may already be imported
        # Instead, test that the config system works when vars are unset
        
        # Save current values
        orig_root = os.environ.get("FINN_ROOT")
        orig_deps = os.environ.get("FINN_DEPS_DIR")
        
        try:
            # Clear env vars
            os.environ.pop("FINN_ROOT", None)
            os.environ.pop("FINN_DEPS_DIR", None)
            reset_config()
            
            # Get config (simulates what import finn does)
            config = get_config()
            
            # Check env vars are set
            assert "FINN_ROOT" in os.environ
            assert "FINN_DEPS_DIR" in os.environ
            assert "FINN_BUILD_DIR" in os.environ
            
            # Verify they match config
            assert os.environ["FINN_ROOT"] == str(config.finn_root)
            assert os.environ["FINN_DEPS_DIR"] == str(config.finn_deps_dir)
        finally:
            # Restore original values
            if orig_root:
                os.environ["FINN_ROOT"] = orig_root
            if orig_deps:
                os.environ["FINN_DEPS_DIR"] = orig_deps