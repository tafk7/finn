############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN HLS Configuration System - FINN-specific configuration management
############################################################################

import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import json
from copy import deepcopy
from enum import Enum


logger = logging.getLogger(__name__)


class FINNConfigLevel(Enum):
    """FINN-specific configuration levels."""
    SYSTEM = "system"
    USER = "user"
    RUNTIME = "runtime"


@dataclass
class FINNCompilerSettings:
    """FINN HLS compiler-specific settings."""
    
    # Standard include paths (FINN-specific)
    include_paths: List[str] = field(default_factory=lambda: [
        "$FINN_ROOT/src/finn/qnn-data/cpp",
        "$FINN_DEPS_DIR/cnpy/",
        "$FINN_DEPS_DIR/finn-hlslib",
        "$FINN_ROOT/custom_hls"
    ])
    
    # Compiler flags (FINN defaults)
    cxx_standard: str = "c++14"
    optimization_level: str = "-O3"
    custom_flags: List[str] = field(default_factory=list)
    
    # Library linking (FINN-specific)
    link_libraries: List[str] = field(default_factory=lambda: ["z"])
    
    # HLS-specific settings
    clock_period: float = 10.0  # nanoseconds
    target_fpga: Optional[str] = None
    
    # FINN-specific compilation settings
    enable_debug_defines: bool = False
    ap_int_max_w: int = 8191
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "include_paths": self.include_paths,
            "cxx_standard": self.cxx_standard,
            "optimization_level": self.optimization_level,
            "custom_flags": self.custom_flags,
            "link_libraries": self.link_libraries,
            "clock_period": self.clock_period,
            "target_fpga": self.target_fpga,
            "enable_debug_defines": self.enable_debug_defines,
            "ap_int_max_w": self.ap_int_max_w
        }


@dataclass
class FINNTemplateSettings:
    """FINN template system configuration."""
    
    # Template search paths (FINN-specific)
    template_paths: List[str] = field(default_factory=list)
    
    # FINN template settings
    enable_caching: bool = True
    variable_prefix: str = "$"
    variable_suffix: str = "$"
    
    # FINN-specific template names
    ipgen_template_name: str = "ipgen_cpp"
    ipgen_tcl_template_name: str = "ipgen_tcl"
    docompute_template_name: str = "docompute"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "template_paths": self.template_paths,
            "enable_caching": self.enable_caching,
            "variable_prefix": self.variable_prefix,
            "variable_suffix": self.variable_suffix,
            "ipgen_template_name": self.ipgen_template_name,
            "ipgen_tcl_template_name": self.ipgen_tcl_template_name,
            "docompute_template_name": self.docompute_template_name
        }


@dataclass
class FINNPathSettings:
    """FINN path resolution configuration."""
    
    # FINN environment variables
    finn_root: Optional[str] = None
    finn_deps_dir: Optional[str] = None
    finn_hlslib_dir: Optional[str] = None
    
    # Tool paths
    vitis_path: Optional[str] = None
    hls_path: Optional[str] = None
    
    # Path resolution settings
    enable_path_caching: bool = True
    auto_discover_libraries: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "finn_root": self.finn_root,
            "finn_deps_dir": self.finn_deps_dir,
            "finn_hlslib_dir": self.finn_hlslib_dir,
            "vitas_path": self.vitis_path,
            "hls_path": self.hls_path,
            "enable_path_caching": self.enable_path_caching,
            "auto_discover_libraries": self.auto_discover_libraries
        }


class FINNHLSConfig:
    """Main FINN HLS configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.compiler: FINNCompilerSettings = FINNCompilerSettings()
        self.templates: FINNTemplateSettings = FINNTemplateSettings()
        self.paths: FINNPathSettings = FINNPathSettings()
        
        # Configuration metadata
        self.config_level: FINNConfigLevel = FINNConfigLevel.RUNTIME
        self.config_source: Optional[str] = None
        self.custom_settings: Dict[str, Any] = {}
        
        if config_file:
            self.load_from_file(config_file)
        else:
            self._load_finn_defaults()
    
    def _load_finn_defaults(self):
        """Load FINN-specific default configuration."""
        # Compiler settings are already set in FINNCompilerSettings default factory
        
        # Set up FINN template paths
        self.templates = FINNTemplateSettings(
            template_paths=[
                # Will be populated by template engine based on search paths
            ]
        )
        
        # Set up FINN path settings from environment
        self.paths = FINNPathSettings(
            finn_root=os.environ.get('FINN_ROOT'),
            finn_deps_dir=os.environ.get('FINN_DEPS_DIR'),
            finn_hlslib_dir=os.environ.get('FINN_HLSLIB_DIR'),
            vitis_path=os.environ.get('VITIS_PATH'),
            hls_path=os.environ.get('HLS_PATH')
        )
        
        self.config_level = FINNConfigLevel.SYSTEM
        self.config_source = "finn_defaults"
        
        logger.debug("Loaded FINN default configuration")
    
    def load_from_file(self, config_file: str):
        """Load FINN configuration from JSON file."""
        if not os.path.exists(config_file):
            logger.warning(f"FINN config file not found: {config_file}")
            self._load_finn_defaults()
            return
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            self._apply_finn_config_data(config_data)
            self.config_source = config_file
            
            logger.info(f"Loaded FINN configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load FINN config from {config_file}: {e}")
            self._load_finn_defaults()
    
    def _apply_finn_config_data(self, config_data: Dict[str, Any]):
        """Apply FINN configuration data from loaded file."""
        # Start with defaults
        self._load_finn_defaults()
        
        # Load compiler settings
        if 'compiler' in config_data:
            compiler_data = config_data['compiler']
            self.compiler = FINNCompilerSettings(
                include_paths=compiler_data.get('include_paths', self.compiler.include_paths),
                cxx_standard=compiler_data.get('cxx_standard', self.compiler.cxx_standard),
                optimization_level=compiler_data.get('optimization_level', self.compiler.optimization_level),
                custom_flags=compiler_data.get('custom_flags', self.compiler.custom_flags),
                link_libraries=compiler_data.get('link_libraries', self.compiler.link_libraries),
                clock_period=compiler_data.get('clock_period', self.compiler.clock_period),
                target_fpga=compiler_data.get('target_fpga', self.compiler.target_fpga),
                enable_debug_defines=compiler_data.get('enable_debug_defines', self.compiler.enable_debug_defines),
                ap_int_max_w=compiler_data.get('ap_int_max_w', self.compiler.ap_int_max_w)
            )
        
        # Load template settings
        if 'templates' in config_data:
            template_data = config_data['templates']
            self.templates = FINNTemplateSettings(
                template_paths=template_data.get('template_paths', self.templates.template_paths),
                enable_caching=template_data.get('enable_caching', self.templates.enable_caching),
                variable_prefix=template_data.get('variable_prefix', self.templates.variable_prefix),
                variable_suffix=template_data.get('variable_suffix', self.templates.variable_suffix),
                ipgen_template_name=template_data.get('ipgen_template_name', self.templates.ipgen_template_name),
                ipgen_tcl_template_name=template_data.get('ipgen_tcl_template_name', self.templates.ipgen_tcl_template_name),
                docompute_template_name=template_data.get('docompute_template_name', self.templates.docompute_template_name)
            )
        
        # Load path settings
        if 'paths' in config_data:
            path_data = config_data['paths']
            self.paths = FINNPathSettings(
                finn_root=path_data.get('finn_root', self.paths.finn_root),
                finn_deps_dir=path_data.get('finn_deps_dir', self.paths.finn_deps_dir),
                finn_hlslib_dir=path_data.get('finn_hlslib_dir', self.paths.finn_hlslib_dir),
                vitis_path=path_data.get('vitis_path', self.paths.vitis_path),
                hls_path=path_data.get('hls_path', self.paths.hls_path),
                enable_path_caching=path_data.get('enable_path_caching', self.paths.enable_path_caching),
                auto_discover_libraries=path_data.get('auto_discover_libraries', self.paths.auto_discover_libraries)
            )
        
        # Store custom settings
        self.custom_settings = {k: v for k, v in config_data.items() 
                               if k not in ['compiler', 'templates', 'paths']}
        
        # Determine config level
        if 'user' in (self.config_source or ''):
            self.config_level = FINNConfigLevel.USER
        else:
            self.config_level = FINNConfigLevel.SYSTEM
    
    def save_to_file(self, config_file: str):
        """Save current FINN configuration to JSON file."""
        config_data = {
            'compiler': self.compiler.to_dict(),
            'templates': self.templates.to_dict(),
            'paths': self.paths.to_dict(),
            **self.custom_settings
        }
        
        # Remove None values to keep file clean
        config_data = self._clean_config_data(config_data)
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved FINN configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save FINN config to {config_file}: {e}")
    
    def _clean_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and empty lists to keep config clean."""
        cleaned = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                cleaned_dict = self._clean_config_data(value)
                if cleaned_dict:
                    cleaned[key] = cleaned_dict
            elif isinstance(value, list):
                if value:
                    cleaned[key] = value
            elif value is not None and value != "":
                cleaned[key] = value
        
        return cleaned
    
    def get_expanded_paths(self, env_vars: Optional[Dict[str, str]] = None) -> 'FINNHLSConfig':
        """Return configuration with environment variables expanded in paths."""
        expanded = deepcopy(self)
        
        # Use provided env_vars or fall back to os.environ
        env_dict = env_vars or dict(os.environ)
        
        # Expand paths in compiler settings
        expanded.compiler.include_paths = [self._expand_path(p, env_dict) for p in self.compiler.include_paths]
        
        # Expand paths in template settings
        expanded.templates.template_paths = [self._expand_path(p, env_dict) for p in self.templates.template_paths]
        
        return expanded
    
    def _expand_path(self, path: str, env_dict: Dict[str, str]) -> str:
        """Expand environment variables in a path string using FINN patterns."""
        expanded = path
        for env_var, env_value in env_dict.items():
            expanded = expanded.replace(f"${env_var}", env_value)
            expanded = expanded.replace(f"${{{env_var}}}", env_value)
        return expanded
    
    def validate_finn_environment(self) -> List[str]:
        """Validate FINN environment configuration and return list of issues."""
        issues = []
        
        # Check critical FINN environment variables
        if not self.paths.finn_root:
            issues.append("FINN_ROOT not configured")
        
        if not self.paths.finn_deps_dir:
            issues.append("FINN_DEPS_DIR not configured")
        
        # Check tool paths
        if not self.paths.hls_path:
            issues.append("HLS_PATH not configured")
        
        if not self.paths.vitis_path:
            issues.append("VITIS_PATH not configured")
        
        # Validate compiler settings
        if self.compiler.clock_period <= 0:
            issues.append("Clock period must be positive")
        
        if self.compiler.ap_int_max_w <= 0 or self.compiler.ap_int_max_w > 8191:
            issues.append("AP_INT_MAX_W must be between 1 and 8191")
        
        return issues
    
    def get_finn_summary(self) -> Dict[str, Any]:
        """Get FINN configuration summary for debugging/logging."""
        return {
            "config_level": self.config_level.value,
            "config_source": self.config_source,
            "finn_root": self.paths.finn_root,
            "finn_deps_dir": self.paths.finn_deps_dir,
            "include_paths_count": len(self.compiler.include_paths),
            "template_paths_count": len(self.templates.template_paths),
            "cxx_standard": self.compiler.cxx_standard,
            "clock_period": self.compiler.clock_period,
            "ap_int_max_w": self.compiler.ap_int_max_w,
            "custom_settings_count": len(self.custom_settings)
        }
    
    def create_code_gen_dict(self, additional_vars: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Create FINN-compatible code generation dictionary."""
        code_gen_dict = {
            "$AP_INT_MAX_W$": [str(self.compiler.ap_int_max_w)]
        }
        
        # Add additional variables if provided
        if additional_vars:
            for key, value in additional_vars.items():
                if isinstance(value, list):
                    code_gen_dict[key] = value
                else:
                    code_gen_dict[key] = [str(value)]
        
        return code_gen_dict


# Convenience functions
def create_finn_config(config_file: Optional[str] = None) -> FINNHLSConfig:
    """Create FINN HLS configuration with optional config file."""
    return FINNHLSConfig(config_file)


def get_default_finn_config() -> FINNHLSConfig:
    """Get default FINN HLS configuration."""
    return FINNHLSConfig()


def validate_finn_environment() -> Dict[str, Any]:
    """Validate current FINN environment and return report."""
    config = get_default_finn_config()
    issues = config.validate_finn_environment()
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "summary": config.get_finn_summary()
    }