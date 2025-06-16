############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Flexible HLS Backend System - Main module exports
############################################################################

"""
FINN Flexible HLS Backend System

This module provides a flexible, environment-agnostic HLS backend system for FINN
that replaces the existing rigid HLSBackend implementation while maintaining 
100% backward compatibility.

Key Features:
- Flexible path resolution for FINN environments
- Template inheritance and customization
- Enhanced compilation strategies
- 100% API compatibility with existing FINN kernels

Usage:
    from finn.util.flexible_hls import FlexibleHLSBackend, FINNHLSConfig
    
    # Create flexible backend with automatic environment detection
    backend = FlexibleHLSBackend()
    
    # Or use with custom configuration
    config = FINNHLSConfig("custom_config.json")
    backend = FlexibleHLSBackend(config=config)
"""

__version__ = "1.0.0"
__author__ = "Advanced Micro Devices, Inc."

# Core backend implementation
from .flexible_backend import FlexibleHLSBackend

# Configuration management
from .config import (
    FINNHLSConfig,
    FINNCompilerSettings,
    FINNTemplateSettings,
    FINNPathSettings,
    FINNConfigLevel,
    create_finn_config,
    get_default_finn_config,
    validate_finn_environment
)

# Path resolution system
from .path_resolver import (
    FINNPathResolver,
    FINNEnvironmentConfig,
    FINNLibraryRegistry,
    FINNPathDiscovery
)

# Template engine
from .template_engine import (
    FINNTemplateEngine,
    FINNTemplateLoader,
    FINNVariableResolver,
    FINNTemplateMigrator,
    FINNTemplateType,
    FINNTemplateMetadata
)

# Compiler builder
from .compiler_builder import (
    FINNCompilerBuilder,
    FINNBuildConfiguration,
    FINNBuildStrategy,
    FINNIncludeManager,
    create_finn_build_config,
    create_finn_compiler_builder,
    build_finn_executable
)

# Main exports for easy import
__all__ = [
    # Core backend
    "FlexibleHLSBackend",
    
    # Configuration
    "FINNHLSConfig",
    "FINNCompilerSettings", 
    "FINNTemplateSettings",
    "FINNPathSettings",
    "FINNConfigLevel",
    "create_finn_config",
    "get_default_finn_config",
    "validate_finn_environment",
    
    # Path resolution
    "FINNPathResolver",
    "FINNEnvironmentConfig",
    "FINNLibraryRegistry",
    "FINNPathDiscovery",
    
    # Template engine
    "FINNTemplateEngine",
    "FINNTemplateLoader",
    "FINNVariableResolver", 
    "FINNTemplateMigrator",
    "FINNTemplateType",
    "FINNTemplateMetadata",
    
    # Compiler
    "FINNCompilerBuilder",
    "FINNBuildConfiguration",
    "FINNBuildStrategy",
    "FINNIncludeManager",
    "create_finn_build_config",
    "create_finn_compiler_builder",
    "build_finn_executable",
]

# Convenience functions for quick setup
def create_flexible_backend(config_file=None, **kwargs):
    """
    Create a FlexibleHLSBackend with optional configuration.
    
    Args:
        config_file: Optional path to configuration file
        **kwargs: Additional configuration parameters
    
    Returns:
        FlexibleHLSBackend: Configured backend instance
    """
    config = create_finn_config(config_file) if config_file else get_default_finn_config()
    return FlexibleHLSBackend(config=config, **kwargs)


def check_finn_environment():
    """
    Check FINN environment and return validation report.
    
    Returns:
        dict: Environment validation report with issues and summary
    """
    return validate_finn_environment()


# Module initialization message
import logging
logger = logging.getLogger(__name__)
logger.debug("FINN Flexible HLS Backend System initialized")