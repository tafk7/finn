############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN CompilerBuilder System - Flexible compilation for FINN HLS kernels
############################################################################

import os
import logging
import hashlib
import time
import subprocess
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from finn.util.basic import CppBuilder

from .path_resolver import FINNPathResolver


logger = logging.getLogger(__name__)


class FINNBuildStrategy(Enum):
    """Available FINN build strategies."""
    FINN_DEFAULT = "finn_default"
    FINN_DEBUG = "finn_debug"
    CUSTOM = "custom"


@dataclass
class FINNBuildConfiguration:
    """Configuration for FINN-specific builds."""
    strategy: FINNBuildStrategy = FINNBuildStrategy.FINN_DEFAULT
    optimization_level: str = "-O3"
    debug_mode: bool = False
    enable_profiling: bool = False
    
    # Compilation settings
    cxx_standard: str = "c++14"
    custom_flags: List[str] = field(default_factory=list)
    custom_includes: List[str] = field(default_factory=list)
    custom_sources: List[str] = field(default_factory=list)
    
    # FINN-specific settings
    enable_debug_defines: bool = False
    executable_name: str = "node_model"
    
    def get_cache_key(self) -> str:
        """Generate cache key for this build configuration."""
        config_str = f"{self.strategy.value}_{self.optimization_level}_{self.cxx_standard}"
        config_str += "_".join(sorted(self.custom_flags))
        config_str += "_".join(sorted(self.custom_includes))
        return hashlib.md5(config_str.encode()).hexdigest()


class FINNIncludeManager:
    """FINN-specific include path management."""
    
    def __init__(self, path_resolver: FINNPathResolver):
        self.path_resolver = path_resolver
        self._resolved_includes: Optional[List[str]] = None
    
    def get_finn_include_paths(self, additional_includes: Optional[List[str]] = None) -> List[str]:
        """Get all FINN include paths needed for compilation."""
        if self._resolved_includes is None:
            self._resolved_includes = self.path_resolver.get_finn_include_paths()
        
        include_paths = self._resolved_includes.copy()
        
        # Add additional custom includes
        if additional_includes:
            for include_path in additional_includes:
                if os.path.exists(include_path):
                    include_paths.append(include_path)
                else:
                    logger.warning(f"Custom include path does not exist: {include_path}")
        
        return include_paths
    
    def get_finn_source_paths(self) -> List[str]:
        """Get FINN source file paths for linking."""
        return self.path_resolver.get_finn_source_paths()
    
    def validate_environment(self) -> List[str]:
        """Validate FINN compilation environment."""
        return self.path_resolver.validate_finn_environment()


class FINNCompilerBuilder:
    """Main FINN compiler builder with flexible path resolution."""
    
    def __init__(self, path_resolver: Optional[FINNPathResolver] = None):
        self.path_resolver = path_resolver or FINNPathResolver()
        self.include_manager = FINNIncludeManager(self.path_resolver)
        
        # Build cache
        self._build_cache: Dict[str, Tuple[str, float]] = {}  # cache_key -> (executable_path, timestamp)
        
        # Validate environment on initialization
        env_issues = self.include_manager.validate_environment()
        if env_issues:
            logger.warning(f"FINN environment issues detected: {env_issues}")
    
    def build(self, source_directory: str, output_directory: str, 
              build_config: Optional[FINNBuildConfiguration] = None) -> str:
        """Build FINN project using flexible path resolution."""
        
        if build_config is None:
            build_config = FINNBuildConfiguration()
        
        # Check cache first
        cached_executable = self._check_build_cache(build_config, source_directory)
        if cached_executable:
            logger.info(f"Using cached FINN build: {cached_executable}")
            return cached_executable
        
        # Create FINN CppBuilder
        builder = CppBuilder()
        
        # Configure builder with FINN-specific settings
        builder = self._configure_finn_builder(builder, build_config)
        
        # Add source files
        builder.append_sources(f"{source_directory}/*.cpp")
        
        # Add FINN source dependencies
        finn_sources = self.include_manager.get_finn_source_paths()
        for source in finn_sources:
            builder.append_sources(source)
        
        # Add custom sources
        for source in build_config.custom_sources:
            if os.path.exists(source):
                builder.append_sources(source)
            else:
                logger.warning(f"Custom source does not exist: {source}")
        
        # Set output executable
        executable_path = os.path.join(output_directory, build_config.executable_name)
        builder.set_executable_path(executable_path)
        
        # Build
        try:
            start_time = time.time()
            builder.build(output_directory)
            build_time = time.time() - start_time
            
            logger.info(f"FINN build completed in {build_time:.2f}s: {executable_path}")
            
            # Cache the result
            cache_key = self._generate_cache_key(build_config, source_directory)
            self._build_cache[cache_key] = (executable_path, time.time())
            
            return executable_path
            
        except Exception as e:
            logger.error(f"FINN build failed: {e}")
            raise
    
    def _configure_finn_builder(self, builder: CppBuilder, build_config: FINNBuildConfiguration) -> CppBuilder:
        """Configure CppBuilder with FINN-specific settings."""
        
        # Add FINN include paths using flexible path resolution
        include_paths = self.include_manager.get_finn_include_paths(build_config.custom_includes)
        
        for include_path in include_paths:
            builder.append_includes(f"-I{include_path}")
        
        # Add FINN-specific compiler flags
        builder.append_includes(f"--std={build_config.cxx_standard}")
        builder.append_includes(build_config.optimization_level)
        
        # Add debug flags if enabled
        if build_config.debug_mode or build_config.enable_debug_defines:
            builder.append_includes("-DDEBUG")
        
        if build_config.debug_mode:
            builder.append_includes("-g")
        
        # Add FINN-specific libraries
        builder.append_includes("-lz")
        
        # Add custom flags
        for flag in build_config.custom_flags:
            builder.append_includes(flag)
        
        return builder
    
    def _check_build_cache(self, build_config: FINNBuildConfiguration, source_directory: str) -> Optional[str]:
        """Check if cached build exists and is valid."""
        cache_key = self._generate_cache_key(build_config, source_directory)
        
        if cache_key not in self._build_cache:
            return None
        
        executable_path, build_timestamp = self._build_cache[cache_key]
        
        # Check if executable still exists
        if not os.path.exists(executable_path):
            del self._build_cache[cache_key]
            return None
        
        # Check if source files are newer than cached build
        latest_source_time = self._get_latest_modification_time(source_directory)
        if latest_source_time > build_timestamp:
            del self._build_cache[cache_key]
            return None
        
        return executable_path
    
    def _generate_cache_key(self, build_config: FINNBuildConfiguration, source_directory: str) -> str:
        """Generate cache key for build configuration and source directory."""
        config_key = build_config.get_cache_key()
        source_key = hashlib.md5(source_directory.encode()).hexdigest()
        return f"finn_{config_key}_{source_key}"
    
    def _get_latest_modification_time(self, directory: str) -> float:
        """Get latest modification time of files in directory."""
        latest_time = 0.0
        
        if not os.path.exists(directory):
            return latest_time
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    mtime = os.path.getmtime(file_path)
                    latest_time = max(latest_time, mtime)
                except OSError:
                    continue
        
        return latest_time
    
    def clear_cache(self):
        """Clear build cache."""
        self._build_cache.clear()
        logger.debug("FINN build cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get build cache statistics."""
        total_size = 0
        for executable_path, _ in self._build_cache.values():
            if os.path.exists(executable_path):
                total_size += os.path.getsize(executable_path)
        
        return {
            "cached_builds": len(self._build_cache),
            "cache_size_mb": total_size / (1024 * 1024),
            "environment_summary": self.path_resolver.get_environment_summary()
        }
    
    def get_compilation_command_preview(self, source_directory: str, output_directory: str,
                                       build_config: Optional[FINNBuildConfiguration] = None) -> List[str]:
        """Get preview of compilation command that would be executed."""
        if build_config is None:
            build_config = FINNBuildConfiguration()
        
        # Create a temporary builder to get command preview
        builder = CppBuilder()
        builder = self._configure_finn_builder(builder, build_config)
        
        # Add sources
        builder.append_sources(f"{source_directory}/*.cpp")
        
        finn_sources = self.include_manager.get_finn_source_paths()
        for source in finn_sources:
            builder.append_sources(source)
        
        executable_path = os.path.join(output_directory, build_config.executable_name)
        builder.set_executable_path(executable_path)
        
        # Extract command from builder (this is a simplified preview)
        command_parts = []
        command_parts.append("g++")  # Default compiler
        
        # Add includes and flags (simplified extraction)
        if hasattr(builder, 'includes'):
            command_parts.extend(builder.includes)
        
        if hasattr(builder, 'sources'):
            command_parts.extend(builder.sources)
        
        command_parts.extend(["-o", executable_path])
        
        return command_parts
    
    def validate_build_environment(self) -> Dict[str, Any]:
        """Validate FINN build environment and return detailed report."""
        validation_report = {
            "environment_valid": True,
            "issues": [],
            "warnings": [],
            "environment_summary": self.path_resolver.get_environment_summary()
        }
        
        # Check FINN environment
        env_issues = self.include_manager.validate_environment()
        if env_issues:
            validation_report["issues"].extend(env_issues)
            validation_report["environment_valid"] = False
        
        # Check include paths
        include_paths = self.include_manager.get_finn_include_paths()
        missing_includes = []
        for include_path in include_paths:
            if not os.path.exists(include_path):
                missing_includes.append(include_path)
        
        if missing_includes:
            validation_report["warnings"].append(f"Missing include paths: {missing_includes}")
        
        # Check source paths
        source_paths = self.include_manager.get_finn_source_paths()
        missing_sources = []
        for source_path in source_paths:
            if not os.path.exists(source_path):
                missing_sources.append(source_path)
        
        if missing_sources:
            validation_report["warnings"].append(f"Missing source files: {missing_sources}")
        
        return validation_report


# Convenience functions
def create_finn_build_config(debug: bool = False, **kwargs) -> FINNBuildConfiguration:
    """Create FINNBuildConfiguration with common defaults."""
    strategy = FINNBuildStrategy.FINN_DEBUG if debug else FINNBuildStrategy.FINN_DEFAULT
    return FINNBuildConfiguration(
        strategy=strategy,
        debug_mode=debug,
        **kwargs
    )


def create_finn_compiler_builder() -> FINNCompilerBuilder:
    """Create FINNCompilerBuilder with default configuration."""
    return FINNCompilerBuilder()


def build_finn_executable(source_directory: str, output_directory: str, 
                         debug: bool = False, **kwargs) -> str:
    """Convenience function to build FINN executable with minimal configuration."""
    builder = create_finn_compiler_builder()
    build_config = create_finn_build_config(debug=debug, **kwargs)
    return builder.build(source_directory, output_directory, build_config)