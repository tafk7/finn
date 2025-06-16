############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN PathResolver System - Environment-agnostic path management for FINN
############################################################################

import os
import logging
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class FINNEnvironmentConfig:
    """FINN-specific environment variable abstraction with fallback support."""
    
    # Core FINN paths
    finn_root: Optional[str] = None
    finn_deps_dir: Optional[str] = None
    finn_hlslib_dir: Optional[str] = None
    finn_qnn_data_dir: Optional[str] = None
    finn_custom_hls_dir: Optional[str] = None
    
    # Tool paths
    vitis_path: Optional[str] = None
    hls_path: Optional[str] = None
    
    # Additional dependency paths
    cnpy_dir: Optional[str] = None
    
    # Custom library paths
    custom_paths: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-populate from environment variables with intelligent fallbacks."""
        self._populate_from_env()
        self._apply_fallbacks()
        self._validate_paths()
    
    def _populate_from_env(self):
        """Load values from environment variables."""
        env_mapping = {
            'finn_root': 'FINN_ROOT',
            'finn_deps_dir': 'FINN_DEPS_DIR',
            'finn_hlslib_dir': 'FINN_HLSLIB_DIR',
            'vitis_path': 'VITIS_PATH',
            'hls_path': 'HLS_PATH'
        }
        
        for attr, env_var in env_mapping.items():
            if getattr(self, attr) is None:
                setattr(self, attr, os.environ.get(env_var))
    
    def _apply_fallbacks(self):
        """Apply intelligent fallbacks for missing paths."""
        # Derive FINN subdirectories from FINN_ROOT
        if self.finn_root:
            if not self.finn_qnn_data_dir:
                self.finn_qnn_data_dir = os.path.join(self.finn_root, "src", "finn", "qnn-data", "cpp")
            
            if not self.finn_custom_hls_dir:
                self.finn_custom_hls_dir = os.path.join(self.finn_root, "custom_hls")
        
        # Derive dependency paths from FINN_DEPS_DIR
        if self.finn_deps_dir:
            if not self.finn_hlslib_dir:
                hlslib_candidate = os.path.join(self.finn_deps_dir, "finn-hlslib")
                if os.path.exists(hlslib_candidate):
                    self.finn_hlslib_dir = hlslib_candidate
            
            if not self.cnpy_dir:
                cnpy_candidate = os.path.join(self.finn_deps_dir, "cnpy")
                if os.path.exists(cnpy_candidate):
                    self.cnpy_dir = cnpy_candidate
    
    def _validate_paths(self):
        """Validate that critical paths exist."""
        critical_paths = ['finn_root', 'finn_deps_dir']
        for path_attr in critical_paths:
            path_value = getattr(self, path_attr)
            if path_value and not os.path.exists(path_value):
                logger.warning(f"Critical FINN path {path_attr}={path_value} does not exist")
    
    def get_path(self, key: str) -> Optional[str]:
        """Get path value with custom path support."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_paths.get(key)
    
    def set_custom_path(self, key: str, path: str):
        """Set custom path for library-specific usage."""
        self.custom_paths[key] = path
        logger.debug(f"Set custom FINN path {key}={path}")


class FINNLibraryRegistry:
    """Registry for managing FINN-specific library paths and metadata."""
    
    def __init__(self):
        self._libraries: Dict[str, Dict] = {}
        self._library_paths: Dict[str, str] = {}
        self._header_cache: Dict[str, Set[str]] = {}
    
    def register_library(self, name: str, base_path: str, metadata: Optional[Dict] = None):
        """Register a FINN library with its base path and optional metadata."""
        if not os.path.exists(base_path):
            logger.warning(f"FINN library path does not exist: {base_path}")
            # Don't raise exception for FINN libraries as they may be optional
            
        self._libraries[name] = metadata or {}
        self._library_paths[name] = base_path
        
        # Clear header cache for this library
        if name in self._header_cache:
            del self._header_cache[name]
            
        logger.info(f"Registered FINN library '{name}' at {base_path}")
    
    def get_library_path(self, name: str) -> Optional[str]:
        """Get the base path for a registered library."""
        return self._library_paths.get(name)
    
    def get_library_metadata(self, name: str) -> Dict:
        """Get metadata for a registered library."""
        return self._libraries.get(name, {})
    
    def list_libraries(self) -> List[str]:
        """List all registered library names."""
        return list(self._libraries.keys())
    
    def discover_headers(self, library_name: str, refresh_cache: bool = False) -> Set[str]:
        """Discover header files for a FINN library."""
        if library_name not in self._library_paths:
            return set()
            
        if library_name in self._header_cache and not refresh_cache:
            return self._header_cache[library_name]
        
        library_path = self._library_paths[library_name]
        if not os.path.exists(library_path):
            return set()
            
        headers = set()
        
        # Search for .hpp, .h files recursively
        for root, _, files in os.walk(library_path):
            for file in files:
                if file.endswith(('.hpp', '.h')):
                    rel_path = os.path.relpath(os.path.join(root, file), library_path)
                    headers.add(rel_path)
        
        self._header_cache[library_name] = headers
        logger.debug(f"Discovered {len(headers)} headers in FINN library '{library_name}'")
        return headers


class FINNPathDiscovery:
    """Automatic path discovery specifically for FINN libraries and headers."""
    
    def __init__(self, env_config: FINNEnvironmentConfig, registry: FINNLibraryRegistry):
        self.env_config = env_config
        self.registry = registry
        self._search_paths: List[str] = []
        self._init_search_paths()
    
    def _init_search_paths(self):
        """Initialize search paths from FINN environment configuration."""
        potential_paths = [
            self.env_config.finn_hlslib_dir,
            self.env_config.finn_qnn_data_dir,
            self.env_config.finn_custom_hls_dir,
            self.env_config.cnpy_dir,
            self.env_config.finn_deps_dir,
        ]
        
        self._search_paths = [p for p in potential_paths if p and os.path.exists(p)]
        logger.debug(f"Initialized {len(self._search_paths)} FINN search paths")
    
    def find_header(self, header_name: str, library_hint: Optional[str] = None) -> Optional[str]:
        """Find a header file across FINN search paths and registered libraries."""
        # Try library-specific search first if hint provided
        if library_hint:
            library_path = self.registry.get_library_path(library_hint)
            if library_path and os.path.exists(library_path):
                candidate = self._search_in_directory(header_name, library_path)
                if candidate:
                    return candidate
        
        # Search in all registered libraries
        for lib_name in self.registry.list_libraries():
            lib_path = self.registry.get_library_path(lib_name)
            if lib_path and os.path.exists(lib_path):
                candidate = self._search_in_directory(header_name, lib_path)
                if candidate:
                    return candidate
        
        # Search in standard FINN search paths
        for search_path in self._search_paths:
            candidate = self._search_in_directory(header_name, search_path)
            if candidate:
                return candidate
        
        logger.warning(f"FINN header '{header_name}' not found in any search path")
        return None
    
    def _search_in_directory(self, header_name: str, directory: str) -> Optional[str]:
        """Search for header in a specific directory recursively."""
        if not os.path.exists(directory):
            return None
            
        for root, _, files in os.walk(directory):
            if header_name in files:
                return os.path.join(root, header_name)
        return None
    
    def get_include_paths(self, required_headers: List[str]) -> List[str]:
        """Get list of include paths needed for the specified headers."""
        include_paths = set()
        
        for header in required_headers:
            header_path = self.find_header(header)
            if header_path:
                include_dir = os.path.dirname(header_path)
                include_paths.add(include_dir)
            else:
                logger.warning(f"Could not locate FINN header: {header}")
        
        return list(include_paths)


class FINNPathResolver:
    """Main FINN path resolver orchestrating environment config, registry, and discovery."""
    
    def __init__(self):
        self.env_config = FINNEnvironmentConfig()
        self.registry = FINNLibraryRegistry()
        self.discovery = FINNPathDiscovery(self.env_config, self.registry)
        
        # Auto-register FINN libraries
        self._auto_register_finn_libraries()
    
    def _auto_register_finn_libraries(self):
        """Automatically register FINN libraries from standard locations."""
        # Register core FINN libraries
        libraries_to_register = [
            ("finn-hlslib", self.env_config.finn_hlslib_dir, {"type": "hls_library"}),
            ("finn-qnn-data", self.env_config.finn_qnn_data_dir, {"type": "cpp_utilities"}),
            ("finn-custom-hls", self.env_config.finn_custom_hls_dir, {"type": "custom_hls"}),
            ("cnpy", self.env_config.cnpy_dir, {"type": "numpy_cpp"}),
        ]
        
        for name, path, metadata in libraries_to_register:
            if path:
                self.registry.register_library(name, path, metadata)
        
        logger.info(f"Auto-registered {len(self.registry.list_libraries())} FINN libraries")
    
    def get_finn_include_paths(self) -> List[str]:
        """Get all FINN include paths needed for compilation."""
        include_paths = []
        
        # Add core FINN include paths
        core_paths = [
            self.env_config.finn_qnn_data_dir,
            self.env_config.cnpy_dir,
            self.env_config.finn_hlslib_dir,
            self.env_config.finn_custom_hls_dir,
        ]
        
        for path in core_paths:
            if path and os.path.exists(path):
                include_paths.append(path)
        
        # Add tool include paths
        if self.env_config.hls_path:
            hls_include = os.path.join(self.env_config.hls_path, "include")
            if os.path.exists(hls_include):
                include_paths.append(hls_include)
        
        if self.env_config.vitis_path:
            vitis_include = os.path.join(self.env_config.vitis_path, "include")
            if os.path.exists(vitis_include):
                include_paths.append(vitis_include)
        
        return include_paths
    
    def get_finn_source_paths(self) -> List[str]:
        """Get FINN source file paths for linking."""
        source_paths = []
        
        # Add cnpy source if available
        if self.env_config.cnpy_dir:
            cnpy_source = os.path.join(self.env_config.cnpy_dir, "cnpy.cpp")
            if os.path.exists(cnpy_source):
                source_paths.append(cnpy_source)
        
        return source_paths
    
    def resolve_header_paths(self, headers: List[str], library_context: Optional[str] = None) -> Dict[str, str]:
        """Resolve header file paths, returning mapping of header -> full_path."""
        resolved = {}
        
        for header in headers:
            path = self.discovery.find_header(header, library_context)
            if path:
                resolved[header] = path
            else:
                logger.error(f"Failed to resolve FINN header: {header}")
        
        return resolved
    
    def get_include_directories(self, headers: List[str]) -> List[str]:
        """Get list of include directories needed for the specified headers."""
        return self.discovery.get_include_paths(headers)
    
    def get_library_info(self, library_name: str) -> Dict:
        """Get comprehensive information about a FINN library."""
        return {
            "name": library_name,
            "path": self.registry.get_library_path(library_name),
            "metadata": self.registry.get_library_metadata(library_name),
            "headers": list(self.registry.discover_headers(library_name))
        }
    
    def validate_finn_environment(self) -> List[str]:
        """Validate FINN environment configuration and return list of issues."""
        issues = []
        
        # Check critical FINN environment paths
        if not self.env_config.finn_root:
            issues.append("FINN_ROOT not configured")
        
        if not self.env_config.finn_deps_dir:
            issues.append("FINN_DEPS_DIR not configured")
        
        # Check tool paths
        if not self.env_config.hls_path:
            issues.append("HLS_PATH not configured")
        
        if not self.env_config.vitis_path:
            issues.append("VITIS_PATH not configured")
        
        # Check registered libraries
        for lib_name in self.registry.list_libraries():
            lib_path = self.registry.get_library_path(lib_name)
            if not os.path.exists(lib_path):
                issues.append(f"FINN library '{lib_name}' path does not exist: {lib_path}")
        
        return issues
    
    def get_environment_summary(self) -> Dict[str, str]:
        """Get summary of current FINN environment configuration."""
        return {
            "finn_root": self.env_config.finn_root,
            "finn_deps_dir": self.env_config.finn_deps_dir,
            "finn_hlslib_dir": self.env_config.finn_hlslib_dir,
            "finn_qnn_data_dir": self.env_config.finn_qnn_data_dir,
            "finn_custom_hls_dir": self.env_config.finn_custom_hls_dir,
            "cnpy_dir": self.env_config.cnpy_dir,
            "hls_path": self.env_config.hls_path,
            "vitis_path": self.env_config.vitis_path,
            "libraries_registered": len(self.registry.list_libraries())
        }