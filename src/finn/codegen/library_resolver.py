"""
Library Resolver for Unified FINN Code Generation

This module provides dynamic library dependency resolution for FINN code generation,
automatically determining and managing include files and library dependencies
based on operation requirements.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import logging


class LibraryType(Enum):
    """Types of libraries that can be resolved."""
    HLS_LIBRARY = "hls_library"
    RTL_LIBRARY = "rtl_library"
    SYSTEM_LIBRARY = "system_library"
    CUSTOM_LIBRARY = "custom_library"


@dataclass
class LibrarySpec:
    """Specification for a library dependency."""
    name: str
    path: str  # Can contain environment variables like $FINN_ROOT
    include_files: List[str]
    library_type: LibraryType
    required_for: Optional[List[str]] = None  # Operation types that require this library
    dependencies: Optional[List[str]] = None  # Other libraries this depends on
    
    def __post_init__(self):
        if self.required_for is None:
            self.required_for = []
        if self.dependencies is None:
            self.dependencies = []


class LibraryResolver:
    """
    Dynamic library dependency resolver for FINN operations.
    
    Automatically determines which libraries and include files are needed
    for a given operation, resolving paths and handling dependencies.
    """
    
    def __init__(self):
        """Initialize the library resolver with default FINN libraries."""
        self.logger = logging.getLogger(__name__)
        self.libraries: Dict[str, LibrarySpec] = {}
        self._auto_detected_paths = self._auto_detect_finn_paths()
        self._register_default_libraries()
    
    def _get_auto_detected_path(self, path_key: str) -> Optional[str]:
        """Get an auto-detected path by key."""
        return self._auto_detected_paths.get(path_key)
        
    def _register_default_libraries(self):
        """Register default FINN libraries and their specifications."""
        
        # FINN HLS Library - Updated with actual available files and fallback paths
        auto_deps_dir = self._get_auto_detected_path('AUTO_FINN_DEPS_DIR')
        finn_hlslib_path = f'$FINN_DEPS_DIR/finn-hlslib' if not auto_deps_dir else f'{auto_deps_dir}/finn-hlslib'
        
        self.register_library(LibrarySpec(
            name='finn-hlslib',
            path=finn_hlslib_path,
            include_files=[
                'mvau.hpp',           # Matrix-Vector-Activation Unit
                'utils.hpp',          # Utility functions
                'pool.hpp',           # Pooling operations
                'activations.hpp',    # Activation functions
                'weights.hpp',        # Weight handling
                'mac.hpp',            # Multiply-accumulate operations
                'concat.hpp',         # Concatenation operations
                'split.hpp',          # Split operations
                'interpret.hpp',      # Interpretation utilities
                'vvau.hpp',           # Vector-Vector-Activation Unit
                'tmrcheck.hpp'        # Triple Modular Redundancy check
            ],
            library_type=LibraryType.HLS_LIBRARY,
            required_for=['MatrixVectorActivation', 'Pool_Batch', 'StreamingDataWidthConverter', 'VectorVectorActivation']
        ))
        
        # BNN Library (Legacy) - with auto-detected paths
        auto_root_dir = self._get_auto_detected_path('AUTO_FINN_ROOT')
        bnn_library_path = f'$FINN_ROOT/custom_hls' if not auto_root_dir else f'{auto_root_dir}/custom_hls'
        
        self.register_library(LibrarySpec(
            name='bnn-library',
            path=bnn_library_path,
            include_files=[
                'bnn-library.h',
                'activations.hpp',
                'weights.hpp',
                'fclayer.h',
                'convlayer.h'
            ],
            library_type=LibraryType.HLS_LIBRARY,
            required_for=['MVAU_hls', 'ConvolutionInputGenerator', 'MatrixVectorActivation']
        ))
        
        # FINN RTL Library - with auto-detected paths
        finn_rtllib_path = f'$FINN_ROOT/finn-rtllib' if not auto_root_dir else f'{auto_root_dir}/finn-rtllib'
        
        self.register_library(LibrarySpec(
            name='finn-rtllib',
            path=finn_rtllib_path,
            include_files=[
                'swg/swg_pkg.sv',
                'swg/swg_common.sv',
                'swg/swg_template_default.sv',
                'swg/swg_template_parallel.sv',
                'swg/swg_template_axilite.v',
                'swg/swg_template_wrapper.v'
            ],
            library_type=LibraryType.RTL_LIBRARY,
            required_for=['StreamingFIFO_rtl', 'Thresholding_rtl', 'MatrixVectorActivation_rtl']
        ))
        
        # Xilinx HLS Libraries
        self.register_library(LibrarySpec(
            name='xilinx-hls',
            path='',  # System paths
            include_files=[
                'ap_int.h',
                'ap_fixed.h', 
                'hls_stream.h',
                'hls_math.h'
            ],
            library_type=LibraryType.SYSTEM_LIBRARY,
            required_for=['*']  # Required for all HLS operations
        ))
        
        # Standard C++ Libraries  
        self.register_library(LibrarySpec(
            name='std-cpp',
            path='',  # System paths
            include_files=[
                'iostream',
                'vector',
                'string',
                'memory',
                'algorithm',
                'cassert'
            ],
            library_type=LibraryType.SYSTEM_LIBRARY,
            required_for=['*']  # Potentially required for any operation
        ))
        
        # SystemVerilog Standard Libraries
        self.register_library(LibrarySpec(
            name='systemverilog-std',
            path='',  # System paths  
            include_files=[],  # Built-in to SystemVerilog
            library_type=LibraryType.SYSTEM_LIBRARY,
            required_for=['*_rtl']  # Required for all RTL operations
        ))
    
    def register_library(self, library_spec: LibrarySpec):
        """
        Register a new library specification.
        
        Args:
            library_spec: Library specification to register
        """
        self.libraries[library_spec.name] = library_spec
        self.logger.debug(f"Registered library: {library_spec.name}")
    
    def resolve_includes(self, operation) -> List[str]:
        """
        Resolve include files needed for an operation.
        
        Args:
            operation: HWCustomOp instance
            
        Returns:
            List of resolved include file paths
        """
        operation_type = self._get_operation_type(operation)
        required_libraries = self._get_required_libraries(operation_type)
        
        includes = []
        for lib_name in required_libraries:
            if lib_name in self.libraries:
                lib_spec = self.libraries[lib_name]
                lib_includes = self._resolve_library_includes(lib_spec)
                includes.extend(lib_includes)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(includes))
    
    def resolve_libraries(self, operation) -> List[str]:
        """
        Resolve library dependencies for an operation.
        
        Args:
            operation: HWCustomOp instance
            
        Returns:
            List of library names/paths needed
        """
        operation_type = self._get_operation_type(operation)
        return self._get_required_libraries(operation_type)
    
    def resolve_library_paths(self, operation) -> Dict[str, str]:
        """
        Resolve library paths for an operation.
        
        Args:
            operation: HWCustomOp instance
            
        Returns:
            Dictionary mapping library names to resolved paths
        """
        operation_type = self._get_operation_type(operation)
        required_libraries = self._get_required_libraries(operation_type)
        
        paths = {}
        for lib_name in required_libraries:
            if lib_name in self.libraries:
                lib_spec = self.libraries[lib_name]
                resolved_path = self._resolve_path(lib_spec.path)
                if resolved_path:  # Only include if path exists
                    paths[lib_name] = resolved_path
                    
        return paths
    
    def _get_operation_type(self, operation) -> str:
        """
        Get the operation type string for library resolution.
        
        Args:
            operation: HWCustomOp instance
            
        Returns:
            Operation type string
        """
        if hasattr(operation, 'onnx_node') and operation.onnx_node:
            return operation.onnx_node.op_type
        elif hasattr(operation, '__class__'):
            return operation.__class__.__name__
        else:
            return str(type(operation).__name__)
    
    def _get_required_libraries(self, operation_type: str) -> List[str]:
        """
        Get libraries required for an operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            List of required library names
        """
        required = []
        
        for lib_name, lib_spec in self.libraries.items():
            if self._operation_requires_library(operation_type, lib_spec):
                required.append(lib_name)
                
                # Add dependencies recursively
                for dep in lib_spec.dependencies:
                    if dep not in required and dep in self.libraries:
                        required.append(dep)
        
        return required
    
    def _operation_requires_library(self, operation_type: str, lib_spec: LibrarySpec) -> bool:
        """
        Check if an operation type requires a specific library.
        
        Args:
            operation_type: Type of operation
            lib_spec: Library specification
            
        Returns:
            True if operation requires this library
        """
        # Check for wildcard match
        if '*' in lib_spec.required_for:
            return True
            
        # Check for exact match
        if operation_type in lib_spec.required_for:
            return True
            
        # Check for pattern matches (e.g., *_rtl, *_hls)
        for pattern in lib_spec.required_for:
            if pattern.endswith('*') and operation_type.startswith(pattern[:-1]):
                return True
            elif pattern.startswith('*') and operation_type.endswith(pattern[1:]):
                return True
                
        return False
    
    def _resolve_library_includes(self, lib_spec: LibrarySpec) -> List[str]:
        """
        Resolve include file paths for a library.
        
        Args:
            lib_spec: Library specification
            
        Returns:
            List of resolved include file paths
        """
        includes = []
        base_path = self._resolve_path(lib_spec.path)
        
        for include_file in lib_spec.include_files:
            if base_path and lib_spec.library_type != LibraryType.SYSTEM_LIBRARY:
                # For non-system libraries, prepend the base path
                full_path = os.path.join(base_path, include_file)
                if os.path.exists(full_path):
                    includes.append(full_path)
                else:
                    self.logger.debug(f"Include file not found: {full_path}")
            else:
                # For system libraries, use the include file name as-is
                includes.append(include_file)
                
        return includes
    
    def _resolve_path(self, path: str) -> Optional[str]:
        """
        Resolve a path with environment variable substitution and fallbacks.
        
        Args:
            path: Path that may contain environment variables
            
        Returns:
            Resolved path, or None if path doesn't exist
        """
        if not path:
            return None
            
        # First try direct environment variable expansion
        resolved = os.path.expandvars(path)
        if os.path.exists(resolved):
            return os.path.abspath(resolved)
            
        # Try fallback paths for common FINN directory structures
        fallback_paths = self._get_fallback_paths(path)
        for fallback in fallback_paths:
            if os.path.exists(fallback):
                self.logger.info(f"Using fallback path: {fallback} (for {path})")
                return os.path.abspath(fallback)
        
        self.logger.debug(f"Path not found: {resolved} (from {path})")
        return None
    
    def _get_fallback_paths(self, original_path: str) -> List[str]:
        """
        Generate fallback paths for common FINN directory structures.
        
        Args:
            original_path: Original path with environment variables
            
        Returns:
            List of fallback paths to try
        """
        fallbacks = []
        
        # Common workspace directory patterns
        workspace_dirs = [
            os.getcwd(),  # Current working directory
            '/workspace',  # Docker workspace
            '/home/tafk/dev/tafk-finn-1',  # Current project directory
            os.path.expanduser('~/finn'),  # User home finn directory
        ]
        
        # Extract the relative part after environment variables
        if '$FINN_DEPS_DIR' in original_path:
            relative_part = original_path.replace('$FINN_DEPS_DIR/', '')
            for workspace in workspace_dirs:
                fallbacks.extend([
                    os.path.join(workspace, 'deps', relative_part),
                    os.path.join(workspace, 'dependencies', relative_part),
                    os.path.join(workspace, relative_part),
                ])
                
        elif '$FINN_ROOT' in original_path:
            relative_part = original_path.replace('$FINN_ROOT/', '')
            for workspace in workspace_dirs:
                fallbacks.extend([
                    os.path.join(workspace, relative_part),
                    os.path.join(workspace, 'finn', relative_part),
                    os.path.join(workspace, 'src', 'finn', relative_part),
                ])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(fallbacks))
    
    def detect_finn_environment(self) -> Dict[str, str]:
        """
        Detect FINN environment variables and paths.
        
        Returns:
            Dictionary of detected environment variables and paths
        """
        env_info = {}
        
        # Check for environment variables
        finn_vars = ['FINN_ROOT', 'FINN_DEPS_DIR', 'FINN_HLS_ROOT', 'XILINX_VIVADO']
        for var in finn_vars:
            value = os.environ.get(var)
            if value and os.path.exists(value):
                env_info[var] = value
            else:
                env_info[var] = None
                
        # Try to detect common paths automatically
        auto_detected = self._auto_detect_finn_paths()
        env_info.update(auto_detected)
        
        return env_info
        
    def _auto_detect_finn_paths(self) -> Dict[str, Optional[str]]:
        """
        Auto-detect common FINN paths in the environment.
        
        Returns:
            Dictionary of auto-detected paths
        """
        detected = {}
        
        # Look for deps directory
        deps_candidates = [
            './deps',
            '../deps',
            '/workspace/deps',
            os.path.join(os.getcwd(), 'deps')
        ]
        
        for candidate in deps_candidates:
            if os.path.exists(candidate):
                detected['AUTO_FINN_DEPS_DIR'] = os.path.abspath(candidate)
                break
        else:
            detected['AUTO_FINN_DEPS_DIR'] = None
            
        # Look for FINN root
        root_candidates = [
            '.',
            '/workspace',
            os.getcwd(),
            os.path.dirname(os.getcwd())
        ]
        
        for candidate in root_candidates:
            # Check if this looks like a FINN root (has src/finn directory)
            if os.path.exists(os.path.join(candidate, 'src', 'finn')):
                detected['AUTO_FINN_ROOT'] = os.path.abspath(candidate)
                break
        else:
            detected['AUTO_FINN_ROOT'] = None
            
        return detected
    
    def get_library_info(self, library_name: str) -> Optional[LibrarySpec]:
        """
        Get information about a registered library.
        
        Args:
            library_name: Name of the library
            
        Returns:
            LibrarySpec if found, None otherwise
        """
        return self.libraries.get(library_name)
    
    def list_libraries(self, library_type: Optional[LibraryType] = None) -> List[str]:
        """
        List registered libraries, optionally filtered by type.
        
        Args:
            library_type: Optional filter by library type
            
        Returns:
            List of library names
        """
        if library_type is None:
            return list(self.libraries.keys())
        else:
            return [name for name, spec in self.libraries.items() 
                    if spec.library_type == library_type]
    
    def validate_dependencies(self, operation) -> Dict[str, bool]:
        """
        Validate that all required dependencies are available.
        
        Args:
            operation: HWCustomOp instance
            
        Returns:
            Dictionary mapping library names to availability status
        """
        operation_type = self._get_operation_type(operation)
        required_libraries = self._get_required_libraries(operation_type)
        
        status = {}
        for lib_name in required_libraries:
            if lib_name in self.libraries:
                lib_spec = self.libraries[lib_name]
                resolved_path = self._resolve_path(lib_spec.path)
                status[lib_name] = resolved_path is not None or lib_spec.library_type == LibraryType.SYSTEM_LIBRARY
            else:
                status[lib_name] = False
                
        return status