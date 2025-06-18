"""
Explicit Backend Registry for FINN Code Generation

This module provides simple explicit backend registration without auto-discovery
complexity. Replaces the complex BackendRegistry with straightforward dictionary
lookups for deterministic backend selection.
"""

import logging
from typing import Dict, Type, Optional, List


class BackendRegistry:
    """
    Simple explicit backend registration - no auto-discovery.
    
    Eliminates 500+ lines of auto-discovery complexity in favor of
    explicit registration that is predictable and fast.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._hls_backends: Dict[str, Type] = {}
        self._rtl_backends: Dict[str, Type] = {}
        
        self.logger.debug("Initialized BackendRegistry")
    
    def register_hls_backend(self, operation_name: str, backend_class: Type):
        """
        Register HLS backend for operation.
        
        Args:
            operation_name: Name of the operation (e.g., 'Thresholding', 'MVAU')
            backend_class: Backend class that handles this operation
        """
        self._hls_backends[operation_name] = backend_class
        self.logger.debug(f"Registered HLS backend: {operation_name} -> {backend_class.__name__}")
    
    def register_rtl_backend(self, operation_name: str, backend_class: Type):
        """
        Register RTL backend for operation.
        
        Args:
            operation_name: Name of the operation (e.g., 'Thresholding', 'MVAU')
            backend_class: Backend class that handles this operation
        """
        self._rtl_backends[operation_name] = backend_class
        self.logger.debug(f"Registered RTL backend: {operation_name} -> {backend_class.__name__}")
    
    def get_hls_backend(self, operation_name: str) -> Optional[Type]:
        """
        Get HLS backend for operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Backend class if found, None otherwise
        """
        backend_class = self._hls_backends.get(operation_name)
        if backend_class:
            self.logger.debug(f"Found HLS backend for {operation_name}: {backend_class.__name__}")
        else:
            self.logger.debug(f"No HLS backend found for operation: {operation_name}")
        return backend_class
    
    def get_rtl_backend(self, operation_name: str) -> Optional[Type]:
        """
        Get RTL backend for operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Backend class if found, None otherwise
        """
        backend_class = self._rtl_backends.get(operation_name)
        if backend_class:
            self.logger.debug(f"Found RTL backend for {operation_name}: {backend_class.__name__}")
        else:
            self.logger.debug(f"No RTL backend found for operation: {operation_name}")
        return backend_class
    
    def list_hls_operations(self) -> List[str]:
        """
        List all operations with HLS backends.
        
        Returns:
            List of operation names that have HLS backends
        """
        return list(self._hls_backends.keys())
    
    def list_rtl_operations(self) -> List[str]:
        """
        List all operations with RTL backends.
        
        Returns:
            List of operation names that have RTL backends
        """
        return list(self._rtl_backends.keys())
    
    def get_all_hls_backends(self) -> Dict[str, Type]:
        """
        Get all registered HLS backends.
        
        Returns:
            Dictionary mapping operation names to backend classes
        """
        return self._hls_backends.copy()
    
    def get_all_rtl_backends(self) -> Dict[str, Type]:
        """
        Get all registered RTL backends.
        
        Returns:
            Dictionary mapping operation names to backend classes
        """
        return self._rtl_backends.copy()
    
    def clear_registry(self):
        """Clear all registered backends."""
        self._hls_backends.clear()
        self._rtl_backends.clear()
        self.logger.debug("Backend registry cleared")
    
    def get_registry_stats(self) -> Dict[str, int]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with backend counts
        """
        return {
            'hls_backends': len(self._hls_backends),
            'rtl_backends': len(self._rtl_backends),
            'total_backends': len(self._hls_backends) + len(self._rtl_backends)
        }