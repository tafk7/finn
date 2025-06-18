"""
Clean Backend Registration for FINN Operations

This module provides clean backend registration without legacy compatibility bloat.
Supports A/B testing between old and new backends with configuration-driven selection.
"""

import logging
from typing import Dict, Type, Optional, Any
from .backend_registry import BackendRegistry


class CG_BackendRegistry(BackendRegistry):
    """
    Clean backend registry with A/B testing support.
    
    Extends the existing BackendRegistry to support both legacy and clean
    implementations, allowing for gradual migration and validation.
    """
    
    def __init__(self):
        """Initialize clean backend registry."""
        super().__init__()
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
        
        # Track both clean and legacy backends
        self._clean_hls_backends: Dict[str, Type] = {}
        self._clean_rtl_backends: Dict[str, Type] = {}
        
        # Configuration for A/B testing
        self._use_clean_backends = False  # Default to legacy until validation complete
        self._clean_backend_whitelist = set()  # Operations approved for clean backends
        
        self.logger.debug("Initialized clean backend registry")
    
    def register_clean_hls_backend(self, operation_name: str, backend_class: Type):
        """
        Register a clean HLS backend implementation.
        
        Args:
            operation_name: Name of the operation (e.g., 'Thresholding', 'MatrixVectorActivation')
            backend_class: Clean backend class to register
        """
        self._clean_hls_backends[operation_name] = backend_class
        self.logger.debug(f"Registered clean HLS backend: {operation_name} -> {backend_class.__name__}")
    
    def register_clean_rtl_backend(self, operation_name: str, backend_class: Type):
        """
        Register a clean RTL backend implementation.
        
        Args:
            operation_name: Name of the operation
            backend_class: Clean backend class to register
        """
        self._clean_rtl_backends[operation_name] = backend_class
        self.logger.debug(f"Registered clean RTL backend: {operation_name} -> {backend_class.__name__}")
    
    def get_hls_backend(self, operation_name: str, prefer_clean: bool = None) -> Optional[Type]:
        """
        Get HLS backend with clean/legacy selection logic.
        
        Args:
            operation_name: Name of the operation
            prefer_clean: Override to prefer clean backend (for testing)
            
        Returns:
            Backend class (clean or legacy based on configuration)
        """
        # Determine whether to use clean backend
        use_clean = self._should_use_clean_backend(operation_name, prefer_clean)
        
        if use_clean and operation_name in self._clean_hls_backends:
            backend_class = self._clean_hls_backends[operation_name]
            self.logger.debug(f"Using clean HLS backend for {operation_name}: {backend_class.__name__}")
            return backend_class
        
        # Fallback to legacy backend
        legacy_backend = super().get_hls_backend(operation_name)
        if legacy_backend:
            self.logger.debug(f"Using legacy HLS backend for {operation_name}: {legacy_backend.__name__}")
        return legacy_backend
    
    def get_rtl_backend(self, operation_name: str, prefer_clean: bool = None) -> Optional[Type]:
        """
        Get RTL backend with clean/legacy selection logic.
        
        Args:
            operation_name: Name of the operation
            prefer_clean: Override to prefer clean backend (for testing)
            
        Returns:
            Backend class (clean or legacy based on configuration)
        """
        # Determine whether to use clean backend
        use_clean = self._should_use_clean_backend(operation_name, prefer_clean)
        
        if use_clean and operation_name in self._clean_rtl_backends:
            backend_class = self._clean_rtl_backends[operation_name]
            self.logger.debug(f"Using clean RTL backend for {operation_name}: {backend_class.__name__}")
            return backend_class
        
        # Fallback to legacy backend
        legacy_backend = super().get_rtl_backend(operation_name)
        if legacy_backend:
            self.logger.debug(f"Using legacy RTL backend for {operation_name}: {legacy_backend.__name__}")
        return legacy_backend
    
    def _should_use_clean_backend(self, operation_name: str, prefer_clean_override: bool = None) -> bool:
        """
        Determine whether to use clean backend for given operation.
        
        Args:
            operation_name: Name of the operation
            prefer_clean_override: Override for testing purposes
            
        Returns:
            True if clean backend should be used
        """
        # Test override takes precedence
        if prefer_clean_override is not None:
            return prefer_clean_override
        
        # Check if operation is whitelisted for clean backends
        if operation_name in self._clean_backend_whitelist:
            return True
        
        # Global clean backend flag
        return self._use_clean_backends
    
    def enable_clean_backends(self, operation_names: Optional[list] = None):
        """
        Enable clean backends for all operations or specific operations.
        
        Args:
            operation_names: List of operations to enable, or None for all
        """
        if operation_names is None:
            self._use_clean_backends = True
            self.logger.info("Enabled clean backends globally")
        else:
            self._clean_backend_whitelist.update(operation_names)
            self.logger.info(f"Enabled clean backends for: {operation_names}")
    
    def disable_clean_backends(self, operation_names: Optional[list] = None):
        """
        Disable clean backends for all operations or specific operations.
        
        Args:
            operation_names: List of operations to disable, or None for all
        """
        if operation_names is None:
            self._use_clean_backends = False
            self._clean_backend_whitelist.clear()
            self.logger.info("Disabled clean backends globally")
        else:
            self._clean_backend_whitelist.difference_update(operation_names)
            self.logger.info(f"Disabled clean backends for: {operation_names}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        legacy_stats = super().get_registry_stats()
        clean_stats = {
            'clean_hls_backends': len(self._clean_hls_backends),
            'clean_rtl_backends': len(self._clean_rtl_backends),
            'clean_backends_enabled': self._use_clean_backends,
            'whitelisted_operations': len(self._clean_backend_whitelist),
            'clean_hls_operations': list(self._clean_hls_backends.keys()),
            'clean_rtl_operations': list(self._clean_rtl_backends.keys()),
        }
        
        return {**legacy_stats, **clean_stats}
    
    def list_clean_backends(self) -> Dict[str, Dict[str, Type]]:
        """
        List all registered clean backends.
        
        Returns:
            Dictionary with 'hls' and 'rtl' keys containing backend mappings
        """
        return {
            'hls': dict(self._clean_hls_backends),
            'rtl': dict(self._clean_rtl_backends)
        }


def register_all_clean_backends() -> CG_BackendRegistry:
    """
    Register all available clean backend implementations.
    
    Returns:
        Configured CG_BackendRegistry with clean backends registered
    """
    logger = logging.getLogger(__name__)
    registry = CG_BackendRegistry()
    
    # Register clean HLS backends
    logger.debug("Registering clean HLS backends...")
    
    try:
        from ..custom_op.fpgadataflow.hls.CG_thresholding_hls import CG_ThresholdingHLS
        registry.register_clean_hls_backend('Thresholding', CG_ThresholdingHLS)
    except ImportError as e:
        logger.debug(f"Could not import CG_ThresholdingHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.CG_mvau_hls import CG_MVAU_HLS
        registry.register_clean_hls_backend('MatrixVectorActivation', CG_MVAU_HLS)
        registry.register_clean_hls_backend('MVAU', CG_MVAU_HLS)  # Alternative name
    except ImportError as e:
        logger.debug(f"Could not import CG_MVAU_HLS: {e}")
    
    # Register clean RTL backends
    logger.debug("Registering clean RTL backends...")
    
    try:
        from ..custom_op.fpgadataflow.rtl.CG_thresholding_rtl import CG_ThresholdingRTL
        registry.register_clean_rtl_backend('Thresholding', CG_ThresholdingRTL)
    except ImportError as e:
        logger.debug(f"Could not import CG_ThresholdingRTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.CG_mvau_rtl import CG_MVAU_RTL
        registry.register_clean_rtl_backend('MatrixVectorActivation', CG_MVAU_RTL)
        registry.register_clean_rtl_backend('MVAU', CG_MVAU_RTL)  # Alternative name
    except ImportError as e:
        logger.debug(f"Could not import CG_MVAU_RTL: {e}")
    
    # Register all legacy backends as well
    from .backend_registration import register_all_backends
    legacy_registry = register_all_backends()
    
    # Copy legacy registrations to clean registry
    for op_name, backend_class in legacy_registry._hls_backends.items():
        registry.register_hls_backend(op_name, backend_class)
    
    for op_name, backend_class in legacy_registry._rtl_backends.items():
        registry.register_rtl_backend(op_name, backend_class)
    
    stats = registry.get_registry_stats()
    logger.info(f"Clean backend registration complete: {stats['clean_hls_backends']} clean HLS, "
                f"{stats['clean_rtl_backends']} clean RTL, {stats['hls_backends']} legacy HLS, "
                f"{stats['rtl_backends']} legacy RTL backends")
    
    return registry


# Global clean registry instance
_global_clean_registry = None


def get_clean_backend_registry() -> CG_BackendRegistry:
    """
    Get the global clean backend registry instance.
    
    Returns:
        Global CG_BackendRegistry instance
    """
    global _global_clean_registry
    if _global_clean_registry is None:
        _global_clean_registry = register_all_clean_backends()
    return _global_clean_registry


def reset_clean_backend_registry():
    """Reset the global clean backend registry (useful for testing)."""
    global _global_clean_registry
    _global_clean_registry = None


# Convenience functions for clean backend lookup
def get_clean_hls_backend(operation_name: str, prefer_clean: bool = True):
    """
    Get clean HLS backend class for operation.
    
    Args:
        operation_name: Name of the operation
        prefer_clean: Whether to prefer clean implementation
        
    Returns:
        Clean HLS backend class if found, legacy fallback otherwise
    """
    return get_clean_backend_registry().get_hls_backend(operation_name, prefer_clean)


def get_clean_rtl_backend(operation_name: str, prefer_clean: bool = True):
    """
    Get clean RTL backend class for operation.
    
    Args:
        operation_name: Name of the operation
        prefer_clean: Whether to prefer clean implementation
        
    Returns:
        Clean RTL backend class if found, legacy fallback otherwise
    """
    return get_clean_backend_registry().get_rtl_backend(operation_name, prefer_clean)