"""
Backend Registration for FINN Operations

This module explicitly registers all available backends with the registry.
Replaces auto-discovery with explicit, predictable registration.
"""

import logging
from .backend_registry import BackendRegistry


def register_all_backends() -> BackendRegistry:
    """
    Register all available backends explicitly.
    
    Returns:
        Configured BackendRegistry with all backends registered
    """
    logger = logging.getLogger(__name__)
    registry = BackendRegistry()
    
    # Register HLS backends
    logger.debug("Registering HLS backends...")
    
    try:
        from ..custom_op.fpgadataflow.hls.thresholding_hls import ThresholdingHLS
        registry.register_hls_backend('Thresholding', ThresholdingHLS)
    except ImportError as e:
        logger.debug(f"Could not import ThresholdingHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.mvau_hls import MVAU_HLS
        registry.register_hls_backend('MatrixVectorActivation', MVAU_HLS)
        registry.register_hls_backend('MVAU', MVAU_HLS)  # Alternative name
    except ImportError as e:
        logger.debug(f"Could not import MVAU_HLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MatrixVectorActivationHLS
        registry.register_hls_backend('MatrixVectorActivation', MatrixVectorActivationHLS)
    except ImportError as e:
        logger.debug(f"Could not import MatrixVectorActivationHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.addstreams_hls import AddStreamsHLS
        registry.register_hls_backend('AddStreams', AddStreamsHLS)
    except ImportError as e:
        logger.debug(f"Could not import AddStreamsHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.concat_hls import ConcatHLS
        registry.register_hls_backend('Concat', ConcatHLS)
    except ImportError as e:
        logger.debug(f"Could not import ConcatHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.convolutioninputgenerator_hls import ConvolutionInputGeneratorHLS
        registry.register_hls_backend('ConvolutionInputGenerator', ConvolutionInputGeneratorHLS)
    except ImportError as e:
        logger.debug(f"Could not import ConvolutionInputGeneratorHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.duplicatestreams_hls import DuplicateStreamsHLS
        registry.register_hls_backend('DuplicateStreams', DuplicateStreamsHLS)
    except ImportError as e:
        logger.debug(f"Could not import DuplicateStreamsHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.elementwise_binary_hls import ElementwiseBinaryHLS
        registry.register_hls_backend('ElementwiseBinary', ElementwiseBinaryHLS)
    except ImportError as e:
        logger.debug(f"Could not import ElementwiseBinaryHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.fmpadding_hls import FMPaddingHLS
        registry.register_hls_backend('FMPadding', FMPaddingHLS)
    except ImportError as e:
        logger.debug(f"Could not import FMPaddingHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.globalaccpool_hls import GlobalAccPoolHLS
        registry.register_hls_backend('GlobalAccPool', GlobalAccPoolHLS)
    except ImportError as e:
        logger.debug(f"Could not import GlobalAccPoolHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.labelselect_hls import LabelSelectHLS
        registry.register_hls_backend('LabelSelect', LabelSelectHLS)
    except ImportError as e:
        logger.debug(f"Could not import LabelSelectHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.lookup_hls import LookupHLS
        registry.register_hls_backend('Lookup', LookupHLS)
    except ImportError as e:
        logger.debug(f"Could not import LookupHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.pool_hls import PoolHLS
        registry.register_hls_backend('Pool', PoolHLS)
    except ImportError as e:
        logger.debug(f"Could not import PoolHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.streamingeltwise_hls import StreamingEltwiseHLS
        registry.register_hls_backend('StreamingEltwise', StreamingEltwiseHLS)
    except ImportError as e:
        logger.debug(f"Could not import StreamingEltwiseHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.streamingmaxpool_hls import StreamingMaxPoolHLS
        registry.register_hls_backend('StreamingMaxPool', StreamingMaxPoolHLS)
    except ImportError as e:
        logger.debug(f"Could not import StreamingMaxPoolHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.upsampler_hls import UpsamplerHLS
        registry.register_hls_backend('Upsampler', UpsamplerHLS)
    except ImportError as e:
        logger.debug(f"Could not import UpsamplerHLS: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.vectorvectoractivation_hls import VectorVectorActivationHLS
        registry.register_hls_backend('VectorVectorActivation', VectorVectorActivationHLS)
    except ImportError as e:
        logger.debug(f"Could not import VectorVectorActivationHLS: {e}")
    
    # Register RTL backends
    logger.debug("Registering RTL backends...")
    
    try:
        from ..custom_op.fpgadataflow.rtl.thresholding_rtl import ThresholdingRTL
        registry.register_rtl_backend('Thresholding', ThresholdingRTL)
    except ImportError as e:
        logger.debug(f"Could not import ThresholdingRTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MatrixVectorActivationRTL
        registry.register_rtl_backend('MatrixVectorActivation', MatrixVectorActivationRTL)
    except ImportError as e:
        logger.debug(f"Could not import MatrixVectorActivationRTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.dynmvau_rtl import DynMVAURTL
        registry.register_rtl_backend('DynMVAU', DynMVAURTL)
    except ImportError as e:
        logger.debug(f"Could not import DynMVAURTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl import ConvolutionInputGeneratorRTL
        registry.register_rtl_backend('ConvolutionInputGenerator', ConvolutionInputGeneratorRTL)
    except ImportError as e:
        logger.debug(f"Could not import ConvolutionInputGeneratorRTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.fmpadding_rtl import FMPaddingRTL
        registry.register_rtl_backend('FMPadding', FMPaddingRTL)
    except ImportError as e:
        logger.debug(f"Could not import FMPaddingRTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl import StreamingDataWidthConverterRTL
        registry.register_rtl_backend('StreamingDataWidthConverter', StreamingDataWidthConverterRTL)
    except ImportError as e:
        logger.debug(f"Could not import StreamingDataWidthConverterRTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.streamingfifo_rtl import StreamingFIFORTL
        registry.register_rtl_backend('StreamingFIFO', StreamingFIFORTL)
    except ImportError as e:
        logger.debug(f"Could not import StreamingFIFORTL: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl import VectorVectorActivationRTL
        registry.register_rtl_backend('VectorVectorActivation', VectorVectorActivationRTL)
    except ImportError as e:
        logger.debug(f"Could not import VectorVectorActivationRTL: {e}")
    
    stats = registry.get_registry_stats()
    logger.info(f"Backend registration complete: {stats['hls_backends']} HLS, {stats['rtl_backends']} RTL backends")
    
    return registry


# Global registry instance
_global_registry = None


def get_backend_registry() -> BackendRegistry:
    """
    Get the global backend registry instance.
    
    Returns:
        Global BackendRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = register_all_backends()
    return _global_registry


def reset_backend_registry():
    """Reset the global backend registry (useful for testing)."""
    global _global_registry
    _global_registry = None


# Convenience functions for backend lookup
def get_hls_backend(operation_name: str):
    """
    Get HLS backend class for operation.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        HLS backend class if found, None otherwise
    """
    return get_backend_registry().get_hls_backend(operation_name)


def get_rtl_backend(operation_name: str):
    """
    Get RTL backend class for operation.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        RTL backend class if found, None otherwise
    """
    return get_backend_registry().get_rtl_backend(operation_name)