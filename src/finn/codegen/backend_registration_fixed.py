"""
Backend Registration for FINN Operations - FIXED VERSION

This module explicitly registers all available backends with the registry.
Replaces auto-discovery with explicit, predictable registration.

Fixed to use correct class names matching actual implementation files.
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
    
    # Register HLS backends with CORRECT class names
    logger.debug("Registering HLS backends...")
    
    # Core operations
    try:
        from ..custom_op.fpgadataflow.hls.thresholding_hls import Thresholding_hls
        registry.register_hls_backend('Thresholding', Thresholding_hls)
        logger.debug("Registered Thresholding_hls")
    except ImportError as e:
        logger.warning(f"Could not import Thresholding_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
        registry.register_hls_backend('MatrixVectorActivation', MVAU_hls)
        registry.register_hls_backend('MVAU', MVAU_hls)  # Alternative name
        logger.debug("Registered MVAU_hls")
    except ImportError as e:
        logger.warning(f"Could not import MVAU_hls: {e}")
    
    # Fixed class names for other operations
    try:
        from ..custom_op.fpgadataflow.hls.addstreams_hls import AddStreams_hls
        registry.register_hls_backend('AddStreams', AddStreams_hls)
        logger.debug("Registered AddStreams_hls")
    except ImportError as e:
        logger.warning(f"Could not import AddStreams_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.concat_hls import StreamingConcat_hls
        registry.register_hls_backend('StreamingConcat', StreamingConcat_hls)
        registry.register_hls_backend('Concat', StreamingConcat_hls)  # Alternative name
        logger.debug("Registered StreamingConcat_hls")
    except ImportError as e:
        logger.warning(f"Could not import StreamingConcat_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.convolutioninputgenerator_hls import ConvolutionInputGenerator_hls
        registry.register_hls_backend('ConvolutionInputGenerator', ConvolutionInputGenerator_hls)
        logger.debug("Registered ConvolutionInputGenerator_hls")
    except ImportError as e:
        logger.warning(f"Could not import ConvolutionInputGenerator_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.duplicatestreams_hls import DuplicateStreams_hls
        registry.register_hls_backend('DuplicateStreams', DuplicateStreams_hls)
        logger.debug("Registered DuplicateStreams_hls")
    except ImportError as e:
        logger.warning(f"Could not import DuplicateStreams_hls: {e}")
    
    # ElementwiseBinary has multiple subclasses
    try:
        from ..custom_op.fpgadataflow.hls.elementwise_binary_hls import (
            ElementwiseAdd_hls, ElementwiseSub_hls, ElementwiseMul_hls, 
            ElementwiseDiv_hls, ElementwiseAnd_hls, ElementwiseOr_hls,
            ElementwiseXor_hls
        )
        registry.register_hls_backend('ElementwiseAdd', ElementwiseAdd_hls)
        registry.register_hls_backend('ElementwiseSub', ElementwiseSub_hls)
        registry.register_hls_backend('ElementwiseMul', ElementwiseMul_hls)
        registry.register_hls_backend('ElementwiseDiv', ElementwiseDiv_hls)
        registry.register_hls_backend('ElementwiseAnd', ElementwiseAnd_hls)
        registry.register_hls_backend('ElementwiseOr', ElementwiseOr_hls)
        registry.register_hls_backend('ElementwiseXor', ElementwiseXor_hls)
        logger.debug("Registered ElementwiseBinary operations")
    except ImportError as e:
        logger.warning(f"Could not import ElementwiseBinary operations: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.fmpadding_hls import FMPadding_hls
        registry.register_hls_backend('FMPadding', FMPadding_hls)
        logger.debug("Registered FMPadding_hls")
    except ImportError as e:
        logger.warning(f"Could not import FMPadding_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.globalaccpool_hls import GlobalAccPool_hls
        registry.register_hls_backend('GlobalAccPool', GlobalAccPool_hls)
        logger.debug("Registered GlobalAccPool_hls")
    except ImportError as e:
        logger.warning(f"Could not import GlobalAccPool_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.labelselect_hls import LabelSelect_hls
        registry.register_hls_backend('LabelSelect', LabelSelect_hls)
        logger.debug("Registered LabelSelect_hls")
    except ImportError as e:
        logger.warning(f"Could not import LabelSelect_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.lookup_hls import Lookup_hls
        registry.register_hls_backend('Lookup', Lookup_hls)
        logger.debug("Registered Lookup_hls")
    except ImportError as e:
        logger.warning(f"Could not import Lookup_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.pool_hls import Pool_hls
        registry.register_hls_backend('Pool', Pool_hls)
        logger.debug("Registered Pool_hls")
    except ImportError as e:
        logger.warning(f"Could not import Pool_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.streamingeltwise_hls import StreamingEltwise_hls
        registry.register_hls_backend('StreamingEltwise', StreamingEltwise_hls)
        logger.debug("Registered StreamingEltwise_hls")
    except ImportError as e:
        logger.warning(f"Could not import StreamingEltwise_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.streamingmaxpool_hls import StreamingMaxPool_hls
        registry.register_hls_backend('StreamingMaxPool', StreamingMaxPool_hls)
        logger.debug("Registered StreamingMaxPool_hls")
    except ImportError as e:
        logger.warning(f"Could not import StreamingMaxPool_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls import StreamingDataWidthConverter_hls
        registry.register_hls_backend('StreamingDataWidthConverter', StreamingDataWidthConverter_hls)
        logger.debug("Registered StreamingDataWidthConverter_hls")
    except ImportError as e:
        logger.warning(f"Could not import StreamingDataWidthConverter_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.upsampler_hls import UpsampleNearestNeighbour_hls
        registry.register_hls_backend('UpsampleNearestNeighbour', UpsampleNearestNeighbour_hls)
        registry.register_hls_backend('Upsampler', UpsampleNearestNeighbour_hls)  # Alternative name
        logger.debug("Registered UpsampleNearestNeighbour_hls")
    except ImportError as e:
        logger.warning(f"Could not import UpsampleNearestNeighbour_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.vectorvectoractivation_hls import VVAU_hls
        registry.register_hls_backend('VectorVectorActivation', VVAU_hls)
        registry.register_hls_backend('VVAU', VVAU_hls)  # Alternative name
        logger.debug("Registered VVAU_hls")
    except ImportError as e:
        logger.warning(f"Could not import VVAU_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.channelwise_op_hls import ChannelwiseOp_hls
        registry.register_hls_backend('ChannelwiseOp', ChannelwiseOp_hls)
        logger.debug("Registered ChannelwiseOp_hls")
    except ImportError as e:
        logger.warning(f"Could not import ChannelwiseOp_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.checksum_hls import CheckSum_hls
        registry.register_hls_backend('CheckSum', CheckSum_hls)
        logger.debug("Registered CheckSum_hls")
    except ImportError as e:
        logger.warning(f"Could not import CheckSum_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.downsampler_hls import DownSampler_hls
        registry.register_hls_backend('DownSampler', DownSampler_hls)
        logger.debug("Registered DownSampler_hls")
    except ImportError as e:
        logger.warning(f"Could not import DownSampler_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.fmpadding_pixel_hls import FMPadding_Pixel_hls
        registry.register_hls_backend('FMPadding_Pixel', FMPadding_Pixel_hls)
        logger.debug("Registered FMPadding_Pixel_hls")
    except ImportError as e:
        logger.warning(f"Could not import FMPadding_Pixel_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.iodma_hls import IODMA_hls
        registry.register_hls_backend('IODMA', IODMA_hls)
        logger.debug("Registered IODMA_hls")
    except ImportError as e:
        logger.warning(f"Could not import IODMA_hls: {e}")
    
    try:
        from ..custom_op.fpgadataflow.hls.tlastmarker_hls import TLastMarker_hls
        registry.register_hls_backend('TLastMarker', TLastMarker_hls)
        logger.debug("Registered TLastMarker_hls")
    except ImportError as e:
        logger.warning(f"Could not import TLastMarker_hls: {e}")
    
    # Register RTL backends with CORRECT class names
    logger.debug("Registering RTL backends...")
    
    try:
        from ..custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
        registry.register_rtl_backend('Thresholding', Thresholding_rtl)
        logger.debug("Registered Thresholding_rtl")
    except ImportError as e:
        logger.warning(f"Could not import Thresholding_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
        registry.register_rtl_backend('MatrixVectorActivation', MVAU_rtl)
        registry.register_rtl_backend('MVAU', MVAU_rtl)  # Alternative name
        logger.debug("Registered MVAU_rtl")
    except ImportError as e:
        logger.warning(f"Could not import MVAU_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl import ConvolutionInputGenerator_rtl
        registry.register_rtl_backend('ConvolutionInputGenerator', ConvolutionInputGenerator_rtl)
        logger.debug("Registered ConvolutionInputGenerator_rtl")
    except ImportError as e:
        logger.warning(f"Could not import ConvolutionInputGenerator_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.fmpadding_rtl import FMPadding_rtl
        registry.register_rtl_backend('FMPadding', FMPadding_rtl)
        logger.debug("Registered FMPadding_rtl")
    except ImportError as e:
        logger.warning(f"Could not import FMPadding_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl import StreamingDataWidthConverter_rtl
        registry.register_rtl_backend('StreamingDataWidthConverter', StreamingDataWidthConverter_rtl)
        logger.debug("Registered StreamingDataWidthConverter_rtl")
    except ImportError as e:
        logger.warning(f"Could not import StreamingDataWidthConverter_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.streamingfifo_rtl import StreamingFIFO_rtl
        registry.register_rtl_backend('StreamingFIFO', StreamingFIFO_rtl)
        logger.debug("Registered StreamingFIFO_rtl")
    except ImportError as e:
        logger.warning(f"Could not import StreamingFIFO_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl import VVAU_rtl
        registry.register_rtl_backend('VectorVectorActivation', VVAU_rtl)
        registry.register_rtl_backend('VVAU', VVAU_rtl)  # Alternative name
        logger.debug("Registered VVAU_rtl")
    except ImportError as e:
        logger.warning(f"Could not import VVAU_rtl: {e}")
    
    try:
        from ..custom_op.fpgadataflow.rtl.mvau_rtl import DynMVU_rtl
        registry.register_rtl_backend('DynMVU', DynMVU_rtl)
        registry.register_rtl_backend('DynMVAU', DynMVU_rtl)  # Alternative name
        logger.debug("Registered DynMVU_rtl")
    except ImportError as e:
        logger.warning(f"Could not import DynMVU_rtl: {e}")
    
    stats = registry.get_registry_stats()
    logger.info(f"Backend registration complete: {stats['hls_backends']} HLS, {stats['rtl_backends']} RTL backends")
    
    # Log detailed registry contents for debugging
    logger.debug("HLS backends registered:")
    for op_name in sorted(registry._hls_backends.keys()):
        logger.debug(f"  - {op_name}: {registry._hls_backends[op_name].__name__}")
    
    logger.debug("RTL backends registered:")
    for op_name in sorted(registry._rtl_backends.keys()):
        logger.debug(f"  - {op_name}: {registry._rtl_backends[op_name].__name__}")
    
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