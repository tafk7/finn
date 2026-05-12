############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Brainsmith integration for FINN.

Entry point: brainsmith.plugins -> finn = finn.brainsmith_integration:register_all

Strategy:
- Auto-discovery of components via module inspection (~5-10ms, cached)
- Zero dependencies on brainsmith (returns metadata only)
- Lazy loading: components imported only when accessed
- Convention-based metadata inference with explicit overrides

Components:
- Steps: Auto-discovered from build_dataflow_steps module
- Kernels: Auto-discovered HWCustomOp subclasses
- Backends: Auto-discovered HLS/RTL implementations
- Infer Transforms: Convention-based discovery (Infer{Name}Layer)

Infrastructure Kernels:
Infrastructure kernels (DuplicateStreams, StreamingFIFO, StreamingDataWidthConverter)
are marked with is_infrastructure=True and are filtered out of InferKernelList
when kernel_classes=None. They are inserted by topology transforms (InsertFIFO,
InsertDWC, InsertDuplicateStreams) rather than pattern matching.

Legacy Components:
CheckSum_hls, TLastMarker_hls, IODMA_hls are legacy FINN backend-only components
(no base kernel class) and are not registered in Brainsmith.

Maintenance:
Components are auto-discovered. To exclude components or mark as infrastructure,
update the configuration sets below (EXCLUDED_KERNELS, INFRASTRUCTURE_KERNELS, etc.).
"""

import inspect
from typing import Any, Optional


# ============================================================================
# DISCOVERY CONFIGURATION
# ============================================================================

# Components intentionally excluded from registration
EXCLUDED_KERNELS = {
    # Sub-components created by transforms, not from ONNX directly
    "FMPadding",
    "FMPadding_Pixel",
    # Base kernel with no backends (backends target specialized variants)
    "ElementwiseBinaryOperation",
    # Specialized ElementwiseBinary variants (backends target these, but they're
    # created by InferElementwiseBinaryOperation, not from ONNX directly)
    "ElementwiseAdd",
    "ElementwiseMul",
    "ElementwiseSub",
    "ElementwiseDiv",
    "ElementwiseAnd",
    "ElementwiseOr",
    "ElementwiseXor",
    "ElementwiseGreater",
    "ElementwiseGreaterOrEqual",
    "ElementwiseLess",
    "ElementwiseLessOrEqual",
    "ElementwiseEqual",
    "ElementwiseBitwiseAnd",
    "ElementwiseBitwiseOr",
    "ElementwiseBitwiseXor",
}

EXCLUDED_BACKENDS = {
    # Legacy backend-only components without base kernel classes
    "CheckSum_hls",
    "TLastMarker_hls",
    "IODMA_hls",
    # Base ElementwiseBinary backend
    "ElementwiseBinaryOperation_hls",
    # ElementwiseBinary variants (not registered as they target specialized kernels)
    "ElementwiseAdd_hls",
    "ElementwiseMul_hls",
    "ElementwiseSub_hls",
    "ElementwiseDiv_hls",
    "ElementwiseAnd_hls",
    "ElementwiseOr_hls",
    "ElementwiseXor_hls",
    "ElementwiseGreater_hls",
    "ElementwiseGreaterOrEqual_hls",
    "ElementwiseLess_hls",
    "ElementwiseLessOrEqual_hls",
    "ElementwiseEqual_hls",
    "ElementwiseBitwiseAnd_hls",
    "ElementwiseBitwiseOr_hls",
    "ElementwiseBitwiseXor_hls",
    # Backends for padding sub-components
    "FMPadding_hls",
    "FMPadding_Pixel_hls",
    "FMPadding_rtl",
    # Loop components (internal infrastructure)
    "FINNLoop",
}

# Infrastructure kernels inserted by topology transforms (not pattern matching)
INFRASTRUCTURE_KERNELS = {
    "StreamingFIFO",  # Inserted by InsertFIFO/InsertAndSetFIFODepths
    "StreamingDataWidthConverter",  # Inserted by InsertDWC
    "DuplicateStreams",  # Inserted by InsertDuplicateStreams
    "InnerShuffle",  # Inserted by InferInnerOuterShuffles
    "OuterShuffle",  # Inserted by InferInnerOuterShuffles
}

# Manual override for kernels without standard infer transforms
KERNEL_INFER_OVERRIDES = {
    # Infrastructure kernels may not have infer transforms
    "StreamingFIFO": None,
    "StreamingDataWidthConverter": None,
    # Non-standard infer transform names
    "MVAU": {
        "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
        "class_name": "InferQuantizedMatrixVectorActivation",
    },
    "VVAU": {
        "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
        "class_name": "InferVectorVectorActivation",
    },
}


# ============================================================================
# AUTO-DISCOVERY FUNCTIONS
# ============================================================================

def _find_infer_transform(kernel_name: str) -> Optional[dict[str, str]]:
    """Find infer transform for a kernel by convention.

    Searches for Infer{KernelName}Layer in convert_to_hw_layers module.

    Args:
        kernel_name: Name of kernel (e.g., 'AddStreams')

    Returns:
        Infer transform spec dict, or None if not found
    """
    # Check for manual override first
    if kernel_name in KERNEL_INFER_OVERRIDES:
        return KERNEL_INFER_OVERRIDES[kernel_name]

    # Import the module containing infer transforms
    try:
        from finn.transformation.fpgadataflow import convert_to_hw_layers
    except ImportError:
        return None

    # Try common patterns
    patterns = [
        f"Infer{kernel_name}Layer",
        f"Infer{kernel_name}",
    ]

    for pattern in patterns:
        if hasattr(convert_to_hw_layers, pattern):
            return {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": pattern,
            }

    return None


def _discover_kernels_auto() -> list[dict[str, Any]]:
    """Auto-discover FINN kernels by scanning fpgadataflow module.

    Discovers all HWCustomOp subclasses, excluding base classes and
    explicitly excluded components.

    Returns:
        List of kernel metadata dicts with auto-detected info
    """
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    import finn.custom_op.fpgadataflow as fpgadataflow

    kernels = []

    # Scan all modules in fpgadataflow for HWCustomOp subclasses
    for name in dir(fpgadataflow):
        try:
            obj = getattr(fpgadataflow, name)

            # Skip if not a class
            if not inspect.isclass(obj):
                continue

            # Skip if not a HWCustomOp subclass
            if not issubclass(obj, HWCustomOp):
                continue

            # Skip base class itself
            if obj is HWCustomOp:
                continue

            # Skip explicitly excluded kernels
            if obj.__name__ in EXCLUDED_KERNELS:
                continue

            # Extract metadata
            kernel_name = obj.__name__

            # Check for class-level metadata (future enhancement)
            is_infrastructure = (
                getattr(obj, '_brainsmith_infrastructure', False) or
                kernel_name in INFRASTRUCTURE_KERNELS
            )

            # Infer transform discovery
            infer_transform = _find_infer_transform(kernel_name)

            metadata = {
                "name": kernel_name,
                "module": obj.__module__,
                "class_name": obj.__name__,
            }

            # Add optional fields only if present
            if is_infrastructure:
                metadata["is_infrastructure"] = True

            if infer_transform is not None:
                metadata["infer_transform"] = infer_transform

            kernels.append(metadata)

        except (AttributeError, TypeError, ImportError):
            # Skip items that can't be inspected
            continue

    return kernels


def _infer_target_kernel(backend_name: str, language: str) -> str:
    """Infer target kernel from backend name.

    Args:
        backend_name: Backend class name (e.g., 'MVAU_hls')
        language: Language ('hls' or 'rtl')

    Returns:
        Qualified kernel name (e.g., 'finn:MVAU')
    """
    # Remove language suffix
    suffix = f"_{language}"
    if backend_name.endswith(suffix):
        kernel_name = backend_name[:-len(suffix)]
    else:
        kernel_name = backend_name

    return f"finn:{kernel_name}"


def _discover_backends_in_module(
    module: Any,
    base_class: type,
    language: str
) -> list[dict[str, Any]]:
    """Discover backends in a specific module.

    Args:
        module: Module to scan
        base_class: Base backend class (HLSBackend or RTLBackend)
        language: Language identifier ('hls' or 'rtl')

    Returns:
        List of backend metadata dicts
    """
    backends = []

    for name in dir(module):
        try:
            obj = getattr(module, name)

            # Skip if not a class
            if not inspect.isclass(obj):
                continue

            # Skip if not a backend subclass
            if not issubclass(obj, base_class):
                continue

            # Skip base class itself
            if obj is base_class:
                continue

            # Skip explicitly excluded backends
            if obj.__name__ in EXCLUDED_BACKENDS:
                continue

            # Extract metadata
            backend_name = obj.__name__

            # Infer target kernel from name (e.g., "MVAU_hls" -> "finn:MVAU")
            target_kernel = _infer_target_kernel(backend_name, language)

            # Check for class-level override (future enhancement)
            target_kernel = getattr(
                obj, '_brainsmith_target_kernel', target_kernel
            )

            metadata = {
                "name": backend_name,
                "module": obj.__module__,
                "class_name": obj.__name__,
                "target_kernel": target_kernel,
                "language": language,
            }

            backends.append(metadata)

        except (AttributeError, TypeError, ImportError):
            continue

    return backends


def _discover_backends_auto() -> list[dict[str, Any]]:
    """Auto-discover FINN backends by scanning hls/ and rtl/ directories.

    Discovers all HLSBackend and RTLBackend subclasses, inferring:
    - Target kernel from class name pattern
    - Language from directory/suffix

    Returns:
        List of backend metadata dicts
    """
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
    from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

    backends = []

    # Discover HLS backends
    try:
        import finn.custom_op.fpgadataflow.hls as hls_module
        backends.extend(_discover_backends_in_module(
            hls_module, HLSBackend, "hls"
        ))
    except ImportError:
        pass

    # Discover RTL backends
    try:
        import finn.custom_op.fpgadataflow.rtl as rtl_module
        backends.extend(_discover_backends_in_module(
            rtl_module, RTLBackend, "rtl"
        ))
    except ImportError:
        pass

    return backends


def _discover_steps_auto() -> list[dict[str, Any]]:
    """Auto-discover FINN build steps by scanning build_dataflow_steps module.

    Discovers all functions matching pattern step_{name}.

    Returns:
        List of step metadata dicts
    """
    from finn.builder import build_dataflow_steps

    steps = []

    for name in dir(build_dataflow_steps):
        if not name.startswith("step_"):
            continue

        obj = getattr(build_dataflow_steps, name)

        # Skip if not a function
        if not inspect.isfunction(obj):
            continue

        # Extract step name (remove 'step_' prefix)
        step_name = name[5:]  # len("step_") == 5

        metadata = {
            "name": step_name,
            "module": "finn.builder.build_dataflow_steps",
            "func_name": name,
        }

        steps.append(metadata)

    return steps


# ============================================================================
# ENTRY POINT
# ============================================================================

def register_all():
    """Return FINN components for Brainsmith registration.

    This is the entry point function called by Brainsmith's plugin discovery.
    FINN has no dependency on brainsmith - this just returns component data.

    Uses auto-discovery for improved maintainability.

    Returns:
        Dict with keys 'kernels', 'backends', 'steps', each containing
        lists of component metadata dicts.

    Example:
        >>> components = register_all()
        >>> 'steps' in components and 'kernels' in components
        True
        >>> all(isinstance(components[k], list) for k in components)
        True
    """
    return {
        "kernels": _discover_kernels_auto(),
        "backends": _discover_backends_auto(),
        "steps": _discover_steps_auto(),
    }
