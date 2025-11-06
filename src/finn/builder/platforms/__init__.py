############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""FINN Platform Abstraction.

Public API for platform definitions, build methods, and target specifications.

This package provides:
- Platform protocols and types
- Built-in platform definitions
- Platform-specific build handlers (Vivado, Vitis, RTL)
- Registration and lookup API

Example:
    >>> from finn.builder.platforms import get_platform, FPGATargetSpec
    >>> platform = get_platform("Pynq-Z1")
    >>> print(platform.part_number)
    'xc7z020clg400-1'
"""

# Core framework
from finn.builder.platforms.core import (
    # Protocols
    PlatformProtocol,
    BuildMethodProtocol,
    get_platform_attribute,

    # Core types
    Platform,
    SynthesisBackend,
    IntegrationFlow,
    FPGATargetSpec,

    # Validation
    AttributeMatchBuilder,
    validate_handler_kwargs,

    # Registry API
    register_platform,
    register_synthesis_backend,
    register_integration_flow,
    get_platform,
    get_synthesis_backend,
    get_integration_flow,
    get_compatible_synthesis_backends,
    get_compatible_integration_flows,

    # Schema introspection
    get_synthesis_backend_schema,
    get_integration_flow_schema,
)

# Built-in platforms (auto-registers on import)
from finn.builder.platforms import builtin

# Base FPGA platforms
from finn.builder.platforms.builtin import (
    # Zynq-7000 series
    XC7Z020_BASE,
    # Zynq UltraScale+ series
    XCZU3EG_BASE,
    XCZU7EV_BASE,
    XCZU9EG_BASE,
    XCZU28DR_BASE,
    XCZU48DR_BASE,
    XCK26_BASE,
    # UltraScale+ datacenter series
    XCU50_BASE,
    XCU200_BASE,
    XCU250_BASE,
    XCU280_BASE,
    XCU55C_BASE,
)

# Build handlers
from finn.builder.platforms.vivado_handlers import (
    vivado_hls_handler,
    zynq_ps_handler,
)
from finn.builder.platforms.vitis_handlers import (
    vitis_hls_handler,
    alveo_xrt_handler,
)

__all__ = [
    # Protocols
    "PlatformProtocol",
    "BuildMethodProtocol",
    "get_platform_attribute",

    # Core types
    "Platform",
    "SynthesisBackend",
    "IntegrationFlow",
    "FPGATargetSpec",

    # Validation
    "AttributeMatchBuilder",
    "validate_handler_kwargs",

    # Registry API
    "register_platform",
    "register_synthesis_backend",
    "register_integration_flow",
    "get_platform",
    "get_synthesis_backend",
    "get_integration_flow",
    "get_compatible_synthesis_backends",
    "get_compatible_integration_flows",

    # Schema introspection
    "get_synthesis_backend_schema",
    "get_integration_flow_schema",

    # Base FPGA platforms (Zynq-7000)
    "XC7Z020_BASE",
    # Base FPGA platforms (Zynq UltraScale+)
    "XCZU3EG_BASE",
    "XCZU7EV_BASE",
    "XCZU9EG_BASE",
    "XCZU28DR_BASE",
    "XCZU48DR_BASE",
    "XCK26_BASE",
    # Base FPGA platforms (UltraScale+ datacenter)
    "XCU50_BASE",
    "XCU200_BASE",
    "XCU250_BASE",
    "XCU280_BASE",
    "XCU55C_BASE",

    # Handlers
    "vivado_hls_handler",
    "zynq_ps_handler",
    "vitis_hls_handler",
    "alveo_xrt_handler",
]
