############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Platform abstraction framework for FINN.

This module provides the core platform abstraction framework including:
- Protocols (PlatformProtocol, BuildMethodProtocol)
- Core types (Platform, SynthesisBackend, IntegrationFlow, FPGATargetSpec)
- Registration API (register_*, get_*)
- Validation utilities (AttributeMatchBuilder)
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)
import warnings


# =============================================================================
# Protocols (Minimal Contracts)
# =============================================================================


@runtime_checkable
class PlatformProtocol(Protocol):
    """Contract for platform objects used by FINN.

    This protocol defines the minimal interface that all platforms must implement.
    It enables duck typing while providing clear expectations and type safety.

    Required Attributes:
        part_number: FPGA part number (e.g., "xc7z020clg400-1")
        vendor: Vendor name (e.g., "amd")
        has_ps: Whether platform has Processing System (ARM cores, etc.)

    Required Methods:
        get_attribute: Access platform-specific attributes with fallback
        total_resources: Property returning aggregated FPGA resources

    Platform-Specific Attributes (accessed via get_attribute):
        family: FPGA family (e.g., "zynq-7000", "ultrascale+")
        board_name: Human-readable board name
        axi_port_width: AXI port width in bits (for Zynq platforms)
        vitis_platform: Vitis platform string (for Alveo platforms)
        resources: per-SLR resource breakdown
        ... (custom attributes as needed)
    """

    part_number: str
    vendor: str
    has_ps: bool

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get platform-specific attribute with fallback.

        Args:
            key: Attribute name
            default: Value to return if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        ...

    @property
    def total_resources(self) -> Dict[str, int]:
        """Get total FPGA resources aggregated across all dies/SLRs.

        This is the only resource-related method required by the protocol.
        Platforms answer "can this design fit?" without mandating how
        resources are internally structured (per-SLR, per-tile, etc.).

        Returns:
            Dict mapping resource types to total counts
            Example: {"lut": 53200, "ff": 106400, "bram_18k": 280, "dsp": 220}
        """
        ...


@runtime_checkable
class BuildMethodProtocol(Protocol):
    """Minimal contract for build method objects.

    This protocol enables duck typing for both core generation and
    integration flow methods.

    Required attributes:
        name: Unique identifier for the method (e.g., "vivado_hls")

    Required methods:
        supports: Check if method is compatible with a platform
        execute: Execute the build method
    """

    name: str

    def supports(self, platform: PlatformProtocol) -> bool:
        """Check if this method supports the given platform."""
        ...

    def execute(self, model: Any, **kwargs) -> Any:
        """Execute the build method on the model."""
        ...


# =============================================================================
# Protocol Helper Functions
# =============================================================================


def get_platform_attribute(
    platform: PlatformProtocol,
    key: str,
    default: Any = None
) -> Any:
    """Get attribute from platform using multiple access patterns.

    This helper enables accessing platform attributes whether the platform
    implements a get_attribute() method or uses direct attributes.

    Args:
        platform: Platform object
        key: Attribute name to retrieve
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    # Try get_attribute method first (if platform implements it)
    if hasattr(platform, 'get_attribute') and callable(platform.get_attribute):
        return platform.get_attribute(key, default)
    # Fall back to direct attribute access
    return getattr(platform, key, default)


# =============================================================================
# FINN's Built-In Platform Implementation
# =============================================================================


@dataclass(frozen=True)
class Platform:
    """FINN's built-in platform implementation for AMD FPGAs.

    This is a typed, validated dataclass for AMD platforms. Custom
    platforms don't need to use this - they only need to match the
    PlatformProtocol interface.

    Attributes:
        part_number: FPGA part number (e.g., "xc7z020clg400-1")
        vendor: Vendor name (default: "amd")
        family: FPGA family (e.g., "zynq-7000", "ultrascale+")
        has_ps: Whether platform has Processing System (ARM cores)
        resources: List of resource dicts (per die/SLR), vendor-neutral
        attributes: Additional platform-specific attributes
        base_platform: Optional parent platform (for board-specific platforms derived from generic FPGAs)

    Resources are defined as dicts mapping resource type to count:
        Zynq-7000: {"lut": 53200, "ff": 106400, "bram_18k": 280, "dsp": 220}
        UltraScale+: {"lut": 1500000, "ff": 3000000, "bram_18k": 2304, "uram": 1280, "dsp": 11520}
        Versal: {"lut": 900000, "ff": 1800000, "bram_18k": 960, "uram": 320, "dsp": 1968}

    Example:
        # Base FPGA platform
        xc7z020_base = Platform(
            part_number="xc7z020clg400-1",
            vendor="amd",
            family="zynq-7000",
            has_ps=True,
            resources=[{
                "lut": 53200,
                "ff": 106400,
                "bram_18k": 280,
                "dsp": 220
            }]
        )

        # Board platform derived from base FPGA
        my_board = Platform(
            part_number="xc7z020clg400-1",
            base_platform=xc7z020_base,  # Inherit FPGA specs
            attributes={
                "board_name": "MyBoard",
                "constraints_file": "my_board.xdc",
                "dram_size_gb": 2
            }
        )
        # my_board inherits resources from xc7z020_base
    """

    part_number: Optional[str] = None
    vendor: Optional[str] = None
    family: Optional[str] = None
    has_ps: Optional[bool] = None
    resources: List[Dict[str, int]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    base_platform: Optional["Platform"] = None

    def __post_init__(self):
        """Copy fields from base_platform if not explicitly set.

        This enables true inheritance - derived platforms automatically
        inherit all fields from their base_platform, with the ability to
        override any field by specifying it explicitly.

        Example:
            >>> base = Platform(part_number="xc7z020clg400-1", vendor="amd", family="zynq-7000", has_ps=True)
            >>> # Clean - inherits everything from base
            >>> board = Platform(base_platform=base, attributes={"board_name": "Pynq-Z1"})
            >>> board.part_number
            'xc7z020clg400-1'
            >>> board.vendor
            'amd'
            >>> # Override if needed
            >>> custom = Platform(base_platform=base, vendor="custom")
            >>> custom.vendor
            'custom'
        """
        if self.base_platform:
            # Inherit each field from base if None (None = inherit)
            if self.part_number is None:
                object.__setattr__(self, 'part_number', self.base_platform.part_number)
            if self.vendor is None:
                object.__setattr__(self, 'vendor', self.base_platform.vendor)
            if self.family is None:
                object.__setattr__(self, 'family', self.base_platform.family)
            if self.has_ps is None:
                object.__setattr__(self, 'has_ps', self.base_platform.has_ps)

            # Inherit resources if not specified (empty list = not specified)
            if not self.resources and self.base_platform.resources:
                object.__setattr__(self, 'resources', self.base_platform.resources)
        else:
            # Standalone platform - validate and apply defaults
            if self.part_number is None:
                raise ValueError(
                    "part_number is required for platforms without base_platform"
                )
            if self.vendor is None:
                object.__setattr__(self, 'vendor', 'amd')
            if self.family is None:
                object.__setattr__(self, 'family', '')
            if self.has_ps is None:
                object.__setattr__(self, 'has_ps', False)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get platform-specific attribute with inheritance from base_platform.

        Inheritance chain:
        1. Check this platform's attributes dict first (overrides)
        2. Check this platform's dataclass fields
        3. Fall back to base_platform (recursive)
        4. Return default

        Args:
            key: Attribute name
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default

        Example:
            >>> platform.get_attribute("clock_domains", 1)
            1
            >>> platform.get_attribute("vendor")
            'amd'
            >>> # With base_platform
            >>> base_fpga = Platform(part_number="xc7z020clg400-1", vendor="amd", family="zynq-7000", has_ps=True)
            >>> board = Platform(base_platform=base_fpga, attributes={"board_name": "Pynq-Z1"})
            >>> board.get_attribute("board_name")
            'Pynq-Z1'
            >>> board.get_attribute("vendor")  # Inherited from base
            'amd'
            >>> board.get_attribute("family")  # Inherited from base
            'zynq-7000'
        """
        # First check attributes dict (overrides everything)
        if key in self.attributes:
            return self.attributes[key]

        # Then check dataclass fields
        if hasattr(self, key):
            value = getattr(self, key)
            return value

        # Fall back to base platform (recursive)
        if self.base_platform:
            return self.base_platform.get_attribute(key, default)

        return default

    @property
    def total_resources(self) -> Dict[str, int]:
        """Aggregate resources across all dies/SLRs, with base_platform fallback.

        If this platform has no resources defined, inherits from base_platform.

        Returns:
            Dict mapping resource types to total counts

        Example:
            >>> platform.total_resources
            {"lut": 53200, "ff": 106400, "bram_18k": 280, "dsp": 220}
        """
        # Use resources from base if not overridden
        resources_to_use = self.resources or (
            self.base_platform.resources if self.base_platform else []
        )

        if not resources_to_use:
            return {}

        total = Counter()
        for die_resources in resources_to_use:
            total.update(die_resources)

        return dict(total)

    @property
    def is_derived_platform(self) -> bool:
        """Check if this platform is derived from a base FPGA platform.

        Returns:
            True if this platform has a base_platform

        Example:
            >>> base_fpga = Platform(part_number="xc7z020clg400-1", vendor="amd")
            >>> board = Platform(part_number="xc7z020clg400-1", base_platform=base_fpga)
            >>> base_fpga.is_derived_platform
            False
            >>> board.is_derived_platform
            True
        """
        return self.base_platform is not None



# =============================================================================
# Build Method Wrappers
# =============================================================================


@dataclass(frozen=True)
class SynthesisBackend:
    """Wrapper for synthesis backend methods with compatibility checking.

    Synthesis backends handle IP synthesis from the dataflow model using
    FPGA vendor toolchains (Vivado HLS, Vitis HLS, etc.). This wrapper
    allows registration of handler functions without requiring inheritance.

    Attributes:
        name: Unique identifier (e.g., "vivado_hls", "vitis_hls")
        handler: Function that executes synthesis
        compatibility_check: Function for platform compatibility checking
        tested_platforms: List of tested part numbers/families
        required_params: Parameters this method requires
        optional_params: Optional parameters with defaults

    Compatibility Checking:
        The supports() method checks if a platform is compatible.
        Platforms passing compatibility_check return True.
        If the platform is not in tested_platforms, a warning is emitted.

    Example:
        vivado_hls = SynthesisBackend(
            name="vivado_hls",
            handler=vivado_hls_handler,
            compatibility_check=lambda p: p.vendor in ["xilinx", "amd"],
            tested_platforms=["zynq-7000", "ultrascale", "ultrascale+"],
            required_params=["model", "clk_period_ns"],
            optional_params={"mem_mode": "internal"}
        )

        # Check compatibility (returns bool, warns if untested)
        if vivado_hls.supports(platform):
            print("Compatible!")
    """

    name: str
    handler: Callable[[Any], Any]
    compatibility_check: Optional[Callable[[PlatformProtocol], bool]] = None

    # Compatibility checking
    tested_platforms: List[str] = field(default_factory=list)

    # Flow-driven parameters
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.compatibility_check is None:
            object.__setattr__(self, 'compatibility_check', lambda p: True)

    def supports(self, platform: PlatformProtocol) -> bool:
        """Check if this method supports the given platform.

        Emits a warning if the platform passes compatibility check but
        has not been explicitly tested.

        Args:
            platform: Platform to check compatibility against

        Returns:
            True if this method can build for the platform

        Example:
            >>> vivado_hls = get_core_generation("vivado_hls")
            >>> pynq = get_platform("Pynq-Z1")
            >>> if vivado_hls.supports(pynq):
            ...     print("Compatible")
        """
        return _check_platform_support(
            self.name,
            platform,
            self.compatibility_check,
            self.tested_platforms
        )

    def execute(self, model: Any, **kwargs) -> Any:
        """Execute synthesis backend on the model."""
        return self.handler(model, **kwargs)


@dataclass(frozen=True)
class IntegrationFlow:
    """Wrapper for integration flow methods with compatibility checking.

    Integration flow connects the accelerator to the rest of the system
    (memory, processing system, host interface, etc.).

    Attributes:
        name: Unique identifier (e.g., "zynq_ps", "alveo_xrt")
        handler: Function that executes integration
        compatibility_check: Function for platform compatibility checking
        tested_platforms: List of confirmed compatible platforms (part numbers, families, board names)
        required_params: List of required config parameter names
        optional_params: Dict of optional parameter names to default values

    Compatibility Checking:
        The supports() method checks if a platform is compatible.
        Platforms passing compatibility_check return True.
        If the platform is not in tested_platforms, a warning is emitted.

    Example:
        def zynq_ps_handler(model, **kwargs):
            # Zynq PS integration implementation
            pass

        zynq_ps = IntegrationFlow(
            name="zynq_ps",
            handler=zynq_ps_handler,
            compatibility_check=lambda p: p.vendor == "amd" and p.has_ps,
            tested_platforms=["Pynq-Z1", "ZCU104", "zynq-7000"],
            required_params=["board", "clk_period_ns", "output_dir"],
            optional_params={"enable_hw_debug": False}
        )
    """

    name: str
    handler: Callable[[Any], Any]
    compatibility_check: Optional[Callable[[PlatformProtocol], bool]] = None

    # Compatibility checking
    tested_platforms: List[str] = field(default_factory=list)

    # Flow-driven parameters
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.compatibility_check is None:
            object.__setattr__(self, 'compatibility_check', lambda p: True)

    def supports(self, platform: PlatformProtocol) -> bool:
        """Check if this flow supports the platform.

        Emits a warning if the platform passes compatibility check but
        has not been explicitly tested.

        Args:
            platform: Platform to check compatibility against

        Returns:
            True if this flow can integrate on the platform

        Example:
            >>> zynq_ps = get_integration_flow("zynq_ps")
            >>> pynq = get_platform("Pynq-Z1")
            >>> if zynq_ps.supports(pynq):
            ...     print("Compatible")
        """
        return _check_platform_support(
            self.name,
            platform,
            self.compatibility_check,
            self.tested_platforms
        )

    def execute(self, model: Any, **kwargs) -> Any:
        """Execute integration flow on the model."""
        return self.handler(model, **kwargs)


# =============================================================================
# FPGA Target Specification
# =============================================================================


@dataclass(frozen=True)
class FPGATargetSpec:
    """Immutable FPGA target specification: platform + build methods.

    This class composes a platform with specific build methods to create
    a complete, validated target specification for the build system.

    Scope Hierarchy:
        DataflowBuildConfig (full build: verify, optimize, output formats)
        └── FPGATargetSpec (platform-specific: where + how to synthesize/deploy)
            ├── Platform (hardware: part number, resources, features)
            ├── SynthesisBackend (how to synthesize IP: HLS, RTL, custom)
            └── IntegrationFlow (how to integrate: Zynq PS, Alveo XRT, custom)

    Attributes:
        platform: Target FPGA platform
        synthesis_backend: Method for IP synthesis
        integration_flow: Method for system integration

    The spec is immutable (frozen=True) to prevent accidental modification
    during the build process.

    Example:
        target_spec = FPGATargetSpec(
            platform=get_platform("Pynq-Z1"),
            synthesis_backend=get_synthesis_backend("vivado_hls"),
            integration_flow=get_integration_flow("zynq_ps")
        )

    Multi-stage builds can compose multiple specs:
        # Stage 1: HLS synthesis
        spec1 = FPGATargetSpec(platform, vivado_hls, standalone)
        model = spec1.synthesis_backend.execute(model)

        # Stage 2: Integration
        spec2 = FPGATargetSpec(platform, rtl_only, zynq_ps)
        bitfile = spec2.integration_flow.execute(model)
    """

    platform: PlatformProtocol
    synthesis_backend: BuildMethodProtocol
    integration_flow: BuildMethodProtocol

    def __post_init__(self):
        """Validate that build methods are compatible with platform."""
        if not self.synthesis_backend.supports(self.platform):
            raise ValueError(
                f"Synthesis backend '{self.synthesis_backend.name}' "
                f"does not support platform '{self.platform.part_number}' "
                f"(vendor: {self.platform.vendor})"
            )

        if not self.integration_flow.supports(self.platform):
            raise ValueError(
                f"Integration flow '{self.integration_flow.name}' "
                f"does not support platform '{self.platform.part_number}' "
                f"(vendor: {self.platform.vendor})"
            )


# =============================================================================
# Attribute-Based Compatibility Helper
# =============================================================================


class AttributeMatchBuilder:
    """Declarative helper for building attribute-based compatibility checks.

    This builder pattern simplifies creating compatibility functions
    that check multiple platform attributes.

    Example:
        # Simple existence check
        zynq_check = (
            AttributeMatchBuilder()
            .require("has_ps", True)
            .require_any("vendor", ["xilinx", "amd"])
            .build()
        )

        # Multiple values
        family_check = (
            AttributeMatchBuilder()
            .require_any("family", ["zynq-7000", "zynq-ultrascale-plus"])
            .build()
        )

        # Custom logic
        complex_check = (
            AttributeMatchBuilder()
            .require_any("vendor", ["amd", "xilinx"])
            .custom(lambda p: get_platform_attribute(p, "has_ps", False))
            .build()
        )
    """

    def __init__(self):
        self.checks: List[Callable[[PlatformProtocol], bool]] = []

    def require(self, attribute: str, value: Any = None) -> "AttributeMatchBuilder":
        """Require platform to have attribute, optionally matching a value.

        Args:
            attribute: Attribute name
            value: Expected value (None = just check existence and truthy)

        Returns:
            Self for chaining

        Example:
            # Check attribute exists and is truthy
            builder.require("has_ps")

            # Check attribute matches specific value
            builder.require("vendor", "amd")
        """
        # Capture in closure using a function to avoid lambda late-binding issues
        def make_check(attr, val):
            if val is None:
                return lambda p: bool(get_platform_attribute(p, attr, False))
            else:
                return lambda p: get_platform_attribute(p, attr) == val

        self.checks.append(make_check(attribute, value))
        return self

    def require_any(self, attribute: str, values: List[Any]) -> "AttributeMatchBuilder":
        """Require attribute to match any value in list.

        Args:
            attribute: Attribute name
            values: List of acceptable values

        Returns:
            Self for chaining

        Example:
            # Check vendor is AMD or Xilinx
            builder.require_any("vendor", ["amd", "xilinx"])
        """
        def make_check(attr, vals):
            if attr == "vendor":
                # For vendor, do case-insensitive comparison
                lower_vals = [v.lower() for v in vals]
                return lambda p: (get_platform_attribute(p, attr, "") or "").lower() in lower_vals
            else:
                return lambda p: get_platform_attribute(p, attr) in vals

        self.checks.append(make_check(attribute, values))
        return self

    def custom(self, check: Callable[[PlatformProtocol], bool]) -> "AttributeMatchBuilder":
        """Add custom compatibility check function.

        Args:
            check: Function that takes platform and returns bool

        Returns:
            Self for chaining

        Example:
            builder.custom(lambda p: p.family in ["ultrascale", "ultrascale+"])
        """
        self.checks.append(check)
        return self

    def build(self) -> Callable[[PlatformProtocol], bool]:
        """Build the final compatibility check function.

        Returns:
            Function that returns True if all checks pass
        """
        def compatibility_check(platform: PlatformProtocol) -> bool:
            return all(check(platform) for check in self.checks)

        return compatibility_check


# =============================================================================
# Build Method Support Checking
# =============================================================================


def _check_platform_support(
    method_name: str,
    platform: PlatformProtocol,
    compatibility_check: Callable[[PlatformProtocol], bool],
    tested_platforms: List[str]
) -> bool:
    """Shared logic for checking platform support with testing warnings.

    This helper is used by both DataflowCoreGeneration and IntegrationFlow
    to avoid code duplication.

    Args:
        method_name: Name of the build method (for warning messages)
        platform: Platform to check compatibility against
        compatibility_check: Function that returns True if platform is compatible
        tested_platforms: List of tested platform identifiers (part numbers, families, board names)

    Returns:
        True if platform is supported, False otherwise

    Side Effects:
        Emits UserWarning if platform is supported but not explicitly tested
    """
    # Check compatibility first
    if not compatibility_check(platform):
        return False

    # Check if platform is explicitly tested
    # Try multiple platform identifiers: part_number, family, board_name
    identifiers = [
        platform.part_number,
        getattr(platform, 'family', ''),
        get_platform_attribute(platform, 'board_name', '')
    ]
    is_tested = any(id in tested_platforms for id in identifiers if id)

    # Warn if supported but not tested
    if not is_tested and tested_platforms:
        tested_sample = tested_platforms[:3]
        tested_suffix = "..." if len(tested_platforms) > 3 else ""
        warnings.warn(
            f"{method_name}: {platform.part_number} not tested. "
            f"Tested: {', '.join(tested_sample)}{tested_suffix}",
            UserWarning,
            stacklevel=3  # Adjust for helper function
        )

    return True


# =============================================================================
# Handler Validation Utilities
# =============================================================================


def validate_handler_kwargs(
    handler_name: str,
    kwargs: Dict[str, Any],
    required_params: List[str]
) -> None:
    """Validate that required kwargs are present for a handler.

    This is a shared validation utility used by all build method handlers
    to ensure required parameters are provided.

    Args:
        handler_name: Name of the handler (for error messages)
        kwargs: Dictionary of keyword arguments passed to handler
        required_params: List of required parameter names

    Raises:
        KeyError: If any required parameters are missing

    Example:
        >>> validate_handler_kwargs(
        ...     "vivado_hls_handler",
        ...     {"fpga_part": "xc7z020clg400-1"},
        ...     ["fpga_part", "clk_period_ns"]
        ... )
        KeyError: vivado_hls_handler requires 'clk_period_ns' in kwargs. Provided: ['fpga_part']
    """
    for param in required_params:
        if param not in kwargs:
            raise KeyError(
                f"{handler_name} requires '{param}' in kwargs. "
                f"Provided: {list(kwargs.keys())}"
            )


# =============================================================================
# Registration API
# =============================================================================

# Module-level registries
_platforms: Dict[str, PlatformProtocol] = {}
_synthesis_backends: Dict[str, BuildMethodProtocol] = {}
_integration_flows: Dict[str, BuildMethodProtocol] = {}


def register_platform(name: str, platform: PlatformProtocol) -> None:
    """Register a platform for use by name.

    This allows extension packages to register custom platforms that
    can be referenced by name in build configurations.

    Args:
        name: Platform identifier (e.g., "Pynq-Z1", "DE10-Nano")
        platform: Platform object matching PlatformProtocol

    Raises:
        TypeError: If platform doesn't match PlatformProtocol

    Example:
        # FINN's built-in registration
        register_platform("Pynq-Z1", Platform(
            part_number="xc7z020clg400-1",
            vendor="amd",
            has_ps=True,
            resources=[{"lut": 53200, "ff": 106400, "bram_18k": 280, "dsp": 220}]
        ))

        # Custom board platform registration
        register_platform("MyZynqBoard", Platform(
            part_number="xc7z020clg400-1",
            vendor="amd",
            has_ps=True,
            attributes={"board_name": "MyZynqBoard", "custom_constraints": "/path/to/constraints.xdc"}
        ))
    """
    # Check if platform matches protocol
    if not isinstance(platform, PlatformProtocol):
        # Provide helpful error message showing what's missing
        missing_attrs = []
        for attr in ['part_number', 'vendor', 'has_ps']:
            if not hasattr(platform, attr):
                missing_attrs.append(attr)

        missing_methods = []
        if not hasattr(platform, 'get_attribute') or not callable(getattr(platform, 'get_attribute', None)):
            missing_methods.append('get_attribute')
        if not hasattr(platform, 'total_resources'):
            missing_methods.append('total_resources')

        error_parts = []
        if missing_attrs:
            error_parts.append(f"missing attributes: {', '.join(missing_attrs)}")
        if missing_methods:
            error_parts.append(f"missing methods: {', '.join(missing_methods)}")

        raise TypeError(
            f"Platform must match PlatformProtocol. "
            f"Found {type(platform).__name__} with {'; '.join(error_parts)}"
        )

    if name in _platforms:
        warnings.warn(
            f"Overwriting existing platform '{name}' "
            f"(was {_platforms[name].part_number}, now {platform.part_number})",
            UserWarning,
            stacklevel=2
        )

    _platforms[name] = platform


def register_synthesis_backend(method: BuildMethodProtocol) -> None:
    """Register a synthesis backend.

    Args:
        method: Synthesis backend matching BuildMethodProtocol

    Raises:
        TypeError: If method doesn't match BuildMethodProtocol

    Example:
        # FINN's built-in registration
        register_synthesis_backend(SynthesisBackend(
            name="vivado_hls",
            handler=vivado_hls_handler,
            compatibility_check=lambda p: p.vendor in ["xilinx", "amd"]
        ))

        # Experimental synthesis flow registration
        register_synthesis_backend(SynthesisBackend(
            name="experimental_hls",
            handler=experimental_hls_handler,
            compatibility_check=lambda p: p.vendor == "amd"
        ))
    """
    if not isinstance(method, BuildMethodProtocol):
        raise TypeError(
            f"Synthesis backend must match BuildMethodProtocol "
            f"(have 'name', 'supports', and 'execute' attributes)"
        )

    if method.name in _synthesis_backends:
        warnings.warn(
            f"Overwriting existing synthesis backend '{method.name}'",
            UserWarning,
            stacklevel=2
        )

    _synthesis_backends[method.name] = method


def register_integration_flow(flow: BuildMethodProtocol) -> None:
    """Register an integration flow method.

    Args:
        flow: Integration flow matching BuildMethodProtocol

    Raises:
        TypeError: If flow doesn't match BuildMethodProtocol

    Example:
        # FINN's built-in registration
        register_integration_flow(IntegrationFlow(
            name="zynq_ps",
            handler=zynq_ps_handler,
            compatibility_check=lambda p: p.get_attribute("has_ps", False)
        ))

        # Versal AI Engine integration registration
        register_integration_flow(IntegrationFlow(
            name="versal_aie",
            handler=versal_aie_handler,
            compatibility_check=lambda p: p.vendor == "amd" and p.family == "versal"
        ))
    """
    if not isinstance(flow, BuildMethodProtocol):
        raise TypeError(
            f"Integration flow must match BuildMethodProtocol "
            f"(have 'name', 'supports', and 'execute' attributes)"
        )

    if flow.name in _integration_flows:
        warnings.warn(
            f"Overwriting existing integration flow '{flow.name}'",
            UserWarning,
            stacklevel=2
        )

    _integration_flows[flow.name] = flow


def _format_platform_entry(name: str, platform: PlatformProtocol) -> str:
    """Format a single platform for error message listing.

    Args:
        name: Platform registry name
        platform: Platform object to format

    Returns:
        Formatted string for display in error messages
    """
    # Check if derived platform (show part number)
    if hasattr(platform, 'base_platform') and platform.base_platform:
        return f"  {name} (part: {platform.part_number})"

    # Base platform (just show name)
    return f"  {name}"


def get_platform(name: str) -> PlatformProtocol:
    """Get a registered platform by name.

    Args:
        name: Platform identifier (board name or part number)

    Returns:
        Platform object

    Raises:
        KeyError: If platform not found

    Example:
        >>> platform = get_platform("Pynq-Z1")  # Board name
        >>> platform.part_number
        'xc7z020clg400-1'
        >>> platform = get_platform("xc7z020clg400-1")  # Part number (generic FPGA)
    """
    if name not in _platforms:
        # Build helpful error message showing platform hierarchy
        platform_list = [
            _format_platform_entry(key, platform)
            for key, platform in sorted(_platforms.items())
        ]
        raise KeyError(
            f"Platform '{name}' not found.\n"
            f"Available platforms:\n" + "\n".join(platform_list)
        )
    return _platforms[name]


def get_synthesis_backend(name: str) -> BuildMethodProtocol:
    """Get a registered synthesis backend by name.

    Args:
        name: Synthesis backend name

    Returns:
        Synthesis backend

    Raises:
        KeyError: If backend not found
    """
    if name not in _synthesis_backends:
        raise KeyError(
            f"Synthesis backend '{name}' not found. "
            f"Available backends: {list(_synthesis_backends.keys())}"
        )
    return _synthesis_backends[name]


def get_integration_flow(name: str) -> BuildMethodProtocol:
    """Get a registered integration flow by name.

    Args:
        name: Integration flow name

    Returns:
        Integration flow

    Raises:
        KeyError: If flow not found
    """
    if name not in _integration_flows:
        raise KeyError(
            f"Integration flow '{name}' not found. "
            f"Available flows: {list(_integration_flows.keys())}"
        )
    return _integration_flows[name]


def get_compatible_synthesis_backends(
    platform: PlatformProtocol,
) -> List[BuildMethodProtocol]:
    """Find all synthesis backends compatible with a platform.

    Args:
        platform: Platform to check compatibility against

    Returns:
        List of compatible synthesis backends

    Example:
        >>> platform = get_platform("Pynq-Z1")
        >>> backends = get_compatible_synthesis_backends(platform)
        >>> [b.name for b in backends]
        ['vivado_hls', 'vitis_hls']
    """
    return [
        method
        for method in _synthesis_backends.values()
        if method.supports(platform)
    ]


def get_compatible_integration_flows(
    platform: PlatformProtocol,
) -> List[BuildMethodProtocol]:
    """Find all integration flows compatible with a platform.

    Args:
        platform: Platform to check compatibility against

    Returns:
        List of compatible integration flows

    Example:
        >>> platform = get_platform("Pynq-Z1")
        >>> flows = get_compatible_integration_flows(platform)
        >>> [f.name for f in flows]
        ['zynq_ps']
    """
    return [
        flow
        for flow in _integration_flows.values()
        if flow.supports(platform)
    ]


# =============================================================================
# Schema Introspection API (Priority 2b)
# =============================================================================


def get_synthesis_backend_schema(name: str) -> Dict[str, Any]:
    """Get parameter schema for a synthesis backend.

    Returns the required and optional parameters for the specified
    synthesis backend, useful for validation and documentation.

    Args:
        name: Name of the synthesis backend (e.g., "vivado_hls")

    Returns:
        Dictionary with keys:
            - "required": List of required parameter names
            - "optional": Dict mapping optional parameter names to default values

    Raises:
        KeyError: If synthesis backend not found

    Example:
        >>> schema = get_synthesis_backend_schema("vivado_hls")
        >>> schema["required"]
        ['fpga_part', 'clk_period_ns']
        >>> schema["optional"]
        {}
    """
    method = get_synthesis_backend(name)
    return {
        "required": method.required_params,
        "optional": method.optional_params
    }


def get_integration_flow_schema(name: str) -> Dict[str, Any]:
    """Get parameter schema for an integration flow.

    Returns the required and optional parameters for the specified
    integration flow, useful for validation and documentation.

    Args:
        name: Name of the integration flow (e.g., "zynq_ps")

    Returns:
        Dictionary with keys:
            - "required": List of required parameter names
            - "optional": Dict mapping optional parameter names to default values

    Raises:
        KeyError: If integration flow not found

    Example:
        >>> schema = get_integration_flow_schema("zynq_ps")
        >>> schema["required"]
        ['board', 'clk_period_ns', 'output_dir']
        >>> schema["optional"]
        {'enable_hw_debug': False, 'partition_model_dir': None}
    """
    flow = get_integration_flow(name)
    return {
        "required": flow.required_params,
        "optional": flow.optional_params
    }


