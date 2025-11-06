# Copyright (c) 2025, AMD
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Unit tests for platform abstraction core APIs.

Tests protocol compliance, registration APIs, and compatibility checking.
Integration tests are in test_platforms_extensibility.py (external vendor support)
and tests/builder/test_platform_pipeline_integration.py (build pipeline integration).
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict

from finn.builder.platforms import (
    PlatformProtocol,
    BuildMethodProtocol,
    Platform,
    SynthesisBackend,
    IntegrationFlow,
    FPGATargetSpec,
    AttributeMatchBuilder,
    register_platform,
    register_synthesis_backend,
    register_integration_flow,
    get_platform,
    get_synthesis_backend,
    get_integration_flow,
    get_compatible_synthesis_backends,
    get_compatible_integration_flows,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def amd_platform():
    """AMD Pynq-Z1 platform for testing."""
    return Platform(
        part_number="xc7z020clg400-1",
        vendor="amd",
        family="zynq-7000",
        has_ps=True,
        resources=[{
            "lut": 53200,
            "ff": 106400,
            "bram_18k": 280,
            "dsp": 220
        }],
        attributes={"clock_domains": 1}
    )


@pytest.fixture
def intel_platform():
    """Custom Intel platform for testing external extensibility."""
    @dataclass(frozen=True)
    class IntelPlatform:
        part_number: str = "10AS066N3F40E2SG"
        vendor: str = "intel"
        family: str = "arria-10"
        has_ps: bool = False  # Intel has HPS (Hard Processor System), not PS
        attributes: Dict[str, Any] = field(default_factory=dict)

        def get_attribute(self, key: str, default=None):
            return self.attributes.get(key, default)

        @property
        def total_resources(self) -> Dict[str, int]:
            """Intel Arria 10 resources (example values)."""
            return {
                "alm": 251680,
                "m20k": 2713,
                "dsp_blocks": 1518
            }

    return IntelPlatform(attributes={"has_hps": True})


@pytest.fixture
def vivado_hls():
    """Vivado HLS core generation method."""
    return SynthesisBackend(
        name="vivado_hls",
        handler=lambda model, **kwargs: f"vivado_hls({model})",
        compatibility_check=lambda p: p.vendor.lower() in ["xilinx", "amd"]
    )


@pytest.fixture
def quartus_hls():
    """Custom Quartus HLS for testing external methods."""
    return SynthesisBackend(
        name="quartus_hls",
        handler=lambda model, **kwargs: f"quartus_hls({model})",
        compatibility_check=lambda p: p.vendor == "intel"
    )


@pytest.fixture
def zynq_ps():
    """Zynq PS integration flow."""
    return IntegrationFlow(
        name="zynq_ps",
        handler=lambda model, **kwargs: f"zynq_ps({model})",
        compatibility_check=lambda p: getattr(p, "has_ps", False)
    )


@pytest.fixture
def nios_ii():
    """Custom Nios II integration for testing external flows."""
    return IntegrationFlow(
        name="nios_ii",
        handler=lambda model, **kwargs: f"nios_ii({model})",
        compatibility_check=lambda p: p.get_attribute("has_hps", False)
    )


@pytest.fixture
def isolated_registry():
    """Provide completely empty registry (no built-ins).

    Use when testing external vendor registration or when you need
    to register platforms without collision warnings.
    """
    import finn.builder.platforms.core as plat_core
    import finn.builder.platforms.builtin as builtin_module

    # Save current state
    saved_platforms = dict(plat_core._platforms)
    saved_backends = dict(plat_core._synthesis_backends)
    saved_flows = dict(plat_core._integration_flows)

    # Clear all registries
    plat_core._platforms.clear()
    plat_core._synthesis_backends.clear()
    plat_core._integration_flows.clear()

    yield

    # Restore original state
    plat_core._platforms.clear()
    plat_core._platforms.update(saved_platforms)
    plat_core._synthesis_backends.clear()
    plat_core._synthesis_backends.update(saved_backends)
    plat_core._integration_flows.clear()
    plat_core._integration_flows.update(saved_flows)


@pytest.fixture
def builtin_registry():
    """Provide registry with FINN built-ins only (default state).

    Use when testing built-in platform behavior or when you need
    to access existing built-in platforms.
    """
    import finn.builder.platforms.core as plat_core
    import finn.builder.platforms.builtin as builtin_module

    # Clear and re-register built-ins
    plat_core._platforms.clear()
    plat_core._synthesis_backends.clear()
    plat_core._integration_flows.clear()

    builtin_module._register_builtin_platforms()
    builtin_module._register_builtin_synthesis_backends()
    builtin_module._register_builtin_integration_flows()

    yield

    # Leave as-is (subsequent tests will manage their own state)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_platform_protocol_compliance(amd_platform, intel_platform):
    """Test that both AMD and custom platforms match PlatformProtocol."""
    assert isinstance(amd_platform, PlatformProtocol)
    assert isinstance(intel_platform, PlatformProtocol)

    # Check required attributes
    assert hasattr(amd_platform, "part_number")
    assert hasattr(amd_platform, "vendor")
    assert hasattr(intel_platform, "part_number")
    assert hasattr(intel_platform, "vendor")


def test_build_method_protocol_compliance(vivado_hls, quartus_hls, zynq_ps, nios_ii):
    """Test that build methods match BuildMethodProtocol."""
    methods = [vivado_hls, quartus_hls, zynq_ps, nios_ii]

    for method in methods:
        assert isinstance(method, BuildMethodProtocol)
        assert hasattr(method, "name")
        assert callable(method.supports)
        assert callable(method.execute)


# =============================================================================
# Platform Tests
# =============================================================================


def test_platform_creation(amd_platform):
    """Test Platform dataclass creation and attributes."""
    assert amd_platform.part_number == "xc7z020clg400-1"
    assert amd_platform.vendor == "amd"
    assert amd_platform.family == "zynq-7000"
    assert amd_platform.has_ps is True


def test_platform_resources_vendor_neutral():
    """Test that resources can use vendor-specific terminology."""
    # AMD platform
    amd_plat = Platform(
        part_number="xc7z020clg400-1",
        vendor="amd",
        resources=[{"lut": 53200, "ff": 106400, "bram_18k": 280, "dsp": 220}]
    )

    # Intel-style resources (even in Platform dataclass)
    intel_style = Platform(
        part_number="test",
        vendor="custom",
        resources=[{"alm": 251680, "m20k": 2713, "dsp_blocks": 1518}]
    )

    # Custom resources
    custom = Platform(
        part_number="custom_asic",
        vendor="custom",
        resources=[{"gates": 10000000, "sram_kb": 2048}]
    )

    assert "lut" in amd_plat.total_resources
    assert "alm" in intel_style.total_resources
    assert "gates" in custom.total_resources


def test_platform_total_resources():
    """Test total_resources aggregation across multiple dies/SLRs."""
    multi_slr = Platform(
        part_number="test-multi-slr",
        vendor="amd",
        resources=[
            {"lut": 100000, "dsp": 1000},
            {"lut": 150000, "dsp": 1500},
            {"lut": 100000, "dsp": 1000}
        ]
    )

    total = multi_slr.total_resources
    assert total["lut"] == 350000
    assert total["dsp"] == 3500


def test_platform_get_attribute(amd_platform):
    """Test platform attribute access."""
    assert amd_platform.get_attribute("clock_domains") == 1
    assert amd_platform.get_attribute("nonexistent", "default") == "default"
    assert amd_platform.get_attribute("nonexistent") is None


def test_platform_immutability(amd_platform):
    """Test that Platform is frozen (immutable)."""
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        amd_platform.vendor = "intel"


# =============================================================================
# Build Method Tests
# =============================================================================


def test_core_generation_supports(vivado_hls, quartus_hls, amd_platform, intel_platform):
    """Test build method compatibility checking."""
    # supports() returns bool
    is_supported = vivado_hls.supports(amd_platform)
    assert is_supported is True
    is_supported = vivado_hls.supports(intel_platform)
    assert is_supported is False
    is_supported = quartus_hls.supports(amd_platform)
    assert is_supported is False
    is_supported = quartus_hls.supports(intel_platform)
    assert is_supported is True


def test_core_generation_execute(vivado_hls):
    """Test core generation execution."""
    result = vivado_hls.execute("test_model")
    assert result == "vivado_hls(test_model)"


def test_integration_flow_supports(zynq_ps, nios_ii, amd_platform, intel_platform):
    """Test integration flow compatibility checking."""
    # supports() returns bool
    is_supported = zynq_ps.supports(amd_platform)
    assert is_supported is True
    is_supported = zynq_ps.supports(intel_platform)
    assert is_supported is False
    is_supported = nios_ii.supports(amd_platform)
    assert is_supported is False
    is_supported = nios_ii.supports(intel_platform)
    assert is_supported is True


def test_integration_flow_execute(zynq_ps):
    """Test integration flow execution."""
    result = zynq_ps.execute("test_model")
    assert result == "zynq_ps(test_model)"


# =============================================================================
# FPGATargetSpec Tests
# =============================================================================


def test_fpga_target_spec_valid(amd_platform, vivado_hls, zynq_ps):
    """Test valid FPGATargetSpec creation."""
    spec = FPGATargetSpec(
        platform=amd_platform,
        synthesis_backend=vivado_hls,
        integration_flow=zynq_ps
    )

    assert spec.platform == amd_platform
    assert spec.synthesis_backend == vivado_hls
    assert spec.integration_flow == zynq_ps


def test_fpga_target_spec_incompatible_core_generation(amd_platform, quartus_hls, zynq_ps):
    """Test FPGATargetSpec validation catches incompatible core generation."""
    with pytest.raises(ValueError, match="Synthesis backend.*does not support"):
        FPGATargetSpec(
            platform=amd_platform,
            synthesis_backend=quartus_hls,  # Intel method on AMD platform
            integration_flow=zynq_ps
        )


def test_fpga_target_spec_incompatible_integration(amd_platform, vivado_hls, nios_ii):
    """Test FPGATargetSpec validation catches incompatible integration flow."""
    with pytest.raises(ValueError, match="Integration flow.*does not support"):
        FPGATargetSpec(
            platform=amd_platform,
            synthesis_backend=vivado_hls,
            integration_flow=nios_ii  # Intel flow on AMD platform
        )


def test_fpga_target_spec_immutability(amd_platform, vivado_hls, zynq_ps):
    """Test that FPGATargetSpec is frozen (immutable)."""
    spec = FPGATargetSpec(
        platform=amd_platform,
        synthesis_backend=vivado_hls,
        integration_flow=zynq_ps
    )

    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        spec.platform = amd_platform


# =============================================================================
# AttributeMatchBuilder Tests
# =============================================================================


def test_attribute_match_builder_require_any():
    """Test require_any filter."""
    check = AttributeMatchBuilder().require_any("vendor", ["amd", "xilinx"]).build()

    amd_plat = Platform(part_number="test", vendor="amd")
    intel_plat = Platform(part_number="test", vendor="intel")

    assert check(amd_plat) is True
    assert check(intel_plat) is False


def test_attribute_match_builder_require():
    """Test require filter."""
    check = AttributeMatchBuilder().require("has_ps", True).build()

    with_ps = Platform(part_number="test", vendor="amd", has_ps=True)
    without_ps = Platform(part_number="test", vendor="amd", has_ps=False)

    assert check(with_ps) is True
    assert check(without_ps) is False


def test_attribute_match_builder_require_existence():
    """Test require filter with no value (existence check)."""
    check = AttributeMatchBuilder().require("special_feature").build()

    with_attr = Platform(
        part_number="test",
        vendor="amd",
        attributes={"special_feature": True}
    )
    without_attr = Platform(part_number="test", vendor="amd")

    assert check(with_attr) is True
    assert check(without_attr) is False


def test_attribute_match_builder_custom():
    """Test custom filter."""
    check = (
        AttributeMatchBuilder()
        .custom(lambda p: p.family.startswith("zynq"))
        .build()
    )

    zynq_plat = Platform(part_number="test", vendor="amd", family="zynq-7000")
    alveo_plat = Platform(part_number="test", vendor="amd", family="ultrascale+")

    assert check(zynq_plat) is True
    assert check(alveo_plat) is False


def test_attribute_match_builder_chaining():
    """Test chaining multiple filters."""
    check = (
        AttributeMatchBuilder()
        .require_any("vendor", ["amd", "xilinx"])
        .require("has_ps", True)
        .custom(lambda p: p.family == "zynq-7000")
        .build()
    )

    # Matches all criteria
    match = Platform(
        part_number="test",
        vendor="amd",
        family="zynq-7000",
        has_ps=True
    )

    # Fails vendor check
    fail_vendor = Platform(
        part_number="test",
        vendor="intel",
        family="zynq-7000",
        has_ps=True
    )

    # Fails has_ps check
    fail_ps = Platform(
        part_number="test",
        vendor="amd",
        family="zynq-7000",
        has_ps=False
    )

    assert check(match) is True
    assert check(fail_vendor) is False
    assert check(fail_ps) is False


# =============================================================================
# Registration API Tests
# =============================================================================


def test_register_and_get_platform(isolated_registry, amd_platform):
    """Test platform registration and retrieval."""
    register_platform("Pynq-Z1", amd_platform)

    retrieved = get_platform("Pynq-Z1")
    assert retrieved == amd_platform


def test_register_platform_collision_warning(isolated_registry, amd_platform):
    """Test that registering duplicate platform name warns but allows overwrite."""
    register_platform("Pynq-Z1", amd_platform)

    # Second registration should warn but succeed (overwrite)
    with pytest.warns(UserWarning, match="Overwriting existing platform"):
        register_platform("Pynq-Z1", amd_platform)

    # Verify the platform is still retrievable
    assert get_platform("Pynq-Z1") == amd_platform


def test_get_platform_not_found():
    """Test getting unregistered platform raises KeyError."""
    with pytest.raises(KeyError, match="Platform.*not found"):
        get_platform("NonExistent")


def test_register_and_get_synthesis_backend(isolated_registry, vivado_hls):
    """Test core generation registration and retrieval."""
    register_synthesis_backend(vivado_hls)

    retrieved = get_synthesis_backend("vivado_hls")
    assert retrieved == vivado_hls


def test_register_synthesis_backend_collision_warning(isolated_registry, vivado_hls):
    """Test that registering duplicate method name warns but allows overwrite."""
    register_synthesis_backend(vivado_hls)

    # Second registration should warn but succeed (overwrite)
    with pytest.warns(UserWarning, match="Overwriting existing synthesis backend"):
        register_synthesis_backend(vivado_hls)

    # Verify the method is still retrievable
    assert get_synthesis_backend("vivado_hls") == vivado_hls


def test_get_synthesis_backend_not_found():
    """Test getting unregistered method raises KeyError."""
    with pytest.raises(KeyError, match="Synthesis backend.*not found"):
        get_synthesis_backend("nonexistent")


def test_register_and_get_integration_flow(isolated_registry, zynq_ps):
    """Test integration flow registration and retrieval."""
    register_integration_flow(zynq_ps)

    retrieved = get_integration_flow("zynq_ps")
    assert retrieved == zynq_ps


def test_register_integration_flow_collision_warning(isolated_registry, zynq_ps):
    """Test that registering duplicate flow name warns but allows overwrite."""
    register_integration_flow(zynq_ps)

    # Second registration should warn but succeed (overwrite)
    with pytest.warns(UserWarning, match="Overwriting existing integration flow"):
        register_integration_flow(zynq_ps)

    # Verify the flow is still retrievable
    assert get_integration_flow("zynq_ps") == zynq_ps


def test_get_integration_flow_not_found():
    """Test getting unregistered flow raises KeyError."""
    with pytest.raises(KeyError, match="Integration flow.*not found"):
        get_integration_flow("nonexistent")


# =============================================================================
# Compatibility Query Tests
# =============================================================================


def test_get_compatible_synthesis_backends(isolated_registry, amd_platform, vivado_hls, quartus_hls):
    """Test finding compatible core generation methods."""
    register_synthesis_backend(vivado_hls)
    register_synthesis_backend(quartus_hls)

    compatible = get_compatible_synthesis_backends(amd_platform)
    compatible_names = [m.name for m in compatible]

    assert "vivado_hls" in compatible_names
    assert "quartus_hls" not in compatible_names


def test_get_compatible_integration_flows(isolated_registry, amd_platform, intel_platform, zynq_ps, nios_ii):
    """Test finding compatible integration flows."""
    register_integration_flow(zynq_ps)
    register_integration_flow(nios_ii)

    # AMD platform should be compatible with zynq_ps
    amd_compatible = get_compatible_integration_flows(amd_platform)
    amd_names = [f.name for f in amd_compatible]
    assert "zynq_ps" in amd_names
    assert "nios_ii" not in amd_names

    # Intel platform should be compatible with nios_ii
    intel_compatible = get_compatible_integration_flows(intel_platform)
    intel_names = [f.name for f in intel_compatible]
    assert "nios_ii" in intel_names
    assert "zynq_ps" not in intel_names


def test_get_compatible_methods_empty():
    """Test compatibility query with no custom methods (only built-ins)."""
    import finn.builder.platforms.core as plat_core

    # Manually clear to test with no methods at all
    plat_core._synthesis_backends.clear()
    plat_core._integration_flows.clear()

    platform = Platform(part_number="test", vendor="amd")

    assert get_compatible_synthesis_backends(platform) == []
    assert get_compatible_integration_flows(platform) == []

    # Re-register built-ins for other tests
    import finn.builder.platforms.builtin as builtin_module
    builtin_module._register_builtin_synthesis_backends()
    builtin_module._register_builtin_integration_flows()


# =============================================================================
# Phase 1: Two-Tier Compatibility Tests
# =============================================================================


def test_integration_flow_two_tier_compatibility():
    """Test IntegrationFlow two-tier compatibility (tested vs baseline)."""
    # Platform that matches tested_platforms (Tier 1)
    tested_platform = Platform(
        part_number="xc7z020clg400-1",
        vendor="amd",
        family="zynq-7000",
        has_ps=True
    )

    # Platform with same characteristics but different part (Tier 2)
    baseline_platform = Platform(
        part_number="xc7z030fbg484-1",
        vendor="amd",
        family="zynq-7000",
        has_ps=True
    )

    # Platform incompatible with baseline
    incompatible_platform = Platform(
        part_number="xcu250-figd2104-2L-e",
        vendor="amd",
        family="ultrascale+",
        has_ps=False  # No PS!
    )

    # Create flow with two-tier compatibility
    zynq_ps_flow = IntegrationFlow(
        name="zynq_ps",
        handler=lambda model, **kwargs: model,
        tested_platforms=["xc7z020clg400-1", "zynq-7000"],
        compatibility_check=lambda p: getattr(p, "has_ps", False),
        required_params=["board", "clk_period_ns"],
        optional_params={"mem_mode": "internal"}
    )

    # Test Tier 1 (tested platform)
    is_supported = zynq_ps_flow.supports(tested_platform)
    assert is_supported is True

    # Test Tier 2 (baseline compatible but not tested) - both tested due to family match
    is_supported = zynq_ps_flow.supports(baseline_platform)
    assert is_supported is True

    # Test incompatible
    is_supported = zynq_ps_flow.supports(incompatible_platform)
    assert is_supported is False


def test_core_generation_two_tier_compatibility():
    """Test SynthesisBackend two-tier compatibility."""
    # Tested platform (Tier 1)
    tested_platform = Platform(
        part_number="xc7z020clg400-1",
        vendor="amd",
        family="zynq-7000"
    )

    # Baseline compatible platform (Tier 2)
    baseline_platform = Platform(
        part_number="xcu250-figd2104-2L-e",
        vendor="amd",
        family="ultrascale+"
    )

    # Incompatible platform
    incompatible_platform = Platform(
        part_number="10AS066N3F40E2SG",
        vendor="intel",
        family="arria-10"
    )

    # Create core generation with two-tier compatibility
    vivado_hls = SynthesisBackend(
        name="vivado_hls",
        handler=lambda model, **kwargs: model,
        tested_platforms=["zynq-7000", "ultrascale", "ultrascale+"],
        compatibility_check=lambda p: p.vendor.lower() in ["xilinx", "amd"],
        required_params=["model", "clk_period_ns"],
        optional_params={"mem_mode": "internal"}
    )

    # Test Tier 1 (tested)
    is_supported = vivado_hls.supports(tested_platform)
    assert is_supported is True

    # Test Tier 2 (baseline compatible) - tested due to family match
    is_supported = vivado_hls.supports(baseline_platform)
    assert is_supported is True

    # Test incompatible
    is_supported = vivado_hls.supports(incompatible_platform)
    assert is_supported is False


def test_integration_flow_backward_compatibility():
    """Test IntegrationFlow backward compatibility with legacy compatibility_check."""
    # Legacy flow using compatibility_check (deprecated)
    legacy_flow = IntegrationFlow(
        name="legacy_flow",
        handler=lambda model, **kwargs: model,
        compatibility_check=lambda p: p.vendor == "amd"
    )

    platform = Platform(part_number="test", vendor="amd")

    # Should still work with new API
    is_supported = legacy_flow.supports(platform)
    assert is_supported is True
    # Legacy (no tested_platforms) doesn't issue warnings


def test_core_generation_backward_compatibility():
    """Test SynthesisBackend backward compatibility."""
    # Legacy method using compatibility_check
    legacy_method = SynthesisBackend(
        name="legacy_method",
        handler=lambda model, **kwargs: model,
        compatibility_check=lambda p: p.vendor == "amd"
    )

    platform = Platform(part_number="test", vendor="amd")

    is_supported = legacy_method.supports(platform)
    assert is_supported is True  # No tested_platforms, so no warning


def test_resolve_config_param_infrastructure():
    """Test that _resolve_config_param() infrastructure is in place."""
    from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType

    # Create minimal config that doesn't require platform resolution
    cfg = DataflowBuildConfig(
        output_dir="/tmp/test",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS]
    )

    # Test that helper function exists and is callable
    assert hasattr(cfg, "_resolve_config_param")
    assert callable(cfg._resolve_config_param)

    # Test that it returns None when no FPGA target spec
    result = cfg._resolve_config_param("vitis_platform")
    assert result is None  # No bitfile generation, no spec needed


def test_integration_flow_tested_platform_exact_match(isolated_registry):
    """Test that tested platforms return (True, True) for exact board name match."""
    from finn.builder.platforms import (
        Platform,
        IntegrationFlow,
        AttributeMatchBuilder,
        register_platform,
        register_integration_flow,
        get_platform,
        get_integration_flow,
    )

    # Register Pynq-Z1 platform
    pynq_z1 = Platform(
        part_number="xc7z020clg400-1",
        vendor="amd",
        family="zynq-7000",
        has_ps=True
    )
    register_platform("Pynq-Z1", pynq_z1)

    # Register zynq_ps flow with tested_platforms including Pynq-Z1
    zynq_ps = IntegrationFlow(
        name="zynq_ps",
        handler=lambda model, **kwargs: model,
        tested_platforms=["Pynq-Z1", "zynq-7000"],  # Include both board name and family
        compatibility_check=AttributeMatchBuilder().require_any("vendor", ["amd", "xilinx"]).require("has_ps", True).build(),
        required_params=["board"],
        optional_params={}
    )
    register_integration_flow(zynq_ps)

    # Now test
    platform = get_platform("Pynq-Z1")
    flow = get_integration_flow("zynq_ps")

    is_supported = flow.supports(platform)

    assert is_supported is True, "Pynq-Z1 should be supported by zynq_ps"
    # No warning expected since Pynq-Z1 is in tested_platforms list


def test_integration_flow_untested_but_compatible_platform(isolated_registry):
    """Test that untested Zynq platforms return (True, False) - Tier 2."""
    from finn.builder.platforms import (
        Platform,
        IntegrationFlow,
        AttributeMatchBuilder,
        register_platform,
        register_integration_flow,
        get_integration_flow,
    )

    # Register zynq_ps flow with specific tested platforms (NOT including xc7z030fbg484-1)
    zynq_ps = IntegrationFlow(
        name="zynq_ps",
        handler=lambda model, **kwargs: model,
        tested_platforms=["Pynq-Z1", "Pynq-Z2"],  # Only specific boards, NOT xc7z030
        compatibility_check=AttributeMatchBuilder().require_any("vendor", ["amd", "xilinx"]).require("has_ps", True).build(),
        required_params=["board"],
        optional_params={}
    )
    register_integration_flow(zynq_ps)

    # Register a custom Zynq platform NOT in tested_platforms list
    custom_zynq = Platform(
        part_number="xc7z030fbg484-1",  # Not in tested_platforms
        vendor="amd",
        family="zynq-7000",
        has_ps=True
    )
    register_platform("Custom-Zynq-Z030", custom_zynq)

    # Test - should issue warning since not in tested_platforms
    import warnings
    flow = get_integration_flow("zynq_ps")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        is_supported = flow.supports(custom_zynq)

        assert is_supported is True, "Should pass baseline compatibility (has_ps=True, vendor=amd)"
        assert len(w) == 1, "Should issue warning for untested platform"
        assert "not tested" in str(w[0].message), "Warning should mention platform not tested"


def test_integration_flow_incompatible_platform(isolated_registry):
    """Test that incompatible platforms return (False, False)."""
    from finn.builder.platforms import (
        Platform,
        IntegrationFlow,
        AttributeMatchBuilder,
        register_platform,
        register_integration_flow,
        get_platform,
        get_integration_flow,
    )

    # Register U250 Alveo platform (datacenter, NO PS)
    alveo_u250 = Platform(
        part_number="xcu250-figd2104-2L-e",
        vendor="amd",
        family="ultrascale-plus",
        has_ps=False  # ← No PS!
    )
    register_platform("U250", alveo_u250)

    # Register zynq_ps flow (requires has_ps=True)
    zynq_ps = IntegrationFlow(
        name="zynq_ps",
        handler=lambda model, **kwargs: model,
        tested_platforms=["Pynq-Z1"],
        compatibility_check=AttributeMatchBuilder().require_any("vendor", ["amd", "xilinx"]).require("has_ps", True).build(),
        required_params=["board"],
        optional_params={}
    )
    register_integration_flow(zynq_ps)

    # Test
    platform = get_platform("U250")
    flow = get_integration_flow("zynq_ps")

    is_supported = flow.supports(platform)

    assert is_supported is False, "U250 should fail baseline compatibility (has_ps=False)"


# =============================================================================
# Platform Inheritance Tests (base_platform)
# =============================================================================


def test_platform_inheritance():
    """Test that platforms can inherit from base_platform."""
    base_fpga = Platform(
        part_number="xc7z020clg400-1",
        vendor="amd",
        family="zynq-7000",
        resources=[{"lut": 53200, "ff": 106400}]
    )

    # NEW: No need to specify part_number - it's inherited!
    custom_board = Platform(
        base_platform=base_fpga,
        attributes={"board_name": "CustomBoard"}
    )

    # Board inherits FPGA specs via __post_init__ - direct field access works!
    assert custom_board.part_number == "xc7z020clg400-1"
    assert custom_board.vendor == "amd"
    assert custom_board.family == "zynq-7000"

    # get_attribute also works
    assert custom_board.get_attribute("vendor") == "amd"
    assert custom_board.get_attribute("family") == "zynq-7000"

    # Board has its own attributes
    assert custom_board.get_attribute("board_name") == "CustomBoard"

    # Resources are inherited
    assert custom_board.total_resources == {"lut": 53200, "ff": 106400}


def test_platform_attribute_override():
    """Test that derived platforms can override base attributes."""
    base_fpga = Platform(
        part_number="test",
        vendor="amd",
        attributes={"default_clock": 100}
    )

    # Test overriding attributes dict
    custom_board = Platform(
        base_platform=base_fpga,
        attributes={"default_clock": 200}  # Override
    )

    assert base_fpga.get_attribute("default_clock") == 100
    assert custom_board.get_attribute("default_clock") == 200

    # Test overriding dataclass fields
    custom_vendor = Platform(
        base_platform=base_fpga,
        vendor="custom"  # Override vendor field
    )

    assert custom_vendor.vendor == "custom"
    assert custom_vendor.part_number == "test"  # Still inherited


def test_multiple_boards_same_fpga(isolated_registry):
    """Test that multiple boards can share the same base FPGA."""
    base_fpga = Platform(
        part_number="xc7z020clg400-1",
        vendor="amd"
    )

    # Both boards inherit from same base
    board1 = Platform(
        base_platform=base_fpga,
        attributes={"board_name": "Board1"}
    )

    board2 = Platform(
        base_platform=base_fpga,
        attributes={"board_name": "Board2"}
    )

    # Register all three
    register_platform("xc7z020clg400-1", base_fpga)
    register_platform("Board1", board1)
    register_platform("Board2", board2)

    # Part number gives base FPGA
    fpga = get_platform("xc7z020clg400-1")
    assert not fpga.is_derived_platform

    # Board names give specific boards
    b1 = get_platform("Board1")
    assert b1.get_attribute("board_name") == "Board1"

    b2 = get_platform("Board2")
    assert b2.get_attribute("board_name") == "Board2"


def test_is_derived_platform():
    """Test introspection helpers."""
    base = Platform(part_number="test", vendor="amd")
    derived = Platform(base_platform=base)  # Inherits part_number

    assert not base.is_derived_platform
    assert derived.is_derived_platform


def test_backward_compatibility():
    """Test that platforms without base_platform still work."""
    standalone = Platform(
        part_number="test",
        vendor="amd",
        resources=[{"lut": 1000}]
    )

    # Should work as before
    assert standalone.part_number == "test"
    assert standalone.total_resources == {"lut": 1000}
    assert standalone.get_attribute("vendor") == "amd"
    assert standalone.base_platform is None
    assert not standalone.is_derived_platform


def test_standalone_platform_requires_part_number():
    """Test that standalone platforms must specify part_number."""
    import pytest

    # Should raise ValueError if part_number not specified and no base_platform
    with pytest.raises(ValueError, match="part_number is required"):
        Platform(vendor="amd", family="test")


# =============================================================================
# Integration Tests for Built-In Platform Hierarchy
# =============================================================================


def test_builtin_platform_hierarchy(builtin_registry):
    """Test that built-in platforms use base_platform correctly."""
    # Part number gives base FPGA
    fpga = get_platform("xc7z020clg400-1")
    assert not fpga.is_derived_platform

    # Board names give derived platforms
    pynq_z1 = get_platform("Pynq-Z1")
    assert pynq_z1.is_derived_platform
    assert pynq_z1.base_platform is not None
    assert pynq_z1.get_attribute("board_name") == "Pynq-Z1"

    pynq_z2 = get_platform("Pynq-Z2")
    assert pynq_z2.is_derived_platform
    assert pynq_z2.get_attribute("board_name") == "Pynq-Z2"

    # Both boards share same base FPGA
    assert pynq_z1.base_platform.part_number == "xc7z020clg400-1"
    assert pynq_z2.base_platform.part_number == "xc7z020clg400-1"


def test_builtin_platforms_inherit_resources(builtin_registry):
    """Test that board platforms inherit resources from base FPGAs."""
    pynq_z1 = get_platform("Pynq-Z1")
    base_fpga = get_platform("xc7z020clg400-1")

    # Board should inherit resources from base
    assert pynq_z1.total_resources == base_fpga.total_resources
    assert "lut" in pynq_z1.total_resources
    assert "dsp" in pynq_z1.total_resources


def test_all_builtin_boards_have_base_platform(builtin_registry):
    """Test that all built-in board platforms have base_platform set."""
    board_names = ["Pynq-Z1", "Pynq-Z2", "Ultra96", "ZCU104", "U250", "U280"]

    for board_name in board_names:
        board = get_platform(board_name)
        assert board.is_derived_platform, f"{board_name} should be a derived platform"
        assert board.base_platform is not None, f"{board_name} should have base_platform"
        assert not board.base_platform.is_derived_platform, f"{board_name}'s base should not be derived"


def test_error_message_shows_hierarchy(builtin_registry):
    """Test that error messages show platform hierarchy."""
    try:
        get_platform("NonExistent")
        assert False, "Should raise KeyError"
    except KeyError as e:
        error_msg = str(e)
        # Should show derived platforms with part numbers
        assert "part:" in error_msg.lower() or "Pynq-Z1" in error_msg
        # Should list available platforms
        assert "Available platforms" in error_msg or "available" in error_msg.lower()


# =============================================================================
# Handler Validation Tests (Priority 1)
# =============================================================================


def test_validate_handler_kwargs_success():
    """Test validate_handler_kwargs with all required params present."""
    from finn.builder.platforms.core import validate_handler_kwargs

    kwargs = {"fpga_part": "xc7z020clg400-1", "clk_period_ns": 10.0}
    # Should not raise
    validate_handler_kwargs("test_handler", kwargs, ["fpga_part", "clk_period_ns"])


def test_validate_handler_kwargs_missing_param():
    """Test validate_handler_kwargs raises KeyError when params missing."""
    from finn.builder.platforms.core import validate_handler_kwargs

    kwargs = {"fpga_part": "xc7z020clg400-1"}

    with pytest.raises(KeyError) as exc_info:
        validate_handler_kwargs("test_handler", kwargs, ["fpga_part", "clk_period_ns"])

    error_msg = str(exc_info.value)
    assert "test_handler requires 'clk_period_ns'" in error_msg
    assert "['fpga_part']" in error_msg


def test_handler_validation_consistency():
    """Test that all handlers use consistent validation via validate_handler_kwargs."""
    from finn.builder.platforms.vivado_handlers import vivado_hls_handler
    from qonnx.core.modelwrapper import ModelWrapper
    import tempfile
    import onnx

    # Create minimal test model
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty ONNX model
        graph = onnx.helper.make_graph([], "test", [], [])
        model_proto = onnx.helper.make_model(graph)
        model_path = f"{tmpdir}/test.onnx"
        onnx.save(model_proto, model_path)
        model = ModelWrapper(model_path)

        # Test missing required param
        with pytest.raises(KeyError) as exc_info:
            vivado_hls_handler(model, fpga_part="xc7z020clg400-1")

        error_msg = str(exc_info.value)
        assert "vivado_hls_handler requires 'clk_period_ns'" in error_msg


# =============================================================================
# Error Message Sorting Test (Priority 3)
# =============================================================================


def test_platform_error_message_sorted(builtin_registry):
    """Test that platform error messages show platforms in alphabetical order."""
    try:
        get_platform("NonExistentPlatform")
        assert False, "Should raise KeyError"
    except KeyError as e:
        # Get the args[0] which is the actual message (not the repr string)
        error_msg = e.args[0]

        # Extract platform list from error message
        lines = error_msg.split('\n')

        # Find lines with platform listings (start with two spaces and contain platform info)
        platform_lines = []
        for line in lines:
            if line.startswith('  ') and (line.strip().startswith('Pynq') or
                                         line.strip().startswith('U') or
                                         line.strip().startswith('xc') or
                                         line.strip().startswith('ZCU')):
                platform_lines.append(line.strip())

        # Should have multiple platforms listed
        assert len(platform_lines) > 2, f"Found {len(platform_lines)} platform lines"

        # Extract platform names (first word before space or parenthesis)
        platform_names = []
        for line in platform_lines:
            # Get first word (platform name)
            name = line.split()[0]
            platform_names.append(name)

        # Check they're sorted
        sorted_names = sorted(platform_names)
        assert platform_names == sorted_names, f"Platforms not sorted:\n{platform_names}\nvs expected:\n{sorted_names}"


# =============================================================================
# Schema Introspection Tests (Priority 2b)
# =============================================================================


def test_get_synthesis_backend_schema():
    """Test schema introspection for core generation methods."""
    from finn.builder.platforms import get_synthesis_backend_schema

    # Test vivado_hls schema
    # Note: Core generation methods don't have flow-driven params (they use model object)
    # Handler-level validation is done separately via validate_handler_kwargs
    schema = get_synthesis_backend_schema("vivado_hls")
    assert "required" in schema
    assert "optional" in schema
    assert isinstance(schema["required"], list)
    assert isinstance(schema["optional"], dict)


def test_get_integration_flow_schema():
    """Test schema introspection for integration flows."""
    from finn.builder.platforms import get_integration_flow_schema

    # Test zynq_ps schema
    schema = get_integration_flow_schema("zynq_ps")
    assert "required" in schema
    assert "optional" in schema
    assert "board" in schema["required"]
    assert "clk_period_ns" in schema["required"]
    assert "output_dir" in schema["required"]
    assert "enable_hw_debug" in schema["optional"]
    assert schema["optional"]["enable_hw_debug"] == False
    assert "partition_model_dir" in schema["optional"]


def test_schema_nonexistent_method():
    """Test that schema functions raise KeyError for nonexistent methods."""
    from finn.builder.platforms import (
        get_synthesis_backend_schema,
        get_integration_flow_schema
    )
    import pytest

    with pytest.raises(KeyError):
        get_synthesis_backend_schema("nonexistent_method")

    with pytest.raises(KeyError):
        get_integration_flow_schema("nonexistent_flow")
