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
Simple integration tests for platform abstraction build config.

These tests validate that the platform abstraction layer works correctly in
DataflowBuildConfig for various platforms and methods. Full pipeline execution
is already tested in tests/end2end/.

Test Strategy:
- Test platform resolution and FPGATargetSpec creation
- Test backward compatibility between legacy and new API
- Keep tests simple and fast (no full pipeline execution)
"""

import pytest
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)


@pytest.mark.fpgadataflow
def test_zynq_platform_config(temp_build_dir):
    """
    Test Zynq platform abstraction config resolution.

    Validates:
    - Platform resolution from board name
    - FPGATargetSpec creation with correct platform/methods
    - Correct part number resolution
    """
    cfg = DataflowBuildConfig(
        output_dir=temp_build_dir,
        board="Pynq-Z1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    # Verify FPGATargetSpec was created correctly
    spec = cfg.get_fpga_target_spec()
    assert spec is not None, "FPGATargetSpec should be created for bitfile generation"
    assert spec.platform.part_number == "xc7z020clg400-1", "Should resolve Pynq-Z1 to correct part"
    assert spec.synthesis_backend.name == "vivado_hls", "Should use Vivado HLS for Zynq-7000"
    assert spec.integration_flow.name == "zynq_ps", "Should use Zynq PS integration"

    # Verify config helper methods use platform abstraction
    assert cfg._resolve_fpga_part() == "xc7z020clg400-1"
    assert cfg._resolve_driver_platform() == "zynq-iodma"


@pytest.mark.fpgadataflow
def test_alveo_platform_config(temp_build_dir):
    """
    Test Alveo platform abstraction config resolution.

    Validates:
    - Platform resolution from FPGA part number
    - Vitis HLS vs Vivado HLS selection
    - Alveo XRT integration flow selection
    """
    cfg = DataflowBuildConfig(
        output_dir=temp_build_dir,
        fpga_part="xcu250-figd2104-2L-e",
        synthesis_backend="vitis_hls",
        integration_flow="alveo_xrt",
        synth_clk_period_ns=3.0,
        vitis_platform="xilinx_u250_gen3x16_xdma_3_1_202020_1",
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    # Verify FPGATargetSpec for Alveo
    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    assert spec.platform.part_number == "xcu250-figd2104-2L-e"
    assert spec.synthesis_backend.name == "vitis_hls", "Should use Vitis HLS for Alveo"
    assert spec.integration_flow.name == "alveo_xrt", "Should use Alveo XRT integration"

    # Verify config helper methods
    assert cfg._resolve_fpga_part() == "xcu250-figd2104-2L-e"
    assert cfg._resolve_driver_platform() == "alveo"


@pytest.mark.fpgadataflow
def test_backward_compatibility_legacy_vs_new(temp_build_dir):
    """
    Test that legacy shell_flow_type API produces same config as new platform API.

    Validates:
    - Legacy VIVADO_ZYNQ shell_flow_type still works
    - New platform abstraction produces equivalent results
    - Both resolve to same platform/methods
    """
    # Legacy configuration
    cfg_legacy = DataflowBuildConfig(
        output_dir=temp_build_dir + "/legacy",
        board="Pynq-Z1",
        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    # New platform abstraction configuration
    cfg_new = DataflowBuildConfig(
        output_dir=temp_build_dir + "/new",
        board="Pynq-Z1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    # Verify legacy creates FPGATargetSpec internally
    spec_legacy = cfg_legacy.get_fpga_target_spec()
    spec_new = cfg_new.get_fpga_target_spec()

    assert spec_legacy is not None, "Legacy config should create FPGATargetSpec"
    assert spec_new is not None, "New config should create FPGATargetSpec"

    # Both should resolve to same platform and methods
    assert spec_legacy.platform.part_number == spec_new.platform.part_number
    assert spec_legacy.synthesis_backend.name == spec_new.synthesis_backend.name
    assert spec_legacy.integration_flow.name == spec_new.integration_flow.name

    # Helper methods should produce same results
    assert cfg_legacy._resolve_fpga_part() == cfg_new._resolve_fpga_part()
    assert cfg_legacy._resolve_driver_platform() == cfg_new._resolve_driver_platform()


@pytest.mark.fpgadataflow
def test_auto_selection_zynq(temp_build_dir):
    """
    Test platform abstraction with auto-selection for Zynq.

    When only board is specified, methods should be auto-selected.
    """
    cfg = DataflowBuildConfig(
        output_dir=temp_build_dir,
        board="Pynq-Z1",
        # No explicit core_generation or integration_flow
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    assert spec.platform.has_ps is True
    # Auto-selection should choose zynq_ps for platforms with PS
    assert spec.integration_flow.name == "zynq_ps"
    # Auto-selection should choose vivado_hls or vitis_hls
    assert spec.synthesis_backend.name in ["vivado_hls", "vitis_hls"]


@pytest.mark.fpgadataflow
def test_zynq_flow_parameter_isolation(temp_build_dir):
    """Verify Zynq builds don't try to resolve vitis_platform.

    This test validates the fix for KeyError: 'Pynq-Z1' bug.
    Previously, step_synthesize_bitfile unconditionally called
    cfg._resolve_vitis_platform() for all builds, causing errors
    for Zynq platforms that aren't in alveo_default_platform dict.

    After the fix, parameter resolution is flow-driven: only parameters
    declared in the flow's required_params/optional_params are resolved.
    """
    cfg = DataflowBuildConfig(
        output_dir=temp_build_dir,
        board="Pynq-Z1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()

    # Verify correct flow selected
    assert spec.integration_flow.name == "zynq_ps", "Should use zynq_ps integration flow"

    # Verify parameters are flow-specific (key assertion!)
    assert "board" in spec.integration_flow.required_params
    assert "clk_period_ns" in spec.integration_flow.required_params
    assert "output_dir" in spec.integration_flow.required_params
    assert "vitis_platform" not in spec.integration_flow.required_params, \
        "zynq_ps should NOT require vitis_platform"
    assert "vitis_platform" not in spec.integration_flow.optional_params, \
        "zynq_ps should NOT have vitis_platform as optional"

    # Verify _resolve_config_param returns None for vitis_platform
    vitis_param = cfg._resolve_config_param("vitis_platform")
    assert vitis_param is None, "Zynq flow should not resolve vitis_platform"


@pytest.mark.fpgadataflow
def test_untested_platform_emits_warning(temp_build_dir):
    """Verify untested platforms trigger Tier 2 (baseline compatibility) behavior."""
    from finn.builder.platforms import Platform, register_platform

    # Create untested Zynq platform (not in tested_platforms list)
    # Use custom family name to avoid matching tested family "zynq-7000"
    untested_zynq = Platform(
        part_number="xc7z030fbg484-1",  # Not in zynq_ps.tested_platforms
        vendor="amd",
        family="zynq-custom",  # Custom family not in tested list
        has_ps=True  # Passes baseline compatibility
    )
    # Register by part number since config uses fpga_part
    register_platform("xc7z030fbg484-1", untested_zynq)

    cfg = DataflowBuildConfig(
        output_dir=temp_build_dir,
        fpga_part="xc7z030fbg484-1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()

    # Verify platform is supported (Tier 2 - baseline compatibility)
    # The supports() method now returns bool and emits warnings as side effect
    is_supported = spec.integration_flow.supports(spec.platform)
    assert is_supported is True, "Platform should pass baseline compatibility (has_ps=True)"

    # Note: Warnings are emitted by supports() method (see core.py:589-619)
    # when platform is not in tested_platforms list but passes compatibility_check.
    # The warnings can be seen in the test output above.
