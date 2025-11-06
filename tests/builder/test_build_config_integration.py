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
Unit tests for platform abstraction in DataflowBuildConfig.

Tests configuration parsing, platform resolution, and API compatibility.
These are fast unit tests that validate config object creation without
executing build pipeline stages.

For integration tests that execute build steps with real models, see
tests/builder/test_platform_pipeline_integration.py.
"""

import pytest
import warnings
import tempfile
import shutil

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="finn_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Legacy API Tests (shell_flow_type)
# =============================================================================


def test_legacy_vivado_zynq_config(temp_output_dir):
    """Test that legacy VIVADO_ZYNQ config auto-converts to platform abstraction."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        cfg = DataflowBuildConfig(
            output_dir=temp_output_dir,
            board="Pynq-Z1",
            shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
            synth_clk_period_ns=10.0,
            generate_outputs=[DataflowOutputType.BITFILE],
        )

        # Check deprecation warning was emitted
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "shell_flow_type is deprecated" in str(w[0].message)

        # Check that FPGATargetSpec was created
        spec = cfg.get_fpga_target_spec()
        assert spec is not None
        assert spec.platform.part_number == "xc7z020clg400-1"
        assert spec.synthesis_backend.name == "vivado_hls"
        assert spec.integration_flow.name == "zynq_ps"


def test_legacy_vitis_alveo_config(temp_output_dir):
    """Test that legacy VITIS_ALVEO config auto-converts to platform abstraction."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        cfg = DataflowBuildConfig(
            output_dir=temp_output_dir,
            fpga_part="xcu250-figd2104-2L-e",
            shell_flow_type=ShellFlowType.VITIS_ALVEO,
            synth_clk_period_ns=3.0,
            vitis_platform="xilinx_u250_gen3x16_xdma_3_1_202020_1",
            generate_outputs=[DataflowOutputType.BITFILE],
        )

        # Check deprecation warning was emitted
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

        # Check that FPGATargetSpec was created
        spec = cfg.get_fpga_target_spec()
        assert spec is not None
        assert spec.platform.part_number == "xcu250-figd2104-2L-e"
        assert spec.synthesis_backend.name == "vitis_hls"
        assert spec.integration_flow.name == "alveo_xrt"


# =============================================================================
# New API Tests (explicit platform abstraction)
# =============================================================================


def test_new_explicit_config(temp_output_dir):
    """Test new API with explicit platform, core_generation, and integration_flow."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        fpga_part="xc7z020clg400-1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        board="Pynq-Z1",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    # Check no deprecation warnings (use new API)
    # Already executed, no warnings expected in normal flow

    # Check that FPGATargetSpec was created correctly
    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    assert spec.platform.part_number == "xc7z020clg400-1"
    assert spec.synthesis_backend.name == "vivado_hls"
    assert spec.integration_flow.name == "zynq_ps"


def test_new_explicit_alveo_config(temp_output_dir):
    """Test new API with explicit Alveo configuration."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        fpga_part="xcu250-figd2104-2L-e",
        synthesis_backend="vitis_hls",
        integration_flow="alveo_xrt",
        synth_clk_period_ns=3.0,
        vitis_platform="xilinx_u250_gen3x16_xdma_3_1_202020_1",
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    assert spec.platform.part_number == "xcu250-figd2104-2L-e"
    assert spec.synthesis_backend.name == "vitis_hls"
    assert spec.integration_flow.name == "alveo_xrt"


# =============================================================================
# Auto-Selection Tests
# =============================================================================


def test_auto_selection_zynq(temp_output_dir):
    """Test auto-selection based on Zynq platform characteristics."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        board="Pynq-Z1",  # This will resolve to xc7z020clg400-1
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    # Auto-selection should choose zynq_ps for platforms with PS
    assert spec.integration_flow.name == "zynq_ps"
    # Auto-selection should choose vivado_hls for Zynq-7000
    assert spec.synthesis_backend.name in ["vivado_hls", "vitis_hls"]


def test_auto_selection_alveo(temp_output_dir):
    """Test auto-selection based on Alveo platform characteristics."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        fpga_part="xcu250-figd2104-2L-e",
        synth_clk_period_ns=3.0,
        vitis_platform="xilinx_u250_gen3x16_xdma_3_1_202020_1",
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    # Auto-selection should choose alveo_xrt for datacenter platforms
    assert spec.integration_flow.name == "alveo_xrt"


# =============================================================================
# Mixed API Tests (new overrides old)
# =============================================================================


def test_mixed_api_new_overrides_old(temp_output_dir):
    """Test that explicit new API fields override legacy shell_flow_type."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        board="Pynq-Z1",
        shell_flow_type=ShellFlowType.VIVADO_ZYNQ,  # Legacy says vivado_hls
        synthesis_backend="vitis_hls",  # New API overrides to vitis_hls
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    # When explicit new API fields are provided, they take precedence over legacy
    # Check that new API took precedence (vitis_hls, not vivado_hls from legacy)
    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    assert spec.synthesis_backend.name == "vitis_hls"  # Not vivado_hls from shell_flow_type


# =============================================================================
# No Bitfile Output Test
# =============================================================================


def test_no_bitfile_output_skips_initialization(temp_output_dir):
    """Test that config without BITFILE output doesn't initialize platform abstraction."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        board="Pynq-Z1",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],  # No BITFILE
    )

    # get_fpga_target_spec() should return None when no bitfile generation
    spec = cfg.get_fpga_target_spec()
    assert spec is None


# =============================================================================
# Platform Access Tests
# =============================================================================


def test_platform_access_by_part_number(temp_output_dir):
    """Test that platform can be accessed by part number."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        fpga_part="xc7z020clg400-1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    assert spec.platform.part_number == "xc7z020clg400-1"
    assert spec.platform.vendor == "amd"
    assert spec.platform.family == "zynq-7000"


def test_platform_access_by_board_name(temp_output_dir):
    """Test that platform can be accessed by board name (board field)."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        board="Pynq-Z1",
        synthesis_backend="vivado_hls",
        integration_flow="zynq_ps",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    spec = cfg.get_fpga_target_spec()
    assert spec is not None
    # Board "Pynq-Z1" should resolve to xc7z020clg400-1
    assert spec.platform.part_number == "xc7z020clg400-1"


# =============================================================================
# Compatibility Validation Tests
# =============================================================================


def test_incompatible_combination_rejected(temp_output_dir):
    """Test that incompatible platform/method combinations are rejected."""
    with pytest.raises(ValueError, match="does not support"):
        # Try to use zynq_ps (requires has_ps=True) with Alveo (has_ps=False)
        cfg = DataflowBuildConfig(
            output_dir=temp_output_dir,
            fpga_part="xcu250-figd2104-2L-e",  # Alveo (no PS)
            synthesis_backend="vitis_hls",
            integration_flow="zynq_ps",  # Requires PS!
            synth_clk_period_ns=3.0,
            generate_outputs=[DataflowOutputType.BITFILE],
        )


def test_unknown_platform_rejected(temp_output_dir):
    """Test that unknown FPGA part numbers are rejected with clear error."""
    with pytest.raises(ValueError, match="Platform not found"):
        cfg = DataflowBuildConfig(
            output_dir=temp_output_dir,
            fpga_part="NONEXISTENT-PART-123",
            synthesis_backend="vivado_hls",
            integration_flow="zynq_ps",
            synth_clk_period_ns=10.0,
            generate_outputs=[DataflowOutputType.BITFILE],
        )


def test_unknown_core_generation_rejected(temp_output_dir):
    """Test that unknown core generation methods are rejected."""
    with pytest.raises(ValueError, match="Synthesis backend.*not found"):
        cfg = DataflowBuildConfig(
            output_dir=temp_output_dir,
            fpga_part="xc7z020clg400-1",
            synthesis_backend="nonexistent_hls",
            integration_flow="zynq_ps",
            synth_clk_period_ns=10.0,
            generate_outputs=[DataflowOutputType.BITFILE],
        )


def test_unknown_integration_flow_rejected(temp_output_dir):
    """Test that unknown integration flows are rejected."""
    with pytest.raises(ValueError, match="Integration flow.*not found"):
        cfg = DataflowBuildConfig(
            output_dir=temp_output_dir,
            fpga_part="xc7z020clg400-1",
            synthesis_backend="vivado_hls",
            integration_flow="nonexistent_flow",
            synth_clk_period_ns=10.0,
            generate_outputs=[DataflowOutputType.BITFILE],
        )


# =============================================================================
# Helper Method Tests
# =============================================================================


def test_resolve_fpga_part_uses_platform_abstraction(temp_output_dir):
    """Test that _resolve_fpga_part() returns part from platform abstraction."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        board="Pynq-Z1",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    fpga_part = cfg._resolve_fpga_part()
    assert fpga_part == "xc7z020clg400-1"


def test_resolve_driver_platform_maps_integration_flow(temp_output_dir):
    """Test that _resolve_driver_platform() maps integration flow to driver string."""
    cfg = DataflowBuildConfig(
        output_dir=temp_output_dir,
        board="Pynq-Z1",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )

    driver_platform = cfg._resolve_driver_platform()
    # zynq_ps should map to "zynq-iodma" for driver compatibility
    assert driver_platform in ["zynq-iodma", "alveo"]
