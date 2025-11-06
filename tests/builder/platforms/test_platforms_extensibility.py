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
External vendor extensibility tests for platform abstraction.

These tests validate that external meta-compilers (non-AMD vendors) can register
custom platforms, core generation methods, and integration flows.

NOTE: These tests use minimal realistic handlers that write actual files to validate
the API, but they do NOT execute full synthesis or integration. For full integration
testing with real FINN models, see tests/builder/test_platform_pipeline_integration.py.
"""

import pytest
import os
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict

from finn.builder.platforms import (
    PlatformProtocol,
    SynthesisBackend,
    IntegrationFlow,
    FPGATargetSpec,
    register_platform,
    register_synthesis_backend,
    register_integration_flow,
    get_platform,
    get_synthesis_backend,
    get_integration_flow,
)

# =============================================================================
# Test Fixtures - External Platform (Intel Example)
# =============================================================================


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
def temp_output_dir():
    """Temporary directory for handler file outputs."""
    temp_dir = tempfile.mkdtemp(prefix="finn_test_ext_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def isolated_registry():
    """Provide completely empty registry for external vendor testing."""
    import finn.builder.platforms.core as plat_core

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


# =============================================================================
# Test Fixtures - Realistic Handlers (External Vendor Methods)
# =============================================================================


@pytest.fixture
def quartus_hls_handler():
    """
    Minimal realistic Quartus HLS core generation handler.

    This handler demonstrates external vendor integration by:
    - Creating Quartus project directory structure
    - Writing minimal .qpf (Quartus Project File)
    - Setting model metadata

    NOTE: This is a minimal handler for API testing. A real Quartus HLS
    handler would generate full TCL scripts, IP components, etc.
    """

    def handler(model, fpga_part, clk_period_ns, output_dir, **kwargs):
        # Create Quartus project structure
        proj_dir = os.path.join(output_dir, "quartus_proj")
        os.makedirs(proj_dir, exist_ok=True)

        # Write minimal Quartus Project File (.qpf)
        qpf_path = os.path.join(proj_dir, "finn_accel.qpf")
        with open(qpf_path, "w") as f:
            f.write(f"# Quartus Prime Project File\n")
            f.write(f"# Generated by FINN for {fpga_part}\n")
            f.write(f"# Target clock period: {clk_period_ns} ns\n")
            f.write(f"\n")
            f.write(f"PROJECT_REVISION = \"finn_accel\"\n")

        # Write minimal Quartus Settings File (.qsf)
        qsf_path = os.path.join(proj_dir, "finn_accel.qsf")
        with open(qsf_path, "w") as f:
            f.write(f"set_global_assignment -name FAMILY \"Arria 10\"\n")
            f.write(f"set_global_assignment -name DEVICE {fpga_part}\n")

        # Set metadata on model (simulating what a real handler would do)
        model.set_metadata_prop("quartus_project_dir", proj_dir)
        model.set_metadata_prop("quartus_project_file", qpf_path)
        model.set_metadata_prop("fpga_part", fpga_part)

        return model

    return handler


@pytest.fixture
def nios_ii_handler():
    """
    Minimal realistic Nios II integration flow handler.

    This handler demonstrates external integration flow by:
    - Creating Qsys system directory
    - Writing minimal Platform Designer TCL script
    - Setting integration metadata

    NOTE: This is a minimal handler for API testing. A real Nios II handler
    would generate complete Qsys systems, AXI interconnect, etc.
    """

    def handler(model, fpga_part, output_dir, **kwargs):
        # Create Nios II system directory
        system_dir = os.path.join(output_dir, "nios_system")
        os.makedirs(system_dir, exist_ok=True)

        # Write minimal Platform Designer / Qsys TCL
        qsys_tcl_path = os.path.join(system_dir, "finn_system.tcl")
        with open(qsys_tcl_path, "w") as f:
            f.write("# Qsys System for FINN Accelerator\n")
            f.write("# Platform: Nios II with Avalon-MM interface\n")
            f.write("\n")
            f.write("# Add Nios II processor\n")
            f.write("add_instance nios2 altera_nios2_gen2\n")
            f.write("\n")
            f.write("# Add FINN accelerator (would be actual IP in real handler)\n")
            f.write("add_instance finn_accel finn_accelerator\n")
            f.write("\n")
            f.write("# Connect via Avalon-MM bus\n")
            f.write("add_connection nios2.data_master finn_accel.s_avalon\n")

        # Set integration metadata
        model.set_metadata_prop("nios_system_dir", system_dir)
        model.set_metadata_prop("qsys_script", qsys_tcl_path)
        model.set_metadata_prop("integration_flow", "nios_ii")

        return model

    return handler


# =============================================================================
# External Platform Registration Tests
# =============================================================================


def test_external_platform_registration(isolated_registry, intel_platform):
    """Test that external platforms can be registered and retrieved."""
    # External meta-compiler registers Intel platform
    register_platform("DE10-Nano", intel_platform)

    # Retrieve and validate
    retrieved = get_platform("DE10-Nano")
    assert retrieved.vendor == "intel"
    assert retrieved.part_number == "10AS066N3F40E2SG"
    assert retrieved.get_attribute("has_hps") is True


# =============================================================================
# External Build Method Registration Tests
# =============================================================================


def test_external_build_method_registration(
    isolated_registry, quartus_hls_handler, nios_ii_handler, intel_platform, temp_output_dir
):
    """
    Test that external build methods can be registered with realistic handlers.

    Validates that:
    - Methods can be registered
    - Handlers write actual files
    - Metadata is set correctly
    """
    # Create minimal mock model for testing
    class MockModel:
        def __init__(self):
            self.metadata = {}

        def set_metadata_prop(self, key, value):
            self.metadata[key] = value

        def get_metadata_prop(self, key):
            return self.metadata.get(key)

    # Register methods with realistic handlers
    quartus_method = SynthesisBackend(
        name="quartus_hls",
        handler=quartus_hls_handler,
        compatibility_check=lambda p: p.vendor == "intel",
    )

    nios_method = IntegrationFlow(
        name="nios_ii",
        handler=nios_ii_handler,
        compatibility_check=lambda p: p.get_attribute("has_hps", False),
    )

    register_synthesis_backend(quartus_method)
    register_integration_flow(nios_method)

    # Retrieve and validate registration
    core_gen = get_synthesis_backend("quartus_hls")
    integration = get_integration_flow("nios_ii")

    assert core_gen.name == "quartus_hls"
    assert integration.name == "nios_ii"

    # Execute handlers and validate file outputs
    model = MockModel()

    # Test Quartus HLS handler
    result_model = core_gen.execute(
        model, fpga_part=intel_platform.part_number, clk_period_ns=10.0, output_dir=temp_output_dir
    )

    assert result_model.get_metadata_prop("quartus_project_dir") is not None
    quartus_proj_dir = result_model.get_metadata_prop("quartus_project_dir")
    assert os.path.exists(quartus_proj_dir)
    assert os.path.exists(os.path.join(quartus_proj_dir, "finn_accel.qpf"))
    assert os.path.exists(os.path.join(quartus_proj_dir, "finn_accel.qsf"))

    # Test Nios II handler
    result_model = integration.execute(result_model, fpga_part=intel_platform.part_number, output_dir=temp_output_dir)

    assert result_model.get_metadata_prop("nios_system_dir") is not None
    nios_system_dir = result_model.get_metadata_prop("nios_system_dir")
    assert os.path.exists(nios_system_dir)
    assert os.path.exists(os.path.join(nios_system_dir, "finn_system.tcl"))


# =============================================================================
# Full External Workflow Test
# =============================================================================


def test_full_external_workflow(isolated_registry, intel_platform, quartus_hls_handler, nios_ii_handler, temp_output_dir):
    """
    Test complete workflow with external platform and methods.

    This demonstrates the full flow of an external meta-compiler:
    1. Register custom platform (Intel Arria-10)
    2. Register custom core generation method (Quartus HLS)
    3. Register custom integration flow (Nios II)
    4. Create FPGATargetSpec with all custom components
    5. Execute workflow and validate file outputs
    """
    # Create minimal mock model
    class MockModel:
        def __init__(self):
            self.metadata = {}

        def set_metadata_prop(self, key, value):
            self.metadata[key] = value

        def get_metadata_prop(self, key):
            return self.metadata.get(key)

    # External meta-compiler registers its components
    register_platform("Arria-10", intel_platform)

    quartus_method = SynthesisBackend(
        name="quartus_hls",
        handler=quartus_hls_handler,
        compatibility_check=lambda p: p.vendor == "intel",
    )
    register_synthesis_backend(quartus_method)

    nios_method = IntegrationFlow(
        name="nios_ii",
        handler=nios_ii_handler,
        compatibility_check=lambda p: p.get_attribute("has_hps", False),
    )
    register_integration_flow(nios_method)

    # Build configuration uses registered components
    platform = get_platform("Arria-10")
    core_gen = get_synthesis_backend("quartus_hls")
    integration = get_integration_flow("nios_ii")

    # Create and validate target spec
    spec = FPGATargetSpec(platform=platform, synthesis_backend=core_gen, integration_flow=integration)

    assert spec.platform.vendor == "intel"
    assert spec.synthesis_backend.name == "quartus_hls"
    assert spec.integration_flow.name == "nios_ii"

    # Execute complete workflow
    model = MockModel()

    # Core generation step
    model = spec.synthesis_backend.execute(
        model, fpga_part=platform.part_number, clk_period_ns=10.0, output_dir=temp_output_dir
    )

    # Validate Quartus outputs
    assert model.get_metadata_prop("quartus_project_dir") is not None
    assert os.path.exists(model.get_metadata_prop("quartus_project_file"))

    # Integration flow step
    model = spec.integration_flow.execute(model, fpga_part=platform.part_number, output_dir=temp_output_dir)

    # Validate Nios II outputs
    assert model.get_metadata_prop("nios_system_dir") is not None
    assert os.path.exists(model.get_metadata_prop("qsys_script"))

    # Verify complete workflow produced all expected artifacts
    assert os.path.exists(temp_output_dir + "/quartus_proj/finn_accel.qpf")
    assert os.path.exists(temp_output_dir + "/quartus_proj/finn_accel.qsf")
    assert os.path.exists(temp_output_dir + "/nios_system/finn_system.tcl")
