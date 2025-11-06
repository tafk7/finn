# Copyright (c) 2020, Xilinx
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

# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for finn.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock

from finn.builder.platforms import Platform, SynthesisBackend, IntegrationFlow
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType


# =============================================================================
# Platform Abstraction Fixtures
# =============================================================================


@pytest.fixture
def pynq_z1_platform():
    """Create a Pynq-Z1 platform instance for testing."""
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
        attributes={"board_name": "Pynq-Z1"}
    )


@pytest.fixture
def zcu104_platform():
    """Create a ZCU104 platform instance for testing."""
    return Platform(
        part_number="xczu7ev-ffvc1156-2-e",
        vendor="amd",
        family="zynq-ultrascale-plus",
        has_ps=True,
        resources=[{
            "lut": 230400,
            "ff": 460800,
            "bram_36k": 312,
            "uram": 96,
            "dsp": 1728
        }],
        attributes={"board_name": "ZCU104"}
    )


@pytest.fixture
def alveo_u250_platform():
    """Create an Alveo U250 platform instance for testing."""
    return Platform(
        part_number="xcu250-figd2104-2L-e",
        vendor="amd",
        family="ultrascale-plus",
        has_ps=False,
        resources=[
            {"lut": 1341440, "ff": 2682880, "bram_36k": 2688, "uram": 1280, "dsp": 12288},  # SLR0
            {"lut": 1326720, "ff": 2653440, "bram_36k": 1344, "uram": 640, "dsp": 6144},    # SLR1
            {"lut": 1326720, "ff": 2653440, "bram_36k": 1344, "uram": 640, "dsp": 6144},    # SLR2
            {"lut": 1341440, "ff": 2682880, "bram_36k": 2688, "uram": 1280, "dsp": 12288},  # SLR3
        ],
        attributes={"slr_count": 4}
    )


@pytest.fixture
def vivado_hls_method():
    """Create a Vivado HLS synthesis backend for testing."""
    return SynthesisBackend(
        name="vivado_hls",
        handler=lambda model, **kwargs: model,  # Mock handler
        compatibility_check=lambda p: p.vendor.lower() in ["xilinx", "amd"]
    )


@pytest.fixture
def zynq_ps_flow():
    """Create a Zynq PS integration flow for testing."""
    return IntegrationFlow(
        name="zynq_ps",
        handler=lambda model, **kwargs: model,  # Mock handler
        compatibility_check=lambda p: getattr(p, "has_ps", False)
    )


@pytest.fixture
def alveo_xrt_flow():
    """Create an Alveo XRT integration flow for testing."""
    return IntegrationFlow(
        name="alveo_xrt",
        handler=lambda model, **kwargs: model,  # Mock handler
        compatibility_check=lambda p: p.part_number.startswith("xcu")
    )


@pytest.fixture
def temp_build_dir():
    """Create a temporary build directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="finn_test_build_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_model_wrapper():
    """Create a mock ModelWrapper for testing."""
    model = Mock()
    model.get_metadata_prop = Mock(return_value=None)
    model.set_metadata_prop = Mock()
    model.transform = Mock(return_value=model)
    model.graph = Mock()
    model.model = Mock()
    return model


@pytest.fixture
def zynq_build_config(temp_build_dir):
    """Create a Zynq build configuration for testing."""
    return DataflowBuildConfig(
        output_dir=temp_build_dir,
        board="Pynq-Z1",
        synth_clk_period_ns=10.0,
        generate_outputs=[DataflowOutputType.BITFILE],
    )


@pytest.fixture
def alveo_build_config(temp_build_dir):
    """Create an Alveo build configuration for testing."""
    return DataflowBuildConfig(
        output_dir=temp_build_dir,
        fpga_part="xcu250-figd2104-2L-e",
        synth_clk_period_ns=3.0,
        vitis_platform="xilinx_u250_gen3x16_xdma_3_1_202020_1",
        generate_outputs=[DataflowOutputType.BITFILE],
    )
