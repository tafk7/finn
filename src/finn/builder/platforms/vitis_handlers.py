############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Vitis-based build handlers for FINN platform abstraction.

This module provides handler functions for Vitis HLS synthesis and
Alveo XRT integration flows.
"""

import json
import os
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from shutil import copy

from finn.builder.platforms.core import validate_handler_kwargs
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res


def vitis_hls_handler(model: ModelWrapper, **kwargs) -> ModelWrapper:
    """Generate Vitis HLS IP blocks from FINN hardware layers.

    This handler wraps the core generation transformations for Vitis HLS,
    preparing IP blocks and running HLS synthesis.

    Required kwargs:
        fpga_part (str): Target FPGA part number (e.g., "xcu250-figd2104-2L-e")
        clk_period_ns (float): Target clock period in nanoseconds

    Optional kwargs:
        None currently

    Args:
        model: FINN ModelWrapper to transform
        **kwargs: Configuration parameters

    Returns:
        Transformed ModelWrapper with synthesized HLS IP

    Raises:
        KeyError: If required kwargs are missing

    Example:
        >>> model = vitis_hls_handler(
        ...     model,
        ...     fpga_part="xcu250-figd2104-2L-e",
        ...     clk_period_ns=3.3
        ... )
    """
    # Validate required kwargs
    validate_handler_kwargs(
        "vitis_hls_handler",
        kwargs,
        ["fpga_part", "clk_period_ns"]
    )

    fpga_part = kwargs["fpga_part"]
    clk_period_ns = kwargs["clk_period_ns"]

    # Prepare IP (generate HLS code)
    model = model.transform(PrepareIP(fpga_part, clk_period_ns))

    # Synthesize HLS IP blocks
    model = model.transform(HLSSynthIP())

    # Fix relative paths in generated Verilog
    model = model.transform(ReplaceVerilogRelPaths())

    return model


def alveo_xrt_handler(model: ModelWrapper, **kwargs) -> ModelWrapper:
    """Integrate FINN accelerator with Alveo XRT runtime.

    This handler wraps the Vitis/Alveo build flow, creating a complete
    xclbin binary with XRT integration and AXI interfaces.

    Required kwargs:
        fpga_part (str): Target FPGA part number (e.g., "xcu250-figd2104-2L-e")
        clk_period_ns (float): Target clock period in nanoseconds
        vitis_platform (str): Vitis platform file path
        output_dir (str): Base output directory for build artifacts

    Optional kwargs:
        vitis_opt_strategy (str): Vitis optimization strategy (default: "default")
        enable_hw_debug (bool): Enable ILA debug cores (default: False)
        floorplan_file (str): Floorplan TCL file path (default: None)
        partition_model_dir (str): Directory for kernel partitions (default: None)

    Args:
        model: FINN ModelWrapper to transform
        **kwargs: Configuration parameters

    Returns:
        Transformed ModelWrapper with Alveo XRT integration metadata

    Raises:
        KeyError: If required kwargs are missing

    Example:
        >>> model = alveo_xrt_handler(
        ...     model,
        ...     fpga_part="xcu250-figd2104-2L-e",
        ...     clk_period_ns=3.3,
        ...     vitis_platform="/path/to/platform.xpfm",
        ...     output_dir="/tmp/build",
        ...     vitis_opt_strategy="Performance_ExploreWithRemap"
        ... )
    """
    # Validate required kwargs
    validate_handler_kwargs(
        "alveo_xrt_handler",
        kwargs,
        ["fpga_part", "clk_period_ns", "vitis_platform", "output_dir"]
    )

    fpga_part = kwargs["fpga_part"]
    clk_period_ns = kwargs["clk_period_ns"]
    vitis_platform = kwargs["vitis_platform"]
    output_dir = kwargs["output_dir"]
    vitis_opt_strategy = kwargs.get("vitis_opt_strategy", "default")
    enable_hw_debug = kwargs.get("enable_hw_debug", False)
    floorplan_file = kwargs.get("floorplan_file", None)
    partition_model_dir = kwargs.get("partition_model_dir", None)

    # Run Vitis build
    model = model.transform(
        VitisBuild(
            fpga_part,
            clk_period_ns,
            vitis_platform,
            strategy=vitis_opt_strategy,
            enable_debug=enable_hw_debug,
            floorplan_file=floorplan_file,
            partition_model_dir=partition_model_dir,
        )
    )

    # Copy output artifacts
    output_path = Path(output_dir)
    bitfile_dir = output_path / "bitfile"
    report_dir = output_path / "report"
    bitfile_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    copy(model.get_metadata_prop("bitfile"), bitfile_dir / "finn-accel.xclbin")
    copy(
        model.get_metadata_prop("vivado_synth_rpt"),
        report_dir / "post_synth_resources.xml",
    )

    # Generate post-synthesis resource report
    post_synth_resources = model.analysis(post_synth_res)
    with open(report_dir / "post_synth_resources.json", "w") as f:
        json.dump(post_synth_resources, f, indent=2)

    print(f"Bitfile written into {bitfile_dir}")

    return model
