############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Vivado-based build handlers for FINN platform abstraction.

This module provides handler functions for Vivado HLS synthesis and
Zynq PS integration flows.
"""

import json
import os
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from shutil import copy

from finn.builder.platforms.core import validate_handler_kwargs
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res


def vivado_hls_handler(model: ModelWrapper, **kwargs) -> ModelWrapper:
    """Generate Vivado HLS IP blocks from FINN hardware layers.

    This handler wraps the core generation transformations for Vivado HLS,
    preparing IP blocks and running HLS synthesis.

    Required kwargs:
        fpga_part (str): Target FPGA part number (e.g., "xc7z020clg400-1")
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
        >>> model = vivado_hls_handler(
        ...     model,
        ...     fpga_part="xc7z020clg400-1",
        ...     clk_period_ns=10.0
        ... )
    """
    # Validate required kwargs
    validate_handler_kwargs(
        "vivado_hls_handler",
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


def zynq_ps_handler(model: ModelWrapper, **kwargs) -> ModelWrapper:
    """Integrate FINN accelerator with Zynq Processing System.

    This handler wraps the Zynq PS integration flow, creating a complete
    bitfile with ARM PS integration, DMA, and AXI interfaces.

    Required kwargs:
        board (str): Target board name (e.g., "Pynq-Z1")
        clk_period_ns (float): Target clock period in nanoseconds
        output_dir (str): Base output directory for build artifacts

    Optional kwargs:
        enable_hw_debug (bool): Enable ILA debug cores (default: False)
        partition_model_dir (str): Directory for kernel partitions

    Args:
        model: FINN ModelWrapper to transform
        **kwargs: Configuration parameters

    Returns:
        Transformed ModelWrapper with Zynq PS integration metadata

    Raises:
        KeyError: If required kwargs are missing

    Example:
        >>> model = zynq_ps_handler(
        ...     model,
        ...     board="Pynq-Z1",
        ...     clk_period_ns=10.0,
        ...     output_dir="/tmp/build",
        ...     enable_hw_debug=False
        ... )
    """
    # Validate required kwargs
    validate_handler_kwargs(
        "zynq_ps_handler",
        kwargs,
        ["board", "clk_period_ns", "output_dir"]
    )

    board = kwargs["board"]
    clk_period_ns = kwargs["clk_period_ns"]
    output_dir = kwargs["output_dir"]
    enable_hw_debug = kwargs.get("enable_hw_debug", False)
    partition_model_dir = kwargs.get("partition_model_dir", None)

    # Run Zynq build
    model = model.transform(
        ZynqBuild(
            board,
            clk_period_ns,
            enable_hw_debug,
            partition_model_dir=partition_model_dir,
        )
    )

    # Copy output artifacts
    output_path = Path(output_dir)
    bitfile_dir = output_path / "bitfile"
    report_dir = output_path / "report"
    bitfile_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    copy(model.get_metadata_prop("bitfile"), bitfile_dir / "finn-accel.bit")
    copy(model.get_metadata_prop("hw_handoff"), bitfile_dir / "finn-accel.hwh")
    copy(
        model.get_metadata_prop("vivado_synth_rpt"),
        report_dir / "post_synth_resources.xml",
    )

    # Generate post-synthesis resource report
    post_synth_resources = model.analysis(post_synth_res)
    with open(report_dir / "post_synth_resources.json", "w") as f:
        json.dump(post_synth_resources, f, indent=2)

    # Copy timing report
    vivado_pynq_proj_dir = Path(model.get_metadata_prop("vivado_pynq_proj"))
    timing_rpt = (
        vivado_pynq_proj_dir
        / "finn_zynq_link.runs"
        / "impl_1"
        / "top_wrapper_timing_summary_routed.rpt"
    )
    copy(timing_rpt, report_dir / "post_route_timing.rpt")

    print(f"Bitfile written into {bitfile_dir}")

    return model
