# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The builder's kernel-path configuration for the tests."""

from __future__ import annotations

from pathlib import Path

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.platform.request import TargetRequest
from support.models import TARGET


def config(directory: Path) -> DataflowBuildConfig:
    """The kernel path for ``TARGET`` (Ultra96's part at 5 ns on the ``ip`` shell), its
    choices the default exploration's (the baselines, FIFOs sized: ``[size_fifos]``)."""
    return DataflowBuildConfig(
        output_dir=str(directory),
        target=TargetRequest(part=TARGET.part, period_ns=TARGET.platform.period_ns),
        steps=["phase_convert_to_hardware", "phase_create_accelerator", "phase_optimize_hardware"],
        enable_build_pdb_debug=False,
    )
