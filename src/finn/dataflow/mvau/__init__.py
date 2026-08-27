# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete MVAU Kernel authoring support."""

from finn.dataflow.mvau.computation import MVAUBindingSelection, MVAUComputationProfile
from finn.dataflow.mvau.definition import (
    MVAU_COMPUTE_KERNEL,
    MVAU_COMPUTE_KERNEL_SPEC,
    MVAUComputeBinding,
    MVAUComputeKernelPaths,
    MVAUDspBlock,
    build_mvau_compute_kernel_spec,
)
from finn.dataflow.mvau.regions import (
    MVAURegionDeclaration,
    MVAUWeightInterface,
    construct_batch_interleaved_streamed_mvau_region,
    construct_mvau_compute_region,
    construct_standard_embedded_mvau_region,
    construct_standard_streamed_mvau_region,
    construct_streamed_weight_mvau_region,
)

__all__ = [
    "MVAU_COMPUTE_KERNEL",
    "MVAU_COMPUTE_KERNEL_SPEC",
    "MVAUBindingSelection",
    "MVAUComputationProfile",
    "MVAUComputeBinding",
    "MVAUComputeKernelPaths",
    "MVAUDspBlock",
    "MVAURegionDeclaration",
    "MVAUWeightInterface",
    "build_mvau_compute_kernel_spec",
    "construct_batch_interleaved_streamed_mvau_region",
    "construct_mvau_compute_region",
    "construct_standard_embedded_mvau_region",
    "construct_standard_streamed_mvau_region",
    "construct_streamed_weight_mvau_region",
]
