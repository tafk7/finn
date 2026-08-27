# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility facade for the MVAU compute Kernel authoring modules."""

from finn.dataflow.mvau.computation import MVAUBindingSelection, MVAUComputationProfile
from finn.dataflow.mvau.definition import (
    MVAU_COMPUTE_KERNEL,
    MVAU_COMPUTE_KERNEL_SPEC,
    MVAU_DESIGN_SPACE_SPEC,
    MVAUComputeBinding,
    MVAUComputeKernelPaths,
    MVAUDesignPaths,
    MVAUDspBlock,
    build_mvau_compute_kernel_spec,
    build_legacy_mvau_design_space_spec,
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
    "MVAU_DESIGN_SPACE_SPEC",
    "MVAUBindingSelection",
    "MVAUComputationProfile",
    "MVAUComputeBinding",
    "MVAUComputeKernelPaths",
    "MVAUDesignPaths",
    "MVAUDspBlock",
    "MVAURegionDeclaration",
    "MVAUWeightInterface",
    "build_legacy_mvau_design_space_spec",
    "build_mvau_compute_kernel_spec",
    "construct_batch_interleaved_streamed_mvau_region",
    "construct_mvau_compute_region",
    "construct_standard_embedded_mvau_region",
    "construct_standard_streamed_mvau_region",
    "construct_streamed_weight_mvau_region",
]
