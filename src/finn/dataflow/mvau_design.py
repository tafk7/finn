# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility facade for the MVAU Kernel authoring modules."""

from finn.dataflow.mvau import (
    BATCH_INTERLEAVED_DSP_MVAU_KERNEL,
    LEGACY_HLS_MVAU_KERNEL,
    MVAU_COMPUTE_SELECTION,
    MVAU_DESIGN_SPACE_SPEC,
    PACKED_DSP_MVAU_KERNEL,
    SOFT_VECTOR_MVAU_KERNEL,
    MVAUComputationProfile,
    MVAUComputeKernelId,
    MVAUComputeProblemPaths,
    MVAUDesignPaths,
    MVAUDspBlock,
    MVAURegionDeclaration,
    MVAUWeightInterface,
    build_legacy_mvau_design_space_spec,
    construct_batch_interleaved_streamed_mvau_region,
    construct_mvau_compute_region,
    construct_standard_embedded_mvau_region,
    construct_standard_streamed_mvau_region,
    construct_streamed_weight_mvau_region,
)

__all__ = [
    "BATCH_INTERLEAVED_DSP_MVAU_KERNEL",
    "LEGACY_HLS_MVAU_KERNEL",
    "MVAU_COMPUTE_SELECTION",
    "MVAU_DESIGN_SPACE_SPEC",
    "MVAUComputationProfile",
    "MVAUComputeKernelId",
    "MVAUComputeProblemPaths",
    "MVAUDesignPaths",
    "MVAUDspBlock",
    "MVAURegionDeclaration",
    "MVAUWeightInterface",
    "PACKED_DSP_MVAU_KERNEL",
    "SOFT_VECTOR_MVAU_KERNEL",
    "build_legacy_mvau_design_space_spec",
    "construct_batch_interleaved_streamed_mvau_region",
    "construct_mvau_compute_region",
    "construct_standard_embedded_mvau_region",
    "construct_standard_streamed_mvau_region",
    "construct_streamed_weight_mvau_region",
]
