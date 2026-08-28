# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete MVAU Kernel authoring support."""

from finn.dataflow.mvau.computation import MVAUComputationProfile
from finn.dataflow.mvau.compute_kernels import (
    BATCH_INTERLEAVED_DSP_MVAU_KERNEL,
    LEGACY_HLS_MVAU_KERNEL,
    MVAU_COMPUTE_SELECTION,
    PACKED_DSP_MVAU_KERNEL,
    SOFT_VECTOR_MVAU_KERNEL,
    MVAUComputeKernelId,
    MVAUComputeKernelPathSet,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.mvau.legacy_design import (
    MVAU_DESIGN_SPACE_SPEC,
    MVAUDesignPaths,
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
from finn.dataflow.mvau.weight_adapter import (
    construct_weight_sequence_adapter_region,
    weight_sequence_adapter_applicable,
)
from finn.dataflow.mvau.weight_adapter_kernel import (
    FULL_TILE_TO_CHUNKED,
    MVAU_WEIGHT_ADAPTER_SELECTION_NAME,
    build_mvau_weight_adapter_kernel,
    build_mvau_weight_adapter_selection,
)

__all__ = [
    "BATCH_INTERLEAVED_DSP_MVAU_KERNEL",
    "FULL_TILE_TO_CHUNKED",
    "LEGACY_HLS_MVAU_KERNEL",
    "MVAU_COMPUTE_SELECTION",
    "MVAU_DESIGN_SPACE_SPEC",
    "MVAU_WEIGHT_ADAPTER_SELECTION_NAME",
    "MVAUComputationProfile",
    "MVAUComputeKernelId",
    "MVAUComputeKernelPathSet",
    "MVAUDesignPaths",
    "MVAUHlsResource",
    "MVAURegionDeclaration",
    "MVAUWeightInterface",
    "MVAUWeightSource",
    "PACKED_DSP_MVAU_KERNEL",
    "SOFT_VECTOR_MVAU_KERNEL",
    "build_legacy_mvau_design_space_spec",
    "build_mvau_weight_adapter_kernel",
    "build_mvau_weight_adapter_selection",
    "construct_batch_interleaved_streamed_mvau_region",
    "construct_mvau_compute_region",
    "construct_standard_embedded_mvau_region",
    "construct_standard_streamed_mvau_region",
    "construct_streamed_weight_mvau_region",
    "construct_weight_sequence_adapter_region",
    "weight_sequence_adapter_applicable",
]
