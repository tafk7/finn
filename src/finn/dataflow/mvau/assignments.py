# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable MVAU decision-to-node-attribute codec inventory.

Every persistent choice is either one Kernel-pool identity or one decision a
selected Kernel owns.  Nothing here persists a Region form, a binding, or a
delivery topology: those are derived from the selected Kernels.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import TypeVar, cast

from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau.compute_kernels import (
    BATCH_INTERLEAVED_PATHS,
    LEGACY_HLS_PATHS,
    PACKED_DSP_PATHS,
    SOFT_VECTOR_PATHS,
    DECOMPOSED_MVAU_KERNELS,
    MVAU_REPLAY_SELECTION,
    MVAUHlsResource,
    MVAUWeightSource,
)
from finn.dataflow.op import NodeAttrCodec
from finn.dataflow.ops.mvau import (
    MVAU_COMPUTE_SELECTION,
    MVAU_WEIGHT_ADAPTER_SELECTION,
    MVAU_WEIGHT_SUPPLY_SELECTION,
)
from finn.dataflow.parameters.supply_kernels import (
    FINNLIB_MEMSTREAM_PATHS,
    FINN_RTL_MEMSTREAM_PATHS,
    CyclicRamStyle,
    WeightOrganization,
)

E = TypeVar("E", bound=Enum)


def _enum_codec(attribute_name: str, enum_type: type[E]) -> NodeAttrCodec:
    return NodeAttrCodec.finite_enum(
        attribute_name,
        cast(type[Enum], enum_type),
        {cast(str, value.value): value for value in enum_type},
    )


MVAU_DECISION_NODEATTRS: Mapping[QualifiedPath, NodeAttrCodec] = MappingProxyType(
    {
        # Kernel-pool identities.
        MVAU_COMPUTE_SELECTION.paths.kernel: NodeAttrCodec.string("dataflow_compute_kernel"),
        MVAU_WEIGHT_SUPPLY_SELECTION.paths.kernel: NodeAttrCodec.string(
            "dataflow_weight_supply_kernel"
        ),
        MVAU_WEIGHT_ADAPTER_SELECTION.paths.kernel: NodeAttrCodec.string(
            "dataflow_weight_adapter_kernel"
        ),
        # Legacy HLS compute Kernel.
        LEGACY_HLS_PATHS.pe: NodeAttrCodec.integer("dataflow_legacy_hls_pe"),
        LEGACY_HLS_PATHS.simd: NodeAttrCodec.integer("dataflow_legacy_hls_simd"),
        LEGACY_HLS_PATHS.resource: _enum_codec("dataflow_legacy_hls_resource", MVAUHlsResource),
        LEGACY_HLS_PATHS.weight_source: _enum_codec(
            "dataflow_legacy_hls_weight_source", MVAUWeightSource
        ),
        # Soft-vector compute Kernel.
        SOFT_VECTOR_PATHS.pe: NodeAttrCodec.integer("dataflow_softvec_pe"),
        SOFT_VECTOR_PATHS.simd: NodeAttrCodec.integer("dataflow_softvec_simd"),
        SOFT_VECTOR_PATHS.compute_pumping: NodeAttrCodec.boolean("dataflow_softvec_pumping"),
        # Packed-DSP compute Kernel.
        PACKED_DSP_PATHS.pe: NodeAttrCodec.integer("dataflow_packed_pe"),
        PACKED_DSP_PATHS.simd: NodeAttrCodec.integer("dataflow_packed_simd"),
        PACKED_DSP_PATHS.compute_pumping: NodeAttrCodec.boolean("dataflow_packed_pumping"),
        # Batch-interleaved DSP compute Kernel.
        BATCH_INTERLEAVED_PATHS.pe: NodeAttrCodec.integer("dataflow_interleaved_pe"),
        BATCH_INTERLEAVED_PATHS.simd: NodeAttrCodec.integer("dataflow_interleaved_simd"),
        BATCH_INTERLEAVED_PATHS.interleave: NodeAttrCodec.integer("dataflow_interleaved_batch"),
        # Decomposed compute Kernel.  The replay half owns no choices; it is
        # told the folding, so there is nothing of its own to persist beyond
        # which Kernel was selected.
        DECOMPOSED_MVAU_KERNELS.pe.path: NodeAttrCodec.integer("dataflow_dot_product_pe"),
        DECOMPOSED_MVAU_KERNELS.simd.path: NodeAttrCodec.integer("dataflow_dot_product_simd"),
        DECOMPOSED_MVAU_KERNELS.compute_pumping.path: NodeAttrCodec.boolean(
            "dataflow_dot_product_pumping"
        ),
        MVAU_REPLAY_SELECTION.paths.kernel: NodeAttrCodec.string("dataflow_replay_kernel"),
        # FINN RTL memstream supply Kernel.
        FINN_RTL_MEMSTREAM_PATHS.organization: _enum_codec(
            "dataflow_finn_rtl_memstream_organization", WeightOrganization
        ),
        FINN_RTL_MEMSTREAM_PATHS.ram_style: _enum_codec(
            "dataflow_finn_rtl_memstream_ram_style", CyclicRamStyle
        ),
        FINN_RTL_MEMSTREAM_PATHS.pumped_memory: NodeAttrCodec.boolean(
            "dataflow_finn_rtl_memstream_pumping"
        ),
        # FinnLib memstream supply Kernel.
        FINNLIB_MEMSTREAM_PATHS.organization: _enum_codec(
            "dataflow_finnlib_memstream_organization", WeightOrganization
        ),
    }
)


def local_mvau_assignment_path(path: QualifiedPath) -> QualifiedPath | None:
    """Resolve an unqualified or placed MVAU decision path to its local path."""

    matches = tuple(
        candidate
        for candidate in MVAU_DECISION_NODEATTRS
        if path == candidate or path.value.endswith(f".{candidate.value}")
    )
    return matches[0] if len(matches) == 1 else None


__all__ = ["MVAU_DECISION_NODEATTRS", "local_mvau_assignment_path"]
