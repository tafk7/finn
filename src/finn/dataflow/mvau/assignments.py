# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable MVAU decision-to-node-attribute codec inventory."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import TypeVar, cast

from finn.dataflow.design import QualifiedPath
from finn.dataflow.mvau.definition import MVAUComputeBinding, MVAUComputeKernelPaths
from finn.dataflow.mvau.regions import MVAURegionDeclaration
from finn.dataflow.op import NodeAttrCodec
from finn.dataflow.ops.mvau import (
    MVAUConnectionTopology,
    MVAUDataflowOpPaths,
    MVAUParameterTopology,
    MVAUWeightDeliveryDeclaration,
)
from finn.dataflow.parameters.cyclic.definition import (
    CyclicParameterBinding,
    CyclicParameterKernelPaths,
    CyclicRamStyle,
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
        MVAUComputeKernelPaths.PE: NodeAttrCodec.integer("dataflow_compute_pe"),
        MVAUComputeKernelPaths.SIMD: NodeAttrCodec.integer("dataflow_compute_simd"),
        MVAUComputeKernelPaths.REGION_DECLARATION: _enum_codec(
            "dataflow_compute_region", MVAURegionDeclaration
        ),
        MVAUComputeKernelPaths.INTERLEAVE: NodeAttrCodec.integer("dataflow_compute_interleave"),
        MVAUComputeKernelPaths.BINDING: _enum_codec("dataflow_compute_binding", MVAUComputeBinding),
        MVAUComputeKernelPaths.COMPUTE_PUMPING: NodeAttrCodec.boolean("dataflow_compute_pumping"),
        MVAUDataflowOpPaths.PARAMETER_TOPOLOGY: _enum_codec(
            "dataflow_parameter_topology", MVAUParameterTopology
        ),
        MVAUDataflowOpPaths.DELIVERY_PE: NodeAttrCodec.integer("dataflow_delivery_pe"),
        MVAUDataflowOpPaths.DELIVERY_SIMD: NodeAttrCodec.integer("dataflow_delivery_simd"),
        MVAUDataflowOpPaths.DELIVERY_DECLARATION: _enum_codec(
            "dataflow_delivery_region", MVAUWeightDeliveryDeclaration
        ),
        MVAUDataflowOpPaths.DELIVERY_INTERLEAVE: NodeAttrCodec.integer(
            "dataflow_delivery_interleave"
        ),
        MVAUDataflowOpPaths.CONNECTION_TOPOLOGY: _enum_codec(
            "dataflow_connection_topology", MVAUConnectionTopology
        ),
        CyclicParameterKernelPaths.BINDING: _enum_codec(
            "dataflow_delivery_binding", CyclicParameterBinding
        ),
        CyclicParameterKernelPaths.RAM_STYLE: _enum_codec(
            "dataflow_delivery_ram_style", CyclicRamStyle
        ),
        CyclicParameterKernelPaths.PUMPED_MEMORY: NodeAttrCodec.boolean(
            "dataflow_delivery_pumping"
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
