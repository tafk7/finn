# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh v6 MVAU decision-to-node-attribute codec inventory."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import TypeVar, cast

from finn.dataflow.design import QualifiedPath
from finn.dataflow.ops.mvau.inventory import MVAU_DESIGN_INVENTORY
from finn.dataflow.op_contracts import NodeAttrCodec
from finn.dataflow.parameters.cyclic.definition import CyclicRamStyle

E = TypeVar("E", bound=Enum)


def _enum_codec(attribute_name: str, enum_type: type[E]) -> NodeAttrCodec:
    return NodeAttrCodec.finite_enum(
        attribute_name,
        cast(type[Enum], enum_type),
        {cast(str, value.value): value for value in enum_type},
    )


_DESIGN_PATH = MVAU_DESIGN_INVENTORY.inventory.design_path
if _DESIGN_PATH is None:
    raise AssertionError("the v6 MVAU inventory must declare a design decision")

MVAU_DECISION_NODEATTRS: Mapping[QualifiedPath, NodeAttrCodec] = MappingProxyType(
    {
        _DESIGN_PATH: NodeAttrCodec.string("dataflow_design"),
        MVAU_DESIGN_INVENTORY.dot_product.pe.path: NodeAttrCodec.integer("dataflow_dot_product_pe"),
        MVAU_DESIGN_INVENTORY.dot_product.simd.path: NodeAttrCodec.integer(
            "dataflow_dot_product_simd"
        ),
        MVAU_DESIGN_INVENTORY.batch_interleaved.pe.path: NodeAttrCodec.integer(
            "dataflow_interleaved_pe"
        ),
        MVAU_DESIGN_INVENTORY.batch_interleaved.simd.path: NodeAttrCodec.integer(
            "dataflow_interleaved_simd"
        ),
        MVAU_DESIGN_INVENTORY.batch_interleaved.interleave.path: NodeAttrCodec.integer(
            "dataflow_interleaved_batch"
        ),
        MVAU_DESIGN_INVENTORY.input_supply.declaration.choice.path: NodeAttrCodec.string(
            "dataflow_weight_supply"
        ),
        MVAU_DESIGN_INVENTORY.compute_pumping.path: NodeAttrCodec.boolean(
            "dataflow_dotp_axi_pumping"
        ),
        MVAU_DESIGN_INVENTORY.input_supply.settings.ram_style.path: _enum_codec(
            "dataflow_finn_rtl_memstream_ram_style", CyclicRamStyle
        ),
        MVAU_DESIGN_INVENTORY.input_supply.settings.pumped_memory.path: NodeAttrCodec.boolean(
            "dataflow_finn_rtl_memstream_pumping"
        ),
    }
)


def local_mvau_assignment_path(path: QualifiedPath) -> QualifiedPath | None:
    """Resolve an exact or occurrence-prefixed v6 MVAU decision path."""

    matches = tuple(
        candidate
        for candidate in MVAU_DECISION_NODEATTRS
        if path == candidate or path.value.endswith(f".{candidate.value}")
    )
    return matches[0] if len(matches) == 1 else None


__all__ = ["MVAU_DECISION_NODEATTRS", "local_mvau_assignment_path"]
