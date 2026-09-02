# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""MVAU source-to-semantics association values.

These values are shared by the legacy Operation assembly and the new
``DataflowDesign`` path.  Keeping them in a leaf module lets both declarations
produce and compare the same values without either importing the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from finn.dataflow.resolution import ResolvedDataflowOp


class MVAUParameterTopology(str, Enum):
    """The resolved shape of one parameter-supply arrangement."""

    EMBEDDED = "embedded"
    DIRECT = "direct"
    CYCLIC = "cyclic"


class CoordinateMappingKind(str, Enum):
    """Explicit source-to-region coordinate transformations used by MVAU."""

    FLATTEN_LEADING = "flatten_leading"
    TRANSPOSE_2D = "transpose_2d"


@dataclass(frozen=True)
class SemanticOperandDestination:
    """Qualified operand destination in a selected Region or Network node."""

    owner_id: str
    operand_id: str


@dataclass(frozen=True)
class SourceOperandAssociation:
    """Explicit association from one source operand to a semantic target."""

    role: str
    source_operand_id: str
    destination: SemanticOperandDestination
    mapping: CoordinateMappingKind
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]

    def map_position(self, position: tuple[int, ...]) -> tuple[int, ...]:
        if len(position) != len(self.source_shape) or any(
            index < 0 or index >= extent for index, extent in zip(position, self.source_shape)
        ):
            raise ValueError("source position is outside source shape")
        if self.mapping is CoordinateMappingKind.FLATTEN_LEADING:
            leading = position[:-1]
            flattened = 0
            for index, extent in zip(leading, self.source_shape[:-1]):
                flattened = flattened * extent + index
            return (flattened, position[-1])
        if self.mapping is CoordinateMappingKind.TRANSPOSE_2D:
            if len(position) != 2:
                raise ValueError("transpose mapping requires a rank-two position")
            return (position[1], position[0])
        return position


@dataclass(frozen=True)
class MVAUSourceAssociation:
    """Source provenance and operand mappings for one selected MVAU result."""

    source_node_id: str
    fused_source_node_ids: tuple[str, ...]
    region_declaration_id: str
    parameter_topology: MVAUParameterTopology
    operands: tuple[SourceOperandAssociation, ...]


MVAUResolvedDataflowOp: TypeAlias = ResolvedDataflowOp[MVAUSourceAssociation]


__all__ = [
    "CoordinateMappingKind",
    "MVAUParameterTopology",
    "MVAUResolvedDataflowOp",
    "MVAUSourceAssociation",
    "SemanticOperandDestination",
    "SourceOperandAssociation",
]
