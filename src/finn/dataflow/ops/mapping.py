# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-owned source correspondence, derived over an accepted Network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from finn.dataflow.model.network import DataflowNetwork
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.presentation import (
    boundary_presented_positions,
    edge_presented_positions,
    exposing_boundaries,
    exposing_ports,
    unpresented_positions,
)
from finn.dataflow.model.refs import (
    DataflowOperandRef,
    NetworkOperandError,
    RegionInputRef,
    resolve_input,
    resolve_output,
)
from finn.dataflow.model.region import Coordinate
from finn.dataflow.ops.source import SourceNode


class CoordinateMapping(str, Enum):
    IDENTITY = "identity"
    FLATTEN_LEADING = "flatten_leading"
    TRANSPOSE_2D = "transpose_2d"


@dataclass(frozen=True, slots=True)
class External:
    boundary_id: str
    node_id: str
    port_id: str


@dataclass(frozen=True, slots=True)
class InternalStream:
    node_id: str
    port_id: str


@dataclass(frozen=True, slots=True)
class Internal:
    """A Region input with no port; makes no physical storage claim."""

    node_id: str
    operand_id: str


OperandPlacement = External | InternalStream | Internal


@dataclass(frozen=True, slots=True)
class OperandMapping:
    source_operand: str
    tensor: str
    semantic_operand: DataflowOperandRef
    placement: OperandPlacement
    correspondence: CoordinateMapping
    source_shape: tuple[int, ...]
    semantic_shape: tuple[int, ...]
    edge_presented: frozenset[Coordinate] = frozenset()
    boundary_presented: frozenset[Coordinate] = frozenset()
    unpresented: frozenset[Coordinate] = frozenset()


def derive_operand_mappings(
    network: DataflowNetwork,
    source: SourceNode,
    references: Mapping[str, tuple[DataflowOperandRef, ...]],
    correspondences: Mapping[str, CoordinateMapping],
) -> tuple[OperandMapping, ...]:
    """Direct/synthetic entry point: validate once, then derive every mapping.

    DataflowOp uses its accepted Design projection and calls the private
    derivation below without a second whole-Network validation.
    """

    if not isinstance(network, DataflowNetwork):
        raise NetworkOperandError("mapping requires a DataflowNetwork accepted by validation")
    issues = validate_network(network)
    if issues:
        raise NetworkOperandError(f"cannot map a structurally invalid Network: {issues}")
    return _derive_operand_mappings(network, source, references, correspondences)


def _derive_operand_mappings(
    network: DataflowNetwork,
    source: SourceNode,
    references: Mapping[str, tuple[DataflowOperandRef, ...]],
    correspondences: Mapping[str, CoordinateMapping],
) -> tuple[OperandMapping, ...]:
    """Requires the caller's accepted projection or explicit validation."""

    result: list[OperandMapping] = []
    for name, refs in references.items():
        operand = source.operand(name)
        for ref in refs:
            is_input = isinstance(ref, RegionInputRef)
            if is_input != (operand in source.inputs):
                raise NetworkOperandError(f"{name!r} and {ref!r} have different directions")
            ports = exposing_ports(network, ref)
            boundaries = exposing_boundaries(network, ref)
            if len(boundaries) > 1:
                raise NetworkOperandError(f"{ref!r} is exposed by several boundaries")
            placement: OperandPlacement
            if boundaries:
                boundary = boundaries[0]
                placement = External(boundary.id, ref.node_id, boundary.endpoint.port_id)
            elif ports:
                placement = InternalStream(ref.node_id, ports[0].port_id)
            else:
                placement = Internal(ref.node_id, ref.operand_id)
            shape = (
                resolve_input(network, ref).operand.shape
                if isinstance(ref, RegionInputRef)
                else resolve_output(network, ref).port.operand.shape
            )
            result.append(
                OperandMapping(
                    name,
                    operand.tensor,
                    ref,
                    placement,
                    correspondences[name],
                    operand.shape,
                    tuple(shape),
                    edge_presented_positions(network, ref)
                    if isinstance(ref, RegionInputRef)
                    else frozenset(),
                    boundary_presented_positions(network, ref)
                    if isinstance(ref, RegionInputRef)
                    else frozenset(),
                    unpresented_positions(network, ref)
                    if isinstance(ref, RegionInputRef)
                    else frozenset(),
                )
            )
    return tuple(result)


__all__ = [
    "CoordinateMapping",
    "External",
    "Internal",
    "InternalStream",
    "OperandMapping",
    "OperandPlacement",
    "derive_operand_mappings",
]
