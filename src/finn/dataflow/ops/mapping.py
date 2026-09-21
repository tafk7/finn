# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-owned source correspondence, derived over an accepted Network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from finn.dataflow.model.logical.maps import (
    AffineRankMap,
    CoordinateMap,
    CoordinateSet,
    IdentityCoordinateMap,
    ExplicitCoordinateMap,
    MaterializationRequired,
    RectangularDomain,
)
from finn.dataflow.model.logical.network import DataflowNetwork
from finn.dataflow.model.logical.network_validation import validate_network
from finn.dataflow.model.logical.presentation import (
    boundary_presented_position_set,
    edge_presented_position_set,
    exposing_boundaries,
    exposing_ports,
    unpresented_position_set,
)
from finn.dataflow.model.logical.refs import (
    DataflowOperandRef,
    NetworkOperandError,
    RegionInputRef,
    resolve_input,
    resolve_output,
)
from finn.dataflow.model.logical.region import Coordinate, InputInterface
from finn.dataflow.ops.source import SourceNode
from finn.dataflow.model.logical.interface import OperandExport, validate_operand_export


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
    coordinate_map: CoordinateMap
    edge_presented_set: CoordinateSet
    boundary_presented_set: CoordinateSet
    unpresented_set: CoordinateSet
    _presentation_is_explicit: bool = False

    @property
    def edge_presented(self) -> frozenset[Coordinate]:
        return self._legacy_set(self.edge_presented_set, "edge_presented")

    @property
    def boundary_presented(self) -> frozenset[Coordinate]:
        return self._legacy_set(self.boundary_presented_set, "boundary_presented")

    @property
    def unpresented(self) -> frozenset[Coordinate]:
        return self._legacy_set(self.unpresented_set, "unpresented")

    def _legacy_set(self, value: CoordinateSet, field_name: str) -> frozenset[Coordinate]:
        if not self._presentation_is_explicit:
            raise MaterializationRequired(
                f"OperandMapping.{field_name} requires materialize_presentation("
                "max_positions_per_set=...)"
            )
        return frozenset(value.materialize(max_points=value.cardinality))

    def materialize_presentation(
        self, *, max_positions_per_set: int
    ) -> MaterializedOperandPresentation:
        return MaterializedOperandPresentation(
            frozenset(self.edge_presented_set.materialize(max_points=max_positions_per_set)),
            frozenset(self.boundary_presented_set.materialize(max_points=max_positions_per_set)),
            frozenset(self.unpresented_set.materialize(max_points=max_positions_per_set)),
        )


@dataclass(frozen=True, slots=True)
class MaterializedOperandPresentation:
    edge_presented: frozenset[Coordinate]
    boundary_presented: frozenset[Coordinate]
    unpresented: frozenset[Coordinate]


def _checked_coordinate_map(
    correspondence: CoordinateMapping,
    source_shape: tuple[int, ...],
    semantic_shape: tuple[int, ...],
) -> CoordinateMap:
    source = RectangularDomain(source_shape)
    target = RectangularDomain(semantic_shape)
    if correspondence is CoordinateMapping.IDENTITY:
        if source != target:
            raise NetworkOperandError(
                f"identity correspondence requires equal shapes, got {source_shape} and "
                f"{semantic_shape}"
            )
        return IdentityCoordinateMap(CoordinateSet.full(source), target)
    if correspondence is CoordinateMapping.FLATTEN_LEADING:
        if not source_shape:
            raise NetworkOperandError("flatten-leading correspondence requires rank at least one")
        rows = 1
        for extent in source_shape[:-1]:
            rows *= extent
        expected = (rows, source_shape[-1])
        if semantic_shape != expected:
            raise NetworkOperandError(
                "flatten-leading correspondence requires semantic shape "
                f"{expected}, got {semantic_shape}"
            )
        return AffineRankMap.row_major_reshape(source, target)
    if len(source_shape) != 2 or semantic_shape != tuple(reversed(source_shape)):
        raise NetworkOperandError(
            "two-dimensional transpose correspondence requires reversed rank-two shapes, "
            f"got {source_shape} and {semantic_shape}"
        )
    rows, columns = source_shape
    return AffineRankMap.from_mixed_radix(
        source,
        view_extents=(rows, columns),
        target=target,
        offset=0,
        coefficients=(1, rows),
    )


def derive_operand_mappings(
    network: DataflowNetwork,
    source: SourceNode,
    references: Mapping[str, tuple[DataflowOperandRef, ...]],
    correspondences: Mapping[str, CoordinateMapping],
) -> tuple[OperandMapping, ...]:
    """Direct/synthetic entry point: validate once, then derive every mapping.

    DataflowOp uses its accepted Kernel projection and calls the private
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
    coordinate_maps: Mapping[tuple[str, DataflowOperandRef], CoordinateMap] | None = None,
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
            if isinstance(ref, RegionInputRef):
                item = resolve_input(network, ref)
                shape = item.operand.shape
                edge_presented = edge_presented_position_set(network, ref)
                boundary_presented = boundary_presented_position_set(network, ref)
                unpresented = unpresented_position_set(network, ref)
                presentation_is_explicit = item.requirements.is_explicit and (
                    not isinstance(item, InputInterface) or item.port.beat_sequence.is_explicit
                )
            else:
                output = resolve_output(network, ref)
                shape = output.port.operand.shape
                empty = CoordinateSet.empty(output.port.operand.position_domain)
                edge_presented = empty
                boundary_presented = empty
                unpresented = empty
                presentation_is_explicit = True
            result.append(
                OperandMapping(
                    name,
                    operand.tensor,
                    ref,
                    placement,
                    correspondences[name],
                    operand.shape,
                    tuple(shape),
                    coordinate_maps[(name, ref)]
                    if coordinate_maps is not None
                    else _checked_coordinate_map(
                        correspondences[name], operand.shape, tuple(shape)
                    ),
                    edge_presented,
                    boundary_presented,
                    unpresented,
                    presentation_is_explicit,
                )
            )
    return tuple(result)


def _compose_maps(first: CoordinateMap, second: CoordinateMap) -> CoordinateMap:
    """Compose the common compact boundary views without enumerating tensors."""

    if isinstance(first, IdentityCoordinateMap):
        return second
    if isinstance(second, IdentityCoordinateMap):
        return first
    if isinstance(first, AffineRankMap) and first.offset == 0 and first.coefficients == (1,):
        if isinstance(second, AffineRankMap):
            return AffineRankMap(
                first.source, second.view_extents, second.target, second.offset, second.coefficients
            )
    source = first.source if isinstance(first, AffineRankMap) else first.source_domain
    target = second.target if isinstance(second, AffineRankMap) else second.target_domain
    if source is None or target is None:
        raise NetworkOperandError("composed maps require explicit coordinate domains")
    if source.cardinality > 1_000_000:
        raise MaterializationRequired("this boundary map composition needs an explicit size budget")
    return ExplicitCoordinateMap(
        (
            (source.coordinate_at(rank), second.mapped(first.mapped(source.coordinate_at(rank))))
            for rank in range(source.cardinality)
        ),
        source_domain=source,
        target_domain=target,
    )


def derive_public_operand_mappings(
    network: DataflowNetwork,
    source: SourceNode,
    exports: Mapping[str, OperandExport],
    correspondences: Mapping[str, CoordinateMapping],
) -> tuple[OperandMapping, ...]:
    """Bind source values through checked public exports to the actual body.

    The caller supplies an accepted logical Network. Kernel export maps own body
    organization; only the first map adapts the graph coordinate convention.
    """

    expected = {operand.id for operand in (*source.inputs, *source.outputs)}
    if set(exports) != expected:
        raise NetworkOperandError("public bindings must cover every present graph operand")
    references: dict[str, tuple[DataflowOperandRef, ...]] = {}
    maps: dict[tuple[str, DataflowOperandRef], CoordinateMap] = {}
    for name, export in exports.items():
        validate_operand_export(network, export)
        operand = source.operand(name)
        if operand.datatype != export.operand.element_type:
            raise NetworkOperandError(f"{name!r} boundary binding cannot convert datatype")
        if (operand in source.inputs) != (export.operand.direction == "input"):
            raise NetworkOperandError(f"{name!r} boundary binding has the wrong direction")
        boundary = _checked_coordinate_map(
            correspondences[name], operand.shape, export.operand.domain.extents
        )
        references[name] = tuple(target.ref for target in export.targets)
        for target in export.targets:
            if len(target.presentations) > 1:
                raise NetworkOperandError("stream mapping needs an explicit presentation selection")
            maps[name, target.ref] = _compose_maps(boundary, target.position_map.coordinate_map)
    return _derive_operand_mappings(network, source, references, correspondences, maps)


__all__ = [
    "CoordinateMapping",
    "External",
    "Internal",
    "InternalStream",
    "MaterializedOperandPresentation",
    "OperandMapping",
    "OperandPlacement",
    "derive_operand_mappings",
    "derive_public_operand_mappings",
]
