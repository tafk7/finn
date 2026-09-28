# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Operation-owned source correspondence, derived over an accepted Network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from finn.parked.dataflow.logical_values.maps import (
    AffineRankMap,
    CoordinateMap,
    CoordinateSet,
    IdentityCoordinateMap,
    ExplicitCoordinateMap,
    MaterializationRequired,
    RectangularDomain,
)
from finn.parked.dataflow.logical_values.network import DataflowNetwork, PositionMap
from finn.parked.dataflow.logical_values.network_validation import validate_network
from finn.parked.dataflow.logical_values.presentation import (
    boundary_presented_position_set,
    edge_presented_position_set,
    exposing_boundaries,
    exposing_ports,
    unpresented_position_set,
)
from finn.parked.dataflow.logical_values.refs import (
    DataflowOperandRef,
    NetworkOperandError,
    RegionInputRef,
    resolve_input,
    resolve_output,
)
from finn.parked.dataflow.logical_values.region import Coordinate
from finn.parked.dataflow.ops.source import SourceNode
from finn.parked.dataflow.logical_values.interface import OperandExport, validate_operand_export


class CoordinateMapping(str, Enum):
    IDENTITY = "identity"
    FLATTEN_LEADING = "flatten_leading"
    TRANSPOSE_2D = "transpose_2d"


BoundaryAdapter = CoordinateMapping | CoordinateMap


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
    correspondence: BoundaryAdapter
    source_shape: tuple[int, ...]
    semantic_shape: tuple[int, ...]
    coordinate_map: CoordinateMap
    edge_presented_set: CoordinateSet
    boundary_presented_set: CoordinateSet
    unpresented_set: CoordinateSet

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


def checked_boundary_map(
    adapter: BoundaryAdapter | None,
    source_shape: tuple[int, ...],
    public_shape: tuple[int, ...],
    *,
    output: bool = False,
) -> CoordinateMap:
    """Check an explicit value view, including intentional graph-input slices.

    Input slices may select graph positions, but must supply every public input
    position exactly once. Outputs must also cover every graph result position.
    This establishes value correspondence, not a stream delivery service.
    """
    if adapter is None:
        adapter = CoordinateMapping.IDENTITY
    if isinstance(adapter, CoordinateMapping):
        return _checked_coordinate_map(adapter, source_shape, public_shape)
    if not isinstance(adapter, (IdentityCoordinateMap, AffineRankMap, ExplicitCoordinateMap)):
        raise NetworkOperandError("a boundary adapter must be an explicit coordinate map")
    source, public = RectangularDomain(source_shape), RectangularDomain(public_shape)
    mapping = PositionMap.from_coordinate_map(adapter)
    if mapping.bind_domains(source, public) != mapping:
        raise NetworkOperandError("boundary maps require explicit matching coordinate domains")
    if mapping.sink_set != CoordinateSet.full(public):
        raise NetworkOperandError("boundary map omits required public operand positions")
    if output and mapping.source_set != CoordinateSet.full(source):
        raise NetworkOperandError("output boundary map omits required graph result positions")
    if mapping.source_set.cardinality != mapping.sink_set.cardinality:
        raise NetworkOperandError("boundary maps cannot merge or duplicate element values")
    if (
        isinstance(adapter, ExplicitCoordinateMap)
        and len(adapter.entries) != mapping.source_set.cardinality
    ):
        raise NetworkOperandError("boundary maps require one image per selected source value")
    if isinstance(adapter, AffineRankMap) and not adapter.is_bijection:
        raise NetworkOperandError("affine boundary maps must preserve distinct element values")
    return adapter


def derive_operand_mappings(
    network: DataflowNetwork,
    source: SourceNode,
    references: Mapping[str, tuple[DataflowOperandRef, ...]],
    correspondences: Mapping[str, BoundaryAdapter],
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
    correspondences: Mapping[str, BoundaryAdapter],
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
            else:
                output = resolve_output(network, ref)
                shape = output.port.operand.shape
                empty = CoordinateSet.empty(output.port.operand.position_domain)
                edge_presented = empty
                boundary_presented = empty
                unpresented = empty
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
                    else checked_boundary_map(
                        correspondences[name], operand.shape, tuple(shape), output=not is_input
                    ),
                    edge_presented,
                    boundary_presented,
                    unpresented,
                )
            )
    return tuple(result)


def _compose_maps(first: CoordinateMap, second: CoordinateMap) -> CoordinateMap:
    """Compose the common compact boundary views without enumerating tensors."""

    if (
        isinstance(first, IdentityCoordinateMap)
        and first.domain.is_full
        and first.domain.ambient == first.target_domain
    ):
        return second
    if (
        isinstance(second, IdentityCoordinateMap)
        and second.domain.is_full
        and second.domain.ambient == second.target_domain
    ):
        return first
    if isinstance(first, AffineRankMap) and first.offset == 0 and first.coefficients == (1,):
        if isinstance(second, AffineRankMap):
            return AffineRankMap(
                first.source, second.view_extents, second.target, second.offset, second.coefficients
            )
    selected_source = PositionMap.from_coordinate_map(first).source_set
    source = selected_source.ambient
    target = PositionMap.from_coordinate_map(second).sink_set.ambient
    if selected_source.cardinality > 1_000_000:
        raise MaterializationRequired("this boundary map composition needs an explicit size budget")
    return ExplicitCoordinateMap(
        (
            (point, second.mapped(first.mapped(point)))
            for point in selected_source.materialize(max_points=1_000_000)
        ),
        source_domain=source,
        target_domain=target,
    )


def derive_public_operand_mappings(
    network: DataflowNetwork,
    source: SourceNode,
    exports: Mapping[str, OperandExport],
    correspondences: Mapping[str, BoundaryAdapter],
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
        boundary = checked_boundary_map(
            correspondences[name],
            operand.shape,
            export.operand.domain.extents,
            output=export.operand.direction == "output",
        )
        references[name] = tuple(target.ref for target in export.targets)
        for target in export.targets:
            if len(target.presentations) > 1:
                raise NetworkOperandError("stream mapping needs an explicit presentation selection")
            maps[name, target.ref] = _compose_maps(boundary, target.position_map.coordinate_map)
    return _derive_operand_mappings(network, source, references, correspondences, maps)


__all__ = [
    "CoordinateMapping",
    "BoundaryAdapter",
    "checked_boundary_map",
    "External",
    "Internal",
    "InternalStream",
    "MaterializedOperandPresentation",
    "OperandMapping",
    "OperandPlacement",
    "derive_operand_mappings",
    "derive_public_operand_mappings",
]
