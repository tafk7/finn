# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Public operand views over actual Region values; never virtual boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from finn.dataflow.datatypes import QONNXDataType, canonical_qonnx_datatype
from finn.parked.dataflow.logical_values.maps import (
    AffineRankMap,
    Coordinate,
    CoordinateSet,
    ExplicitCoordinateMap,
    IdentityCoordinateMap,
    RectangularDomain,
)
from finn.parked.dataflow.logical_values.network import DataflowNetwork, PositionMap, RegionEndpoint
from finn.parked.dataflow.logical_values.refs import DataflowOperandRef, RegionInputRef, RegionOutputRef
from finn.parked.dataflow.logical_values.region import InputInterface, Operand, Port


class InterfaceError(ValueError):
    """An export fails to describe the actual logical body."""


@dataclass(frozen=True)
class PublicOperand:
    key: str
    direction: Literal["input", "output"]
    element_type: QONNXDataType
    domain: RectangularDomain

    def __post_init__(self) -> None:
        if not self.key or self.direction not in ("input", "output"):
            raise InterfaceError("public operands require a key and input/output direction")
        if not isinstance(self.domain, RectangularDomain):
            raise TypeError("public operands require a rectangular domain")
        object.__setattr__(self, "element_type", canonical_qonnx_datatype(self.element_type))


@dataclass(frozen=True)
class OperandTarget:
    """One actual operand, mapped from public coordinates, and its presentations."""

    ref: DataflowOperandRef
    position_map: PositionMap
    presentations: tuple[RegionEndpoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ref, (RegionInputRef, RegionOutputRef)):
            raise TypeError("an operand target requires a qualified operand reference")
        if not isinstance(self.position_map, PositionMap):
            raise TypeError("an operand target requires an explicit position map")
        object.__setattr__(self, "presentations", tuple(self.presentations))


@dataclass(frozen=True)
class OperandExport:
    operand: PublicOperand
    targets: tuple[OperandTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not isinstance(self.operand, PublicOperand) or not self.targets:
            raise InterfaceError("an export requires a public operand and actual targets")

    def presentation(self, endpoint: RegionEndpoint | None = None) -> OperandTarget:
        pairs = tuple((target, port) for target in self.targets for port in target.presentations)
        if endpoint is not None:
            pairs = tuple(pair for pair in pairs if pair[1] == endpoint)
        if len(pairs) != 1:
            raise InterfaceError("a stream claim must select exactly one actual presentation")
        return pairs[0][0]

    def public_beat(
        self, network: DataflowNetwork, ordinal: int, endpoint: RegionEndpoint | None = None
    ) -> tuple[Coordinate, ...]:
        """Translate one actual ordered beat; invent neither order nor passes."""
        target = self.presentation(endpoint)
        if endpoint is None:
            endpoint = target.presentations[0]
        port = presentation_port(network, target.ref, endpoint)
        return tuple(
            inverse_position(target.position_map, point)
            for point in port.beat_sequence.beat(ordinal)
        )


def body_operand(network: DataflowNetwork, ref: DataflowOperandRef) -> Operand:
    region = network.node(ref.node_id).region
    if isinstance(ref, RegionInputRef):
        matches = tuple(item.operand for item in region.inputs if item.operand.id == ref.operand_id)
    else:
        matches = tuple(
            item.port.operand for item in region.outputs if item.port.operand.id == ref.operand_id
        )
    if not matches or any(operand != matches[0] for operand in matches):
        raise InterfaceError("a public target must resolve to one consistent operand value")
    return matches[0]


def presentation_port(
    network: DataflowNetwork, ref: DataflowOperandRef, endpoint: RegionEndpoint
) -> Port:
    if endpoint.node_id != ref.node_id:
        raise InterfaceError("presentation belongs to a different Region")
    region = network.node(ref.node_id).region
    ports = (
        tuple(item.port for item in region.inputs if isinstance(item, InputInterface))
        if isinstance(ref, RegionInputRef)
        else tuple(item.port for item in region.outputs)
    )
    matches = tuple(
        port for port in ports if port.id == endpoint.port_id and port.operand.id == ref.operand_id
    )
    if len(matches) != 1:
        raise InterfaceError("presentation does not name this operand in its declared direction")
    return matches[0]


def validate_operand_export(network: DataflowNetwork, export: OperandExport) -> None:
    public = export.operand
    seen: set[DataflowOperandRef] = set()
    for target in export.targets:
        if target.ref in seen:
            raise InterfaceError("an export repeats a body operand")
        seen.add(target.ref)
        if (public.direction == "input") != isinstance(target.ref, RegionInputRef):
            raise InterfaceError("public and body operand directions differ")
        operand = body_operand(network, target.ref)
        if operand.element_type != public.element_type:
            raise InterfaceError("an export cannot convert an element datatype")
        mapping = target.position_map
        # Binding refuses contradictory declared domains, including same-size shapes.
        if mapping.bind_domains(public.domain, operand.position_domain) != mapping:
            raise InterfaceError("an export map must carry explicit domains")
        if mapping.source_set != CoordinateSet.full(public.domain):
            raise InterfaceError("an export must cover its public coordinate domain")
        if mapping.sink_set != CoordinateSet.full(operand.position_domain):
            raise InterfaceError("an export must cover every required/produced body position")
        raw = mapping.coordinate_map
        if isinstance(raw, AffineRankMap) and not raw.is_bijection:
            raise InterfaceError(
                "coordinate views must preserve values without broadcast or merging"
            )
        if isinstance(raw, ExplicitCoordinateMap):
            if (
                len(raw.entries) != public.domain.cardinality
                or len(raw.target_set.rank_intervals) < 1
            ):
                raise InterfaceError("an explicit export must be a total one-to-one value map")
            if operand.position_count != len(raw.entries):
                raise InterfaceError("an explicit export cannot broadcast or merge values")
        if len(set(target.presentations)) != len(target.presentations):
            raise InterfaceError("an export repeats a presentation")
        for endpoint in target.presentations:
            presentation_port(network, target.ref, endpoint)


def inverse_position(mapping: PositionMap, position: Coordinate) -> Coordinate:
    """Invert an admitted bijective coordinate view without expanding its domain."""
    raw = mapping.coordinate_map
    if isinstance(raw, IdentityCoordinateMap):
        return raw.mapped(position)
    if isinstance(raw, ExplicitCoordinateMap):
        matches = tuple(source for source, target in raw.entries if target == position)
        if len(matches) != 1:
            raise InterfaceError("a presentation position has no unique public value")
        return matches[0]
    if not raw.is_bijection:
        raise InterfaceError("a presentation requires an invertible coordinate view")
    rank = raw.target.rank_of(position) - raw.offset
    digits = [0] * len(raw.view_extents)
    for extent, coefficient in zip(raw.view_extents, raw.coefficients):
        if coefficient < 0:
            rank -= coefficient * (extent - 1)
    for axis in sorted(
        range(len(digits)), key=lambda index: abs(raw.coefficients[index]), reverse=True
    ):
        extent, coefficient = raw.view_extents[axis], raw.coefficients[axis]
        if extent == 1:
            continue
        if coefficient == 0:
            raise InterfaceError("a coordinate view loses a value axis")
        digit, rank = divmod(rank, abs(coefficient))
        digits[axis] = digit if coefficient > 0 else extent - 1 - digit
    result = raw.source.coordinate_at(RectangularDomain(raw.view_extents).rank_of(tuple(digits)))
    if raw.mapped(result) != position:
        raise InterfaceError("a presentation position has no public inverse")
    return result


__all__ = [
    "InterfaceError",
    "PublicOperand",
    "OperandTarget",
    "OperandExport",
    "body_operand",
    "presentation_port",
    "validate_operand_export",
    "inverse_position",
]
