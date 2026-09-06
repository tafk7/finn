# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Candidate A — scheduled input requirements independent of stream exposure.

The dataflow model says what logical data a Region requires and produces, on
what schedule, through which interfaces.  Today those two facts are welded
together: ``InputInterface = (Port, ScheduledInputRequirements)``, so an
operand with no port has no requirements and a Region that consumes it says
nothing at all.

Candidate A splits them:

```python
@dataclass(frozen=True)
class InputRequirement:
    operand: Operand
    requirements: ScheduledInputRequirements


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    input_requirements: tuple[InputRequirement, ...]
    input_ports: tuple[Port, ...]
    outputs: tuple[OutputInterface, ...]
```

A requirement is stated once per (Region, Operand).  Ports say which positions
of that operand cross a boundary and in what order.  Whether the ports present
every required occurrence, some of them, or none is a *derived* comparison, and
covering the difference is the binding's job -- which is exactly what
``REGION.md`` §5.2 condition 1 already says.

The Region never declares storage, and the words "local state" do not appear in
the value.  ``REGION.md`` uses that phrase for what the *binding witness*
supplies; keeping it out of the model is the point.

The prototype cannot edit ``finn.dataflow.region`` before C1, so it mirrors the
proposed value and reuses every canonical part underneath: ``Operand``,
``Port``, ``BeatSequence``, ``ScheduledInputRequirements``,
``ScheduledOutputAvailability``, ``OutputInterface``, ``RegionEndpoint``,
``Edge``, ``BoundaryContract``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from finn.dataflow.network import BoundaryContract, Edge, RegionEndpoint
from finn.dataflow.region import (
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    element_width,
)

# -- the proposed values ------------------------------------------------------


@dataclass(frozen=True)
class InputRequirement:
    """What one Region's computation logically requires of one operand.

    Stated once per operand, not once per port: ``required(i, p)`` counts the
    computation's uses, and a computation does not use a position twice because
    two ports happen to deliver it.  This is a change to ``REGION.md`` §2.3,
    which currently indexes the requirement map by interface; §7 of the
    submission carries the fold.
    """

    operand: Operand
    requirements: ScheduledInputRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")

    @property
    def id(self) -> str:
        return self.operand.id

    @property
    def occurrence_count(self) -> int:
        return self.requirements.occurrence_count


@dataclass(frozen=True)
class ProtoRegion:
    """``DataflowRegion`` under candidate A."""

    schedule: LogicalSchedule
    input_requirements: tuple[InputRequirement, ...]
    input_ports: tuple[Port, ...]
    outputs: tuple[OutputInterface, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_requirements",
            tuple(sorted(self.input_requirements, key=lambda item: item.operand.id)),
        )
        object.__setattr__(
            self, "input_ports", tuple(sorted(self.input_ports, key=lambda item: item.id))
        )
        object.__setattr__(
            self, "outputs", tuple(sorted(self.outputs, key=lambda item: item.port.id))
        )

    @property
    def ports(self) -> tuple[Port, ...]:
        """Every port, input then output, in deterministic order."""

        return self.input_ports + tuple(interface.port for interface in self.outputs)

    def input_requirement(self, operand_id: str) -> InputRequirement:
        matches = tuple(item for item in self.input_requirements if item.operand.id == operand_id)
        if len(matches) != 1:
            raise KeyError(f"expected one input requirement {operand_id!r}, found {len(matches)}")
        return matches[0]

    def input_port(self, port_id: str) -> Port:
        matches = tuple(port for port in self.input_ports if port.id == port_id)
        if len(matches) != 1:
            raise KeyError(f"expected one input port {port_id!r}, found {len(matches)}")
        return matches[0]

    def output_interface(self, port_id: str) -> OutputInterface:
        matches = tuple(item for item in self.outputs if item.port.id == port_id)
        if len(matches) != 1:
            raise KeyError(f"expected one output port {port_id!r}, found {len(matches)}")
        return matches[0]

    def ports_for(self, operand_id: str) -> tuple[Port, ...]:
        """Every input port presenting positions of one operand.

        Plural on purpose.  One operand delivered by two ports is what a matrix
        split across two suppliers looks like, and ``REGION.md`` §5.1
        condition 3 already presumes an operand identity can recur.
        """

        return tuple(port for port in self.input_ports if port.operand.id == operand_id)

    def presented_positions(self, operand_id: str) -> frozenset[tuple[int, ...]]:
        """The union of positions the input ports present for one operand."""

        return (
            frozenset().union(*(port.beat_sequence.image for port in self.ports_for(operand_id)))
            if self.ports_for(operand_id)
            else frozenset()
        )

    def unpresented_positions(self, operand_id: str) -> frozenset[tuple[int, ...]]:
        """Required positions no input port presents.

        Derived, never stored.  It is the size of the question the binding has
        to answer for this operand, and it is a *report*, not a classification:
        the empty set does not mean "streamed", and a full set does not mean
        "embedded".
        """

        required = frozenset(
            position
            for (_iteration, position), multiplicity in self.input_requirement(
                operand_id
            ).requirements.entries
            if multiplicity > 0
        )
        return required - self.presented_positions(operand_id)


@dataclass(frozen=True)
class ProtoNode:
    id: str
    region: ProtoRegion


@dataclass(frozen=True)
class ProtoNetwork:
    nodes: tuple[ProtoNode, ...]
    edges: tuple[Edge, ...]
    boundaries: tuple[BoundaryContract, ...]

    def node(self, node_id: str) -> ProtoNode:
        matches = tuple(node for node in self.nodes if node.id == node_id)
        if len(matches) != 1:
            raise KeyError(f"expected one node {node_id!r}, found {len(matches)}")
        return matches[0]


# -- validation ---------------------------------------------------------------
#
# This is the whole of ``validate_region`` restated over the new shape, not a
# list of additions.  Conditions 3 and 5 of REGION.md §5.1 change what they
# range over -- operand rules now cover requirement operands, and requirement
# rules are keyed by operand rather than by interface -- and two conditions are
# new.  Writing only the new ones would have hidden that.


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def validate_region(region: ProtoRegion) -> tuple[Issue, ...]:
    """Every independently detectable structural issue, in stable order."""

    issues: list[Issue] = []
    schedule = region.schedule

    # 1. schedule extents and level names -- unchanged.
    for index, level in enumerate(schedule.levels):
        if level.extent <= 0:
            issues.append(
                Issue(
                    "schedule.extent_not_positive",
                    f"schedule.levels[{index}].extent",
                    f"schedule extent must be positive, got {level.extent}",
                )
            )
    for name in _duplicates(schedule.level_names):
        issues.append(
            Issue(
                "schedule.level_name_duplicate",
                "schedule.levels",
                f"schedule level name {name!r} is not unique",
            )
        )

    # 2. port identities unique across inputs and outputs -- unchanged.
    for port_id in _duplicates(tuple(port.id for port in region.ports)):
        issues.append(
            Issue(
                "port.id_duplicate",
                "region.ports",
                f"port identity {port_id!r} is not unique within the region",
            )
        )

    # 2b. NEW: one requirement per operand.  A requirement has no port id to
    # distinguish two entries, so a repeat is a duplicate declaration.
    for operand_id in _duplicates(tuple(item.operand.id for item in region.input_requirements)):
        issues.append(
            Issue(
                "requirement.operand_duplicate",
                "region.input_requirements",
                f"operand {operand_id!r} declares more than one input requirement",
            )
        )

    # 3. operand declarations -- EXPANDED: the same rules, now ranging over
    # requirement operands as well as port operands, which is how an unported
    # operand acquires datatype and shape validation at all.
    operands: dict[str, tuple[Operand, str]] = {}
    checked: list[tuple[Operand, str]] = [
        (item.operand, f"input_requirements[{item.operand.id!r}].operand")
        for item in region.input_requirements
    ] + [(port.operand, f"port[{port.id!r}].operand") for port in region.ports]
    for operand, path in checked:
        previous = operands.get(operand.id)
        if previous is None:
            operands[operand.id] = (operand, path)
        elif previous[0].element_type != operand.element_type or previous[0].shape != operand.shape:
            issues.append(
                Issue(
                    "operand.identity_conflict",
                    path,
                    f"operand identity {operand.id!r} has inconsistent type or shape "
                    f"against {previous[1]}",
                )
            )
        width = element_width(operand.element_type)
        if width <= 0:
            issues.append(
                Issue(
                    "operand.bit_width_not_positive",
                    f"{path}.element_type",
                    f"numeric bit width must be positive, got {width}",
                )
            )
        for dimension, extent in enumerate(operand.shape):
            if extent <= 0:
                issues.append(
                    Issue(
                        "operand.extent_not_positive",
                        f"{path}.shape[{dimension}]",
                        f"operand extent must be positive, got {extent}",
                    )
                )

    # 4. derived beat-field domains -- unchanged, now over ports directly.
    for port in region.ports:
        if port.beat_sequence.elements_per_beat <= 0:
            issues.append(
                Issue(
                    "beat.elements_per_beat_not_positive",
                    f"port[{port.id!r}].beat_sequence.elements_per_beat",
                    "elements_per_beat must be positive",
                )
            )

    # 5. requirement domains -- EXPANDED: keyed by operand, and now reached for
    # operands no port exposes.
    for item in region.input_requirements:
        path = f"input_requirements[{item.operand.id!r}]"
        for index, ((iteration, position), multiplicity) in enumerate(item.requirements.entries):
            entry = f"{path}.entries[{index}]"
            if not schedule.contains_point(iteration):
                issues.append(
                    Issue(
                        "requirement.iteration_out_of_domain",
                        f"{entry}.iteration",
                        f"requirement iteration {iteration!r} is outside the schedule",
                    )
                )
            if not item.operand.contains_position(position):
                issues.append(
                    Issue(
                        "requirement.position_out_of_domain",
                        f"{entry}.position",
                        f"requirement position {position!r} is outside operand {item.operand.id!r}",
                    )
                )
            if multiplicity < 0:
                issues.append(
                    Issue(
                        "requirement.multiplicity_negative",
                        f"{entry}.multiplicity",
                        f"requirement multiplicity must be non-negative, got {multiplicity}",
                    )
                )

    # 5b. NEW: an input port exposes a declared requirement.  Without this an
    # author can present an operand the computation never asked for, which is
    # the mirror of the gap this redesign closes.
    declared = {item.operand.id for item in region.input_requirements}
    for port in region.input_ports:
        if port.operand.id not in declared:
            issues.append(
                Issue(
                    "input_port.requirement_missing",
                    f"port[{port.id!r}].operand",
                    f"input port presents operand {port.operand.id!r}, which declares "
                    "no input requirement",
                )
            )

    # 7. beat positions -- unchanged.
    for port in region.ports:
        for ordinal, beat in enumerate(port.beat_sequence.beats):
            beat_path = f"port[{port.id!r}].beat_sequence.beats[{ordinal}]"
            if len(beat) != port.beat_sequence.elements_per_beat:
                issues.append(
                    Issue(
                        "beat.field_count_mismatch",
                        beat_path,
                        f"beat field count {len(beat)} does not equal elements_per_beat "
                        f"{port.beat_sequence.elements_per_beat}",
                    )
                )
            for field, position in enumerate(beat):
                if not port.operand.contains_position(position):
                    issues.append(
                        Issue(
                            "beat.position_out_of_domain",
                            f"{beat_path}[{field}]",
                            f"beat position {position!r} is outside operand {port.operand.id!r}",
                        )
                    )

    # 6 and 8. output availability -- unchanged, and deliberately still
    # port-local: REGION.md §3.7 binds availability to the port's beat image and
    # explicitly refuses the mirror equality for inputs.  That asymmetry is the
    # canon's, and it is why requirements move out of the interface and
    # availability does not.
    for interface in region.outputs:
        path = f"output[{interface.port.id!r}]"
        operand = interface.port.operand
        for index, (position, iteration) in enumerate(interface.availability.entries):
            entry = f"{path}.availability.entries[{index}]"
            if not operand.contains_position(position):
                issues.append(
                    Issue(
                        "availability.position_out_of_domain",
                        f"{entry}.position",
                        f"availability position {position!r} is outside operand {operand.id!r}",
                    )
                )
            if not schedule.contains_point(iteration):
                issues.append(
                    Issue(
                        "availability.iteration_out_of_domain",
                        f"{entry}.iteration",
                        f"availability point {iteration!r} is outside the schedule",
                    )
                )
        missing = tuple(sorted(interface.port.beat_sequence.image - interface.availability.domain))
        omitted = tuple(sorted(interface.availability.domain - interface.port.beat_sequence.image))
        if missing or omitted:
            issues.append(
                Issue(
                    "output.domain_image_mismatch",
                    path,
                    "output availability domain and beat image differ: "
                    f"missing availability={missing!r}, omitted from sequence={omitted!r}",
                )
            )

    return tuple(sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message)))


# -- source mapping -----------------------------------------------------------
#
# The question is "which selected dataflow requirement or product corresponds to
# this source operand", and the answer is a reference into the Region model.
# Nothing here classifies storage, and nothing repackages a BoundaryContract.


@dataclass(frozen=True, slots=True)
class RegionInputRef:
    """One Region's input requirement for one operand."""

    node_id: str
    operand_id: str


@dataclass(frozen=True, slots=True)
class RegionOutputRef:
    """One Region's produced operand."""

    node_id: str
    operand_id: str


DataflowOperandRef = RegionInputRef | RegionOutputRef


class MappingError(ValueError):
    """A source operand corresponds to no dataflow requirement or product."""


def _edge_sinks(network: ProtoNetwork) -> set[RegionEndpoint]:
    return {sink.endpoint for edge in network.edges for sink in edge.sinks}


def _edge_sources(network: ProtoNetwork) -> set[RegionEndpoint]:
    return {edge.source for edge in network.edges}


def derive_input_mappings(network: ProtoNetwork, operand_id: str) -> tuple[RegionInputRef, ...]:
    """Every requirement for ``operand_id`` that the Network does not supply.

    A requirement whose ports are all edge sinks is fed by something the
    Network already produced; the source tensor reaches it by transport, not by
    correspondence.  Every other requirement for the operand -- unported, or
    ported and exposed, or ported by a mix -- is a place the source operand
    enters the selected construction.

    Plural by construction.  Two Regions initialized from the same source
    tensor are two mappings, not an ambiguity, and refusing to say so would
    bake one operation's cardinality into the dataflow model.
    """

    sinks = _edge_sinks(network)
    mappings: list[RegionInputRef] = []
    for node in sorted(network.nodes, key=lambda item: item.id):
        if operand_id not in {item.operand.id for item in node.region.input_requirements}:
            continue
        ports = node.region.ports_for(operand_id)
        supplied_internally = bool(ports) and all(
            RegionEndpoint(node.id, port.id) in sinks for port in ports
        )
        if not supplied_internally:
            mappings.append(RegionInputRef(node.id, operand_id))
    if not mappings:
        raise MappingError(
            f"the selected Network declares no unfed input requirement for operand {operand_id!r}"
        )
    return tuple(mappings)


def derive_output_mappings(network: ProtoNetwork, operand_id: str) -> tuple[RegionOutputRef, ...]:
    """Every production of ``operand_id`` the Network does not consume itself."""

    sources = _edge_sources(network)
    mappings: list[RegionOutputRef] = []
    for node in sorted(network.nodes, key=lambda item: item.id):
        ports = tuple(
            interface.port
            for interface in node.region.outputs
            if interface.port.operand.id == operand_id
        )
        if not ports:
            continue
        if not all(RegionEndpoint(node.id, port.id) in sources for port in ports):
            mappings.append(RegionOutputRef(node.id, operand_id))
    if not mappings:
        raise MappingError(f"the selected Network produces no unconsumed operand {operand_id!r}")
    return tuple(mappings)


# -- derived exposure, computed on demand -------------------------------------


def exposing_ports(network: ProtoNetwork, ref: DataflowOperandRef) -> tuple[RegionEndpoint, ...]:
    """The endpoints presenting a referenced requirement or product."""

    region = network.node(ref.node_id).region
    ports = (
        region.ports_for(ref.operand_id)
        if isinstance(ref, RegionInputRef)
        else tuple(
            interface.port
            for interface in region.outputs
            if interface.port.operand.id == ref.operand_id
        )
    )
    return tuple(RegionEndpoint(ref.node_id, port.id) for port in ports)


def exposing_boundaries(
    network: ProtoNetwork, ref: DataflowOperandRef
) -> tuple[BoundaryContract, ...]:
    """The Network boundaries, if any, that expose a referenced operand.

    Returns the canonical ``BoundaryContract`` values rather than a derived
    record of the same fields.  A consumer that wants the beat sequence, the
    endpoint or the pass correspondence already has them.
    """

    endpoints = set(exposing_ports(network, ref))
    return tuple(boundary for boundary in network.boundaries if boundary.endpoint in endpoints)


__all__ = [
    "DataflowOperandRef",
    "InputRequirement",
    "Issue",
    "MappingError",
    "ProtoNetwork",
    "ProtoNode",
    "ProtoRegion",
    "RegionInputRef",
    "RegionOutputRef",
    "derive_input_mappings",
    "derive_output_mappings",
    "exposing_boundaries",
    "exposing_ports",
    "validate_region",
]
