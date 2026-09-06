# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The recommended dataflow model — one Region input, optional stream exposure.

Today a Region input is ``(Port, ScheduledInputRequirements)``, so requirements
cannot exist without a port: an operand the computation consumes but no port
presents is absent from the value entirely.  That is the bug.  The fix is to
make the *port* the optional part rather than the requirements:

```python
@dataclass(frozen=True)
class RegionInput:
    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None = None


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]      # element type changes; signature does not
    outputs: tuple[OutputInterface, ...]
```

The requirement is stated whether or not a port presents it.  Whether the port
presents every required occurrence, some of them, or none is a *derived*
comparison, and covering the difference is the binding's obligation under
``REGION.md`` §5.2 -- not a field here.  The Region never declares storage and
the words "local state" do not appear in the value.

The structural claim this shape makes is **at most one input port per operand**.
Zero is the embedded and parameter-source case.  It says nothing about output
ports: the replay Region carries ``X`` on an input and an output, and the output
side is untouched.  ``candidates.py`` holds the two-collection widening for the
day that claim fails, and states its trigger.

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
    Coordinate,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    element_width,
)

# -- the proposed values ------------------------------------------------------


@dataclass(frozen=True)
class RegionInput:
    """One operand the Region requires, and at most one port presenting it.

    ``requirements`` is never optional.  A streamed input and an unported one
    are equally complete statements about the computation -- which positions, at
    which schedule points, how often -- and differ only in whether an ordered
    channel carries any of them.
    """

    operand: Operand
    requirements: ScheduledInputRequirements
    port: Port | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")
        if self.port is not None and not isinstance(self.port, Port):
            raise TypeError("port must be a Port or None")

    @property
    def id(self) -> str:
        """A Region input is named by its operand; it may have no channel."""

        return self.operand.id

    @property
    def occurrence_count(self) -> int:
        return self.requirements.occurrence_count

    @property
    def required_positions(self) -> frozenset[Coordinate]:
        return frozenset(
            position
            for (_iteration, position), multiplicity in self.requirements.entries
            if multiplicity > 0
        )

    @property
    def presented_positions(self) -> frozenset[Coordinate]:
        """Positions the port presents, or none when there is no port."""

        return frozenset() if self.port is None else self.port.beat_sequence.image

    @property
    def unpresented_positions(self) -> frozenset[Coordinate]:
        """Required positions the port does not present.

        Derived, never stored.  It is the size of the question the binding has
        to answer, and it is a *report*, not a classification: empty does not
        mean "streamed" and full does not mean "embedded".
        """

        return self.required_positions - self.presented_positions

    @property
    def presented_field_count(self) -> int:
        return 0 if self.port is None else self.port.beat_sequence.delivered_field_count


@dataclass(frozen=True)
class ProtoRegion:
    """``DataflowRegion`` under the recommendation."""

    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]
    outputs: tuple[OutputInterface, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "inputs", tuple(sorted(self.inputs, key=lambda item: item.operand.id))
        )
        object.__setattr__(
            self, "outputs", tuple(sorted(self.outputs, key=lambda item: item.port.id))
        )

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return tuple(item.port for item in self.inputs if item.port is not None)

    @property
    def ports(self) -> tuple[Port, ...]:
        """Every port that exists, input then output."""

        return self.input_ports + tuple(interface.port for interface in self.outputs)

    def input(self, operand_id: str) -> RegionInput:
        matches = tuple(item for item in self.inputs if item.operand.id == operand_id)
        if len(matches) != 1:
            raise KeyError(f"expected one input operand {operand_id!r}, found {len(matches)}")
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
# The whole of ``validate_region`` restated over the new shape, not a list of
# additions.  One REGION.md §5.1 condition changes what it ranges over, the
# port-shaped conditions learn to skip an input with no port, and two rules are
# new.  Writing only the new ones would have hidden the expansion, which is the
# part that catches what today's model cannot see.


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

    # 2. port identities unique -- unchanged, over the ports that exist.
    for port_id in _duplicates(tuple(port.id for port in region.ports)):
        issues.append(
            Issue(
                "port.id_duplicate",
                "region.ports",
                f"port identity {port_id!r} is not unique within the region",
            )
        )

    # 2b. NEW: one input per operand.  Two inputs for one operand would be two
    # requirement maps for one computation's use of it, with no defined
    # relation between them.  This is also the rule that makes "at most one
    # input port per operand" structural rather than merely observed.
    for operand_id in _duplicates(tuple(item.operand.id for item in region.inputs)):
        issues.append(
            Issue(
                "input.operand_duplicate",
                "region.inputs",
                f"operand {operand_id!r} declares more than one Region input",
            )
        )

    # 2c. NEW: a port presents the operand its input declares.  Local, because
    # both facts are inside one value -- which is the advantage of not splitting
    # requirements and ports into two collections.
    for item in region.inputs:
        if item.port is not None and item.port.operand != item.operand:
            issues.append(
                Issue(
                    "input.port_operand_mismatch",
                    f"input[{item.operand.id!r}].port.operand",
                    f"port {item.port.id!r} presents operand {item.port.operand.id!r}, "
                    f"but its input declares {item.operand.id!r}",
                )
            )

    # 3. operand declarations -- EXPANDED.  The same rules, now ranging over
    # every Region input's operand rather than over port operands only, which
    # is how an operand with no port acquires datatype and shape validation at
    # all.
    seen: dict[str, tuple[Operand, str]] = {}
    checked: list[tuple[Operand, str]] = [
        (item.operand, f"input[{item.operand.id!r}].operand") for item in region.inputs
    ] + [
        (interface.port.operand, f"output[{interface.port.id!r}].port.operand")
        for interface in region.outputs
    ]
    for operand, path in checked:
        previous = seen.get(operand.id)
        if previous is None:
            seen[operand.id] = (operand, path)
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

    # 4 and 7. beat field domain and beat positions -- unchanged rules, applied
    # to the ports that exist.  An input with no port contributes none.
    for port in region.ports:
        if port.beat_sequence.elements_per_beat <= 0:
            issues.append(
                Issue(
                    "beat.elements_per_beat_not_positive",
                    f"port[{port.id!r}].beat_sequence.elements_per_beat",
                    "elements_per_beat must be positive",
                )
            )
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

    # 5. requirement domains -- unchanged rules, now reached for an input with
    # no port, where today there is no input to reach.
    for item in region.inputs:
        path = f"input[{item.operand.id!r}].requirements"
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

    # 6 and 8. output availability -- unchanged, and deliberately still
    # port-local.  REGION.md §3.7 binds availability to the port's beat image
    # and explicitly refuses the mirror equality for inputs; that asymmetry is
    # the canon's, and it is why the input side changes and the output side
    # does not.
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

    A requirement whose port is an edge sink is fed by something the Network
    already produced; the source tensor reaches it by transport, not by
    correspondence.  Every other requirement for the operand -- unported, or
    ported and exposed -- is a place the source operand enters the selected
    construction.

    Plural by construction.  Two Regions initialized from the same source
    tensor are two mappings, not an ambiguity, and refusing to say so would bake
    one operation's cardinality into the dataflow model.
    """

    sinks = _edge_sinks(network)
    mappings: list[RegionInputRef] = []
    for node in sorted(network.nodes, key=lambda item: item.id):
        for item in node.region.inputs:
            if item.operand.id != operand_id:
                continue
            fed_internally = (
                item.port is not None and RegionEndpoint(node.id, item.port.id) in sinks
            )
            if not fed_internally:
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
        if ports and not all(RegionEndpoint(node.id, port.id) in sources for port in ports):
            mappings.append(RegionOutputRef(node.id, operand_id))
    if not mappings:
        raise MappingError(f"the selected Network produces no unconsumed operand {operand_id!r}")
    return tuple(mappings)


# -- derived exposure, computed on demand -------------------------------------


def exposing_ports(network: ProtoNetwork, ref: DataflowOperandRef) -> tuple[RegionEndpoint, ...]:
    """The endpoints presenting a referenced requirement or product."""

    region = network.node(ref.node_id).region
    if isinstance(ref, RegionInputRef):
        port = region.input(ref.operand_id).port
        ports: tuple[Port, ...] = () if port is None else (port,)
    else:
        ports = tuple(
            interface.port
            for interface in region.outputs
            if interface.port.operand.id == ref.operand_id
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
    "Issue",
    "MappingError",
    "ProtoNetwork",
    "ProtoNode",
    "ProtoRegion",
    "RegionInput",
    "RegionInputRef",
    "RegionOutputRef",
    "derive_input_mappings",
    "derive_output_mappings",
    "exposing_boundaries",
    "exposing_ports",
    "validate_region",
]
