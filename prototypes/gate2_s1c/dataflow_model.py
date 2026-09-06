# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The recommended dataflow model — a Region input is ported or it is not.

Today a Region input is ``(Port, ScheduledInputRequirements)``, so requirements
cannot exist without a port and an operand no port presents is absent from the
value entirely.  That is the bug.  The fix adds the missing case rather than
making the existing one nullable:

```python
@dataclass(frozen=True)
class InputInterface:
    port: Port
    requirements: ScheduledInputRequirements

    @property
    def operand(self) -> Operand:
        return self.port.operand


@dataclass(frozen=True)
class UnportedInput:
    operand: Operand
    requirements: ScheduledInputRequirements


RegionInput = InputInterface | UnportedInput


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[RegionInput, ...]
    outputs: tuple[OutputInterface, ...]
```

``InputInterface`` keeps its name, its fields and its constructor.  The operand
is authored once in both cases -- off the port when there is one -- so an input
whose declared operand disagrees with its port's is not a validation rule but an
unrepresentable state.

Requirements are mandatory in both cases.  A ported input and an unported one
are equally complete statements about the computation: which positions, at which
schedule points, how often.  They differ only in whether an ordered channel
carries any of them.

The Region declares no storage and the words "local state" do not appear in the
value.  ``REGION.md`` §5.2 gives "declared local state" to the binding witness,
and this redesign leaves it there.

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
class InputInterface:
    """One operand the Region requires, presented by one ordered channel.

    Unchanged from today in name, fields and construction order.  ``operand`` is
    a property over the port rather than a field, so the two can never disagree.
    """

    port: Port
    requirements: ScheduledInputRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.port, Port):
            raise TypeError("port must be a Port")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")

    @property
    def operand(self) -> Operand:
        return self.port.operand


@dataclass(frozen=True)
class UnportedInput:
    """One operand the Region requires that no ordered channel presents.

    The only genuinely new case.  It says the computation consumes the operand
    and that this factorization gives it no port; it names no memory,
    technology, slot, image or module, and a physical choice can never add or
    remove one.
    """

    operand: Operand
    requirements: ScheduledInputRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Operand):
            raise TypeError("operand must be an Operand")
        if not isinstance(self.requirements, ScheduledInputRequirements):
            raise TypeError("requirements must be ScheduledInputRequirements")


RegionInput = InputInterface | UnportedInput


def required_positions(item: RegionInput) -> frozenset[Coordinate]:
    """Positions the computation requires at least once."""

    return frozenset(
        position
        for (_iteration, position), multiplicity in item.requirements.entries
        if multiplicity > 0
    )


def presented_positions(item: RegionInput) -> frozenset[Coordinate]:
    """Positions this input's port presents, or none when it has no port."""

    return item.port.beat_sequence.image if isinstance(item, InputInterface) else frozenset()


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
        return tuple(item.port for item in self.inputs if isinstance(item, InputInterface))

    @property
    def ports(self) -> tuple[Port, ...]:
        """Every port that exists, input then output."""

        return self.input_ports + tuple(interface.port for interface in self.outputs)

    def input(self, operand_id: str) -> RegionInput:
        matches = tuple(item for item in self.inputs if item.operand.id == operand_id)
        if len(matches) != 1:
            raise KeyError(f"expected one input operand {operand_id!r}, found {len(matches)}")
        return matches[0]

    def input_interface(self, port_id: str) -> InputInterface:
        """Unchanged: a port lookup can only ever find a ported input."""

        matches = tuple(
            item
            for item in self.inputs
            if isinstance(item, InputInterface) and item.port.id == port_id
        )
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


# -- region validation --------------------------------------------------------
#
# The whole of ``validate_region`` restated over the new shape.  One REGION.md
# §5.1 condition changes what it ranges over, the port-shaped conditions range
# over the ports that exist, and one rule is new.  Writing only the new one
# would have hidden the expansion, which is the part that catches what today's
# model cannot see.


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def _input_path(item: RegionInput) -> str:
    return (
        f"input[{item.port.id!r}]"
        if isinstance(item, InputInterface)
        else f"unported_input[{item.operand.id!r}]"
    )


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
    # requirement maps for one computation's use of it, with no defined relation
    # between them -- which is also exactly the relation a multi-port widening
    # would have to define.  See MULTI_PORT_WIDENING.
    for operand_id in _duplicates(tuple(item.operand.id for item in region.inputs)):
        issues.append(
            Issue(
                "input.operand_duplicate",
                "region.inputs",
                f"operand {operand_id!r} declares more than one Region input",
            )
        )

    # There is no operand/port agreement rule.  Under the sum type the operand
    # of a ported input *is* its port's operand, so disagreement is
    # unrepresentable rather than reportable.

    # 3. operand declarations -- EXPANDED.  The same rules, now ranging over
    # every Region input's operand rather than over port operands only, which is
    # how an operand with no port acquires datatype and shape validation at all.
    seen: dict[str, tuple[Operand, str]] = {}
    checked: list[tuple[Operand, str]] = [
        (item.operand, f"{_input_path(item)}.operand") for item in region.inputs
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
    # to the ports that exist.  An unported input contributes none.
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

    # 5. requirement domains -- unchanged rules, now reached for an unported
    # input, where today there is no input to reach.
    for item in region.inputs:
        path = f"{_input_path(item)}.requirements"
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
    # the canon's, and it is why the input side gains a case and the output side
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


# -- network validation, one added rule ---------------------------------------


def network_operand_issues(network: ProtoNetwork) -> tuple[Issue, ...]:
    """One operand identity must mean one logical tensor across the Network.

    ``REGION.md`` §5.1 condition 3 states this within one Region.  Lifting it to
    the Network is what makes ``Operand.id`` usable as the source-matching
    namespace: without it, two unrelated Regions may each call something ``W``
    and a source operand would correspond to both.

    It catches unrelated tensors that differ in type or shape.  Two unrelated
    tensors that happen to agree on both are an authoring collision the model
    cannot see, and the declaration-side qualification in the submission's §5 is
    the answer to those.
    """

    issues: list[Issue] = []
    seen: dict[str, tuple[Operand, str]] = {}
    for node in sorted(network.nodes, key=lambda item: item.id):
        operands = [(item.operand, f"node[{node.id!r}].input") for item in node.region.inputs] + [
            (interface.port.operand, f"node[{node.id!r}].output")
            for interface in node.region.outputs
        ]
        for operand, path in operands:
            previous = seen.get(operand.id)
            if previous is None:
                seen[operand.id] = (operand, path)
            elif (
                previous[0].element_type != operand.element_type
                or previous[0].shape != operand.shape
            ):
                issues.append(
                    Issue(
                        "network.operand_identity_conflict",
                        path,
                        f"operand identity {operand.id!r} has inconsistent type or shape "
                        f"against {previous[1]}",
                    )
                )
    return tuple(issues)


# -- source mapping -----------------------------------------------------------


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


def derive_input_mappings(network: ProtoNetwork, operand_id: str) -> tuple[RegionInputRef, ...]:
    """Every Region input requirement this source operand corresponds to.

    One meaning and only one: **correspondence**.  Every input in the selected
    Network requiring this operand is a target, whether a port presents it,
    whether an edge feeds that port, and whether anything supplies it at all.
    No filtering happens here, because a filtered set is a different question
    and giving both the same name is how the meaning drifted in the earlier
    passes.

    The two other questions are asked separately, over the same refs:

    ```text
    dataflow provenance     internally_supplied_positions / externally_supplied_positions
    what is still owed      unsupplied_positions
    physical provisioning   not in this model at all -- U6 owns it
    ```

    Ordering is by node id for determinism of the returned tuple only.  Nothing
    is *selected* by ordering.
    """

    mappings = tuple(
        RegionInputRef(node.id, operand_id)
        for node in sorted(network.nodes, key=lambda item: item.id)
        for item in node.region.inputs
        if item.operand.id == operand_id
    )
    if not mappings:
        raise MappingError(
            f"the selected Network declares no input requirement for operand {operand_id!r}"
        )
    return mappings


def derive_output_mappings(network: ProtoNetwork, operand_id: str) -> tuple[RegionOutputRef, ...]:
    """Every Region product this source operand corresponds to."""

    mappings = tuple(
        RegionOutputRef(node.id, operand_id)
        for node in sorted(network.nodes, key=lambda item: item.id)
        for interface in node.region.outputs
        if interface.port.operand.id == operand_id
    )
    if not mappings:
        raise MappingError(f"the selected Network produces no operand {operand_id!r}")
    return mappings


# -- derived provenance, computed on demand -----------------------------------
#
# Three disjoint position sets per referenced requirement.  Position-granular,
# not a boolean: REGION.md §3.7 is explicit that a port may present a required
# position "once, repeatedly, or not at all", so "is this input edge-fed" cannot
# by itself say that the Network supplies every required position.
#
# They are *not* occurrence-granular.  A position presented once and required
# three times has entered the construction; serving the re-reads is the
# binding's business, and §3.7 refuses any required-versus-presented equality
# for inputs.


def _region_input(network: ProtoNetwork, ref: RegionInputRef) -> RegionInput:
    return network.node(ref.node_id).region.input(ref.operand_id)


def _port_endpoint(network: ProtoNetwork, ref: RegionInputRef) -> RegionEndpoint | None:
    item = _region_input(network, ref)
    return RegionEndpoint(ref.node_id, item.port.id) if isinstance(item, InputInterface) else None


def internally_supplied_positions(
    network: ProtoNetwork, ref: RegionInputRef
) -> frozenset[Coordinate]:
    """Required positions delivered by a port that a Network edge feeds."""

    endpoint = _port_endpoint(network, ref)
    if endpoint is None:
        return frozenset()
    sinks = {sink.endpoint for edge in network.edges for sink in edge.sinks}
    if endpoint not in sinks:
        return frozenset()
    item = _region_input(network, ref)
    return required_positions(item) & presented_positions(item)


def externally_supplied_positions(
    network: ProtoNetwork, ref: RegionInputRef
) -> frozenset[Coordinate]:
    """Required positions delivered by a port a Network boundary exposes."""

    endpoint = _port_endpoint(network, ref)
    if endpoint is None:
        return frozenset()
    exposed = {boundary.endpoint for boundary in network.boundaries}
    if endpoint not in exposed:
        return frozenset()
    item = _region_input(network, ref)
    return required_positions(item) & presented_positions(item)


def unsupplied_positions(network: ProtoNetwork, ref: RegionInputRef) -> frozenset[Coordinate]:
    """Required positions no port of this input presents.

    What the binding still owes, stated as dataflow rather than as storage.  It
    is a report, not a classification: empty does not mean "streamed" and full
    does not mean "embedded".
    """

    item = _region_input(network, ref)
    return required_positions(item) - presented_positions(item)


def exposing_ports(network: ProtoNetwork, ref: DataflowOperandRef) -> tuple[RegionEndpoint, ...]:
    """The endpoints presenting a referenced requirement or product."""

    region = network.node(ref.node_id).region
    if isinstance(ref, RegionInputRef):
        endpoint = _port_endpoint(network, ref)
        return () if endpoint is None else (endpoint,)
    return tuple(
        RegionEndpoint(ref.node_id, interface.port.id)
        for interface in region.outputs
        if interface.port.operand.id == ref.operand_id
    )


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
    "InputInterface",
    "Issue",
    "MappingError",
    "ProtoNetwork",
    "ProtoNode",
    "ProtoRegion",
    "RegionInput",
    "RegionInputRef",
    "RegionOutputRef",
    "UnportedInput",
    "derive_input_mappings",
    "derive_output_mappings",
    "exposing_boundaries",
    "exposing_ports",
    "externally_supplied_positions",
    "internally_supplied_positions",
    "network_operand_issues",
    "presented_positions",
    "required_positions",
    "unsupplied_positions",
    "validate_region",
]
