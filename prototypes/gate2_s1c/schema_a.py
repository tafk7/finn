# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Schema A — local-state inputs on the Region, placement derived from the Network.

The prototype cannot edit ``finn.dataflow.region`` before C1, so it mirrors the
canonical value with the one proposed field added and reuses every canonical
part underneath: ``Operand``, ``InputInterface``, ``OutputInterface``,
``ScheduledInputRequirements``, ``RegionEndpoint``, ``Edge``,
``BoundaryContract``.  ``streamed_projection`` drops the new field and hands the
result to the canonical ``validate_region``, which is the evidence that the
existing structural rules are untouched rather than reimplemented.

The proposed production change is exactly:

```python
@dataclass(frozen=True)
class LocalStateInput:
    operand: Operand
    requirements: ScheduledInputRequirements


@dataclass(frozen=True)
class DataflowRegion:
    schedule: LogicalSchedule
    inputs: tuple[InputInterface, ...]
    outputs: tuple[OutputInterface, ...]
    local_state: tuple[LocalStateInput, ...] = ()   # new, last, defaulted
```
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.network import BoundaryContract, Edge, RegionEndpoint
from finn.dataflow.region import (
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    ScheduledInputRequirements,
)

# -- the proposed value -------------------------------------------------------


@dataclass(frozen=True)
class LocalStateInput:
    """One mathematical input the Region consumes without a stream port.

    There is no storage, technology, slot, file or module field, and there will
    not be one: a local-state input says the Region's computation consumes the
    operand and that this factorization gives it no port.

    ``requirements`` is the variant the prototype exists to price.  ``None``
    means "consumed; this Region does not state when", which is *not* the same
    as an empty requirement function ("consumed at no iteration").  Supplying
    it makes local state a statement about transport alone -- the embedded dot
    product then reads provably the same positions at the same iterations as
    the streamed one -- and costs the same entry count the streamed weight
    interface already costs.  Placement derivation never reads it; see
    ``run.py`` section 6.
    """

    operand: Operand
    requirements: ScheduledInputRequirements | None = None

    @property
    def id(self) -> str:
        """A local-state input is named by its operand; it has no channel."""

        return self.operand.id


@dataclass(frozen=True)
class ProtoRegion:
    """``DataflowRegion`` plus ``local_state``, for the prototype only."""

    schedule: LogicalSchedule
    inputs: tuple[InputInterface, ...]
    outputs: tuple[OutputInterface, ...]
    local_state: tuple[LocalStateInput, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_state",
            tuple(sorted(self.local_state, key=lambda item: (item.operand.id, repr(item)))),
        )

    @property
    def streamed_projection(self) -> DataflowRegion:
        """The canonical value this Region is, ignoring what it holds."""

        return DataflowRegion(self.schedule, self.inputs, self.outputs)


@dataclass(frozen=True)
class ProtoNode:
    id: str
    region: ProtoRegion


@dataclass(frozen=True)
class ProtoNetwork:
    nodes: tuple[ProtoNode, ...]
    edges: tuple[Edge, ...]
    boundaries: tuple[BoundaryContract, ...]


# -- the two added structural rules -------------------------------------------


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def added_region_issues(region: ProtoRegion) -> tuple[Issue, ...]:
    """Only the rules ``local_state`` adds; everything else is unchanged.

    Two rules, and no more:

    1. local-state operand identities are unique within one Region -- a
       local-state input has no port id to distinguish two entries, so a repeat
       is a duplicate declaration rather than a second channel;
    2. an operand is not both streamed *in* and local state in one Region --
       two contradictory transport claims about one mathematical input.

    Coinciding with an *output* operand is deliberately legal: it is exactly
    what a parameter source is, and forbidding it would forbid the decoupled
    supplier.
    """

    issues: list[Issue] = []
    seen: set[str] = set()
    for item in region.local_state:
        if item.operand.id in seen:
            issues.append(
                Issue(
                    "local_state.operand_duplicate",
                    f"local_state[{item.operand.id!r}]",
                    f"local-state operand {item.operand.id!r} is declared more than once",
                )
            )
        seen.add(item.operand.id)
    streamed = {interface.port.operand.id for interface in region.inputs}
    for operand_id in sorted(seen & streamed):
        issues.append(
            Issue(
                "local_state.operand_also_streamed",
                f"local_state[{operand_id!r}]",
                f"operand {operand_id!r} is declared both streamed and local state",
            )
        )
    for item in region.local_state:
        if item.requirements is None:
            continue
        for (iteration, position), _multiplicity in item.requirements.entries:
            if not region.schedule.contains_point(iteration):
                issues.append(
                    Issue(
                        "local_state.requirement.iteration_out_of_domain",
                        f"local_state[{item.operand.id!r}].requirements",
                        f"requirement iteration {iteration!r} is outside the schedule",
                    )
                )
            if not item.operand.contains_position(position):
                issues.append(
                    Issue(
                        "local_state.requirement.position_out_of_domain",
                        f"local_state[{item.operand.id!r}].requirements",
                        f"requirement position {position!r} is outside {item.operand.id!r}",
                    )
                )
    return tuple(issues)


# -- derived placement --------------------------------------------------------


@dataclass(frozen=True)
class External:
    """The operand crosses the Network's edge at a named boundary."""

    boundary_id: str
    node_id: str
    port_id: str


@dataclass(frozen=True)
class LocalState:
    """The operand is held by a node, and there is no port for it."""

    node_id: str
    operand_id: str


@dataclass(frozen=True)
class InternalStream:
    """The operand reaches a port fed from inside the Network.

    Unreachable for a source *input* operand -- an input fed by an edge is a
    continuation of something the Network already produced, not the place the
    source tensor enters -- and kept only because a fused operation's
    intermediate tensor would land here.  If C1 does not want a case with no
    live producer, delete it; the derivation below loses one branch.
    """

    node_id: str
    port_id: str


Placement = External | LocalState | InternalStream


class PlacementError(ValueError):
    """No entry site, or more than one."""


def _sinked_inputs(network: ProtoNetwork) -> set[RegionEndpoint]:
    return {sink.endpoint for edge in network.edges for sink in edge.sinks}


def _consumed_outputs(network: ProtoNetwork) -> set[RegionEndpoint]:
    return {edge.source for edge in network.edges}


def derive_input_placement(network: ProtoNetwork, operand_id: str) -> Placement:
    """Where a source *input* operand enters the selected Network.

    The rule is one sentence: the entry site is the site carrying this operand
    that nothing inside the Network feeds.  A streamed input that is the sink of
    an edge is fed from inside and is therefore a continuation; a local-state
    input is fed from outside by construction.  Exactly one site must survive, and
    both failures are reported rather than resolved by a preference order --
    "the first node called ``compute``" is the guess this replaces.
    """

    sinked = _sinked_inputs(network)
    boundaries = {boundary.endpoint: boundary.id for boundary in network.boundaries}
    candidates: list[Placement] = []
    for node in sorted(network.nodes, key=lambda item: item.id):
        for interface in node.region.inputs:
            if interface.port.operand.id != operand_id:
                continue
            endpoint = RegionEndpoint(node.id, interface.port.id)
            if endpoint in sinked:
                continue
            boundary_id = boundaries.get(endpoint)
            candidates.append(
                External(boundary_id, node.id, interface.port.id)
                if boundary_id is not None
                # An unfed, unexposed input is already a canonical
                # ``endpoint.input_ownership`` failure; name it as a stream so
                # the placement report does not invent a boundary.
                else InternalStream(node.id, interface.port.id)
            )
        for item in node.region.local_state:
            if item.operand.id == operand_id:
                candidates.append(LocalState(node.id, item.operand.id))
    return _exactly_one(candidates, operand_id, "enters")


def derive_output_placement(network: ProtoNetwork, operand_id: str) -> Placement:
    """Where a source *output* operand leaves, by the mirrored rule."""

    consumed = _consumed_outputs(network)
    boundaries = {boundary.endpoint: boundary.id for boundary in network.boundaries}
    candidates: list[Placement] = []
    for node in sorted(network.nodes, key=lambda item: item.id):
        for interface in node.region.outputs:
            if interface.port.operand.id != operand_id:
                continue
            endpoint = RegionEndpoint(node.id, interface.port.id)
            boundary_id = boundaries.get(endpoint)
            if boundary_id is not None:
                candidates.append(External(boundary_id, node.id, interface.port.id))
            elif endpoint not in consumed:
                candidates.append(InternalStream(node.id, interface.port.id))
    return _exactly_one(candidates, operand_id, "leaves")


def _exactly_one(candidates: list[Placement], operand_id: str, verb: str) -> Placement:
    if not candidates:
        raise PlacementError(f"the selected Network has nowhere operand {operand_id!r} {verb}")
    if len(candidates) > 1:
        raise PlacementError(
            f"operand {operand_id!r} {verb} the selected Network in more than one place: "
            + ", ".join(repr(candidate) for candidate in sorted(candidates, key=repr))
        )
    return candidates[0]


def selected_shape(network: ProtoNetwork, placement: Placement) -> tuple[int, ...] | None:
    """The shape at the placed port, or ``None`` when there is no port."""

    if isinstance(placement, LocalState):
        return None
    node = next(item for item in network.nodes if item.id == placement.node_id)
    for interface in (*node.region.inputs, *node.region.outputs):
        if interface.port.id == placement.port_id:
            return tuple(interface.port.operand.shape)
    return None


__all__ = [
    "External",
    "InternalStream",
    "Issue",
    "Placement",
    "PlacementError",
    "ProtoNetwork",
    "ProtoNode",
    "ProtoRegion",
    "LocalState",
    "LocalStateInput",
    "added_region_issues",
    "derive_input_placement",
    "derive_output_placement",
    "selected_shape",
]
