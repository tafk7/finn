# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Schemas B and C, far enough to count what they cost.

Both are written against the same four cases as schema A so the comparison is
about structure rather than about how much of each was implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from finn.dataflow.network import RegionEndpoint
from finn.dataflow.region import DataflowRegion, Operand, ScheduledInputRequirements

# -- schema B: adjacent immutable Region semantic metadata --------------------
#
# ``DataflowRegion`` is untouched; a companion value carries what it consumes
# without a port.  The value is small.  What is not small is that from here on
# there are two values required to understand one Region, and every place that
# holds, compares, forwards, gates or validates a Region has to hold, compare,
# forward, gate and validate a second one beside it.


@dataclass(frozen=True)
class RegionResidency:
    """What one Region consumes without a port, stated beside it."""

    operands: tuple[Operand, ...] = ()
    requirements: tuple[ScheduledInputRequirements, ...] = ()


@dataclass(frozen=True)
class AnnotatedRegion:
    """The pair a consumer actually needs."""

    region: DataflowRegion
    residency: RegionResidency = RegionResidency()


@dataclass(frozen=True)
class AnnotatedNetwork:
    """A Network plus the companion map its nodes are missing."""

    network: object
    residency_by_node: dict[str, RegionResidency] = field(default_factory=dict)


#: Every seam that has to learn about the companion for B to work end to end.
#: Read off the current code, not estimated.
B_THREADING_SITES = (
    "kernels/kernel.py: Region declaration -- a second exported member, or a "
    "Region declaration that returns a pair",
    "kernels/kernel.py: the no-local-Decision audit and the region type token",
    "model/semantics.py: a second ValueSemantics for the companion",
    "designs/design.py: DataflowDesign.region(role) -- a second accessor",
    "designs/design.py: _network_property -- a second dependency per segment and "
    "a second returned value, gated by the same segment `when`",
    "designs/design.py: _correspondence_constraint -- compares node Regions to "
    "segment Regions; must compare companions too or the pair can disagree",
    "designs/design.py: SelectedNetwork / the `network` export -- what a consumer "
    "receives is no longer one value",
    "network_validation.py: validate_network takes a Network; it would need the "
    "companion map to check residency at all",
    "artifacts/*: ModuleBuildSpec's Region witness becomes a pair",
)


def derive_placement_b(annotated: AnnotatedNetwork, operand_id: str) -> str:
    """The same rule as schema A, over two values instead of one."""

    network = annotated.network
    sinked = {sink.endpoint for edge in network.edges for sink in edge.sinks}  # type: ignore[attr-defined]
    boundaries = {b.endpoint: b.id for b in network.boundaries}  # type: ignore[attr-defined]
    found: list[str] = []
    for node in network.nodes:  # type: ignore[attr-defined]
        for interface in node.region.inputs:
            if interface.port.operand.id != operand_id:
                continue
            endpoint = RegionEndpoint(node.id, interface.port.id)
            if endpoint in sinked:
                continue
            found.append(f"External({boundaries.get(endpoint)},{node.id},{interface.port.id})")
        # The one line that needs the second value, and the reason the second
        # value has to reach every consumer that ever asks this question.
        companion = annotated.residency_by_node.get(node.id, RegionResidency())
        for operand in companion.operands:
            if operand.id == operand_id:
                found.append(f"LocalState({node.id},{operand.id})")
    if len(found) != 1:
        raise ValueError(f"{operand_id}: {found}")
    return found[0]


# -- schema C: a separate requirement / disposition graph ---------------------
#
# Source-visible requirements are modelled independently of the Region, and a
# disposition graph beside the Network says what became of each.


class DispositionKind(str, Enum):
    EXTERNAL = "external"
    INTERNAL_STREAM = "internal_stream"
    LOCAL_STATE = "local_state"


@dataclass(frozen=True)
class SemanticRequirement:
    """A source-visible requirement.  Note the fields."""

    id: str
    element_type: object
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RequirementDisposition:
    """Authored, not derived: what the contributor says became of it."""

    requirement_id: str
    kind: DispositionKind
    node_id: str
    port_id: str | None = None
    boundary_id: str | None = None


@dataclass(frozen=True)
class DispositionGraph:
    requirements: tuple[SemanticRequirement, ...]
    dispositions: tuple[RequirementDisposition, ...]


def derive_placement_c(graph: DispositionGraph, operand_id: str) -> str:
    """Not a derivation.  A lookup in a table someone wrote by hand."""

    matches = [item for item in graph.dispositions if item.requirement_id == operand_id]
    if len(matches) != 1:
        raise ValueError(f"{operand_id}: {matches}")
    item = matches[0]
    return f"{item.kind.value}({item.boundary_id},{item.node_id},{item.port_id})"


def disposition_agrees_with_network(
    graph: DispositionGraph, network: object, operand_id: str
) -> bool:
    """C's extra obligation: the table and the topology can disagree.

    Schema A cannot express this failure, because there is no second statement
    to disagree with.  C can, so C has to check it -- which means C carries the
    whole of A's derivation *as well as* the table.
    """

    # Local import: the prototype's modules are siblings on sys.path, and this
    # dependency direction (C needs A) is itself part of the finding.
    from schema_a import External, LocalState, derive_input_placement  # noqa: PLC0415

    truth = derive_input_placement(network, operand_id)  # type: ignore[arg-type]
    claimed = [item for item in graph.dispositions if item.requirement_id == operand_id][0]
    if isinstance(truth, External):
        return claimed.kind is DispositionKind.EXTERNAL and claimed.node_id == truth.node_id
    if isinstance(truth, LocalState):
        return claimed.kind is DispositionKind.LOCAL_STATE and claimed.node_id == truth.node_id
    return claimed.kind is DispositionKind.INTERNAL_STREAM and claimed.node_id == truth.node_id


__all__ = [
    "AnnotatedNetwork",
    "AnnotatedRegion",
    "B_THREADING_SITES",
    "DispositionGraph",
    "DispositionKind",
    "RegionResidency",
    "RequirementDisposition",
    "SemanticRequirement",
    "derive_placement_b",
    "derive_placement_c",
    "disposition_agrees_with_network",
]
