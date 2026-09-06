# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Schemas D and E, far enough to count what they cost.

D keeps ``DataflowRegion`` and returns a companion value beside it; E models
source-visible requirements independently and authors a disposition graph.  Both
are written against the same cases as candidate A so the comparison is about
structure rather than about how much of each was implemented.

They were written in the first pass and are unchanged in substance.  What
changed is the value they wrap: the companion now has to carry input
requirements rather than a list of held operands, which makes D strictly worse
-- the second value is no longer a small annotation but half the Region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from finn.dataflow.network import RegionEndpoint
from finn.dataflow.region import DataflowRegion, Operand, ScheduledInputRequirements

# -- schema D: adjacent immutable Region metadata -----------------------------
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


#: Every seam that has to learn about the companion for D to work end to end.
#: Read off the current code, not estimated.
D_THREADING_SITES = (
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


def derive_mappings_d(annotated: AnnotatedNetwork, operand_id: str) -> tuple[str, ...]:
    """The same rule as candidate A, over two values instead of one.

    Every line that reads ``node.region`` needs its companion beside it, and the
    companion is keyed by node id in a map the Network does not own.  Nothing
    checks that the two agree.
    """

    network = annotated.network
    sinks = {sink.endpoint for edge in network.edges for sink in edge.sinks}  # type: ignore[attr-defined]
    found: list[str] = []
    for node in network.nodes:  # type: ignore[attr-defined]
        companion = annotated.residency_by_node.get(node.id, RegionResidency())
        declared = {operand.id for operand in companion.operands} | {
            port.operand.id for port in node.region.input_ports
        }
        if operand_id not in declared:
            continue
        ports = tuple(port for port in node.region.input_ports if port.operand.id == operand_id)
        supplied = bool(ports) and all(RegionEndpoint(node.id, port.id) in sinks for port in ports)
        if not supplied:
            found.append(f"RegionInputRef({node.id},{operand_id})")
    return tuple(found)


# -- schema E: a separate requirement / disposition graph ---------------------
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


def derive_mappings_e(graph: DispositionGraph, operand_id: str) -> tuple[str, ...]:
    """Not a derivation.  A lookup in a table someone wrote by hand."""

    matches = [item for item in graph.dispositions if item.requirement_id == operand_id]
    if not matches:
        raise ValueError(operand_id)
    return tuple(
        f"{item.kind.value}({item.node_id})"
        for item in sorted(matches, key=lambda entry: entry.node_id)
    )


def disposition_agrees_with_network(
    graph: DispositionGraph, network: object, operand_id: str
) -> bool:
    """E's extra obligation: the table and the topology can disagree.

    Candidate A cannot express this failure, because there is no second
    statement to disagree with.  E can, so E has to check it -- which means E
    carries the whole of A's derivation *as well as* the table.
    """

    # Local import: the prototype's modules are siblings on sys.path, and this
    # dependency direction (E needs A) is itself part of the finding.
    from dataflow_model import derive_input_mappings  # noqa: PLC0415

    truth = derive_input_mappings(network, operand_id)  # type: ignore[arg-type]
    claimed = tuple(item for item in graph.dispositions if item.requirement_id == operand_id)
    if len(claimed) != len(truth):
        return False
    return all(
        item.node_id == ref.node_id
        for item, ref in zip(sorted(claimed, key=lambda i: i.node_id), truth)
    )


__all__ = [
    "AnnotatedNetwork",
    "AnnotatedRegion",
    "D_THREADING_SITES",
    "DispositionGraph",
    "DispositionKind",
    "RegionResidency",
    "RequirementDisposition",
    "SemanticRequirement",
    "derive_mappings_d",
    "derive_mappings_e",
    "disposition_agrees_with_network",
]
