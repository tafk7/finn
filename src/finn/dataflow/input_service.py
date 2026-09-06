# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualified references into a Network, and what its interfaces present.

A region's ``Operand.id`` is region-local.  ``node_a.W`` and ``node_b.W`` may be
different tensors of different types and shapes in one Network, and an edge is
free to carry ``producer.Y`` into ``consumer.X``.  So the identity a caller
holds across a Network is the *qualified* one -- a node and an operand together
-- and that is all these references are.

This module answers two questions about an explicit reference:

```text
exposure       which endpoints and boundaries present this operand
presentation   which of its required positions arrive over a port, from inside
               the Network or from outside it, and which arrive over none
```

It answers no others.  In particular it does not know which source ONNX operand
a reference corresponds to: nothing here can tell whether two region-local
``W``\\ s are the same tensor, and inferring that from a bare operand id is
exactly the guess this layer exists to stop.  Cross-stratum correspondence --
one source operand to a tuple of qualified targets -- is the DataflowOp's, built
from its own declarations and its Design's roles and checked against the
selected Network.

**Presented, not supplied.**  A position presented once may satisfy several
scheduled occurrences through binding-owned replay, and ``REGION.md`` 3.7
refuses any required-versus-presented equality for inputs for that reason.  The
sets below describe the dataflow interface.  They are not a binding witness, and
an empty ``unpresented_positions`` does not mean the region's requirements are
satisfiable -- that is the realizability obligation in section 5.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.network import BoundaryContract, DataflowNetwork, RegionEndpoint
from finn.dataflow.region import (
    Coordinate,
    InputInterface,
    OutputInterface,
    RegionInput,
)


@dataclass(frozen=True, slots=True)
class RegionInputRef:
    """One region's input requirement for one operand, qualified by node."""

    node_id: str
    operand_id: str


@dataclass(frozen=True, slots=True)
class RegionOutputRef:
    """One region's produced operand, qualified by node."""

    node_id: str
    operand_id: str


#: A reference to one operand at one node, in one direction.
DataflowOperandRef = RegionInputRef | RegionOutputRef


class InputServiceError(ValueError):
    """A reference does not resolve, or its endpoint ownership is malformed."""


def resolve_input(network: DataflowNetwork, ref: RegionInputRef) -> RegionInput:
    """Return the region input a reference names.

    Raises:
        InputServiceError: the node or the operand is not there, or the operand
            is declared by more than one input.
    """

    try:
        node = network.node(ref.node_id)
    except KeyError as error:
        raise InputServiceError(f"no node {ref.node_id!r} in the selected Network") from error
    try:
        return node.region.input(ref.operand_id)
    except KeyError as error:
        raise InputServiceError(
            f"node {ref.node_id!r} does not declare exactly one input for operand "
            f"{ref.operand_id!r}"
        ) from error


def resolve_output(network: DataflowNetwork, ref: RegionOutputRef) -> OutputInterface:
    """Return the output interface a reference names.

    Refuses an operand emitted by several output ports rather than picking one:
    a reference is qualified by node and operand, and if that is ambiguous the
    caller needs a port, not a preference order.

    Raises:
        InputServiceError: the node is not there, or the operand is produced by
            no output port or by more than one.
    """

    try:
        node = network.node(ref.node_id)
    except KeyError as error:
        raise InputServiceError(f"no node {ref.node_id!r} in the selected Network") from error
    matches = tuple(
        interface
        for interface in node.region.outputs
        if interface.port.operand.id == ref.operand_id
    )
    if len(matches) != 1:
        raise InputServiceError(
            f"node {ref.node_id!r} produces operand {ref.operand_id!r} on {len(matches)} "
            "output ports, expected one"
        )
    return matches[0]


def exposing_ports(network: DataflowNetwork, ref: DataflowOperandRef) -> tuple[RegionEndpoint, ...]:
    """Return the endpoints presenting a referenced requirement or product.

    Empty for an unported input, which is the whole point of the case: there is
    no endpoint, rather than an endpoint with an empty sequence.
    """

    if isinstance(ref, RegionInputRef):
        item = resolve_input(network, ref)
        return (
            (RegionEndpoint(ref.node_id, item.port.id),) if isinstance(item, InputInterface) else ()
        )
    interface = resolve_output(network, ref)
    return (RegionEndpoint(ref.node_id, interface.port.id),)


def exposing_boundaries(
    network: DataflowNetwork, ref: DataflowOperandRef
) -> tuple[BoundaryContract, ...]:
    """Return the boundaries exposing a referenced operand.

    The canonical ``BoundaryContract`` values, not a derived record restating
    their fields: a caller that wants the external beat sequence, the endpoint
    or the pass correspondence already has them.
    """

    endpoints = set(exposing_ports(network, ref))
    return tuple(boundary for boundary in network.boundaries if boundary.endpoint in endpoints)


def _owned_endpoint(
    network: DataflowNetwork, ref: RegionInputRef
) -> tuple[RegionEndpoint | None, bool]:
    """Resolve a referenced input's endpoint and say whether an edge feeds it.

    Enforces the one obligation the presentation queries depend on: a ported
    input's endpoint is either the sink of exactly one edge or exposed by
    exactly one boundary, never both and never neither.  ``validate_network``
    checks that for every endpoint; this checks it for the one endpoint being
    asked about, so a query over a malformed Network refuses instead of
    returning three sets that look authoritative and are not.
    """

    item = resolve_input(network, ref)
    if not isinstance(item, InputInterface):
        return None, False
    endpoint = RegionEndpoint(ref.node_id, item.port.id)
    sinks = sum(1 for edge in network.edges for sink in edge.sinks if sink.endpoint == endpoint)
    exposures = sum(1 for boundary in network.boundaries if boundary.endpoint == endpoint)
    if sinks + exposures != 1:
        raise InputServiceError(
            f"endpoint {endpoint.node_id!r}.{endpoint.port_id!r} is consumed by {sinks} edges "
            f"and exposed by {exposures} boundaries; exactly one is required before its "
            "presentation can be described"
        )
    return endpoint, sinks == 1


def internally_presented_positions(
    network: DataflowNetwork, ref: RegionInputRef
) -> frozenset[Coordinate]:
    """Required positions a port presents that a Network edge feeds."""

    endpoint, fed_by_edge = _owned_endpoint(network, ref)
    if endpoint is None or not fed_by_edge:
        return frozenset()
    item = resolve_input(network, ref)
    return item.requirements.required_positions & _presented(item)


def externally_presented_positions(
    network: DataflowNetwork, ref: RegionInputRef
) -> frozenset[Coordinate]:
    """Required positions a port presents that a Network boundary exposes."""

    endpoint, fed_by_edge = _owned_endpoint(network, ref)
    if endpoint is None or fed_by_edge:
        return frozenset()
    item = resolve_input(network, ref)
    return item.requirements.required_positions & _presented(item)


def unpresented_positions(network: DataflowNetwork, ref: RegionInputRef) -> frozenset[Coordinate]:
    """Required positions no port of this input presents.

    Position-granular, and deliberately not occurrence-granular.  A position
    presented once and required three times has arrived; serving the repeats is
    binding-owned.  A position no port presents has not arrived, and the binding
    must account for it -- through embedded state, a parameter memory, constant
    generation or another supported service, none of which this model names.
    """

    _owned_endpoint(network, ref)
    item = resolve_input(network, ref)
    return item.requirements.required_positions - _presented(item)


def _presented(item: RegionInput) -> frozenset[Coordinate]:
    return item.port.beat_sequence.image if isinstance(item, InputInterface) else frozenset()


__all__ = [
    "DataflowOperandRef",
    "InputServiceError",
    "RegionInputRef",
    "RegionOutputRef",
    "exposing_boundaries",
    "exposing_ports",
    "externally_presented_positions",
    "internally_presented_positions",
    "resolve_input",
    "resolve_output",
    "unpresented_positions",
]
