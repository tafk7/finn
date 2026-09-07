# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualified references into a Network, and how they resolve.

A region's ``Operand.id`` is region-local.  ``node_a.W`` and ``node_b.W`` may be
different tensors of different types and shapes in one Network, and an edge is
free to carry ``producer.Y`` into ``consumer.X``.  So the identity a caller
holds across a Network is the *qualified* one -- a node and an operand together
-- and that is all these references are.

Resolution is the whole of this module's job.  What an interface *presents* is
next door in ``model.presentation``, which is a separate question asked of the
same reference: this module says which region value a name denotes, that one
says which of its positions arrive over a port.

It answers nothing about source identity.  Nothing here can tell whether two
region-local ``W``\\ s are the same tensor, and inferring that from a bare
operand id is exactly the guess this layer exists to stop.  Cross-stratum
correspondence -- one source operand to a tuple of qualified targets -- is the
DataflowOp's, built from its own declarations and its Design's roles and checked
against the selected Network.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.model.network import DataflowNetwork, RegionEndpoint
from finn.dataflow.model.region import InputInterface, OutputInterface, RegionInput


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


class NetworkOperandError(ValueError):
    """A reference does not resolve, or its endpoint ownership is malformed."""


def resolve_input(network: DataflowNetwork, ref: RegionInputRef) -> RegionInput:
    """Return the region input a reference names.

    Raises:
        NetworkOperandError: the node or the operand is not there, or the operand
            is declared by more than one input.
    """

    try:
        node = network.node(ref.node_id)
    except KeyError as error:
        raise NetworkOperandError(f"no node {ref.node_id!r} in the selected Network") from error
    try:
        return node.region.input(ref.operand_id)
    except KeyError as error:
        raise NetworkOperandError(
            f"node {ref.node_id!r} does not declare exactly one input for operand "
            f"{ref.operand_id!r}"
        ) from error


def resolve_output(network: DataflowNetwork, ref: RegionOutputRef) -> OutputInterface:
    """Return the output interface a reference names.

    Refuses an operand emitted by several output ports rather than picking one:
    a reference is qualified by node and operand, and if that is ambiguous the
    caller needs a port, not a preference order.

    Raises:
        NetworkOperandError: the node is not there, or the operand is produced by
            no output port or by more than one.
    """

    try:
        node = network.node(ref.node_id)
    except KeyError as error:
        raise NetworkOperandError(f"no node {ref.node_id!r} in the selected Network") from error
    matches = tuple(
        interface
        for interface in node.region.outputs
        if interface.port.operand.id == ref.operand_id
    )
    if len(matches) != 1:
        raise NetworkOperandError(
            f"node {ref.node_id!r} produces operand {ref.operand_id!r} on {len(matches)} "
            "output ports, expected one"
        )
    return matches[0]


def owned_endpoint(
    network: DataflowNetwork, ref: RegionInputRef
) -> tuple[RegionEndpoint | None, bool]:
    """Resolve a referenced input's endpoint and say whether an edge feeds it.

    Enforces the one obligation the presentation queries depend on: a ported
    input's endpoint is either the sink of exactly one edge or exposed by
    exactly one boundary, never both and never neither.  ``validate_network``
    checks that for every endpoint; this checks it for the one endpoint being
    asked about, so a query over a malformed Network refuses instead of
    returning three sets that look authoritative and are not.

    ``None`` for an internal input, which has no endpoint at all -- not an
    endpoint that happens to be fed by nothing.
    """

    item = resolve_input(network, ref)
    if not isinstance(item, InputInterface):
        return None, False
    endpoint = RegionEndpoint(ref.node_id, item.port.id)
    sinks = sum(1 for edge in network.edges for sink in edge.sinks if sink.endpoint == endpoint)
    exposures = sum(1 for boundary in network.boundaries if boundary.endpoint == endpoint)
    if sinks + exposures != 1:
        raise NetworkOperandError(
            f"endpoint {endpoint.node_id!r}.{endpoint.port_id!r} is consumed by {sinks} edges "
            f"and exposed by {exposures} boundaries; exactly one is required before its "
            "presentation can be described"
        )
    return endpoint, sinks == 1


__all__ = [
    "DataflowOperandRef",
    "NetworkOperandError",
    "RegionInputRef",
    "RegionOutputRef",
    "owned_endpoint",
    "resolve_input",
    "resolve_output",
]
