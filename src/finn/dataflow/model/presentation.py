# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a Network's interfaces present, asked of one qualified reference.

Two questions, over the qualified references ``model.refs`` resolves:

```text
exposure       which endpoints and boundaries present this operand
presentation   which of its required positions arrive over a port, fed by a
               Network edge or exposed at the Network boundary, and which arrive
               over none
```

The two axes are orthogonal and must not be confused with each other:

```text
Region input kind      InputInterface | InternalInput
Network presentation   edge-presented | boundary-presented | unpresented
```

An ``InternalInput`` has no endpoint, so it is neither edge- nor
boundary-presented; every position it requires is unpresented.  The converse
does not hold: a ported input can have unpresented positions too, and does
whenever its beat sequence covers less than its requirements name.

**Presented, not supplied.**  A position presented once may satisfy several
scheduled occurrences through binding-owned replay, and ``REGION.md`` 3.7
refuses any required-versus-presented equality for inputs for that reason.  The
sets below describe the dataflow interface.  They are not a binding witness, and
an empty ``unpresented_positions`` does not mean the region's requirements are
satisfiable -- that is the realizability obligation in section 5.2.  Nor does a
non-empty one name a storage technology: unpresented is a statement about ports,
not about where the data lives.

**Precondition: a structurally valid Network.**  Every function here is a pure
query over a Network that ``validate_network`` has already accepted.  They do
not revalidate, and they are not a validator wearing another name.  Ask
``validate_network`` once and then ask these as often as you like; asking them of
a Network that has not been validated is a caller error, and the answers are
undefined rather than false.

That split is deliberate.  Presentation is meant to be cheap and to be asked
many times -- once per qualified reference during S2-A correspondence, and again
per obligation in U6 -- while validity is a whole-Network property that is
established once.  Folding validation into each query would make the common case
pay for it repeatedly and would still not be a validator, because a query is
scoped to one reference and validity is not.

Concretely, these functions do **not** check that an edge's source exists or is
an output, that its position map is total, that element types or beat sequences
agree across it, that the referenced Region is itself valid, or that the Network
is acyclic.  ``validate_network`` checks all of those, with codes such as
``edge.source_missing_or_not_output``, ``edge.beat_sequence_mismatch`` and
``network.cycle``.  Given a Network carrying one of those defects, a query here
will answer from the consumer side and the answer will look authoritative.  The
one condition ``_owned_endpoint`` re-checks locally is endpoint ownership, and it
does so because that single fact is what selects between the edge and boundary
arms -- not as a substitute for validation.
"""

from __future__ import annotations

from finn.dataflow.model.network import BoundaryContract, DataflowNetwork, RegionEndpoint
from finn.dataflow.model.refs import (
    DataflowOperandRef,
    NetworkOperandError,
    RegionInputRef,
    resolve_input,
    resolve_output,
)
from finn.dataflow.model.region import Coordinate, InputInterface, RegionInput


def exposing_ports(network: DataflowNetwork, ref: DataflowOperandRef) -> tuple[RegionEndpoint, ...]:
    """Return the endpoints presenting a referenced requirement or product.

    Empty for an internal input, which is the whole point of the case: there is
    no endpoint, rather than an endpoint with an empty sequence.

    Requires a Network ``validate_network`` has accepted.
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

    Requires a Network ``validate_network`` has accepted.
    """

    endpoints = set(exposing_ports(network, ref))
    return tuple(boundary for boundary in network.boundaries if boundary.endpoint in endpoints)


def _owned_endpoint(
    network: DataflowNetwork, ref: RegionInputRef
) -> tuple[RegionEndpoint | None, bool]:
    """Resolve a referenced input's endpoint and say whether an edge feeds it.

    Re-checks the one condition the *arm selection* depends on: a ported input's
    endpoint is either the sink of exactly one edge or exposed by exactly one
    boundary, never both and never neither.  Without exactly one owner there is
    no edge-versus-boundary answer to give, so this refuses rather than picking.

    That is arm selection, not validation.  ``validate_network`` checks this for
    every endpoint along with everything else, and callers are required to have
    run it (see the module docstring).  A Network that violates any *other*
    structural rule will still be answered here, from the consumer side.

    ``None`` for an internal input, which has no endpoint at all -- not an
    endpoint that happens to be fed by nothing.

    Private on purpose.  The Boolean is a step in computing the position sets
    below, not an answer: "fed by an edge" reads as a disposition, and a caller
    that took it as one would miss exactly the case
    ``test_an_edge_fed_port_can_still_present_only_part_of_its_requirement``
    exists for.  Ask the position-granular queries.
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


def edge_presented_positions(
    network: DataflowNetwork, ref: RegionInputRef
) -> frozenset[Coordinate]:
    """Required positions a port presents that a Network edge feeds.

    Requires a Network ``validate_network`` has accepted.  In particular this
    does not check that the feeding edge has a source, that the source is an
    output, or that the two sides agree on element type and beat sequence.
    """

    endpoint, fed_by_edge = _owned_endpoint(network, ref)
    if endpoint is None or not fed_by_edge:
        return frozenset()
    item = resolve_input(network, ref)
    return item.requirements.required_positions & _presented(item)


def boundary_presented_positions(
    network: DataflowNetwork, ref: RegionInputRef
) -> frozenset[Coordinate]:
    """Required positions a port presents that a Network boundary exposes.

    Requires a Network ``validate_network`` has accepted.
    """

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

    Requires a Network ``validate_network`` has accepted.
    """

    _owned_endpoint(network, ref)
    item = resolve_input(network, ref)
    return item.requirements.required_positions - _presented(item)


def _presented(item: RegionInput) -> frozenset[Coordinate]:
    return item.port.beat_sequence.image if isinstance(item, InputInterface) else frozenset()


__all__ = [
    "boundary_presented_positions",
    "edge_presented_positions",
    "exposing_boundaries",
    "exposing_ports",
    "unpresented_positions",
]
