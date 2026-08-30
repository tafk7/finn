# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 6g: what every physical object traces back to, and what it does not.

Item 8 of the migration plan's evidence list asks for complete associations.
Phase 4 recorded it as partial and said exactly which half was missing:
port-level association **for the fused reading**, which needs a fused
elaborator producing an ``MVAUPhysicalElaboration``.

Those are two different claims and only one of them was ever open.

**The decomposed reading is complete, and this module proves it** rather than
leaving "partial" to stand for the whole item.  Every component, interface and
connection the composition builds names the Regions it realizes, the semantic
ports it carries, the decisions that chose and configured it, and the Kernels
behind it -- with no entry empty and nothing named that the Network does not
have.

**The fused reading stays open, and §4 of this docstring says why.**  Not
because it is hard: because nothing consumes it.  ``elaborate_fused`` would be
a stage with no caller, and Phase 5's review already rejected inventing a stage
boundary that no consumer asked for.  Since Phase 6e there is a second reason,
and it is the stronger one -- the fused core computes the wrong values on
DSP48E1, so elaborating it into artifacts would make a defective core
buildable.  ``test_fused_hardware`` asserts the absence; this module records
the reason it is still an absence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from finn.dataflow.mvau.decomposed import ACTIVATION_EDGE, DOT_PRODUCT_NODE, REPLAY_NODE
from finn.dataflow.mvau.elaboration import MVAUPhysicalElaboration
from finn.dataflow.mvau.hardware import composition
from finn.dataflow.mvau.hardware.composition import elaborate_decomposed
from finn.dataflow.ops.mvau import NetworkRef

FINN_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(name="elaboration")
def _elaboration() -> MVAUPhysicalElaboration:
    operation = _committed(_model())
    return elaborate_decomposed(operation.resolve_dataflow(_context()))


def _by_id(elaboration: MVAUPhysicalElaboration) -> dict[str, object]:
    return {item.physical_id: item for item in elaboration.associations}


# -- every physical object is accounted for ------------------------------------


def test_every_component_interface_and_connection_has_an_association(
    elaboration: MVAUPhysicalElaboration,
) -> None:
    """No physical object without provenance, and none the other way round.

    Both directions matter.  An object with no association is one nobody can
    trace; an association naming an object that does not exist is a record of
    something that was never built.
    """

    physical = {
        *(item.id for item in elaboration.components),
        *(item.id for item in elaboration.numeric_interfaces),
        *(item.id for item in elaboration.control_interfaces),
        *(item.id for item in elaboration.connections),
    }
    associated = set(_by_id(elaboration))
    assert associated == physical


def test_no_association_is_empty(elaboration: MVAUPhysicalElaboration) -> None:
    """An association naming nothing is the same as none at all.

    The replay buffer is the case that makes this worth asserting: it declares
    no choices of its own, so its decision record is entirely *imported* --
    ``PE`` and ``SIMD`` dimension it without being its to pick.  An empty record
    for hardware the folding literally sizes would be the provenance failure
    this exists to prevent.
    """

    for association in elaboration.associations:
        assert association.semantic_region_ids, association.physical_id
        assert association.decision_paths, association.physical_id
        assert association.kernel_ids, association.physical_id


# -- and traces back to something that exists ----------------------------------


def test_every_named_region_edge_and_port_is_in_the_selected_network(
    elaboration: MVAUPhysicalElaboration,
) -> None:
    """Provenance has to point at the Network that was actually selected.

    A region id or port that no node carries would be a plausible-looking
    string rather than a reference -- the same class of defect as the reported
    signal names fixture 7 found, one layer up.
    """

    result = elaboration.semantic_result
    assert isinstance(result, NetworkRef)
    network = result.network

    nodes = {node.id for node in network.nodes}
    edges = {edge.id for edge in network.edges}
    ports = {
        (node.id, interface.port.id)
        for node in network.nodes
        for interface in node.region.interfaces
    }

    for association in elaboration.associations:
        for region_id in association.semantic_region_ids:
            assert region_id in nodes, (association.physical_id, region_id)
        for edge_id in association.semantic_edge_ids:
            assert edge_id in edges, (association.physical_id, edge_id)
        for port in association.semantic_ports:
            assert (port.region_id, port.port_id) in ports, (association.physical_id, port)


def test_every_named_decision_is_a_decision_of_this_design_space(
    elaboration: MVAUPhysicalElaboration,
) -> None:
    """A decision path that resolves nothing records nothing."""

    operation = _committed(_model())
    resolved = operation.resolve_dataflow(_context())
    space = resolved.point.design_space
    known = set(space.decisions) | set(space.properties)

    for association in elaboration.associations:
        for path in association.decision_paths:
            assert path in known, (association.physical_id, str(path))


# -- port level, specifically --------------------------------------------------


def test_the_two_cores_carry_the_ports_of_the_regions_they_realize(
    elaboration: MVAUPhysicalElaboration,
) -> None:
    """The half of item 8 Phase 4 could not do for the fused reading.

    Each bound Kernel's component is associated with every port of the Region
    it covers -- not merely with the Region -- which is what "complete
    associations" asks for.
    """

    result = elaboration.semantic_result
    assert isinstance(result, NetworkRef)
    network = result.network
    associations = _by_id(elaboration)

    for node_id in (REPLAY_NODE, DOT_PRODUCT_NODE):
        expected = {
            (node_id, interface.port.id) for interface in network.node(node_id).region.interfaces
        }
        found = {
            (port.region_id, port.port_id)
            for association in associations.values()
            for port in association.semantic_ports  # type: ignore[attr-defined]
            if node_id in association.semantic_region_ids  # type: ignore[attr-defined]
        }
        assert expected <= found, node_id


def test_the_absorbed_edge_is_recorded_where_it_was_absorbed(
    elaboration: MVAUPhysicalElaboration,
) -> None:
    """The connection that became a wire still names the edge it realizes."""

    edges = {
        edge_id
        for association in elaboration.associations
        for edge_id in association.semantic_edge_ids
    }
    assert ACTIVATION_EDGE in edges


# -- what stays open, and why --------------------------------------------------


def test_the_fused_reading_has_no_elaborator_and_that_is_the_open_half() -> None:
    """Item 8's remaining gap, stated as a fact about the code.

    Port-level association for the fused Kernel needs ``elaborate_fused``.
    There isn't one, and this asserts there isn't -- so if one appears, this
    fails and whoever added it has to close the association half with it rather
    than leave the record saying the gap is still open.

    Two reasons it stays open, and the second is the stronger:

    - Nothing selects the fused Kernel, so an elaborator would be a stage with
      no consumer.  The Phase 5 review rejected exactly that reasoning when it
      was applied to collapsing the synthesis stage: stage boundaries follow
      actual consumers.
    - Phase 6e measured the fused core computing the wrong values on DSP48E1.
      Giving it an artifact path would make a core with a known defect
      buildable, which is a worse outcome than an open evidence item.
    """

    assert not hasattr(composition, "elaborate_fused")
    assert [name for name in composition.__all__ if "elaborate" in name] == ["elaborate_decomposed"]
