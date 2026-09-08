# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""U3c: the three weight-supply modes, as three resolved Networks.

The forcing case.  Every claim below is about *dataflow*: which nodes exist,
which edges connect them, which boundaries the Design presents.  Two of the
three modes have no build unit at all at this phase, and every one of them
resolves a complete Network anyway -- which is the thing U3 is for.
"""

from __future__ import annotations

import pytest

from finn.dataflow._engine import Absent, Decided, Unresolved
from finn.dataflow.model.presentation import (
    boundary_presented_positions,
    edge_presented_positions,
    exposing_ports,
    unpresented_positions,
)
from finn.dataflow.model.refs import RegionInputRef
from finn.dataflow.model.region import InputInterface, InternalInput
from finn.dataflow.kernels.memstream import MemstreamKernel
from finn.dataflow.ops.mvau.designs.dot_product import (
    DotProductDesign,
    WeightSupply,
)
from dataflow.ops.mvau.test_dot_product_design import _occurrence, _unconfigured


def _network(design: DotProductDesign):
    answer = design.dataflow.accepted_answer
    assert isinstance(answer, Decided), answer
    return answer.value


def test_external_streaming_gives_the_matrix_a_boundary_and_no_supplier() -> None:
    network = _network(_occurrence(WeightSupply.EXTERNAL))
    assert {node.id for node in network.nodes} == {"replay", "compute"}
    assert {edge.id for edge in network.edges} == {"activation_replay"}
    assert {item.id for item in network.boundaries} == {"activation", "weight", "output"}
    weight = next(item for item in network.boundaries if item.id == "weight")
    assert weight.endpoint.node_id == "compute"


def test_embedded_supply_gives_the_compute_region_no_weight_port_at_all() -> None:
    design = _occurrence(WeightSupply.EMBEDDED)
    network = _network(design)
    assert {node.id for node in network.nodes} == {"replay", "compute"}
    assert {edge.id for edge in network.edges} == {"activation_replay"}
    # No boundary to substitute, because there is nothing to substitute for.
    assert {item.id for item in network.boundaries} == {"activation", "output"}
    compute = network.node("compute")
    assert {item.port.id for item in compute.region.input_interfaces} == {"activation"}
    assert design.selected("compute") == Decided("dotp_axi_embedded")
    assert design.region_family("compute") == Decided(("mvau.dot_product.embedded", "1"))

    # The matrix is still required, and still has no endpoint anywhere.
    weight = RegionInputRef("compute", "W")
    assert isinstance(compute.region.input("W"), InternalInput)
    assert exposing_ports(network, weight) == ()
    assert edge_presented_positions(network, weight) == frozenset()
    assert boundary_presented_positions(network, weight) == frozenset()
    assert unpresented_positions(network, weight) == (
        compute.region.input("W").requirements.required_positions
    )


def test_decoupled_supply_gives_the_matrix_its_own_node_and_edge() -> None:
    design = _occurrence(WeightSupply.DECOUPLED)
    network = _network(design)
    assert {node.id for node in network.nodes} == {"replay", "compute", "memory"}
    assert {edge.id for edge in network.edges} == {"activation_replay", "weight_supply_edge"}
    # The matrix no longer crosses the Design's boundary; it is produced inside.
    assert {item.id for item in network.boundaries} == {"activation", "output"}
    edge = next(item for item in network.edges if item.id == "weight_supply_edge")
    assert edge.source.node_id == "memory"
    assert edge.sinks[0].endpoint.node_id == "compute"
    memory = design.kernel("memory")
    assert isinstance(memory, Decided)
    assert isinstance(memory.value, MemstreamKernel)


def test_decoupled_supply_makes_both_qualified_weight_requirements_real() -> None:
    """``memory.W`` is unpresented; ``compute.W`` is edge-presented.

    The corrected decoupled evidence, and the reason the supplier needed an
    internal input of its own: before it had one, the Network could show the
    matrix arriving at ``compute`` and say nothing at all about the node it
    arrived from.  Nothing here relates the two ``W``\\ s -- they are two
    region-local operands at two nodes, and source correspondence is S2-A's.
    """

    network = _network(_occurrence(WeightSupply.DECOUPLED))
    memory = RegionInputRef("memory", "W")
    compute = RegionInputRef("compute", "W")
    supplied = network.node("memory").region.input("W")
    consumed = network.node("compute").region.input("W")

    assert isinstance(supplied, InternalInput)
    assert exposing_ports(network, memory) == ()
    assert unpresented_positions(network, memory) == supplied.requirements.required_positions
    assert edge_presented_positions(network, memory) == frozenset()
    assert boundary_presented_positions(network, memory) == frozenset()

    assert isinstance(consumed, InputInterface)
    assert edge_presented_positions(network, compute) == consumed.requirements.required_positions
    assert unpresented_positions(network, compute) == frozenset()
    assert boundary_presented_positions(network, compute) == frozenset()


def test_the_compute_region_is_identical_in_external_and_decoupled_supply() -> None:
    """Where the matrix comes from does not change what the arithmetic means."""

    external = _network(_occurrence(WeightSupply.EXTERNAL)).node("compute").region
    decoupled = _network(_occurrence(WeightSupply.DECOUPLED)).node("compute").region
    assert external == decoupled
    # And the embedded one genuinely differs, in the one interface.
    embedded = _network(_occurrence(WeightSupply.EMBEDDED)).node("compute").region
    assert embedded != external
    assert embedded.schedule == external.schedule
    assert embedded.outputs == external.outputs
    assert embedded.input("X") == external.input("X")
    # Same weight operand and the same requirement of it; only the port differs.
    assert embedded.input("W").operand == external.input("W").operand
    assert embedded.input("W").requirements == external.input("W").requirements


def test_the_supplier_produces_exactly_what_the_consumer_requires() -> None:
    """The edge is checked position by position, not by a shared import."""

    design = _occurrence(WeightSupply.DECOUPLED)
    network = _network(design)
    produced = network.node("memory").region.output_interface("weight").port
    consumed = network.node("compute").region.input_interface("weight").port
    assert produced.beat_sequence == consumed.beat_sequence
    assert produced.operand.shape == consumed.operand.shape


def test_every_mode_resolves_its_network_with_no_build_unit_available() -> None:
    """Two of the three suppliers cannot be built yet, and it changes nothing."""

    for supply in (WeightSupply.EMBEDDED, WeightSupply.DECOUPLED):
        design = _occurrence(supply)
        assessment = design.dataflow
        assert assessment.readiness.ready is True
        assert isinstance(assessment.accepted_answer, Decided)

    embedded = _occurrence(WeightSupply.EMBEDDED).kernel("compute")
    assert isinstance(embedded, Decided)
    unavailable = embedded.value.physical.accepted_answer
    assert isinstance(unavailable, Absent)
    assert any(finding.code == "kernel-physically-unsupported" for finding in unavailable.findings)

    memory = _occurrence(WeightSupply.DECOUPLED).kernel("memory")
    assert isinstance(memory, Decided)
    assert isinstance(memory.value.physical.accepted_answer, Absent)


def test_a_mode_that_keeps_the_matrix_locally_needs_one_to_keep() -> None:
    for supply in (WeightSupply.EMBEDDED, WeightSupply.DECOUPLED):
        answer = _occurrence(supply, initializer=False).dataflow.accepted_answer
        assert isinstance(answer, Absent), supply
        assert "mvau-local-weights-need-an-initializer" in {
            finding.code for finding in answer.findings
        }


def test_an_initializer_neither_forces_nor_forbids_external_streaming() -> None:
    """The fact narrows what is available; it never picks."""

    for initializer in (True, False):
        design = _occurrence(WeightSupply.EXTERNAL, initializer=initializer)
        assert isinstance(design.dataflow.accepted_answer, Decided)
    # And with an initializer present, all three remain available.
    for supply in WeightSupply:
        assert isinstance(_occurrence(supply).dataflow.accepted_answer, Decided)


def test_the_mode_is_uncommitted_until_it_is_chosen() -> None:
    design = _unconfigured()
    design = design.assign(DotProductDesign.pe, 2).assign(DotProductDesign.simd, 2)
    assert isinstance(design.dataflow.accepted_answer, Unresolved)
    assert design.is_active("memory") == Decided(False) or isinstance(
        design.is_active("memory"), Unresolved
    )


def test_supply_without_a_compute_candidate_is_unresolved() -> None:
    design = _unconfigured()
    design = (
        design.assign(DotProductDesign.pe, 2)
        .assign(DotProductDesign.simd, 2)
        .assign(DotProductDesign.weight_supply, WeightSupply.EXTERNAL)
    )
    assert isinstance(design.dataflow.accepted_answer, Unresolved)


@pytest.mark.parametrize(
    ("supply", "candidate"),
    [
        (WeightSupply.EXTERNAL, "dotp_axi_embedded"),
        (WeightSupply.EMBEDDED, "dotp_axi"),
        (WeightSupply.DECOUPLED, "dotp_axi_embedded"),
    ],
)
def test_inconsistent_supply_and_compute_candidates_are_refused(
    supply: WeightSupply, candidate: str
) -> None:
    design = _unconfigured()
    design = (
        design.assign(DotProductDesign.pe, 2)
        .assign(DotProductDesign.simd, 2)
        .assign(DotProductDesign.weight_supply, supply)
    )
    design = design.compute.select(candidate).root.design
    answer = design.dataflow.accepted_answer
    assert isinstance(answer, Absent)
    assert answer.findings


@pytest.mark.parametrize("supply", list(WeightSupply))
def test_the_memory_role_is_present_exactly_when_the_mode_says(supply: WeightSupply) -> None:
    design = _occurrence(supply)
    expected = supply is WeightSupply.DECOUPLED
    assert design.is_active("memory") == Decided(expected)
    assert design.is_active("compute") == Decided(True)
    assert ("memory" in {node.id for node in _network(design).nodes}) is expected
