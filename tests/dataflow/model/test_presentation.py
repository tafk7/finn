# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a Network's interfaces present, asked of the three supply forms.

Two orthogonal axes meet here and must not be conflated:

```text
Region input kind      InputInterface | InternalInput
Network presentation   edge-presented | boundary-presented | unpresented
```

An ``InternalInput`` has no endpoint, so every position it requires is
unpresented -- but the converse fails, and the partial-presentation case below
is the counterexample: a ported input fed by an edge can still leave a residue.
"""

import ast
from pathlib import Path

import pytest

from finn.dataflow.model import network_validation, presentation
from finn.dataflow.model.network import (
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.network_validation import validate_network
from finn.dataflow.model.presentation import (
    boundary_presented_positions,
    edge_presented_positions,
    exposing_boundaries,
    exposing_ports,
    unpresented_positions,
)
from finn.dataflow.model.refs import NetworkOperandError, RegionInputRef
from finn.dataflow.model.region import (
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from finn.dataflow.model.region_validation import validate_region
from dataflow.model.supply_networks import (
    ACTIVATION,
    RESULT,
    SEQUENTIAL,
    UPPER,
    WEIGHT,
    WHOLE,
    WHOLE_MATRIX,
    boundary,
    compute,
    decoupled_network,
    external_network,
    framed,
    internal_input_network,
    partly_supplied_network,
    supplier,
)

# -- the three forms, asked the same question ---------------------------------


def test_a_boundary_exposed_matrix_is_presented_at_the_boundary():
    network = external_network()
    reference = RegionInputRef("compute", "W")

    assert not validate_network(network)
    assert boundary_presented_positions(network, reference) == WHOLE_MATRIX.required_positions
    assert edge_presented_positions(network, reference) == frozenset()
    assert unpresented_positions(network, reference) == frozenset()
    assert tuple(item.id for item in exposing_boundaries(network, reference)) == ("weight",)


def test_an_internal_matrix_is_required_and_presented_nowhere():
    """The case the model could not state at all before.

    Not "embedded", not "local storage": the region requires the operand and no
    port presents it.  What covers the unpresented positions is the binding's.
    """

    network = internal_input_network()
    reference = RegionInputRef("compute", "W")

    assert not validate_network(network)
    assert unpresented_positions(network, reference) == WHOLE_MATRIX.required_positions
    assert edge_presented_positions(network, reference) == frozenset()
    assert boundary_presented_positions(network, reference) == frozenset()
    assert exposing_ports(network, reference) == ()
    assert exposing_boundaries(network, reference) == ()


def test_a_matrix_supplied_over_an_edge_is_edge_presented():
    """The decoupled evidence: both qualified requirements are real.

    ``memory.W`` is an internal input and unpresented; ``compute.W`` is a ported
    input the weight edge feeds.  One matrix, two nodes, two different answers,
    and no rule anywhere that relates the two ``W``\\ s.
    """

    network = decoupled_network()
    consumer = RegionInputRef("compute", "W")
    memory = RegionInputRef("memory", "W")

    assert not validate_network(network)
    assert edge_presented_positions(network, consumer) == WHOLE_MATRIX.required_positions
    assert unpresented_positions(network, consumer) == frozenset()
    assert boundary_presented_positions(network, consumer) == frozenset()
    # The supplier requires the matrix too, and nothing presents it there.
    assert unpresented_positions(network, memory) == frozenset(WHOLE.image)
    assert edge_presented_positions(network, memory) == frozenset()
    assert boundary_presented_positions(network, memory) == frozenset()
    assert exposing_ports(network, memory) == ()


def test_an_edge_fed_port_can_still_present_only_part_of_its_requirement():
    """Position-granular, because a boolean would lose the remainder.

    ``w_in`` is the sink of ``weight_supply`` and presents two of the four
    required positions.  A caller asking "does the Network supply this input"
    and getting ``True`` would never learn about the other two.
    """

    network = partly_supplied_network()
    consumer = RegionInputRef("compute", "W")

    assert not validate_network(network)
    assert edge_presented_positions(network, consumer) == frozenset(UPPER.image)
    assert unpresented_positions(network, consumer) == frozenset({(0, 0), (1, 0)})
    assert boundary_presented_positions(network, consumer) == frozenset()


def test_one_presentation_can_serve_several_scheduled_occurrences():
    """A re-read is not an unpresented position.

    ``REGION.md`` 3.7 refuses a required-versus-presented equality for inputs
    precisely so a multi-visit kernel can declare its re-reads as multiplicity
    rather than as extra boundary beats.  The position has arrived; serving the
    repeats is binding-owned, and these queries do not pretend otherwise.
    """

    schedule = LogicalSchedule((ScheduleLevel("step", 2), ScheduleLevel("visit", 3)))
    thrice = ScheduledInputRequirements(
        {((step, visit), (step,)): 1 for step in range(2) for visit in range(3)}
    )
    region = DataflowRegion(
        schedule,
        (InputInterface(Port("x_in", ACTIVATION, SEQUENTIAL), thrice),),
        (
            OutputInterface(
                Port("y_out", RESULT, SEQUENTIAL),
                ScheduledOutputAvailability({(0,): (0, 2), (1,): (1, 2)}),
            ),
        ),
    )
    network = DataflowNetwork(
        (NetworkNode("compute", region),),
        (),
        (
            boundary("compute", region.input_interface("x_in").port, "activation"),
            boundary("compute", region.output_interface("y_out").port, "output"),
        ),
    )
    reference = RegionInputRef("compute", "X")

    assert not validate_region(region)
    assert thrice.occurrence_count == 6
    assert region.input_interface("x_in").port.beat_sequence.delivered_field_count == 2
    assert unpresented_positions(network, reference) == frozenset()
    assert boundary_presented_positions(network, reference) == thrice.required_positions


# -- malformed endpoint ownership is refused, not described -------------------


def test_presentation_queries_refuse_an_endpoint_no_one_owns():
    """A malformed Network gets a refusal, not three plausible sets.

    An endpoint neither fed by an edge nor exposed by a boundary has no
    presentation to describe, and answering "edge none, boundary none, all
    unpresented" would read as a fact about an internal input rather than as a
    broken Network.
    """

    unowned = framed(
        compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX)),
        weight_boundary=False,
    )
    reference = RegionInputRef("compute", "W")

    assert validate_network(unowned)
    with pytest.raises(NetworkOperandError, match="exactly one is required"):
        unpresented_positions(unowned, reference)


def test_presentation_queries_refuse_an_endpoint_owned_twice():
    consumer = compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    edge = Edge(
        "weight_supply",
        RegionEndpoint("memory", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(WHOLE.image)),),
    )
    doubly_owned = framed(
        consumer,
        extra_nodes=(NetworkNode("memory", supplier(WHOLE)),),
        extra_edges=(edge,),
        weight_boundary=True,
    )

    with pytest.raises(NetworkOperandError, match="exactly one is required"):
        edge_presented_positions(doubly_owned, RegionInputRef("compute", "W"))


# -- the precondition, and who discharges it ----------------------------------
#
# These queries are pure over a Network `validate_network` has already accepted.
# The tests below establish two halves of that arrangement: that validation
# really does catch the defects the precondition excludes, so the precondition is
# discharged rather than merely asserted; and that the queries do not quietly
# re-run it, which is the performance claim the split exists for.
#
# Deliberately absent: any assertion about what a query *returns* for an invalid
# Network. The contract says that is undefined, and a test pinning today's answer
# would make undefined behaviour normative -- a future implementation that cheaply
# detected one of these defects and raised would be legal under the precondition
# and would read as a regression.


def test_validation_catches_an_edge_with_no_source():
    """The precondition is discharged by `validate_network`, not by hope.

    Presentation does not look at the far end of an edge, so a dangling source
    has to be caught before these queries are asked. It is.
    """

    consumer = compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    dangling = Edge(
        "weight_supply",
        RegionEndpoint("absent", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(WHOLE.image)),),
    )

    assert "edge.source_missing_or_not_output" in {
        issue.code for issue in validate_network(framed(consumer, extra_edges=(dangling,)))
    }


def test_validation_catches_an_edge_whose_sides_disagree():
    """The supplier emits the upper column; the consumer's port declares the whole
    matrix. Presentation compares neither side against the other, so this too has
    to be caught first."""

    consumer = compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    mismatched = Edge(
        "weight_supply",
        RegionEndpoint("memory", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(WHOLE.image)),),
    )
    network = framed(
        consumer,
        extra_nodes=(NetworkNode("memory", supplier(UPPER)),),
        extra_edges=(mismatched,),
    )

    assert {issue.code for issue in validate_network(network)} == {
        "edge.element_count_mismatch",
        "position_map.source_domain_mismatch",
    }


def test_presentation_does_not_revalidate_the_network(monkeypatch):
    """Ask validation once, then ask these as often as you like.

    The whole reason presentation carries a precondition instead of checking for
    itself is that it is asked many times -- per qualified reference in S2-A
    correspondence, per obligation in U6 -- while validity is a whole-Network
    property established once. This is that claim as a test: break
    `validate_network` outright, then run every query over a Network that is in
    fact valid. If any of them reached for it, this fails.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("presentation must not call validate_network")

    monkeypatch.setattr(network_validation, "validate_network", refuse)

    network = decoupled_network()
    consumer = RegionInputRef("compute", "W")
    memory = RegionInputRef("memory", "W")

    assert exposing_ports(network, consumer)
    assert exposing_boundaries(network, RegionInputRef("compute", "X"))
    assert edge_presented_positions(network, consumer) == WHOLE_MATRIX.required_positions
    assert boundary_presented_positions(network, consumer) == frozenset()
    assert unpresented_positions(network, memory) == frozenset(WHOLE.image)


def test_presentation_names_no_validator_at_all():
    """The static half of the same claim, immune to how the import is spelled.

    A monkeypatched module attribute only proves the call did not go through
    *that* name. This proves the module never mentions validation, so there is no
    other name it could have gone through.
    """

    source = Path(presentation.__file__).read_text()
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("validation" in name for name in imported)
