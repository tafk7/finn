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

import pytest

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
    InternalInput,
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


# -- the precondition, stated as tests -----------------------------------------
#
# These queries are pure over a Network `validate_network` has already accepted.
# They are not a validator, and the two cases below are the honest record of what
# that costs: given an invalid Network they answer from the consumer side, and
# the answer looks exactly like a good one.  Pinning that here means a later
# reader learns the contract from the suite rather than from a surprise, and a
# future change that quietly starts revalidating has to come past these.


def test_an_edge_with_no_source_is_validations_business_not_presentations():
    """`edge.source_missing_or_not_output`, and the query still answers.

    The sink endpoint is owned exactly once, which is all `_owned_endpoint`
    re-checks, so the edge-fed arm is selected and the consumer's own presented
    positions come back.  Nothing here looks at the far end of the edge.
    """

    consumer = compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    dangling = Edge(
        "weight_supply",
        RegionEndpoint("absent", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(WHOLE.image)),),
    )
    network = framed(consumer, extra_edges=(dangling,))
    reference = RegionInputRef("compute", "W")

    assert "edge.source_missing_or_not_output" in {
        issue.code for issue in validate_network(network)
    }
    assert edge_presented_positions(network, reference) == WHOLE_MATRIX.required_positions
    assert unpresented_positions(network, reference) == frozenset()


def test_an_edge_whose_sides_disagree_is_also_validations_business():
    """Element counts and position map disagree, and the query still answers.

    The supplier emits the upper column and the consumer's port declares the
    whole matrix.  A caller that skipped `validate_network` gets "the whole
    matrix is edge-presented", which is a statement about the consumer's port and
    not about what the edge can carry.
    """

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
    reference = RegionInputRef("compute", "W")

    assert {issue.code for issue in validate_network(network)} == {
        "edge.element_count_mismatch",
        "position_map.source_domain_mismatch",
    }
    assert edge_presented_positions(network, reference) == WHOLE_MATRIX.required_positions


def test_an_internal_input_is_answered_without_consulting_the_network_at_all():
    """No endpoint means no topology question, so nothing topological is read.

    The Network below is invalid -- its one edge has no source -- and the answer
    for the internal input is unaffected, because an internal input's
    presentation is a property of the Region alone.
    """

    region = compute(InternalInput(WEIGHT, WHOLE_MATRIX))
    dangling = Edge(
        "nowhere",
        RegionEndpoint("absent", "w_out"),
        (SinkContract(RegionEndpoint("absent", "w_in"), PositionMap.identity(WHOLE.image)),),
    )
    network = framed(region, extra_edges=(dangling,))
    reference = RegionInputRef("compute", "W")

    assert "edge.source_missing_or_not_output" in {
        issue.code for issue in validate_network(network)
    }
    assert exposing_ports(network, reference) == ()
    assert unpresented_positions(network, reference) == WHOLE_MATRIX.required_positions
