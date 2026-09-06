# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Qualified references, and what a Network's interfaces present.

The three supply forms of a matrix are the reason this module exists, so they
are built here as Networks and asked the same question.  What separates them is
not a classification stored anywhere -- it is which positions arrive over a
port, and from where.
"""

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.input_service import (
    InputServiceError,
    RegionInputRef,
    RegionOutputRef,
    exposing_boundaries,
    exposing_ports,
    externally_presented_positions,
    internally_presented_positions,
    resolve_input,
    resolve_output,
    unpresented_positions,
)
from finn.dataflow.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.network_validation import validate_network
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    UnportedInput,
)
from finn.dataflow.region_validation import validate_region

WEIGHT_TYPE = DataType["INT8"]
ACTIVATION_TYPE = DataType["INT8"]
OUTPUT_TYPE = DataType["INT32"]

#: A two-by-two matrix, and the requirement that every position of it is used
#: at both steps.  Small enough that the presented sets can be written out.
WEIGHT = Operand("W", WEIGHT_TYPE, (2, 2))
ACTIVATION = Operand("X", ACTIVATION_TYPE, (2,))
RESULT = Operand("Y", OUTPUT_TYPE, (2,))

WHOLE_MATRIX = ScheduledInputRequirements(
    {((step,), (row, column)): 1 for step in range(2) for row in range(2) for column in range(2)}
)
EVERY_STEP = ScheduledInputRequirements({((step,), (step,)): 1 for step in range(2)})

#: The upper column only.  A port carrying this presents half of ``W``.
UPPER = BeatSequence(1, (((0, 1),), ((1, 1),)))
#: Both columns, in step order.
WHOLE = BeatSequence(2, (((0, 0), (0, 1)), ((1, 0), (1, 1))))

SCHEDULE = LogicalSchedule((ScheduleLevel("step", 2),))
SEQUENTIAL = BeatSequence(1, (((0,),), ((1,),)))


def _activation_input():
    return InputInterface(Port("x_in", ACTIVATION, SEQUENTIAL), EVERY_STEP)


def _result_output():
    return OutputInterface(
        Port("y_out", RESULT, SEQUENTIAL),
        ScheduledOutputAvailability({(0,): (0,), (1,): (1,)}),
    )


def _compute(weight_input):
    return DataflowRegion(SCHEDULE, (_activation_input(), weight_input), (_result_output(),))


def _supplier(sequence):
    """A rank-zero source that requires what it emits and exposes no input port."""

    return DataflowRegion(
        LogicalSchedule(()),
        (
            UnportedInput(
                WEIGHT,
                ScheduledInputRequirements({((), position): 1 for position in sequence.image}),
            ),
        ),
        (
            OutputInterface(
                Port("w_out", WEIGHT, sequence),
                ScheduledOutputAvailability({position: () for position in sequence.image}),
            ),
        ),
    )


def _boundary(node_id, port, boundary_id):
    return BoundaryContract(boundary_id, RegionEndpoint(node_id, port.id), port.beat_sequence)


def _framed(compute, extra_nodes=(), extra_edges=(), weight_boundary=False):
    boundaries = [
        _boundary("compute", compute.input_interface("x_in").port, "activation"),
        _boundary("compute", compute.output_interface("y_out").port, "output"),
    ]
    if weight_boundary:
        boundaries.append(_boundary("compute", compute.input_interface("w_in").port, "weight"))
    return DataflowNetwork(
        (NetworkNode("compute", compute), *extra_nodes),
        tuple(extra_edges),
        tuple(boundaries),
    )


def external_network():
    """The matrix crosses the Network's edge: one port, exposed at a boundary."""

    compute = _compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    return _framed(compute, weight_boundary=True)


def unported_network():
    """The matrix is required and no port presents it."""

    return _framed(_compute(UnportedInput(WEIGHT, WHOLE_MATRIX)))


def _supplied_network(sequence):
    compute = _compute(InputInterface(Port("w_in", WEIGHT, sequence), WHOLE_MATRIX))
    supplier = _supplier(sequence)
    edge = Edge(
        "weight_supply",
        RegionEndpoint("memory", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(sequence.image)),),
    )
    return _framed(compute, extra_nodes=(NetworkNode("memory", supplier),), extra_edges=(edge,))


def decoupled_network():
    """A second node presents the whole matrix over an edge."""

    return _supplied_network(WHOLE)


def partly_supplied_network():
    """An edge feeds the port, and the port still presents only half.

    The case that makes a boolean "is this input fed by an edge" answer wrong.
    """

    return _supplied_network(UPPER)


# -- the three forms, asked the same question ---------------------------------


def test_a_boundary_exposed_matrix_is_presented_from_outside():
    network = external_network()
    reference = RegionInputRef("compute", "W")

    assert not validate_network(network)
    assert externally_presented_positions(network, reference) == WHOLE_MATRIX.required_positions
    assert internally_presented_positions(network, reference) == frozenset()
    assert unpresented_positions(network, reference) == frozenset()
    assert tuple(item.id for item in exposing_boundaries(network, reference)) == ("weight",)


def test_an_unported_matrix_is_required_and_presented_nowhere():
    """The case the model could not state at all before.

    Not "embedded", not "local storage": the region requires the operand and no
    port presents it.  What covers the unpresented positions is the binding's.
    """

    network = unported_network()
    reference = RegionInputRef("compute", "W")

    assert not validate_network(network)
    assert unpresented_positions(network, reference) == WHOLE_MATRIX.required_positions
    assert internally_presented_positions(network, reference) == frozenset()
    assert externally_presented_positions(network, reference) == frozenset()
    assert exposing_ports(network, reference) == ()
    assert exposing_boundaries(network, reference) == ()


def test_a_matrix_supplied_over_an_edge_is_presented_from_inside():
    network = decoupled_network()
    consumer = RegionInputRef("compute", "W")
    supplier = RegionInputRef("memory", "W")

    assert not validate_network(network)
    assert internally_presented_positions(network, consumer) == WHOLE_MATRIX.required_positions
    assert unpresented_positions(network, consumer) == frozenset()
    # The supplier requires the matrix too, and nothing presents it there.
    assert unpresented_positions(network, supplier) == frozenset(WHOLE.image)
    assert exposing_ports(network, supplier) == ()


def test_an_edge_fed_port_can_still_present_only_part_of_its_requirement():
    """Position-granular, because a boolean would lose the remainder.

    ``w_in`` is the sink of ``weight_supply`` and presents two of the four
    required positions.  A caller asking "does the Network supply this input"
    and getting ``True`` would never learn about the other two.
    """

    network = partly_supplied_network()
    consumer = RegionInputRef("compute", "W")

    assert not validate_network(network)
    assert internally_presented_positions(network, consumer) == frozenset(UPPER.image)
    assert unpresented_positions(network, consumer) == frozenset({(0, 0), (1, 0)})
    assert externally_presented_positions(network, consumer) == frozenset()


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
            _boundary("compute", region.input_interface("x_in").port, "activation"),
            _boundary("compute", region.output_interface("y_out").port, "output"),
        ),
    )
    reference = RegionInputRef("compute", "X")

    assert not validate_region(region)
    assert thrice.occurrence_count == 6
    assert region.input_interface("x_in").port.beat_sequence.delivered_field_count == 2
    assert unpresented_positions(network, reference) == frozenset()
    assert externally_presented_positions(network, reference) == thrice.required_positions


# -- references resolve, and refuse what they cannot answer -------------------


def test_a_qualified_reference_resolves_an_input_and_an_output():
    network = external_network()

    assert resolve_input(network, RegionInputRef("compute", "W")) == network.node(
        "compute"
    ).region.input("W")
    assert resolve_output(network, RegionOutputRef("compute", "Y")) == network.node(
        "compute"
    ).region.output_interface("y_out")


@pytest.mark.parametrize(
    "reference",
    [RegionInputRef("absent", "W"), RegionInputRef("compute", "Q")],
    ids=["unknown node", "unknown operand"],
)
def test_an_unresolvable_reference_is_refused(reference):
    with pytest.raises(InputServiceError):
        resolve_input(external_network(), reference)


def test_an_output_reference_refuses_an_operand_it_cannot_disambiguate():
    with pytest.raises(InputServiceError):
        resolve_output(external_network(), RegionOutputRef("compute", "W"))


def test_presentation_queries_refuse_an_endpoint_no_one_owns():
    """A malformed Network gets a refusal, not three plausible sets.

    An endpoint neither fed by an edge nor exposed by a boundary has no
    presentation to describe, and answering "internally none, externally none,
    all unpresented" would read as a fact about an embedded operand rather than
    as a broken Network.
    """

    compute = _compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    unowned = _framed(compute, weight_boundary=False)
    reference = RegionInputRef("compute", "W")

    assert validate_network(unowned)
    with pytest.raises(InputServiceError, match="exactly one is required"):
        unpresented_positions(unowned, reference)


def test_presentation_queries_refuse_an_endpoint_owned_twice():
    compute = _compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX))
    supplier = _supplier(WHOLE)
    edge = Edge(
        "weight_supply",
        RegionEndpoint("memory", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(WHOLE.image)),),
    )
    doubly_owned = _framed(
        compute,
        extra_nodes=(NetworkNode("memory", supplier),),
        extra_edges=(edge,),
        weight_boundary=True,
    )

    with pytest.raises(InputServiceError, match="exactly one is required"):
        internally_presented_positions(doubly_owned, RegionInputRef("compute", "W"))


# -- operand identity stays region-local --------------------------------------


def test_one_operand_id_may_mean_two_different_tensors_in_one_network():
    """``node_a.W`` and ``node_b.W`` are qualified identities, not one tensor.

    Nothing in the dataflow model relates them, and nothing should: whether two
    region-local operands correspond to one source ONNX tensor is a question
    only the operation can answer, from its own declarations.
    """

    narrow = Operand("W", DataType["INT4"], (1,))
    wide = Operand("W", DataType["INT8"], (2, 2))
    first = DataflowRegion(
        SCHEDULE,
        (
            _activation_input(),
            UnportedInput(narrow, ScheduledInputRequirements({((0,), (0,)): 1})),
        ),
        (_result_output(),),
    )
    second = DataflowRegion(SCHEDULE, (_activation_input(), UnportedInput(wide, WHOLE_MATRIX)), ())
    network = DataflowNetwork(
        (NetworkNode("node_a", first), NetworkNode("node_b", second)),
        (),
        (
            _boundary("node_a", first.input_interface("x_in").port, "a_in"),
            _boundary("node_a", first.output_interface("y_out").port, "a_out"),
            _boundary("node_b", second.input_interface("x_in").port, "b_in"),
        ),
    )

    assert not validate_region(first)
    assert not validate_region(second)
    assert not validate_network(network)
    assert resolve_input(network, RegionInputRef("node_a", "W")).operand == narrow
    assert resolve_input(network, RegionInputRef("node_b", "W")).operand == wide
