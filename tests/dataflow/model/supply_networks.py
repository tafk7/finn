# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The three supply forms of one matrix, as small hand-built Networks.

Shared by ``test_refs`` and ``test_presentation`` because they ask two different
questions of the same three Networks: which value a qualified name denotes, and
which of its positions arrive over a port.  What separates external, internal
and decoupled supply is not a classification stored anywhere -- it is which
positions arrive over a port, and from where.
"""

from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from finn.dataflow.model.network import (
    BoundaryContract,
    DataflowNetwork,
    Edge,
    NetworkNode,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.model.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    InternalInput,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)

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


def activation_input():
    return InputInterface(Port("x_in", ACTIVATION, SEQUENTIAL), EVERY_STEP)


def result_output():
    return OutputInterface(
        Port("y_out", RESULT, SEQUENTIAL),
        ScheduledOutputAvailability({(0,): (0,), (1,): (1,)}),
    )


def compute(weight_input):
    return DataflowRegion(SCHEDULE, (activation_input(), weight_input), (result_output(),))


def supplier(sequence):
    """A rank-zero source that requires what it emits and exposes no input port."""

    return DataflowRegion(
        LogicalSchedule(()),
        (
            InternalInput(
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


def boundary(node_id, port, boundary_id):
    return BoundaryContract(boundary_id, RegionEndpoint(node_id, port.id), port.beat_sequence)


def framed(region, extra_nodes=(), extra_edges=(), weight_boundary=False):
    boundaries = [
        boundary("compute", region.input_interface("x_in").port, "activation"),
        boundary("compute", region.output_interface("y_out").port, "output"),
    ]
    if weight_boundary:
        boundaries.append(boundary("compute", region.input_interface("w_in").port, "weight"))
    return DataflowNetwork(
        (NetworkNode("compute", region), *extra_nodes),
        tuple(extra_edges),
        tuple(boundaries),
    )


def external_network():
    """The matrix crosses the Network's edge: one port, exposed at a boundary."""

    return framed(
        compute(InputInterface(Port("w_in", WEIGHT, WHOLE), WHOLE_MATRIX)), weight_boundary=True
    )


def internal_input_network():
    """The matrix is required and no port presents it."""

    return framed(compute(InternalInput(WEIGHT, WHOLE_MATRIX)))


def supplied_network(sequence):
    consumer = compute(InputInterface(Port("w_in", WEIGHT, sequence), WHOLE_MATRIX))
    edge = Edge(
        "weight_supply",
        RegionEndpoint("memory", "w_out"),
        (SinkContract(RegionEndpoint("compute", "w_in"), PositionMap.identity(sequence.image)),),
    )
    return framed(
        consumer,
        extra_nodes=(NetworkNode("memory", supplier(sequence)),),
        extra_edges=(edge,),
    )


def decoupled_network():
    """A second node presents the whole matrix over an edge."""

    return supplied_network(WHOLE)


def partly_supplied_network():
    """An edge feeds the port, and the port still presents only half.

    The case that makes a boolean "is this input fed by an edge" answer wrong.
    """

    return supplied_network(UPPER)
