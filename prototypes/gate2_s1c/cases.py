# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The forcing cases, over the real FINN Region constructors where they exist.

```text
external            W required by compute, one port, exposed at a boundary
embedded            W required by compute, no port
decoupled           W required by memory with no port; memory's edge feeds
                    compute's W port, which presents all of it
partial service     one Region whose port presents some of what it requires:
                    X re-read three times and presented once,
                    W half presented and half not
partial internal    compute's W port IS fed by an edge and still presents only
                    half the required positions -- the case a boolean
                    "edge-fed" answer gets wrong
plural target       two Regions requiring the same operand from one source
                    tensor
collision           two unrelated Regions both calling an operand W
```

The first three lift production Regions -- ``construct_dot_product_region``,
``construct_activation_replay_region``, ``construct_weight_stream_region`` --
and change only what is proposed.  The embedded case is the streamed Region with
``drop_ports=("weight",)``: one argument where today there is a separate
constructor.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataflow_model import (
    InputInterface,
    ProtoNetwork,
    ProtoNode,
    ProtoRegion,
    UnportedInput,
)
from finn.dataflow.kernels.memstream import construct_weight_stream_region
from finn.dataflow.network import (
    BoundaryContract,
    Edge,
    PositionMap,
    RegionEndpoint,
    SinkContract,
)
from finn.dataflow.ops.mvau.regions import (
    construct_activation_replay_region,
    construct_dot_product_region,
)
from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)
from qonnx.core.datatype import DataType

REPLAY = "replay"
COMPUTE = "compute"
MEMORY = "memory"

ACTIVATION = DataType["INT8"]
WEIGHT = DataType["INT8"]
OUTPUT = DataType["INT32"]


@dataclass(frozen=True)
class Folding:
    repetitions: int = 4
    matrix_width: int = 64
    matrix_height: int = 64
    pe: int = 8
    simd: int = 8


# -- lifting a production Region into the recommended shape -------------------


def lift(
    region: DataflowRegion,
    *,
    drop_ports: tuple[str, ...] = (),
    add_inputs: tuple[UnportedInput, ...] = (),
) -> ProtoRegion:
    """Rewrite a production Region's interfaces in the recommended shape.

    A ported input is ``InputInterface(port, requirements)`` -- the same class
    name, the same fields, the same construction order it has today, so this
    loop is the identity for every current FINN Region.  ``drop_ports`` turns
    one into an ``UnportedInput`` carrying the same operand and the same
    requirements, which is the entire difference between the streamed and
    embedded dot-product Regions.
    """

    inputs = tuple(
        UnportedInput(interface.port.operand, interface.requirements)
        if interface.port.id in drop_ports
        else InputInterface(interface.port, interface.requirements)
        for interface in region.inputs
    )
    return ProtoRegion(region.schedule, inputs + add_inputs, tuple(region.outputs))


def _streamed_compute(folding: Folding) -> DataflowRegion:
    return construct_dot_product_region(
        folding.repetitions,
        folding.matrix_width,
        folding.matrix_height,
        ACTIVATION,
        WEIGHT,
        OUTPUT,
        folding.pe,
        folding.simd,
    )


def _replay(folding: Folding) -> DataflowRegion:
    return construct_activation_replay_region(
        folding.repetitions,
        folding.matrix_width,
        folding.matrix_height,
        ACTIVATION,
        folding.pe,
        folding.simd,
    )


def _boundary(node_id: str, port: Port, boundary_id: str) -> BoundaryContract:
    return BoundaryContract(boundary_id, RegionEndpoint(node_id, port.id), port.beat_sequence)


def _activation_edge(replay: DataflowRegion) -> Edge:
    produced = replay.output_interface("activation_out").port
    return Edge(
        "activation_replay",
        RegionEndpoint(REPLAY, "activation_out"),
        (
            SinkContract(
                RegionEndpoint(COMPUTE, "activation"),
                PositionMap.identity(produced.beat_sequence.image),
            ),
        ),
    )


def _mvau_frame(
    folding: Folding, compute: ProtoRegion, *, weight_boundary: bool
) -> tuple[tuple[ProtoNode, ...], tuple[Edge, ...], tuple[BoundaryContract, ...]]:
    replay = _replay(folding)
    boundaries = [
        _boundary(REPLAY, replay.input_interface("activation_in").port, "activation"),
        _boundary(COMPUTE, compute.output_interface("output").port, "output"),
    ]
    if weight_boundary:
        boundaries.append(_boundary(COMPUTE, compute.input_interface("weight").port, "weight"))
    return (
        (ProtoNode(REPLAY, lift(replay)), ProtoNode(COMPUTE, compute)),
        (_activation_edge(replay),),
        tuple(boundaries),
    )


def external_network(folding: Folding = Folding()) -> ProtoNetwork:
    compute = lift(_streamed_compute(folding))
    nodes, edges, boundaries = _mvau_frame(folding, compute, weight_boundary=True)
    return ProtoNetwork(nodes, edges, boundaries)


def embedded_network(folding: Folding = Folding()) -> ProtoNetwork:
    """The streamed Region with the weight port dropped.  The requirement stays."""

    compute = lift(_streamed_compute(folding), drop_ports=("weight",))
    nodes, edges, boundaries = _mvau_frame(folding, compute, weight_boundary=False)
    return ProtoNetwork(nodes, edges, boundaries)


def memory_region(folding: Folding) -> ProtoRegion:
    """The parameter source, now saying which operand it requires.

    A rank-zero source has one schedule point and requires every position of the
    matrix there.  Today this Region declares no input at all, and so emits a
    matrix it never says it needs.
    """

    produced = construct_weight_stream_region(
        folding.repetitions,
        folding.matrix_width,
        folding.matrix_height,
        WEIGHT,
        folding.pe,
        folding.simd,
    )
    operand = produced.output_interface("weight").port.operand
    return lift(
        produced,
        add_inputs=(
            UnportedInput(
                operand,
                ScheduledInputRequirements({((), position): 1 for position in operand.positions}),
            ),
        ),
    )


def decoupled_network(folding: Folding = Folding()) -> ProtoNetwork:
    compute = lift(_streamed_compute(folding))
    memory = memory_region(folding)
    nodes, edges, boundaries = _mvau_frame(folding, compute, weight_boundary=False)
    weight_edge = Edge(
        "weight_supply",
        RegionEndpoint(MEMORY, "weight"),
        (
            SinkContract(
                RegionEndpoint(COMPUTE, "weight"),
                PositionMap.identity(memory.output_interface("weight").port.beat_sequence.image),
            ),
        ),
    )
    return ProtoNetwork((*nodes, ProtoNode(MEMORY, memory)), (*edges, weight_edge), boundaries)


# -- partial service, within one Region ---------------------------------------


def partial_service_region() -> ProtoRegion:
    """One Region whose ports present part of what it requires, two ways.

    ``X`` is read once per visit and its port presents each position once: the
    position set is fully presented and two thirds of the *occurrences* are not.
    This is the multi-visit kernel the deadlock proofs name -- softmax reads its
    row three times -- and the canon's instruction is to declare the re-reads as
    multiplicity, not as extra boundary beats.  The tensor has entered; serving
    the re-reads is the binding's business.

    ``W`` is required in full and its port presents only the upper columns: half
    the *positions* are not presented at all, so half of W has not entered.
    ``REGION.md`` §3.7 permits exactly this, per position, which is why
    "streamed" and "locally supplied" cannot be classifications of a whole
    operand -- and why the position sets below are position-granular rather than
    occurrence-granular.
    """

    schedule = LogicalSchedule(
        (ScheduleLevel("rep", 2), ScheduleLevel("visit", 3), ScheduleLevel("col", 4))
    )
    activation = Operand("X", ACTIVATION, (2, 4))
    weight = Operand("W", WEIGHT, (2, 4))
    result = Operand("Y", OUTPUT, (2,))

    read_every_visit = ScheduledInputRequirements(
        {
            ((rep, visit, col), (rep, col)): 1
            for rep in range(2)
            for visit in range(3)
            for col in range(4)
        }
    )
    activation_port = Port(
        "x_in",
        activation,
        BeatSequence(4, tuple(tuple((rep, col) for col in range(4)) for rep in range(2))),
    )
    weight_port = Port(
        "w_hi",
        weight,
        BeatSequence(2, tuple(tuple((rep, col) for col in (2, 3)) for rep in range(2))),
    )
    output = OutputInterface(
        Port("y_out", result, BeatSequence(1, (((0,),), ((1,),)))),
        ScheduledOutputAvailability({(0,): (0, 2, 3), (1,): (1, 2, 3)}),
    )
    return ProtoRegion(
        schedule,
        (
            InputInterface(activation_port, read_every_visit),
            InputInterface(weight_port, read_every_visit),
        ),
        (output,),
    )


# -- partial service across an edge -------------------------------------------


def partial_internal_network() -> ProtoNetwork:
    """A port that an edge feeds, presenting only half of what is required.

    The case a boolean "is this input edge-fed" answer gets wrong.  ``compute``
    requires all four positions of ``W``; its ``w_hi`` port is the sink of
    ``weight_supply``; and that port presents two of the four.  The Network
    supplies half.  The other half is owed by something else, and the source
    tensor corresponds to both the memory Region's requirement and the residue
    at compute.
    """

    weight = Operand("W", WEIGHT, (2, 2))
    activation = Operand("X", ACTIVATION, (2,))
    result = Operand("Y", OUTPUT, (2,))
    upper = BeatSequence(1, (((0, 1),), ((1, 1),)))

    compute = ProtoRegion(
        LogicalSchedule((ScheduleLevel("step", 2),)),
        (
            InputInterface(
                Port("x_in", activation, BeatSequence(1, (((0,),), ((1,),)))),
                ScheduledInputRequirements({((step,), (step,)): 1 for step in range(2)}),
            ),
            InputInterface(
                Port("w_hi", weight, upper),
                ScheduledInputRequirements(
                    {
                        ((step,), (row, col)): 1
                        for step in range(2)
                        for row in range(2)
                        for col in range(2)
                    }
                ),
            ),
        ),
        (
            OutputInterface(
                Port("y_out", result, BeatSequence(1, (((0,),), ((1,),)))),
                ScheduledOutputAvailability({(0,): (0,), (1,): (1,)}),
            ),
        ),
    )
    memory = ProtoRegion(
        LogicalSchedule(()),
        (
            UnportedInput(
                weight,
                ScheduledInputRequirements({((), position): 1 for position in upper.image}),
            ),
        ),
        (
            OutputInterface(
                Port("w_out", weight, upper),
                ScheduledOutputAvailability({position: () for position in upper.image}),
            ),
        ),
    )
    return ProtoNetwork(
        (ProtoNode(COMPUTE, compute), ProtoNode(MEMORY, memory)),
        (
            Edge(
                "weight_supply",
                RegionEndpoint(MEMORY, "w_out"),
                (SinkContract(RegionEndpoint(COMPUTE, "w_hi"), PositionMap.identity(upper.image)),),
            ),
        ),
        (
            _boundary(COMPUTE, compute.input_interface("x_in").port, "activation"),
            _boundary(COMPUTE, compute.output_interface("y_out").port, "output"),
        ),
    )


# -- plural targets and operand-id collision ----------------------------------


def _twin(suffix: str, weight: Operand) -> ProtoRegion:
    """A tiny Region requiring its own activation and a shared matrix."""

    schedule = LogicalSchedule((ScheduleLevel("step", 2),))
    activation = Operand(f"X{suffix}", ACTIVATION, (2,))
    result = Operand(f"Y{suffix}", OUTPUT, (2,))
    uses = ScheduledInputRequirements({((step,), (step,)): 1 for step in range(2)})
    weight_uses = ScheduledInputRequirements(
        {((step,), position): 1 for step in range(2) for position in weight.positions}
    )
    return ProtoRegion(
        schedule,
        (
            InputInterface(
                Port(f"x{suffix}", activation, BeatSequence(1, (((0,),), ((1,),)))), uses
            ),
            UnportedInput(weight, weight_uses),
        ),
        (
            OutputInterface(
                Port(f"y{suffix}", result, BeatSequence(1, (((0,),), ((1,),)))),
                ScheduledOutputAvailability({(0,): (0,), (1,): (1,)}),
            ),
        ),
    )


def _twin_network(left: Operand, right: Operand) -> ProtoNetwork:
    nodes = (
        ProtoNode("compute_a", _twin("a", left)),
        ProtoNode("compute_b", _twin("b", right)),
    )
    boundaries = tuple(
        boundary
        for node in nodes
        for boundary in (
            _boundary(node.id, node.region.input_ports[0], f"{node.id}_in"),
            _boundary(node.id, node.region.outputs[0].port, f"{node.id}_out"),
        )
    )
    return ProtoNetwork(nodes, (), boundaries)


def plural_target_network() -> ProtoNetwork:
    """Two Regions requiring the same matrix, neither exposing it.

    One source tensor, two dataflow requirements.  Under a rule that insists on
    exactly one mapping this Network is an error.  It is not one.
    """

    shared = Operand("W", WEIGHT, (2,))
    return _twin_network(shared, shared)


def colliding_network() -> ProtoNetwork:
    """Two unrelated tensors that both happen to be called ``W``.

    The one the Network can catch: they disagree on shape.  Two unrelated
    tensors that agree on type and shape are an authoring collision no
    structural rule can see, and the declaration-side qualification is the
    answer to those.
    """

    return _twin_network(Operand("W", WEIGHT, (2,)), Operand("W", WEIGHT, (4,)))


__all__ = [
    "ACTIVATION",
    "COMPUTE",
    "Folding",
    "MEMORY",
    "OUTPUT",
    "REPLAY",
    "WEIGHT",
    "colliding_network",
    "decoupled_network",
    "embedded_network",
    "external_network",
    "lift",
    "memory_region",
    "partial_internal_network",
    "partial_service_region",
    "plural_target_network",
]
