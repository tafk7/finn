# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The forcing cases, over the real FINN Region constructors where they exist.

```text
external              W required by compute, one port, exposed at a boundary
embedded              W required by compute, no port
decoupled             W required by memory, no port; memory's output edge
                      transports W to compute's W requirement
partial service       one Region whose ports present some of what it requires:
                      X re-read three times and presented once,
                      W half presented and half not
plural mapping        two Regions requiring the same operand from one source
                      tensor
multi-port operand    one operand presented by two ports -- the named risk, not
                      a forcing case; the recommendation refuses it
```

The first three lift production Regions -- ``construct_dot_product_region``,
``construct_activation_replay_region``, ``construct_weight_stream_region`` --
and change only what the recommendation proposes.  Note what the embedded case
becomes: the streamed Region with the weight *port* set to ``None`` and the
weight *requirement* kept, which is one argument rather than a separate
constructor.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataflow_model import ProtoNetwork, ProtoNode, ProtoRegion, RegionInput
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
from candidates import InputRequirement, SplitRegion
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
    add_inputs: tuple[RegionInput, ...] = (),
) -> ProtoRegion:
    """Rewrite a production Region's interfaces in the recommended shape.

    ``(Port, requirements)`` becomes ``(operand, requirements, port)``.  The
    operand comes off the port, so the rewrite is lossless and mechanical --
    which is the migration this recommendation asks for, performed here on the
    real values before asking for it.

    ``drop_ports`` sets the port to ``None`` and keeps the requirement.  That
    one argument is the entire difference between the streamed and embedded
    dot-product Regions.
    """

    inputs = tuple(
        RegionInput(
            interface.port.operand,
            interface.requirements,
            None if interface.port.id in drop_ports else interface.port,
        )
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
        boundaries.append(_boundary(COMPUTE, compute.input_port("weight"), "weight"))
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
            RegionInput(
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


# -- partial service ----------------------------------------------------------


def partial_service_region() -> ProtoRegion:
    """One Region whose ports present part of what it requires, two ways.

    ``X`` is read once per visit and the port presents each position once: the
    position set is fully presented and two thirds of the *occurrences* are not.
    This is the multi-visit kernel the deadlock proofs name -- softmax reads its
    row three times -- and the canon's instruction is to declare the re-reads as
    multiplicity, not as extra boundary beats.

    ``W`` is required in full and its port presents only the upper columns: half
    the *positions* are not presented at all.  ``REGION.md`` §3.7 permits exactly
    this, per position, which is why "streamed" and "locally supplied" cannot be
    classifications of a whole operand.
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
            RegionInput(activation, read_every_visit, activation_port),
            RegionInput(weight, read_every_visit, weight_port),
        ),
        (output,),
    )


# -- plural source mapping ----------------------------------------------------


def _twin(suffix: str) -> ProtoRegion:
    """A tiny Region requiring its own activation and a shared matrix ``W``."""

    schedule = LogicalSchedule((ScheduleLevel("step", 2),))
    activation = Operand(f"X{suffix}", ACTIVATION, (2,))
    weight = Operand("W", WEIGHT, (2,))
    result = Operand(f"Y{suffix}", OUTPUT, (2,))
    uses = ScheduledInputRequirements({((step,), (step,)): 1 for step in range(2)})
    return ProtoRegion(
        schedule,
        (
            RegionInput(
                activation,
                uses,
                Port(f"x{suffix}", activation, BeatSequence(1, (((0,),), ((1,),)))),
            ),
            RegionInput(weight, uses),
        ),
        (
            OutputInterface(
                Port(f"y{suffix}", result, BeatSequence(1, (((0,),), ((1,),)))),
                ScheduledOutputAvailability({(0,): (0,), (1,): (1,)}),
            ),
        ),
    )


def plural_mapping_network() -> ProtoNetwork:
    """Two Regions requiring the same matrix, neither exposing it.

    One source tensor, two dataflow requirements.  Under a rule that insists on
    exactly one mapping this Network is an error.  It is not one.
    """

    nodes = tuple(ProtoNode(f"compute_{suffix}", _twin(suffix)) for suffix in ("a", "b"))
    boundaries = tuple(
        boundary
        for node in nodes
        for boundary in (
            _boundary(node.id, node.region.input_ports[0], f"{node.id}_in"),
            _boundary(node.id, node.region.outputs[0].port, f"{node.id}_out"),
        )
    )
    return ProtoNetwork(nodes, (), boundaries)


# -- one operand, two ports ---------------------------------------------------


def split_supply_region() -> SplitRegion:
    """``W`` delivered by two ports -- the one shape the recommendation refuses.

    Built in the widening shape so the refusal can be triggered on demand rather
    than reasoned about.  Nothing in FINN or in the canon requires it today; it
    is here as the named risk, not as a forcing case.
    """

    schedule = LogicalSchedule((ScheduleLevel("step", 2),))
    weight = Operand("W", WEIGHT, (2, 2))
    result = Operand("Y", OUTPUT, (2,))
    uses = ScheduledInputRequirements(
        {((step,), (row, col)): 1 for step in range(2) for row in range(2) for col in range(2)}
    )
    return SplitRegion(
        schedule,
        (InputRequirement(weight, uses),),
        (
            Port("w_lo", weight, BeatSequence(2, (((0, 0), (0, 1)),))),
            Port("w_hi", weight, BeatSequence(2, (((1, 0), (1, 1)),))),
        ),
        (
            OutputInterface(
                Port("y", result, BeatSequence(1, (((0,),), ((1,),)))),
                ScheduledOutputAvailability({(0,): (0,), (1,): (1,)}),
            ),
        ),
    )


__all__ = [
    "ACTIVATION",
    "COMPUTE",
    "Folding",
    "MEMORY",
    "OUTPUT",
    "REPLAY",
    "WEIGHT",
    "decoupled_network",
    "embedded_network",
    "external_network",
    "lift",
    "memory_region",
    "partial_service_region",
    "plural_mapping_network",
    "split_supply_region",
]
