# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The four forcing cases, over the real FINN Region constructors.

Nothing here invents a Region.  Each case calls the production constructor the
selected Design would call and then adds the one thing schema A proposes: the
local-state input that says the node consumes the matrix.  That is deliberate --
if the cases were hand-built the prototype would prove only that the prototype
is consistent with itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from finn.dataflow.kernels.dotp_axi import construct_embedded_dot_product_region
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
from finn.dataflow.region import DataflowRegion, ScheduledInputRequirements
from qonnx.core.datatype import DataType

from schema_a import ProtoNetwork, ProtoNode, ProtoRegion, LocalStateInput

REPLAY = "replay"
COMPUTE = "compute"
MEMORY = "memory"


@dataclass(frozen=True)
class Folding:
    repetitions: int = 4
    matrix_width: int = 64
    matrix_height: int = 64
    pe: int = 8
    simd: int = 8


ACTIVATION = DataType["INT8"]
WEIGHT = DataType["INT8"]
OUTPUT = DataType["INT32"]


def _lift(region: DataflowRegion, held: tuple[LocalStateInput, ...] = ()) -> ProtoRegion:
    return ProtoRegion(region.schedule, region.inputs, region.outputs, held)


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


def weight_local_state(folding: Folding) -> LocalStateInput:
    """The matrix as a local-state input, taken from the streamed Region itself.

    The operand and the requirement function are lifted from the streamed
    weight interface rather than rebuilt, so "the embedded core reads exactly
    what the streamed one reads" is a fact about the value here and not a claim
    in a docstring.
    """

    interface = _streamed_compute(folding).input_interface("weight")
    return LocalStateInput(interface.port.operand, interface.requirements)


def _boundary(
    node_id: str, region: DataflowRegion, port_id: str, *, output: bool
) -> BoundaryContract:
    interface = region.output_interface(port_id) if output else region.input_interface(port_id)
    return BoundaryContract(
        port_id if port_id != "activation_in" else "activation",
        RegionEndpoint(node_id, port_id),
        interface.port.beat_sequence,
    )


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


def external_network(folding: Folding = Folding()) -> ProtoNetwork:
    """The matrix crosses the Design's edge.  No local state anywhere."""

    replay = _replay(folding)
    compute = _streamed_compute(folding)
    return ProtoNetwork(
        (ProtoNode(REPLAY, _lift(replay)), ProtoNode(COMPUTE, _lift(compute))),
        (_activation_edge(replay),),
        (
            _boundary(REPLAY, replay, "activation_in", output=False),
            _boundary(COMPUTE, compute, "weight", output=False),
            _boundary(COMPUTE, compute, "output", output=True),
        ),
    )


def embedded_network(folding: Folding = Folding()) -> ProtoNetwork:
    """The compute node holds the matrix.  No weight port, no weight boundary."""

    replay = _replay(folding)
    compute = construct_embedded_dot_product_region(
        folding.repetitions,
        folding.matrix_width,
        folding.matrix_height,
        ACTIVATION,
        WEIGHT,
        OUTPUT,
        folding.pe,
        folding.simd,
    )
    return ProtoNetwork(
        (
            ProtoNode(REPLAY, _lift(replay)),
            ProtoNode(COMPUTE, _lift(compute, (weight_local_state(folding),))),
        ),
        (_activation_edge(replay),),
        (
            _boundary(REPLAY, replay, "activation_in", output=False),
            _boundary(COMPUTE, compute, "output", output=True),
        ),
    )


def decoupled_network(folding: Folding = Folding()) -> ProtoNetwork:
    """A second node holds the matrix and streams it into the compute node.

    The compute Region is the *streamed* one, unchanged: external and decoupled
    differ only in what feeds the weight input.  The memory Region gains the
    local-state input the production ``construct_cyclic_parameter_region`` cannot
    currently express -- today it emits a matrix it never says it has.
    """

    replay = _replay(folding)
    compute = _streamed_compute(folding)
    memory = construct_weight_stream_region(
        folding.repetitions,
        folding.matrix_width,
        folding.matrix_height,
        WEIGHT,
        folding.pe,
        folding.simd,
    )
    produced = memory.output_interface("weight").port
    consumed = compute.input_interface("weight").port
    weight_edge = Edge(
        "weight_supply",
        RegionEndpoint(MEMORY, "weight"),
        (
            SinkContract(
                RegionEndpoint(COMPUTE, "weight"),
                PositionMap.identity(produced.beat_sequence.image),
            ),
        ),
    )
    assert produced.beat_sequence.image == consumed.beat_sequence.image
    # A rank-zero source has one schedule point, so every position it holds is
    # required there.  This is the memory's own requirement function, not the
    # consumer's.
    held = LocalStateInput(
        produced.operand,
        ScheduledInputRequirements(
            {((), position): 1 for position in produced.beat_sequence.image}
        ),
    )
    return ProtoNetwork(
        (
            ProtoNode(REPLAY, _lift(replay)),
            ProtoNode(COMPUTE, _lift(compute)),
            ProtoNode(MEMORY, _lift(memory, (held,))),
        ),
        (_activation_edge(replay), weight_edge),
        (
            _boundary(REPLAY, replay, "activation_in", output=False),
            _boundary(COMPUTE, compute, "output", output=True),
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
    "weight_local_state",
]
