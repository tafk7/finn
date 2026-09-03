# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A local-state source that presents a parameter matrix as an ordered stream.

Decoupled supply is not "the same weights, arranged differently".  It is a
second Region in the Network -- a rank-zero source whose whole content is
available at its one schedule point -- feeding a real edge into the compute
Region's weight input.  Naming it that way is what makes the three supply modes
comparable: external streaming has a boundary where this has a node, and
embedded has neither.

The Region is exact about the order it produces, because the edge it feeds is
compared position by position against what the consumer requires.  Nothing here
knows how the matrix reaches the memory: that is physical, and this Kernel's
physical projection says so.
"""

from __future__ import annotations

from collections.abc import Mapping

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.kernels.kernel import (
    Kernel,
    PhysicallyUnsupported,
    Region,
    RegionRefused,
)
from finn.dataflow.model.declarations import Input
from finn.dataflow.model.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.parameters.cyclic.computation import CYCLIC_PARAMETER_DELIVERY
from finn.dataflow.parameters.cyclic.region import construct_cyclic_parameter_region
from finn.dataflow.region import (
    BeatSequence,
    Coordinate,
    DataflowRegion,
    NumericElementType,
    Operand,
    Port,
)


def _weight_beats(
    repetitions: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
) -> tuple[tuple[Coordinate, ...], ...]:
    """The exact order the folded dot product consumes its matrix in.

    Restated here rather than imported from the consumer: a supplier that
    derived its output order from a particular consumer's module would be a
    supplier bound to that consumer.  The two agreeing is a property the Network
    checks, position by position, and it should be checkable rather than
    guaranteed by a shared import.
    """

    return tuple(
        tuple(
            (neuron_fold * pe + pe_index, synapse_fold * simd + lane)
            for pe_index in range(pe)
            for lane in range(simd)
        )
        for _repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def construct_weight_stream_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    weight_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """One rank-zero source emitting the matrix in folded consumption order."""

    dimensions = (repetitions, matrix_width, matrix_height, pe, simd)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise RegionRefused("weight-stream dimensions and folding must be positive integers")
    if matrix_width % simd:
        raise RegionRefused("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise RegionRefused("PE must divide matrix_height exactly")

    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    weight = Operand("W", weight_type, (matrix_height, matrix_width))
    return construct_cyclic_parameter_region(
        Port(
            "weight",
            weight,
            BeatSequence(
                pe * simd,
                _weight_beats(repetitions, neuron_folds, synapse_folds, pe, simd),
            ),
        )
    )


class MemstreamKernel(Kernel):
    """The decoupled weight supplier as one Region and, later, one module.

    It owns no Decision.  Depth, width and repetition are the Design's folding
    restated in the memory's vocabulary, exactly as the replay buffer's are, and
    a supplier that picked its own depth would be picking a fold.

    Its physical projection is explicitly unavailable at this phase.  A real
    memstream needs a data slot, an initializer written into it, a RAM style and
    the artifact stages that carry them; U3 owns the semantics and U6 owns all
    of that.  The Network placing this Kernel resolves regardless, which is the
    separation the phase exists to demonstrate.
    """

    id = "finn_rtl_memstream"
    version = "1"
    computation = CYCLIC_PARAMETER_DELIVERY

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    weight_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    pe = Input(int)
    simd = Input(int)

    region = Region(
        family="parameter.cyclic_delivery",
        version="1",
        construct=construct_weight_stream_region,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        weight_type=weight_type,
        pe=pe,
        simd=simd,
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        raise PhysicallyUnsupported(
            "memstream realization arrives in U6 with its data slot, loader and RAM style"
        )


__all__ = [
    "CYCLIC_PARAMETER_DELIVERY",
    "MemstreamKernel",
    "construct_weight_stream_region",
]
