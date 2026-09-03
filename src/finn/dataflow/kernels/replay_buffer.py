# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative one-Region model of FinnLib's ``replay_buffer``.

The buffer presents each activation row once per neuron fold.  That expansion is
what the monolithic MVAU Region performed implicitly by scheduling its activation
input across ``nf``; naming it as its own Region makes it a composable unit and
leaves the dot-product half with nothing but arithmetic.

It owns no decision at all.  ``LEN``, ``REP``, and ``W`` are the folding restated
in the buffer's own vocabulary, derived from facts its Design supplies -- a
buffer that picked its own depth would be picking a fold.  It is kept even at one
neuron fold, where it is an identity: eliding the physical buffer is a choice for
this Kernel's own realization to make, not a reason for the Region to disappear.

The core predates FINN's AXI naming and takes ``clk`` with an active-high
``rst``, so the ABI says so rather than smoothing it over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from finn.dataflow.artifacts.abi import (
    Bus,
    Clock,
    ComponentABI,
    Direction,
    Endpoint,
    Free,
    Member,
    Reset,
    Signal,
    StandardProtocol,
)
from finn.dataflow.artifacts.contributions import CopiedSource
from finn.dataflow.computation import ACTIVATION_REPLAY_COMPUTATION
from finn.dataflow.model.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.dataflow.model.declarations import Input, derived
from finn.dataflow.kernels.kernel import Kernel, Parameter, Region, RegionRefused
from finn.dataflow.region import (
    BeatSequence,
    Coordinate,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    element_width,
)

FINNLIB_ROOT = "finnlib"
FINNLIB_SOURCES = ("rtl/infra/replay_buffer.sv",)


def _compact_beats(
    repetitions: int, synapse_folds: int, simd: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
    )


def _expanded_beats(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for _neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
    )


def construct_activation_replay_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Expand a compact activation sequence to one presentation per neuron fold.

    ``R x SF`` beats in, ``R x NF x SF`` beats out: same operand, same position
    image, same elements per beat.
    """

    dimensions = (repetitions, matrix_width, matrix_height, pe, simd)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise RegionRefused("replay dimensions and folding must be positive integers")
    if matrix_width % simd:
        raise RegionRefused("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise RegionRefused("PE must divide matrix_height exactly")

    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    schedule = LogicalSchedule(
        (
            ScheduleLevel("rep", repetitions),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
        )
    )
    activation = Operand("X", activation_type, (repetitions, matrix_width))
    requirements: dict[RequirementKey, int] = {
        (
            (repetition, neuron_fold, synapse_fold),
            (repetition, synapse_fold * simd + lane),
        ): 1
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for lane in range(simd)
    }
    # A position occurs NF times in the expanded sequence; availability is keyed
    # by position, so it records the first occurrence -- at nf = 0.
    availability: dict[Coordinate, Coordinate] = {
        (repetition, synapse_fold * simd + lane): (repetition, 0, synapse_fold)
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
        for lane in range(simd)
    }
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation_in",
                    activation,
                    BeatSequence(simd, _compact_beats(repetitions, synapse_folds, simd)),
                ),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port(
                    "activation_out",
                    activation,
                    BeatSequence(
                        simd, _expanded_beats(repetitions, neuron_folds, synapse_folds, simd)
                    ),
                ),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


class ReplayBufferKernel(Kernel):
    """Present each activation row once per neuron fold."""

    id = "replay_buffer"
    version = "1"
    computation = ACTIVATION_REPLAY_COMPUTATION

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)
    pe = Input(int)
    simd = Input(int)

    region = Region(
        family="mvau.activation_replay",
        version="1",
        construct=construct_activation_replay_region,
        repetitions=repetitions,
        matrix_width=matrix_width,
        matrix_height=matrix_height,
        activation_type=activation_type,
        pe=pe,
        simd=simd,
    )

    @derived(int, matrix_width=matrix_width, simd=simd)
    def sequence_length(*, matrix_width: int, simd: int) -> int:
        return matrix_width // simd

    @derived(int, matrix_height=matrix_height, pe=pe)
    def replay_count(*, matrix_height: int, pe: int) -> int:
        return matrix_height // pe

    @derived(int, activation_type=activation_type, simd=simd)
    def data_width(*, activation_type: NumericElementType, simd: int) -> int:
        return simd * element_width(activation_type)

    LEN = Parameter(sequence_length)
    REP = Parameter(replay_count)
    W = Parameter(data_width)

    sources = (
        CopiedSource(
            FINNLIB_ROOT,
            FINNLIB_SOURCES[0],
            provides=("module:replay_buffer",),
        ),
    )

    @classmethod
    def component_abi(cls, parameters: Mapping[str, bool | int | float | str]) -> ComponentABI:
        width = cast(int, parameters["W"])
        return ComponentABI(
            "replay_buffer",
            (
                Signal("clk", Direction.IN, 1, Clock(Free())),
                Signal("rst", Direction.IN, 1, Reset(active_low=False)),
                Bus(
                    "in0",
                    StandardProtocol.AXIS,
                    (
                        Member("tdata", "idat", width),
                        Member("tvalid", "ivld"),
                        Member("tready", "irdy"),
                    ),
                    endpoint=Endpoint.TARGET,
                    associated_clock="clk",
                    associated_reset="rst",
                ),
                Bus(
                    "out0",
                    StandardProtocol.AXIS,
                    (
                        Member("tdata", "odat", width),
                        Member("tvalid", "ovld"),
                        Member("tready", "ordy"),
                        Member("tlast", "olast"),
                    ),
                    endpoint=Endpoint.INITIATOR,
                    associated_clock="clk",
                    associated_reset="rst",
                ),
                # `ofin` marks the end of the whole replayed run rather than of
                # one sequence, so it is not an AXI-Stream member of `out0`.
                Signal("ofin", Direction.OUT, 1),
            ),
            tuple((name, str(value)) for name, value in parameters.items()),
        )


__all__ = [
    "ACTIVATION_REPLAY_COMPUTATION",
    "FINNLIB_ROOT",
    "FINNLIB_SOURCES",
    "ReplayBufferKernel",
    "construct_activation_replay_region",
]
