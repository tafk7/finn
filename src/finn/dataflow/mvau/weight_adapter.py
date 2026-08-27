# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The first explicit MVAU weight-sequence adapter region."""

from __future__ import annotations

from finn.dataflow.region import (
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    OutputInterface,
    Port,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
)


def weight_sequence_adapter_applicable(source: Port, sink: Port) -> bool:
    """Return whether the evidenced adapter can preserve one weight pass."""
    return source.operand == sink.operand and (
        source.beat_sequence.image == sink.beat_sequence.image
    )


def construct_weight_sequence_adapter_region(source: Port, sink: Port) -> DataflowRegion:
    """Convert one fixed full-tile/chunked weight sequence into another.

    The schedule has one point per output beat.  Its requirements record every
    output field use at that point; the input boundary remains the independently
    selected producer sequence.  Buffering and physical packing are intentionally
    absent from this semantic declaration.
    """
    if not weight_sequence_adapter_applicable(source, sink):
        raise ValueError("weight adapter requires equal operands and beat-sequence images")
    requirements: dict[RequirementKey, int] = {}
    availability: dict[tuple[int, ...], tuple[int, ...]] = {}
    for ordinal, beat in enumerate(sink.beat_sequence.beats):
        iteration = (ordinal,)
        for position in beat:
            key = (iteration, position)
            requirements[key] = requirements.get(key, 0) + 1
            availability.setdefault(position, iteration)
    return DataflowRegion(
        LogicalSchedule((ScheduleLevel("output_beat", sink.beat_sequence.beat_count),)),
        (
            InputInterface(
                Port("weight_in", source.operand, source.beat_sequence),
                ScheduledInputRequirements(requirements),
            ),
        ),
        (
            OutputInterface(
                Port("weight_out", sink.operand, sink.beat_sequence),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


__all__ = [
    "construct_weight_sequence_adapter_region",
    "weight_sequence_adapter_applicable",
]
