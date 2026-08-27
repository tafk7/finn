# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized region declarations for cyclic parameter delivery."""

from __future__ import annotations

from finn.dataflow.region import (
    DataflowRegion,
    LogicalSchedule,
    OutputInterface,
    Port,
    ScheduledOutputAvailability,
)


def construct_cyclic_parameter_region(output_port: Port) -> DataflowRegion:
    """Construct a rank-zero local-state source for an exact output sequence.

    All parameter positions are logically available at the sole schedule point.
    Beat ordinal remains the independent boundary-order domain, including when
    the selected sequence repeats positions.
    """
    if not isinstance(output_port, Port):
        raise TypeError("output_port must be a Port")
    port = Port("weight", output_port.operand, output_port.beat_sequence)
    availability = ScheduledOutputAvailability(
        {position: () for position in port.beat_sequence.image}
    )
    return DataflowRegion(
        LogicalSchedule(()),
        (),
        (OutputInterface(port, availability),),
    )


__all__ = ["construct_cyclic_parameter_region"]
