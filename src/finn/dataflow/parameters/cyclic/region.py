# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized region declarations for cyclic parameter delivery."""

from __future__ import annotations

from finn.dataflow.model.region import (
    DataflowRegion,
    InternalInput,
    LogicalSchedule,
    OutputInterface,
    Port,
    RequirementKey,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)
from finn.dataflow.model.maps import MapCapabilityError, OccurrenceAxis


def construct_cyclic_parameter_region(output_port: Port) -> DataflowRegion:
    """Construct a rank-zero local-state source for an exact output sequence.

    All parameter positions are logically available at the sole schedule point.
    Beat ordinal remains the independent boundary-order domain, including when
    the selected sequence repeats positions.

    The region *requires* what it emits, and now says so.  A source declaring an
    output and no input was claiming to produce a matrix out of nothing: the
    positions have to come from somewhere, and the point of the decoupled form
    is that a Network can see the demand.  So there is one ``InternalInput`` over
    the output operand, requiring each position in the output image once at the
    rank-zero point ``()``.

    Once, not once per beat.  Repeated output presentation does not by itself
    create repeated input requirement occurrences -- a binding may retain one
    required position and emit it several times, exactly as one presented input
    position may satisfy several scheduled uses.  A supplier whose computation
    genuinely re-reads can author a different requirement relation.

    "Internal" here is the absence of an input port and nothing more.  It names
    no memory, no initializer and no data slot; how the positions are serviced
    is the binding's, and U6 owns it.
    """
    if not isinstance(output_port, Port):
        raise TypeError("output_port must be a Port")
    port = Port("weight", output_port.operand, output_port.beat_sequence)
    image = port.beat_sequence.image_set
    schedule = LogicalSchedule(())
    if image.is_full:
        requirements = ScheduledInputRequirements.affine(
            schedule.iteration_domain,
            port.operand.position_domain,
            base=(0,) * port.operand.rank,
            iteration_coefficients=((),) * port.operand.rank,
            occurrences=tuple(
                OccurrenceAxis(axis, extent, 1) for axis, extent in enumerate(port.operand.shape)
            ),
        )
        availability = ScheduledOutputAvailability.affine(
            port.operand.position_domain,
            schedule.iteration_domain,
            view_extents=port.operand.shape,
            offset=0,
            coefficients=(0,) * port.operand.rank,
        )
    else:
        if not port.beat_sequence.is_explicit:
            raise MapCapabilityError(
                "cyclic delivery of a compact partial image requires a supported "
                "partial requirement rule or an explicit caller-bounded conversion"
            )
        requirements_table: dict[RequirementKey, int] = {
            ((), position): 1 for position in image.iter_coordinates()
        }
        availability_table = {position: () for position in image.iter_coordinates()}
        requirements = ScheduledInputRequirements(requirements_table)
        availability = ScheduledOutputAvailability(availability_table)
    return DataflowRegion(
        schedule,
        (InternalInput(port.operand, requirements),),
        (OutputInterface(port, availability),),
    )


__all__ = ["construct_cyclic_parameter_region"]
