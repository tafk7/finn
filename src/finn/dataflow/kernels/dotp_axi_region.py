# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Standard streamed dotp traffic, expressed without MVAU family constructors.

At (rep, nf, sf), consume SIMD activations and a PE-by-SIMD weight tile.
The activation row has already been replayed for each neuron fold. Produce PE
outputs at the last synapse fold. This is a dataflow/schedule description; the
multiply-accumulate arithmetic is implemented by the referenced dotp RTL.
"""

from finn.kernels.datatypes.values import QONNXDataType
from finn.dataflow.model.logical.maps import OccurrenceAxis, RectangularDomain
from finn.dataflow.model.logical.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    Operand,
    OutputInterface,
    Port,
    RegionRefused,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
    ScheduleLevel,
    is_element_type,
)


def construct_dotp_region(
    *,
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: QONNXDataType,
    weight_element_type: QONNXDataType,
    output_element_type: QONNXDataType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    for name, value in (
        ("repetitions", repetitions),
        ("matrix_width", matrix_width),
        ("matrix_height", matrix_height),
        ("pe", pe),
        ("simd", simd),
    ):
        if type(value) is not int or value <= 0:
            raise RegionRefused(f"{name} must be a positive integer")
    for name, datatype in (
        ("activation", activation_element_type),
        ("weight", weight_element_type),
        ("output", output_element_type),
    ):
        if not is_element_type(datatype):
            raise RegionRefused(f"{name} must have a complete numeric element type")
    if matrix_width % simd:
        raise RegionRefused("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise RegionRefused("PE must divide matrix_height exactly")

    nf, sf = matrix_height // pe, matrix_width // simd
    iterations = RectangularDomain((repetitions, nf, sf))
    schedule = LogicalSchedule(
        (ScheduleLevel("rep", repetitions), ScheduleLevel("nf", nf), ScheduleLevel("sf", sf))
    )
    activation = Operand("XR", activation_element_type, (repetitions * nf, matrix_width))
    weights = Operand("W", weight_element_type, (matrix_height, matrix_width))
    output = Operand("Y", output_element_type, (repetitions, matrix_height))

    # Activation lane s at (rep,nf,sf) is XR[rep*NF+nf, sf*SIMD+s].
    activation_port = Port(
        "activation",
        activation,
        BeatSequence.affine(
            activation.position_domain,
            elements_per_beat=simd,
            beat_count=iterations.cardinality,
            view_extents=(repetitions, nf, sf, simd),
            offset=0,
            coefficients=(nf * matrix_width, matrix_width, simd, 1),
        ),
    )
    activation_requirements = ScheduledInputRequirements.affine(
        iterations,
        activation.position_domain,
        base=(0, 0),
        iteration_coefficients=((nf, 1, 0), (0, 0, simd)),
        occurrences=(OccurrenceAxis(1, simd, 1),),
    )

    # Weight field p*SIMD+s is W[nf*PE+p, sf*SIMD+s]. Repeat W for each rep.
    weight_port = Port(
        "weight",
        weights,
        BeatSequence.affine(
            weights.position_domain,
            elements_per_beat=pe * simd,
            beat_count=iterations.cardinality,
            view_extents=(repetitions, nf, sf, pe, simd),
            offset=0,
            coefficients=(0, pe * matrix_width, simd, matrix_width, 1),
        ),
    )
    weight_requirements = ScheduledInputRequirements.affine(
        iterations,
        weights.position_domain,
        base=(0, 0),
        iteration_coefficients=((0, pe, 0), (0, 0, simd)),
        occurrences=(OccurrenceAxis(0, pe, 1), OccurrenceAxis(1, simd, 1)),
    )

    # Output lane p is Y[rep,nf*PE+p], available after sf == SF-1.
    output_port = Port(
        "output",
        output,
        BeatSequence.affine(
            output.position_domain,
            elements_per_beat=pe,
            beat_count=repetitions * nf,
            view_extents=(repetitions, nf, pe),
            offset=0,
            coefficients=(matrix_height, pe, 1),
        ),
    )
    availability = ScheduledOutputAvailability.affine(
        output.position_domain,
        iterations,
        view_extents=(repetitions, nf, pe),
        offset=sf - 1,
        coefficients=(nf * sf, sf, 0),
    )
    return DataflowRegion(
        schedule,
        (
            InputInterface(activation_port, activation_requirements),
            InputInterface(weight_port, weight_requirements),
        ),
        (OutputInterface(output_port, availability),),
    )


__all__ = ["construct_dotp_region"]
