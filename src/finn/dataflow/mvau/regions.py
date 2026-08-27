# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Complete normalized region declarations for the MVAU compute Kernel."""

from __future__ import annotations

from enum import Enum

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
)


class MVAURegionDeclaration(str, Enum):
    """Stable identities for the initial complete MVAU region declarations."""

    STANDARD_EMBEDDED = "standard.embedded"
    STANDARD_STREAMED = "standard.streamed"
    BATCH_INTERLEAVED_STREAMED = "batch_interleaved.streamed"


class MVAUWeightInterface(str, Enum):
    """Compatibility selector for the two standard MVAU constructors."""

    EMBEDDED = "embedded"
    STREAMED = "streamed"


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _complete_numeric_element_type(value: object) -> bool:
    return type(value) is NumericElementType and bool(value.type_id) and value.bit_width > 0


def _validate_common_arguments(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> None:
    dimensions = {
        "repetitions": repetitions,
        "matrix_width": matrix_width,
        "matrix_height": matrix_height,
        "pe": pe,
        "simd": simd,
    }
    for dimension_name, dimension_value in dimensions.items():
        if not _positive_integer(dimension_value):
            raise ValueError(f"{dimension_name} must be a positive integer")
    element_types = {
        "activation_element_type": activation_element_type,
        "weight_element_type": weight_element_type,
        "output_element_type": output_element_type,
    }
    for type_name, element_type in element_types.items():
        if not _complete_numeric_element_type(element_type):
            raise ValueError(f"{type_name} must be a complete numeric element type")
    if matrix_width % simd:
        raise ValueError("SIMD must divide matrix_width exactly")
    if matrix_height % pe:
        raise ValueError("PE must divide matrix_height exactly")


def _activation_operand(
    repetitions: int, matrix_width: int, element_type: NumericElementType
) -> Operand:
    return Operand("X", element_type, (repetitions, matrix_width))


def _weight_operand(
    matrix_height: int, matrix_width: int, element_type: NumericElementType
) -> Operand:
    return Operand("W", element_type, (matrix_height, matrix_width))


def _output_operand(
    repetitions: int, matrix_height: int, element_type: NumericElementType
) -> Operand:
    return Operand("Y", element_type, (repetitions, matrix_height))


def _compact_activation_beats(
    repetitions: int, synapse_folds: int, simd: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, synapse_fold * simd + lane) for lane in range(simd))
        for repetition in range(repetitions)
        for synapse_fold in range(synapse_folds)
    )


def _canonical_output_beats(
    repetitions: int, neuron_folds: int, pe: int
) -> tuple[tuple[Coordinate, ...], ...]:
    return tuple(
        tuple((repetition, neuron_fold * pe + pe_index) for pe_index in range(pe))
        for repetition in range(repetitions)
        for neuron_fold in range(neuron_folds)
    )


def _standard_activation_requirements(
    repetitions: int, neuron_folds: int, synapse_folds: int, simd: int
) -> ScheduledInputRequirements:
    requirements: dict[RequirementKey, int] = {}
    for repetition in range(repetitions):
        for neuron_fold in range(neuron_folds):
            for synapse_fold in range(synapse_folds):
                iteration = (repetition, neuron_fold, synapse_fold)
                for lane in range(simd):
                    requirements[(iteration, (repetition, synapse_fold * simd + lane))] = 1
    return ScheduledInputRequirements(requirements)


def _standard_weight_requirements(
    repetitions: int,
    neuron_folds: int,
    synapse_folds: int,
    pe: int,
    simd: int,
) -> ScheduledInputRequirements:
    requirements: dict[RequirementKey, int] = {}
    for repetition in range(repetitions):
        for neuron_fold in range(neuron_folds):
            for synapse_fold in range(synapse_folds):
                iteration = (repetition, neuron_fold, synapse_fold)
                for pe_index in range(pe):
                    for lane in range(simd):
                        position = (
                            neuron_fold * pe + pe_index,
                            synapse_fold * simd + lane,
                        )
                        requirements[(iteration, position)] = 1
    return ScheduledInputRequirements(requirements)


def _standard_output_availability(
    repetitions: int, neuron_folds: int, synapse_folds: int, pe: int
) -> ScheduledOutputAvailability:
    availability: dict[Coordinate, Coordinate] = {}
    for repetition in range(repetitions):
        for neuron_fold in range(neuron_folds):
            for pe_index in range(pe):
                availability[(repetition, neuron_fold * pe + pe_index)] = (
                    repetition,
                    neuron_fold,
                    synapse_folds - 1,
                )
    return ScheduledOutputAvailability(availability)


def _standard_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
    *,
    streamed_weights: bool,
) -> DataflowRegion:
    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
    )
    synapse_folds = matrix_width // simd
    neuron_folds = matrix_height // pe
    schedule = LogicalSchedule(
        (
            ScheduleLevel("rep", repetitions),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
        )
    )
    activation = _activation_operand(repetitions, matrix_width, activation_element_type)
    output = _output_operand(repetitions, matrix_height, output_element_type)
    inputs = [
        InputInterface(
            Port(
                "activation",
                activation,
                BeatSequence(simd, _compact_activation_beats(repetitions, synapse_folds, simd)),
            ),
            _standard_activation_requirements(repetitions, neuron_folds, synapse_folds, simd),
        )
    ]
    if streamed_weights:
        weight = _weight_operand(matrix_height, matrix_width, weight_element_type)
        weight_beats = tuple(
            tuple(
                (
                    neuron_fold * pe + pe_index,
                    synapse_fold * simd + lane,
                )
                for pe_index in range(pe)
                for lane in range(simd)
            )
            for _repetition in range(repetitions)
            for neuron_fold in range(neuron_folds)
            for synapse_fold in range(synapse_folds)
        )
        inputs.append(
            InputInterface(
                Port("weight", weight, BeatSequence(pe * simd, weight_beats)),
                _standard_weight_requirements(repetitions, neuron_folds, synapse_folds, pe, simd),
            )
        )
    output_interface = OutputInterface(
        Port(
            "output",
            output,
            BeatSequence(pe, _canonical_output_beats(repetitions, neuron_folds, pe)),
        ),
        _standard_output_availability(repetitions, neuron_folds, synapse_folds, pe),
    )
    return DataflowRegion(schedule, tuple(inputs), (output_interface,))


def construct_standard_embedded_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Construct the standard folded MVAU region with binding-local weights."""
    return _standard_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
        streamed_weights=False,
    )


def construct_standard_streamed_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Construct the standard folded MVAU region with full-tile streamed weights."""
    return _standard_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
        streamed_weights=True,
    )


def construct_batch_interleaved_streamed_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
    interleave: int,
) -> DataflowRegion:
    """Construct the batch-interleaved MVAU region with chunked streamed weights."""
    _validate_common_arguments(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
    )
    if not _positive_integer(interleave) or interleave <= 1:
        raise ValueError("interleave must be an integer greater than one")
    if repetitions % interleave:
        raise ValueError("interleave must divide repetitions exactly")
    if (pe * simd) % interleave:
        raise ValueError("interleave must divide PE * SIMD exactly")

    batches = repetitions // interleave
    neuron_folds = matrix_height // pe
    synapse_folds = matrix_width // simd
    weight_fields = pe * simd // interleave
    schedule = LogicalSchedule(
        (
            ScheduleLevel("batch", batches),
            ScheduleLevel("nf", neuron_folds),
            ScheduleLevel("sf", synapse_folds),
            ScheduleLevel("t", interleave),
        )
    )
    activation = _activation_operand(repetitions, matrix_width, activation_element_type)
    weight = _weight_operand(matrix_height, matrix_width, weight_element_type)
    output = _output_operand(repetitions, matrix_height, output_element_type)

    activation_requirements: dict[RequirementKey, int] = {}
    weight_requirements: dict[RequirementKey, int] = {}
    output_availability: dict[Coordinate, Coordinate] = {}
    for batch in range(batches):
        for neuron_fold in range(neuron_folds):
            for synapse_fold in range(synapse_folds):
                for reuse_index in range(interleave):
                    iteration = (batch, neuron_fold, synapse_fold, reuse_index)
                    repetition = batch * interleave + reuse_index
                    for lane in range(simd):
                        activation_requirements[
                            (iteration, (repetition, synapse_fold * simd + lane))
                        ] = 1
                    for pe_index in range(pe):
                        for lane in range(simd):
                            weight_requirements[
                                (
                                    iteration,
                                    (
                                        neuron_fold * pe + pe_index,
                                        synapse_fold * simd + lane,
                                    ),
                                )
                            ] = 1
            for reuse_index in range(interleave):
                repetition = batch * interleave + reuse_index
                for pe_index in range(pe):
                    output_availability[(repetition, neuron_fold * pe + pe_index)] = (
                        batch,
                        neuron_fold,
                        synapse_folds - 1,
                        reuse_index,
                    )

    weight_beats = tuple(
        tuple(
            (
                neuron_fold * pe + (chunk * weight_fields + field) // simd,
                synapse_fold * simd + (chunk * weight_fields + field) % simd,
            )
            for field in range(weight_fields)
        )
        for _batch in range(batches)
        for neuron_fold in range(neuron_folds)
        for synapse_fold in range(synapse_folds)
        for chunk in range(interleave)
    )
    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port(
                    "activation",
                    activation,
                    BeatSequence(
                        simd,
                        _compact_activation_beats(repetitions, synapse_folds, simd),
                    ),
                ),
                ScheduledInputRequirements(activation_requirements),
            ),
            InputInterface(
                Port("weight", weight, BeatSequence(weight_fields, weight_beats)),
                ScheduledInputRequirements(weight_requirements),
            ),
        ),
        (
            OutputInterface(
                Port(
                    "output",
                    output,
                    BeatSequence(
                        pe,
                        _canonical_output_beats(repetitions, neuron_folds, pe),
                    ),
                ),
                ScheduledOutputAvailability(output_availability),
            ),
        ),
    )


def construct_mvau_compute_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
    weight_interface: MVAUWeightInterface,
) -> DataflowRegion:
    """Compatibility facade for the two standard region declarations."""
    if type(weight_interface) is not MVAUWeightInterface:
        raise TypeError("weight_interface must be an MVAUWeightInterface")
    constructor = (
        construct_standard_embedded_mvau_region
        if weight_interface is MVAUWeightInterface.EMBEDDED
        else construct_standard_streamed_mvau_region
    )
    return constructor(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
    )


def construct_streamed_weight_mvau_region(
    repetitions: int,
    matrix_width: int,
    matrix_height: int,
    activation_element_type: NumericElementType,
    weight_element_type: NumericElementType,
    output_element_type: NumericElementType,
    pe: int,
    simd: int,
) -> DataflowRegion:
    """Backward-compatible name for the standard streamed declaration."""
    return construct_standard_streamed_mvau_region(
        repetitions,
        matrix_width,
        matrix_height,
        activation_element_type,
        weight_element_type,
        output_element_type,
        pe,
        simd,
    )


__all__ = [
    "MVAURegionDeclaration",
    "MVAUWeightInterface",
    "construct_batch_interleaved_streamed_mvau_region",
    "construct_mvau_compute_region",
    "construct_standard_embedded_mvau_region",
    "construct_standard_streamed_mvau_region",
    "construct_streamed_weight_mvau_region",
]
