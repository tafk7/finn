# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.dataflow.region import (
    BeatSequence,
    DataflowRegion,
    InputInterface,
    LogicalSchedule,
    NumericElementType,
    Operand,
    OutputInterface,
    Port,
    ScheduledInputRequirements,
    ScheduledOutputAvailability,
)
from finn.dataflow.region_validation import validate_region


def _mvau_region(
    repetitions,
    matrix_width,
    matrix_height,
    simd,
    pe,
    *,
    weight_field_order=None,
    output_field_order=None,
):
    if matrix_width % simd != 0 or matrix_height % pe != 0:
        raise ValueError("SIMD must divide MW and PE must divide MH")
    input_folds = matrix_width // simd
    output_folds = matrix_height // pe
    schedule = LogicalSchedule((("rep", repetitions), ("nf", output_folds), ("sf", input_folds)))
    element_type = NumericElementType("int", 8)
    activation = Operand("X", element_type, (repetitions, matrix_width))
    weight = Operand("W", element_type, (matrix_height, matrix_width))
    output = Operand("Y", element_type, (repetitions, matrix_height))

    activation_requirements = {}
    weight_requirements = {}
    availability = {}
    for rep in range(repetitions):
        for nf in range(output_folds):
            for sf in range(input_folds):
                iteration = (rep, nf, sf)
                for lane in range(simd):
                    activation_requirements[(iteration, (rep, sf * simd + lane))] = 1
                for pe_index in range(pe):
                    for lane in range(simd):
                        weight_requirements[(iteration, (nf * pe + pe_index, sf * simd + lane))] = 1
            for pe_index in range(pe):
                availability[(rep, nf * pe + pe_index)] = (rep, nf, input_folds - 1)

    activation_beats = tuple(
        tuple((rep, sf * simd + lane) for lane in range(simd))
        for rep in range(repetitions)
        for sf in range(input_folds)
    )
    if weight_field_order is None:
        weight_field_order = tuple(
            (pe_index, lane) for pe_index in range(pe) for lane in range(simd)
        )
    else:
        weight_field_order = tuple(weight_field_order)
    weight_beats = tuple(
        tuple((nf * pe + pe_index, sf * simd + lane) for pe_index, lane in weight_field_order)
        for rep in range(repetitions)
        for nf in range(output_folds)
        for sf in range(input_folds)
    )
    if output_field_order is None:
        output_field_order = tuple(range(pe))
    else:
        output_field_order = tuple(output_field_order)
    output_beats = tuple(
        tuple((rep, nf * pe + pe_index) for pe_index in output_field_order)
        for rep in range(repetitions)
        for nf in range(output_folds)
    )

    return DataflowRegion(
        schedule,
        (
            InputInterface(
                Port("activation", activation, BeatSequence(simd, activation_beats)),
                ScheduledInputRequirements(activation_requirements),
            ),
            InputInterface(
                Port("weight", weight, BeatSequence(pe * simd, weight_beats)),
                ScheduledInputRequirements(weight_requirements),
            ),
        ),
        (
            OutputInterface(
                Port("output", output, BeatSequence(pe, output_beats)),
                ScheduledOutputAvailability(availability),
            ),
        ),
    )


def test_small_streamed_weight_mvau_region_matches_authoring_semantics():
    repetitions, matrix_width, matrix_height, simd, pe = 2, 4, 4, 2, 2
    input_folds = matrix_width // simd
    output_folds = matrix_height // pe
    region = _mvau_region(repetitions, matrix_width, matrix_height, simd, pe)
    activation = region.input_interface("activation")
    weight = region.input_interface("weight")
    output = region.output_interface("output")

    assert validate_region(region) == ()
    assert region.schedule.level_names == ("rep", "nf", "sf")
    assert region.schedule.rank((1, 0, 1)) == 5
    assert activation.requirements.occurrence_count == (
        repetitions * output_folds * input_folds * simd
    )
    assert activation.port.beat_sequence.delivered_field_count == (repetitions * input_folds * simd)
    assert (
        activation.requirements.occurrence_count
        == output_folds * activation.port.beat_sequence.delivered_field_count
    )
    assert weight.requirements.occurrence_count == (
        repetitions * output_folds * input_folds * pe * simd
    )

    for iteration in region.schedule.iter_points():
        ordinal = region.schedule.rank(iteration)
        rep, nf, sf = iteration
        expected_weight_beat = tuple(
            (nf * pe + pe_index, sf * simd + lane) for pe_index in range(pe) for lane in range(simd)
        )
        assert weight.port.beat_sequence.beat(ordinal) == expected_weight_beat
        assert all(
            weight.requirements.required(iteration, position) == 1
            for position in expected_weight_beat
        )
        for pe_index in range(pe):
            position = (rep, nf * pe + pe_index)
            if sf == input_folds - 1:
                assert output.availability.available_at(position) == iteration
            else:
                assert output.availability.available_at(position) != iteration

    assert output.availability.domain == output.port.beat_sequence.image


def test_mvau_field_bijections_change_sequences_without_changing_widths_or_shapes():
    original = _mvau_region(2, 4, 4, 2, 2)
    reordered = _mvau_region(
        2,
        4,
        4,
        2,
        2,
        weight_field_order=((1, 1), (1, 0), (0, 1), (0, 0)),
        output_field_order=(1, 0),
    )
    original_weight = original.input_interface("weight").port
    reordered_weight = reordered.input_interface("weight").port
    original_output = original.output_interface("output").port
    reordered_output = reordered.output_interface("output").port

    assert validate_region(reordered) == ()
    assert original_weight.operand.shape == reordered_weight.operand.shape
    assert original_weight.logical_beat_bits == reordered_weight.logical_beat_bits
    assert original_weight.beat_sequence != reordered_weight.beat_sequence
    assert original_output.operand.shape == reordered_output.operand.shape
    assert original_output.logical_beat_bits == reordered_output.logical_beat_bits
    assert original_output.beat_sequence != reordered_output.beat_sequence


def test_corrupting_mvau_output_availability_is_a_structural_failure():
    region = _mvau_region(2, 4, 4, 2, 2)
    output = region.output_interface("output")
    corrupt = OutputInterface(
        output.port,
        ScheduledOutputAvailability(dict(output.availability.entries[:-1])),
    )
    malformed = DataflowRegion(region.schedule, region.inputs, (corrupt,))

    assert "output.domain_image_mismatch" in tuple(
        issue.code for issue in validate_region(malformed)
    )


@pytest.mark.parametrize(
    "repetitions,matrix_width,matrix_height,simd,pe,expected_activation,expected_weight",
    [
        (1, 4, 4, 2, 2, 8, 16),
        (1, 64, 64, 8, 8, 512, 4096),
    ],
)
def test_mvau_concrete_storage_scales_with_occurrences(
    repetitions,
    matrix_width,
    matrix_height,
    simd,
    pe,
    expected_activation,
    expected_weight,
):
    region = _mvau_region(repetitions, matrix_width, matrix_height, simd, pe)

    assert validate_region(region) == ()
    assert len(region.input_interface("activation").requirements.entries) == expected_activation
    assert len(region.input_interface("weight").requirements.entries) == expected_weight
